"""Guarded hybrid DVQA runner: baseline + multi-agent + program solver + scratch verifier.

This runner is intentionally conservative: the direct VLM baseline is treated as
an anchor, and the multi-agent/program answer only overrides it when an
independent scratch answer or a high-confidence verifier supports the override.
It also normalizes final short answers into DVQA-style forms, e.g. count answers
like "0" -> "zero" when the question asks "How many groups/items/...".

The output schema remains compatible with scripts/aggregate_dvqa.py: `pred` is
what gets scored, while diagnostic fields expose the candidates and final source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional, Tuple

import litellm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chart_agents.config import settings  # noqa: E402
from chart_agents.crew import ChartReasoningCrew  # noqa: E402
from chart_agents.utils import encode_image_b64, setup_logging  # noqa: E402


BASELINE_PROMPT = """You are reading a bar chart. Answer the question below concisely with just the final answer.
No explanation, no reasoning, no units, no quotes, no leading phrase like "The answer is".

If the answer is a number, give just the number.
If the answer is a label from the chart, give just the label (no quotes).
If the answer is yes or no, give just "yes" or "no".

Question: {question}"""


SCRATCH_PROMPT = """You are solving a DVQA-style bar-chart question.
Read the chart carefully and answer the question. Think through the chart values internally, but output only STRICT JSON:
{{"answer": "final short answer only", "confidence": 0.0}}

DVQA answer-format rules:
- For yes/no questions, answer exactly "yes" or "no".
- For "Which ..." questions, answer the chart label/name, not a numeric value.
- For counting questions such as "How many groups/items/algorithms...", answer with a small number word when natural, e.g. zero, one, two.
- For numeric value/difference/sum/unit questions, answer with the numeric value only.

Question: {question}"""


VERIFIER_PROMPT = """You are a strict chart-question answering verifier.
You are given the chart image, the user question, and several candidate answers.
Use the image as the source of truth. The baseline is a strong anchor; override it only when another candidate is clearly better.

Question:
{question}

baseline_candidate: {baseline_pred}
agent_candidate: {agent_pred}
program_candidate: {program_pred}
scratch_candidate: {scratch_pred}

Agent explanation:
{agent_explanation}

Agent evidence table:
{evidence_json}

Program rationale:
{program_rationale}

Rules:
- If the question asks which/which bar/which group/which category/which item/which algorithm/which object, the final answer should be a chart label, not a numeric value.
- If the question asks how many/how much/what value/sum/difference/units, the final answer should be the requested number, formatted concisely.
- If the question asks yes/no, the final answer must be exactly "yes" or "no".
- Prefer the baseline candidate unless the chart image or multiple independent candidates clearly support a different answer.
- If agent/program/scratch agree against baseline and the chart supports them, you may override baseline.
- Keep the final answer concise. Do not include explanation.

Return STRICT JSON only:
{{
  "choice": "baseline" | "agent" | "program" | "scratch" | "other",
  "answer": "final short answer only",
  "confidence": 0.0
}}"""


NUM_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]
WORD_TO_NUM = {w: i for i, w in enumerate(NUM_WORDS)}


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean_short_answer(text: str) -> str:
    text = (text or "").strip().strip("\"'").strip()
    # Remove common lead-ins that hurt exact match.
    text = re.sub(r"^(the\s+answer\s+is\s+|answer\s*:\s*)", "", text, flags=re.I).strip()
    # Keep labels like "St. Louis" intact; only trim sentence-ending punctuation.
    while len(text) > 1 and text[-1] in ".。":
        text = text[:-1].strip()
    return text


def _as_float(text: str) -> Optional[float]:
    t = clean_short_answer(text).lower().replace(",", "").strip()
    if t in WORD_TO_NUM:
        return float(WORD_TO_NUM[t])
    t = t.rstrip("%")
    try:
        return float(t)
    except Exception:
        return None


def _fmt_number(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return (f"{x:.6f}".rstrip("0").rstrip("."))


def _number_word_if_small(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        n = int(round(x))
        if 0 <= n < len(NUM_WORDS):
            return NUM_WORDS[n]
    return _fmt_number(x)


def wants_count_word(question: str) -> bool:
    """DVQA often stores pure count answers as words: zero, one, two, ...

    Do NOT use words for value/difference/unit questions such as "How many units"
    or "How many more"; those are numeric amount questions in the provided set.
    """
    q = question.lower()
    if not q.startswith("how many"):
        return False
    if any(s in q for s in ["how many more", "how many units", "how many unit"]):
        return False
    return any(s in q for s in [
        "how many groups", "how many algorithms", "how many items", "how many bars", "how many objects",
    ])


def canonicalize_answer(question: str, answer: str) -> str:
    ans = clean_short_answer(answer)
    q = question.lower()
    a_low = ans.lower().strip()

    # Boolean normalization.
    if a_low in {"true", "correct", "smaller", "greater"}:
        ans = "yes"
    elif a_low in {"false", "incorrect"}:
        ans = "no"
    if q.startswith("is ") or q.startswith("are ") or q.startswith("does "):
        if ans.lower() in {"yes", "no"}:
            return ans.lower()

    # Numeric normalization.
    val = _as_float(ans)
    if val is not None:
        if wants_count_word(question):
            return _number_word_if_small(val)
        return _fmt_number(val)

    # Label normalization: remove extra articles, but preserve casing mostly.
    ans = re.sub(r"^(the|a|an)\s+", "", ans, flags=re.I).strip()
    return ans


def norm(text: str) -> str:
    t = canonicalize_answer("", text).lower().replace("_", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def parse_jsonish(text: str) -> Optional[dict]:
    t = (text or "").strip()
    if t.startswith("```"):
        first = t.find("\n")
        if first != -1:
            t = t[first + 1 :]
        if t.endswith("```"):
            t = t[:-3]
    # Robustly pull the first JSON object if the model added text around it.
    if not t.startswith("{"):
        m = re.search(r"\{.*\}", t, flags=re.S)
        if m:
            t = m.group(0)
    try:
        return json.loads(t)
    except Exception:
        return None


def query_vlm_json(model: str, image_path: str, prompt: str, max_tokens: int = 128) -> tuple[Optional[dict], str, int]:
    b64, media_type = encode_image_b64(image_path)
    t0 = time.time()
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    latency_ms = int((time.time() - t0) * 1000)
    raw = response.choices[0].message.content or ""
    return parse_jsonish(raw), raw, latency_ms


def query_baseline(model: str, image_path: str, question: str) -> tuple[str, str, int]:
    b64, media_type = encode_image_b64(image_path)
    t0 = time.time()
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BASELINE_PROMPT.format(question=question)},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=64,
    )
    latency_ms = int((time.time() - t0) * 1000)
    raw = response.choices[0].message.content or ""
    return canonicalize_answer(question, raw), raw, latency_ms


def query_scratch(model: str, image_path: str, question: str) -> tuple[str, Optional[dict], str, int]:
    parsed, raw, latency_ms = query_vlm_json(
        model, image_path, SCRATCH_PROMPT.format(question=question), max_tokens=128
    )
    if isinstance(parsed, dict):
        ans = canonicalize_answer(question, str(parsed.get("answer", "")))
        return ans, parsed, raw, latency_ms
    return "", None, raw, latency_ms


def _data_as_dict(chart_dsl) -> list[dict]:
    out = []
    for d in getattr(chart_dsl, "data", []) or []:
        out.append({"category": d.category, "series": d.series, "value": float(d.value)})
    return out


def _lookup(data: list[dict], label: str, context: str | None = None) -> Optional[float]:
    """Try both orientations: (series=label, category=context) and vice versa."""
    lab = label.strip().lower()
    ctx = context.strip().lower() if context else None
    candidates = []
    for d in data:
        cat = d["category"].lower()
        ser = d["series"].lower()
        if ctx:
            if ser == lab and cat == ctx:
                candidates.append(d["value"])
            if cat == lab and ser == ctx:
                candidates.append(d["value"])
        else:
            if ser == lab or cat == lab:
                candidates.append(d["value"])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _label_exists(data: list[dict], label: str) -> bool:
    l = label.lower()
    return any(d["category"].lower() == l or d["series"].lower() == l for d in data)


def programmatic_answer(question: str, chart_dsl, parsed_question=None) -> tuple[str, float, str]:
    """Small deterministic solver for common DVQA reasoning templates.

    It uses ONLY the ChartDSL values produced by the agents. When it cannot map a
    question safely, it returns ("", 0.0, rationale).
    """
    data = _data_as_dict(chart_dsl)
    if not data:
        return "", 0.0, "no ChartDSL data"
    q = question.lower().strip()
    cats = sorted({d["category"] for d in data})
    series = sorted({d["series"] for d in data})

    # Yes/no pairwise comparison: "Is the value of X in Y smaller than ... Z in W?"
    # Yes/no pairwise comparison:
    # "Did item A in store X sell fewer units than item B in store Y?"
    m = re.search(
        r"(?:is|are|did|does|do)\s+"
        r"(?:the\s+)?"
        r"(?:accuracy\s+of\s+the\s+algorithm\s+|value\s+of\s+|item\s+|object\s+|product\s+|algorithm\s+)?"
        r"(.+?)\s+in\s+"
        r"(?:the\s+)?(?:dataset\s+|category\s+|store\s+|group\s+)?"
        r"(.+?)\s+"
        r"(?:sold\s+)?"
        r"(?:smaller|less|lower|fewer|larger|greater|higher|more)\s+"
        r"(?:units\s+)?than\s+"
        r"(?:the\s+)?"
        r"(?:accuracy\s+of\s+the\s+algorithm\s+|value\s+of\s+|item\s+|object\s+|product\s+|algorithm\s+)?"
        r"(.+?)\s+in\s+"
        r"(?:the\s+)?(?:dataset\s+|category\s+|store\s+|group\s+)?"
        r"(.+?)\??$",
        q,
    )

    if m:
        a, ac, b, bc = [x.strip(" ?") for x in m.groups()]
        v1 = _lookup(data, a, ac)
        v2 = _lookup(data, b, bc)

        if v1 is not None and v2 is not None:
            wants_less = any(k in q for k in ["smaller", "less", "lower", "fewer"])
            ok = v1 < v2 if wants_less else v1 > v2
            sign = "<" if wants_less else ">"

            return (
                "yes" if ok else "no",
                0.85,
                f"compare {a}/{ac}={v1} {sign} {b}/{bc}={v2}",
            )

    # Sum of two named labels.
    m = re.search(r"sum\s+of\s+the\s+values\s+of\s+(.+?)\s+and\s+(.+?)\??$", q)
    if m:
        a, b = [x.strip(" ?") for x in m.groups()]
        va = _lookup(data, a)
        vb = _lookup(data, b)
        if va is not None and vb is not None:
            return canonicalize_answer(question, _fmt_number(va + vb)), 0.80, f"sum {a}={va}, {b}={vb}"

    # Largest/least individual bar value.
    values = [d["value"] for d in data]
    max_d = max(data, key=lambda d: d["value"])
    min_d = min(data, key=lambda d: d["value"])
    if "value of the largest individual bar" in q:
        return canonicalize_answer(question, _fmt_number(max_d["value"])), 0.85, f"max individual value={max_d['value']}"
    if "least sold item" in q and ("how many units" in q or "units" in q):
        return canonicalize_answer(question, _fmt_number(min_d["value"])), 0.80, f"min individual value={min_d['value']}"
    if "most accurate" in q and "least accurate" in q and ("how much more" in q or "compared" in q):
        diff = max(values) - min(values)
        return canonicalize_answer(question, _fmt_number(diff)), 0.80, f"max-min={max(values)}-{min(values)}"

    # Counting thresholds.
    m = re.search(r"how many groups of bars contain at least one bar with value (greater|higher|smaller|less) than ([\d.]+)", q)
    if m:
        op, thresh_s = m.groups()
        thresh = float(thresh_s)
        by_cat = defaultdict(list)
        for d in data:
            by_cat[d["category"]].append(d["value"])
        if op in {"greater", "higher"}:
            n = sum(1 for vals in by_cat.values() if any(v > thresh for v in vals))
        else:
            n = sum(1 for vals in by_cat.values() if any(v < thresh for v in vals))
        return canonicalize_answer(question, _fmt_number(n)), 0.85, f"count categories with any value {op} than {thresh} = {n}"

    m = re.search(r"how many (algorithms|items|objects|bars).*?(higher|greater|smaller|less) than ([\d.]+)", q)
    if m:
        noun, op, thresh_s = m.groups()
        thresh = float(thresh_s)
        # Choose the axis with more distinct non-generic labels as the count dimension.
        # For simple horizontal bar charts series may be a placeholder like "value".
        count_by = "series"
        if len(cats) >= len(series) or series == ["value"]:
            count_by = "category"
        by_label = defaultdict(list)
        for d in data:
            by_label[d[count_by]].append(d["value"])
        if op in {"greater", "higher"}:
            n = sum(1 for vals in by_label.values() if any(v > thresh for v in vals))
        else:
            n = sum(1 for vals in by_label.values() if any(v < thresh for v in vals))
        return canonicalize_answer(question, _fmt_number(n)), 0.70, f"count {count_by} labels with any value {op} than {thresh} = {n}"

    # Which group/object summed across categories.
    if "summed" in q and ("which group" in q or "which object" in q or "which item" in q):
        if "object" in q or "item" in q:
            dim = "series" if len(series) > 1 else "category"
        else:
            dim = "category"
        sums = defaultdict(float)
        for d in data:
            sums[d[dim]] += d["value"]
        if sums:
            label, val = max(sums.items(), key=lambda kv: kv[1])
            return canonicalize_answer(question, label), 0.78, f"max sum by {dim}: {label}={val}"

    # Which bar/algorithm/object has highest/largest value.
    if q.startswith("which") and any(k in q for k in ["largest value", "highest accuracy", "largest individual", "largest bar"]):
        if "algorithm" in q or "object" in q or "item" in q:
            label = max_d["series"] if len(series) > 1 else max_d["category"]
        elif "bar" in q:
            label = max_d["category"] if len(cats) >= len(series) or series == ["value"] else max_d["series"]
        else:
            label = max_d["category"]
        return canonicalize_answer(question, label), 0.75, f"max individual bar {label}={max_d['value']}"

    return "", 0.0, "no safe program template matched"


def verify_choice(
    model: str,
    image_path: str,
    question: str,
    baseline_pred: str,
    agent_pred: str,
    program_pred: str,
    program_conf: float,
    program_rationale: str,
    scratch_pred: str,
    agent_explanation: str,
    evidence_items: list[dict],
) -> tuple[str, str, Optional[dict], int]:
    """Return (final_pred, source, verifier_json, latency_ms)."""
    # Canonicalize every candidate before comparing/scoring.
    baseline_pred = canonicalize_answer(question, baseline_pred)
    agent_pred = canonicalize_answer(question, agent_pred)
    program_pred = canonicalize_answer(question, program_pred)
    scratch_pred = canonicalize_answer(question, scratch_pred)

    # If all non-empty useful candidates agree, no verifier needed.
    candidates = [x for x in [baseline_pred, agent_pred, program_pred, scratch_pred] if x]
    if candidates and all(norm(x) == norm(candidates[0]) for x in candidates):
        return canonicalize_answer(question, candidates[0]), "all_agree", None, 0

    # Conservative but useful override rule: two independent non-baseline
    # candidates must agree before we switch away from the baseline.
    non_base = []
    for source, ans in [("agent", agent_pred), ("program", program_pred), ("scratch", scratch_pred)]:
        if ans and baseline_pred and norm(ans) != norm(baseline_pred):
            non_base.append((source, ans))
    for i in range(len(non_base)):
        for j in range(i + 1, len(non_base)):
            si, ai = non_base[i]
            sj, aj = non_base[j]
            if norm(ai) == norm(aj):
                if "program" not in {si, sj} or program_conf >= 0.65:
                    return canonicalize_answer(question, ai), f"two_signal_override_{si}_{sj}", None, 0

    evidence_json = json.dumps(evidence_items, ensure_ascii=False)
    prompt = VERIFIER_PROMPT.format(
        question=question,
        baseline_pred=baseline_pred,
        agent_pred=agent_pred or "<empty>",
        program_pred=program_pred or "<empty>",
        scratch_pred=scratch_pred or "<empty>",
        agent_explanation=agent_explanation or "",
        evidence_json=evidence_json[:4000],
        program_rationale=program_rationale or "",
    )
    t0 = time.time()
    try:
        parsed, raw, latency_ms = query_vlm_json(model, image_path, prompt, max_tokens=160)
        if isinstance(parsed, dict):
            choice = str(parsed.get("choice", "")).lower().strip()
            answer = canonicalize_answer(question, str(parsed.get("answer", "")))
            try:
                conf = float(parsed.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            parsed["raw"] = raw

            # Do not move off baseline unless the verifier is very confident.
            # This is what prevents the hard split from losing baseline-correct cases.
            if choice == "baseline" and answer:
                return answer, "verifier_baseline", parsed, latency_ms
            if answer and conf >= 0.85:
                # Even with high confidence, prefer overrides supported by scratch/program/agent agreement.
                if baseline_pred and norm(answer) == norm(baseline_pred):
                    return baseline_pred, "verifier_baseline_equiv", parsed, latency_ms
                support = 0
                for cand in [agent_pred, program_pred if program_conf >= 0.65 else "", scratch_pred]:
                    if cand and norm(cand) == norm(answer):
                        support += 1
                if support >= 1:
                    return answer, f"verifier_{choice}_highconf", parsed, latency_ms
            return baseline_pred, "fallback_baseline_guarded", parsed, latency_ms
        return baseline_pred, "fallback_baseline_parse", {"raw": raw}, latency_ms
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        return baseline_pred, "fallback_baseline_error", {"error": f"{type(e).__name__}: {e}"}, latency_ms


def run(
    samples_path: Path,
    out_dir: Path,
    save_pipeline: bool,
    limit: int | None,
    baseline_model: str,
    verifier_model: str,
    scratch_model: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    meta_path = out_dir / "run_meta.json"
    pipeline_dir = out_dir / "pipeline" if save_pipeline else None
    if pipeline_dir is not None:
        pipeline_dir.mkdir(exist_ok=True)

    done_ids: set[int] = set()
    if pred_path.exists():
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(int(json.loads(line)["question_id"]))
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} predictions already written.")

    crew = ChartReasoningCrew(save_intermediates=False)
    n_written = 0
    n_errors = 0
    total_seen = 0
    t_start = time.time()

    with open(pred_path, "a", encoding="utf-8") as out_f:
        for rec in iter_jsonl(samples_path):
            total_seen += 1
            if limit is not None and total_seen > limit:
                break
            qid = int(rec["question_id"])
            if qid in done_ids:
                continue

            image_path = rec["image_path"]
            question = rec["question"]
            t0 = time.time()
            err = None

            baseline_pred = ""
            baseline_raw = ""
            baseline_latency_ms = 0
            agent_pred = ""
            explanation = ""
            evidence_items: list[dict] = []
            critic_consistent = None
            critic_issues: list = []
            critic_suggested = None
            chart_type = None
            verifier = None
            final_source = ""
            verifier_latency_ms = 0
            scratch_pred = ""
            scratch_raw = ""
            scratch_json = None
            scratch_latency_ms = 0
            program_pred = ""
            program_conf = 0.0
            program_rationale = ""
            pred = ""

            try:
                baseline_pred, baseline_raw, baseline_latency_ms = query_baseline(
                    baseline_model, image_path, question
                )

                result = crew.run(image_path=image_path, question=question, tag=None)
                agent_pred = canonicalize_answer(question, result.answer.answer)
                explanation = result.answer.explanation
                evidence_items = [
                    {
                        "category": e.category,
                        "series": e.series,
                        "value": e.value,
                        "visual_ref": e.visual_ref,
                    }
                    for e in result.evidence.items
                ]
                critic_consistent = bool(result.critic.consistent)
                critic_issues = list(result.critic.issues)
                critic_suggested = result.critic.suggested_revisit
                chart_type = result.perception.chart_type

                program_pred, program_conf, program_rationale = programmatic_answer(
                    question, result.chart_dsl, result.parsed_question
                )
                program_pred = canonicalize_answer(question, program_pred)

                scratch_pred, scratch_json, scratch_raw, scratch_latency_ms = query_scratch(
                    scratch_model, image_path, question
                )

                pred, final_source, verifier, verifier_latency_ms = verify_choice(
                    verifier_model,
                    image_path,
                    question,
                    baseline_pred,
                    agent_pred,
                    program_pred,
                    program_conf,
                    program_rationale,
                    scratch_pred,
                    explanation,
                    evidence_items,
                )
                pred = canonicalize_answer(question, pred)

                if pipeline_dir is not None:
                    with open(pipeline_dir / f"q{qid}.json", "w", encoding="utf-8") as pf:
                        json.dump(result.model_dump(), pf, indent=2, ensure_ascii=False, default=str)
            except Exception as e:
                pred = canonicalize_answer(question, baseline_pred or "")
                final_source = "error_fallback_baseline"
                err = f"{type(e).__name__}: {e}"
                n_errors += 1
                if n_errors <= 5:
                    print(f"\n[error on qid={qid}] {err}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

            latency_ms = int((time.time() - t0) * 1000)
            out = {
                "question_id": qid,
                "image": rec["image"],
                "image_path": image_path,
                "question": question,
                "gold": rec["answer"],
                "question_type": rec["question_type"],
                "bbox_answer": rec.get("bbox_answer"),
                "pred": pred,
                "baseline_pred": baseline_pred,
                "baseline_raw_response": baseline_raw,
                "agent_pred": agent_pred,
                "program_pred": program_pred,
                "program_confidence": program_conf,
                "program_rationale": program_rationale,
                "scratch_pred": scratch_pred,
                "scratch_raw_response": scratch_raw,
                "scratch_json": scratch_json,
                "final_source": final_source,
                "verifier": verifier,
                "explanation": explanation,
                "evidence": evidence_items,
                "critic_consistent": critic_consistent,
                "critic_issues": critic_issues,
                "critic_suggested_revisit": critic_suggested,
                "perception_chart_type": chart_type,
                "latency_ms": latency_ms,
                "baseline_latency_ms": baseline_latency_ms,
                "scratch_latency_ms": scratch_latency_ms,
                "verifier_latency_ms": verifier_latency_ms,
                "error": err,
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_f.flush()
            n_written += 1

            if n_written % 5 == 0:
                elapsed = time.time() - t_start
                rate = n_written / elapsed if elapsed > 0 else 0
                print(
                    f"  {n_written} done, {n_errors} errors, {rate:.2f} q/s, "
                    f"final='{pred[:30]}' source={final_source} gold='{rec['answer'][:30]}'",
                    flush=True,
                )

    meta = {
        "experiment": "guarded_multi_signal_hybrid",
        "samples_path": str(samples_path),
        "total_predicted": n_written + len(done_ids),
        "errors": n_errors,
        "elapsed_sec": round(time.time() - t_start, 2),
        "save_pipeline": save_pipeline,
        "baseline_model": baseline_model,
        "agent_llm_model": settings.llm_model,
        "agent_vision_model": settings.vision_model,
        "scratch_model": scratch_model,
        "verifier_model": verifier_model,
        "notes": "Baseline-anchored hybrid with programmatic solver, scratch answer, guarded verifier, and DVQA-style answer canonicalization.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {pred_path}")
    print(f"Wrote {meta_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-pipeline", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline-model", default=settings.llm_model)
    parser.add_argument("--verifier-model", default=settings.llm_model)
    parser.add_argument("--scratch-model", default=settings.llm_model)
    args = parser.parse_args()
    setup_logging()
    run(
        args.samples,
        args.out,
        args.save_pipeline,
        args.limit,
        args.baseline_model,
        args.verifier_model,
        args.scratch_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
