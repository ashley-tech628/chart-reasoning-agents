"""Aggregate baseline + multi-agent predictions into a comparison report.

Reads pairs of predictions.jsonl from `eval_runs/`, matches them by
question_id, and writes:

    <out>/results.md          # human-readable summary
    <out>/results.json        # machine-readable rollup
    <out>/per_split.csv       # accuracy by (split, experiment, question_type)
    <out>/disagreements.jsonl # items where the two systems disagree
                              # (for case-study mining)

Usage:
    python scripts/aggregate_dvqa.py \\
        --baseline-easy eval_runs/.../predictions.jsonl \\
        --baseline-hard eval_runs/.../predictions.jsonl \\
        --agents-easy   eval_runs/.../predictions.jsonl \\
        --agents-hard   eval_runs/.../predictions.jsonl \\
        --out results/preliminary
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dvqa_eval.evaluation.metrics import (  # noqa: E402
    iou_xywh,
    mcnemar_from_paired,
    relaxed_match,
    strict_exact_match,
)


def load_predictions(path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[int(r["question_id"])] = r
    return out


def pair_records(
    baseline: Dict[int, dict],
    agents: Dict[int, dict],
) -> List[Tuple[dict, dict]]:
    """Inner-join on question_id, dropping anything either side errored on."""
    common = sorted(set(baseline.keys()) & set(agents.keys()))
    pairs = []
    for qid in common:
        b = baseline[qid]
        a = agents[qid]
        if b.get("error") or a.get("error"):
            continue
        pairs.append((b, a))
    return pairs


def evidence_iou_for_record(rec: dict) -> Optional[float]:
    """If the gold has a bbox and we can find a matching component, return IoU."""
    gold_bbox = rec.get("bbox_answer")
    if not gold_bbox:
        return None
    # We only persisted EvidenceItems (no bbox). For now, treat IoU as
    # available only when the visual_ref points to a perception component
    # with a bbox — wiring this through requires either loading the saved
    # PipelineResult JSON (when --save-pipeline was set on the agent run)
    # or extending run_dvqa_agents.py to inline the bbox per evidence item.
    # We leave the hook here and skip when bbox is unavailable.
    return None


def summarize_split(pairs: List[Tuple[dict, dict]], split_label: str) -> dict:
    """Compute per-split accuracy and the paired McNemar p-value."""
    base_correct_strict: List[bool] = []
    agents_correct_strict: List[bool] = []
    base_correct_relaxed: List[bool] = []
    agents_correct_relaxed: List[bool] = []
    per_type: Dict[str, Dict[str, List[bool]]] = defaultdict(
        lambda: {"baseline_strict": [], "agents_strict": [], "baseline_relaxed": [], "agents_relaxed": []}
    )

    critic_consistent: List[bool] = []
    critic_consistent_and_correct: int = 0
    critic_consistent_and_wrong: int = 0
    critic_inconsistent_and_correct: int = 0
    critic_inconsistent_and_wrong: int = 0

    for b, a in pairs:
        qtype = b["question_type"]
        gold = b["gold"]
        bs = strict_exact_match(b["pred"], gold)
        as_ = strict_exact_match(a["pred"], gold)
        br = relaxed_match(b["pred"], gold)
        ar = relaxed_match(a["pred"], gold)
        base_correct_strict.append(bs)
        agents_correct_strict.append(as_)
        base_correct_relaxed.append(br)
        agents_correct_relaxed.append(ar)
        per_type[qtype]["baseline_strict"].append(bs)
        per_type[qtype]["agents_strict"].append(as_)
        per_type[qtype]["baseline_relaxed"].append(br)
        per_type[qtype]["agents_relaxed"].append(ar)

        cc = a.get("critic_consistent")
        if cc is not None:
            critic_consistent.append(bool(cc))
            if cc and ar:
                critic_consistent_and_correct += 1
            elif cc and not ar:
                critic_consistent_and_wrong += 1
            elif (not cc) and ar:
                critic_inconsistent_and_correct += 1
            else:
                critic_inconsistent_and_wrong += 1

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "split": split_label,
        "n_paired": len(pairs),
        "baseline_strict_em": mean(base_correct_strict),
        "agents_strict_em": mean(agents_correct_strict),
        "baseline_relaxed": mean(base_correct_relaxed),
        "agents_relaxed": mean(agents_correct_relaxed),
        "per_question_type": {
            t: {
                "n": len(v["baseline_strict"]),
                "baseline_strict": mean(v["baseline_strict"]),
                "agents_strict": mean(v["agents_strict"]),
                "baseline_relaxed": mean(v["baseline_relaxed"]),
                "agents_relaxed": mean(v["agents_relaxed"]),
            }
            for t, v in per_type.items()
        },
        "mcnemar_strict": mcnemar_from_paired(agents_correct_strict, base_correct_strict),
        "mcnemar_relaxed": mcnemar_from_paired(agents_correct_relaxed, base_correct_relaxed),
        "critic": {
            "n_with_verdict": len(critic_consistent),
            "p_consistent": mean(critic_consistent) if critic_consistent else None,
            "consistent_and_correct": critic_consistent_and_correct,
            "consistent_and_wrong": critic_consistent_and_wrong,
            "inconsistent_and_correct": critic_inconsistent_and_correct,
            "inconsistent_and_wrong": critic_inconsistent_and_wrong,
        },
    }
    return summary


def write_markdown(summaries: List[dict], out_path: Path) -> None:
    lines: List[str] = ["# DVQA Preliminary Results\n"]
    for s in summaries:
        lines.append(f"## {s['split']} (n={s['n_paired']} paired)\n")
        lines.append("| Metric | Baseline | Agents | Delta |")
        lines.append("|---|---:|---:|---:|")
        bs, as_ = s["baseline_strict_em"], s["agents_strict_em"]
        br, ar = s["baseline_relaxed"], s["agents_relaxed"]
        lines.append(f"| Strict EM | {bs:.3f} | {as_:.3f} | {as_ - bs:+.3f} |")
        lines.append(f"| Relaxed   | {br:.3f} | {ar:.3f} | {ar - br:+.3f} |")
        lines.append("")
        lines.append("### Per question_type (relaxed)\n")
        lines.append("| Type | n | Baseline | Agents | Delta |")
        lines.append("|---|---:|---:|---:|---:|")
        for t, v in sorted(s["per_question_type"].items()):
            lines.append(
                f"| {t} | {v['n']} | {v['baseline_relaxed']:.3f} | "
                f"{v['agents_relaxed']:.3f} | {v['agents_relaxed'] - v['baseline_relaxed']:+.3f} |"
            )
        lines.append("")
        m = s["mcnemar_relaxed"]
        lines.append(
            f"**McNemar (relaxed)**: b={m['b_only_a_right']} "
            f"(only agents right), c={m['c_only_b_right']} (only baseline right), "
            f"both={m['both_right']}, neither={m['neither_right']}, "
            f"**p = {m['p_value']:.4g}**.\n"
        )
        crit = s["critic"]
        if crit["n_with_verdict"]:
            lines.append("### Critic analysis\n")
            lines.append(f"- P(consistent) = {crit['p_consistent']:.3f}")
            lines.append(f"- consistent ∧ correct   : {crit['consistent_and_correct']}")
            lines.append(f"- consistent ∧ wrong     : {crit['consistent_and_wrong']}")
            lines.append(f"- inconsistent ∧ correct : {crit['inconsistent_and_correct']}")
            lines.append(f"- inconsistent ∧ wrong   : {crit['inconsistent_and_wrong']}\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def dump_disagreements(pairs: List[Tuple[dict, dict]], out_path: Path) -> None:
    """One JSONL line per (baseline, agents) disagreement — feed into case studies."""
    with open(out_path, "w", encoding="utf-8") as f:
        for b, a in pairs:
            gold = b["gold"]
            b_correct = relaxed_match(b["pred"], gold)
            a_correct = relaxed_match(a["pred"], gold)
            if b_correct == a_correct:
                continue
            f.write(
                json.dumps(
                    {
                        "question_id": b["question_id"],
                        "image": b.get("image"),
                        "question": b.get("question"),
                        "gold": gold,
                        "question_type": b["question_type"],
                        "baseline_pred": b["pred"],
                        "agents_pred": a["pred"],
                        "agents_explanation": a.get("explanation"),
                        "baseline_correct": b_correct,
                        "agents_correct": a_correct,
                        "critic_consistent": a.get("critic_consistent"),
                    }
                )
                + "\n"
            )


def write_csv(summaries: List[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "question_type", "n", "metric", "baseline", "agents", "delta"])
        for s in summaries:
            for t, v in s["per_question_type"].items():
                w.writerow([
                    s["split"], t, v["n"], "strict_em",
                    f"{v['baseline_strict']:.4f}", f"{v['agents_strict']:.4f}",
                    f"{v['agents_strict'] - v['baseline_strict']:+.4f}",
                ])
                w.writerow([
                    s["split"], t, v["n"], "relaxed",
                    f"{v['baseline_relaxed']:.4f}", f"{v['agents_relaxed']:.4f}",
                    f"{v['agents_relaxed'] - v['baseline_relaxed']:+.4f}",
                ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-easy", type=Path, required=True)
    parser.add_argument("--baseline-hard", type=Path, required=True)
    parser.add_argument("--agents-easy", type=Path, required=True)
    parser.add_argument("--agents-hard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    summaries: List[dict] = []
    for split, base_path, agents_path in [
        ("val_easy", args.baseline_easy, args.agents_easy),
        ("val_hard", args.baseline_hard, args.agents_hard),
    ]:
        base = load_predictions(base_path)
        agents = load_predictions(agents_path)
        pairs = pair_records(base, agents)
        print(f"{split}: paired {len(pairs)} records (baseline={len(base)}, agents={len(agents)})")
        summary = summarize_split(pairs, split)
        summaries.append(summary)
        dump_disagreements(pairs, args.out / f"disagreements_{split}.jsonl")

    write_markdown(summaries, args.out / "results.md")
    write_csv(summaries, args.out / "per_split.csv")
    with open(args.out / "results.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote {args.out}/results.md (+ json/csv/disagreements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
