"""Single-shot GPT-4o baseline for DVQA.

Sends one vision call per question: <image> + question -> answer. No DSL,
no decomposition, no critic. This is the "what does the underlying VLM
do on its own" reference point that the multi-agent system has to beat.

Usage:
    python scripts/run_dvqa_baseline.py \\
        --samples samples/val_easy_n500.jsonl \\
        --out eval_runs/$(date +%Y%m%d-%H%M%S)-baseline-val_easy \\
        --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import litellm

# Make the package importable when running this script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chart_agents.utils import encode_image_b64, setup_logging  # noqa: E402


BASELINE_PROMPT = """You are reading a bar chart. Answer the question below \
concisely with just the final answer. No explanation, no reasoning, no units, \
no quotes, no leading phrase like "The answer is".

If the answer is a number, give just the number.
If the answer is a label from the chart, give just the label (no quotes).
If the answer is yes or no, give just "yes" or "no".

Question: {question}"""


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def query_baseline(model: str, image_path: str, question: str) -> tuple[str, str, int]:
    """Returns (parsed_answer, raw_response_text, latency_ms)."""
    b64, media_type = encode_image_b64(image_path)
    t0 = time.time()
    response = litellm.completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BASELINE_PROMPT.format(question=question)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.0,
        max_tokens=64,
    )
    latency_ms = int((time.time() - t0) * 1000)
    raw = response.choices[0].message.content or ""
    # Strip any quotes / trailing punctuation; keep it close to a leaf answer.
    parsed = raw.strip().strip("\"'").strip(".").strip()
    return parsed, raw, latency_ms


def run(samples_path: Path, out_dir: Path, model: str, limit: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    meta_path = out_dir / "run_meta.json"

    # Resume support: skip any question_ids already in pred_path.
    done_ids: set[int] = set()
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["question_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} predictions already written.")

    total_seen = 0
    n_written = 0
    n_errors = 0
    t_start = time.time()

    with open(pred_path, "a") as out_f:
        for rec in iter_jsonl(samples_path):
            total_seen += 1
            if limit is not None and total_seen > limit:
                break
            qid = rec["question_id"]
            if qid in done_ids:
                continue
            image_path = rec["image_path"]
            question = rec["question"]
            try:
                pred, raw, latency_ms = query_baseline(model, image_path, question)
                err = None
            except Exception as e:
                pred, raw, latency_ms = "", "", 0
                err = f"{type(e).__name__}: {e}"
                n_errors += 1

            out = {
                "question_id": qid,
                "image": rec["image"],
                "image_path": image_path,
                "question": question,
                "gold": rec["answer"],
                "question_type": rec["question_type"],
                "bbox_answer": rec.get("bbox_answer"),
                "pred": pred,
                "raw_response": raw,
                "latency_ms": latency_ms,
                "error": err,
            }
            out_f.write(json.dumps(out) + "\n")
            out_f.flush()
            n_written += 1

            if n_written % 25 == 0:
                elapsed = time.time() - t_start
                rate = n_written / elapsed if elapsed > 0 else 0
                print(
                    f"  {n_written} done, {n_errors} errors, "
                    f"{rate:.2f} q/s, latest pred='{pred[:40]}' gold='{rec['answer'][:40]}'",
                    flush=True,
                )

    meta = {
        "experiment": "baseline_single_shot",
        "model": model,
        "samples_path": str(samples_path),
        "total_predicted": n_written + len(done_ids),
        "errors": n_errors,
        "elapsed_sec": round(time.time() - t_start, 2),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {pred_path}")
    print(f"Wrote {meta_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True, help="JSONL from sample_dvqa.py.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N records (for smoke tests).",
    )
    args = parser.parse_args()
    setup_logging()
    run(args.samples, args.out, args.model, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
