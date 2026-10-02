"""Run the multi-agent ChartReasoningCrew over a JSONL of DVQA samples.

For each question, runs the full perception → DSL → parse → evidence →
reasoning → critic pipeline and persists the prediction plus the rich
intermediates (evidence items, critic verdict) needed for the analyses
in EVAL_PLAN.md §4.

Usage:
    python scripts/run_dvqa_agents.py \\
        --samples samples/val_easy_n500.jsonl \\
        --out eval_runs/$(date +%Y%m%d-%H%M%S)-agents-val_easy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

# Make the package importable when running this script directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chart_agents.crew import ChartReasoningCrew  # noqa: E402
from chart_agents.utils import setup_logging  # noqa: E402


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run(samples_path: Path, out_dir: Path, save_pipeline: bool, limit: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    meta_path = out_dir / "run_meta.json"
    pipeline_dir = out_dir / "pipelines" if save_pipeline else None
    if pipeline_dir is not None:
        pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Resume support
    done_ids: set[int] = set()
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["question_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} predictions already written.")

    # The Crew saves per-stage intermediates to its own `outputs/` by default.
    # We turn that off here (we keep a single compact pipeline.json per
    # question instead) — but `--save-pipeline` re-enables Crew's verbose
    # JSON dumps for case studies.
    crew = ChartReasoningCrew(save_intermediates=False)

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

            t0 = time.time()
            pred = ""
            explanation = ""
            evidence_items: list = []
            critic_consistent = None
            critic_issues: list = []
            critic_suggested = None
            chart_type = None
            err = None
            try:
                result = crew.run(image_path=image_path, question=question, tag=None)
                pred = result.answer.answer.strip()
                explanation = result.answer.explanation
                evidence_items = [
                    {
                        "group": e.group,
                        "category": e.category,
                        "value": e.value,
                        "visual_ref": e.visual_ref,
                    }
                    for e in result.evidence.items
                ]
                critic_consistent = bool(result.critic.consistent)
                critic_issues = list(result.critic.issues)
                critic_suggested = result.critic.suggested_revisit
                chart_type = result.perception.chart_type

                if pipeline_dir is not None:
                    with open(pipeline_dir / f"q{qid}.json", "w") as pf:
                        json.dump(result.model_dump(), pf, indent=2, default=str)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_errors += 1
                # Save the traceback for the first few failures to help debug.
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
                "explanation": explanation,
                "evidence": evidence_items,
                "critic_consistent": critic_consistent,
                "critic_issues": critic_issues,
                "critic_suggested_revisit": critic_suggested,
                "perception_chart_type": chart_type,
                "latency_ms": latency_ms,
                "error": err,
            }
            out_f.write(json.dumps(out) + "\n")
            out_f.flush()
            n_written += 1

            if n_written % 10 == 0:
                elapsed = time.time() - t_start
                rate = n_written / elapsed if elapsed > 0 else 0
                print(
                    f"  {n_written} done, {n_errors} errors, "
                    f"{rate:.2f} q/s, latest pred='{pred[:40]}' gold='{rec['answer'][:40]}'",
                    flush=True,
                )

    meta = {
        "experiment": "multi_agent_pipeline",
        "samples_path": str(samples_path),
        "total_predicted": n_written + len(done_ids),
        "errors": n_errors,
        "elapsed_sec": round(time.time() - t_start, 2),
        "save_pipeline": save_pipeline,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {pred_path}")
    print(f"Wrote {meta_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--save-pipeline",
        action="store_true",
        help="Also dump full PipelineResult JSON per question (for case studies).",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    setup_logging()
    run(args.samples, args.out, args.save_pipeline, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
