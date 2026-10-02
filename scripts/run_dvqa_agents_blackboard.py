"""DVQA runner for the structured round-based blackboard workflow.

It keeps the upstream perception/ChartDSL pipeline unchanged, then replaces the
final downstream reasoning with a blackboard-style round workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chart_agents.config import settings  # noqa: E402
from chart_agents.crew import ChartReasoningCrew  # noqa: E402
from chart_agents.utils import setup_logging  # noqa: E402
from chart_agents.workflows.blackboard_workflow import run_blackboard_workflow  # noqa: E402


def _dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def run(samples_path: Path, out_dir: Path, save_blackboard: bool, limit: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    meta_path = out_dir / "run_meta.json"
    bb_dir = out_dir / "blackboards" if save_blackboard else None
    if bb_dir is not None:
        bb_dir.mkdir(exist_ok=True)

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

            t0 = time.time()
            pred = ""
            explanation = ""
            source = "blackboard"
            blackboard_messages = []
            chart_type = None
            upstream_agent_pred = ""
            err = None

            try:
                upstream = crew.run(rec["image_path"], rec["question"], tag=None)
                chart_type = upstream.perception.chart_type
                upstream_agent_pred = upstream.answer.answer

                blackboard, answer = run_blackboard_workflow(rec["question"], upstream.chart_dsl)
                pred = answer.answer.strip()
                explanation = answer.explanation
                blackboard_messages = [_dump(m) for m in blackboard.messages]

                # MVP safety fallback: if blackboard cannot produce an answer, keep upstream answer.
                if not pred:
                    pred = upstream.answer.answer.strip()
                    source = "fallback_upstream_agent"

                if bb_dir is not None:
                    with open(bb_dir / f"q{qid}.json", "w", encoding="utf-8") as f:
                        json.dump(blackboard.to_dict(), f, indent=2, ensure_ascii=False)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                source = "error"
                n_errors += 1
                if n_errors <= 5:
                    print(f"\n[error on qid={qid}] {err}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

            out = {
                "question_id": qid,
                "image": rec.get("image"),
                "image_path": rec.get("image_path"),
                "question": rec.get("question"),
                "gold": rec.get("answer"),
                "question_type": rec.get("question_type"),
                "bbox_answer": rec.get("bbox_answer"),
                "pred": pred,
                "explanation": explanation,
                "source": source,
                "upstream_agent_pred": upstream_agent_pred,
                "perception_chart_type": chart_type,
                "blackboard_messages": blackboard_messages,
                "latency_ms": int((time.time() - t0) * 1000),
                "error": err,
            }
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_f.flush()
            n_written += 1

            if n_written % 5 == 0:
                elapsed = time.time() - t_start
                rate = n_written / elapsed if elapsed else 0
                print(f"  {n_written} done, {n_errors} errors, {rate:.2f} q/s, pred='{pred}' gold='{rec.get('answer')}'")

    meta = {
        "experiment": "structured_round_based_blackboard_mvp",
        "samples_path": str(samples_path),
        "total_predicted": n_written + len(done_ids),
        "errors": n_errors,
        "elapsed_sec": round(time.time() - t_start, 2),
        "save_blackboard": save_blackboard,
        "agent_llm_model": settings.llm_model,
        "agent_vision_model": settings.vision_model,
        "notes": "Structured MVP: blackboard package + separate router/answer-type/solvers/critics/judge modules.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {pred_path}")
    print(f"Wrote {meta_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-blackboard", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    setup_logging()
    run(args.samples, args.out, args.save_blackboard, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
