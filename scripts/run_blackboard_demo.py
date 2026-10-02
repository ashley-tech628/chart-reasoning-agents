"""Run the structured round-based blackboard workflow on a fixed sample ChartDSL.

This demo intentionally does NOT import the old ChartReasoningCrew, so it is
independent from upstream perception / OCR files. It is only for showing the
new downstream blackboard workflow and agent communication.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chart_agents.schemas import ChartDSL, ChartDataPoint  # noqa: E402
from chart_agents.utils import new_run_dir, save_intermediate, setup_logging  # noqa: E402
from chart_agents.workflows.blackboard_workflow import run_blackboard_workflow  # noqa: E402


def sample_chart_dsl() -> ChartDSL:
    """Fixed ChartDSL for examples/sample_chart.png.

    This keeps the demo focused on downstream reasoning.
    The upstream vision/OCR group can later replace this with real perception output.
    """
    return ChartDSL(
        type="bar_chart",
        title="Quarterly Sales by Product",
        x_axis="Quarter",
        y_axis="Sales",
        groups=["Q1", "Q2", "Q3", "Q4"],
        categories=["Product A", "Product B"],
        data=[
            ChartDataPoint(group="Q1", category="Product A", value=42.0, visual_mark_index=0),
            ChartDataPoint(group="Q1", category="Product B", value=35.0, visual_mark_index=1),
            ChartDataPoint(group="Q2", category="Product A", value=58.0, visual_mark_index=2),
            ChartDataPoint(group="Q2", category="Product B", value=47.0, visual_mark_index=3),
            ChartDataPoint(group="Q3", category="Product A", value=71.0, visual_mark_index=4),
            ChartDataPoint(group="Q3", category="Product B", value=58.0, visual_mark_index=5),
            ChartDataPoint(group="Q4", category="Product A", value=88.0, visual_mark_index=6),
            ChartDataPoint(group="Q4", category="Product B", value=80.0, visual_mark_index=7),
        ],
    )


def _dump(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def print_messages(messages) -> None:
    current_round = None
    for msg in messages:
        if msg.round != current_round:
            current_round = msg.round
            print(f"\n--- Round {current_round} ---")
        print(json.dumps(_dump(msg), indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run blackboard multi-agent demo.")
    parser.add_argument("--image", help="Optional reference image; this structured demo does not parse pixels.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--tag", default="blackboard-demo")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--dsl", type=Path, help="Use a ChartDSL JSON file instead of the built-in fixture.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    setup_logging()
    image_path = Path(args.image) if args.image else None
    if image_path is not None and not image_path.exists():
        parser.error(f"Reference image not found: {image_path}")
    print("Structured demo: deterministic agents over ChartDSL; no OCR or API calls.")

    chart_dsl = (
        ChartDSL.model_validate_json(args.dsl.read_text(encoding="utf-8"))
        if args.dsl else sample_chart_dsl()
    )
    blackboard, answer = run_blackboard_workflow(args.question, chart_dsl)

    if not args.no_save:
        run_dir = new_run_dir(args.output_dir, args.tag)
        save_intermediate({"image_path": str(image_path) if image_path else None, "question": args.question}, "0_input", run_dir)
        save_intermediate(chart_dsl, "1_sample_chart_dsl", run_dir)
        save_intermediate(blackboard.to_dict(), "2_blackboard_messages", run_dir)
        save_intermediate(answer, "3_blackboard_final_answer", run_dir)
        print(f"\nSaved blackboard run to: {run_dir}")

    print("\n" + "=" * 26 + " BLACKBOARD WORKFLOW " + "=" * 26)
    print(f"Question: {args.question}")
    print(f"Final Answer: {answer.answer}")
    print(f"Explanation: {answer.explanation}")

    if answer.intermediate_values:
        print("Intermediate values:")
        for iv in answer.intermediate_values:
            print(f"  {iv.name} = {iv.value}")

    print_messages(blackboard.messages)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
