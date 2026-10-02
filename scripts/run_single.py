"""Run the chart reasoning pipeline on a single (image, question) example.

Usage:
    python scripts/run_single.py \\
        --image examples/sample_chart.png \\
        --question "Which quarter has the largest gap between Product A and Product B?" \\
        --tag demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this script directly without `pip install -e .`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chart_agents.crew import ChartReasoningCrew  # noqa: E402
from chart_agents.utils import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the chart reasoning pipeline on a single example."
    )
    parser.add_argument("--image", required=True, help="Path to chart image (PNG/JPG).")
    parser.add_argument(
        "--question", required=True, help="Natural-language question about the chart."
    )
    parser.add_argument(
        "--tag", default=None, help="Optional tag suffix for the run output directory."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save per-stage JSON files to outputs/.",
    )
    args = parser.parse_args()

    setup_logging()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 2

    crew = ChartReasoningCrew(save_intermediates=not args.no_save)
    result = crew.run(image_path=str(image_path), question=args.question, tag=args.tag)

    print("\n" + "=" * 30 + " RESULT " + "=" * 30)
    print(f"Question : {args.question}")
    print(f"Answer   : {result.answer.answer}")
    print(f"\nExplanation:\n  {result.answer.explanation}")
    if result.answer.intermediate_values:
        print("\nIntermediate values:")
        for iv in result.answer.intermediate_values:
            print(f"  {iv.name} = {iv.value}")
    print(f"\nEvidence ({len(result.evidence.items)} items):")
    for item in result.evidence.items:
        ref = f"  ({item.visual_ref})" if item.visual_ref else ""
        print(f"  - {item.group} / {item.category} = {item.value}{ref}")
    print(f"\nCritic consistent: {result.critic.consistent}")
    if result.critic.issues:
        for issue in result.critic.issues:
            print(f"  ! {issue}")
        if result.critic.suggested_revisit != "none":
            print(f"  -> suggested revisit: {result.critic.suggested_revisit}")
    print("=" * 68)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())