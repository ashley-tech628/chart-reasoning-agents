"""Smoke-test the offline CV perception backend without calling any LLM/API.

Usage:
    python scripts/smoke_offline_perception.py --image examples/sample_chart.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chart_agents.schemas import PerceptionOutput  # noqa: E402
from chart_agents.tools.offline_vision import analyze_chart_image_offline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="examples/sample_chart.png")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    result = analyze_chart_image_offline(args.image)
    if "error" in result:
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 2

    # Validate against the same schema used by the CrewAI perception task.
    parsed = PerceptionOutput.model_validate(result)
    payload = parsed.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if "ocr_unavailable=" in (parsed.notes or "") or "ocr_returned_no_words" in (parsed.notes or ""):
        print(
            "WARNING: OCR did not run successfully. Numeric bar geometry may still work, "
            "but title/category/legend text will be missing or generic. Run "
            "`python scripts/check_ocr_setup.py`.",
            file=sys.stderr,
        )

    bar_count = sum(1 for c in parsed.components if c.type == "bar")
    if parsed.chart_type != "bar_chart" or bar_count == 0:
        print("Smoke test failed: no bars detected.", file=sys.stderr)
        return 1

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
