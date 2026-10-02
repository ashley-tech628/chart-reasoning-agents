"""Vision tool for the Perception Agent.

Wraps a vision-language model call via litellm so any provider that supports
images works (OpenAI, Anthropic, Google, ...). The prompt is baked into the
tool so the agent only has to supply the image path.

Swap this out (e.g. for DePlot, Pix2Struct, or a chart-specific detector) to
upgrade the perception stage without touching the rest of the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Type

import litellm
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from chart_agents.config import settings
from chart_agents.utils import encode_image_b64


PERCEPTION_PROMPT = """You are analyzing a chart image. Extract every visible \
chart element in detail and return a STRICT JSON object (no markdown fences, \
no prose) matching this schema:

{
  "chart_type": "bar_chart | line_chart | scatter_plot | pie_chart | area_chart | other",
  "title": "<chart title or null>",
  "x_axis_label": "<x label or null>",
  "y_axis_label": "<y label or null>",
  "x_tick_labels": ["..."],
  "y_tick_labels": ["..."],
  "legend": ["<category_1>", "<category_2>", ...],
  "components": [
    {
      "type": "bar | line_segment | point | legend_entry | axis_label | tick | other",
      "label": "<text label or null>",
      "value": <number or null>,
      "group": "<axis-side group label or null>",
      "category": "<legend-side category label or null>",
      "bbox": {"x": <num>, "y": <num>, "w": <num>, "h": <num>}
    }
  ],
  "notes": "<short notes only if the chart is ambiguous, occluded, or unusual>"
}

Rules:
- Estimate numeric `value`s by reading off tick marks; mark uncertainty in `notes`.
- bbox coordinates can be normalised (0..1) or pixels; be consistent.
- Every distinct bar/point/line-segment should appear as its own component
  with its (group, category, value).
- `group` means the axis-side label, such as an x-axis bar group or y-axis bucket.
- `category` means the legend-side label, such as a color/legend category in a grouped chart.
- Copy visible text exactly for legend, group, category, title, axis labels, and tick labels.
- Do NOT use generic placeholder names such as "Product A", "Product B", "Category 1", or "Q1"
  unless those exact words are visibly printed in the chart.
- If a visible label is unreadable, use a stable neutral placeholder such as "unreadable_group_0"
  or "unreadable_category_0" and explain the uncertainty in notes.
- Include legend entries and axis labels as components too.
- Output JSON ONLY. No commentary, no markdown fences."""


class VisionToolInput(BaseModel):
    image_path: str = Field(..., description="Local path to the chart image to analyze.")


class ChartVisionTool(BaseTool):
    name: str = "chart_vision"
    description: str = (
        "Analyze a chart image with a vision-language model. Pass `image_path` (a local "
        "filesystem path). Returns a JSON string describing chart type, title, axes, "
        "tick labels, legend, and every detected bar/line/point with its group, "
        "category, estimated value, and approximate bounding box."
    )
    args_schema: Type[BaseModel] = VisionToolInput

    def _run(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            return json.dumps({"error": f"Image file not found: {image_path}"})

        try:
            b64, media_type = encode_image_b64(path)
        except Exception as e:  # pragma: no cover
            return json.dumps({"error": f"Could not read image: {e}"})

        try:
            response = litellm.completion(
                model=settings.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PERCEPTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{b64}"},
                            },
                        ],
                    }
                ],
                temperature=settings.vision_temperature,
                max_tokens=settings.vision_max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:  # pragma: no cover
            return json.dumps({"error": f"Vision call failed: {e}"})
