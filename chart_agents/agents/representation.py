"""Representation Agent.

Converts the raw PerceptionOutput into a clean ChartDSL using the current
group/category naming. Also preserves a mapping from
(group, category) back to the component index that produced each value,
so the Evidence Agent can cite the original visual marks.
"""

from __future__ import annotations

from crewai import Agent, LLM, Task

from chart_agents.config import settings
from chart_agents.schemas import ChartDSL


def build_representation_agent() -> Agent:
    return Agent(
        role="Chart DSL Builder",
        goal=(
            "Convert a PerceptionOutput into a clean, structured ChartDSL with "
            "group/category data and references back to the original visual marks."
        ),
        backstory=(
            "You translate raw chart perception into a canonical table-like "
            "representation. You normalize group/category names, merge redundant entries, "
            "and never hallucinate data points that did not appear in the "
            "perception output."
        ),
        llm=LLM(model=settings.llm_model, temperature=settings.llm_temperature),
        verbose=True,
        allow_delegation=False,
    )


def build_representation_task(agent: Agent, perception_task: Task) -> Task:
    description = (
        "You are given the PerceptionOutput JSON from the previous task. Convert it "
        "into a ChartDSL JSON with this exact shape (no markdown fences, no commentary):\n\n"
        "{\n"
        '  "type": "string: chart type copied from perception.chart_type",\n'
        '  "title": "string or null: copied from perception.title",\n'
        '  "x_axis": "string or null: copied from perception.x_axis_label",\n'
        '  "y_axis": "string or null: copied from perception.y_axis_label",\n'
        '  "groups": ["string: actual axis group label copied from tick labels/components"],\n'
        '  "categories": ["string: actual legend/category label copied from legend/components"],\n'
        '  "data": [\n'
        '    {"group": "string: actual x-axis or y-axis group label", "category": "string: actual legend/category label", "value": 0, "visual_mark_index": 0}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Treat the block above as schema documentation only. Do not copy literal strings like "
        "'actual axis group label', 'actual legend/category label', or generic names such as 'Product A'.\n"
        "- Preserve real chart labels exactly. `group` must come from components[*].group or visible "
        "axis tick labels such as PerceptionOutput.x_tick_labels / y_tick_labels. `category` must come "
        "from PerceptionOutput.legend or components[*].category.\n"
        "- In this schema, `group` means the axis-side label, such as the name under a bar group. "
        "`category` means the legend-side label, such as the color/category name in a grouped bar chart.\n"
        "- Never swap `group` and `category`. Never rename real labels to generic placeholders. "
        "For example, if the chart says Clothes, Books, or Furniture, keep those names exactly.\n"
        "- Emit ONE ChartDataPoint per (group, category) pair. Only include components whose "
        "`value` is non-null in the PerceptionOutput.\n"
        "- If multiple perceived components map to the same (group, category) pair, deduplicate them: "
        "emit only one ChartDataPoint for that pair and choose the component with the largest numeric value.\n"
        "- Normalize only whitespace/case inconsistencies for the same visible label; do not invent new names.\n"
        "- `visual_mark_index` is the 0-based index of the matching component in "
        "PerceptionOutput.components. Use null if no clean match exists.\n"
        "- For line/scatter charts, use the x value or x tick label as `group`.\n"
        "- For pie charts, use the slice label as `group` and a single category name like 'value' only "
        "if the chart has no visible legend/category labels.\n"
        "- Single-category charts still use the list form; just emit one point per group with that category.\n"
        "- Output JSON only."
    )
    return Task(
        description=description,
        expected_output="A ChartDSL JSON object.",
        agent=agent,
        context=[perception_task],
        output_pydantic=ChartDSL,
    )
