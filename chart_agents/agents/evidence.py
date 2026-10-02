"""Evidence Agent.

Selects the minimal set of (group, category, value, visual_ref) tuples
needed to answer the ParsedQuestion. Values are taken verbatim from the
ChartDSL; visual_ref points back to a component index when available.
"""

from __future__ import annotations

from crewai import Agent, LLM, Task

from chart_agents.config import settings
from chart_agents.schemas import EvidenceSet


def build_evidence_agent() -> Agent:
    return Agent(
        role="Evidence Selector",
        goal=(
            "Select the minimal set of chart data points needed to answer the parsed "
            "question, drawn strictly from the ChartDSL, with references back to the "
            "original visual marks."
        ),
        backstory=(
            "You are precise and parsimonious. You include every data point the answer "
            "depends on, and nothing more. You never invent values: if a value is not "
            "in ChartDSL.data, you do not include it."
        ),
        llm=LLM(model=settings.llm_model, temperature=settings.llm_temperature),
        verbose=True,
        allow_delegation=False,
    )


def build_evidence_task(
    agent: Agent, representation_task: Task, parser_task: Task
) -> Task:
    description = (
        "You have access to two prior outputs:\n"
        "  - ChartDSL (from the Representation Agent)\n"
        "  - ParsedQuestion (from the Question Parser Agent)\n\n"
        "Select the minimal evidence set needed to answer the question. Return JSON of this form "
        "(no fences, no prose):\n\n"
        "{\n"
        '  "items": [\n'
        '    {"group": "Q1", "category": "Product A", "value": 42, "visual_ref": "components[3]"},\n'
        '    {"group": "Q1", "category": "Product B", "value": 35, "visual_ref": "components[7]"}\n'
        "  ],\n"
        '  "rationale": "Need values of Product A and Product B in Q1 to compare them."\n'
        "}\n\n"
        "Definitions:\n"
        "- `group` means the axis-side label, such as an x-axis bar group or y-axis group.\n"
        "- `category` means the legend-side label, such as the color/category name in a grouped chart.\n"
        "- Use these names literally. Do not emit any old field names.\n\n"
        "Rules:\n"
        "- `value` MUST come verbatim from a ChartDataPoint in ChartDSL.data — find the entry whose "
        "`group` and `category` match. Do not round, do not invent.\n"
        "- Preserve exact `group` and `category` strings from ChartDSL.data. Do not rename, swap, or "
        "normalize them beyond trivial whitespace consistency.\n"
        "- For comparison / trend / interval_change / max / min tasks over all groups, include every "
        "relevant group for each target category.\n"
        "- For comparison / trend / interval_change / max / min tasks over all categories, include every "
        "relevant category for each target group.\n"
        "- For questions asking for the largest/smallest individual bar or max/min over the whole chart, "
        "include the individual candidate cells needed to identify the single largest/smallest cell. "
        "Do NOT convert this into a sum across groups or categories.\n"
        "- For max(value) - min(value) or questions comparing the most and least accurate/sold/etc., "
        "include the true global maximum cell and true global minimum cell from ChartDSL.data.\n"
        "- For value_lookup, include only the requested (group, category) pair.\n"
        "- For counting and multi_condition_filter, include every cell that satisfies the predicate.\n"
        "- For grouped-any counting operations such as `count groups where any value > threshold`, "
        "`count groups where any value < threshold`, or questions containing 'at least one' / "
        "'groups of bars', include at least one satisfying cell for EVERY qualifying group. "
        "Count each group once even if multiple cells in that group satisfy the predicate.\n"
        "- For category-any counting operations such as `count categories where any value > threshold` "
        "or `count categories where any value < threshold`, include at least one satisfying cell for "
        "EVERY qualifying category. Count each category once even if multiple groups in that category "
        "satisfy the predicate.\n"
        "- If the question asks which category/object/item/product has the largest or smallest total "
        "summed across groups, include all groups for each candidate category needed to compute the sums.\n"
        "- If the question asks which group has the largest or smallest total summed across categories, "
        "include all categories for each candidate group needed to compute the sums.\n"
        "- `visual_ref`: if the matching ChartDataPoint has a non-null `visual_mark_index` N, set this "
        "to `\"components[N]\"`. Otherwise omit it or set it to null.\n"
        "- `rationale`: one sentence stating why this set is sufficient.\n"
        "- Output JSON only."
    )
    return Task(
        description=description,
        expected_output="An EvidenceSet JSON object.",
        agent=agent,
        context=[representation_task, parser_task],
        output_pydantic=EvidenceSet,
    )
