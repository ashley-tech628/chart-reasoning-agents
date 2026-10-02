"""Question Parser Agent.

Maps a natural-language question into a structured operation over the
ChartDSL. Output schema: ParsedQuestion = (task, target_groups,
target_categories, operation, raw_question).
"""

from __future__ import annotations

from crewai import Agent, LLM, Task

from chart_agents.config import settings
from chart_agents.schemas import ParsedQuestion


def build_question_parser_agent() -> Agent:
    return Agent(
        role="Question Parser",
        goal=(
            "Parse a natural-language question about a chart into a typed operation "
            "over the ChartDSL, drawn from a fixed task taxonomy."
        ),
        backstory=(
            "You are a semantic parser specialised in tabular and chart-style "
            "questions. You map free-form questions to a small set of operation "
            "types and identify which groups and categories the question refers to."
        ),
        llm=LLM(model=settings.llm_model, temperature=settings.llm_temperature),
        verbose=True,
        allow_delegation=False,
    )


def build_question_parser_task(
    agent: Agent, question: str, representation_task: Task
) -> Task:
    description = (
        f"The user question is: {question!r}\n\n"
        "Using the ChartDSL from the previous task as context (its exact group names, "
        "category names, and axes), produce a ParsedQuestion JSON with this exact schema "
        "(no fences, no prose):\n\n"
        "{\n"
        '  "task": "value_lookup | max | min | comparison | trend | interval_change | counting | multi_condition_filter | other",\n'
        '  "target_groups": ["string: exact axis-side group name from ChartDSL.groups or ChartDSL.data[*].group"],\n'
        '  "target_categories": ["string: exact legend-side category name from ChartDSL.categories or ChartDSL.data[*].category"],\n'
        '  "operation": "string: concise symbolic operation over the exact chart names",\n'
        f'  "raw_question": "{question}"\n'
        "}\n\n"
        "Definitions:\n"
        "- `group` means the axis-side label, such as an x-axis bar group or y-axis bucket.\n"
        "- `category` means the legend-side label, such as a color/legend category in a grouped chart.\n\n"
        "Rules:\n"
        "- Treat the block above as schema documentation only. Do not copy placeholder text into the output.\n"
        "- target_groups MUST use exact group names that appear in ChartDSL.groups or ChartDSL.data[*].group when possible.\n"
        "- target_categories MUST use exact category names that appear in ChartDSL.categories or ChartDSL.data[*].category when possible.\n"
        "- Use an empty target list to mean 'all groups' or 'all categories'.\n"
        "- Do not introduce generic names such as Product A/Product B or Q1/Q2 unless those exact strings appear in ChartDSL.\n"
        "- If the question refers to a visible label by synonym or paraphrase, map it to the closest exact ChartDSL name.\n"
        "- If the question asks 'which group/groups', prefer operations over ChartDSL.groups, "
        "for example max(group) over all categories or argmax over groups.\n"
        "- If the question asks 'which category/categories', prefer operations over ChartDSL.categories, "
        "for example max(category) over all groups or argmax over categories.\n"
        "- For counting questions containing 'at least one' or 'groups of bars', do NOT describe the operation as a direct count of matching cells/bars. "
        "Use a grouped-any operation instead: `count groups where any value > threshold` for groups of bars/items, "
        "counting each group label once even if multiple bars in that group satisfy the condition.\n"
        "- operation: concise symbolic description with plain math operators and exact chart names, e.g. "
        "lookup(<group>, <category>), max(<category>) over all groups, max(<category_1> - <category_2>) over all groups, "
        "trend(<category>) across all groups, count groups where any value > <threshold>, "
        "count groups where any value < <threshold>, or sum where category=<category> and group in <groups>.\n"
        "- If you cannot map cleanly, set task='other' and write a clear operation string describing the ambiguity.\n"
        "- Output JSON only."
    )
    return Task(
        description=description,
        expected_output="A ParsedQuestion JSON object.",
        agent=agent,
        context=[representation_task],
        output_pydantic=ParsedQuestion,
    )
