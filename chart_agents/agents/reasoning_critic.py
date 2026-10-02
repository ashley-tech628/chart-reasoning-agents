"""Reasoning Agent and Critic Agent.

The Reasoning Agent computes the final answer from the EvidenceSet and
emits a grounded explanation. The Critic Agent then audits the answer
against the evidence and ChartDSL, returning a CriticVerdict that flags
inconsistencies and (if needed) suggests which earlier stage to revisit.

These two are paired in one file because they share most of their context
and are conceptually a single check-and-answer step.
"""

from __future__ import annotations

import json

from crewai import Agent, LLM, Task
from pydantic import BaseModel

from chart_agents.agents.answer_type import infer_answer_type
from chart_agents.config import settings
from chart_agents.schemas import (
    ChartDSL,
    CriticVerdict,
    EvidenceSet,
    FinalAnswer,
    ParsedQuestion,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _json(obj: BaseModel) -> str:
    """Pretty JSON for embedding validated intermediate objects into prompts."""
    return obj.model_dump_json(indent=2)


def _answer_type_hint(parsed_question: ParsedQuestion, chart_dsl: ChartDSL) -> str:
    hint = infer_answer_type(parsed_question.raw_question, chart_dsl)
    return json.dumps(hint, indent=2)


# ---------------------------------------------------------------------------
# Reasoning Agent
# ---------------------------------------------------------------------------


def build_reasoning_agent() -> Agent:
    return Agent(
        role="Chart Reasoner",
        goal=(
            "Compute the final answer to the parsed question using only the EvidenceSet, "
            "and produce a short grounded explanation that references the same "
            "groups, categories, and values."
        ),
        backstory=(
            "You perform precise numerical and logical reasoning on small structured "
            "tables. You always show your intermediate work so the Critic can verify "
            "every step."
        ),
        llm=LLM(model=settings.llm_model, temperature=settings.llm_temperature),
        verbose=True,
        allow_delegation=False,
    )


def build_reasoning_task(
    agent: Agent, evidence_task: Task, parser_task: Task
) -> Task:
    description = (
        "Given the EvidenceSet and the ParsedQuestion from the previous tasks, perform the "
        "required computation. Return JSON with this exact schema (no fences, no prose):\n\n"
        "{\n"
        '  "answer": "string: concise final answer",\n'
        '  "explanation": "string: short explanation using only evidence groups, categories, and values",\n'
        '  "intermediate_values": [\n'
        '    {"name": "string: descriptive computation name", "value": 0}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Use ONLY numbers present in EvidenceSet.items[*].value. Do not pull in new data.\n"
        "- Use the exact group and category names from EvidenceSet / ChartDSL. Do not copy names from prompt examples.\n"
        "- Put every intermediate computation as an entry in `intermediate_values` (delta_*, sum_*, mean_*, etc.).\n"
        "- `explanation` must reference the same groups/categories that appear in the evidence.\n"
        "- `answer` must be short and directly gradable, using the same units/scale as the chart's y-axis for numeric answers.\n"
        "- If ParsedQuestion.operation says `count categories where any value > threshold` or `count categories where any value < threshold`, count unique categories that have at least one qualifying evidence item. Do NOT count qualifying cells/bars.\n"
        "- If the raw question contains 'at least one' or 'groups of bars', count each qualifying group/category once, even when multiple categories in that group satisfy the threshold.\n"
        "- If the raw question asks for the value of the largest/smallest individual bar, return the single max/min EvidenceSet item value. Do NOT sum values across groups or categories.\n"
        "- If the raw question asks 'which object/item/product ... summed across categories', sum values by category label and return the winning label, not the numeric sum.\n"
        "- If the raw question asks 'which algorithm has highest accuracy for any dataset', return the group or category label requested by the question for the single highest cell.\n"
        "- Output JSON only."
        
        "- STRICT GROUP/CATEGORY RULE: Use the field names in the EvidenceSet literally. "
        "`group` means the axis entry/group name from the chart, and `category` means the "
        "legend/category name from the chart. Do not reinterpret these names using older "
        "old ChartDSL terminology.\n"
        "- If the raw question asks about a group or groups, answer using evidence item "
        "`group` values only. A group-level computation should aggregate or compare items "
        "that share the same `group`.\n"
        "- If the raw question asks about a category or categories, answer using evidence item "
        "`category` values only. A category-level computation should aggregate or compare "
        "items that share the same `category`.\n"
        "- If the raw question asks for a single bar/cell/value, return the value from the "
        "single qualifying EvidenceSet item, while still using that item's exact `group` and "
        "`category` names in the explanation.\n"
        "- Do not invent, rename, or swap `group` and `category`. The final `answer`, "
        "`explanation`, and `intermediate_values` must use only exact strings that appear "
        "in EvidenceSet items.\n"
    )
    return Task(
        description=description,
        expected_output="A FinalAnswer JSON object.",
        agent=agent,
        context=[evidence_task, parser_task],
        output_pydantic=FinalAnswer,
    )


def build_reasoning_revision_task(
    agent: Agent,
    evidence: EvidenceSet,
    parsed_question: ParsedQuestion,
    previous_answer: FinalAnswer,
    critic: CriticVerdict,
) -> Task:
    """Build a reasoning-only repair task driven by the previous critic verdict.

    Use this only when the critic says the evidence is adequate but the reasoning
    step is faulty, i.e. `suggested_revisit == "none"`.
    """
    description = (
        "The previous FinalAnswer was rejected by the Critic. Revise ONLY the reasoning. "
        "Do not change the evidence, parsed question, group names, category names, or values.\n\n"
        f"ParsedQuestion JSON:\n{_json(parsed_question)}\n\n"
        f"EvidenceSet JSON:\n{_json(evidence)}\n\n"
        f"Previous FinalAnswer JSON:\n{_json(previous_answer)}\n\n"
        f"CriticVerdict JSON:\n{_json(critic)}\n\n"
        "Return a corrected FinalAnswer JSON with this exact schema:\n"
        "{\n"
        '  "answer": "string: concise final answer",\n'
        '  "explanation": "string: corrected explanation",\n'
        '  "intermediate_values": [\n'
        '    {"name": "string: descriptive computation name", "value": 0}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Address every item in CriticVerdict.issues.\n"
        "- Use ONLY EvidenceSet.items[*].value; do not invent or re-read chart values.\n"
        "- Use exact group and category strings from EvidenceSet.\n"
        "- Include all arithmetic needed to support the final answer in `intermediate_values`.\n"
        "- Output JSON only."
        "- STRICT GROUP/CATEGORY RULE: Use the field names in the EvidenceSet literally. "
        "`group` means the axis-side label, and `category` means the legend-side label. "
        "Do not invent, rename, or swap these fields.\n"
        "- If the raw question asks about a group or groups, answer using evidence item `group` values.\n"
        "- If the raw question asks about a category or categories, answer using evidence item `category` values.\n"
    )
    return Task(
        description=description,
        expected_output="A corrected FinalAnswer JSON object.",
        agent=agent,
        output_pydantic=FinalAnswer,
    )




# ---------------------------------------------------------------------------
# Short Answer Extractor Agent
# ---------------------------------------------------------------------------


def build_short_answer_extractor_agent() -> Agent:
    return Agent(
        role="Short Answer Extractor",
        goal=(
            "Derive the benchmark-style short answer from the reasoning explanation, "
            "so the concise answer is consistent with the explanation."
        ),
        backstory=(
            "You convert a correct chart-QA explanation into the shortest answer that "
            "an exact-match evaluator expects. You do not re-read the chart; you only "
            "extract the answer implied by the explanation and the question wording."
        ),
        llm=LLM(model=settings.llm_model, temperature=0.0),
        verbose=True,
        allow_delegation=False,
    )


def _short_answer_rules(answer_type_hint: str | None = None) -> str:
    hint_block = ""
    if answer_type_hint:
        hint_block = (
            "AnswerTypeHint JSON:\n"
            f"{answer_type_hint}\n\n"
            "Use this AnswerTypeHint as the source of truth for whether the final answer should be "
            "a group label, category label, yes/no, or number.\n\n"
        )
    return (
        hint_block +
        "Rewrite ONLY the `answer` field so it is derived from the previous "
        "FinalAnswer.explanation and matches the raw question. Copy the explanation "
        "and intermediate_values from the previous FinalAnswer without changing their meaning.\n\n"
        "Critical rules for `answer`:\n"
        "- First determine the expected answer type from AnswerTypeHint if present; otherwise infer it from ParsedQuestion.raw_question and ChartDSL.\n"
        "- If expected answer_type is `category`, return one exact legend-side category label from ChartDSL.data[*].category whenever possible.\n"
        "- If expected answer_type is `group`, return one exact axis-side group label from ChartDSL.data[*].group whenever possible.\n"
        "- If expected answer_type is `boolean`, return exactly `yes` or `no`.\n"
        "- If expected answer_type is `number`, return only the numeric value, not the label where it occurs.\n"
        "- Treat the explanation as the source of truth. Do not invent new chart values.\n"
        "- If the question asks 'which', 'which quarter', 'which label', 'which category', "
        "or asks for the item/name/category with max/min, return the category/name, not the numeric value.\n"
        "  Example: if the explanation says 'The largest gap occurs in Q3 ... gap of 13.0', "
        "the answer should be 'Q3', not '13.0'.\n"
        "- If the question asks 'what value', 'how many', 'how much', 'what is the difference', "
        "or directly asks for a numeric amount, return the numeric value only.\n"
        "- If the explanation gives both a label and a number, choose the part requested by the raw question.\n"
        "- Keep the answer concise: usually one label, one group, one category name, yes/no, "
        "or one number.\n"
        "- Output JSON only with exactly this schema:\n"
        "{\n"
        '  "answer": "string: short answer extracted from the explanation",\n'
        '  "explanation": "string: same explanation as the previous FinalAnswer",\n'
        '  "intermediate_values": [\n'
        '    {"name": "string", "value": 0}\n'
        "  ]\n"
        "}"
    )


def build_short_answer_extraction_task(
    agent: Agent, reasoning_task: Task, parser_task: Task, representation_task: Task
) -> Task:
    description = (
        "You will receive ChartDSL, ParsedQuestion, and a previous FinalAnswer from context.\n"
        + _short_answer_rules()
    )
    return Task(
        description=description,
        expected_output="A FinalAnswer JSON object with the answer extracted from the explanation.",
        agent=agent,
        context=[representation_task, parser_task, reasoning_task],
        output_pydantic=FinalAnswer,
    )


def build_short_answer_extraction_task_from_objects(
    agent: Agent,
    parsed_question: ParsedQuestion,
    reasoning_task: Task,
    chart_dsl: ChartDSL | None = None,
) -> Task:
    hint = _answer_type_hint(parsed_question, chart_dsl) if chart_dsl is not None else None
    description = (
        "ParsedQuestion JSON:\n"
        f"{_json(parsed_question)}\n\n"
        "You will receive the previous FinalAnswer from context.\n"
        + _short_answer_rules(hint)
    )
    return Task(
        description=description,
        expected_output="A FinalAnswer JSON object with the answer extracted from the explanation.",
        agent=agent,
        context=[reasoning_task],
        output_pydantic=FinalAnswer,
    )


# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------


def build_critic_agent() -> Agent:
    return Agent(
        role="Reasoning Critic",
        goal=(
            "Audit the final answer against the evidence set and the chart DSL. "
            "Flag inconsistencies and, if found, suggest the earlier stage most "
            "likely to be at fault."
        ),
        backstory=(
            "You play the role of an adversarial reviewer. You verify arithmetic, "
            "check that every value cited in the explanation appears in the evidence, "
            "and confirm the answer is actually supported by the intermediate values. "
            "You are strict but fair; you do not flag stylistic issues."
        ),
        llm=LLM(model=settings.llm_model, temperature=settings.llm_temperature),
        verbose=True,
        allow_delegation=False,
    )


def _critic_instructions() -> str:
    return (
        "Audit the FinalAnswer against the EvidenceSet and the ChartDSL. Return JSON of "
        "this form (no fences, no prose):\n\n"
        "{\n"
        '  "consistent": true,\n'
        '  "issues": [],\n'
        '  "suggested_revisit": "none"\n'
        "}\n\n"
        "Checklist:\n"
        "  1. Are all numbers explicitly cited in `explanation` present in EvidenceSet.items[*].value or derived in intermediate_values?\n"
        "  2. Do `intermediate_values` arithmetically match the values from the evidence set?\n"
        "  3. Does `answer` match the type requested by ParsedQuestion.raw_question, using these concrete expectations: label/group/category for which/max/min item questions, number for numeric questions, yes/no for boolean questions? If the question asks `which object`, `which algorithm`, `which item`, `which product`, or `which category` and the answer is a ChartDSL category label, that is a correct label-type answer; do NOT flag it as a numeric type mismatch.\n"
        "  4. Is the final answer supported by the intermediate values and explanation?\n"
        "  5. Do referenced group/category names exist in ChartDSL?\n\n"
        "Answer-type rules:\n"
        "- Use ParsedQuestion.raw_question and ChartDSL to infer the expected answer type.\n"
        "- For `which product`, `which item`, `which algorithm`, `which object`, or `which category`, expect a category/item label if ChartDSL has non-generic categories; otherwise expect a group label.\n"
        "- For `which group`, `which bar`, or generic axis-label `which ...` questions, expect a group/bar label unless the ParsedQuestion explicitly says otherwise.\n"
        "- For label-type `which ...` questions, do not require a numeric final answer just because the reasoning uses numeric intermediate values such as max, min, total, sum, or gap. The number supports selecting the label; it is not the requested final answer.\n"
        "- For yes/no questions beginning with is/are/does/do/did/was/were, accept only `yes` or `no`.\n"
        "- For value/difference/sum/count questions beginning with what/how many/how much, expect a number.\n"
        "- Treat normalized numeric forms as equivalent, e.g. `7`, `7.0`, and `7.000`.\n"
        "- Treat boolean aliases as answer-format errors only: `true` should become `yes`, and `false` should become `no`.\n\n"
        "Important fairness rules:\n"
        "- Do NOT require the explanation to mention every intermediate value. It only needs to mention enough evidence to support the final answer.\n"
        "- Do NOT mark an answer inconsistent just because non-winning categories or non-final computations are omitted from the explanation, as long as they appear in intermediate_values when needed.\n"
        "- Do NOT flag cosmetic wording such as saying `category` when the winning label is a group/bar label, or saying `group` when the winning label is a category/item label, if the named label and value are supported by ChartDSL and the final answer has the requested label type.\n"
        "- Do NOT raise a contradiction if the cited comparison supports the boolean answer, e.g. if the evidence shows 1.0 < 5.0 and the answer is `yes` to an `is ... smaller than ...` question.\n"
        "- If an issue says a value is incorrect but the expected value is identical, do not flag it.\n"
        "- If the explanation or intermediate_values compute the same result as the final answer after normalization, do not claim the answer fails to match that computation.\n"
        "- For questions asking 'which', a label answer can be correct even if max_gap is numeric in intermediate_values.\n\n"
        "If everything is consistent, return `consistent: true`, empty `issues`, and\n"
        "`suggested_revisit: \"none\"`.\n"
        "Otherwise set `consistent: false`, list every real issue as a short string, and set\n"
        "`suggested_revisit` to the single most useful stage to redo:\n"
        "  - 'none'            -- only final reasoning or answer formatting is at fault; use this for label-vs-number, true-vs-yes, false-vs-no, or arithmetic errors when evidence is adequate\n"
        "  - 'evidence'        — evidence set is missing or wrong\n"
        "  - 'representation'  — ChartDSL values look wrong / mis-grouped\n"
        "  - 'perception'      — underlying perception is faulty (e.g. wrong chart_type)\n"
        "Output JSON only."
    )


def build_critic_task(
    agent: Agent,
    reasoning_task: Task,
    evidence_task: Task,
    representation_task: Task,
    parser_task: Task,
) -> Task:
    return Task(
        description=(
            "You will receive FinalAnswer, EvidenceSet, ChartDSL, and ParsedQuestion as context.\n\n"
            + _critic_instructions()
        ),
        expected_output="A CriticVerdict JSON object.",
        agent=agent,
        context=[reasoning_task, evidence_task, representation_task, parser_task],
        output_pydantic=CriticVerdict,
    )


def build_critic_task_from_objects(
    agent: Agent,
    reasoning_task: Task,
    evidence: EvidenceSet,
    chart_dsl: ChartDSL,
    parsed_question: ParsedQuestion | None = None,
) -> Task:
    """Critic task for a revised answer where evidence/DSL are fixed objects."""
    parsed_block = ""
    if parsed_question is not None:
        parsed_block = (
            f"ParsedQuestion JSON:\n{_json(parsed_question)}\n\n"
            f"AnswerTypeHint JSON:\n{_answer_type_hint(parsed_question, chart_dsl)}\n\n"
        )
    description = (
        "Audit the revised FinalAnswer from the previous task against these fixed inputs.\n\n"
        f"EvidenceSet JSON:\n{_json(evidence)}\n\n"
        f"ChartDSL JSON:\n{_json(chart_dsl)}\n\n"
        + parsed_block
        + _critic_instructions()
    )
    return Task(
        description=description,
        expected_output="A CriticVerdict JSON object.",
        agent=agent,
        context=[reasoning_task],
        output_pydantic=CriticVerdict,
    )
