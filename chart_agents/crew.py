"""ChartReasoningCrew — orchestrates the five-agent pipeline.

Build all agents, wire up the sequential tasks, run the crew, and collect every
stage's Pydantic output into a single PipelineResult.

If `save_intermediates=True`, every stage's output is written as JSON under
`outputs/<timestamp>-<tag>/`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from crewai import Crew, Process
from pydantic import BaseModel, ValidationError

from chart_agents.agents.evidence import build_evidence_agent, build_evidence_task
from chart_agents.agents.perception import (
    build_perception_agent,
    build_perception_task,
)
from chart_agents.agents.question_parser import (
    build_question_parser_agent,
    build_question_parser_task,
)
from chart_agents.agents.reasoning_critic import (
    build_critic_agent,
    build_critic_task,
    build_critic_task_from_objects,
    build_short_answer_extraction_task,
    build_short_answer_extraction_task_from_objects,
    build_short_answer_extractor_agent,
    build_reasoning_agent,
    build_reasoning_revision_task,
    build_reasoning_task,
)
from chart_agents.agents.representation import (
    build_representation_agent,
    build_representation_task,
)
from chart_agents.config import settings
from chart_agents.schemas import (
    ChartDSL,
    CriticVerdict,
    EvidenceSet,
    FinalAnswer,
    ParsedQuestion,
    PerceptionOutput,
    PipelineResult,
)
from chart_agents.solvers.deterministic_overrides import (
    try_solve_sum_value,
    try_solve_minmax_numeric_value,
    try_solve_threshold_count,
)
from chart_agents.utils import new_run_dir, parse_json_safe, save_intermediate


log = logging.getLogger("chart_agents.crew")


def _coerce_task_output(task, model: type[BaseModel]) -> BaseModel:
    """Return a parsed Pydantic instance from a finished CrewAI Task."""
    out = task.output
    if out is None:
        raise RuntimeError(f"Task {task!r} produced no output.")

    pyd = getattr(out, "pydantic", None)
    if isinstance(pyd, model):
        return pyd

    raw = getattr(out, "raw", None) or str(out)
    try:
        data = parse_json_safe(raw)
        return model.model_validate(data)
    except (ValueError, ValidationError) as e:
        raise RuntimeError(
            f"Could not coerce task output to {model.__name__}: {e}\n--- raw ---\n{raw[:1500]}"
        ) from e


def _unique_categories(chart_dsl: ChartDSL) -> list[str]:
    values: list[str] = []
    for row in getattr(chart_dsl, "data", []) or []:
        category = getattr(row, "category", None)
        if category is not None:
            category = str(category)
            if category not in values:
                values.append(category)
    return values


def _unique_series(chart_dsl: ChartDSL) -> list[str]:
    values: list[str] = []
    for row in getattr(chart_dsl, "data", []) or []:
        series = getattr(row, "series", None)
        if series is not None:
            series = str(series)
            if series not in values:
                values.append(series)

    declared = getattr(chart_dsl, "series", None)
    if isinstance(declared, list):
        for item in declared:
            item = str(item)
            if item not in values:
                values.append(item)

    return values


def infer_answer_type(question: str, chart_dsl: ChartDSL) -> dict[str, Any]:
    """Small local answer-type heuristic used by deterministic overrides."""
    q = question.lower()
    categories = _unique_categories(chart_dsl)
    series = _unique_series(chart_dsl)

    valid_answers = categories + [s for s in series if s not in categories]

    if q.startswith(("is ", "are ", "does ", "do ", "was ", "were ")):
        return {
            "answer_type": "boolean",
            "expected_format": "yes/no",
            "valid_answers": ["yes", "no"],
        }

    if "how many" in q or "number of" in q or "count" in q:
        return {
            "answer_type": "number",
            "expected_format": "integer count",
            "valid_answers": [],
        }

    if (
        q.startswith("which ")
        or "which " in q
        or "what algorithm" in q
        or "what item" in q
        or "what product" in q
        or "what category" in q
        or "what group" in q
    ):
        return {
            "answer_type": "category",
            "expected_format": "category or label",
            "valid_answers": valid_answers,
        }

    return {
        "answer_type": "number",
        "expected_format": "number",
        "valid_answers": valid_answers,
    }


def normalize_answer(answer: Any, answer_type: str, valid_answers: list[str]) -> str:
    """Normalize final answer for DVQA-style evaluation."""
    text = str(answer).strip()

    lower = text.lower()
    if answer_type == "boolean":
        if lower in {"true", "yes", "y"}:
            return "yes"
        if lower in {"false", "no", "n"}:
            return "no"

    try:
        value = float(text)
        if value.is_integer():
            return str(int(value))
    except Exception:
        pass

    for candidate in valid_answers:
        if text.lower() == str(candidate).lower():
            return str(candidate)

    return text


class ChartReasoningCrew:
    """End-to-end grounded chart reasoning pipeline."""

    def __init__(self, save_intermediates: Optional[bool] = None):
        self.save = (
            settings.save_intermediates if save_intermediates is None else save_intermediates
        )

    def run(
        self,
        image_path: str,
        question: str,
        tag: Optional[str] = None,
    ) -> PipelineResult:
        run_dir: Optional[Path] = None
        if self.save:
            run_dir = new_run_dir(settings.output_dir, tag)
            log.info("Run directory: %s", run_dir)
            save_intermediate(
                {"image_path": image_path, "question": question}, "0_input", run_dir
            )

        # ----- Build agents -----
        perception_agent = build_perception_agent()
        representation_agent = build_representation_agent()
        question_parser_agent = build_question_parser_agent()
        evidence_agent = build_evidence_agent()
        reasoning_agent = build_reasoning_agent()
        short_answer_agent = build_short_answer_extractor_agent()
        critic_agent = build_critic_agent()

        # ----- Build tasks -----
        perception_task = build_perception_task(perception_agent, image_path)
        representation_task = build_representation_task(
            representation_agent, perception_task
        )
        question_parser_task = build_question_parser_task(
            question_parser_agent, question, representation_task
        )
        evidence_task = build_evidence_task(
            evidence_agent, representation_task, question_parser_task
        )
        reasoning_task = build_reasoning_task(
            reasoning_agent, evidence_task, question_parser_task
        )
        short_answer_task = build_short_answer_extraction_task(
            short_answer_agent, reasoning_task, question_parser_task, representation_task,
        )
        critic_task = build_critic_task(
            critic_agent,
            short_answer_task,
            evidence_task,
            representation_task,
            question_parser_task,
        )

        crew = Crew(
            agents=[
                perception_agent,
                representation_agent,
                question_parser_agent,
                evidence_agent,
                reasoning_agent,
                short_answer_agent,
                critic_agent,
            ],
            tasks=[
                perception_task,
                representation_task,
                question_parser_task,
                evidence_task,
                reasoning_task,
                short_answer_task,
                critic_task,
            ],
            process=Process.sequential,
            verbose=True,
        )

        crew.kickoff(inputs={"image_path": image_path, "question": question})

        # ----- Collect outputs -----
        perception = _coerce_task_output(perception_task, PerceptionOutput)
        chart_dsl = _coerce_task_output(representation_task, ChartDSL)
        parsed_question = _coerce_task_output(question_parser_task, ParsedQuestion)
        evidence = _coerce_task_output(evidence_task, EvidenceSet)
        raw_answer = _coerce_task_output(reasoning_task, FinalAnswer)
        final_answer = _coerce_task_output(short_answer_task, FinalAnswer)
        critic = _coerce_task_output(critic_task, CriticVerdict)

        # ----- Deterministic answer repair / override -----
        final_answer = self._apply_sum_value_override(final_answer, question, chart_dsl)
        final_answer = self._apply_minmax_numeric_override(final_answer, question, chart_dsl)
        final_answer = self._apply_threshold_count_override(final_answer, question, chart_dsl)
        final_answer = self._normalize_final_answer(final_answer, question, chart_dsl)

        result = PipelineResult(
            perception=perception,
            chart_dsl=chart_dsl,
            parsed_question=parsed_question,
            evidence=evidence,
            answer=final_answer,
            critic=critic,
        )

        if run_dir is not None:
            save_intermediate(result.perception, "1_perception", run_dir)
            save_intermediate(result.chart_dsl, "2_representation", run_dir)
            save_intermediate(result.parsed_question, "3_question", run_dir)
            save_intermediate(result.evidence, "4_evidence", run_dir)
            save_intermediate(raw_answer, "5_answer_raw", run_dir)
            save_intermediate(result.answer, "5_answer", run_dir)
            save_intermediate(result.critic, "6_critic", run_dir)

        # ----- Optional critic-driven reasoning repair -----
        for step in range(settings.max_refinement_steps):
            if result.critic.consistent:
                break

            if result.critic.suggested_revisit != "none":
                log.info(
                    "Critic suggested revisiting %s; skipping reasoning-only repair.",
                    result.critic.suggested_revisit,
                )
                break

            revised_answer, revised_critic = self._revise_reasoning_once(result)

            revised_answer = self._apply_sum_value_override(
                revised_answer, question, result.chart_dsl
            )
            revised_answer = self._apply_minmax_numeric_override(
                revised_answer, question, result.chart_dsl
            )
            revised_answer = self._apply_threshold_count_override(
                revised_answer, question, result.chart_dsl
            )
            revised_answer = self._normalize_final_answer(
                revised_answer, question, result.chart_dsl
            )

            result = result.model_copy(
                update={"answer": revised_answer, "critic": revised_critic}
            )

            if run_dir is not None:
                idx = step + 1
                save_intermediate(result.answer, f"5_answer_refined_{idx}", run_dir)
                save_intermediate(result.critic, f"6_critic_refined_{idx}", run_dir)

        if run_dir is not None:
            save_intermediate(result, "result", run_dir)
            log.info("Saved intermediates to %s", run_dir)

        return result

    def _normalize_final_answer(
        self,
        answer: FinalAnswer,
        question: str,
        chart_dsl: ChartDSL,
    ) -> FinalAnswer:
        answer_type = infer_answer_type(question, chart_dsl)
        normalized = normalize_answer(
            answer.answer,
            answer_type.get("answer_type", "unknown"),
            answer_type.get("valid_answers", []),
        )
        if normalized == answer.answer:
            return answer
        return answer.model_copy(update={"answer": normalized})

    def _apply_sum_value_override(
        self,
        answer: FinalAnswer,
        question: str,
        chart_dsl: ChartDSL,
    ) -> FinalAnswer:
        answer_type = infer_answer_type(question, chart_dsl)
        override = try_solve_sum_value(question, chart_dsl, answer_type)
        if override is None:
            return answer
        return override

    def _apply_minmax_numeric_override(
        self,
        answer: FinalAnswer,
        question: str,
        chart_dsl: ChartDSL,
    ) -> FinalAnswer:
        answer_type = infer_answer_type(question, chart_dsl)
        override = try_solve_minmax_numeric_value(question, chart_dsl, answer_type)
        if override is None:
            return answer
        return override

    def _apply_threshold_count_override(
        self,
        answer: FinalAnswer,
        question: str,
        chart_dsl: ChartDSL,
    ) -> FinalAnswer:
        answer_type = infer_answer_type(question, chart_dsl)
        override = try_solve_threshold_count(question, chart_dsl, answer_type)
        if override is None:
            return answer
        return override

    def _revise_reasoning_once(
        self, result: PipelineResult
    ) -> tuple[FinalAnswer, CriticVerdict]:
        """Run one critic-driven reasoning-only repair pass."""
        reasoning_agent = build_reasoning_agent()
        short_answer_agent = build_short_answer_extractor_agent()
        critic_agent = build_critic_agent()

        revision_task = build_reasoning_revision_task(
            reasoning_agent,
            evidence=result.evidence,
            parsed_question=result.parsed_question,
            previous_answer=result.answer,
            critic=result.critic,
        )
        short_answer_task = build_short_answer_extraction_task_from_objects(
            short_answer_agent,
            parsed_question=result.parsed_question,
            reasoning_task=revision_task,
            chart_dsl=result.chart_dsl,
        )
        critic_task = build_critic_task_from_objects(
            critic_agent,
            reasoning_task=short_answer_task,
            evidence=result.evidence,
            chart_dsl=result.chart_dsl,
        )

        crew = Crew(
            agents=[reasoning_agent, short_answer_agent, critic_agent],
            tasks=[revision_task, short_answer_task, critic_task],
            process=Process.sequential,
            verbose=True,
        )
        crew.kickoff()

        return (
            _coerce_task_output(short_answer_task, FinalAnswer),
            _coerce_task_output(critic_task, CriticVerdict),
        )