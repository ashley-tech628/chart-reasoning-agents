"""Round-based blackboard workflow orchestrator.

This file is the only place that controls round order. Individual agents live
in separate modules, so future teammates can add solvers/critics without
editing one huge script.
"""

from __future__ import annotations

from typing import List, Tuple

from chart_agents.agents.answer_type import AnswerTypeAgent
from chart_agents.agents.judge import SimpleJudgeAgent
from chart_agents.agents.router import QuestionRouterAgent
from chart_agents.blackboard import BlackboardState
from chart_agents.critics.evidence_critic import EvidenceCritic
from chart_agents.critics.format_critic import FormatCritic
from chart_agents.schemas import ChartDSL, FinalAnswer, IntermediateValue
from chart_agents.solvers.registry import SOLVER_REGISTRY


class BlackboardWorkflow:
    """MVP round-based multi-agent workflow.

    Current MVP rounds:
    - Round 1: Router + Answer-Type Agent
    - Round 2: selected solver branches from SOLVER_REGISTRY
    - Round 3: Format Critic + Evidence Critic
    - Round 4: Simple Judge
    """

    def __init__(self) -> None:
        self.router = QuestionRouterAgent()
        self.answer_type = AnswerTypeAgent()
        self.format_critic = FormatCritic()
        self.evidence_critic = EvidenceCritic()
        self.judge = SimpleJudgeAgent()

    def run(self, question: str, chart_dsl: ChartDSL) -> Tuple[BlackboardState, FinalAnswer]:
        state = BlackboardState(question=question, chart_dsl=chart_dsl)

        # Round 1: interpretation/routing. These agents read the same state and post claims.
        self.router.run(state)
        self.answer_type.run(state)

        # Round 2: candidate generation. Activate only selected solver branches.
        selected = self._selected_solvers(state)
        for solver_name in selected:
            solver = SOLVER_REGISTRY.get(solver_name)
            if solver is not None:
                solver.run(state)

        # Round 3: critique. Critics add support/objection messages.
        self.format_critic.run(state)
        self.evidence_critic.run(state)

        # Round 4: consensus. Judge reads candidates + critiques and writes final answer.
        self.judge.run(state)

        return state, self._to_final_answer(state)

    @staticmethod
    def _selected_solvers(state: BlackboardState) -> List[str]:
        router_msg = state.latest_claim_from("question_router")
        if router_msg is None or not isinstance(router_msg.content, dict):
            return ["comparison_solver", "arithmetic_solver"]
        required = list(router_msg.content.get("required_solvers", []) or [])
        # For MVP, include optional arithmetic/comparison only when present and available.
        optional = list(router_msg.content.get("optional_solvers", []) or [])
        names = []
        for name in required + optional:
            if name in SOLVER_REGISTRY and name not in names:
                names.append(name)
        return names or ["comparison_solver", "arithmetic_solver"]

    @staticmethod
    def _to_final_answer(state: BlackboardState) -> FinalAnswer:
        intermediate_values: list[IntermediateValue] = []
        seen_names = set()
        for cand in state.candidates():
            if not cand.support:
                continue
            details = cand.support.get("details", {}) or {}
            for name, value in details.items():
                if str(name).startswith("gap_") and name not in seen_names:
                    try:
                        intermediate_values.append(IntermediateValue(name=str(name), value=float(value)))
                        seen_names.add(str(name))
                    except Exception:
                        pass
        return FinalAnswer(
            answer=state.final_answer or "",
            explanation=state.final_explanation or "",
            intermediate_values=intermediate_values,
        )


def run_blackboard_workflow(question: str, chart_dsl: ChartDSL) -> Tuple[BlackboardState, FinalAnswer]:
    return BlackboardWorkflow().run(question, chart_dsl)
