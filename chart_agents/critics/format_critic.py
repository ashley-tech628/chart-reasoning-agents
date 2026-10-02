"""Round 3 Format Critic Agent."""

from __future__ import annotations

from chart_agents.agents.answer_type import get_answer_type_claim
from chart_agents.agents.normalizer import normalize_answer
from chart_agents.blackboard import BlackboardState, make_objection, make_support
from chart_agents.solvers.utils import exact_valid_label, is_numeric


class FormatCritic:
    name = "answer_format_critic"

    def run(self, state: BlackboardState):
        answer_type = get_answer_type_claim(state)
        expected = answer_type.get("answer_type", "unknown")
        valid = answer_type.get("valid_answers", []) or []
        messages = []

        for cand in state.candidates():
            ans = cand.answer or ""
            if not ans:
                continue

            if expected in {"group", "category", "label"} and is_numeric(ans):
                content = (
                    f"{ans} is a correct numeric/intermediate value, but the question expects "
                    f"{expected}: {answer_type.get('expected_format', '')}."
                )
                messages.append(state.post(make_objection(self.name, cand.from_agent, content, severity="high", confidence=0.98, round_id=3)))
            elif expected in {"group", "category", "label"} and exact_valid_label(ans, valid) is not None:
                content = f"{ans} matches the expected answer type ({expected}) and is in the valid answer space."
                messages.append(state.post(make_support(self.name, cand.from_agent, content, confidence=0.95, round_id=3)))
            elif expected in {"number", "count"} and is_numeric(ans):
                content = f"{ans} is numeric, matching the expected answer type."
                messages.append(state.post(make_support(self.name, cand.from_agent, content, confidence=0.90, round_id=3)))
            elif expected == "boolean" and ans in {"yes", "no"}:
                content = f"{ans} is yes/no, matching the expected boolean answer type."
                messages.append(state.post(make_support(self.name, cand.from_agent, content, confidence=0.90, round_id=3)))

        return messages
