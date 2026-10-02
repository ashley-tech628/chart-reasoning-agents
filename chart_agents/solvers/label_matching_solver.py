"""Round 2 Label / Group / Category / Legend Solver Agent."""

from __future__ import annotations

from chart_agents.agents.answer_type import get_answer_type_claim
from chart_agents.agents.normalizer import normalize_answer
from chart_agents.blackboard import BlackboardState, make_candidate
from chart_agents.solvers.utils import mentioned_labels, solver_is_active, unique_categories, unique_groups


class LabelMatchingSolver:
    name = "label_matching_solver"

    def run(self, state: BlackboardState):
        if not solver_is_active(state, self.name):
            return None

        q = state.question.lower()
        groups = unique_groups(state.chart_dsl)
        categories = unique_categories(state.chart_dsl)
        answer_type = get_answer_type_claim(state)
        expected = answer_type.get("answer_type", "unknown")
        valid = answer_type.get("valid_answers", []) or []

        answer = None
        explanation = "Label solver could not find a safe label answer."
        confidence = 0.20

        hit_groups = mentioned_labels(state.question, groups)
        hit_categories = mentioned_labels(state.question, categories)

        if "how many" in q and ("legend" in q or "categor" in q):
            answer = str(len(categories))
            explanation = f"The chart has {len(categories)} legend/category label(s)."
            confidence = 0.78
        elif "how many" in q and ("group" in q or "x-axis" in q or "x axis" in q):
            answer = str(len(groups))
            explanation = f"The chart has {len(groups)} group label(s)."
            confidence = 0.78
        elif "legend" in q or "category" in q or "product" in q:
            if hit_categories:
                answer = hit_categories[0]
                explanation = f"The mentioned legend/category label is {answer}."
                confidence = 0.72
        elif "group" in q or "label" in q or "x-axis" in q or "x axis" in q:
            if hit_groups:
                answer = hit_groups[0]
                explanation = f"The mentioned group label is {answer}."
                confidence = 0.72

        ans = normalize_answer(answer, expected, valid) if answer is not None else None
        support = {
            "details": {"groups": groups, "categories": categories, "hit_groups": hit_groups, "hit_categories": hit_categories},
            "explanation": explanation,
        }
        return state.post(make_candidate(self.name, ans, support=support, confidence=confidence, round_id=2))
