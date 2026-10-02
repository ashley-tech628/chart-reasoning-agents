"""Round 2 Comparison Solver Agent."""

from __future__ import annotations

from chart_agents.agents.answer_type import get_answer_type_claim
from chart_agents.agents.normalizer import normalize_answer
from chart_agents.blackboard import BlackboardState, make_candidate
from chart_agents.solvers.utils import chart_records, compute_gap_by_group, format_number


class ComparisonSolver:
    name = "comparison_solver"

    def run(self, state: BlackboardState):
        q = state.question.lower()
        answer_type = get_answer_type_claim(state)
        expected = answer_type.get("answer_type", "unknown")
        valid = answer_type.get("valid_answers", []) or []

        answer = ""
        confidence = 0.0
        support = {}
        explanation = "comparison_solver could not safely answer this question."

        if any(k in q for k in ["largest gap", "biggest gap", "largest difference", "biggest difference", "gap between"]):
            gaps, rationale = compute_gap_by_group(state.question, state.chart_dsl)
            if gaps:
                best_group, best_gap = max(gaps.items(), key=lambda kv: kv[1])
                answer = best_group if expected == "group" else format_number(best_gap)
                answer = normalize_answer(answer, expected, valid)
                support = {f"gap_{group}": format_number(v) for group, v in gaps.items()}
                support.update({"max_gap": format_number(best_gap), "max_gap_group": best_group, "rationale": rationale})
                explanation = f"{best_group} has the largest gap ({format_number(best_gap)})."
                confidence = 0.95
        elif "largest individual bar" in q:
            records = chart_records(state.chart_dsl)
            if records:
                best = max(records, key=lambda d: d["value"])
                answer = format_number(best["value"]) if expected in {"number", "count"} else best["group"]
                answer = normalize_answer(answer, expected, valid)
                support = {"max_value": format_number(best["value"]), "group": best["group"], "category": best["category"]}
                explanation = f"{best['group']} / {best['category']} has the largest individual value, {format_number(best['value'])}."
                confidence = 0.94
        else:
            # MVP fallback for "which has the highest/largest value" style questions.
            records = chart_records(state.chart_dsl)
            if records and expected in {"group", "category"}:
                best = max(records, key=lambda d: d["value"])
                label_key = "category" if expected == "category" else "group"
                answer = normalize_answer(best[label_key], expected, valid)
                support = {"max_value": format_number(best["value"]), "group": best["group"], "category": best["category"]}
                explanation = f"{best[label_key]} has the largest individual value."
                confidence = 0.70

        return state.post(
            make_candidate(
                self.name,
                answer or None,
                support={"details": support, "explanation": explanation},
                confidence=confidence,
                round_id=2,
            )
        )
