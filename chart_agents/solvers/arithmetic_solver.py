"""Round 2 Arithmetic Solver Agent.

This solver is intentionally allowed to propose numeric intermediate values.
The Format Critic can object when the numeric value is not the final answer type.
"""

from __future__ import annotations

from chart_agents.blackboard import BlackboardState, make_candidate
from chart_agents.solvers.utils import chart_records, compute_gap_by_group, format_number


class ArithmeticSolver:
    name = "arithmetic_solver"

    def run(self, state: BlackboardState):
        q = state.question.lower()
        answer = ""
        confidence = 0.0
        support = {}
        note = "arithmetic_solver did not match a safe arithmetic template."

        if any(k in q for k in ["largest gap", "biggest gap", "largest difference", "biggest difference", "gap between"]):
            gaps, rationale = compute_gap_by_group(state.question, state.chart_dsl)
            if gaps:
                best_group, best_gap = max(gaps.items(), key=lambda kv: kv[1])
                answer = format_number(best_gap)
                support = {f"gap_{group}": format_number(v) for group, v in gaps.items()}
                support.update({"max_gap": format_number(best_gap), "max_gap_group": best_group, "rationale": rationale})
                note = f"{answer} is the largest gap value, but it may not be the final answer if the question asks for a group."
                confidence = 0.80
        elif (
            "most accurate" in q
            and "least accurate" in q
            and ("how much more" in q or "compared" in q)
        ):
            records = chart_records(state.chart_dsl)
            if records:
                values = [r["value"] for r in records]
                diff = max(values) - min(values)
                answer = format_number(diff)
                support = {
                    "max_value": format_number(max(values)),
                    "min_value": format_number(min(values)),
                    "difference": answer,
                }
                note = f"The max-minus-min difference is {answer}."
                confidence = 0.90

        return state.post(
            make_candidate(
                self.name,
                answer or None,
                support={"details": support, "note": note},
                confidence=confidence,
                round_id=2,
            )
        )
