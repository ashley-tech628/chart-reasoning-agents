"""Round 2 Boolean Solver Agent."""

from __future__ import annotations

from chart_agents.blackboard import BlackboardState, make_candidate
from chart_agents.solvers.utils import (
    chart_records, evidence_from_record, mentioned_labels,
    solver_is_active, unique_categories, unique_groups,
)


class BooleanSolver:
    name = "boolean_solver"

    def run(self, state: BlackboardState):
        if not solver_is_active(state, self.name):
            return None

        q = state.question.lower()
        records = chart_records(state.chart_dsl)
        hit_groups = mentioned_labels(state.question, unique_groups(state.chart_dsl))
        hit_categories = mentioned_labels(state.question, unique_categories(state.chart_dsl))

        matches = [
            r for r in records
            if (not hit_groups or r["group"] in hit_groups) and (not hit_categories or r["category"] in hit_categories)
        ]

        answer = None
        evidence = []
        explanation = "Boolean solver could not isolate a safe comparison."

        if len(matches) == 1 and any(k in q for k in ["greater than", "larger than", "higher than", "more than", "above", "over", "less than", "smaller than", "lower than", "fewer than", "below", "under"]):
            import re
            m = re.search(r"(-?\d+(?:\.\d+)?)", q)
            if m:
                threshold = float(m.group(1))
                r = matches[0]
                if any(k in q for k in ["greater than", "larger than", "higher than", "more than", "above", "over"]):
                    answer_bool = r["value"] > threshold
                    comp = ">"
                else:
                    answer_bool = r["value"] < threshold
                    comp = "<"
                answer = "yes" if answer_bool else "no"
                evidence = [evidence_from_record(r)]
                explanation = f"Compared {r['category']} at {r['group']}={r['value']} {comp} {threshold}: {answer}."
        elif len(matches) >= 2:
            # Use first two mentioned records for pairwise smaller/larger questions.
            a, b = matches[0], matches[1]
            if any(k in q for k in ["larger", "greater", "higher", "more", "above"]):
                answer_bool = a["value"] > b["value"]
                comp = ">"
            elif any(k in q for k in ["smaller", "less", "lower", "fewer", "below"]):
                answer_bool = a["value"] < b["value"]
                comp = "<"
            else:
                answer_bool = abs(a["value"] - b["value"]) < 1e-9
                comp = "=="
            answer = "yes" if answer_bool else "no"
            evidence = [evidence_from_record(a), evidence_from_record(b)]
            explanation = f"Compared {a['category']} at {a['group']}={a['value']} {comp} {b['category']} at {b['group']}={b['value']}: {answer}."

        confidence = 0.90 if answer else 0.10
        support = {"details": {"hit_groups": hit_groups, "hit_categories": hit_categories}, "evidence": evidence, "explanation": explanation}
        return state.post(make_candidate(self.name, answer, support=support, confidence=confidence, round_id=2))
