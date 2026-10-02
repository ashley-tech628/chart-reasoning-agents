"""Round 2 Lookup Solver Agent."""

from __future__ import annotations

from chart_agents.blackboard import BlackboardState, make_candidate
from chart_agents.solvers.utils import (
    chart_records, evidence_from_record, format_number, mentioned_labels,
    solver_is_active, unique_categories, unique_groups,
)


class LookupSolver:
    name = "lookup_solver"

    def run(self, state: BlackboardState):
        if not solver_is_active(state, self.name):
            return None

        records = chart_records(state.chart_dsl)
        hit_groups = mentioned_labels(state.question, unique_groups(state.chart_dsl))
        hit_categories = mentioned_labels(state.question, unique_categories(state.chart_dsl))

        matches = []
        for r in records:
            group_ok = not hit_groups or r["group"] in hit_groups
            category_ok = not hit_categories or r["category"] in hit_categories
            if group_ok and category_ok:
                matches.append(r)

        if len(matches) == 1:
            r = matches[0]
            ans = format_number(r["value"])
            support = {
                "details": {"group": r["group"], "category": r["category"], "value": ans},
                "evidence": [evidence_from_record(r)],
                "explanation": f"The value for {r['category']} at {r['group']} is {ans}.",
            }
            return state.post(make_candidate(self.name, ans, support=support, confidence=0.90, round_id=2))

        support = {
            "details": {"hit_groups": hit_groups, "hit_categories": hit_categories, "num_matches": len(matches)},
            "explanation": "Lookup solver could not isolate exactly one chart cell.",
        }
        return state.post(make_candidate(self.name, None, support=support, confidence=0.10, round_id=2))
