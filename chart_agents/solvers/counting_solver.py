"""Round 2 Counting Solver Agent."""

from __future__ import annotations

from collections import defaultdict

from chart_agents.blackboard import BlackboardState, make_candidate
from chart_agents.solvers.utils import (
    chart_records, evidence_from_record, format_number, mentioned_labels,
    parse_threshold, solver_is_active, unique_categories, unique_groups,
)


class CountingSolver:
    name = "counting_solver"

    def run(self, state: BlackboardState):
        if not solver_is_active(state, self.name):
            return None

        q = state.question.lower()
        records = chart_records(state.chart_dsl)
        groups = unique_groups(state.chart_dsl)
        categories = unique_categories(state.chart_dsl)
        hit_groups = mentioned_labels(state.question, groups)
        hit_categories = mentioned_labels(state.question, categories)
        op, threshold = parse_threshold(state.question)

        candidates = records
        if hit_groups:
            candidates = [r for r in candidates if r["group"] in hit_groups]
        if hit_categories:
            candidates = [r for r in candidates if r["category"] in hit_categories]

        selected_records = []
        selected_labels: list[str] = []
        group_mode = _grouped_any_mode(q)
        if op and threshold is not None:
            if group_mode is not None:
                grouped: dict[str, list[dict]] = defaultdict(list)
                for r in candidates:
                    grouped[str(r[group_mode])].append(r)

                for label, records_for_label in grouped.items():
                    matching = [r for r in records_for_label if _matches_threshold(r["value"], op, threshold)]
                    if matching:
                        selected_labels.append(label)
                        selected_records.extend(matching)
            else:
                for r in candidates:
                    if _matches_threshold(r["value"], op, threshold):
                        selected_records.append(r)
        elif "group" in q or "x-axis" in q or "x axis" in q or "bars" in q:
            selected_labels = hit_groups or groups
        elif "categor" in q or "legend" in q or "lines" in q:
            selected_labels = hit_categories or categories
        else:
            selected_records = candidates

        count = len(selected_records) if selected_records else len(selected_labels)
        if group_mode is not None:
            count = len(selected_labels)
        support = {
            "details": {
                "count": count,
                "operator": op,
                "threshold": threshold,
                "hit_groups": hit_groups,
                "hit_categories": hit_categories,
                "counted_labels": selected_labels,
                "count_mode": f"group_by_{group_mode}_with_any" if group_mode else "records",
            },
            "evidence": [evidence_from_record(r) for r in selected_records],
            "explanation": _explain_count(count, group_mode, op, threshold),
        }
        return state.post(make_candidate(self.name, format_number(count), support=support, confidence=0.84, round_id=2))


def _matches_threshold(value: float, op: str, threshold: float) -> bool:
    return (
        (op == ">" and value > threshold)
        or (op == "<" and value < threshold)
        or (op == "==" and abs(value - threshold) < 1e-9)
    )


def _grouped_any_mode(question_lower: str) -> str | None:
    """Return the label dimension for "at least one" count questions.

    These questions ask how many labels have any qualifying bar/cell. Counting
    each matching bar directly overcounts grouped charts.
    """

    if "groups of bars" in question_lower or "group of bars" in question_lower:
        return "group"
    if "at least one" not in question_lower:
        return None
    # "items/objects/products in at least one store/group/dataset" -> count legend-side categories.
    if any(k in question_lower for k in ["item", "object", "product", "algorithm", "category"]):
        return "category"
    # "groups with at least one bar" -> count axis-side groups.
    return "group"


def _explain_count(count: int, group_mode: str | None, op: str | None, threshold: float | None) -> str:
    if group_mode and op and threshold is not None:
        return (
            f"There are {format_number(count)} {group_mode} label(s) with at least one "
            f"value {op} {format_number(threshold)}."
        )
    return f"There are {format_number(count)} matching item(s)."
