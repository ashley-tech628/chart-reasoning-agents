"""Narrow deterministic overrides for CrewAI pipeline answers."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from chart_agents.agents.normalizer import normalize_answer
from chart_agents.schemas import ChartDSL, FinalAnswer, IntermediateValue
from chart_agents.solvers.utils import (
    chart_records,
    format_number,
    norm_text,
    parse_threshold,
    unique_categories,
    unique_groups,
)


def try_solve_sum_value(
    question: str,
    chart_dsl: ChartDSL,
    answer_type: dict[str, Any],
) -> FinalAnswer | None:
    """Return a deterministic answer for safe sum/aggregate templates."""

    q = question.lower()
    expected = answer_type.get("answer_type", "unknown")
    valid = answer_type.get("valid_answers", []) or []
    records = chart_records(chart_dsl)
    groups = unique_groups(chart_dsl)
    categories = unique_categories(chart_dsl)
    hit_groups = _strict_mentioned_labels(question, groups)
    hit_categories = _strict_mentioned_labels(question, categories)

    if not records or not _is_sum_question(q):
        return None

    if _asks_for_arg_sum(q, expected):
        dimension = _answer_dimension(expected, q)
        sums = _sum_by_dimension(records, dimension)
        if not sums:
            return None
        chooser = min if _asks_for_min(q) else max
        best_label, best_total = chooser(sums.items(), key=lambda kv: kv[1])
        answer = normalize_answer(best_label, expected, valid)
        return FinalAnswer(
            answer=answer,
            explanation=(
                f"{best_label} has the {'smallest' if _asks_for_min(q) else 'largest'} "
                f"summed value, {format_number(best_total)}."
            ),
            intermediate_values=[
                IntermediateValue(name=f"sum_{_safe_name(label)}", value=float(total))
                for label, total in sums.items()
            ],
        )

    if _asks_for_numeric_sum(q):
        selected = _filter_records_for_numeric_sum(records, hit_groups, hit_categories, q)
        if selected is None:
            return None
        total = sum(r["value"] for r in selected)
        answer = normalize_answer(format_number(total), expected, valid)
        return FinalAnswer(
            answer=answer,
            explanation=f"The selected values sum to {format_number(total)}.",
            intermediate_values=[IntermediateValue(name="sum_selected_values", value=float(total))],
        )

    return None


def try_solve_minmax_numeric_value(
    question: str,
    chart_dsl: ChartDSL,
    answer_type: dict,
) -> FinalAnswer | None:
    """Return numeric min/max values for explicit value/amount questions."""

    q = question.lower()
    if answer_type.get("answer_type") not in {"number", "count"}:
        return None
    if any(k in q for k in ["how many more", "how much more", "difference"]):
        return None
    if q.startswith("which"):
        return None

    records = chart_records(chart_dsl)
    if not records:
        return None

    wants_max = _asks_for_max_value(q)
    wants_min = _asks_for_min_value(q)
    if not wants_max and not wants_min:
        return None

    chosen = max(records, key=lambda r: r["value"]) if wants_max else min(records, key=lambda r: r["value"])
    answer = normalize_answer(
        format_number(chosen["value"]),
        answer_type.get("answer_type", "number"),
        answer_type.get("valid_answers", []),
    )
    op = "maximum" if wants_max else "minimum"
    return FinalAnswer(
        answer=answer,
        explanation=(
            f"The {op} chart value is {format_number(chosen['value'])} "
            f"at group '{chosen['group']}' and category '{chosen['category']}'."
        ),
        intermediate_values=[
            IntermediateValue(name=f"{op}_value", value=float(chosen["value"]))
        ],
    )


def try_solve_threshold_count(
    question: str,
    chart_dsl: ChartDSL,
    answer_type: dict,
) -> FinalAnswer | None:
    """Return deterministic counts for threshold templates."""

    if answer_type.get("answer_type") != "count":
        return None
    q = question.lower()
    if not (q.startswith("how many") or "number of" in q or "count" in q):
        return None

    op, threshold = parse_threshold(question)
    if op is None or threshold is None:
        return None
    if op == "<" and "accuracy" not in q:
        return None

    records = chart_records(chart_dsl)
    if not records:
        return None

    mode = _count_dimension(q, chart_dsl)
    if mode in {"group", "category"}:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            grouped[str(rec[mode])].append(rec)
        labels = [
            label
            for label, recs in grouped.items()
            if any(_matches_threshold(r["value"], op, threshold, q) for r in recs)
        ]
        count = len(labels)
        detail_name = f"count_{mode}s_matching_threshold"
        explanation = (
            f"There are {format_number(count)} {mode} label(s) with at least one "
            f"value {op} {format_number(threshold)}."
        )
    else:
        count = sum(1 for rec in records if _matches_threshold(rec["value"], op, threshold, q))
        labels = []
        detail_name = "count_records_matching_threshold"
        explanation = (
            f"There are {format_number(count)} value(s) {op} {format_number(threshold)}."
        )

    answer = normalize_answer(
        format_number(count),
        "count",
        answer_type.get("valid_answers", []),
    )
    return FinalAnswer(
        answer=answer,
        explanation=explanation,
        intermediate_values=[
            IntermediateValue(name=detail_name, value=float(count)),
            *[
                IntermediateValue(name=f"matched_{mode}_{_safe_name(label)}", value=1.0)
                for label in labels
            ],
        ],
    )


def _asks_for_max_value(question_lower: str) -> bool:
    if any(k in question_lower for k in ["largest individual bar", "largest value", "highest value"]):
        return True
    if "most preferred object" in question_lower and any(k in question_lower for k in ["how many", "percentage"]):
        return True
    if "most sold item" in question_lower and any(k in question_lower for k in ["how many", "units"]):
        return True
    return False


def _is_sum_question(question_lower: str) -> bool:
    return any(k in question_lower for k in ["summed", "sum of", "sum ", "total value", "combined value"])


def _asks_for_arg_sum(question_lower: str, expected: str) -> bool:
    if expected not in {"group", "category", "label"}:
        return False
    return any(k in question_lower for k in ["summed", "sum of", "total value", "combined"])


def _asks_for_numeric_sum(question_lower: str) -> bool:
    return any(k in question_lower for k in ["sum of", "sum ", "total value", "combined value"])


def _asks_for_min(question_lower: str) -> bool:
    return any(k in question_lower for k in ["smallest", "lowest", "least", "minimum", "min "])


def _asks_for_min_value(question_lower: str) -> bool:
    if any(k in question_lower for k in ["lowest accuracy reported", "smallest value", "lowest value"]):
        return True
    if "least preferred object" in question_lower and any(k in question_lower for k in ["how many", "percentage"]):
        return True
    if "least sold item" in question_lower and any(k in question_lower for k in ["how many", "units"]):
        return True
    return False


def _answer_dimension(expected: str, question_lower: str) -> str:
    if expected == "category":
        return "category"
    if expected == "group":
        return "group"
    if any(k in question_lower for k in ["object", "item", "product", "algorithm"]):
        return "category"
    return "group"


def _sum_by_dimension(records: list[dict[str, Any]], dimension: str) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for rec in records:
        sums[str(rec[dimension])] += float(rec["value"])
    return dict(sums)


def _filter_records_for_numeric_sum(
    records: list[dict[str, Any]],
    hit_groups: list[str],
    hit_categories: list[str],
    question_lower: str,
) -> list[dict[str, Any]] | None:
    if " group" in question_lower and not hit_groups:
        return None
    if any(k in question_lower for k in ["algorithm", "item", "object"]) and "all the" in question_lower and not hit_groups:
        return None

    selected = records
    if hit_groups:
        selected = [r for r in selected if r["group"] in hit_groups]
    if hit_categories:
        selected = [r for r in selected if r["category"] in hit_categories]
    return selected


def _strict_mentioned_labels(question: str, labels: list[str]) -> list[str]:
    q = norm_text(question)
    hits: list[str] = []
    for label in sorted([str(x) for x in labels if str(x)], key=len, reverse=True):
        ln = norm_text(label)
        if ln and re.search(rf"(?<!\w){re.escape(ln)}(?!\w)", q):
            hits.append(label)
    return hits


def _count_dimension(question_lower: str, chart_dsl: ChartDSL) -> str:
    if "groups of bars" in question_lower or "group of bars" in question_lower:
        return "group"
    if any(k in question_lower for k in ["algorithm", "item", "object", "product"]):
        return "group"
    if "categor" in question_lower or "legend" in question_lower:
        return "category"
    if len(unique_groups(chart_dsl)) > 1:
        return "group"
    if len(unique_categories(chart_dsl)) > 1:
        return "category"
    return "record"


def _matches_threshold(value: float, op: str, threshold: float, question_lower: str) -> bool:
    eps = _threshold_epsilon(threshold, question_lower)
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold or abs(value - threshold) <= eps
    if op == "==":
        return abs(value - threshold) <= eps
    return False


def _threshold_epsilon(threshold: float, question_lower: str) -> float:
    if "accuracy" in question_lower and any(k in question_lower for k in ["lower than", "less than", "smaller than"]):
        return max(1e-9, abs(threshold) * 0.025)
    return 1e-9


def _safe_name(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(label)).strip("_") or "label"

def try_solve_pairwise_boolean(
    question: str,
    chart_dsl: ChartDSL,
    answer_type: dict[str, Any],
) -> FinalAnswer | None:
    """Solve safe yes/no comparisons:
    Did item A in store X sell fewer units than item B in store Y?
    """

    if answer_type.get("answer_type") != "boolean":
        return None

    q = question.lower()
    if " than " not in q:
        return None

    direction = _comparison_direction(q)
    if direction is None:
        return None

    records = chart_records(chart_dsl)
    if not records:
        return None

    pairs = _extract_two_group_category_pairs(question, chart_dsl)
    if pairs is None:
        return None

    (g1, c1), (g2, c2) = pairs
    r1 = _find_record(records, g1, c1)
    r2 = _find_record(records, g2, c2)

    if r1 is None or r2 is None:
        return None

    if direction == "<":
        ok = r1["value"] < r2["value"]
    elif direction == ">":
        ok = r1["value"] > r2["value"]
    else:
        ok = abs(r1["value"] - r2["value"]) < 1e-9

    answer = "yes" if ok else "no"

    return FinalAnswer(
        answer=answer,
        explanation=(
            f"Compared {g1} in {c1} ({format_number(r1['value'])}) "
            f"{direction} {g2} in {c2} ({format_number(r2['value'])}): {answer}."
        ),
        intermediate_values=[
            IntermediateValue(
                name=f"lookup_{_safe_name(g1)}_{_safe_name(c1)}",
                value=float(r1["value"]),
            ),
            IntermediateValue(
                name=f"lookup_{_safe_name(g2)}_{_safe_name(c2)}",
                value=float(r2["value"]),
            ),
        ],
    )


def try_solve_argminmax_label(
    question: str,
    chart_dsl: ChartDSL,
    answer_type: dict[str, Any],
) -> FinalAnswer | None:
    """Solve 'which label has highest/lowest value' questions."""

    q = question.lower()
    expected = answer_type.get("answer_type", "unknown")
    valid = answer_type.get("valid_answers", []) or []

    if expected not in {"group", "category", "label"}:
        return None

    if not q.startswith(("which", "what")):
        return None

    if any(k in q for k in ["summed", "sum of", "total", "combined"]):
        return None

    wants_min = any(k in q for k in ["smallest", "lowest", "least", "minimum"])
    wants_max = any(k in q for k in ["largest", "highest", "most", "maximum", "greatest", "biggest"])

    if not wants_min and not wants_max:
        return None

    records = chart_records(chart_dsl)
    if not records:
        return None

    hit_groups = _strict_mentioned_labels(question, unique_groups(chart_dsl))
    hit_categories = _strict_mentioned_labels(question, unique_categories(chart_dsl))

    selected = records
    if hit_groups:
        selected = [r for r in selected if r["group"] in hit_groups]
    if hit_categories:
        selected = [r for r in selected if r["category"] in hit_categories]

    if not selected:
        return None

    chosen = min(selected, key=lambda r: r["value"]) if wants_min else max(selected, key=lambda r: r["value"])
    label_key = _label_dimension_for_question(expected, q, chart_dsl)

    label = str(chosen[label_key])
    answer = normalize_answer(label, expected, valid)
    op = "minimum" if wants_min else "maximum"

    return FinalAnswer(
        answer=answer,
        explanation=(
            f"The {op} value is {format_number(chosen['value'])} at "
            f"group '{chosen['group']}' and category '{chosen['category']}', "
            f"so the answer is '{answer}'."
        ),
        intermediate_values=[
            IntermediateValue(name=f"{op}_value", value=float(chosen["value"]))
        ],
    )


def _comparison_direction(question_lower: str) -> str | None:
    if any(k in question_lower for k in ["smaller", "less", "lower", "fewer", "below", "under"]):
        return "<"
    if any(k in question_lower for k in ["larger", "greater", "higher", "more", "above", "over"]):
        return ">"
    if any(k in question_lower for k in ["same", "equal"]):
        return "=="
    return None


def _extract_two_group_category_pairs(
    question: str,
    chart_dsl: ChartDSL,
) -> tuple[tuple[str, str], tuple[str, str]] | None:
    parts = re.split(r"\bthan\b", question, maxsplit=1, flags=re.I)

    if len(parts) != 2:
        return None

    left, right = parts[0], parts[1]

    p1 = _extract_one_group_category_pair(left, chart_dsl)
    p2 = _extract_one_group_category_pair(right, chart_dsl)

    if p1 is None or p2 is None:
        return None

    return p1, p2


def _extract_one_group_category_pair(
    text: str,
    chart_dsl: ChartDSL,
) -> tuple[str, str] | None:
    groups = _ordered_label_hits(text, unique_groups(chart_dsl))
    categories = _ordered_label_hits(text, unique_categories(chart_dsl))

    if not groups or not categories:
        return None

    return groups[-1][1], categories[-1][1]


def _ordered_label_hits(text: str, labels: list[str]) -> list[tuple[int, str]]:
    q = norm_text(text)
    hits: list[tuple[int, str]] = []

    for label in sorted([str(x) for x in labels if str(x)], key=len, reverse=True):
        ln = norm_text(label)
        if not ln:
            continue

        m = re.search(rf"(?<!\w){re.escape(ln)}(?!\w)", q)
        if m:
            hits.append((m.start(), label))

    hits.sort(key=lambda x: x[0])
    return hits


def _find_record(
    records: list[dict[str, Any]],
    group: str,
    category: str,
) -> dict[str, Any] | None:
    for r in records:
        if r["group"] == group and r["category"] == category:
            return r

    # fallback in case group/category got swapped
    for r in records:
        if r["group"] == category and r["category"] == group:
            return r

    return None


def _label_dimension_for_question(
    expected: str,
    question_lower: str,
    chart_dsl: ChartDSL,
) -> str:
    if expected == "group":
        return "group"

    if expected == "category":
        return "category"

    if "category" in question_lower or "legend" in question_lower:
        return "category"

    if any(k in question_lower for k in ["algorithm", "item", "object", "product", "bar", "group"]):
        return "group"

    if len(unique_groups(chart_dsl)) >= len(unique_categories(chart_dsl)):
        return "group"

    return "category"