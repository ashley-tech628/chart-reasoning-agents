"""Utility helpers shared by blackboard agents and solvers."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from chart_agents.schemas import ChartDSL


def clean_text(text: str | None) -> str:
    return (text or "").strip().strip("\"'").strip()


def norm_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", clean_text(text).lower().replace("_", " ")).strip()


def format_number(x: float | int | str) -> str:
    try:
        s = str(x).replace(",", "").strip()
        # Strip common wrappers from solver answers.
        s = re.sub(r"^(answer\s*:\s*|the\s+answer\s+is\s+)", "", s, flags=re.I).strip()
        v = float(s)
    except Exception:
        return str(x)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def is_numeric(text: str | None) -> bool:
    try:
        float(clean_text(text).replace(",", ""))
        return True
    except Exception:
        return False


def number_word(n: int) -> str:
    words = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    }
    return words.get(n, str(n))


def chart_records(chart_dsl: ChartDSL) -> List[Dict[str, Any]]:
    return [
        {
            "group": d.group,
            "category": d.category,
            "value": float(d.value),
            "visual_mark_index": d.visual_mark_index,
        }
        for d in chart_dsl.data
    ]


def unique_groups(chart_dsl: ChartDSL) -> List[str]:
    """Return axis-side group labels, preserving ChartDSL order when available."""

    seen: List[str] = []
    for g in getattr(chart_dsl, "groups", []) or []:
        if g and g not in seen:
            seen.append(g)
    for d in chart_dsl.data:
        if d.group not in seen:
            seen.append(d.group)
    return seen


def unique_categories(chart_dsl: ChartDSL) -> List[str]:
    """Return legend-side category labels, preserving ChartDSL order when available."""

    seen: List[str] = []
    for c in getattr(chart_dsl, "categories", []) or []:
        if c and c not in seen:
            seen.append(c)
    for d in chart_dsl.data:
        if d.category not in seen:
            seen.append(d.category)
    return seen


def build_table(chart_dsl: ChartDSL) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = defaultdict(dict)
    for d in chart_dsl.data:
        table[d.group][d.category] = float(d.value)
    return dict(table)


def infer_two_categories(question: str, chart_dsl: ChartDSL) -> Tuple[Optional[str], Optional[str]]:
    q = norm_text(question)
    categories = unique_categories(chart_dsl)
    mentioned = [c for c in categories if norm_text(c) in q]
    if len(mentioned) >= 2:
        return mentioned[0], mentioned[1]
    if len(categories) >= 2:
        return categories[0], categories[1]
    return None, None


def compute_gap_by_group(question: str, chart_dsl: ChartDSL) -> Tuple[Dict[str, float], str]:
    """Compute gap between two legend-side categories within each axis-side group.

    For MVP, "gap" means absolute difference. This supports questions such as:
    "Which quarter has the largest gap between Product A and Product B?"
    """

    c1, c2 = infer_two_categories(question, chart_dsl)
    if not c1 or not c2:
        return {}, "could not infer two categories"
    table = build_table(chart_dsl)
    gaps: Dict[str, float] = {}
    for group, vals in table.items():
        if c1 in vals and c2 in vals:
            gaps[group] = abs(vals[c1] - vals[c2])
    return gaps, f"absolute gap between {c1} and {c2} by group"


def exact_valid_label(answer: str | None, valid_answers: Iterable[str]) -> Optional[str]:
    ans_norm = norm_text(answer)
    for v in valid_answers or []:
        if ans_norm == norm_text(v):
            return v
    return None


def mentioned_labels(question: str, labels: Iterable[str]) -> List[str]:
    """Return exact labels that appear in the question, preserving label strings."""
    q = norm_text(question)
    hits: List[str] = []
    for label in sorted([str(x) for x in labels if str(x)], key=len, reverse=True):
        ln = norm_text(label)
        if not ln:
            continue
        # Use both boundary and substring, because chart labels can be short or contain punctuation.
        if re.search(rf"(?<!\w){re.escape(ln)}(?!\w)", q) or ln in q:
            if label not in hits:
                hits.append(label)
    return hits


def evidence_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    idx = rec.get("visual_mark_index")
    return {
        "group": rec.get("group"),
        "category": rec.get("category"),
        "value": format_number(rec.get("value")),
        "visual_ref": f"components[{idx}]" if idx is not None else None,
    }


def get_router_claim(state: Any) -> Dict[str, Any]:
    msg = state.latest_claim_from("question_router") if hasattr(state, "latest_claim_from") else None
    if msg is not None and isinstance(msg.content, dict):
        return msg.content
    return {"required_solvers": [], "optional_solvers": [], "task": "unknown", "operation": "unknown"}


def solver_is_active(state: Any, name: str) -> bool:
    route = get_router_claim(state)
    active = set((route.get("required_solvers") or []) + (route.get("optional_solvers") or []))
    # If no router claim exists, allow solver to run safely for manual tests.
    return not active or name in active


def parse_threshold(question: str) -> Tuple[Optional[str], Optional[float]]:
    q = question.lower()
    patterns = [
        (r"(?:greater than|larger than|higher than|more than|above|over)\s+(-?\d+(?:\.\d+)?)", ">"),
        (r"(?:less than|smaller than|lower than|fewer than|below|under)\s+(-?\d+(?:\.\d+)?)", "<"),
        (r"(?:equal to|equals?|exactly)\s+(-?\d+(?:\.\d+)?)", "=="),
    ]
    for pat, op in patterns:
        m = re.search(pat, q)
        if m:
            return op, float(m.group(1))
    return None, None
