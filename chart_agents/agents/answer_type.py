"""Round 1 Answer-Type Agent.

This agent predicts the benchmark-style answer type before solvers run.
The key rule is that value/amount questions must stay numeric even if they
contain words such as largest / smallest / most / least.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from chart_agents.blackboard import BlackboardState, make_claim
from chart_agents.solvers.utils import unique_categories, unique_groups


_NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
]

GENERIC_CATEGORIES = {"", "value", "values", "count", "counts"}


def infer_answer_type(question: str, chart_dsl: Optional[object] = None) -> Dict[str, Any]:
    """Infer the expected short-answer type for a chart-QA question.

    This deterministic helper is shared by the blackboard workflow and the
    CrewAI pipeline so answer extraction, normalization, and critique use the
    same label-vs-number policy.

    Naming convention:
    - group: axis-side label, e.g. x-axis/y-axis group.
    - category: legend-side label, e.g. colored legend/category.
    """

    q = question.lower().strip()
    groups: List[str] = unique_groups(chart_dsl) if chart_dsl is not None else []
    categories: List[str] = unique_categories(chart_dsl) if chart_dsl is not None else []
    meaningful_categories = [c for c in categories if c and c.strip().lower() not in GENERIC_CATEGORIES]

    # 1. Boolean questions
    if q.startswith(("is ", "are ", "does ", "do ", "did ", "was ", "were ", "can ", "has ", "have ")):
        return {
            "answer_type": "boolean",
            "expected_format": "yes or no",
            "valid_answers": ["yes", "no"],
            "reject_numeric_only_answer": True,
        }

    if q.startswith("which") and any(k in q for k in ["summed", "sum of", "total"]):
        if "group" in q:
            return {
                "answer_type": "group",
                "expected_format": "one axis-side group label",
                "valid_answers": groups,
                "reject_numeric_only_answer": True,
            }
        if any(k in q for k in ["object", "product", "item", "algorithm"]):
            return {
                "answer_type": "group",
                "expected_format": "one axis-side object/item/algorithm label",
                "valid_answers": groups,
                "reject_numeric_only_answer": True,
            }
        if any(k in q for k in ["categor", "legend"]):
            return {
                "answer_type": "category",
                "expected_format": "one legend-side object/category label",
                "valid_answers": categories,
                "reject_numeric_only_answer": True,
            }
        if any(k in q for k in ["store", "shop", "dataset"]):
            return {
                "answer_type": "group",
                "expected_format": "one axis-side item/algorithm label",
                "valid_answers": groups,
                "reject_numeric_only_answer": True,
            }

    if q.startswith("which") and any(k in q for k in ["object", "product", "item", "category"]) and "number of" in q:
        if meaningful_categories:
            return {
                "answer_type": "category",
                "expected_format": "one legend/category/item label",
                "valid_answers": categories,
                "reject_numeric_only_answer": True,
            }
        return {
            "answer_type": "group",
            "expected_format": "one group/item label",
            "valid_answers": groups,
            "reject_numeric_only_answer": True,
        }

     # 2. Numeric value questions
    if any(
        k in q
        for k in [
            "what is the value",
            "what's the value",
            "what value",
            "how much",
            "how many units",
            "how many more",
            "how much more",
            "sum of",
            "total value",
            "difference between",
            "difference in",
            "compared to the least",
            "compared the least",
        ]
    ):
        return {
            "answer_type": "number",
            "expected_format": "one numeric value",
            "valid_answers": [],
            "reject_numeric_only_answer": False,
        }

    # 3. Count questions
    if q.startswith("how many") or "number of" in q or "count" in q:
        return {
            "answer_type": "count",
            "expected_format": "one count value; prefer dataset style number words when possible",
            "valid_answers": [str(i) for i in range(0, 21)] + _NUMBER_WORDS,
            "reject_numeric_only_answer": False,
        }

    # 4. Which group has largest summed value?
    if q.startswith("which") and "summed value" in q and "group" in q:
        return {
            "answer_type": "group",
            "expected_format": "one axis-side group label",
            "valid_answers": groups,
            "reject_numeric_only_answer": True,
        }

    # 5. Which bar/group/quarter...
    if q.startswith("which") and any(k in q for k in ["bar", "group", "quarter", "x-axis", "x axis", "y-axis", "y axis", "label"]):
        return {
            "answer_type": "group",
            "expected_format": "one axis-side group/bar label",
            "valid_answers": groups,
            "reject_numeric_only_answer": True,
        }

    # 6. Which category/algorithm/object/product/item...
    if q.startswith("which") and any(k in q for k in ["category", "algorithm", "object", "product", "item"]):
        if meaningful_categories:
            return {
                "answer_type": "category",
                "expected_format": "one legend-side category/item label",
                "valid_answers": categories,
                "reject_numeric_only_answer": True,
            }
        return {
            "answer_type": "group",
            "expected_format": "one group/item label",
            "valid_answers": groups,
            "reject_numeric_only_answer": True,
        }

    if q.startswith("which"):
        if meaningful_categories:
            return {
                "answer_type": "category",
                "expected_format": "one legend-side category/item label",
                "valid_answers": categories,
                "reject_numeric_only_answer": True,
            }
        return {
            "answer_type": "group",
            "expected_format": "one group/bar label",
            "valid_answers": groups,
            "reject_numeric_only_answer": True,
        }

    # 7. Direct label questions
    if any(k in q for k in ["what label", "what category", "name of", "legend", "x-axis label", "x axis label"]):
        return {
            "answer_type": "label",
            "expected_format": "one group/category/legend string",
            "valid_answers": groups + categories,
            "reject_numeric_only_answer": True,
        }

    # 8. Default
    return {
        "answer_type": "number",
        "expected_format": "one numeric value",
        "valid_answers": [],
        "reject_numeric_only_answer": False,
    }


class AnswerTypeAgent:
    name = "answer_type_agent"

    def run(self, state: BlackboardState):
        content = infer_answer_type(state.question, state.chart_dsl)
        return state.post(make_claim(self.name, content, confidence=0.96, round_id=1))


def get_answer_type_claim(state: BlackboardState) -> Dict[str, Any]:
    msg = state.latest_claim_from("answer_type_agent")
    if msg is not None and isinstance(msg.content, dict):
        return msg.content
    return {"answer_type": "unknown", "valid_answers": [], "reject_numeric_only_answer": False}
