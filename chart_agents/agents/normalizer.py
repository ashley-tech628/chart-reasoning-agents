"""Answer normalization for benchmark-style short answers."""

from __future__ import annotations

import re
from typing import Iterable

from chart_agents.solvers.utils import clean_text, exact_valid_label, format_number, is_numeric, norm_text, number_word

def normalize_answer(answer: str | None, answer_type: str = "unknown", valid_answers: Iterable[str] = (), count_as_words: bool | None = None) -> str:
    ans = clean_text(answer)
    ans = re.sub(r"^(the\s+answer\s+is\s+|answer\s*:\s*)", "", ans, flags=re.I).strip()
    while len(ans) > 1 and ans[-1] in ".。":
        ans = ans[:-1].strip()

    if answer_type == "boolean":
        low = ans.lower()
        if low in {"true", "correct", "yes", "y", "1"}:
            return "yes"
        if low in {"false", "incorrect", "no", "n", "0"}:
            return "no"
        return low
    
    if answer_type == "count":
        if is_numeric(ans):
            n = int(round(float(ans.replace(",", ""))))
            valid_norms = {norm_text(v) for v in valid_answers or []}
            if count_as_words is True or number_word(n) in valid_norms:
                return number_word(n)
            return str(n)
        return ans.lower()

    if answer_type == "number" or is_numeric(ans):
        return format_number(ans)

    if answer_type in {"group", "category", "label"}:
        exact = exact_valid_label(ans, valid_answers)
        if exact is not None:
            return exact
        # If answer is a sentence containing a valid label, return the exact label.
        low = norm_text(ans)
        for v in valid_answers or []:
            if re.search(rf"\b{re.escape(norm_text(v))}\b", low):
                return v
        return ans

    return ans
