"""Scoring functions for DVQA-style chart QA.

Two flavors of accuracy:

- `strict_exact_match`: matches the official DVQA protocol. Lowercase,
  collapse whitespace, strip trailing punctuation, then equality test.

- `relaxed_match`: more forgiving — useful for grading verbose LLM outputs
  against terse gold strings:
    * "five" ↔ "5" (cardinal words 0..20)
    * strip articles ("the", "a", "an") and common units ("%", "$",
      "units", "people", "items", ...)
    * numeric answers: 5% relative tolerance when both sides parse as floats.

Plus evidence IoU for the small subset of DVQA questions where the gold
answer carries a bounding box (`bbox_answer != []`).
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_WORDS_TO_DIGITS = {w: str(n) for w, n in _NUMBER_WORDS.items()}

_ARTICLES = {"the", "a", "an"}
_FILLER_UNITS = {"units", "unit", "people", "items", "item", "things"}

_PUNCT_STRIP = string.punctuation.replace("-", "")  # keep hyphens (e.g. Jan-Apr)


def _basic_normalize(s: str) -> str:
    s = s.strip().lower()
    s = s.translate(str.maketrans("", "", _PUNCT_STRIP))
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strict_exact_match(pred: str, gold: str) -> bool:
    """DVQA-style strict exact match."""
    return _basic_normalize(pred) == _basic_normalize(gold)


def _try_float(s: str) -> Optional[float]:
    """Parse a possibly-stripped numeric string. Returns None if not numeric."""
    s = s.strip().replace(",", "").rstrip("%$").lstrip("$")
    try:
        return float(s)
    except ValueError:
        return None


def _relaxed_normalize(s: str) -> str:
    s = _basic_normalize(s)
    tokens = [t for t in s.split() if t and t not in _ARTICLES and t not in _FILLER_UNITS]
    tokens = [_WORDS_TO_DIGITS.get(t, t) for t in tokens]
    return " ".join(tokens).strip()


def relaxed_match(pred: str, gold: str, numeric_tol: float = 0.05) -> bool:
    """Lenient grading: number-word folding + numeric tolerance."""
    pn = _relaxed_normalize(pred)
    gn = _relaxed_normalize(gold)
    if pn == gn:
        return True
    # Numeric path: both sides parse as floats.
    pf = _try_float(pred)
    gf = _try_float(gold)
    if pf is not None and gf is not None:
        if gf == 0.0:
            return abs(pf) < 1e-6
        return abs(pf - gf) / abs(gf) <= numeric_tol
    return False


# ---------------------------------------------------------------------------
# Bounding-box IoU
# ---------------------------------------------------------------------------

# DVQA bboxes are [x, y, w, h] in image pixel coordinates.

def _xywh_to_xyxy(b: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return float(x), float(y), float(x + w), float(y + h)


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU between two [x, y, w, h] boxes. Returns 0.0 if either box is invalid."""
    if a is None or b is None:
        return 0.0
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def accuracy(preds: Iterable[str], golds: Iterable[str], strict: bool = True) -> float:
    """Mean exact-match (strict or relaxed) over a list."""
    fn = strict_exact_match if strict else relaxed_match
    correct = 0
    total = 0
    for p, g in zip(preds, golds):
        total += 1
        if fn(p, g):
            correct += 1
    return correct / total if total else 0.0


def per_group_accuracy(
    preds: Iterable[str],
    golds: Iterable[str],
    groups: Iterable[str],
    strict: bool = True,
) -> dict:
    """Mean accuracy bucketed by a grouping key (e.g. question_type)."""
    fn = strict_exact_match if strict else relaxed_match
    bucket: dict = {}
    for p, g, k in zip(preds, golds, groups):
        b = bucket.setdefault(k, [0, 0])  # [correct, total]
        b[1] += 1
        if fn(p, g):
            b[0] += 1
    return {k: (c / t if t else 0.0) for k, (c, t) in bucket.items()}


# ---------------------------------------------------------------------------
# Paired-test helper (McNemar's test, no SciPy dependency)
# ---------------------------------------------------------------------------

def mcnemar_pvalue(b: int, c: int) -> float:
    """Two-sided exact McNemar's test p-value for paired binary outcomes.

    b = number of items where system A is right, B is wrong.
    c = number of items where B is right, A is wrong.

    Uses the exact binomial test on min(b, c) ~ Binomial(b+c, 0.5). Fine for
    n up to a few thousand.
    """
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) under Binomial(n, 0.5)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return p


def mcnemar_from_paired(correct_a: List[bool], correct_b: List[bool]) -> dict:
    """McNemar p-value + the underlying 2x2 contingency, given paired booleans."""
    assert len(correct_a) == len(correct_b)
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if bb and not a)
    both = sum(1 for a, bb in zip(correct_a, correct_b) if a and bb)
    neither = sum(1 for a, bb in zip(correct_a, correct_b) if not a and not bb)
    return {
        "b_only_a_right": b,
        "c_only_b_right": c,
        "both_right": both,
        "neither_right": neither,
        "p_value": mcnemar_pvalue(b, c),
    }
