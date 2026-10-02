"""Offline computer-vision chart perception tool.

This module replaces the previous VLM-only perception step with a deterministic
CV/OCR pipeline.  It is intentionally built as a set of small detectors rather
than one monolithic model:

1. OCR detector: title, axis labels, tick labels, legend text.
2. Plot-area / axis detector: chart frame and grid/tick positions.
3. Color-mark detector: bars and legend swatches from connected components.
4. Legend linker: maps swatch colors to category names.
5. Scale calibrator: maps mark endpoints in pixels to chart values.

The tool targets the DVQA-style synthetic bar charts used in this project.  It is
not meant to be a universal chart parser, but it gives the pipeline a clearly
computer-vision-centered perception stage while keeping the downstream LLM agents
unchanged.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type

import cv2
import numpy as np
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None
try:
    from crewai.tools import BaseTool
except Exception:  # pragma: no cover - allows offline smoke tests without CrewAI installed
    class BaseTool:  # type: ignore[no-redef]
        pass
from pydantic import BaseModel, Field

try:  # OCR is optional at import time so unit tests can still import the module.
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover - depends on local system packages
    pytesseract = None
    Output = None


@lru_cache(maxsize=1)
def _tesseract_status() -> tuple[bool, str]:
    """Return whether the Tesseract executable is callable.

    `pip install pytesseract` only installs the Python wrapper.  On Windows in
    particular, it is common for the import to succeed while the actual
    `tesseract.exe` binary is missing from PATH.  The perception output includes
    this status so OCR failures are visible rather than silently producing empty
    labels.
    """

    if pytesseract is None or Output is None:
        return False, "pytesseract_import_failed"
    try:
        _ = pytesseract.get_tesseract_version()
        return True, "ok"
    except Exception as exc:  # pragma: no cover - depends on local system packages
        msg = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        msg = re.sub(r"\s+", " ", msg)[:180]
        return False, f"tesseract_binary_unavailable: {msg}"


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> int:
        return int(max(0, self.w) * max(0, self.h))

    def as_schema(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "w": float(self.w), "h": float(self.h)}


@dataclass
class OCRWord:
    text: str
    conf: float
    box: Box


@dataclass
class ColorComponent:
    box: Box
    area: int
    rgb: Tuple[int, int, int]
    color_key: Tuple[int, int, int]


@dataclass
class LinearCalibrator:
    """value = a * pixel + b."""

    a: float
    b: float
    pixel_min: float
    pixel_max: float
    value_min: float
    value_max: float
    source: str

    def value_at(self, pixel: float) -> float:
        return self.a * pixel + self.b

    def pixel_for(self, value: float) -> float:
        if abs(self.a) < 1e-9:
            return self.pixel_min
        return (value - self.b) / self.a


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def _is_useful_text(text: str) -> bool:
    t = _clean_text(text)
    if not t:
        return False
    if len(t) == 1 and not t.isalnum():
        return False
    # Drop common OCR junk generated from grid lines / hatch marks.
    if re.fullmatch(r"[|_~`.,:;^]+", t):
        return False
    return True


def _parse_float_token(text: str) -> Optional[float]:
    """Parse OCR text as a numeric tick label with conservative cleanup.

    This function must be conservative: words such as "Title" or "Values"
    should not become numeric merely because they contain OCR-confusable letters
    like "l" or "I".  We only apply letter-to-digit repairs when the token
    is already mostly numeric-like.
    """

    raw = _clean_text(text)
    if not raw:
        return None
    alpha = sum(ch.isalpha() for ch in raw)
    numeric_like = sum(ch.isdigit() or ch in "+-.−–—" for ch in raw)
    if alpha and numeric_like == 0:
        return None
    if alpha > max(1, numeric_like):
        return None

    # If the token contains a letter that is NOT a common OCR digit-confusable
    # ({l, I, o, O}), treat it as a group label, not a number.  Without this
    # check, short category names like "Q2", "A3", "B1" were being misread as
    # floats 2.0 / 3.0 / 1.0 and getting filtered out of category candidates.
    confusable = {"l", "I", "o", "O"}
    if any(ch.isalpha() and ch not in confusable for ch in raw):
        return None

    t = raw.replace("O", "0").replace("o", "0")
    if numeric_like > 0:
        t = t.replace("l", "1").replace("I", "1")
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")
    # Keep only numeric-looking characters.
    t = re.sub(r"[^0-9+\-.]", "", t)
    if not t or t in {"-", ".", "-.", "+", "+."}:
        return None
    # Tesseract often emits multiple dots; keep the first one.
    if t.count(".") > 1:
        first = t.find(".")
        t = t[: first + 1] + t[first + 1 :].replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def _round_chart_value(v: float) -> float:
    """Round values the way synthetic chart data are normally encoded."""

    if not math.isfinite(v):
        return v
    nearest_int = round(v)
    if abs(v - nearest_int) <= 0.18:
        return float(nearest_int)
    nearest_half = round(v * 2.0) / 2.0
    if abs(v - nearest_half) <= 0.18:
        return float(nearest_half)
    return float(round(v, 2))


def _group_positions(indices: np.ndarray, gap: int = 2) -> list[int]:
    if len(indices) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for val in indices[1:]:
        val = int(val)
        if val <= prev + gap:
            prev = val
        else:
            groups.append((start, prev))
            start = prev = val
    groups.append((start, prev))
    return [(a + b) // 2 for a, b in groups]


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / float(a.area + b.area - inter + 1e-9)


def _contained_fraction(inner: Box, outer: Box) -> float:
    ix1, iy1 = max(inner.x, outer.x), max(inner.y, outer.y)
    ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / float(inner.area + 1e-9)


def _merge_line_text(words: Sequence[OCRWord]) -> Optional[str]:
    good = [w for w in words if _is_useful_text(w.text)]
    if not good:
        return None
    good = sorted(good, key=lambda w: w.box.x)
    return _clean_text(" ".join(w.text for w in good)) or None


def _line_groups(words: Sequence[OCRWord], y_tol: int = 12) -> list[list[OCRWord]]:
    groups: list[list[OCRWord]] = []
    for word in sorted(words, key=lambda w: w.box.cy):
        if not groups or abs(np.mean([w.box.cy for w in groups[-1]]) - word.box.cy) > y_tol:
            groups.append([word])
        else:
            groups[-1].append(word)
    for g in groups:
        g.sort(key=lambda w: w.box.x)
    return groups


# ---------------------------------------------------------------------------
# OCR layer
# ---------------------------------------------------------------------------


def _ocr_words(rgb: np.ndarray, psm_values: Sequence[int] = (6, 11)) -> list[OCRWord]:
    """Run Tesseract OCR and return de-duplicated word boxes in original pixels.

    Pre-processing before OCR:

    * **Edge padding (white border, w//5 horizontal and h//5 vertical on each
      side).**  Tesseract's page-layout analyzer occasionally drops the first
      or last character of a token that touches the figure edge, and OCR is
      generally more robust when there is breathing room around text.
      Padding before upscaling means the border is also resampled, which keeps
      anti-aliasing consistent with the rest of the image.
    * **2x upscale.**  Matplotlib's small tick fonts recognize substantially
      better at ~2x.  Kept at a fixed factor rather than an adaptive one to
      keep OCR latency predictable across figure sizes.

    The pad is subtracted off after OCR so all reported bboxes are in the
    *original* image coordinate system; downstream geometry (plot detection,
    bar bbox math, calibration) is unaffected.

    These steps help small / tight-margin figures.  They do NOT help labels
    that are rotated 90 degrees -- Tesseract reads horizontally and cannot
    auto-detect rotated text.  Rotated tick labels are handled separately by
    the per-group rotated-OCR fallback in ``_ocr_rotated_bottom_labels``.
    """

    ready, _ = _tesseract_status()
    if not ready:
        return []

    h_src, w_src = rgb.shape[:2]
    pad_x = max(1, w_src // 5)
    pad_y = max(1, h_src // 5)
    padded = cv2.copyMakeBorder(
        rgb, pad_y, pad_y, pad_x, pad_x,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )
    scale = 2
    up = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    words: list[OCRWord] = []
    for psm in psm_values:
        try:
            data = pytesseract.image_to_data(
                up,
                output_type=Output.DICT,
                config=f"--psm {psm}",
            )
        except Exception:  # pragma: no cover - depends on OCR binary
            continue
        n = len(data.get("text", []))
        for i in range(n):
            text = _clean_text(str(data["text"][i]))
            if not _is_useful_text(text):
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0
            if conf < 25:
                continue
            # OCR coordinates are in the padded+upscaled image.  Undo upscale
            # first (divide by scale), then translate back by the per-side pad
            # to recover original-image pixel coordinates.
            x = int(round(data["left"][i] / scale)) - pad_x
            y = int(round(data["top"][i] / scale)) - pad_y
            w = int(round(data["width"][i] / scale))
            h = int(round(data["height"][i] / scale))
            if w <= 1 or h <= 1:
                continue
            # Clip to source-image bounds: a token that landed entirely inside
            # the white pad is a layout-analysis artifact and should be ignored,
            # but a token straddling the edge is real and gets clipped to the
            # visible image.
            if x + w <= 0 or y + h <= 0 or x >= w_src or y >= h_src:
                continue
            x_clip = max(0, x)
            y_clip = max(0, y)
            w_clip = max(1, min(w_src - x_clip, w - (x_clip - x)))
            h_clip = max(1, min(h_src - y_clip, h - (y_clip - y)))
            words.append(OCRWord(text=text, conf=conf, box=Box(x_clip, y_clip, w_clip, h_clip)))

    # De-duplicate near-identical boxes from different PSM passes.
    out: list[OCRWord] = []
    for word in sorted(words, key=lambda w: (-w.conf, w.box.y, w.box.x)):
        duplicate = False
        for kept in out:
            if word.text.lower() == kept.text.lower() and _iou(word.box, kept.box) > 0.45:
                duplicate = True
                break
        if not duplicate:
            out.append(word)
    return sorted(out, key=lambda w: (w.box.y, w.box.x))


def _ocr_rotated_left_labels(rgb: np.ndarray, plot: Box) -> list[str]:
    """OCR y/group labels that matplotlib often rotates vertically.

    For horizontal DVQA charts, y tick/category names may be rotated 90 degrees
    and placed left of the plotting area.  Rotating that strip 270 degrees turns
    top-to-bottom labels into a left-to-right string; reversing the tokens maps
    them back to image top-to-bottom order.
    """

    if pytesseract is None:
        return []
    # Include a small strip inside the plot because tick labels can touch the
    # y-axis; do not go far enough to include the legend text.
    x2 = max(1, min(rgb.shape[1], plot.x + 60))
    crop = rgb[max(0, plot.y - 10) : min(rgb.shape[0], plot.y2 + 10), 0:x2]
    if crop.size == 0:
        return []
    # Clockwise rotation turns top-to-bottom vertical labels into a left-to-right
    # string in bottom-to-top order; reverse tokens below to recover top-to-bottom.
    rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    try:
        text = pytesseract.image_to_string(rot, config="--psm 6")
    except Exception:  # pragma: no cover
        return []
    tokens = []
    for tok in re.split(r"\s+", text):
        tok = _clean_text(tok)
        if not tok or _parse_float_token(tok) is not None:
            continue
        if len(tok) <= 1:
            continue
        # Keep alphabetic category names and reject OCR junk with too many symbols.
        if sum(ch.isalpha() for ch in tok) >= max(2, len(tok) // 2):
            tokens.append(re.sub(r"[^A-Za-z0-9_-]", "", tok))
    # 270-degree rotation reads bottom-to-top, so reverse for top-to-bottom.
    return list(reversed([t for t in tokens if t]))


# ---------------------------------------------------------------------------
# Axis / plot area detection
# ---------------------------------------------------------------------------


def _detect_long_lines(gray: np.ndarray, dark_background: bool) -> tuple[list[int], list[int]]:
    h, w = gray.shape[:2]
    if dark_background:
        mask = (gray > 170).astype(np.uint8) * 255
    else:
        # Conservative grayscale fallback.  _detect_plot_area uses a stronger
        # RGB-aware spine/grid detector before reaching this helper.
        mask = (gray < 235).astype(np.uint8) * 255
    hk = max(12, w // 22)
    vk = max(12, h // 22)
    horizontal = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    )
    vertical = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    )
    cols = np.where((vertical > 0).sum(axis=0) > 0.10 * h)[0]
    rows = np.where((horizontal > 0).sum(axis=1) > 0.10 * w)[0]
    return _group_positions(cols), _group_positions(rows)


def _detect_spine_box(rgb: np.ndarray) -> Optional[tuple[Box, list[int], list[int]]]:
    """Detect the chart rectangle from black axes/spines.

    DVQA/matplotlib charts often have a very clean black rectangular frame.  It
    is safer to recover the plot area from these long black spines than from a
    connected component over all non-white pixels, because bars, hatching, and
    grid lines can merge into a misleading panel-sized component.
    """

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    black = (gray < 95).astype(np.uint8) * 255
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, w // 3), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, h // 3)))
    horizontal = cv2.morphologyEx(black, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(black, cv2.MORPH_OPEN, v_kernel)
    rows = _group_positions(np.where((horizontal > 0).sum(axis=1) > 0.45 * w)[0], gap=2)
    cols = _group_positions(np.where((vertical > 0).sum(axis=0) > 0.30 * h)[0], gap=2)
    if len(rows) < 2 or len(cols) < 2:
        return None

    x_min, x_max = min(cols), max(cols)
    y_min, y_max = min(rows), max(rows)
    if x_max - x_min < 0.45 * w or y_max - y_min < 0.45 * h:
        return None
    if y_min > 0.35 * h or y_max < 0.55 * h:
        return None
    plot = Box(int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    # Recover gray grid/tick rows/columns with a true low-saturation mask.  These
    # are returned for debugging and for tick-label synthesis; the plot box still
    # comes from the black spines above.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    grayish = ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] < 245)).astype(np.uint8) * 255
    hk = max(20, w // 18)
    vk = max(20, h // 18)
    h_lines_img = cv2.morphologyEx(grayish, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    v_lines_img = cv2.morphologyEx(grayish, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    h_rows = _group_positions(
        np.where((h_lines_img > 0).sum(axis=1) > 0.45 * plot.w)[0], gap=2
    )
    v_cols = _group_positions(
        np.where((v_lines_img > 0).sum(axis=0) > 0.45 * plot.h)[0], gap=2
    )
    h_rows = [r for r in h_rows if plot.y - 3 <= r <= plot.y2 + 3]
    v_cols = [c for c in v_cols if plot.x - 3 <= c <= plot.x2 + 3]
    if plot.y not in h_rows:
        h_rows.append(plot.y)
    if plot.y2 not in h_rows:
        h_rows.append(plot.y2)
    if plot.x not in v_cols:
        v_cols.append(plot.x)
    if plot.x2 not in v_cols:
        v_cols.append(plot.x2)
    return plot, sorted(v_cols), sorted(h_rows)


def _detect_plot_area(rgb: np.ndarray) -> tuple[Box, list[int], list[int], str]:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark_background = float(np.median(gray)) < 100.0

    if not dark_background:
        spine = _detect_spine_box(rgb)
        if spine is not None:
            plot, v_cols, h_rows = spine
            return plot, v_cols, h_rows, "black_spines"

    v_lines, h_lines = _detect_long_lines(gray, dark_background=dark_background)

    # On dark-background charts the left spine may be too short/fragmented for
    # the vertical-line detector.  Recover x-extents from long horizontal grid
    # rows, especially the top/bottom spines.
    dark_x_extents: list[tuple[int, int]] = []
    if dark_background and h_lines:
        bright = gray > 170
        for row in h_lines:
            r0, r1 = max(0, row - 1), min(h, row + 2)
            cols = np.where(bright[r0:r1, :].any(axis=0))[0]
            if len(cols) == 0:
                continue
            # Use the longest contiguous run so nearby white tick-label text on
            # the left does not pull the plot boundary outward.
            runs: list[tuple[int, int]] = []
            start = prev = int(cols[0])
            for c in cols[1:]:
                c = int(c)
                if c <= prev + 2:
                    prev = c
                else:
                    runs.append((start, prev))
                    start = prev = c
            runs.append((start, prev))
            a, b = max(runs, key=lambda ab: ab[1] - ab[0])
            if (b - a) > 0.45 * w:
                dark_x_extents.append((a, b))

    # Candidate from connected non-background plot panel.  This captures seaborn's
    # gray plotting rectangle and matplotlib's grid/spine rectangle.
    if dark_background:
        # On black-themed charts, rely primarily on long grid/spine lines.
        panel_candidate = None
    else:
        panel_mask = (gray < 248).astype(np.uint8) * 255
        panel_mask = cv2.morphologyEx(
            panel_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        n, _, stats, _ = cv2.connectedComponentsWithStats(panel_mask, 8)
        comps: list[tuple[int, Box]] = []
        for i in range(1, n):
            x, y, ww, hh, area = [int(v) for v in stats[i]]
            if area > 0.10 * w * h and ww > 0.35 * w and hh > 0.35 * h:
                comps.append((area, Box(x, y, ww, hh)))
        panel_candidate = sorted(comps, key=lambda t: t[0], reverse=True)[0][1] if comps else None

    if panel_candidate is not None:
        plot = panel_candidate
        source = "panel_connected_component"
    elif len(v_lines) >= 2 and len(h_lines) >= 2:
        x_min, x_max = min(v_lines), max(v_lines)
        if dark_x_extents:
            x_min = min(x_min, min(a for a, _ in dark_x_extents))
            x_max = max(x_max, max(b for _, b in dark_x_extents))
        plot = Box(x_min, min(h_lines), x_max - x_min, max(h_lines) - min(h_lines))
        source = "long_grid_lines"
    else:
        # Last-resort fallback: central chart-like region.
        plot = Box(int(0.12 * w), int(0.10 * h), int(0.82 * w), int(0.75 * h))
        source = "fallback_central_region"

    # Refine with line extremes only when they genuinely coincide with the
    # plot-panel edges.  Internal grid lines can otherwise be mistaken for
    # plot boundaries; on grouped bar charts this shrinks the plot area, drops
    # the outer bars, and later causes category-label truncation.
    if v_lines and h_lines:
        xs = [x for x in v_lines if plot.x - 12 <= x <= plot.x2 + 12]
        ys = [y for y in h_lines if plot.y - 12 <= y <= plot.y2 + 12]

        if len(xs) >= 2:
            x_min, x_max = min(xs), max(xs)
            x_span = x_max - x_min
            x_near_edges = abs(x_min - plot.x) <= 8 and abs(x_max - plot.x2) <= 8
            # For line-only detections, the line extremes define the plot.  For
            # connected-component panels, only accept refinement if the extremes
            # already agree with the panel edges; do not shrink to internal grid
            # lines such as category-center grid lines.
            if source == "long_grid_lines" or (x_near_edges and x_span >= 0.85 * plot.w):
                if dark_background and dark_x_extents:
                    x_min = min(x_min, min(a for a, _ in dark_x_extents))
                    x_max = max(x_max, max(b for _, b in dark_x_extents))
                plot = Box(x_min, plot.y, x_max - x_min, plot.h)

        if len(ys) >= 2:
            y_min, y_max = min(ys), max(ys)
            y_span = y_max - y_min
            y_near_edges = abs(y_min - plot.y) <= 8 and abs(y_max - plot.y2) <= 8
            if source == "long_grid_lines" or (y_near_edges and y_span >= 0.85 * plot.h):
                plot = Box(plot.x, y_min, plot.w, y_max - y_min)

    # Keep sane bounds.
    plot.x = max(0, min(plot.x, w - 2))
    plot.y = max(0, min(plot.y, h - 2))
    plot.w = max(2, min(plot.w, w - plot.x))
    plot.h = max(2, min(plot.h, h - plot.y))
    return plot, v_lines, h_lines, source


# ---------------------------------------------------------------------------
# Bar / legend mark detection
# ---------------------------------------------------------------------------


def _dominant_color_components(rgb: np.ndarray) -> list[ColorComponent]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Saturation threshold filters text/grid/background.  The value bounds keep
    # pure white/black grid/text out while allowing dark-theme colored marks.
    color_mask = (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 25)
    if not np.any(color_mask):
        return []

    q = (rgb // 16) * 16
    flat = q[color_mask].reshape(-1, 3)
    if flat.size == 0:
        return []
    colors, counts = np.unique(flat, axis=0, return_counts=True)

    min_color_count = max(60, int(0.00025 * rgb.shape[0] * rgb.shape[1]))
    dominant = [(tuple(map(int, c)), int(n)) for c, n in zip(colors, counts) if n >= min_color_count]
    # Avoid an explosion on anti-aliased / hatched charts; the largest clusters
    # carry almost all bar information.
    dominant = sorted(dominant, key=lambda item: item[1], reverse=True)[:14]

    comps: list[ColorComponent] = []
    for color, _ in dominant:
        c = np.array(color, dtype=np.int16)
        dist = np.linalg.norm(rgb.astype(np.int16) - c.reshape(1, 1, 3), axis=2)
        mask = ((dist < 42) & color_mask).astype(np.uint8) * 255
        # DVQA bars are frequently hatched and crossed by gray grid lines.  A
        # slightly larger close operation reconnects pieces of the same mark
        # without merging neighboring bars of the same color, whose gaps are
        # normally much wider than this kernel.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 9)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, ww, hh, area = [int(v) for v in stats[i]]
            if area < 70 or ww < 4 or hh < 4:
                continue
            patch = rgb[labels == i]
            if patch.size == 0:
                mean_rgb = color
            else:
                mean_rgb = tuple(int(round(v)) for v in patch.reshape(-1, 3).mean(axis=0))
            comps.append(ColorComponent(Box(x, y, ww, hh), area, mean_rgb, color))

    # Non-maximum suppression removes duplicate detections from nearby quantized shades.
    comps = sorted(comps, key=lambda c: c.area, reverse=True)
    kept: list[ColorComponent] = []
    for comp in comps:
        if any(_iou(comp.box, k.box) > 0.50 or _contained_fraction(comp.box, k.box) > 0.82 for k in kept):
            continue
        kept.append(comp)
    return sorted(kept, key=lambda c: (c.box.y, c.box.x))



def _grayscale_mark_components(rgb: np.ndarray, plot: Box | None = None) -> list[ColorComponent]:
    """Detect solid black/gray bars and legend swatches.

    The saturated-color detector intentionally ignores low-saturation pixels,
    which is correct for text/grid noise but fails on charts whose category colors
    are black and gray.  This fallback looks for dominant grayscale intensities
    and keeps only dense rectangular components.  Thin axes, tick marks, grid
    lines, and ordinary text are rejected by the density/size filters.
    """

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h_img, w_img = gray.shape[:2]

    # Candidate grayscale pixels: low saturation, not near-white background.
    grayish = (hsv[:, :, 1] < 35) & (gray < 235)
    if plot is not None:
        # Work inside the plot interior so black bars do not connect to the
        # black x/y spines.  This is essential for grayscale charts: otherwise
        # all black bars plus the baseline become one huge sparse component and
        # get rejected by the density filter.
        region = np.zeros_like(grayish, dtype=bool)
        y1 = max(0, plot.y + 1)
        y2 = min(h_img, plot.y2 - 1)
        x1 = max(0, plot.x + 1)
        x2 = min(w_img, plot.x2 - 1)
        if y2 > y1 and x2 > x1:
            region[y1:y2, x1:x2] = True
        grayish = grayish & region
    if not np.any(grayish):
        return []

    vals, counts = np.unique(gray[grayish], return_counts=True)
    min_count = max(60, int(0.00020 * h_img * w_img))
    # Use exact dominant grayscale values rather than coarse quantization.  This
    # keeps black (0) and matplotlib gray (102) as two separate category colors.
    dominant_vals = [int(v) for v, c in sorted(zip(vals, counts), key=lambda t: t[1], reverse=True) if c >= min_count]
    dominant_vals = dominant_vals[:12]

    comps: list[ColorComponent] = []
    for val in dominant_vals:
        # Very light gray is normally the matplotlib/seaborn panel background
        # or grid/legend-frame anti-aliasing, not data.  The previous threshold
        # (>=230) let the common light-gray panel value 229 pass through; on
        # diverging horizontal charts it was split by colored bars into dense
        # rectangular background slabs and then mistaken for huge bars.  Real
        # grayscale categories used in DVQA-style charts are black/dark gray, so keep
        # the fallback conservative and skip light grays.
        if val >= 205:
            continue
        mask = ((np.abs(gray.astype(np.int16) - int(val)) <= 8) & grayish).astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, ww, hh, area = [int(v) for v in stats[i]]
            if area < 70 or ww < 6 or hh < 6:
                continue
            density = area / float(max(1, ww * hh))
            if density < 0.55:
                continue
            # Reject long 1--2 px lines from axes, ticks, and legend frames.
            if min(ww, hh) <= 3:
                continue
            # Reject ordinary glyph blobs: they are usually small and much less
            # rectangular than bars/swatches.  Keep both large bars and small
            # legend swatches.
            is_large_mark = (ww >= 8 and hh >= 20) or (ww >= 20 and hh >= 8)
            is_swatch = 8 <= ww <= 65 and 6 <= hh <= 24 and ww / max(1, hh) >= 1.25
            if not (is_large_mark or is_swatch):
                continue

            patch = rgb[labels == i]
            if patch.size == 0:
                mean_rgb = (val, val, val)
            else:
                mean_rgb = tuple(int(round(v)) for v in patch.reshape(-1, 3).mean(axis=0))
            comps.append(ColorComponent(Box(x, y, ww, hh), area, mean_rgb, mean_rgb))

    # NMS across nearby grayscale values / anti-aliased duplicates.
    comps = sorted(comps, key=lambda c: c.area, reverse=True)
    kept: list[ColorComponent] = []
    for comp in comps:
        if any(_iou(comp.box, k.box) > 0.50 or _contained_fraction(comp.box, k.box) > 0.82 for k in kept):
            continue
        kept.append(comp)
    return sorted(kept, key=lambda c: (c.box.y, c.box.x))


def _merge_component_sources(*sources: Sequence[ColorComponent]) -> list[ColorComponent]:
    """Merge saturated-color and grayscale component detections with NMS."""

    comps: list[ColorComponent] = []
    for source in sources:
        comps.extend(list(source))
    comps = sorted(comps, key=lambda c: c.area, reverse=True)
    kept: list[ColorComponent] = []
    for comp in comps:
        if any(_iou(comp.box, k.box) > 0.55 or _contained_fraction(comp.box, k.box) > 0.88 for k in kept):
            continue
        kept.append(comp)
    return sorted(kept, key=lambda c: (c.box.y, c.box.x))

def _detect_legend(
    comps: Sequence[ColorComponent], words: Sequence[OCRWord], plot: Box
) -> tuple[list[dict[str, Any]], list[ColorComponent]]:
    """Return legend entries and components considered to be legend swatches."""

    entries: list[dict[str, Any]] = []
    swatches: list[ColorComponent] = []
    used_text: set[tuple[int, int, str]] = set()

    def _ocr_text_right_of_swatch(comp: ColorComponent) -> Optional[str]:
        if pytesseract is None:
            return None
        # The swatch itself is excluded from the crop; otherwise Tesseract often
        # reads the hatch pattern as leading junk such as "mmm".
        b = comp.box
        # `comp` boxes are in image coordinates, but this helper is called from
        # analyze_chart_image_offline where the global RGB image is available via
        # a closure-like module variable?  To keep the public signature stable,
        # this OCR path is implemented below in _detect_legend_from_image.
        return None

    for comp in comps:
        b = comp.box
        # Legend swatches in DVQA/matplotlib are small horizontal rectangles,
        # often inside the plot area or just below it.
        if not (8 <= b.w <= 55 and 6 <= b.h <= 22 and comp.area <= 1200):
            continue
        # Collect ALL words on the same baseline immediately right of the swatch.
        # This lets multi-word labels like "Product A" / "young adults" survive
        # even when Tesseract emits them as separate OCRWord entries.
        same_row: list[OCRWord] = []
        for word in words:
            wb = word.box
            if _parse_float_token(word.text) is not None:
                continue
            if wb.x < b.x2 - 2:
                continue
            if abs(wb.cy - b.cy) > max(14, b.h * 1.3):
                continue
            if wb.x - b.x2 > 200:
                continue
            if not _is_useful_text(word.text):
                continue
            same_row.append(word)
        if not same_row:
            continue
        same_row.sort(key=lambda w: w.box.x)
        # Tesseract often emits duplicate readings of the same glyphs ("Product"
        # AND "ProductA" with overlapping bboxes).  Collapse overlapping words by
        # IoU, keeping the longer text (or higher confidence on tie).
        deduped: list[OCRWord] = []
        for w in same_row:
            absorbed = False
            for i, kept_w in enumerate(deduped):
                if _iou(w.box, kept_w.box) > 0.4:
                    keep_new = (
                        len(w.text) > len(kept_w.text)
                        or (len(w.text) == len(kept_w.text) and w.conf > kept_w.conf)
                    )
                    if keep_new:
                        deduped[i] = w
                    absorbed = True
                    break
            if not absorbed:
                deduped.append(w)
        # Keep words whose gap to the previous word is small enough to belong to
        # the same label.  Using ~1.5x the swatch height as a horizontal gap
        # threshold roughly matches matplotlib's default inter-word spacing.
        kept: list[OCRWord] = [deduped[0]]
        gap_thresh = max(18, int(b.h * 1.8))
        for w in deduped[1:]:
            prev = kept[-1]
            if w.box.x - prev.box.x2 <= gap_thresh:
                kept.append(w)
            else:
                break
        label = _clean_text(" ".join(w.text for w in kept))
        if not label:
            continue
        # camelCase split inside any single token Tesseract glued together
        # ("ProductB" -> "Product B"), then drop adjacent duplicate tokens that
        # come from Tesseract reading the same glyphs twice ("ProductA A").
        split_tokens: list[str] = []
        for tok in label.split(" "):
            for piece in re.split(r"(?<=[a-z])(?=[A-Z])", tok):
                if piece:
                    split_tokens.append(piece)
        cleaned: list[str] = []
        for tok in split_tokens:
            if cleaned and cleaned[-1] == tok:
                continue
            # Drop a trailing 1-char token that just repeats the suffix of the
            # previous token ("ProductA" + "A" -> "ProductA"); the camelCase
            # split above will then re-separate it as "Product A".
            if cleaned and len(tok) == 1 and cleaned[-1].endswith(tok):
                continue
            cleaned.append(tok)
        label = " ".join(cleaned)
        x1 = min(w.box.x for w in kept)
        y1 = min(w.box.y for w in kept)
        x2 = max(w.box.x2 for w in kept)
        y2 = max(w.box.y2 for w in kept)
        text_bbox = Box(x1, y1, x2 - x1, y2 - y1)
        key = (text_bbox.x, text_bbox.y, label.lower())
        if key in used_text:
            continue
        used_text.add(key)
        swatches.append(comp)
        entries.append(
            {
                "category": label,
                "rgb": comp.rgb,
                "bbox": b,
                "text_bbox": text_bbox,
            }
        )

    # Sort top-to-bottom then left-to-right; this matches visible legend order.
    entries.sort(key=lambda e: (e["bbox"].y, e["bbox"].x))
    return entries, swatches


def _ocr_small_text_crop(rgb: np.ndarray, box: Box, psm: int = 7) -> Optional[str]:
    if pytesseract is None or box.w <= 0 or box.h <= 0:
        return None
    h, w = rgb.shape[:2]
    x1, y1 = max(0, box.x), max(0, box.y)
    x2, y2 = min(w, box.x2), min(h, box.y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = rgb[y1:y2, x1:x2]
    up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    # Allow space inside the whitelist so multi-word labels like "Product A" survive.
    config = (
        f"--psm {psm} -c "
        "tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- "
    )
    # Try the requested PSM first, then fall back to single-line and block layouts.
    psm_order = []
    for p in (psm, 7, 6, 11):
        if p not in psm_order:
            psm_order.append(p)
    best_line: Optional[str] = None
    for p in psm_order:
        up_padded = cv2.copyMakeBorder(
            up, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        try:
            text = pytesseract.image_to_string(
                up_padded,
                config=(
                    f"--psm {p} -c "
                    "tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- "
                ),
            )
        except Exception:  # pragma: no cover
            continue
        text = _clean_text(text)
        # Keep multi-token labels intact (e.g. "Product A", "young adults").
        # We accept any sequence of alphabetic / alphanumeric tokens of length >=1,
        # then re-join with single spaces.  Leading single-char garbage is dropped.
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
        if not tokens:
            continue
        # Drop leading/trailing 1-char fragments (typical OCR noise from swatch hatching).
        while tokens and len(tokens[0]) == 1 and tokens[0].isalpha() and len(tokens) > 1:
            tokens.pop(0)
        while tokens and len(tokens[-1]) == 1 and tokens[-1].isalpha() and len(tokens) > 1:
            # Keep a trailing single uppercase letter when it could be a category suffix
            # ("Product A", "Plan B"); drop only obvious noise lowercase.
            if tokens[-1].isupper():
                break
            tokens.pop()
        joined = " ".join(tokens).strip()
        if joined and (len(joined) >= 2 or len(tokens) >= 2):
            # camelCase split: Tesseract often glues a category suffix to the prior
            # word ("ProductB" instead of "Product B").  Split on lower→upper
            # transitions, but only inside a single token (do not touch the gap
            # between space-separated tokens).
            split_tokens: list[str] = []
            for tok in joined.split(" "):
                pieces = re.split(r"(?<=[a-z])(?=[A-Z])", tok)
                split_tokens.extend(p for p in pieces if p)
            best_line = " ".join(split_tokens).strip()
            break
    return best_line


def _detect_legend_swatches_cv_only(
    rgb: np.ndarray, comps: Sequence[ColorComponent], plot: Box
) -> list[ColorComponent]:
    """Detect likely legend swatches without OCR text.

    This is used both as a normal pre-filter and as a fallback when the local
    Tesseract binary is not available.  Without this step, small legend swatches
    get treated as bars and may be merged into nearby bars of the same color.
    """

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    raw: list[ColorComponent] = []
    for comp in sorted(comps, key=lambda c: (c.box.y, c.box.x)):
        b = comp.box
        if not (10 <= b.w <= 60 and 6 <= b.h <= 22 and b.w / max(1, b.h) >= 1.4):
            continue
        if comp.area > 1400:
            continue
        # Matplotlib's `loc='best'` puts the legend wherever there's room, so
        # accept top-left, top-right, and bottom-right legend bands.  Centered
        # mid-plot rectangles are still rejected (those are typically bars).
        in_top_left = (plot.x - 10 <= b.x <= plot.x + 0.45 * plot.w) and (
            plot.y - 5 <= b.y <= plot.y + 0.35 * plot.h
        )
        in_top_right = (plot.x + 0.55 * plot.w <= b.x <= plot.x2 + 25) and (
            plot.y - 5 <= b.y <= plot.y + 0.35 * plot.h
        )
        in_bottom_right = (plot.x + 0.55 * plot.w <= b.x <= plot.x2 + 25) and (
            plot.y + 0.65 * plot.h <= b.y <= plot.y2 + 25
        )
        if not (in_top_left or in_top_right or in_bottom_right):
            continue
        # Check for black-ish legend text to the right.  This is not OCR; it is
        # only a geometric/ink cue to avoid confusing bar fragments with swatches.
        tx1, tx2 = min(w, b.x2 + 4), min(w, b.x2 + 118)
        ty1, ty2 = max(0, b.y - 10), min(h, b.y2 + 24)
        if tx2 <= tx1 or ty2 <= ty1:
            continue
        crop = gray[ty1:ty2, tx1:tx2]
        dark = crop < 105
        # Text has several dark pixels/components; hatching alone is sparse.
        if int(dark.sum()) < 10:
            continue
        raw.append(comp)

    if not raw:
        return []

    # Keep swatches that form a legend-like vertical stack: similar x and width.
    groups: list[list[ColorComponent]] = []
    for comp in raw:
        placed = False
        for group in groups:
            gx = float(np.median([g.box.x for g in group]))
            gw = float(np.median([g.box.w for g in group]))
            if abs(comp.box.x - gx) <= 12 and abs(comp.box.w - gw) <= 18:
                group.append(comp)
                placed = True
                break
        if not placed:
            groups.append([comp])
    best = max(groups, key=len)
    if len(best) >= 2:
        return sorted(best, key=lambda c: (c.box.y, c.box.x))
    # If only one candidate exists, still return it; some charts are single-category.
    return sorted(best, key=lambda c: (c.box.y, c.box.x))


def _detect_legend_from_image(
    rgb: np.ndarray, comps: Sequence[ColorComponent], words: Sequence[OCRWord], plot: Box
) -> tuple[list[dict[str, Any]], list[ColorComponent]]:
    """Detect legend swatches using CV and OCR the text crop beside each swatch.

    The swatches are detected with OpenCV first.  OCR is only used to name them.
    If OCR is unavailable, we still return generic category names so color-to-category
    assignment remains stable and swatches are excluded from the bar list.
    """

    # Count how many swatches CV alone finds.  When the full-image OCR path
    # covers all of them with unique labels, prefer it (full-image OCR has
    # global layout context and reads more accurately than re-OCRing a tiny
    # per-swatch crop, e.g. "north" stays "north" instead of becoming "noth").
    # Otherwise fall back to CV swatch detection + per-crop re-OCR so we don't
    # silently drop swatches the full-image OCR couldn't label.
    swatch_candidates = _detect_legend_swatches_cv_only(rgb, comps, plot)
    legend_word_entries, legend_word_swatches = _detect_legend(comps, words, plot)
    if legend_word_entries:
        unique_categories = {str(e["category"]).lower() for e in legend_word_entries}
        full_covers_all = (
            len(unique_categories) == len(legend_word_entries)
            and len(legend_word_entries) >= len(swatch_candidates)
        )
        if full_covers_all:
            return legend_word_entries, legend_word_swatches

    if not swatch_candidates:
        # Last resort: pair colored rectangles with full-image OCR words.
        return _detect_legend(comps, words, plot)

    entries: list[dict[str, Any]] = []
    swatches: list[ColorComponent] = []
    seen_labels: set[str] = set()
    # Widen the text crop so multi-word labels like "Product A" fit fully.  The
    # OCR helper internally drops single-char leading/trailing junk introduced by
    # the wider crop, so the extra width is safe.
    text_crop_width = 180
    for idx, comp in enumerate(swatch_candidates):
        b = comp.box
        text_box = Box(b.x2 + 4, b.y - 10, text_crop_width, b.h + 24)
        label = _ocr_small_text_crop(rgb, text_box, psm=7)
        if label is None:
            label = _ocr_small_text_crop(rgb, text_box, psm=6)
        if label is None:
            # Do not hallucinate a real label.  Generic names keep the structured
            # table usable for numeric/color reasoning when OCR is not configured.
            label = f"category_{idx}"
        key = label.lower()
        if key in seen_labels:
            # Same OCR string for different swatches usually means the label has
            # a per-row suffix the crop missed (e.g. "Product A" / "Product B"
            # both read as "Product").  Disambiguate with an A/B/... suffix so
            # downstream color-to-category mapping stays unique.
            suffix = chr(ord("A") + idx)
            label = f"{label} {suffix}"
            key = label.lower()
        seen_labels.add(key)
        swatches.append(comp)
        entries.append({"category": label, "rgb": comp.rgb, "bbox": b, "text_bbox": text_box})

    return entries, swatches


def _filter_bar_components(
    comps: Sequence[ColorComponent], legend_swatches: Sequence[ColorComponent], plot: Box
) -> list[ColorComponent]:
    bars: list[ColorComponent] = []
    for comp in comps:
        b = comp.box
        if any(_iou(b, s.box) > 0.35 or _contained_fraction(b, s.box) > 0.70 for s in legend_swatches):
            continue
        # Mark must overlap the plot area substantially.
        if _contained_fraction(b, plot) < 0.35:
            continue
        # Drop tiny residual OCR/color artifacts.
        if comp.area < 120 or min(b.w, b.h) < 5:
            continue
        bars.append(comp)

    # Second NMS pass after dropping legend swatches.
    bars = sorted(bars, key=lambda c: c.area, reverse=True)
    kept: list[ColorComponent] = []
    for comp in bars:
        if any(_iou(comp.box, k.box) > 0.55 or _contained_fraction(comp.box, k.box) > 0.88 for k in kept):
            continue
        kept.append(comp)
    return sorted(kept, key=lambda c: (c.box.y, c.box.x))


def _merge_bar_fragments(bars: Sequence[ColorComponent], orientation: str) -> list[ColorComponent]:
    """Merge same-color pieces of a hatched/grid-cut bar.

    The color connected-component stage can split one visual bar into several
    pieces whenever a gray grid line crosses the bar.  This routine merges
    components that share a color and overlap strongly on the non-value axis.
    """

    def color_close(a: ColorComponent, b: ColorComponent) -> bool:
        return float(np.linalg.norm(np.array(a.rgb, dtype=float) - np.array(b.rgb, dtype=float))) <= 55.0

    def axis_overlap(a: Box, b: Box) -> float:
        if orientation == "vertical":
            inter = max(0, min(a.x2, b.x2) - max(a.x, b.x))
            return inter / float(min(a.w, b.w) + 1e-9)
        inter = max(0, min(a.y2, b.y2) - max(a.y, b.y))
        return inter / float(min(a.h, b.h) + 1e-9)

    items = list(bars)
    changed = True
    while changed:
        changed = False
        used = [False] * len(items)
        merged: list[ColorComponent] = []
        for i, item in enumerate(items):
            if used[i]:
                continue
            group = [item]
            used[i] = True
            grew = True
            while grew:
                grew = False
                for j, other in enumerate(items):
                    if used[j]:
                        continue
                    if any(color_close(other, g) and axis_overlap(other.box, g.box) >= 0.58 for g in group):
                        group.append(other)
                        used[j] = True
                        grew = True
                        changed = True
            if len(group) == 1:
                merged.append(group[0])
                continue
            x1 = min(g.box.x for g in group)
            y1 = min(g.box.y for g in group)
            x2 = max(g.box.x2 for g in group)
            y2 = max(g.box.y2 for g in group)
            area = int(sum(g.area for g in group))
            weights = np.array([max(1, g.area) for g in group], dtype=float)
            rgbs = np.array([g.rgb for g in group], dtype=float)
            rgb = tuple(int(round(v)) for v in np.average(rgbs, axis=0, weights=weights))
            merged.append(ColorComponent(Box(x1, y1, x2 - x1, y2 - y1), area, rgb, group[0].color_key))
        items = merged
    return sorted(items, key=lambda c: (c.box.x if orientation == "vertical" else c.box.y, c.box.y, c.box.x))



def _merge_occluded_vertical_bar_fragments(bars: Sequence[ColorComponent]) -> list[ColorComponent]:
    """Recover vertical bars partially covered by a translucent legend box.

    In some DVQA/matplotlib charts the legend sits on top of the plot with a
    semi-transparent white frame.  The part of a black/gray bar behind that
    frame is still present in the image, but its color is lightened.  The normal
    color-component detector therefore splits one real bar into two pieces: a
    light top segment under the legend and the normal-color visible segment
    below.  This function merges such vertically aligned pieces before value
    calibration.
    """

    items = list(bars)
    if len(items) < 2:
        return items

    def gray_level(c: ColorComponent) -> float:
        return float(np.mean(np.array(c.rgb, dtype=float)))

    used: set[int] = set()
    replacements: dict[int, ColorComponent] = {}

    for i, lower in enumerate(items):
        if i in used:
            continue
        lb = lower.box
        best_j: int | None = None
        best_score = -1.0
        for j, upper in enumerate(items):
            if i == j or j in used:
                continue
            ub = upper.box
            # Upper fragment must sit immediately above the lower visible bar.
            vertical_gap = lb.y - ub.y2
            if vertical_gap < -2 or vertical_gap > 5:
                continue
            x_overlap = max(0, min(lb.x2, ub.x2) - max(lb.x, ub.x))
            overlap_frac = x_overlap / float(max(1, min(lb.w, ub.w)))
            if overlap_frac < 0.72:
                continue
            if abs(lb.w - ub.w) > max(4, 0.30 * max(lb.w, ub.w)):
                continue
            # The occluded piece is usually much lighter because the legend
            # background is semi-transparent white.  Do not merge two normal
            # same-color bars stacked by mistake.
            if gray_level(upper) <= gray_level(lower) + 35.0:
                continue
            # Prefer the closest/tallest aligned upper fragment.
            score = overlap_frac * 100.0 + upper.box.h - 5.0 * max(0, vertical_gap)
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is None:
            continue
        upper = items[best_j]
        ub = upper.box
        x1 = min(lb.x, ub.x)
        y1 = min(lb.y, ub.y)
        x2 = max(lb.x2, ub.x2)
        y2 = max(lb.y2, ub.y2)
        # Keep the lower component's RGB so legend color matching still maps to
        # the true category color, not the lightened overlay color.
        replacements[i] = ColorComponent(
            Box(x1, y1, x2 - x1, y2 - y1),
            int(lower.area + upper.area),
            lower.rgb,
            lower.color_key,
        )
        used.add(best_j)

    merged: list[ColorComponent] = []
    for i, item in enumerate(items):
        if i in used:
            continue
        merged.append(replacements.get(i, item))
    return sorted(merged, key=lambda c: (c.box.y, c.box.x))

def _infer_orientation(bars: Sequence[ColorComponent], plot: Box) -> str:
    if not bars:
        return "vertical"
    ratios = [b.box.w / max(1.0, b.box.h) for b in bars]
    med = float(np.median(ratios))
    if med > 1.25:
        return "horizontal"
    if med < 0.80:
        return "vertical"
    # Fallback from global span.
    all_box = Box(
        min(b.box.x for b in bars),
        min(b.box.y for b in bars),
        max(b.box.x2 for b in bars) - min(b.box.x for b in bars),
        max(b.box.y2 for b in bars) - min(b.box.y for b in bars),
    )
    return "horizontal" if all_box.w > all_box.h else "vertical"


# ---------------------------------------------------------------------------
# Scale calibration and label assignment
# ---------------------------------------------------------------------------


def _numeric_tick_candidates(
    words: Sequence[OCRWord], plot: Box, axis: str
) -> list[tuple[float, float, OCRWord]]:
    """Return (pixel_position, numeric_value, OCRWord) for an axis."""

    out: list[tuple[float, float, OCRWord]] = []
    for word in words:
        val = _parse_float_token(word.text)
        if val is None:
            continue
        b = word.box
        if axis == "x":
            if not (plot.y2 - 4 <= b.cy <= plot.y2 + 55):
                continue
            if not (plot.x - 35 <= b.cx <= plot.x2 + 35):
                continue
            pos = b.cx
        else:
            # Y tick labels are normally right-aligned just to the left of the
            # y-axis.  Using only the bbox center can accidentally admit rotated
            # y-axis-title fragments once the plot area is correctly expanded
            # leftward.  Keep labels whose right edge is close to the y-axis.
            if not (plot.x - 75 <= b.cx <= plot.x + 20):
                continue
            if not (plot.x - 28 <= b.x2 <= plot.x + 20):
                continue
            if not (plot.y - 25 <= b.cy <= plot.y2 + 25):
                continue
            pos = b.cy
        out.append((float(pos), float(val), word))
    # Deduplicate numeric tokens at same position.
    out = sorted(out, key=lambda t: (t[0], -t[2].conf))
    dedup: list[tuple[float, float, OCRWord]] = []
    for cand in out:
        if dedup and abs(dedup[-1][0] - cand[0]) < 5:
            # Keep the candidate that looks more like an actual tick label.
            # Decimal points / minus signs are often lost by OCR, so prefer them
            # over a slightly higher-confidence integer reading at the same box.
            def _tick_score(t):
                txt = t[2].text
                return t[2].conf + (18 if "." in txt else 0) + (14 if "-" in txt or "−" in txt else 0)
            if _tick_score(cand) > _tick_score(dedup[-1]):
                dedup[-1] = cand
        else:
            dedup.append(cand)
    return dedup


def _repair_tick_values(ticks: list[tuple[float, float, OCRWord]], axis: str) -> list[tuple[float, float, OCRWord]]:
    if len(ticks) < 2:
        return ticks
    # Synthetic DVQA axes usually span roughly [-10, 10] or [0, 10].  OCR often
    # drops decimal points, reading -7.5 or 10.0 as 75 or 100.  Repair only the
    # affected integer-looking tokens rather than scaling the entire axis.
    # First decide whether large integer-looking ticks are real large-scale
    # values (e.g. a 0--100 percent axis) or isolated decimal-point-loss OCR
    # errors (e.g. 7.5 read as 75).  The old code divided every value >=20 by
    # 10, which broke percent stacked charts by turning 100/80/60/... into
    # 10/8/6/... .  Only apply the /10 repair when large integer tokens are
    # sparse outliers; when many ticks are large, keep the real 0--100-style
    # scale.
    large_no_decimal = 0
    for _, v0, w0 in ticks:
        txt0 = _clean_text(w0.text)
        if abs(v0) >= 20 and "." not in txt0 and not re.fullmatch(r"([0-9])\1", txt0):
            large_no_decimal += 1
    divide_sparse_large_ticks = large_no_decimal <= max(1, len(ticks) // 3)

    repaired = []
    for p, v, w in ticks:
        txt = _clean_text(w.text)
        # OCR often duplicates a single tick glyph, e.g. the y tick "4" becomes
        # "44".  In the DVQA 0--10 scale this should be interpreted as 4, not
        # 4.4.  Keep the more general decimal-point-loss repair below only for
        # sparse large outliers.
        if re.fullmatch(r"([0-9])\1", txt):
            v = float(txt[0])
        elif abs(v) >= 20 and "." not in txt and divide_sparse_large_ticks:
            v = v / 10.0
        repaired.append((p, v, w))

    # If there is a zero tick, values on the left/below zero should have the
    # correct sign even if OCR lost the minus sign.
    zero_positions = [p for p, v, _ in repaired if abs(v) < 1e-6]
    if zero_positions:
        zero = float(np.median(zero_positions))
        fixed = []
        for p, v, w in repaired:
            if axis == "x" and p < zero - 3 and v > 0:
                v = -abs(v)
            if axis == "y" and p > zero + 3 and v > 0:
                v = -abs(v)
            fixed.append((p, v, w))
        repaired = fixed
    return repaired


def _fit_calibrator_from_ticks(
    ticks: list[tuple[float, float, OCRWord]], plot: Box, axis: str
) -> Optional[LinearCalibrator]:
    ticks = _repair_tick_values(ticks, axis=axis)
    if len(ticks) < 2:
        return None

    # Robust extreme-tick calibration.  Tesseract often misreads intermediate
    # labels (e.g., 8 -> 3 or 4 -> 44) on small matplotlib figures, but the top
    # and bottom/left and right extreme labels are usually correct.  When they
    # are close to the plot spines and define a plausible monotonic scale, use
    # only those two points instead of a least-squares fit polluted by bad OCR.
    if axis == "y":
        top = min(ticks, key=lambda t: abs(t[0] - plot.y))
        bottom = min(ticks, key=lambda t: abs(t[0] - plot.y2))
        if abs(top[0] - plot.y) <= 18 and abs(bottom[0] - plot.y2) <= 18:
            if top[1] > bottom[1] and 1.0 <= abs(top[1] - bottom[1]) <= 200.0:
                denom = top[0] - bottom[0]
                if abs(denom) < 1e-9:
                    return None
                a = (top[1] - bottom[1]) / denom
                b = top[1] - a * top[0]
                return LinearCalibrator(
                    float(a),
                    float(b),
                    float(plot.y),
                    float(plot.y2),
                    float(min(top[1], bottom[1])),
                    float(max(top[1], bottom[1])),
                    "ocr_y_extreme_ticks",
                )
    elif axis == "x":
        left = min(ticks, key=lambda t: abs(t[0] - plot.x))
        right = min(ticks, key=lambda t: abs(t[0] - plot.x2))
        if abs(left[0] - plot.x) <= 18 and abs(right[0] - plot.x2) <= 18:
            if right[1] > left[1] and 1.0 <= abs(right[1] - left[1]) <= 200.0:
                a = (right[1] - left[1]) / max(1e-9, right[0] - left[0])
                b = left[1] - a * left[0]
                return LinearCalibrator(
                    float(a),
                    float(b),
                    float(plot.x),
                    float(plot.x2),
                    float(min(left[1], right[1])),
                    float(max(left[1], right[1])),
                    "ocr_x_extreme_ticks",
                )

    pixels = np.array([p for p, _, _ in ticks], dtype=float)
    values = np.array([v for _, v, _ in ticks], dtype=float)
    # Remove impossible duplicate pixels/values.
    if np.std(pixels) < 2 or np.std(values) < 1e-6:
        return None

    pmin, pmax = (plot.x, plot.x2) if axis == "x" else (plot.y, plot.y2)

    # Robust pairwise fit before ordinary least squares.  Small chart fonts often
    # produce a single badly OCR'd tick, e.g. -7.5 -> -1.5, while the remaining
    # ticks are perfectly monotone.  A plain least-squares fit lets that one
    # outlier compress the whole axis and underestimates bars near 7.5.  Try all
    # tick pairs, keep the model with the most low-residual inliers, then refit
    # using only those inliers.
    best: Optional[tuple[int, float, float, np.ndarray]] = None
    n = len(ticks)
    if n >= 3:
        for i in range(n):
            for j in range(i + 1, n):
                dp = pixels[j] - pixels[i]
                dv = values[j] - values[i]
                if abs(dp) < 5 or abs(dv) < 0.5:
                    continue
                a0 = dv / dp
                if axis == "x" and a0 <= 0:
                    continue
                if axis == "y" and a0 >= 0:
                    continue
                b0 = values[i] - a0 * pixels[i]
                pred = a0 * pixels + b0
                residual = np.abs(pred - values)
                inliers = residual <= 0.35
                count = int(inliers.sum())
                if count < max(3, min(n, n // 2 + 1)):
                    continue
                # Prefer more inliers, then smaller residual among inliers, then
                # larger covered value span.
                med_res = float(np.median(residual[inliers]))
                span = float(values[inliers].max() - values[inliers].min())
                score = (count, -med_res, span)
                if best is None or score > (best[0], -best[1], best[2]):
                    best = (count, med_res, span, inliers)
        if best is not None:
            inliers = best[3]
            a, b = np.polyfit(pixels[inliers], values[inliers], 1)
            if math.isfinite(a) and abs(a) >= 1e-9:
                if not (axis == "y" and a >= 0) and not (axis == "x" and a <= 0):
                    v0, v1 = a * pmin + b, a * pmax + b
                    if 1.0 <= abs(v1 - v0) <= 200.0:
                        return LinearCalibrator(
                            a=float(a),
                            b=float(b),
                            pixel_min=float(pmin),
                            pixel_max=float(pmax),
                            value_min=float(min(v0, v1)),
                            value_max=float(max(v0, v1)),
                            source=f"ocr_{axis}_ticks_robust",
                        )

    a, b = np.polyfit(pixels, values, 1)
    if not math.isfinite(a) or abs(a) < 1e-9:
        return None
    if axis == "y" and a >= 0:
        return None
    if axis == "x" and a <= 0:
        return None
    v0, v1 = a * pmin + b, a * pmax + b
    if abs(v1 - v0) < 1.0 or abs(v1 - v0) > 200.0:
        return None
    return LinearCalibrator(
        a=float(a),
        b=float(b),
        pixel_min=float(pmin),
        pixel_max=float(pmax),
        value_min=float(min(v0, v1)),
        value_max=float(max(v0, v1)),
        source=f"ocr_{axis}_ticks",
    )


def _default_calibrator(plot: Box, axis: str, bars: Sequence[ColorComponent]) -> LinearCalibrator:
    if axis == "x":
        # If bars cross around the middle, assume [-10, 10]; otherwise [0, 10].
        min_x = min((b.box.x for b in bars), default=plot.x)
        if min_x < plot.x + 0.20 * plot.w:
            vmin, vmax = 0.0, 10.0
        else:
            vmin, vmax = -10.0, 10.0
        a = (vmax - vmin) / max(1.0, plot.w)
        b = vmin - a * plot.x
        return LinearCalibrator(a, b, float(plot.x), float(plot.x2), vmin, vmax, "default_x_range")
    # y-axis: image y increases downward, values usually increase upward.
    a = (0.0 - 10.0) / max(1.0, plot.h)
    b = 10.0 - a * plot.y
    return LinearCalibrator(a, b, float(plot.y), float(plot.y2), 0.0, 10.0, "default_y_range")


def _axis_calibrator(
    words: Sequence[OCRWord], plot: Box, axis: str, bars: Sequence[ColorComponent]
) -> LinearCalibrator:
    ticks = _numeric_tick_candidates(words, plot, axis=axis)
    cal = _fit_calibrator_from_ticks(ticks, plot, axis=axis)
    if cal is not None:
        return cal
    return _default_calibrator(plot, axis=axis, bars=bars)




def _format_numeric_tick_labels(ticks: list[tuple[float, float, OCRWord]], axis: str) -> list[str]:
    """Format numeric tick labels after applying the same OCR repairs used for scale fitting."""

    repaired = _repair_tick_values(ticks, axis=axis)
    labels: list[str] = []
    for _, value, _ in repaired:
        value = _round_chart_value(float(value))
        if abs(value) < 1e-9:
            value = 0.0
        labels.append(str(value))
    return labels

def _prepare_tick_label_crop_for_ocr(crop: np.ndarray, angle: float) -> np.ndarray:
    """Rotate, trim, upscale, and pad one tick-label crop before OCR."""

    arr = np.array(Image.fromarray(crop).rotate(angle, expand=True, fillcolor=(255, 255, 255)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Use a fairly loose threshold so anti-aliased text strokes are kept after
    # rotating the image.  This matters for 90-degree labels because the letters
    # are already thin in the original small figure.
    ink = gray < 220
    if np.any(ink):
        ys, xs = np.where(ink)
        arr = arr[
            max(0, int(ys.min()) - 5) : min(arr.shape[0], int(ys.max()) + 6),
            max(0, int(xs.min()) - 5) : min(arr.shape[1], int(xs.max()) + 6),
        ]
    arr = cv2.resize(arr, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    arr = cv2.copyMakeBorder(arr, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return arr


def _label_tokens_from_ocr_data(data: dict[str, Any]) -> list[tuple[str, float]]:
    """Extract plausible category-label tokens and their OCR confidence."""

    out: list[tuple[str, float]] = []
    texts = data.get("text", [])
    confs = data.get("conf", [])
    for raw, raw_conf in zip(texts, confs):
        try:
            conf = float(raw_conf)
        except Exception:
            conf = -1.0
        if conf < 0:
            continue
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", _clean_text(str(raw))):
            tok = tok.strip("_-")
            if len(tok) < 2:
                continue
            if _parse_float_token(tok) is not None:
                continue
            if sum(ch.isalpha() for ch in tok) == 0:
                continue
            out.append((tok, conf))
    return out


def _score_tick_label_token(token: str, conf: float, angle: float, crop: np.ndarray) -> float:
    """Score one OCR candidate for a bottom group label.

    The score intentionally has a geometry prior.  Tall/narrow crops under the
    x-axis are very likely to be matplotlib `rotation=90` tick labels, so +/-90
    candidates get a bonus there.  This lets the new path handle perpendicular
    labels while leaving horizontal and 45-degree cases mostly unchanged.
    """

    if not token:
        return -1e9
    alpha = sum(ch.isalpha() for ch in token)
    alnum = sum(ch.isalnum() for ch in token)
    if len(token) < 2 or alpha == 0:
        return -1e9

    h, w = crop.shape[:2]
    aspect = h / max(1.0, float(w))
    score = float(conf) + 4.0 * min(len(token), 10) + 2.0 * alpha + 0.5 * alnum

    # Geometry-aware angle prior.
    if aspect >= 1.25:
        if abs(angle) == 90:
            score += 22.0
        elif abs(angle) == 45:
            score -= 6.0
    elif aspect <= 0.75:
        if angle == 0:
            score += 10.0
        elif abs(angle) == 90:
            score -= 14.0
    else:
        if abs(angle) == 45:
            score += 10.0

    # Wrong-direction 90-degree OCR often returns all-caps garbage with very low
    # confidence, such as "SQWOODUI" for "income".  Keep real acronyms possible,
    # but penalize long all-uppercase tokens when Tesseract is not confident.
    if len(token) >= 4 and token.isupper() and conf < 35:
        score -= 45.0
    if conf <= 0:
        score -= 25.0
    return score


def _ocr_one_bottom_tick_crop(crop: np.ndarray) -> Optional[str]:
    """OCR one grouped bottom tick-label crop.

    We try a small set of rotations:
    - 0 degrees for horizontal labels,
    - +/-45 degrees for the existing diagonal matplotlib case,
    - +/-90 degrees for the newly added perpendicular tick-label case.

    The order is chosen from the crop aspect ratio for speed, but the final
    result is selected by OCR confidence plus the geometry-aware score above.
    """

    if pytesseract is None or Output is None or Image is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    aspect = h / max(1.0, float(w))
    if aspect >= 1.25:
        angle_order = (-90, 90, -45, 45, 0)
    elif aspect <= 0.75:
        angle_order = (0, -45, 45, -90, 90)
    else:
        angle_order = (-45, 45, 0, -90, 90)

    best: tuple[float, str] | None = None
    for angle in angle_order:
        arr = _prepare_tick_label_crop_for_ocr(crop, angle)
        for psm in (7, 8, 13):
            try:
                data = pytesseract.image_to_data(
                    arr,
                    output_type=Output.DICT,
                    config=(
                        f"--psm {psm} -c "
                        "tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- "
                    ),
                )
            except Exception:  # pragma: no cover
                continue
            tokens = _label_tokens_from_ocr_data(data)
            if not tokens:
                continue
            # Multiple tokens can happen for multi-word labels.  Keep the longest
            # confident sequence, but score the joined form so it can beat a
            # shorter accidental fragment.
            joined = " ".join(tok for tok, _ in tokens).strip()
            joined_conf = float(np.mean([conf for _, conf in tokens])) if tokens else -1.0
            candidates = [(joined, joined_conf)] + tokens
            for token, conf in candidates:
                score = _score_tick_label_token(token, conf, angle, crop)
                if best is None or score > best[0]:
                    best = (score, token)
            # Fast path: high-confidence result for the aspect-prior angle.
            if best is not None and best[0] >= 110:
                return best[1]

    if best is None or best[0] < 35:
        return None
    return best[1]


def _ocr_rotated_bottom_labels(
    rgb: np.ndarray,
    plot: Box,
    group_centers: Sequence[float],
    axis_y: float | None = None,
) -> list[tuple[str, float]]:
    """Read bottom x tick labels that may be horizontal, 45-deg, or 90-deg.

    Full-image OCR tends to fail on small matplotlib tick labels when they are
    rotated.  This function first groups black character blobs below the x-axis
    by the already-detected bar-group centers.  For each group, it OCRs the tight
    crop after trying several deskew rotations.  The previous implementation only
    rotated by -45 degrees; this version adds +/-90-degree candidates for fully
    perpendicular x tick labels while still keeping the 0 and +/-45 paths.

    `axis_y` is the y-pixel of the real x-axis.  Defaults to ``plot.y2`` when not
    provided, which is correct whenever the detected plot rectangle ends exactly
    at the x-axis.  When the panel connected-component overshoots downward and
    swallows the rotated tick labels, ``plot.y2`` sits *below* the labels, so
    the caller should pass the consensus bar-bottom instead (see
    ``_effective_axis_y``); otherwise the search band misses the labels entirely.
    """

    if pytesseract is None or Image is None or not group_centers:
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    ay = float(plot.y2 if axis_y is None else axis_y)
    # The search band must start just below the real x-axis and stretch far
    # enough to cover long perpendicular labels.  When `axis_y` is well above
    # the (overshot) `plot.y2`, we still extend the band past `plot.y2` so the
    # bottom edges of tall labels are not clipped.
    y_start = min(h - 1, int(ay) + 4)
    y_end = min(h, int(max(plot.y2, ay)) + 140)
    x_start = max(0, plot.x - 60)
    x_end = min(w, plot.x2 + 60)
    if y_end <= y_start or x_end <= x_start:
        return []

    mask = (gray < 145).astype(np.uint8) * 255
    roi = mask[y_start:y_end, x_start:x_end]
    n, _, stats, cent = cv2.connectedComponentsWithStats(roi, 8)
    comps: list[tuple[int, int, int, int, int, float, float]] = []
    for i in range(1, n):
        x, y, ww, hh, area = [int(v) for v in stats[i]]
        cx, cy = float(cent[i][0] + x_start), float(cent[i][1] + y_start)
        x += x_start
        y += y_start
        # Individual glyphs after antialiasing are small; the filters exclude
        # the bottom axis line, tick marks, and large accidental blobs.
        if 4 <= area <= 700 and 2 <= ww <= 50 and 3 <= hh <= 55 and cy > ay + 6:
            comps.append((x, y, ww, hh, area, cx, cy))
    if not comps:
        return []

    centers = sorted(float(c) for c in group_centers)
    bounds = [plot.x - 80.0]
    bounds += [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]
    bounds += [plot.x2 + 80.0]

    labels: list[tuple[str, float]] = []
    for idx, center in enumerate(centers):
        group = [co for co in comps if bounds[idx] <= co[5] < bounds[idx + 1]]
        if not group:
            continue
        x1 = max(0, min(co[0] for co in group) - 6)
        y1 = max(0, min(co[1] for co in group) - 6)
        x2 = min(w, max(co[0] + co[2] for co in group) + 6)
        y2 = min(h, max(co[1] + co[3] for co in group) + 6)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = rgb[y1:y2, x1:x2]
        text = _ocr_one_bottom_tick_crop(crop)
        if text and len(text) >= 2:
            labels.append((text, center))
    return labels


def _ocr_left_tick_labels_by_group(
    rgb: np.ndarray, plot: Box, group_centers: Sequence[float]
) -> list[tuple[str, float]]:
    """Read left-side y tick/group labels grouped by bar centers.

    Horizontal bar charts put the group/category names on the y-axis.  In many
    DVQA/matplotlib figures those labels are rotated 90 degrees, and normal
    full-image OCR often reads them as fragments such as ``D_`` or ``oO``.

    The key fix here is to crop **only the tick-label text strip** to the left
    of the plot.  The previous implementation allowed a small strip inside the
    plot; for hatched bars this pulled in the first colored bar/axis fragment,
    which then dominated the crop and made Tesseract return junk.  We now:

    1. split the left strip by detected bar-group centers;
    2. keep the right-most dark-text cluster before the plot boundary, so a
       far-left y-axis title is not confused with a tick label;
    3. OCR the tight crop with the existing rotation-aware helper, which tries
       0, +/-45, and +/-90 degrees.
    """

    if pytesseract is None or Image is None or not group_centers:
        return []

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    centers = sorted(float(c) for c in group_centers)
    if not centers:
        return []

    # Never include colored bar pixels in this crop.  Including plot.x + 18 was
    # the direct cause of the "D_ / oO / hed" failure on the attached smoke-test
    # chart: the first hatched bar begins around x=49 while the text ends around
    # x=37 and the plot starts at x=42.
    x_left = 0
    x_right = max(1, min(w, int(round(plot.x)) + 2))

    # Build y-bounds from the visual groups.  The small margin protects long
    # descenders/ascenders after rotation but still keeps adjacent labels apart.
    bounds: list[float] = [max(0.0, centers[0] - max(35.0, 0.65 * (centers[1] - centers[0]) if len(centers) > 1 else 45.0))]
    bounds += [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]
    bounds += [min(float(h), centers[-1] + max(35.0, 0.65 * (centers[-1] - centers[-2]) if len(centers) > 1 else 45.0))]

    labels: list[tuple[str, float]] = []
    for idx, center in enumerate(centers):
        y_top = max(0, int(math.floor(bounds[idx])) - 4)
        y_bot = min(h, int(math.ceil(bounds[idx + 1])) + 4)
        if y_bot <= y_top or x_right <= x_left:
            continue

        strip = rgb[y_top:y_bot, x_left:x_right]
        strip_gray = gray[y_top:y_bot, x_left:x_right]
        # Text is dark; ignore colored marks by staying outside the plot.  Use a
        # conservative threshold so anti-aliased strokes remain connected after
        # the later upscaling/rotation.
        mask = (strip_gray < 170).astype(np.uint8) * 255
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)

        comps: list[tuple[int, int, int, int, int, float, float]] = []
        for i in range(1, n):
            x, y, ww, hh, area = [int(v) for v in stats[i]]
            if not (2 <= ww <= 40 and 2 <= hh <= 80 and 3 <= area <= 900):
                continue
            cx = float(cent[i][0])
            cy = float(cent[i][1])
            # Drop tiny tick/axis fragments close to the plot edge; real label
            # glyphs have more ink and form a vertical stack.
            if x + ww >= x_right - 1 and area < 12:
                continue
            comps.append((x, y, ww, hh, area, cx, cy))

        crop: Optional[np.ndarray] = None
        if comps:
            # Cluster components by x-overlap/proximity.  A separate y-axis title
            # usually forms a farther-left cluster; group labels are nearest
            # the plot, so choose the cluster with the largest right edge.
            comps_sorted = sorted(comps, key=lambda co: co[0])
            clusters: list[list[tuple[int, int, int, int, int, float, float]]] = []
            for co in comps_sorted:
                if not clusters:
                    clusters.append([co])
                    continue
                prev_right = max(c[0] + c[2] for c in clusters[-1])
                if co[0] - prev_right <= 10:
                    clusters[-1].append(co)
                else:
                    clusters.append([co])
            chosen = max(clusters, key=lambda cl: (max(c[0] + c[2] for c in cl), sum(c[4] for c in cl)))
            x1 = max(0, min(c[0] for c in chosen) - 6)
            y1 = max(0, min(c[1] for c in chosen) - 6)
            x2 = min(strip.shape[1], max(c[0] + c[2] for c in chosen) + 6)
            y2 = min(strip.shape[0], max(c[1] + c[3] for c in chosen) + 6)
            if x2 > x1 and y2 > y1:
                crop = strip[y1:y2, x1:x2]

        # Fallback: if component grouping somehow fails, use the whole text strip
        # for this group.  It still excludes plot/bar pixels.
        if crop is None or crop.size == 0:
            crop = strip

        text = _ocr_one_bottom_tick_crop(crop)
        if text:
            text = re.sub(r"[^A-Za-z0-9 _\-]", "", _clean_text(text)).strip()
        if text and len(text) >= 2 and _parse_float_token(text) is None:
            labels.append((text, center))

    return labels

def _labels_are_plausible(labels: Sequence[tuple[str, float]], needed: int) -> bool:
    """Return whether current OCR labels are enough to trust as categories."""

    if len(labels) < needed:
        return False
    good = 0
    for text, _ in labels:
        text = _clean_text(text)
        if len(text) >= 2 and sum(ch.isalnum() for ch in text) >= 2 and sum(ch.isalpha() for ch in text) >= 1:
            good += 1
    return good >= needed


def _group_label_text_score(labels: Sequence[tuple[str, float]]) -> int:
    """Small heuristic for deciding whether a fallback OCR result is cleaner."""

    score = 0
    for text, _ in labels:
        t = _clean_text(text)
        alpha = sum(ch.isalpha() for ch in t)
        alnum = sum(ch.isalnum() for ch in t)
        score += 3 * min(alpha, 12) + alnum
        if len(t) < 2:
            score -= 8
        if re.fullmatch(r"[A-Z]{4,}", t):
            score -= 5
    return score


def _effective_axis_y(
    plot: Box, bars: Sequence[ColorComponent], baseline: float | None = None
) -> float:
    """Return the y-pixel of the real x-axis for label-region searches.

    ``plot.y2`` is normally exactly the x-axis position, but when the plot
    detector latches onto a connected component that overshoots downward, the
    rotated x-tick labels can be swallowed *inside* the plot rectangle.  In
    that case ``plot.y2`` is far below the labels and any "below the plot"
    search band misses them.

    The overshoot signal we trust is: *every* bar (including any negative bars)
    has its bottom well above ``plot.y2``.  We measure that with
    ``gap = plot.y2 - max(bar.bottom)``.  When the gap is both large in absolute
    pixels and in a fraction of the plot height, we treat the consensus
    bar-bottom as the true x-axis.  A chart with negative bars normally has at
    least one bar reaching close to ``plot.y2`` so the gap stays small and we
    keep the original ``plot.y2`` reference.

    ``baseline`` is accepted for future use but is not required for the
    overshoot decision; the gap-based test is robust enough on its own.
    """

    _ = baseline  # currently unused; reserved for future heuristics
    if not bars:
        return float(plot.y2)
    bottoms = sorted(float(b.box.y2) for b in bars)
    max_bottom = bottoms[-1]
    gap = float(plot.y2) - max_bottom
    # Require both a meaningful absolute gap (~ height of a tick label) and a
    # meaningful fraction of plot height to call this an overshoot.  These two
    # thresholds together separate the "rotated labels swallowed by plot
    # rectangle" case (large gap) from the "chart with negative bars" case
    # (small gap; lowest bar nearly reaches plot.y2).
    if gap < 22.0 or gap < 0.08 * float(plot.h):
        return float(plot.y2)
    # Median of the lower-on-screen half is a robust estimate of the axis line:
    # one floating bar (e.g. a bar with value 0 or a thin stray fragment) will
    # not move it.
    n = len(bottoms)
    lower_half = bottoms[max(0, n - max(2, (n + 1) // 2)):]
    return float(np.median(lower_half))


_LABEL_JUNK_CHARS = re.compile(r"[^A-Za-z0-9 _\-]")


def _label_has_junk_chars(text: str) -> bool:
    r"""Return True when `text` contains characters that almost never appear in
    a real DVQA-style category name.

    Tesseract reading rotated text horizontally often emits punctuation like
    ``)``, ``]``, ``|``, ``/``, or ``\`` for the disconnected glyph fragments
    of a perpendicular label.  Real group labels in these synthetic charts
    only use letters, digits, spaces, hyphens, and underscores.  Anything else
    is a strong signal the token is OCR garbage and should be dropped before
    deciding whether to fall back to the per-group rotated OCR path.
    """

    return bool(_LABEL_JUNK_CHARS.search(text or ""))


def _group_labels_from_ocr(
    words: Sequence[OCRWord],
    plot: Box,
    orientation: str,
    rgb: np.ndarray,
    group_centers: Sequence[float] | None = None,
    axis_y: float | None = None,
) -> list[tuple[str, float]]:
    labels: list[tuple[str, float]] = []
    if orientation == "vertical":
        # When `axis_y` is below `plot.y2` we still search up to `plot.y2 + 70`
        # so legitimate "below the plot rectangle" labels remain in scope.  When
        # `axis_y` is well above `plot.y2` (overshot plot rect), the band
        # starts at `axis_y - 2` and reaches just past `plot.y2` so labels
        # straddling the overshoot are captured.
        ay = float(plot.y2 if axis_y is None else axis_y)
        band_low = ay - 2.0
        band_high = max(float(plot.y2), ay) + 70.0
        candidates = []
        for w in words:
            if _parse_float_token(w.text) is not None:
                continue
            b = w.box
            # x tick labels sit below the plot and above/beside the x-axis label.
            if band_low <= b.cy <= band_high and plot.x - 30 <= b.cx <= plot.x2 + 30:
                # Short tokens are accepted only when Tesseract is confident.
                # Rotated DVQA labels produce low-conf 2-char noise like "Ss"
                # / "fF" that would otherwise crowd out the per-group rotated
                # OCR fallback; real horizontal labels like "Q1" come back at
                # conf >= 90.
                if len(w.text) >= 3:
                    candidates.append(w)
                elif len(w.text) == 2 and w.conf >= 80:
                    candidates.append(w)
        # Keep likely category row, not the axis label line: choose group closest to plot bottom.
        groups = _line_groups(candidates, y_tol=14)
        if groups:
            group = min(groups, key=lambda g: abs(np.mean([w.box.cy for w in g]) - (ay + 16)))
            labels = [(_clean_text(w.text), float(w.box.cx)) for w in sorted(group, key=lambda w: w.box.cx)]
        # If normal OCR fails or returns obvious junk, switch to the CV-assisted
        # per-group OCR (which tries horizontal, 45-deg, and 90-deg rotations).
        # Short labels like "Q1"/"Q2"/"A3" are legitimate, so accept any token of
        # length >= 2 whose characters are alphanumeric.
        if group_centers:
            n_groups = len(group_centers)
            # Drop any token with non-alnum punctuation (e.g. "Re)") before we
            # decide whether the bottom-band OCR was good enough.  These are
            # nearly always misread fragments of a rotated label and would
            # otherwise silently propagate to the bars as category names.
            pre_filter = list(labels)
            labels = [(t, p) for t, p in labels if not _label_has_junk_chars(t)]
            useful = [
                t for t, _ in labels
                if len(t) >= 2 and sum(ch.isalnum() for ch in t) >= 2
            ]
            had_junk = len(labels) < len(pre_filter)
            # Trigger the per-group rotated OCR whenever the bottom-band OCR is
            # sparse relative to the expected category count, or contained any
            # junk tokens that we just dropped.  The previous threshold capped
            # at 3 labels, which silently accepted 3 noisy tokens for a chart
            # with 7 categories.
            sparse = len(useful) < max(1, min(n_groups, 3))
            far_short = len(useful) * 2 < n_groups
            if sparse or far_short or had_junk:
                rot_labels = _ocr_rotated_bottom_labels(
                    rgb, plot, group_centers, axis_y=ay
                )
                rot_clean = [t for t, _ in rot_labels if not _label_has_junk_chars(t)]
                # Prefer the per-group result when it found more labels overall,
                # OR when it produced cleaner labels (no junk punctuation) and is
                # at least as dense as what we had.  If neither path produced
                # something usable, leave `labels` as the junk-filtered list so
                # downstream gets fewer-but-cleaner names instead of bogus
                # `Re)`-style categories.
                if len(rot_labels) > len(useful):
                    labels = rot_labels
                elif (
                    had_junk
                    and len(rot_clean) >= max(2, n_groups // 2)
                    and len(rot_clean) >= len(useful)
                ):
                    labels = rot_labels
    else:
        candidates = []
        for w in words:
            if _parse_float_token(w.text) is not None:
                continue
            b = w.box
            if plot.x - 110 <= b.cx <= plot.x + 12 and plot.y - 20 <= b.cy <= plot.y2 + 20:
                if len(w.text) >= 2:
                    candidates.append(w)
        # If OCR can read non-rotated y labels, use their real positions.
        for w in sorted(candidates, key=lambda w: w.box.cy):
            labels.append((_clean_text(w.text), float(w.box.cy)))

        if group_centers:
            needed = max(1, min(len(group_centers), 3))
            side_labels = _ocr_left_tick_labels_by_group(rgb, plot, group_centers)
            side_plausible = _labels_are_plausible(side_labels, needed)
            # The per-group side crop is more reliable than full-image OCR for
            # horizontal charts with rotated y labels.  Prefer it when it gives
            # a plausible label per visual group, or when the full-image OCR is
            # missing/incomplete/lower-quality.  This prevents junk fragments
            # such as D_/oO/hed from being accepted just because they are numerous.
            if side_labels and side_plausible and (
                len(side_labels) >= len(group_centers)
                or not _labels_are_plausible(labels, needed)
                or len(side_labels) > len(labels)
                or len(labels) != len(group_centers)
                or _group_label_text_score(side_labels) > _group_label_text_score(labels) + 4
            ):
                labels = side_labels

        # Last resort for 90-degree rotated labels when group centers are not
        # available.  Positions are unknown, so downstream assignment maps them
        # by sorted order instead of nearest pixel position.
        if not labels:
            rot_tokens = _ocr_rotated_left_labels(rgb, plot)
            labels = [(tok, float("nan")) for tok in rot_tokens]
    # OCR-confusable repair: if the labels look like a "<letter-prefix><digit>"
    # labels (e.g. Q1/Q2/Q3/Q4, A1/A2/A3) but one entry came through with 'l'
    # or 'I' instead of '1' (e.g. "Ql"), restore the digit so downstream
    # category matching stays consistent.
    if labels:
        digit_suffix_count = sum(
            1 for t, _ in labels if len(t) >= 2 and t[-1].isdigit() and t[:-1].isalpha()
        )
        if digit_suffix_count >= max(2, len(labels) // 2 + 1):
            repaired: list[tuple[str, float]] = []
            for text, pos in labels:
                # Pattern "<letters><l|I>" with no digit at end → likely a misread.
                if (
                    len(text) >= 2
                    and text[-1] in {"l", "I"}
                    and text[:-1].isalpha()
                ):
                    repaired.append((text[:-1] + "1", pos))
                else:
                    repaired.append((text, pos))
            labels = repaired

    # Remove obvious duplicates while preserving order.
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for text, pos in labels:
        key = text.lower()
        if key in seen or not text:
            continue
        seen.add(key)
        out.append((text, pos))
    return out


def _group_bar_centers(bars: Sequence[ColorComponent], orientation: str) -> list[float]:
    """Return category/group centers along the discrete axis.

    For grouped bar charts, several bars share one category and sit close to one
    another; larger gaps separate neighboring categories.  The old heuristic
    handled that case, but it collapsed single-category charts because all bar
    spacings are almost uniform, so there was no "larger" inter-category gap.
    In a single-color/no-legend chart, each bar is itself one category, so we
    return every bar center directly before applying the grouped-bar gap logic.
    """

    if not bars:
        return []

    def _coord(comp: ColorComponent) -> float:
        return float(comp.box.cx if orientation == "vertical" else comp.box.cy)

    ordered = sorted(bars, key=_coord)
    coords = [_coord(b) for b in ordered]
    if len(coords) == 1:
        return [coords[0]]

    # Single-category DVQA/matplotlib charts often have no legend and all bars use
    # the same color.  Their bar centers are evenly spaced, so the gap heuristic
    # below would merge every bar into one category.  Detect that case by color:
    # if all detected bars belong to one RGB cluster, each bar is a category.
    color_clusters: list[np.ndarray] = []
    for bar in ordered:
        rgb = np.array(bar.rgb, dtype=float)
        matched = False
        for i, center in enumerate(color_clusters):
            if float(np.linalg.norm(rgb - center)) <= 55.0:
                color_clusters[i] = 0.7 * center + 0.3 * rgb
                matched = True
                break
        if not matched:
            color_clusters.append(rgb)
    if len(color_clusters) <= 1:
        return coords

    diffs = np.diff(coords)
    # Large gaps separate categories; within-category bars are closer together.
    # Use a two-mode gap split when possible.  This is more robust than a fixed
    # median*1.35 rule when some groups have missing bars/segments.
    if len(diffs):
        threshold = max(12.0, float(np.median(diffs) * 1.35))
        sorted_diffs = np.sort(diffs.astype(float))
        if len(sorted_diffs) >= 2:
            ratios = sorted_diffs[1:] / np.maximum(sorted_diffs[:-1], 1e-6)
            j = int(np.argmax(ratios))
            small_gap = float(sorted_diffs[j])
            large_gap = float(sorted_diffs[j + 1])
            if large_gap >= max(18.0, small_gap * 1.45):
                threshold = (small_gap + large_gap) / 2.0
    else:
        threshold = 12.0
    groups: list[list[float]] = [[coords[0]]]
    for c, d in zip(coords[1:], diffs):
        if d > threshold:
            groups.append([c])
        else:
            groups[-1].append(c)
    return [float(np.mean(g)) for g in groups]



def _discrete_axis_overlap(a: Box, b: Box, orientation: str) -> float:
    """Overlap fraction on the category/discrete axis."""

    if orientation == "vertical":
        inter = max(0, min(a.x2, b.x2) - max(a.x, b.x))
        return inter / float(max(1, min(a.w, b.w)))
    inter = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    return inter / float(max(1, min(a.h, b.h)))


def _value_axis_touch(a: Box, b: Box, orientation: str, tol: int = 4) -> bool:
    """Return True when two same-position segments touch along the value axis."""

    if orientation == "vertical":
        return abs(a.y2 - b.y) <= tol or abs(b.y2 - a.y) <= tol
    return abs(a.x2 - b.x) <= tol or abs(b.x2 - a.x) <= tol


def _stacked_bar_indices(bars: Sequence[ColorComponent], orientation: str) -> set[int]:
    """Detect bar segments that form stacked bars.

    Grouped bars are side-by-side on the discrete axis, so their boxes have little
    overlap along that axis.  Stacked bars share nearly the same x/y span and
    touch along the value axis.  This detector only marks those touching,
    same-position segments, so normal grouped bars keep the old endpoint-based
    value computation.
    """

    stacked: set[int] = set()
    for i, a in enumerate(bars):
        for j in range(i + 1, len(bars)):
            b = bars[j]
            if _discrete_axis_overlap(a.box, b.box, orientation) < 0.72:
                continue
            if not _value_axis_touch(a.box, b.box, orientation):
                continue
            if float(np.linalg.norm(np.array(a.rgb, dtype=float) - np.array(b.rgb, dtype=float))) < 35.0:
                continue
            stacked.add(i)
            stacked.add(j)
    return stacked


def _bar_value_from_geometry(
    bar: ColorComponent,
    orientation: str,
    calibrator: LinearCalibrator,
    baseline: float,
    stacked_indices: set[int],
    bar_index: int,
) -> float:
    """Calibrate one bar/segment value.

    For ordinary bars, the visible rectangle extends from the zero baseline to
    the value endpoint, so the old endpoint rule is correct.  For stacked bars,
    upper segments do not start from zero; their value is the calibrated length
    of the segment itself, i.e. the difference between its two edges.
    """

    b = bar.box
    if bar_index in stacked_indices:
        if orientation == "horizontal":
            value = calibrator.value_at(b.x2) - calibrator.value_at(b.x)
        else:
            value = calibrator.value_at(b.y) - calibrator.value_at(b.y2)
        return _round_chart_value(abs(value))

    if orientation == "horizontal":
        endpoint = b.x2 if b.cx >= baseline else b.x
        value = _round_chart_value(calibrator.value_at(endpoint))
    else:
        endpoint = b.y if b.cy <= baseline else b.y2
        value = _round_chart_value(calibrator.value_at(endpoint))
    return value


def _assign_group(
    bar: ColorComponent,
    orientation: str,
    labels: Sequence[tuple[str, float]],
    group_centers: Sequence[float],
) -> str:
    pos = bar.box.cx if orientation == "vertical" else bar.box.cy
    if labels:
        # If OCR labels have real positions, use nearest.  If positions are NaN
        # (rotated-OCR fallback), map by sorted group index.
        if all(math.isfinite(p) for _, p in labels):
            text, _ = min(labels, key=lambda item: abs(item[1] - pos))
            return text
        if group_centers:
            idx = int(np.argmin([abs(c - pos) for c in group_centers]))
            if idx < len(labels):
                return labels[idx][0]
    if group_centers:
        idx = int(np.argmin([abs(c - pos) for c in group_centers]))
        return f"group_{idx}"
    return "group_0"


def _category_from_color(
    bar: ColorComponent, legend_entries: Sequence[dict[str, Any]], default: Optional[str]) -> Optional[str]:
    if not legend_entries:
        return default
    rgb = np.array(bar.rgb, dtype=float)
    best = min(
        legend_entries,
        key=lambda e: float(np.linalg.norm(rgb - np.array(e["rgb"], dtype=float))),
    )
    return str(best["category"])


def _axis_label(words: Sequence[OCRWord], plot: Box, orientation: str) -> tuple[Optional[str], Optional[str]]:
    # x-axis label: choose text line lower than tick labels, centered under plot.
    bottom_words = [
        w
        for w in words
        if plot.y2 + 24 <= w.box.cy <= plot.y2 + 80
        and plot.x - 40 <= w.box.cx <= plot.x2 + 40
        and _parse_float_token(w.text) is None
    ]
    x_label = None
    if bottom_words:
        groups = _line_groups(bottom_words, y_tol=12)
        if orientation == "horizontal":
            # For horizontal charts the value-axis label usually sits just below
            # the numeric ticks, while the legend may sit even lower.  The old
            # rule picked the lowest text line and returned legend text such as
            # "window help mile" as the x-axis label.  Choose the line closest
            # to the expected axis-label band and centered under the plot.
            target_y = plot.y2 + 38.0
            plot_cx = plot.x + plot.w / 2.0

            def _x_label_score(g: list[OCRWord]) -> float:
                cy = float(np.mean([w.box.cy for w in g]))
                cx = float(np.mean([w.box.cx for w in g]))
                text = _merge_line_text(g) or ""
                alpha = sum(ch.isalpha() for ch in text)
                centered = 1.0 - min(1.0, abs(cx - plot_cx) / max(1.0, plot.w / 2.0))
                return 12.0 * centered + 0.25 * alpha - abs(cy - target_y) / 4.0

            line = max(groups, key=_x_label_score)
        else:
            # In vertical charts, this is only a weak fallback and is later
            # filtered by the caller.
            line = max(groups, key=lambda g: np.mean([w.box.cy for w in g]))
        x_label = _merge_line_text(line)

    # y-axis label is often rotated, so OCR may be unreliable.  Keep a detected
    # vertical-side word only if it is clearly outside group labels.
    left_words = [
        w
        for w in words
        if w.box.cx < max(25, plot.x - 30)
        and plot.y - 20 <= w.box.cy <= plot.y2 + 20
        and _parse_float_token(w.text) is None
    ]
    y_label = None
    if orientation == "vertical" and left_words:
        # This usually captures the vertical axis label such as "Values".
        candidate = max(left_words, key=lambda w: (w.conf, w.box.h))
        if len(candidate.text) >= 3:
            y_label = _clean_text(candidate.text)
    return x_label, y_label


def _rotated_y_axis_label_from_image(rgb: np.ndarray, plot: Box) -> Optional[str]:
    if pytesseract is None or Image is None:
        return None
    h, _ = rgb.shape[:2]
    x1 = 0
    x2 = max(1, plot.x - 8)
    y1 = max(0, plot.y - 8)
    y2 = min(h, plot.y2 + 8)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Counter-clockwise rotation turns a standard vertical y-axis label into a
    # left-to-right word.  Numeric tick labels remain present, so we keep only
    # alphabetic tokens and choose the longest one.
    arr = np.array(Image.fromarray(crop).rotate(-90, expand=True, fillcolor=(255, 255, 255)))
    arr = cv2.resize(arr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    arr = cv2.copyMakeBorder(arr, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    tokens: list[str] = []
    for psm in (6, 11, 7):
        try:
            raw = pytesseract.image_to_string(arr, config=f"--psm {psm}")
        except Exception:  # pragma: no cover
            continue
        tokens.extend(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", _clean_text(raw)))
    if not tokens:
        return None
    token = max(tokens, key=len)
    return token if len(token) >= 3 else None


def _title_from_ocr(words: Sequence[OCRWord], plot: Box, image_width: int) -> Optional[str]:
    top_words = [
        w
        for w in words
        if w.box.cy < max(plot.y - 2, 0) and _parse_float_token(w.text) is None and len(w.text) >= 2
    ]
    if not top_words:
        return None
    groups = _line_groups(top_words, y_tol=14)
    if not groups:
        return None
    centered_groups = []
    for group in groups:
        cx = np.mean([w.box.cx for w in group])
        text_len = sum(len(w.text) for w in group)
        if text_len >= 3 and abs(cx - image_width / 2.0) <= 0.42 * image_width:
            centered_groups.append(group)
    if len(centered_groups) >= 2:
        centered_groups = sorted(centered_groups, key=lambda g: np.mean([w.box.cy for w in g]))
        merged_lines = [_merge_line_text(g) for g in centered_groups]
        merged = _clean_text(" ".join(t for t in merged_lines if t))
        if merged:
            return merged
    # Title is normally the top centered, longest non-numeric line.
    def score(group: list[OCRWord]) -> float:
        text_len = sum(len(w.text) for w in group)
        cx = np.mean([w.box.cx for w in group])
        centered = 1.0 - min(1.0, abs(cx - image_width / 2.0) / (image_width / 2.0))
        return text_len + 5.0 * centered

    best = max(groups, key=score)
    return _merge_line_text(best)


# ---------------------------------------------------------------------------
# Public analyzer
# ---------------------------------------------------------------------------


def analyze_chart_image_offline(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image file not found: {image_path}"}
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": f"Could not read image: {image_path}"}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    plot, v_lines, h_lines, plot_source = _detect_plot_area(rgb)
    ocr_ready, ocr_status = _tesseract_status()
    words = _ocr_words(rgb)
    color_components = _merge_component_sources(
        _dominant_color_components(rgb),
        _grayscale_mark_components(rgb, plot),
    )
    legend_entries, legend_swatches = _detect_legend_from_image(rgb, color_components, words, plot)
    bars = _filter_bar_components(color_components, legend_swatches, plot)
    orientation = _infer_orientation(bars, plot)
    if orientation == "vertical":
        bars = _merge_occluded_vertical_bar_fragments(bars)
    bars = _merge_bar_fragments(bars, orientation)
    orientation = _infer_orientation(bars, plot)

    value_axis = "x" if orientation == "horizontal" else "y"
    calibrator = _axis_calibrator(words, plot, value_axis, bars)
    baseline = calibrator.pixel_for(0.0)
    if value_axis == "x" and not (plot.x - 20 <= baseline <= plot.x2 + 20):
        baseline = plot.x if calibrator.value_at(plot.x) >= -0.5 else plot.x2
    if value_axis == "y" and not (plot.y - 20 <= baseline <= plot.y2 + 20):
        baseline = plot.y2

    group_centers = _group_bar_centers(bars, orientation)
    axis_y = _effective_axis_y(plot, bars, baseline=baseline if orientation == "vertical" else None)
    labels = _group_labels_from_ocr(
        words, plot, orientation, rgb, group_centers=group_centers, axis_y=axis_y
    )
    # Rotated-label OCR can pull in stray legend/axis fragments.  Only trim
    # order-only labels whose positions are NaN.  If labels have real pixel
    # positions, keep them all; nearest-position assignment is safer when some
    # bars or even whole groups are missing.
    labels_have_real_positions = bool(labels) and all(math.isfinite(pos) for _, pos in labels)
    if group_centers and len(labels) > len(group_centers) and not labels_have_real_positions:
        if orientation == "horizontal":
            labels = labels[-len(group_centers) :]
        else:
            labels = labels[: len(group_centers)]

    x_axis_label, y_axis_label = _axis_label(words, plot, orientation)
    if x_axis_label is not None:
        alpha_tokens = re.findall(r"[A-Za-z]{2,}", x_axis_label)
        if not alpha_tokens or max(len(t) for t in alpha_tokens) < 4:
            x_axis_label = None
    if y_axis_label is None and orientation == "vertical":
        y_axis_label = _rotated_y_axis_label_from_image(rgb, plot)
    title = _title_from_ocr(words, plot, image_width=w)

    stacked_indices = _stacked_bar_indices(bars, orientation)

    if not legend_entries:
        # No legend means there is no visible legend-derived category.
        # Do NOT write a placeholder such as "value" into each bar's category;
        # that makes all bars look identical to the reasoning/DSL stage.
        # The downstream ChartDSL adapter should use bar["group"] as the x/y
        # category and use a separate default category name only for the single
        # legend-side category dimension.
        default_category = None
    else:
        default_category = str(legend_entries[0]["category"])

    components: list[dict[str, Any]] = []

    # Add legend entries first as visible components.  In the output schema,
    # legend names are called "category" and discrete axis tick names are called
    # "group".  This matches the downstream JSON convention requested for DVQA:
    # e.g. {"group": "phrase", "category": "wound"}.
    for entry in legend_entries:
        components.append(
            {
                "type": "legend_entry",
                "label": entry["category"],
                "value": None,
                "category": entry["category"],
                "group": None,
                "bbox": entry["bbox"].as_schema(),
            }
        )

    # Add the plot area as an axis-like component for debugging/traceability.
    components.append(
        {
            "type": "other",
            "label": "plot_area",
            "value": None,
            "category": None,
            "group": None,
            "bbox": plot.as_schema(),
        }
    )

    # Add bar components with calibrated values.
    for bar_idx, bar in enumerate(bars):
        b = bar.box
        value = _bar_value_from_geometry(
            bar, orientation, calibrator, baseline, stacked_indices, bar_idx
        )
        group = _assign_group(bar, orientation, labels, group_centers)
        category = _category_from_color(bar, legend_entries, default_category)
        components.append(
            {
                "type": "bar",
                "label": None,
                "value": value,
                "group": group,
                "category": category,
                "bbox": b.as_schema(),
            }
        )

    # Tick labels are useful to show the CV stage really extracted plot geometry.
    x_ticks = _numeric_tick_candidates(words, plot, axis="x")
    y_ticks = _numeric_tick_candidates(words, plot, axis="y")
    x_tick_labels = _format_numeric_tick_labels(x_ticks, axis="x")
    y_tick_labels = _format_numeric_tick_labels(y_ticks, axis="y")
    if orientation == "horizontal" and x_ticks and calibrator.source.startswith("ocr_x"):
        # Report tick labels from the robust calibration rather than raw OCR.
        # This repairs isolated OCR mistakes such as -7.5 -> -1.5 while keeping
        # the actual tick positions returned by Tesseract.
        repaired_ticks = []
        for pos, _, _word in x_ticks:
            val = _round_chart_value(calibrator.value_at(float(pos)))
            if abs(val) < 1e-9:
                val = 0.0
            repaired_ticks.append(str(val))
        x_tick_labels = repaired_ticks

    # If the value axis is the common 0--10 DVQA scale and grid lines are clean,
    # synthesize stable tick labels from the calibrated grid positions.  This is
    # more reliable than trusting every OCR token on small figures.
    if orientation == "vertical" and h_lines and 0.0 <= calibrator.value_min <= 0.5 and 9.5 <= calibrator.value_max <= 10.5:
        vals = sorted({_round_chart_value(calibrator.value_at(float(r))) for r in h_lines})
        vals = [v for v in vals if -0.25 <= v <= 10.25]
        if len(vals) >= 4:
            y_tick_labels = [str(int(v)) if abs(v - int(v)) < 1e-6 else str(v) for v in vals]
    if orientation == "horizontal" and v_lines and 0.0 <= calibrator.value_min <= 0.5 and 9.5 <= calibrator.value_max <= 10.5:
        vals = sorted({_round_chart_value(calibrator.value_at(float(c))) for c in v_lines})
        vals = [v for v in vals if -0.25 <= v <= 10.25]
        if len(vals) >= 4:
            x_tick_labels = [str(int(v)) if abs(v - int(v)) < 1e-6 else str(v) for v in vals]

    # Some matplotlib figures have no grid lines, and OCR may miss the bottom
    # zero tick.  When the fitted value axis is clearly the standard DVQA 0--10
    # scale, synthesize the stable tick sequence instead of returning a partial
    # OCR list such as [10, 8, 6, 4, 2].
    if 0.0 <= calibrator.value_min <= 0.6 and 9.4 <= calibrator.value_max <= 10.6:
        standard_ticks = ["0", "2", "4", "6", "8", "10"]
        if orientation == "vertical":
            y_tick_labels = standard_ticks
        else:
            x_tick_labels = standard_ticks
    elif abs(calibrator.value_min) <= 5.0 and 94.0 <= calibrator.value_max <= 106.0:
        # Common percent axis. OCR often misses the bottom zero tick, so synthesize
        # the complete tick sequence when the calibrated range clearly indicates
        # a 0--100 scale.
        standard_ticks = ["0", "20", "40", "60", "80", "100"]
        if orientation == "vertical":
            y_tick_labels = standard_ticks
        else:
            x_tick_labels = standard_ticks

    # For horizontal charts, y_tick_labels are groups; for vertical charts,
    # x_tick_labels are groups.  This keeps the schema intuitive downstream.
    if orientation == "horizontal" and labels:
        y_tick_labels = [text for text, _ in labels]
    if orientation == "vertical" and labels:
        x_tick_labels = [text for text, _ in labels]

    notes_parts = [
        "offline_cv_perception",
        f"plot_area={plot_source}",
        f"orientation={orientation}",
        f"value_axis={value_axis}",
        f"scale={calibrator.source}",
        f"bars={len(bars)}",
        f"stacked_segments={len(stacked_indices)}",
        f"legend_entries={len(legend_entries)}",
        f"ocr_words={len(words)}",
    ]
    if not ocr_ready:
        notes_parts.append(f"ocr_unavailable={ocr_status}")
    elif len(words) == 0:
        notes_parts.append("ocr_returned_no_words")
    if not labels:
        notes_parts.append("group_ocr_uncertain")

    return {
        "chart_type": "bar_chart" if bars else "other",
        "title": title,
        "x_axis_label": x_axis_label,
        "y_axis_label": y_axis_label,
        "x_tick_labels": x_tick_labels,
        "y_tick_labels": y_tick_labels,
        "legend": [str(e["category"]) for e in legend_entries],
        "components": components,
        "notes": "; ".join(notes_parts),
    }


# ---------------------------------------------------------------------------
# CrewAI tool wrapper
# ---------------------------------------------------------------------------


class OfflineVisionToolInput(BaseModel):
    image_path: str = Field(..., description="Local path to the chart image to analyze.")


class OfflineChartVisionTool(BaseTool):
    name: str = "offline_chart_vision"
    description: str = (
        "Analyze a chart image using offline OCR and OpenCV detectors only. Pass "
        "`image_path` as a local filesystem path. Returns JSON with chart type, "
        "title, axes, tick labels, legend entries, and detected bars with group, "
        "category, calibrated value, and pixel bounding boxes."
    )
    args_schema: Type[BaseModel] = OfflineVisionToolInput

    def _run(self, image_path: str) -> str:
        return json.dumps(analyze_chart_image_offline(image_path), ensure_ascii=False)