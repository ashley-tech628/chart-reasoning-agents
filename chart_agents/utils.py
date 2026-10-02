"""I/O, logging, and small parsing helpers shared across agents."""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("chart_agents")


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


def encode_image_b64(path: str | Path) -> Tuple[str, str]:
    """Return (base64_string, media_type) for a local image file."""
    p = Path(path)
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    suffix = p.suffix.lstrip(".").lower() or "png"
    media = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return b64, media


# ---------------------------------------------------------------------------
# Run directory & intermediate saving
# ---------------------------------------------------------------------------


def new_run_dir(base: Path, tag: Optional[str] = None) -> Path:
    """Create and return a per-run output directory: <base>/<timestamp>[-<tag>]."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{tag}" if tag else stamp
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_intermediate(obj: Any, name: str, run_dir: Path) -> Path:
    """Save an arbitrary object (Pydantic model, dict, or string) as JSON."""
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"{name}.json"
    if isinstance(obj, BaseModel):
        data: Any = obj.model_dump()
    elif isinstance(obj, str):
        data = {"text": obj}
    else:
        data = obj
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# JSON parsing (LLM output is sometimes fenced)
# ---------------------------------------------------------------------------


def strip_json_fences(text: str) -> str:
    """Remove markdown ```json ... ``` fences from an LLM response, if present."""
    t = text.strip()
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def parse_json_safe(text: str) -> dict:
    """Parse JSON from a possibly fenced model response. Raises ValueError on failure."""
    cleaned = strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON: {e}\n--- raw ---\n{text[:1000]}") from e
