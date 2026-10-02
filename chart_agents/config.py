"""Centralized runtime configuration. Values are read from environment vars
(or a local .env file) at import time. See .env.example for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Models (litellm-format strings) ---
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    # llm_model: str = os.getenv("LLM_MODEL", "gpt-5-mini")
    # vision_model: str = os.getenv("VISION_MODEL", "o4-mini")
    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o")
    # vision_model: str = os.getenv("VISION_MODEL", "gpt-4o-mini")

    # --- Generation ---
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    vision_temperature: float = float(os.getenv("VISION_TEMPERATURE", "0.3"))
    vision_max_tokens: int = int(os.getenv("VISION_MAX_TOKENS", "2048"))

    # --- Perception backend ---
    # offline_cv keeps the perception stage local / CV-style. Set to "vlm"
    # to recover the previous vision-language-model perception call.
    perception_backend: str = os.getenv("PERCEPTION_BACKEND", "offline_cv").strip().lower()

    # --- Output ---
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    save_intermediates: bool = _bool("SAVE_INTERMEDIATES", True)

    # --- Bounded reasoning-only repair in ChartReasoningCrew ---
    max_refinement_steps: int = int(os.getenv("MAX_REFINEMENT_STEPS", "3"))


settings = Settings()
