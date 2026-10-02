"""Structured messages for the round-based blackboard workflow.

Agents communicate by posting these messages to the shared BlackboardState.
The goal is to make downstream reasoning look like agent collaboration, not a
single sequential chain.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BlackboardMessage(BaseModel):
    """One message posted by an agent.

    message_type is intentionally simple so new agents can be added later:
    - claim: interpretation/routing/answer-type claims
    - candidate: a solver's proposed answer
    - support: a critic or agent supports a candidate
    - objection: a critic or agent objects to a candidate
    - final_decision: judge's final choice
    - repair_request: branch-level repair request, for future extension
    """

    round: int = Field(..., description="Round number: 1 interpretation, 2 candidate, 3 critique, 4 consensus")
    message_type: str = Field(..., description="claim | candidate | support | objection | final_decision | repair_request")
    from_agent: str
    target: Optional[str] = None
    content: Any = None
    answer: Optional[str] = None
    support: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    severity: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")


def make_claim(from_agent: str, content: Any, confidence: float = 0.0, *, round_id: int = 1) -> BlackboardMessage:
    return BlackboardMessage(round=round_id, message_type="claim", from_agent=from_agent, content=content, confidence=confidence)


def make_candidate(from_agent: str, answer: str | None, support: Dict[str, Any] | None = None, confidence: float = 0.0, *, round_id: int = 2) -> BlackboardMessage:
    return BlackboardMessage(round=round_id, message_type="candidate", from_agent=from_agent, answer=answer, support=support or {}, confidence=confidence)


def make_support(from_agent: str, target: str, content: str, confidence: float = 0.0, *, round_id: int = 3) -> BlackboardMessage:
    return BlackboardMessage(round=round_id, message_type="support", from_agent=from_agent, target=target, content=content, severity="support", confidence=confidence)


def make_objection(from_agent: str, target: str, content: str, severity: str = "high", confidence: float = 0.0, *, round_id: int = 3) -> BlackboardMessage:
    return BlackboardMessage(round=round_id, message_type="objection", from_agent=from_agent, target=target, content=content, severity=severity, confidence=confidence)


def make_final_decision(from_agent: str, answer: str, content: Dict[str, Any], confidence: float = 0.0, *, round_id: int = 4) -> BlackboardMessage:
    return BlackboardMessage(round=round_id, message_type="final_decision", from_agent=from_agent, answer=answer, content=content, confidence=confidence)
