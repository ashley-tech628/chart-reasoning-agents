"""Shared blackboard state for round-based downstream chart reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chart_agents.blackboard.messages import BlackboardMessage


def _dump_obj(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


@dataclass
class BlackboardState:
    """A shared workspace used by all downstream agents.

    Agents do not directly pass outputs one-by-one. Instead, each agent reads
    the current blackboard and posts messages: claims, candidates, supports,
    objections, and final decisions.
    """

    question: str
    chart_dsl: Any
    messages: List[BlackboardMessage] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_explanation: Optional[str] = None
    final_confidence: float = 0.0

    def post(self, message: BlackboardMessage) -> BlackboardMessage:
        self.messages.append(message)
        return message

    def by_type(self, message_type: str) -> List[BlackboardMessage]:
        return [m for m in self.messages if m.message_type == message_type]

    def claims(self) -> List[BlackboardMessage]:
        return self.by_type("claim")

    def candidates(self) -> List[BlackboardMessage]:
        return self.by_type("candidate")

    def supports(self) -> List[BlackboardMessage]:
        return self.by_type("support")

    def objections(self) -> List[BlackboardMessage]:
        return self.by_type("objection")

    def decisions(self) -> List[BlackboardMessage]:
        return self.by_type("final_decision")

    def latest_claim_from(self, agent_name: str) -> Optional[BlackboardMessage]:
        for msg in reversed(self.claims()):
            if msg.from_agent == agent_name:
                return msg
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "chart_dsl": _dump_obj(self.chart_dsl),
            "messages": [_dump_obj(m) for m in self.messages],
            "final_answer": self.final_answer,
            "final_explanation": self.final_explanation,
            "final_confidence": self.final_confidence,
        }
