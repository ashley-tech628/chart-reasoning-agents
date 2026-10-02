"""Round 3 Evidence Critic Agent."""

from __future__ import annotations

from chart_agents.blackboard import BlackboardState, make_support


class EvidenceCritic:
    name = "evidence_critic"

    def run(self, state: BlackboardState):
        messages = []
        for cand in state.candidates():
            details = ((cand.support or {}).get("details") if cand.support else {}) or {}
            if cand.answer and details:
                content = f"Candidate {cand.answer} is supported by computed evidence: {details}"
                messages.append(state.post(make_support(self.name, cand.from_agent, content, confidence=0.85, round_id=3)))
        return messages
