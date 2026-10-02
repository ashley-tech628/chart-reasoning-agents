"""Round 4 Simple Judge Agent."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

from chart_agents.agents.answer_type import get_answer_type_claim
from chart_agents.agents.normalizer import normalize_answer
from chart_agents.blackboard import BlackboardState, make_final_decision
from chart_agents.solvers.utils import is_numeric, norm_text


def _valid_label(answer: str, valid_answers: list[str]) -> bool:
    return any(norm_text(answer) == norm_text(v) for v in valid_answers or [])


class SimpleJudgeAgent:
    name = "judge_agent"

    def run(self, state: BlackboardState):
        answer_type = get_answer_type_claim(state)
        expected = answer_type.get("answer_type", "unknown")
        valid = answer_type.get("valid_answers", []) or []

        objections_by_target: Dict[str, list] = defaultdict(list)
        supports_by_target: Dict[str, list] = defaultdict(list)
        for obj in state.objections():
            objections_by_target[obj.target].append(obj)
        for sup in state.supports():
            supports_by_target[sup.target].append(sup)

        scored: list[tuple[float, str, Any]] = []
        for cand in state.candidates():
            if not cand.answer:
                continue
            ans = normalize_answer(cand.answer, expected, valid)
            score = float(cand.confidence or 0.0)
            score += 0.15 * len(supports_by_target.get(cand.from_agent, []))
            score -= 0.45 * len(objections_by_target.get(cand.from_agent, []))

            # Answer-type gate: wrong answer type should be heavily downweighted.
            if expected in {"group", "category", "label"}:
                if is_numeric(ans):
                    score -= 1.0
                elif valid and _valid_label(ans, valid):
                    score += 0.40
            elif expected == "number":
                if is_numeric(ans):
                    score += 1.0
                else:
                    score -= 2.0
            elif expected == "boolean":
                if ans in {"yes", "no"}:
                    score += 1.0
                else:
                    score -= 2.0
            elif expected == "count":
                count_words = {
                    "zero", "one", "two", "three", "four", "five",
                    "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve", "thirteen", "fourteen",
                    "fifteen", "sixteen", "seventeen", "eighteen",
                    "nineteen", "twenty"
                }
                if is_numeric(ans) or ans.lower() in count_words:
                    score += 1.0
                else:
                    score -= 2.0
            scored.append((score, ans, cand))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, final_answer, best_cand = scored[0]
            rejected = [
                {
                    "answer": ans,
                    "from_agent": cand.from_agent,
                    "score": round(score, 3),
                    "reason": "lower score or answer-type mismatch",
                }
                for score, ans, cand in scored[1:]
            ]
            confidence = max(0.0, min(1.0, best_score))
            explanation = self._build_explanation(best_cand, final_answer)
            selected_from = [best_cand.from_agent]
        else:
            final_answer = ""
            rejected = []
            confidence = 0.0
            explanation = "No solver produced a valid candidate."
            selected_from = []

        state.final_answer = final_answer
        state.final_explanation = explanation
        state.final_confidence = confidence

        content = {
            "final_answer": final_answer,
            "selected_from": selected_from,
            "rejected_candidates": rejected,
            "explanation": explanation,
            "answer_type_used": answer_type,
            "decision_rule": "candidate confidence + support votes - objections + answer-type gate",
        }
        return state.post(make_final_decision(self.name, final_answer, content, confidence=confidence, round_id=4))

    @staticmethod
    def _build_explanation(candidate, final_answer: str) -> str:
        support = candidate.support or {}
        if isinstance(support, dict):
            if support.get("explanation"):
                return str(support["explanation"])
            details = support.get("details", {}) or {}
            if "max_gap" in details and "max_gap_group" in details:
                return (
                    f"The largest gap is {details['max_gap']} in {details['max_gap_group']}; "
                    f"the selected final answer is {final_answer}."
                )
        return f"The judge selected {final_answer} from {candidate.from_agent} based on evidence, answer type, and critique messages."
