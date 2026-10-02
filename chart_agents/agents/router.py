"""Round 1 Router Agent.

The router decides which solver branches should be activated. This is important
for cost control: not every question needs every solver.
"""

from __future__ import annotations

from chart_agents.blackboard import BlackboardState, make_claim


class QuestionRouterAgent:
    name = "question_router"

    def run(self, state: BlackboardState):
        q = state.question.lower().strip()
        required: list[str]
        optional: list[str] = []
        task = "other"
        operation = "unknown"
        confidence = 0.70

        if any(k in q for k in ["largest gap", "biggest gap", "largest difference", "biggest difference"]):
            task = "comparison"
            operation = "max_difference"
            required = ["comparison_solver", "arithmetic_solver"]
            optional = ["format_critic", "evidence_critic"]
            confidence = 0.94
        elif q.startswith("which") and any(k in q for k in ["largest", "highest", "maximum", "most"]):
            task = "comparison"
            operation = "max"
            required = ["comparison_solver"]
            optional = ["arithmetic_solver", "format_critic", "evidence_critic"]
            confidence = 0.82
        elif any(k in q for k in ["difference", "gap", "how much more", "how many more"]):
            task = "arithmetic"
            operation = "difference"
            required = ["arithmetic_solver"]
            optional = ["comparison_solver", "format_critic", "evidence_critic"]
            confidence = 0.80
        elif q.startswith("how many") or "number of" in q or "count" in q:
            task = "counting"
            operation = "count"
            required = ["counting_solver"]
            optional = ["label_matching_solver"]
            confidence = 0.90
        elif q.startswith(("is ", "are ", "does ", "do ", "did ", "was ", "were ", "has ", "have ")):
            task = "boolean"
            operation = "yes_no_comparison"
            required = ["boolean_solver"]
            optional = ["lookup_solver", "comparison_solver", "arithmetic_solver"]
            confidence = 0.88
        elif any(k in q for k in ["what is the value", "what value", "value of", "how much is", "what was the value"]):
            task = "lookup"
            operation = "lookup_cell"
            required = ["lookup_solver"]
            optional = ["label_matching_solver"]
            confidence = 0.88
        elif any(k in q for k in ["label", "category", "legend", "x-axis", "x axis", "y-axis", "y axis"]):
            task = "label_matching"
            operation = "label_or_metadata"
            required = ["label_matching_solver"]
            confidence = 0.78
        else:
            # MVP fallback: keep these two solvers available for demo/debug.
            task = "other_or_unknown"
            operation = "fallback"
            required = ["lookup_solver", "comparison_solver", "arithmetic_solver", "label_matching_solver"]
            optional = ["format_critic", "evidence_critic"]
            confidence = 0.55

        all_known = [
            "lookup_solver",
            "arithmetic_solver",
            "comparison_solver",
            "counting_solver",
            "label_matching_solver",
            "boolean_solver"
        ]
        content = {
            "task": task,
            "operation": operation,
            "required_solvers": required,
            "optional_solvers": optional,
            "skip_solvers": [s for s in all_known if s not in required + optional],
            "cost_policy": "activate selected branches only; do not run all solvers by default",
        }
        return state.post(make_claim(self.name, content, confidence=confidence, round_id=1))
