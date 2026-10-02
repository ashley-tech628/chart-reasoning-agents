"""Offline behavior and historical-evidence integrity checks."""
import json
import tempfile
import unittest
from pathlib import Path

from chart_agents.blackboard import BlackboardState
from chart_agents.agents.router import QuestionRouterAgent
from chart_agents.schemas import ChartDSL
from chart_agents.workflows.blackboard_workflow import run_blackboard_workflow
from scripts.run_blackboard_demo import sample_chart_dsl
from scripts.verify_results import audit, load_rows


class BlackboardTests(unittest.TestCase):
    def test_largest_gap_and_trace(self):
        state, answer = run_blackboard_workflow(
            "Which quarter has the largest gap between Product A and Product B?", sample_chart_dsl())
        self.assertEqual(answer.answer, "Q3")
        self.assertEqual({v.name: v.value for v in answer.intermediate_values},
                         {"gap_Q1": 7, "gap_Q2": 11, "gap_Q3": 13, "gap_Q4": 8})
        self.assertEqual({m.round for m in state.messages}, {1, 2, 3, 4})

    def test_changed_data_changes_answer(self):
        dsl = sample_chart_dsl()
        dsl.data[0].value = 100
        _, answer = run_blackboard_workflow(
            "Which quarter has the largest gap between Product A and Product B?", dsl)
        self.assertEqual(answer.answer, "Q1")

    def test_empty_chart_abstains(self):
        _, answer = run_blackboard_workflow("Which quarter has the largest gap?", ChartDSL(type="bar_chart"))
        self.assertEqual(answer.answer, "")

    def test_label_route_does_not_crash(self):
        state = BlackboardState(question="What is the x-axis label?", chart_dsl=sample_chart_dsl())
        msg = QuestionRouterAgent().run(state)
        self.assertEqual(msg.content["task"], "label_matching")

    def test_fixture_roundtrip(self):
        dsl = sample_chart_dsl()
        self.assertEqual(ChartDSL.model_validate_json(dsl.model_dump_json()), dsl)


class HistoricalEvidenceTests(unittest.TestCase):
    def test_saved_scores_match_report(self):
        results = audit()
        self.assertEqual([r["n_paired"] for r in results], [30, 30])

    def invalid_rows(self, rows, expected):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, expected):
                load_rows(p)

    def test_duplicate_ids_rejected(self):
        row = {"question_id": 1, "pred": "x", "gold": "x"}
        self.invalid_rows([row, row], "Duplicate")

    def test_failed_predictions_not_dropped(self):
        self.invalid_rows([{"question_id": 1, "error": "timeout"}], "Failed")

    def test_missing_prediction_rejected(self):
        self.invalid_rows([{"question_id": 1, "gold": "x"}], "Expected string")


if __name__ == "__main__":
    unittest.main()
