import csv
import tempfile
import unittest
from pathlib import Path

from real_ecology_benchmark.evaluator import ContinuousEvaluator
from real_ecology_benchmark.gate import gate_decision, run_decision_gate
from real_ecology_benchmark.manifest import make_manifest
from real_ecology_benchmark.methods.mopo import MOPOPolicy

from .common import tiny_data


class EvaluatorTests(unittest.TestCase):
    def test_paired_evaluator_outputs_metrics(self):
        cfg, dataset, _private, factory, cache = tiny_data()
        method = MOPOPolicy(cfg.environment, cfg.model, cfg.planner, seed=5)
        method.fit(dataset, cache)
        rows = ContinuousEvaluator(cfg, factory).run(method)
        self.assertEqual(len(rows), 1)
        self.assertIn("filter_rmse", rows[0])
        self.assertIn("filter_coverage90", rows[0])
        self.assertIn("filter_unsafe_brier", rows[0])
        self.assertIn("true_return", rows[0])
        self.assertLessEqual(rows[0]["n_steps"], cfg.evaluation.horizon)

    def test_policy_observe_receives_public_feedback_only(self):
        cfg, _dataset, _private, factory, _cache = tiny_data()

        class InspectPolicy:
            name = "inspect"
            num_actions = cfg.environment.num_actions
            last_diagnostics = {}
            saw_private_channel = False

            def reset(self, seed):
                return None

            def act(self, belief, observation):
                return 0

            def observe(self, belief, action, result):
                self.saw_private_channel |= hasattr(result, "evaluator_info")

        policy = InspectPolicy()
        ContinuousEvaluator(cfg, factory).run(policy)
        self.assertFalse(policy.saw_private_channel)

    def test_three_controller_gate_schema(self):
        cfg, _dataset, _private, _factory, _cache = tiny_data()
        cfg.filter.particles = 24
        cfg.planner.sequences = 8
        cfg.planner.particles = 4
        cfg.planner.horizon = 2
        cfg.evaluation.horizon = 4
        result = run_decision_gate(cfg, episodes=1)
        self.assertIn("belief_oracle_return", result)
        self.assertIn("belief_ricker_return", result)
        self.assertIn("clairvoyant_return", result)
        self.assertIn("passed", result)

    def test_gate_rule_has_explicit_pass_and_fail_cases(self):
        self.assertTrue(gate_decision("allee", 1.1, 0.06)["passed"])
        self.assertFalse(gate_decision("allee", 0.9, 0.06)["passed"])
        theta = gate_decision("theta", 0.6, 0.0)
        self.assertFalse(theta["hard_gate"])
        self.assertTrue(theta["passed"])

    def test_manifest_contains_baseline_fidelity_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            self.assertEqual(make_manifest(path), 384)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        ricker = [row for row in rows if row["filter"] == "ricker"]
        self.assertEqual(len(ricker), 48)
        self.assertEqual({row["method"] for row in ricker}, {"plus", "moor"})


if __name__ == "__main__":
    unittest.main()
