import unittest
from dataclasses import replace

import numpy as np

from real_ecology_benchmark.dynamics import _design
from real_ecology_benchmark.methods import FAITHFUL_METHODS, METHODS, NATIVE_METHODS
from real_ecology_benchmark.methods.ensemble_value_disagreement import (
    EnsembleValueDisagreementPolicy,
)

from .common import tiny_config, tiny_data


class MethodSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.dataset, cls.private, cls.factory, cls.cache = tiny_data()

    def test_all_methods_fit_and_act(self):
        first = self.dataset.episode_indices()[0][0]
        for name, method_cls in METHODS.items():
            if name in NATIVE_METHODS or name in FAITHFUL_METHODS or name == "delphic":
                continue
            with self.subTest(method=name):
                kwargs = {}
                if name == "bamcts":
                    kwargs.update(simulations=16, depth=3)
                if name == "ensemble_value_disagreement_pessimism":
                    kwargs.update(ensemble_size=4)
                if name == "ogsrl":
                    kwargs.update(train_iterations=3)
                method = method_cls(
                    self.cfg.environment, self.cfg.model, self.cfg.planner,
                    seed=42, **kwargs,
                )
                metrics = method.fit(self.dataset, self.cache)
                self.assertIsInstance(metrics, dict)
                belief = self.factory().reset(float(self.dataset.observations[first]), 123)
                method.reset(123)
                action = method.act(belief, float(self.dataset.observations[first]))
                self.assertGreaterEqual(action, 0)
                self.assertLess(action, self.cfg.environment.num_actions)
                if name == "ogsrl":
                    self.assertIn("unsafe_probability_all", method.last_diagnostics)
                    method.deployment_safety_limit = -1.0
                    method.deployment_ood_limit = -1.0
                    method.act(belief, float(self.dataset.observations[first]))
                    self.assertEqual(method.last_diagnostics["hard_fallback"], 1.0)

    def test_all_methods_fit_and_act_cumulative_controls(self):
        cfg = tiny_config(kind="allee", sigma=0.1, actions=5)
        cfg.environment = replace(cfg.environment, control_mode="cumulative_capped")
        cfg.dataset = replace(cfg.dataset, transitions=120, episode_length=8)
        cfg.filter = replace(cfg.filter, particles=32)
        cfg.planner = replace(cfg.planner, horizon=2, sequences=8, particles=4)
        cfg, dataset, _private, factory, cache = tiny_data(cfg)
        first = dataset.episode_indices()[0][0]
        for name, method_cls in METHODS.items():
            if name in NATIVE_METHODS or name in FAITHFUL_METHODS or name == "delphic":
                continue
            with self.subTest(method=name):
                kwargs = {}
                if name == "bamcts":
                    kwargs.update(simulations=8, depth=2)
                if name == "ensemble_value_disagreement_pessimism":
                    kwargs.update(ensemble_size=3)
                if name == "ogsrl":
                    kwargs.update(train_iterations=2)
                method = method_cls(
                    cfg.environment, cfg.model, cfg.planner, seed=12, **kwargs,
                )
                metrics = method.fit(dataset, cache)
                self.assertIsInstance(metrics, dict)
                belief = factory().reset(float(dataset.observations[first]), 123)
                self.assertIsNotNone(belief.rho)
                method.reset(123)
                action = method.act(belief, float(dataset.observations[first]))
                self.assertGreaterEqual(action, 0)
                self.assertLess(action, cfg.environment.num_actions)

    def test_dynamics_design_has_action_state_interactions(self):
        design = _design(
            np.asarray([100.0, 200.0]), np.asarray([0, 1]), 5, 500.0
        )
        self.assertEqual(design.shape[1], 3 + 3 * 5)
        interaction = design[:, 3 + 5 : 3 + 10]
        self.assertGreater(interaction[0, 0], 0.0)
        self.assertEqual(interaction[0, 1], 0.0)
        self.assertGreater(interaction[1, 1], interaction[0, 1])

    def test_bootstrap_q_disagreement_is_fitted_and_deterministic(self):
        policy = EnsembleValueDisagreementPolicy(
            self.cfg.environment, self.cfg.model, self.cfg.planner,
            seed=42, ensemble_size=4,
        )
        metrics = policy.fit(self.dataset, self.cache)
        repeat = EnsembleValueDisagreementPolicy(
            self.cfg.environment, self.cfg.model, self.cfg.planner,
            seed=42, ensemble_size=4,
        )
        repeat.fit(self.dataset, self.cache)
        self.assertGreater(metrics["q_ensemble_disagreement_observed"], 0.0)
        self.assertTrue(np.array_equal(policy.q_weights, repeat.q_weights))


if __name__ == "__main__":
    unittest.main()
