import unittest
from dataclasses import replace

import numpy as np

from real_ecology_benchmark.dynamics import _design
from real_ecology_benchmark.methods import FAITHFUL_METHODS, METHODS, NATIVE_METHODS
from real_ecology_benchmark.methods.delphic import DelphicCQLPolicy

from .common import tiny_config, tiny_data


class MethodSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.dataset, cls.private, cls.factory, cls.cache = tiny_data()

    def test_all_methods_fit_and_act(self):
        first = self.dataset.episode_indices()[0][0]
        for name, method_cls in METHODS.items():
            if name in NATIVE_METHODS or name in FAITHFUL_METHODS:
                continue
            with self.subTest(method=name):
                kwargs = {}
                if name == "bamcts":
                    kwargs.update(simulations=16, depth=3)
                if name == "delphic":
                    kwargs.update(world_count=4)
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
            if name in NATIVE_METHODS or name in FAITHFUL_METHODS:
                continue
            with self.subTest(method=name):
                kwargs = {}
                if name == "bamcts":
                    kwargs.update(simulations=8, depth=2)
                if name == "delphic":
                    kwargs.update(world_count=3)
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

    def test_delphic_uncertainty_responds_to_hidden_state_ambiguity(self):
        metrics = []
        for sigma in (0.0, 0.1, 0.2, 0.4):
            cfg, dataset, _private, _factory, cache = tiny_data(tiny_config(sigma=sigma))
            policy = DelphicCQLPolicy(
                cfg.environment, cfg.model, cfg.planner, seed=42, world_count=4
            )
            metrics.append(policy.fit(dataset, cache))
        self.assertLess(metrics[0]["posterior_ambiguity_mean"], 1e-10)
        self.assertLess(metrics[0]["mean_delphic_uncertainty"], 1e-10)
        ambiguity = [row["posterior_ambiguity_mean"] for row in metrics]
        self.assertTrue(
            all(right > left for left, right in zip(ambiguity, ambiguity[1:]))
        )
        self.assertTrue(all(
            row["mean_delphic_uncertainty"]
            > metrics[0]["mean_delphic_uncertainty"] + 1e-8
            for row in metrics[1:]
        ))


if __name__ == "__main__":
    unittest.main()
