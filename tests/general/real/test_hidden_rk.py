"""Acceptance gates for the hidden-r/K real-ecology information regime."""

from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from real_ecology_benchmark import realdata
from real_ecology_benchmark.beliefs import (
    PublicObservationFilter,
    cache_public_beliefs,
)
from real_ecology_benchmark.collector import collect_dataset
from real_ecology_benchmark.config import (
    BenchmarkConfig,
    FilterConfig,
    MethodContext,
    ModelConfig,
    PlannerConfig,
    TrainingConfig,
    real_environment,
)
from real_ecology_benchmark.dataset import (
    PUBLIC_CONTROL_FIELDS,
    PUBLIC_FIELDS,
    assert_public_schema,
    save_public,
)
from real_ecology_benchmark.envs import make_env
from real_ecology_benchmark.manifest import make_manifest
from real_ecology_benchmark.methods import FAITHFUL_METHODS, METHODS
from real_ecology_benchmark.native_fit import (
    PUBLIC_FORM_CANDIDATES,
    build_fitted_solver,
)
from real_ecology_benchmark.pipeline import build_method
from real_ecology_benchmark.privacy import (
    assert_hidden_method_artifact,
    forbidden_method_name,
)
from real_ecology_benchmark.public_surrogate import (
    evaluator_only_surrogate_diagnostics,
    fit_public_surrogate,
)


class HiddenRKContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_cfg = real_environment(
            "Amur tiger",
            "ricker",
            expose_rk="hidden",
            observation_noise_sigma=0.1,
        )
        cls.dataset, cls.private = collect_dataset(
            make_env(cls.env_cfg), 144, 12, seed=116
        )
        cls.surrogate = fit_public_surrogate(cls.dataset, seed=20_116)
        cls.context = MethodContext(
            num_actions=cls.env_cfg.num_actions,
            action_costs=tuple(float(value) for value in cls.dataset.action_costs),
            action_channels=realdata.public_action_channels(),
            observation_noise_sigma=cls.env_cfg.observation_noise_sigma,
            horizon=cls.env_cfg.horizon,
            observation_scale=cls.surrogate.observation_scale,
            pop_id=str(cls.dataset.pop_ids[0]),
            reward_mode=cls.env_cfg.reward_mode,
            surrogate=cls.surrogate,
        )
        cls.filter_cfg = FilterConfig(particles=32, proposal="learned")
        cls.factory = staticmethod(
            lambda: PublicObservationFilter(cls.context, cls.filter_cfg)
        )
        cls.cache = cache_public_beliefs(
            cls.dataset, cls.factory, seed=30_116,
            observation_scale=cls.context.observation_scale,
        )

    def test_hidden_schema_has_cost_identity_events_and_no_controls(self):
        dataset = self.dataset
        self.assertTrue(np.array_equal(dataset.dones, dataset.terminated | dataset.truncated))
        self.assertFalse(np.any(dataset.terminated & dataset.truncated))
        self.assertTrue(np.array_equal(dataset.costs, dataset.action_costs[dataset.actions]))
        self.assertEqual(len(np.unique(dataset.pop_ids)), 1)
        self.assertTrue(str(dataset.pop_ids[0]).startswith("pop_"))
        self.assertTrue(all(getattr(dataset, key) is None for key in PUBLIC_CONTROL_FIELDS))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "public.npz"
            save_public(path, dataset)
            assert_public_schema(path)

    def test_full_and_hidden_legacy_arrays_are_identical(self):
        hidden = self.env_cfg
        full = replace(hidden, expose_rk="full")
        hidden_data, _ = collect_dataset(make_env(hidden), 120, 12, seed=91)
        full_data, _ = collect_dataset(make_env(full), 120, 12, seed=91)
        for field in PUBLIC_FIELDS:
            self.assertTrue(np.array_equal(getattr(hidden_data, field), getattr(full_data, field)), field)
        self.assertTrue(all(getattr(hidden_data, key) is None for key in PUBLIC_CONTROL_FIELDS))
        self.assertTrue(any(getattr(full_data, key) is not None for key in PUBLIC_CONTROL_FIELDS))

    def test_hidden_environment_config_cannot_construct_policy_directly(self):
        with self.assertRaisesRegex(ValueError, "MethodContext"):
            METHODS["mopo"](self.env_cfg, ModelConfig(), PlannerConfig())

    def test_all_hidden_methods_fit_without_table_or_exact_native_access(self):
        cfg = BenchmarkConfig(
            seed=116,
            environment=self.env_cfg,
            filter=self.filter_cfg,
            model=ModelConfig(
                ensemble_size=2,
                native_state_bins=15,
                native_vi_iterations=30,
            ),
            planner=PlannerConfig(
                horizon=2,
                sequences=8,
                particles=4,
                bamcts_depth=2,
                bamcts_simulations=4,
                ogsrl_cost_horizon=2,
            ),
            training=TrainingConfig(enabled=False),
        )

        def forbidden(*_args, **_kwargs):
            raise AssertionError("private table/exact solver accessed")

        patches = (
            patch("real_ecology_benchmark.realdata.pops_for", side_effect=forbidden),
            patch("real_ecology_benchmark.realdata.effects_for", side_effect=forbidden),
            patch("real_ecology_benchmark.realdata.actions_for", side_effect=forbidden),
            patch("real_ecology_benchmark.actions.resolve_actions", side_effect=forbidden),
            patch("real_ecology_benchmark.methods.moor.resolve_actions", side_effect=forbidden),
            patch("real_ecology_benchmark.native_solver.NativeSolver.build", side_effect=forbidden),
        )
        for active in patches:
            active.start()
        try:
            for method in METHODS:
                if method in FAITHFUL_METHODS:
                    continue
                policy, _ = build_method(
                    method,
                    cfg,
                    self.dataset,
                    self.factory,
                    self.cache,
                    method_context=self.context,
                )
                policy.reset(44)
                if method == "moor_native":
                    belief = policy.hidden_filter_factory()().reset(
                        float(self.dataset.observations[0]), 55
                    )
                else:
                    belief = self.factory().reset(float(self.dataset.observations[0]), 55)
                action = policy.act(belief, float(self.dataset.observations[0]))
                self.assertIn(action, range(self.env_cfg.num_actions), method)
                self.assertFalse(hasattr(policy, "env_cfg"), method)
                assert_hidden_method_artifact(policy, root=method)
        finally:
            for active in reversed(patches):
                active.stop()

    def test_private_family_relabel_does_not_change_hidden_native_fit(self):
        relabeled_cfg = replace(self.env_cfg, kind="allee")
        self.assertNotEqual(relabeled_cfg.kind, self.env_cfg.kind)
        first_fit, first_solver = build_fitted_solver(
            self.dataset, self.context, "ricker", ModelConfig(native_state_bins=15),
            PlannerConfig(),
        )
        second_fit, second_solver = build_fitted_solver(
            self.dataset, self.context, "ricker", ModelConfig(native_state_bins=15),
            PlannerConfig(),
        )
        self.assertTrue(np.array_equal(first_fit.coefficients, second_fit.coefficients))
        self.assertTrue(np.array_equal(first_solver.transition, second_solver.transition))
        self.assertTrue(np.array_equal(first_solver.q_values, second_solver.q_values))
        self.assertEqual(PUBLIC_FORM_CANDIDATES, ("ricker", "allee", "theta", "regime"))
        model = ModelConfig(
            ensemble_size=2,
            native_state_bins=15,
            native_vi_iterations=30,
        )
        planner = PlannerConfig(
            horizon=2,
            sequences=8,
            particles=4,
            bamcts_depth=2,
            bamcts_simulations=4,
            ogsrl_cost_horizon=2,
        )
        for method in METHODS:
            if method in FAITHFUL_METHODS:
                continue
            actions = []
            diagnostics = []
            for environment in (self.env_cfg, relabeled_cfg):
                cfg = BenchmarkConfig(
                    seed=116,
                    environment=environment,
                    filter=self.filter_cfg,
                    model=model,
                    planner=planner,
                    training=TrainingConfig(enabled=False),
                )
                policy, _ = build_method(
                    method,
                    cfg,
                    self.dataset,
                    self.factory,
                    self.cache,
                    method_context=self.context,
                )
                policy.reset(44)
                belief = (
                    policy.hidden_filter_factory()().reset(
                        float(self.dataset.observations[0]), 55
                    )
                    if method == "moor_native"
                    else self.factory().reset(float(self.dataset.observations[0]), 55)
                )
                actions.append(policy.act(belief, float(self.dataset.observations[0])))
                diagnostics.append(policy.fit_diagnostics)
            self.assertEqual(actions[0], actions[1], method)
            self.assertEqual(diagnostics[0], diagnostics[1], method)

    def test_private_safety_diagnostics_cannot_change_surrogate(self):
        first = fit_public_surrogate(self.dataset, seed=20_116)
        changed_private = replace(
            self.private,
            safety_penalty_applied=~self.private.safety_penalty_applied,
        )
        before = first.reward_coefficients.copy()
        first_diag = evaluator_only_surrogate_diagnostics(first, self.dataset, self.private)
        changed_diag = evaluator_only_surrogate_diagnostics(first, self.dataset, changed_private)
        self.assertTrue(np.array_equal(before, first.reward_coefficients))
        self.assertNotEqual(first_diag, changed_diag)
        second = fit_public_surrogate(self.dataset, seed=20_116)
        self.assertTrue(np.array_equal(first.reward_coefficients, second.reward_coefficients))
        self.assertEqual(first.diagnostics["risk_fallback"], "constant_single_class")
        self.assertIn("risk_holdout_brier", first.diagnostics)
        self.assertIn("reward_holdout_low_tail_rmse", first.diagnostics)

    def test_serialized_hidden_caches_and_surrogate_have_no_private_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_path = root / "regime_hidden.beliefs.npz"
            surrogate_path = root / "regime_hidden.surrogate.npz"
            self.cache.save(cache_path)
            self.surrogate.save(surrogate_path)
            for label, path in (("cache", cache_path), ("surrogate", surrogate_path)):
                with np.load(path, allow_pickle=False) as payload:
                    self.assertFalse(
                        any(forbidden_method_name(name) for name in payload.files), label
                    )
                    artifact = {
                        "array_names": tuple(payload.files),
                        "metadata": json.loads(str(payload["metadata_json"].item())),
                    }
                assert_hidden_method_artifact(artifact, root=label)

    def test_episode_preserving_4000_target_bound(self):
        dataset, _ = collect_dataset(make_env(self.env_cfg), 4000, 25, seed=117)
        metadata = dataset.metadata
        self.assertEqual(metadata["target_rows"], 4000)
        self.assertGreaterEqual(metadata["actual_rows"], 4000)
        self.assertLessEqual(metadata["actual_rows"], 4024)
        self.assertEqual(
            metadata["overshoot_rows"], metadata["actual_rows"] - 4000
        )
        self.assertLess(metadata["overshoot_rows"], metadata["episode_length"])

    def test_real_manifest_defaults_hidden_and_labels_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            make_manifest(
                path,
                methods=("refplan", "moor_native"),
                data_mode="real",
                populations=("Amur tiger",),
                families=("ricker",),
                sigmas=(0.0,),
                reward_modes=("safe",),
                method_filters={
                    "refplan": ("learned",),
                    "moor_native": ("native_discrete",),
                },
            )
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual({row["expose_rk"] for row in rows}, {"hidden"})
        self.assertEqual({row["target_rows"] for row in rows}, {"4000"})
        self.assertTrue(all(row["actual_rows"] == "" for row in rows))
        self.assertNotIn("ricker", {row["filter"] for row in rows})


if __name__ == "__main__":
    unittest.main()
