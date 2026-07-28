from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from real_ecology_benchmark.beliefs import DiscreteGridFilter, ParticleFilter, ReferenceProposal
from real_ecology_benchmark.config import (
    FilterConfig,
    ModelConfig,
    PlannerConfig,
    real_environment as _real_environment,
)
from real_ecology_benchmark.dataset import TrajectoryDataset
from real_ecology_benchmark.discretize import build_native_grid
from real_ecology_benchmark.manifest import aggregate_summaries, make_motivation_native_manifest
from real_ecology_benchmark.methods.moor_native import MOORNativePolicy
from real_ecology_benchmark.methods.plus_native import PLUSNativePolicy
from real_ecology_benchmark.native_solver import NativeSolver, predict_ricker_next
from real_ecology_benchmark.types import PublicTransition


def real_environment(*args, **kwargs):
    """Native baseline fixtures exercise the retained parameter-known arm."""

    kwargs["expose_rk"] = "full"
    return _real_environment(*args, **kwargs)


def _tiny_public_dataset(cfg) -> TrajectoryDataset:
    observations = np.asarray([cfg.N0, cfg.N0 * 0.9, cfg.N0 * 0.8, cfg.N0 * 0.7])
    actions = np.asarray([0, 3, 4, 5], dtype=np.int16)
    kappa = np.zeros(len(actions), dtype=np.float64)
    next_observations = predict_ricker_next(cfg, observations, actions, kappa)
    dataset = TrajectoryDataset(
        observations=observations,
        actions=actions,
        rewards=np.zeros(len(actions), dtype=np.float64),
        next_observations=next_observations,
        dones=np.ones(len(actions), dtype=bool),
        episode_id=np.arange(len(actions), dtype=np.int32),
        timestep=np.zeros(len(actions), dtype=np.int16),
        metadata={"environment": cfg.__dict__},
        kappa=kappa,
    )
    dataset.validate()
    return dataset


def _write_episode(path: Path, model: str, filter_name: str, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "compute_backend_effective": "numpy",
        "expose_rk": "full",
        "reward_mode": "safe",
        "population": "Amur tiger",
        "environment": "ricker",
        "num_actions": "11",
        "sigma_obs": "0.0",
        "filter": filter_name,
        "block_seed": "7001",
        "model": model,
        "operational_return": str(value),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


class NativeBaselineTests(unittest.TestCase):
    def test_native_k_lattice_is_population_normalized_for_non_250_population(self):
        cfg = real_environment("Bottlenose dolphin", "ricker")
        grid = build_native_grid(cfg, state_bins=21)
        self.assertTrue(np.allclose(grid.kappa_normalized, np.linspace(0.0, 1.0, 11)))
        self.assertTrue(np.isclose(grid.kappa_values[1], 3.5))
        self.assertFalse(np.isclose(grid.kappa_values[1], 25.0))

    def test_sigma_zero_emission_is_deterministic_not_all_zero(self):
        cfg = real_environment("Amur tiger", "ricker", observation_noise_sigma=0.0)
        grid = build_native_grid(cfg, state_bins=21)
        ll = grid.log_likelihood(187.3)
        self.assertEqual(np.sum(np.isfinite(ll)), 1)
        weights = np.exp(grid.initial_log_weights(187.3))
        self.assertTrue(np.isclose(weights.sum(), 1.0))
        self.assertEqual(weights.max(), 1.0)

    def test_native_transition_interpolates_between_state_bins(self):
        cfg = real_environment("Amur tiger", "ricker", observation_noise_sigma=0.0)
        solver = NativeSolver.build(cfg, state_bins=51, iterations=5)
        state_idx = int(np.argmin(np.abs(solver.grid.state_values - 100.0)))
        row = solver.transition[0, 0, state_idx]
        self.assertGreater(np.count_nonzero(row), 1)
        self.assertAlmostEqual(float(row.sum()), 1.0)
        self.assertLess(float(row[state_idx]), 1.0)

    def test_native_policies_act_with_finite_discrete_beliefs(self):
        cfg = real_environment(
            "Amur tiger",
            "ricker",
            observation_noise_sigma=0.0,
            collapse_penalty=5.0,
        )
        model = ModelConfig(
            native_state_bins=21,
            native_fit_grid=5,
            native_vi_iterations=20,
        )
        planner = PlannerConfig(discount=0.95)
        dataset = _tiny_public_dataset(cfg)
        solver = NativeSolver.build(
            cfg,
            state_bins=model.native_state_bins,
            discount=planner.discount,
            iterations=model.native_vi_iterations,
        )
        belief = DiscreteGridFilter(cfg, solver).reset(cfg.N0, 123)

        moor = MOORNativePolicy(cfg, model, planner, seed=1)
        moor.fit(dataset)
        action = moor.act(belief, cfg.N0)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, cfg.num_actions)
        self.assertTrue(np.all(np.isfinite(moor.last_diagnostics["action_scores"])))

        plus = PLUSNativePolicy(cfg, model, planner, seed=2)
        plus.fit(dataset)
        action = plus.act(belief, cfg.N0)
        plus.observe(
            belief,
            action,
            PublicTransition(observation=cfg.N0, done=False, truncated=False, public_info={}),
        )
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, cfg.num_actions)
        self.assertTrue(np.isclose(plus.posterior.sum(), 1.0))
        self.assertTrue(np.all(np.isfinite(plus.posterior)))

    def test_native_policy_rejects_non_native_belief(self):
        cfg = real_environment("Amur tiger", "ricker", observation_noise_sigma=0.0)
        model = ModelConfig(native_state_bins=21, native_vi_iterations=20)
        planner = PlannerConfig(discount=0.95)
        dataset = _tiny_public_dataset(cfg)
        belief = ParticleFilter(
            cfg,
            FilterConfig(particles=24),
            ReferenceProposal(cfg),
        ).reset(cfg.N0, 123)

        policy = MOORNativePolicy(cfg, model, planner, seed=1)
        policy.fit(dataset)
        with self.assertRaisesRegex(ValueError, "native_discrete"):
            policy.act(belief, cfg.N0)

    def test_motivation_manifest_routes_native_methods_to_native_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            rows = make_motivation_native_manifest(path)
            self.assertEqual(rows, 9 * 4 * 4 * 2 * 5)
            with path.open("r", encoding="utf-8") as handle:
                by_method: dict[str, set[str]] = {}
                for row in csv.DictReader(handle):
                    by_method.setdefault(row["method"], set()).add(row["filter"])
        self.assertEqual(by_method["refplan"], {"learned"})
        self.assertEqual(by_method["bamcts"], {"learned"})
        self.assertEqual(by_method["ogsrl"], {"learned"})
        self.assertEqual(by_method["plus_native"], {"native_discrete"})
        self.assertEqual(by_method["moor_native"], {"native_discrete"})

    def test_aggregate_summaries_pairs_learned_challenger_against_native_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            _write_episode(root / "refplan" / "learned" / "episodes.csv", "refplan", "learned", 10.0)
            _write_episode(
                root / "plus_native" / "native_discrete" / "episodes.csv",
                "plus_native",
                "native_discrete",
                6.0,
            )
            _write_episode(
                root / "moor_native" / "native_discrete" / "episodes.csv",
                "moor_native",
                "native_discrete",
                7.0,
            )
            output = Path(tmp) / "aggregate.json"
            result = aggregate_summaries(
                root,
                output,
                challenger_filter="learned",
                baselines=(("plus_native", "native_discrete"), ("moor_native", "native_discrete")),
            )
            self.assertTrue(output.exists())
            self.assertEqual(len(result["beats_both_cells"]), 1)
            self.assertEqual(result["beats_both_cells"][0]["paired_mean_difference"], 3.0)
            self.assertEqual(result["beats_both_rate"][0]["rate"], 1.0)
            with output.open("r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["beats_both_cells"][0]["method"], "refplan")


class NativeCandidateBankTests(unittest.TestCase):
    """PLUS-native's hypothesis class is the four mechanistic forms, not Ricker K.

    A bank over Ricker K carries no signal here: 6 of the 11 actions have
    ``r_setpoint <= 0``, which zeroes ``r_pos`` and cancels K out of the exponent,
    leaving the posterior flat and collapsing PLUS onto MOOR.
    """

    def _policy(self, cfg):
        policy = PLUSNativePolicy(cfg, ModelConfig(native_state_bins=21), PlannerConfig(), seed=0)
        policy.fit(_tiny_public_dataset(cfg), None)
        return policy

    def test_plus_native_bank_spans_the_four_mechanistic_forms(self):
        cfg = real_environment("Amur tiger", "ricker")
        policy = self._policy(cfg)
        self.assertEqual(policy.candidates, ["ricker", "allee", "theta", "regime"])
        self.assertEqual([s.family for s in policy.solvers], policy.candidates)

    def test_regime_candidate_carries_a_hidden_regime_the_others_do_not(self):
        cfg = real_environment("Amur tiger", "ricker")
        for solver in self._policy(cfg).solvers:
            expected = 2 if solver.family == "regime" else 1
            self.assertEqual(solver.grid.num_regimes, expected)
            self.assertEqual(
                solver.grid.num_hidden, solver.grid.num_states * expected
            )
            # regime is Markov with the cell's persistence; the others are inert.
            kernel = solver.grid.regime_kernel()
            np.testing.assert_allclose(kernel.sum(axis=1), 1.0)
            if solver.family == "regime":
                self.assertAlmostEqual(kernel[0, 0], cfg.regime_persistence)

    def test_candidate_forms_predict_different_dynamics(self):
        # The forms differ only through the density-dependent growth term, which is
        # gated by r_pos = max(r_eff, 0) -- so they coincide exactly on every action
        # whose set-point rate is <= 0.  Scan for the actions where the form bites.
        cfg = real_environment("Iberian lynx", "allee")
        solvers = self._policy(cfg).solvers
        spread = 0.0
        for action in range(cfg.num_actions):
            for state in (0.5 * cfg.N0, cfg.N0, cfg.K_base):
                predictions = [
                    float(
                        solver.transition[0, action, solver.grid.state_index(float(state))]
                        @ solver.grid.hidden_state_values
                    )
                    for solver in solvers
                ]
                spread = max(spread, max(predictions) - min(predictions))
        self.assertGreater(spread, 1e-6)

    def test_forms_disagree_on_the_optimal_action_somewhere(self):
        # If the greedy policy were invariant to the assumed form, a model-averaging
        # baseline could not differ from a single-model one for any posterior.
        cfg = real_environment("Iberian lynx", "allee")
        solvers = self._policy(cfg).solvers
        num_states = solvers[0].grid.num_states
        greedy = np.stack(
            [s.q_values[:, :num_states, :].argmax(axis=2) for s in solvers]
        )
        self.assertTrue(bool(np.any(greedy != greedy[0])))

    def test_plus_posterior_is_a_proper_distribution_over_forms(self):
        cfg = real_environment("Iberian lynx", "allee", observation_noise_sigma=0.2)
        policy = self._policy(cfg)
        solver = NativeSolver.build(cfg, state_bins=21)
        belief = DiscreteGridFilter(cfg, solver).reset(float(cfg.N0), 0)
        policy.act(belief, float(cfg.N0))
        for step in range(4):
            observation = float(cfg.N0) * (0.95 ** (step + 1))
            policy.observe(
                belief,
                0,
                PublicTransition(observation=observation, done=False, truncated=False),
            )
            self.assertEqual(len(policy.posterior), 4)
            self.assertTrue(np.all(np.isfinite(policy.posterior)))
            self.assertTrue(np.all(policy.posterior >= 0.0))
            self.assertAlmostEqual(float(policy.posterior.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
