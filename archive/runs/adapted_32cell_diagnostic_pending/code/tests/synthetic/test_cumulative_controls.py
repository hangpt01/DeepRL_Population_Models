import unittest

import numpy as np

from real_ecology_benchmark.actions import (
    CUMULATIVE_CONTROL_FIVE_ACTIONS,
    CUMULATIVE_CONTROL_TEN_ACTIONS,
    action_table,
    action_table_hash,
)
from real_ecology_benchmark.beliefs import LearnedLinearProposal, MechanisticProposal, ReferenceProposal
from real_ecology_benchmark.collector import calibration_summary, collect_dataset
from real_ecology_benchmark.config import environment_with_kind_defaults, synthetic_environment
from real_ecology_benchmark.controls import advance_public_controls
from real_ecology_benchmark.dataset import TrajectoryDataset, assert_public_schema, save_public
from real_ecology_benchmark.envs import make_env
from real_ecology_benchmark.rollout import AugmentedRolloutState


class OneStepSyntheticFreezeTests(unittest.TestCase):
    def test_one_step_synthetic_golden_trajectory(self):
        cfg = synthetic_environment(
            kind="allee", observation_noise_sigma=0.0, horizon=10, num_actions=5
        )
        env = make_env(cfg)
        reset = env.reset(7, state_override=120.0)
        self.assertEqual(action_table_hash(action_table(5)), "e7f0f72a0b22fc19")
        self.assertEqual(reset.public_info, {"timestep": 0})
        self.assertNotIn("rho", reset.evaluator_info)
        actions = [0, 1, 2, 3, 4, 0]
        expected_observations = np.asarray([
            127.12134004749828,
            59.40579951291802,
            126.88387792546285,
            298.7076099076742,
            691.5473340444486,
            361.5718265364622,
        ])
        expected_rewards = np.asarray([
            0.1935483870967742,
            0.6027061302647907,
            -0.09380553515061263,
            -0.09759588275685488,
            -0.22601131352410292,
            0.5803775597375065,
        ])
        observations = []
        rewards = []
        entries = []
        for action in actions:
            result = env.step(action)
            observations.append(result.observation)
            rewards.append(result.reward)
            entries.append(result.evaluator_info["entered_safety_region"])
            self.assertNotIn("rho", result.evaluator_info)
        np.testing.assert_allclose(observations, expected_observations, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(rewards, expected_rewards, rtol=0.0, atol=1e-12)
        self.assertFalse(any(entries))


class CumulativeControlTests(unittest.TestCase):
    def test_kind_defaults_are_control_mode_aware(self):
        one_step_ricker = environment_with_kind_defaults(synthetic_environment(), "ricker")
        one_step_theta = environment_with_kind_defaults(synthetic_environment(), "theta")
        self.assertEqual((one_step_ricker.r_base_low, one_step_ricker.r_base_high), (0.95, 1.00))
        self.assertEqual((one_step_theta.r_base_low, one_step_theta.r_base_high), (0.18, 0.40))

        base = synthetic_environment(control_mode="cumulative_capped")
        cumulative_ricker = environment_with_kind_defaults(base, "ricker")
        cumulative_theta = environment_with_kind_defaults(base, "theta")
        self.assertEqual((cumulative_ricker.r_base_low, cumulative_ricker.r_base_high), (0.12, 0.30))
        self.assertEqual((cumulative_theta.r_base_low, cumulative_theta.r_base_high), (0.12, 0.40))
        self.assertLessEqual(cumulative_ricker.r_base_high, cumulative_ricker.r_max)

    def test_cumulative_control_tables_remove_direct_stock_authority(self):
        for spec in (*CUMULATIVE_CONTROL_FIVE_ACTIONS, *CUMULATIVE_CONTROL_TEN_ACTIONS):
            self.assertEqual(spec.harvest_fraction, 0.0)
            self.assertEqual(spec.stocking_delta, 0.0)
        self.assertLess(CUMULATIVE_CONTROL_TEN_ACTIONS[1].cost, 0.0)
        self.assertGreater(CUMULATIVE_CONTROL_TEN_ACTIONS[6].delta_K, 0.0)

    def test_public_controls_advance_without_r_eff_leak(self):
        cfg = synthetic_environment(
            kind="allee",
            observation_noise_sigma=0.0,
            horizon=10,
            num_actions=5,
            control_mode="cumulative_capped",
        )
        env = make_env(cfg)
        reset = env.reset(7, state_override=120.0)
        self.assertEqual(reset.public_info["rho"], 0.0)
        self.assertEqual(reset.public_info["kappa"], 0.0)
        self.assertEqual(reset.public_info["K_eff"], 500.0)
        self.assertNotIn("r_eff", reset.public_info)
        observed = []
        for action in [3, 3, 1, 4]:
            result = env.step(action)
            self.assertNotIn("r_eff", result.public_info)
            self.assertIn("r_eff_true", result.evaluator_info)
            observed.append((
                result.public_info["rho"],
                result.public_info["kappa"],
                result.public_info["K_eff"],
            ))
        self.assertEqual(observed, [(0.0, 40.0, 540.0), (0.0, 80.0, 580.0),
                                    (-0.02, 80.0, 580.0), (0.0, 120.0, 620.0)])

    def test_negative_effective_rate_never_overflows(self):
        # Regression: sustained harvest drives r_eff to the r_min floor (negative).
        # A negative rate inside the density-dependent exponent used to flip sign
        # for s>K (and below C) and explode to a FloatingPointError during data
        # collection; it must stay finite and bounded, with the net-negative part
        # acting as exploitation mortality.  Once r_eff is clipped negative the
        # population must strictly decline.
        for kind in ("ricker", "allee", "regime", "theta"):
            cfg = synthetic_environment(
                kind=kind,
                observation_noise_sigma=0.0,
                horizon=60,
                num_actions=5,
                control_mode="cumulative_capped",
                r_base_low=0.12,
                r_base_high=0.12,  # low r_base so harvest reaches the negative floor fast
            )
            env = make_env(cfg)
            env.reset(3, state_override=560.0)  # just above K_base, like the real crash (s~527)
            saw_negative_decline = False
            previous = env.state
            for _ in range(45):
                result = env.step(1)  # aggressive harvest: delta_r < 0
                self.assertTrue(np.isfinite(env.state), f"{kind} produced non-finite state")
                self.assertLess(env.state, 100_000.0, f"{kind} exploded under harvest")
                # once r_eff is pinned negative the map is pure mortality: declining.
                if float(result.evaluator_info["r_eff_true"]) < 0.0 and env.state > 0.0:
                    self.assertLess(env.state, previous + 1e-9, f"{kind} grew at negative r_eff")
                    saw_negative_decline = True
                previous = env.state
                if result.done:
                    break
            self.assertTrue(saw_negative_decline, f"{kind} never exercised negative r_eff")

    def test_theta_cumulative_prior_has_base_stability_headroom(self):
        cfg = synthetic_environment(
            kind="theta",
            control_mode="cumulative_capped",
            observation_noise_sigma=0.0,
            num_actions=5,
        )
        for seed in range(50):
            reset = make_env(cfg).reset(seed)
            product = reset.evaluator_info["r_base"] * reset.evaluator_info["theta"]
            self.assertLessEqual(product, cfg.theta_base_stability_margin + 1e-12)

    def test_reference_proposal_conditions_on_public_controls(self):
        cfg = synthetic_environment(
            kind="allee",
            control_mode="cumulative_capped",
            observation_noise_sigma=0.0,
            num_actions=5,
        )
        proposal = ReferenceProposal(cfg, sigma=0.0)
        states = np.asarray([400.0, 400.0])
        contexts = np.zeros((2, 3), dtype=np.float64)
        regimes = np.zeros(2, dtype=np.int8)
        rng = np.random.default_rng(1)
        base, _ = proposal.sample_next(
            states, np.asarray([0, 0]), contexts, regimes, rng, rho=0.0, kappa=0.0
        )
        invested, _ = proposal.sample_next(
            states, np.asarray([0, 0]), contexts, regimes, rng, rho=0.10, kappa=200.0
        )
        harvest, _ = proposal.sample_next(
            states, np.asarray([1, 1]), contexts, regimes, rng, rho=0.0, kappa=0.0
        )
        self.assertTrue(np.all(invested > base))
        self.assertTrue(np.all(harvest < base))

    def test_learned_linear_proposal_uses_public_control_features(self):
        cfg = synthetic_environment(
            kind="allee",
            control_mode="cumulative_capped",
            observation_noise_sigma=0.0,
            num_actions=5,
        )
        n = 6
        observations = np.linspace(100.0, 160.0, n)
        next_rho = np.linspace(0.0, 0.10, n)
        next_kappa = np.linspace(0.0, 120.0, n)
        next_K_eff = cfg.K_base + next_kappa
        dataset = TrajectoryDataset(
            observations=observations,
            actions=np.zeros(n, dtype=np.int16),
            rewards=np.zeros(n, dtype=np.float64),
            next_observations=observations * np.exp(0.05 + next_rho),
            dones=np.asarray([False, False, False, False, False, True]),
            episode_id=np.zeros(n, dtype=np.int32),
            timestep=np.arange(n, dtype=np.int16),
            metadata={"environment": cfg.__dict__},
            rho=np.zeros(n, dtype=np.float64),
            kappa=np.zeros(n, dtype=np.float64),
            K_eff=np.full(n, cfg.K_base, dtype=np.float64),
            next_rho=next_rho,
            next_kappa=next_kappa,
            next_K_eff=next_K_eff,
        )
        proposal = LearnedLinearProposal.fit(dataset, cfg, ridge=1e-3)
        self.assertTrue(proposal.uses_controls)
        self.assertEqual(proposal.coefficients.shape[1], 8)
        proposal.residual_sigma[:] = 0.0
        sparse_base, _ = proposal.sample_next(
            np.asarray([125.0, 125.0]),
            np.asarray([1, 1]),
            np.zeros((2, 3), dtype=np.float64),
            np.zeros(2, dtype=np.int8),
            np.random.default_rng(2),
            rho=0.0,
            kappa=0.0,
        )
        sparse_invested, _ = proposal.sample_next(
            np.asarray([125.0, 125.0]),
            np.asarray([1, 1]),
            np.zeros((2, 3), dtype=np.float64),
            np.zeros(2, dtype=np.int8),
            np.random.default_rng(2),
            rho=0.10,
            kappa=200.0,
        )
        self.assertTrue(np.all(sparse_invested > sparse_base))

        coefficients = np.zeros((cfg.num_actions, 8), dtype=np.float64)
        coefficients[:, 1] = 1.0
        coefficients[:, 5] = 1.0
        coefficients[:, 6] = 0.5
        deterministic = LearnedLinearProposal(
            cfg, coefficients, np.zeros(cfg.num_actions, dtype=np.float64), True
        )
        states = np.asarray([125.0, 125.0])
        contexts = np.zeros((2, 3), dtype=np.float64)
        regimes = np.zeros(2, dtype=np.int8)
        rng = np.random.default_rng(2)
        base, _ = deterministic.sample_next(
            states, np.asarray([0, 0]), contexts, regimes, rng, rho=0.0, kappa=0.0
        )
        invested, _ = deterministic.sample_next(
            states, np.asarray([0, 0]), contexts, regimes, rng, rho=0.10, kappa=200.0
        )
        self.assertTrue(np.all(invested > base))

    def test_cumulative_learned_proposal_requires_public_controls(self):
        cfg = synthetic_environment(
            kind="allee",
            control_mode="cumulative_capped",
            observation_noise_sigma=0.0,
            num_actions=5,
        )
        n = 4
        dataset = TrajectoryDataset(
            observations=np.ones(n) * 100.0,
            actions=np.zeros(n, dtype=np.int16),
            rewards=np.zeros(n, dtype=np.float64),
            next_observations=np.ones(n) * 105.0,
            dones=np.asarray([False, False, False, True]),
            episode_id=np.zeros(n, dtype=np.int32),
            timestep=np.arange(n, dtype=np.int16),
            metadata={"environment": cfg.__dict__},
        )
        with self.assertRaises(ValueError):
            LearnedLinearProposal.fit(dataset, cfg)

    def test_public_dataset_controls_do_not_leak_private_r_eff(self):
        cfg = synthetic_environment(
            kind="allee",
            control_mode="cumulative_capped",
            observation_noise_sigma=0.1,
            horizon=8,
            num_actions=5,
        )
        dataset, private = collect_dataset(make_env(cfg), 80, 8, 123, True)
        self.assertIsNotNone(dataset.rho)
        self.assertIsNotNone(dataset.next_kappa)
        self.assertIsNotNone(private.r_eff_true)
        summary = calibration_summary(dataset, private, cfg.safety_threshold)
        self.assertIn("harvest_driven_delayed_collapse_count", summary)
        self.assertIn("harvest_driven_delayed_collapse_rate", summary)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "public.npz"
            save_public(path, dataset)
            assert_public_schema(path)

    def test_augmented_rollout_state_matches_env_constant_action(self):
        cfg = synthetic_environment(
            kind="allee",
            observation_noise_sigma=0.0,
            horizon=10,
            num_actions=5,
            control_mode="cumulative_capped",
        )
        env = make_env(cfg)
        reset = env.reset(11, state_override=120.0)
        hidden = reset.evaluator_info
        proposal = MechanisticProposal(cfg, cfg.kind)
        context = np.asarray([[0.0, 0.0, 0.0]])
        # Encode exact hidden params through the proposal overrides.
        proposal._env = env
        state = AugmentedRolloutState(
            states=np.asarray([hidden["state"]]),
            contexts=context,
            regimes=np.asarray([hidden["regime"]], dtype=np.int8),
            rho=np.asarray([0.0]),
            kappa=np.asarray([0.0]),
        )
        # MechanisticProposal maps context to prior parameters, so use env.transition_value
        # directly in a tiny exact proposal to test the shared rollout controls.
        class ExactProposal:
            def sample_next(self, states, action, contexts, regimes, rng, rho=None, kappa=None):
                out = []
                for s, a, rr, kk in zip(states, action, rho, kappa):
                    out.append(env.transition_value(
                        float(s), env.actions[int(a)], hidden["r_base"], hidden["C"],
                        hidden["theta"], int(hidden["regime"]), 0.0, float(rr), float(kk)
                    ))
                return np.asarray(out), regimes.copy()

        rng = np.random.default_rng(1)
        env_states = []
        rollout_states = []
        for _ in range(4):
            result = env.step(3)
            state = state.advance(cfg, ExactProposal(), np.asarray([3]), rng)
            env_states.append(result.evaluator_info["state"])
            rollout_states.append(float(state.states[0]))
            self.assertAlmostEqual(float(state.rho[0]), result.public_info["rho"])
            self.assertAlmostEqual(float(state.kappa[0]), result.public_info["kappa"])
        np.testing.assert_allclose(rollout_states, env_states, rtol=0.0, atol=1e-12)


class CumulativeControlEconomicsSmokeTests(unittest.TestCase):
    def _return_for_fixed_investment(self, invest_steps: int, action: int = 5) -> float:
        cfg = synthetic_environment(
            kind="ricker",
            num_actions=10,
            control_mode="cumulative_capped",
            observation_noise_sigma=0.0,
            horizon=50,
            r_base_low=0.30,
            r_base_high=0.30,
            low_start_probability=0.0,
            initial_log_sigma=0.0,
        )
        env = make_env(cfg)
        env.reset(1, state_override=400.0)
        total = 0.0
        gamma = 1.0
        for step in range(50):
            result = env.step(action if step < invest_steps else 0)
            total += gamma * result.reward
            gamma *= 0.95
        return total

    def test_restoration_has_interior_optimum(self):
        grid = [0, 1, 2, 4, 8, 12, 16, 20, 25, 30]
        returns = [self._return_for_fixed_investment(n, 5) for n in grid]
        best = int(np.argmax(returns))
        self.assertGreater(best, 0)
        self.assertLess(best, len(grid) - 1)
        self.assertGreater(max(returns), returns[0])
        self.assertGreater(max(returns), returns[-1])

    def test_no_investment_does_not_dominate_conservation(self):
        no_invest = self._return_for_fixed_investment(0, 5)
        moderate = max(self._return_for_fixed_investment(n, 5) for n in [4, 8, 12, 16])
        intensive = max(self._return_for_fixed_investment(n, 6) for n in [2, 4, 8, 12])
        self.assertGreater(moderate, no_invest)
        self.assertGreater(intensive, no_invest)


if __name__ == "__main__":
    unittest.main()
