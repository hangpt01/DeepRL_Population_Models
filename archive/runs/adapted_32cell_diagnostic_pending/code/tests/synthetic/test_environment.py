import math
import unittest

import numpy as np

from real_ecology_benchmark.actions import FIVE_ACTIONS, TEN_ACTIONS, action_table
from real_ecology_benchmark.config import synthetic_environment
from real_ecology_benchmark.envs import make_env
from real_ecology_benchmark.observation import LogNormalObservationModel


class ObservationTests(unittest.TestCase):
    def test_exact_and_zero_atoms(self):
        model = LogNormalObservationModel(0.0)
        states = np.asarray([0.0, 10.0, 11.0])
        self.assertEqual(model.sample(10.0, np.random.default_rng(1)), 10.0)
        ll = model.log_prob(10.0, states)
        self.assertTrue(np.isneginf(ll[0]))
        self.assertEqual(ll[1], 0.0)
        self.assertTrue(np.isneginf(ll[2]))
        zero = model.log_prob(0.0, states)
        self.assertEqual(zero[0], 0.0)
        self.assertTrue(np.all(np.isneginf(zero[1:])))

    def test_lognormal_median(self):
        model = LogNormalObservationModel(0.2)
        draws = model.sample(np.full(20000, 100.0), np.random.default_rng(3))
        self.assertAlmostEqual(float(np.median(draws)), 100.0, delta=2.0)


class EnvironmentTests(unittest.TestCase):
    def test_locked_action_tables(self):
        self.assertEqual(len(FIVE_ACTIONS), 5)
        self.assertEqual(len(TEN_ACTIONS), 10)
        self.assertEqual(FIVE_ACTIONS[1].harvest_fraction, 0.5)
        self.assertEqual(FIVE_ACTIONS[4].stocking_delta, 160.0)
        self.assertEqual(TEN_ACTIONS[9].delta_K, 200.0)

    def test_ricker_authority_before_growth(self):
        cfg = synthetic_environment(kind="ricker", observation_noise_sigma=0.0)
        env = make_env(cfg)
        env.reset(1, state_override=200.0)
        r = env._r_base
        action = action_table(5)[1]
        expected_managed = 100.0
        expected = expected_managed * math.exp((r - 0.02) * (1 - expected_managed / 500.0))
        result = env.step(1)
        self.assertAlmostEqual(result.evaluator_info["state"], expected, places=10)

    def test_all_dynamics_finite_and_nonnegative(self):
        for kind in ("ricker", "allee", "theta", "regime"):
            env = make_env(synthetic_environment(kind=kind, observation_noise_sigma=0.1))
            reset = env.reset(10)
            self.assertGreaterEqual(reset.observation, 0.0)
            for action in (0, 1, 2):
                if env._done:
                    break
                result = env.step(action)
                self.assertTrue(np.isfinite(result.evaluator_info["state"]))
                self.assertGreaterEqual(result.evaluator_info["state"], 0.0)

    def test_exact_extinction_is_terminal(self):
        cfg = synthetic_environment(kind="theta", observation_noise_sigma=0.0)
        env = make_env(cfg)
        reset = env.reset(2, state_override=0.0)
        self.assertTrue(reset.done)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_no_discrete_geometry(self):
        env = make_env(synthetic_environment())
        self.assertFalse(hasattr(env, "num_states"))
        self.assertFalse(hasattr(env, "s_max"))


if __name__ == "__main__":
    unittest.main()
