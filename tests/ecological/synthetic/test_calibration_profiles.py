import unittest
from dataclasses import replace

import numpy as np

from real_ecology_benchmark.actions import FIVE_ACTIONS, TEN_ACTIONS
from real_ecology_benchmark.collector import (
    DEFAULT_PROFILE,
    PROFILES,
    CollectorProfile,
    collect_dataset,
    collector_profile_for,
)
from real_ecology_benchmark.config import environment_with_kind_defaults, synthetic_environment
from real_ecology_benchmark.envs import make_env


def _collect(kind, actions, sigma, seed=116, n=4000, profile=None):
    cfg = environment_with_kind_defaults(synthetic_environment(), kind)
    cfg = replace(cfg, num_actions=actions, observation_noise_sigma=sigma)
    return collect_dataset(make_env(cfg), n, 25, seed, True, profile)


class ProfileSerializationTests(unittest.TestCase):
    def test_round_trip(self):
        for profile in [DEFAULT_PROFILE, *PROFILES.values()]:
            restored = CollectorProfile.from_dict(profile.to_dict())
            self.assertEqual(restored, profile)
            self.assertIsInstance(profile.to_dict()["component_probabilities"], list)

    def test_validation_rejects_bad_profiles(self):
        with self.assertRaises(ValueError):
            CollectorProfile("x", (0.5, 0.5, 0.5, 0.5)).validate()  # sums to 2
        with self.assertRaises(ValueError):
            CollectorProfile("x", (0.25, 0.25, 0.25, 0.25), harvest_probe_prob=1.4).validate()
        with self.assertRaises(ValueError):
            CollectorProfile("x", (0.5, 0.5)).validate()  # wrong length


class LockedSettingsTests(unittest.TestCase):
    def test_environment_defaults_unchanged(self):
        cfg = synthetic_environment()
        self.assertEqual(cfg.safety_threshold, 50.0)
        self.assertEqual(cfg.initial_log_sigma, 0.7)
        self.assertEqual(cfg.low_start_probability, 0.2)
        self.assertEqual(cfg.collapse_penalty, 20.0)
        self.assertEqual(cfg.K_base, 500.0)
        self.assertEqual(cfg.K_ref, 500.0)
        self.assertEqual(cfg.alpha, 1.0)
        self.assertTrue(cfg.privileged_behavior)

    def test_hidden_priors_unchanged(self):
        allee = environment_with_kind_defaults(synthetic_environment(), "allee")
        regime = environment_with_kind_defaults(synthetic_environment(), "regime")
        theta = environment_with_kind_defaults(synthetic_environment(), "theta")
        self.assertEqual((allee.r_base_low, allee.r_base_high), (0.12, 0.30))
        self.assertEqual((regime.r_base_low, regime.r_base_high), (0.12, 0.30))
        self.assertEqual((theta.r_base_low, theta.r_base_high), (0.18, 0.40))
        self.assertEqual((allee.C_low, allee.C_high), (90.0, 150.0))
        self.assertEqual((theta.theta_low, theta.theta_high), (3.0, 6.0))
        self.assertEqual(allee.regime_persistence, 0.90)

    def test_action_tables_unchanged(self):
        self.assertEqual(FIVE_ACTIONS[1].harvest_fraction, 0.50)
        self.assertEqual(FIVE_ACTIONS[4].stocking_delta, 160.0)
        self.assertEqual(TEN_ACTIONS[2].harvest_fraction, 0.50)
        self.assertEqual(TEN_ACTIONS[9].delta_K, 200.0)

    def test_default_profile_is_original_mixture(self):
        # The default profile must reproduce the originally deployed behavior.
        self.assertEqual(DEFAULT_PROFILE.component_probabilities, (0.25, 0.30, 0.25, 0.20))
        self.assertEqual(DEFAULT_PROFILE.harvest_probe_prob, 0.85)
        self.assertEqual(DEFAULT_PROFILE.rescue_harvest_prob, 0.45)
        self.assertEqual(DEFAULT_PROFILE.rescue_donothing_prob, 0.60)

    def test_successful_families_keep_default_profile(self):
        # allee_10a and regime_10a completed already; they must stay on default.
        self.assertIs(collector_profile_for("allee", 10), DEFAULT_PROFILE)
        self.assertIs(collector_profile_for("regime", 10), DEFAULT_PROFILE)
        self.assertIs(collector_profile_for("ricker", 5), DEFAULT_PROFILE)
        # The four failed families have explicit overrides.
        for key in (("allee", 5), ("regime", 5), ("theta", 5), ("theta", 10)):
            self.assertIn(key, PROFILES)
            self.assertIsNot(collector_profile_for(*key), DEFAULT_PROFILE)


class CollectionInvariantTests(unittest.TestCase):
    def test_metadata_records_full_profile(self):
        dataset, _ = _collect("theta", 5, 0.2)
        recorded = dataset.metadata["collector_profile"]
        self.assertEqual(recorded, PROFILES[("theta", 5)].to_dict())

    def test_deterministic_collection(self):
        a_pub, a_pri = _collect("allee", 5, 0.2)
        b_pub, b_pri = _collect("allee", 5, 0.2)
        np.testing.assert_array_equal(a_pub.observations, b_pub.observations)
        np.testing.assert_array_equal(a_pub.actions, b_pub.actions)
        np.testing.assert_array_equal(a_pri.entry, b_pri.entry)

    def test_latent_collapse_is_sigma_independent(self):
        # Privileged behavior acts on true state, so latent collapse must be
        # identical across observation-noise levels for a fixed seed+profile.
        _, p0 = _collect("theta", 10, 0.0)
        _, p4 = _collect("theta", 10, 0.4)
        np.testing.assert_array_equal(p0.entry, p4.entry)
        np.testing.assert_array_equal(p0.states, p4.states)

    def test_overrides_keep_full_action_coverage(self):
        for kind, actions in (("allee", 5), ("regime", 5), ("theta", 5), ("theta", 10)):
            dataset, _ = _collect(kind, actions, 0.2, n=6000)
            present = set(int(a) for a in np.unique(dataset.actions))
            self.assertEqual(present, set(range(actions)), f"{kind}_{actions}a missing actions")


if __name__ == "__main__":
    unittest.main()
