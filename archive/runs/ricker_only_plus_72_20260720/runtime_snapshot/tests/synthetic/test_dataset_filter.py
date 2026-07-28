import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from real_ecology_benchmark.beliefs import ParticleFilter, ReferenceProposal
from real_ecology_benchmark.collector import calibration_summary
from real_ecology_benchmark.config import FilterConfig, synthetic_environment
from real_ecology_benchmark.dataset import (
    PrivateTrajectoryData,
    TrajectoryDataset,
    assert_public_schema,
    load_private,
    load_public,
    save_private,
    save_public,
)

from .common import tiny_config, tiny_data


class DatasetTests(unittest.TestCase):
    def test_public_private_round_trip_and_no_leak(self):
        cfg, dataset, private, _factory, _cache = tiny_data()
        with tempfile.TemporaryDirectory() as tmp:
            public_path = Path(tmp) / "public.npz"
            private_path = Path(tmp) / "private.npz"
            save_public(public_path, dataset)
            save_private(private_path, private)
            assert_public_schema(public_path)
            loaded = load_public(public_path)
            truth = load_private(private_path)
            self.assertEqual(len(loaded), len(dataset))
            self.assertEqual(len(truth.states), len(dataset))
            self.assertNotIn("states", loaded.__dict__)

    def test_complete_episode_boundaries(self):
        _cfg, dataset, _private, _factory, _cache = tiny_data()
        dataset.validate()
        for idx in dataset.episode_indices():
            self.assertTrue(dataset.dones[idx[-1]])
            np.testing.assert_array_equal(dataset.timestep[idx], np.arange(len(idx)))

    def test_calibration_band_is_an_enforced_boolean(self):
        n = 20
        entries = np.zeros(n, dtype=bool)
        entries[:4] = True
        public = TrajectoryDataset(
            observations=np.full(n, 100.0),
            actions=np.zeros(n, dtype=np.int16),
            rewards=np.zeros(n),
            next_observations=np.full(n, 100.0),
            dones=np.ones(n, dtype=bool),
            episode_id=np.arange(n, dtype=np.int32),
            timestep=np.zeros(n, dtype=np.int16),
            metadata={},
        )
        private = PrivateTrajectoryData(
            states=np.full(n, 100.0),
            next_states=np.where(entries, 40.0, 100.0),
            r_base=np.full(n, 0.2),
            C=np.full(n, 100.0),
            theta=np.full(n, 4.0),
            regime=np.zeros(n, dtype=np.int8),
            next_regime=np.zeros(n, dtype=np.int8),
            entry=entries,
            reward_true=np.zeros(n),
            initially_unsafe=np.zeros(n, dtype=bool),
            metadata={},
        )
        summary = calibration_summary(public, private, 50.0)
        self.assertAlmostEqual(summary["incident_collapse_rate_healthy_starts"], 0.2)
        self.assertTrue(summary["collapse_band_pass"])


class FilterTests(unittest.TestCase):
    def test_sigma_zero_is_exact(self):
        env_cfg = synthetic_environment(observation_noise_sigma=0.0)
        filter_cfg = FilterConfig(particles=32, proposal="reference")
        filt = ParticleFilter(env_cfg, filter_cfg, ReferenceProposal(env_cfg, 0.1))
        belief = filt.reset(123.4, 9)
        self.assertAlmostEqual(belief.mean_state(), 123.4, places=12)
        updated = filt.update(belief, 0, 111.2)
        self.assertAlmostEqual(updated.mean_state(), 111.2, places=12)

    def test_features_and_cache_are_finite(self):
        _cfg, _dataset, _private, _factory, cache = tiny_data()
        self.assertEqual(cache.features.shape[1], 12)
        self.assertTrue(np.all(np.isfinite(cache.features)))
        self.assertTrue(np.all(cache.mean_states >= 0))

    def test_filter_rmse_increases_over_the_noise_sweep(self):
        # Use a default-profile family (allee_10a) at a signal-dominated scale:
        # at very small samples / few particles the weak reference proposal makes
        # RMSE noise-dominated rather than observation-noise-dominated.
        rmse = []
        for sigma in (0.0, 0.2, 0.4):
            base = tiny_config(kind="allee", sigma=sigma, actions=10)
            cfg = replace(
                base,
                dataset=replace(base.dataset, transitions=1500, episode_length=25),
                filter=replace(base.filter, particles=96),
            )
            _cfg, _dataset, private, _factory, cache = tiny_data(cfg)
            rmse.append(float(np.sqrt(np.mean((cache.mean_states - private.states) ** 2))))
        self.assertLess(rmse[0], 1e-10)
        self.assertTrue(all(right > left for left, right in zip(rmse, rmse[1:])))


if __name__ == "__main__":
    unittest.main()
