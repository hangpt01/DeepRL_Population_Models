import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from real_ecology_benchmark import realdata
from real_ecology_benchmark.actions import resolve_actions
from real_ecology_benchmark.collector import collect_dataset
from real_ecology_benchmark.config import (
    SETPOINT_CUMULATIVE,
    BenchmarkConfig,
    dummy_environment,
    load_config,
)
from real_ecology_benchmark.controls import private_r_eff
from real_ecology_benchmark.envs import make_env
from real_ecology_benchmark.pipeline import _validate_dataset_cell


class DummySetpointTests(unittest.TestCase):
    def test_dummy_config_uses_setpoint_semantics(self):
        cfg = load_config("configs/dummy_setpoint_smoke.yaml")
        env_cfg = cfg.environment
        self.assertEqual(env_cfg.data_mode, "dummy")
        self.assertEqual(env_cfg.control_mode, SETPOINT_CUMULATIVE)
        self.assertEqual(env_cfg.accumulator_decay_r, 1.0)
        self.assertEqual(env_cfg.accumulator_decay_K, 0.0)
        self.assertEqual(env_cfg.r_base_low, 0.0)
        self.assertEqual(env_cfg.r_base_high, 0.0)

    def test_legacy_real_setpoint_alias_normalizes(self):
        cfg = dummy_environment(control_mode="real_setpoint")
        self.assertEqual(cfg.control_mode, SETPOINT_CUMULATIVE)

    def test_legacy_cache_metadata_normalizes_against_canonical_config(self):
        cfg = dummy_environment(
            control_mode="real_setpoint",
            observation_noise_sigma=0.0,
            horizon=4,
        )
        dataset, _private = collect_dataset(make_env(cfg), 6, 3, 123, True)
        self.assertEqual(
            dataset.metadata["environment"]["control_mode"],
            SETPOINT_CUMULATIVE,
        )
        legacy_env = dict(dataset.metadata["environment"])
        legacy_env["control_mode"] = "real_setpoint"
        legacy_dataset = SimpleNamespace(metadata={"environment": legacy_env})
        _validate_dataset_cell(BenchmarkConfig(environment=cfg), legacy_dataset)

    def test_dummy_rate_setpoint_does_not_accumulate(self):
        cfg = dummy_environment(observation_noise_sigma=0.0)
        env = make_env(cfg)
        actions = resolve_actions(cfg)
        expected = actions[2].delta_r
        env.reset(seed=1, state_override=cfg.K_base)
        first = env.step(2)
        second = env.step(2)
        self.assertAlmostEqual(first.evaluator_info["rho"], expected, delta=1e-12)
        self.assertAlmostEqual(second.evaluator_info["rho"], expected, delta=1e-12)
        self.assertAlmostEqual(second.evaluator_info["r_eff_true"], expected, delta=1e-12)
        self.assertNotAlmostEqual(second.evaluator_info["rho"], 2.0 * expected, delta=1e-6)

    def test_dummy_capacity_accumulates_to_cap(self):
        cfg = dummy_environment(observation_noise_sigma=0.0)
        env = make_env(cfg)
        actions = resolve_actions(cfg)
        delta_K = actions[4].delta_K
        env.reset(seed=1, state_override=cfg.K_base)
        last = cfg.K_base
        for step in range(1, 10):
            result = env.step(4)
            expected = min(cfg.K_base + step * delta_K, cfg.K_max)
            self.assertGreaterEqual(result.evaluator_info["K_eff"], last - 1e-9)
            self.assertAlmostEqual(result.evaluator_info["K_eff"], expected, delta=1e-9)
            last = result.evaluator_info["K_eff"]
        self.assertAlmostEqual(last, cfg.K_max, delta=1e-9)

    def test_dummy_out_of_cap_rate_raises(self):
        cfg = dummy_environment()
        with self.assertRaises(ValueError):
            private_r_eff(cfg, 0.0, cfg.r_max + 1.0, None)

    def test_dummy_env_path_does_not_call_real_loaders(self):
        with mock.patch.object(realdata, "actions_for", side_effect=AssertionError):
            with mock.patch.object(realdata, "effects_for", side_effect=AssertionError):
                with mock.patch.object(realdata, "pops_for", side_effect=AssertionError):
                    cfg = load_config("configs/dummy_setpoint_smoke.yaml")
                    env = make_env(cfg.environment)
                    env.reset(seed=1, state_override=cfg.environment.K_base)
                    env.step(2)

    def test_realdata_dir_points_to_promoted_top_level_data(self):
        self.assertTrue((realdata.DATA_DIR / "species.csv").exists())
        self.assertEqual(realdata.DATA_DIR.name, "real_ecology_data")

    def test_cache_validation_rejects_cross_data_mode(self):
        real_env = load_config("configs/real_smoke.yaml").environment
        dummy_env = load_config("configs/dummy_setpoint_smoke.yaml").environment
        dataset = SimpleNamespace(metadata={"environment": dict(real_env.__dict__)})
        with self.assertRaises(ValueError):
            _validate_dataset_cell(BenchmarkConfig(environment=dummy_env), dataset)


if __name__ == "__main__":
    unittest.main()
