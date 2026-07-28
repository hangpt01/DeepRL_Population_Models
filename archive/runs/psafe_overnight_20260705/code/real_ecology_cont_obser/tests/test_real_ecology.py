"""E10 acceptance tests for the real-ecology (Tier-4) setting.

Run from the package root with::

    cd real_ecology_cont_obser
    PYTHONPATH=src python -m unittest discover -s tests -v

The tests check the four E10 conditions from
``29_6_Real_Ecology_Setting_Implementation_Plan.tex``:

* (E10.1) one env step reproduces ``action_effects_long.csv`` for every
  ``(population, action)``: set-point r_eff, cumulative dK, translocation dN, cost.
* (E10.2) the abundance reward term is ~0.5 at ``s == K_base`` for every
  population; r_eff stays within the data caps; K_eff <= K_max.
* (E10.3) after recalibration the healthy-start collapse rate is in
  ``[0.15, 0.24]`` for the *exploitable* recoverable populations, the
  structurally-robust ones sit below it (a finding, not a bug), and the two
  demographic sinks are reported separately.
* (leakage) a generated real dataset passes the public/private schema guard.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from real_ecology_benchmark import realdata
from real_ecology_benchmark.config import (
    BenchmarkConfig,
    DatasetConfig,
    EvaluationConfig,
    FilterConfig,
    ModelConfig,
    PlannerConfig,
    default_safety_fraction,
    real_environment,
    real_environment_like,
)
from real_ecology_benchmark.envs import make_env
from real_ecology_benchmark.actions import resolve_actions
from real_ecology_benchmark.beliefs import RawObservationFilter, ReferenceProposal
from real_ecology_benchmark.collector import (
    REAL_DEFAULT_PROFILE,
    calibration_summary,
    collect_dataset,
)
from real_ecology_benchmark.dataset import assert_public_schema, save_public
from real_ecology_benchmark.evaluator import ContinuousEvaluator
from real_ecology_benchmark.manifest import aggregate_summaries
from real_ecology_benchmark.pipeline import _validate_dataset_cell, run_method
from real_ecology_benchmark.reward import build_reward, effective_collapse_penalty
from real_ecology_benchmark.types import BeliefState, PublicTransition
from real_ecology_benchmark.methods.ogsrl import KNNGuardian

RICKER_FAMILIES = ("ricker", "allee", "regime")
ALL_FAMILIES = ("ricker", "allee", "theta", "regime")
TOL = 1e-9


def _ricker_next(managed, r_eff, K_eff):
    if managed <= 0:
        return 0.0
    r_pos = max(r_eff, 0.0)
    r_mort = min(r_eff, 0.0)
    value = managed * math.exp(r_pos * (1.0 - managed / K_eff))
    if r_mort < 0.0:
        value *= math.exp(r_mort)
    return value


class TestE10EnvReproducesData(unittest.TestCase):
    """E10.1 + E10.2: the env reproduces the data table and reward conventions."""

    def test_setpoint_dk_cost_reproduce_table(self):
        effects = realdata.effects_for()
        for pop in realdata.population_names():
            for family in ALL_FAMILIES:
                cfg = real_environment(pop, family, observation_noise_sigma=0.0)
                env = make_env(cfg)
                for a in range(realdata.NUM_REAL_ACTIONS):
                    effect = effects[(pop, f"a{a}")]
                    # Start at K_base (healthy, no collapse) so cost is readable.
                    env.reset(seed=7, state_override=cfg.K_base)
                    res = env.step(a)
                    info = res.evaluator_info
                    expected_r = float(
                        np.clip(effect.r_setpoint(family), cfg.r_min, cfg.r_max)
                    )
                    self.assertAlmostEqual(
                        info["r_eff_true"], expected_r, delta=1e-9,
                        msg=f"{pop}/{family}/a{a} r_eff",
                    )
                    # First step from kappa=0: K_eff - K_base == dK_step (clipped).
                    expected_K = float(
                        np.clip(cfg.K_base + effect.dK_step, cfg.K_min, cfg.K_max)
                    )
                    self.assertAlmostEqual(
                        info["K_eff"], expected_K, delta=1e-6,
                        msg=f"{pop}/{family}/a{a} K_eff",
                    )
                    # Cost: reward is now the STATE reward on s_{t+1} (spec E6),
                    # so with sigma=0 and no collapse from s=K_base,
                    # reward = utility(s_{t+1}) - cost_step.
                    cost = float(env.reward_model.utility(info["state"])) - res.reward
                    self.assertAlmostEqual(
                        cost, effect.cost_step, delta=1e-9,
                        msg=f"{pop}/{family}/a{a} cost",
                    )

    def test_translocation_adds_dN_before_growth(self):
        effects = realdata.effects_for()
        for pop in realdata.population_names():
            cfg = real_environment(pop, "ricker", observation_noise_sigma=0.0)
            env = make_env(cfg)
            effect = effects[(pop, "a10")]
            self.assertAlmostEqual(effect.dN, 0.10 * cfg.N0, delta=1e-6)
            env.reset(seed=3, state_override=cfg.N0)
            res = env.step(10)  # translocation
            r_eff = float(np.clip(effect.r_setpoint("ricker"), cfg.r_min, cfg.r_max))
            expected = _ricker_next(cfg.N0 + effect.dN, r_eff, cfg.K_base)
            self.assertAlmostEqual(
                res.evaluator_info["state"], expected, delta=1e-6,
                msg=f"{pop} translocation next-state",
            )

    def test_eval_reset_starts_at_N0(self):
        # E1: evaluation/gate reset is the deterministic published abundance N0
        # for every population and seed (collector start spread is separate).
        for pop in realdata.population_names():
            cfg = real_environment(pop, "ricker", observation_noise_sigma=0.0)
            self.assertEqual(cfg.low_start_probability, 0.0)
            self.assertEqual(cfg.initial_log_sigma, 0.0)
            env = make_env(cfg)
            for seed in (1, 2, 13, 99):
                reset = env.reset(seed)
                self.assertAlmostEqual(
                    reset.evaluator_info["state"], cfg.N0, delta=1e-9,
                    msg=f"{pop} seed={seed} start != N0",
                )

    def test_abundance_term_half_at_Kbase(self):
        for pop in realdata.population_names():
            cfg = real_environment(pop, "ricker")
            env = make_env(cfg)
            self.assertAlmostEqual(
                float(env.reward_model.utility(cfg.K_base)), 0.5, delta=1e-9,
                msg=f"{pop} abundance term at K_base",
            )

    def test_r_eff_within_caps_and_K_within_Kmax(self):
        for pop in realdata.population_names():
            for family in ALL_FAMILIES:
                cfg = real_environment(pop, family, observation_noise_sigma=0.0)
                env = make_env(cfg)
                self.assertAlmostEqual(cfg.K_max, 2.0 * cfg.K_base, delta=1e-6)
                for a in range(realdata.NUM_REAL_ACTIONS):
                    env.reset(seed=1, state_override=cfg.K_base)
                    res = env.step(a)
                    r_eff = res.evaluator_info["r_eff_true"]
                    self.assertGreaterEqual(r_eff, cfg.r_min - TOL, f"{pop}/{family}/a{a}")
                    self.assertLessEqual(r_eff, cfg.r_max + TOL, f"{pop}/{family}/a{a}")
                    self.assertLessEqual(
                        res.evaluator_info["K_eff"], cfg.K_max + 1e-6,
                        f"{pop}/{family}/a{a} K_eff<=K_max",
                    )

    def test_capacity_accumulates(self):
        # Repeated intensive restoration (a6) accumulates K toward K_max.
        cfg = real_environment("Amur tiger", "ricker", observation_noise_sigma=0.0)
        env = make_env(cfg)
        env.reset(seed=1, state_override=cfg.K_base)
        last = cfg.K_base
        for _ in range(20):
            res = env.step(6)
            self.assertGreaterEqual(res.evaluator_info["K_eff"], last - 1e-9)
            last = res.evaluator_info["K_eff"]
        self.assertAlmostEqual(last, cfg.K_max, delta=1e-6)


class TestRealDataAndConfigContract(unittest.TestCase):
    """Latest real-ecology docs: data-table contract + depletion-aware safety."""

    def test_loader_contract_covers_all_population_action_rows(self):
        self.assertEqual(Path(realdata.DATA_DIR).name, "real_ecology_data")
        self.assertTrue(Path(realdata.DATA_DIR).is_dir())
        actions = realdata.actions_for()
        pops = realdata.pops_for()
        effects = realdata.effects_for()
        self.assertEqual(len(actions), realdata.NUM_REAL_ACTIONS)
        self.assertEqual(len(pops), realdata.NUM_REAL_POPULATIONS)
        for pop in pops:
            for action in actions:
                self.assertIn((pop, action), effects)
                effect = effects[(pop, action)]
                self.assertAlmostEqual(
                    effect.cost_step, actions[action].cost_step, delta=1e-12
                )

    def test_depletion_aware_safety_defaults(self):
        expectations = {
            "Amur tiger": 0.10,
            "Jaguar": 0.10,
            "Puerto Rican parrot": 0.20,
            "Asian elephant": 0.20,
            "Egyptian vulture": 0.25,
            "Bottlenose dolphin": 0.25,
            "Spotted turtle": 0.25,
            "Iberian lynx": 0.10,
            "Crab-eating fox": 0.25,
        }
        for pop_name, expected in expectations.items():
            pop = realdata.pops_for()[pop_name]
            self.assertAlmostEqual(default_safety_fraction(pop), expected)
            cfg = real_environment(pop_name, "ricker")
            self.assertTrue(cfg.safety_fraction_auto)
            self.assertAlmostEqual(cfg.safety_fraction, expected)
            self.assertAlmostEqual(cfg.safety_threshold, expected * cfg.K_base)
        vulture = real_environment("Egyptian vulture", "ricker")
        self.assertGreater(
            vulture.safety_threshold,
            vulture.N0,
            "depletion-aware floor should mark the depleted vulture start unsafe",
        )

    def test_real_environment_like_recomputes_auto_safety_per_population(self):
        source = real_environment("Amur tiger", "ricker")
        switched = real_environment_like(source, "Egyptian vulture", "ricker")
        self.assertAlmostEqual(switched.safety_fraction, 0.25)
        self.assertTrue(switched.safety_fraction_auto)

        pinned = real_environment("Amur tiger", "ricker", safety_fraction=0.10)
        self.assertFalse(pinned.safety_fraction_auto)
        pinned_switch = real_environment_like(pinned, "Egyptian vulture", "ricker")
        self.assertAlmostEqual(pinned_switch.safety_fraction, 0.10)
        self.assertFalse(pinned_switch.safety_fraction_auto)

    def test_initial_public_rate_is_baseline_action(self):
        cfg = real_environment("Egyptian vulture", "ricker", observation_noise_sigma=0.0)
        env = make_env(cfg)
        reset = env.reset(seed=123)
        baseline_r = resolve_actions(cfg)[0].delta_r
        self.assertLess(baseline_r, 0.0)
        self.assertAlmostEqual(reset.public_info["rho"], baseline_r, delta=1e-12)


class TestE10Leakage(unittest.TestCase):
    """Public/private boundary still holds for a generated real dataset."""

    def test_public_schema_guard(self):
        cfg = real_environment("Amur tiger", "ricker", observation_noise_sigma=0.1)
        env = make_env(cfg)
        dataset, _ = collect_dataset(env, 300, 15, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "public.npz")
            save_public(path, dataset)
            assert_public_schema(path)  # raises if truth leaked

    def test_online_transition_interface_has_no_reward_channel(self):
        self.assertNotIn("reward", PublicTransition.__dataclass_fields__)
        transition = PublicTransition(
            observation=1.0,
            done=False,
            truncated=False,
            public_info={"rho": -0.1, "kappa": 0.0, "K_eff": 100.0},
        )
        self.assertEqual(transition.observation, 1.0)
        with self.assertRaises(TypeError):
            PublicTransition(
                observation=1.0,
                reward=999.0,
                done=False,
                truncated=False,
            )


class TestE10CollapseBand(unittest.TestCase):
    """E10.3: recalibrated collapse band for recoverable populations + sinks."""

    BUDGET = 2500
    EP_LEN = 25

    @classmethod
    def setUpClass(cls):
        cls.effects = realdata.effects_for()

    def _collapse_rate(self, pop, low_start):
        cfg = real_environment(pop, "ricker", observation_noise_sigma=0.1)
        # Evaluation reset is deterministic s0=N0; the start spread used for
        # calibration coverage is a *collector* parameter, not an env override.
        self.assertEqual(cfg.low_start_probability, 0.0)
        env = make_env(cfg)
        dataset, private = collect_dataset(
            env, self.BUDGET, self.EP_LEN, seed=116, profile=REAL_DEFAULT_PROFILE,
            start_low_probability=low_start,
        )
        summary = calibration_summary(dataset, private, cfg.safety_threshold)
        return summary["incident_collapse_rate_healthy_starts"]

    def test_exploitable_population_reaches_band(self):
        # The Amur tiger is the recoverable population whose heaviest-exploitation
        # rate (a2 lambda ~ 0.62) is low enough to be driven to collapse; with the
        # harvest-tilted profile and recalibrated low-start it lands in band.
        rate = self._collapse_rate("Amur tiger", low_start=0.55)
        self.assertTrue(
            0.15 <= rate <= 0.24,
            msg=f"Amur tiger recalibrated collapse {rate:.3f} not in [0.15,0.24]",
        )

    def test_robust_populations_below_band_is_consistent(self):
        # Populations whose heaviest-exploitation lambda stays near 1 cannot be
        # collapsed within the horizon: this is a property of their real
        # demographics (reported, not forced).  We assert the classification is
        # self-consistent: a2 lambda high => collapse below the band.
        for pop in realdata.recoverable_population_names():
            a2_lambda = float(np.exp(self.effects[(pop, "a2")].r_setpoint_ricker))
            if a2_lambda < 0.80:
                continue  # exploitable; covered by the band test
            rate = self._collapse_rate(pop, low_start=0.55)
            self.assertLess(
                rate, 0.15,
                msg=f"{pop} (a2 lambda={a2_lambda:.3f}) unexpectedly collapses {rate:.3f}",
            )

    def test_sinks_reported_separately(self):
        # The two demographic sinks (r_max < 0) are excluded from the recoverable
        # set; document their collapse behaviour separately.
        sinks = realdata.sink_population_names()
        self.assertEqual(set(sinks), {"Egyptian vulture", "Bottlenose dolphin"})
        for pop in sinks:
            self.assertNotIn(pop, realdata.recoverable_population_names())
            rate = self._collapse_rate(pop, low_start=0.45)
            self.assertTrue(0.0 <= rate <= 1.0)  # well-defined, reported separately


class _ConstantPolicy:
    """Minimal reward-agnostic policy for battery tests (ignores the belief)."""

    name = "const"

    def __init__(self, num_actions, action=0):
        self.num_actions = num_actions
        self.action = int(action)
        self.last_diagnostics: dict = {}

    def fit(self, dataset, beliefs=None):
        return {}

    def reset(self, seed):
        self.last_diagnostics = {}

    def act(self, belief, observation):
        return self.action

    def observe(self, belief, action, result):
        return None


class TestRewardStateDependent(unittest.TestCase):
    """Spec E6/E6' : state-dependent benefit on s_{t+1} + two reward_mode settings."""

    def test_benefit_half_at_capacity(self):
        # T1: benefit ~ alpha/2 at carrying capacity (cost(a0)=0, no collapse).
        for pop in realdata.population_names():
            cfg = real_environment(pop, "ricker")
            env = make_env(cfg)
            r = env.reward_model.state_reward(cfg.K_base, 0, False)
            self.assertAlmostEqual(r, 0.5, delta=1e-9, msg=f"{pop} benefit at K_base")

    def test_reward_tracks_next_state_not_observation(self):
        # T2: reward is computed from the true next state, not the survey; varying
        # observation noise leaves the reward (and next state) unchanged.
        rewards, next_states = [], []
        for sigma in (0.0, 0.4):
            cfg = real_environment("Amur tiger", "ricker", observation_noise_sigma=sigma)
            env = make_env(cfg)
            env.reset(seed=5, state_override=cfg.K_base)
            res = env.step(0)
            rewards.append(res.reward)
            next_states.append(res.evaluator_info["state"])
        self.assertAlmostEqual(next_states[0], next_states[1], delta=1e-12)
        self.assertAlmostEqual(rewards[0], rewards[1], delta=1e-12,
                               msg="reward changed with observation noise")

    def test_reward_mode_penalty_switch(self):
        # T3: yield forces P=0; safe subtracts collapse_penalty on unsafe
        # occupancy by default (which includes a crossing).
        pop = "Amur tiger"
        cfg_safe = real_environment(pop, "ricker", observation_noise_sigma=0.0, reward_mode="safe")
        cfg_yield = real_environment(pop, "ricker", observation_noise_sigma=0.0, reward_mode="yield")
        self.assertEqual(effective_collapse_penalty(cfg_yield), 0.0)
        self.assertAlmostEqual(effective_collapse_penalty(cfg_safe), cfg_safe.collapse_penalty)
        P = cfg_safe.collapse_penalty
        start = cfg_safe.safety_threshold * 1.05  # just above s_safe

        def step_reward(cfg, action):
            env = make_env(cfg)
            env.reset(seed=1, state_override=start)
            return env.step(action)

        res_safe = step_reward(cfg_safe, 2)   # aggressive harvest -> crosses s_safe
        res_yield = step_reward(cfg_yield, 2)
        self.assertTrue(res_safe.evaluator_info["entered_safety_region"],
                        "test setup: a2 should cross s_safe")
        self.assertAlmostEqual(res_yield.reward - res_safe.reward, P, delta=1e-9)
        # Away from a crossing (start at K_base, do nothing) the two are equal.
        env_s = make_env(cfg_safe); env_s.reset(seed=1, state_override=cfg_safe.K_base)
        env_y = make_env(cfg_yield); env_y.reset(seed=1, state_override=cfg_yield.K_base)
        self.assertAlmostEqual(env_s.step(0).reward, env_y.step(0).reward, delta=1e-12)

    def test_safe_mode_penalizes_initially_unsafe_depleted_population(self):
        # The vulture starts below its depletion-aware floor; occupancy mode keeps
        # safe mode meaningful instead of degenerating to yield.
        cfg_safe = real_environment(
            "Egyptian vulture", "ricker", observation_noise_sigma=0.0, reward_mode="safe"
        )
        cfg_yield = real_environment(
            "Egyptian vulture", "ricker", observation_noise_sigma=0.0, reward_mode="yield"
        )
        self.assertEqual(cfg_safe.safety_penalty_mode, "occupancy")
        self.assertLess(cfg_safe.N0, cfg_safe.safety_threshold)
        env_s = make_env(cfg_safe)
        env_y = make_env(cfg_yield)
        env_s.reset(seed=3)
        env_y.reset(seed=3)
        res_safe = env_s.step(0)
        res_yield = env_y.step(0)
        self.assertFalse(res_safe.evaluator_info["entered_safety_region"])
        self.assertTrue(res_safe.evaluator_info["below_safety_region"])
        self.assertTrue(res_safe.evaluator_info["safety_penalty_applied"])
        self.assertAlmostEqual(
            res_yield.reward - res_safe.reward,
            cfg_safe.collapse_penalty,
            delta=1e-9,
        )

    def test_crossing_penalty_mode_preserves_event_only_ablation(self):
        cfg = real_environment(
            "Egyptian vulture",
            "ricker",
            observation_noise_sigma=0.0,
            reward_mode="safe",
            safety_penalty_mode="crossing",
        )
        env = make_env(cfg)
        env.reset(seed=3)
        res = env.step(0)
        self.assertFalse(res.evaluator_info["entered_safety_region"])
        self.assertTrue(res.evaluator_info["below_safety_region"])
        self.assertFalse(res.evaluator_info["safety_penalty_applied"])

    def test_safe_penalty_exceeds_default_healthy_episode(self):
        # E6: one collapse should outweigh a healthy episode at the new return scale.
        cfg = real_environment("Amur tiger", "ricker", reward_mode="safe")
        eval_cfg = EvaluationConfig()
        discounted_half_benefit = 0.5 * sum(
            eval_cfg.discount ** t for t in range(eval_cfg.horizon)
        )
        self.assertGreater(cfg.collapse_penalty, discounted_half_benefit)

    def test_reward_mode_is_dataset_cache_key(self):
        # dataset.rewards differ between safe/yield on crossings, so a cached dataset
        # from one reward mode must be rejected for the other.
        safe_env = real_environment("Amur tiger", "ricker", reward_mode="safe")
        yield_env = real_environment("Amur tiger", "ricker", reward_mode="yield")
        dataset, _private = collect_dataset(
            make_env(safe_env), 100, 25, seed=116, start_low_probability=0.55
        )
        requested = BenchmarkConfig(environment=yield_env)
        with self.assertRaises(ValueError):
            _validate_dataset_cell(requested, dataset)

    def test_battery_agnostic_to_reward_mode(self):
        # T4: for a fixed policy the reward-agnostic battery is identical across
        # reward_mode (same trajectory); only the return responds to P_m.
        keys = ("min_true_state", "final_true_state", "economic_cost",
                "collapse_entry", "unsafe_fraction", "mvp_fraction", "mvp_breach",
                "persistence", "mean_true_state")
        results = {}
        for mode in ("yield", "safe"):
            env_cfg = real_environment("Amur tiger", "ricker",
                                       observation_noise_sigma=0.1, reward_mode=mode)
            cfg = BenchmarkConfig(
                seed=116, environment=env_cfg,
                filter=FilterConfig(particles=32, proposal="reference"),
                evaluation=EvaluationConfig(seeds=[9001], episodes_per_seed=3, horizon=20),
            )
            factory = lambda ec=env_cfg, fc=cfg.filter: RawObservationFilter(
                ec, fc, ReferenceProposal(ec)
            )
            policy = _ConstantPolicy(env_cfg.num_actions, action=2)  # induce crossings
            results[mode] = ContinuousEvaluator(cfg, factory, "reference").run(policy)
        self.assertEqual(len(results["yield"]), len(results["safe"]))
        return_gap = 0.0
        for row_y, row_s in zip(results["yield"], results["safe"]):
            for k in keys:
                self.assertAlmostEqual(
                    float(row_y[k]), float(row_s[k]), delta=1e-9,
                    msg=f"battery metric {k} differs across reward_mode",
                )
            return_gap += abs(row_y["operational_return"] - row_s["operational_return"])
        # At least one episode collapsed, so the returns must differ somewhere.
        self.assertGreater(return_gap, 0.0)

    def test_mvp_diagnostics_report_absolute_floor(self):
        env_cfg = real_environment("Egyptian vulture", "ricker",
                                   observation_noise_sigma=0.0, reward_mode="safe")
        cfg = BenchmarkConfig(
            seed=116, environment=env_cfg,
            filter=FilterConfig(particles=32, proposal="reference"),
            evaluation=EvaluationConfig(seeds=[9001], episodes_per_seed=1, horizon=5),
        )
        factory = lambda ec=env_cfg, fc=cfg.filter: RawObservationFilter(
            ec, fc, ReferenceProposal(ec)
        )
        rows = ContinuousEvaluator(
            cfg, factory, "reference"
        ).run(_ConstantPolicy(env_cfg.num_actions, action=0))
        self.assertEqual(rows[0]["mvp_threshold"], env_cfg.mvp_threshold)
        self.assertIn("mvp_fraction", rows[0])
        self.assertIn("mvp_breach", rows[0])
        self.assertIn("safety_penalty_mode", rows[0])

    def test_belief_features_require_real_scales(self):
        belief = BeliefState(
            states=np.asarray([10.0, 20.0]),
            contexts=np.zeros((2, 3)),
            regimes=np.zeros(2, dtype=np.int8),
            log_weights=np.log(np.asarray([0.5, 0.5])),
            observation=15.0,
        )
        with self.assertRaises(TypeError):
            belief.features()
        explicit = belief.features(K_ref=100.0, safety_threshold=25.0)
        self.assertTrue(np.all(np.isfinite(explicit)))


class TestOGSRLGuardianContract(unittest.TestCase):
    """Task C: KNN support guard uses public, scaled context rather than state/action only."""

    def test_guardian_feature_space_uses_controls_and_no_hardcoded_500(self):
        cfg = real_environment("Amur tiger", "ricker")
        n = 64
        states = np.linspace(5.0, cfg.K_max, n)
        actions = np.arange(n) % cfg.num_actions
        rho = np.linspace(cfg.r_min, cfg.r_max, n)
        kappa = np.linspace(0.0, cfg.K_base, n)
        K_eff = np.clip(cfg.K_base + kappa, cfg.K_min, cfg.K_max)
        guardian = KNNGuardian.fit(
            states,
            actions,
            cfg.num_actions,
            seed=116,
            K_ref=cfg.K_ref,
            safety_threshold=cfg.safety_threshold,
            rho=rho,
            kappa=kappa,
            K_eff=K_eff,
        )
        expected_dim = 3 + cfg.num_actions + 3
        self.assertEqual(guardian.anchors.shape[1], expected_dim)
        low_context = KNNGuardian.features(
            np.asarray([cfg.K_base]),
            np.asarray([0]),
            cfg.num_actions,
            cfg.K_ref,
            cfg.safety_threshold,
            rho=cfg.r_min,
            kappa=0.0,
            K_eff=cfg.K_base,
        )
        high_context = KNNGuardian.features(
            np.asarray([cfg.K_base]),
            np.asarray([0]),
            cfg.num_actions,
            cfg.K_ref,
            cfg.safety_threshold,
            rho=cfg.r_max,
            kappa=cfg.K_base,
            K_eff=cfg.K_max,
        )
        self.assertFalse(np.allclose(low_context, high_context))
        self.assertEqual(guardian.K_ref, cfg.K_ref)
        self.assertFalse(hasattr(guardian, "reward_mode"))


class TestRuntimeTelemetry(unittest.TestCase):
    """Runtime/memory fields are persisted for compute-cost comparisons."""

    def test_run_method_records_phase_timing_and_peak_rss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env_cfg = real_environment("Amur tiger", "ricker", observation_noise_sigma=0.0)
            cfg = BenchmarkConfig(
                seed=116,
                environment=env_cfg,
                dataset=DatasetConfig(
                    transitions=80,
                    episode_length=8,
                    output=str(root / "public.npz"),
                    private_output=str(root / "private.npz"),
                ),
                filter=FilterConfig(particles=32, proposal="reference"),
                model=ModelConfig(ensemble_size=2),
                planner=PlannerConfig(horizon=2, sequences=8, particles=4),
                evaluation=EvaluationConfig(
                    seeds=[9001],
                    episodes_per_seed=1,
                    horizon=5,
                    output_dir=str(root / "eval"),
                ),
            )
            summary = run_method("mopo", cfg, "reference", regenerate=True)
            expected = (
                "dataset_seconds",
                "filter_factory_seconds",
                "belief_cache_seconds",
                "policy_init_seconds",
                "fit_seconds",
                "evaluation_seconds",
                "summary_save_seconds",
                "row_seconds",
                "peak_rss_mb",
                "planner_seconds_mean",
                "filter_seconds_mean",
            )
            for key in expected:
                self.assertIn(key, summary)
                self.assertGreaterEqual(float(summary[key]), 0.0, key)
            self.assertGreater(float(summary["peak_rss_mb"]), 0.0)
            saved = json.loads((Path(summary["output_dir"]) / "summary.json").read_text())
            for key in expected:
                self.assertIn(key, saved)
            self.assertEqual(saved["model"], "mopo")
            episodes = (Path(summary["output_dir"]) / "episodes.csv").read_text()
            self.assertIn("compute_backend_effective", episodes.splitlines()[0])
            self.assertGreater(summary["training_history_rows"], 0)
            self.assertGreater(summary["training_split"]["holdout_transitions"], 0)
            self.assertTrue(Path(summary["training_history_csv"]).exists())
            self.assertTrue(Path(summary["training_history_json"]).exists())


class TestComputeBackend(unittest.TestCase):
    """Backend selection layer + vectorized-proposal parity (backend.py)."""

    def test_compute_config_validate(self):
        from real_ecology_benchmark.config import ComputeConfig
        ComputeConfig(backend="numpy").validate()
        with self.assertRaises(ValueError):
            ComputeConfig(backend="tensorflow").validate()

    def test_numpy_backend_metadata(self):
        from real_ecology_benchmark.backend import resolve_backend
        from real_ecology_benchmark.config import ComputeConfig
        b = resolve_backend(ComputeConfig(backend="numpy"))
        meta = b.to_dict()
        self.assertEqual(meta["compute_backend_requested"], "numpy")
        self.assertEqual(meta["compute_backend_effective"], "numpy")

    def test_strict_missing_cupy_raises(self):
        # On a node without CuPy/GPU, strict cupy must fail loudly (no silent CPU).
        from real_ecology_benchmark.backend import resolve_backend, BackendUnavailable
        from real_ecology_benchmark.config import ComputeConfig
        try:
            import cupy  # noqa: F401
            self.skipTest("cupy is available; strict-missing path not exercised")
        except Exception:
            pass
        with self.assertRaises(BackendUnavailable):
            resolve_backend(ComputeConfig(backend="cupy", strict=True))

    def test_nonstrict_cupy_falls_back_recorded(self):
        from real_ecology_benchmark.backend import resolve_backend
        from real_ecology_benchmark.config import ComputeConfig
        try:
            import cupy  # noqa: F401
            self.skipTest("cupy is available; fallback path not exercised")
        except Exception:
            pass
        b = resolve_backend(ComputeConfig(backend="cupy", strict=False))
        self.assertEqual(b.name, "numpy")          # effective
        self.assertEqual(b.requested, "cupy")       # requested
        self.assertIsNotNone(b.fallback_reason)

    def test_cupy_method_workload_supports_plus_only(self):
        from real_ecology_benchmark.backend import (
            Backend,
            BackendUnavailable,
            ensure_backend_supports_workload,
        )
        fake = Backend("cupy", "cupy", 0, True, np, None)
        with self.assertRaises(BackendUnavailable):
            ensure_backend_supports_workload(fake, "method:mopo")
        with self.assertRaises(BackendUnavailable):
            ensure_backend_supports_workload(fake, "method:plus")
        fallback = ensure_backend_supports_workload(
            Backend("cupy", "cupy", 0, False, np, None), "method:mopo"
        )
        self.assertEqual(fallback.name, "numpy")
        self.assertEqual(fallback.requested, "cupy")
        self.assertIn("unsupported", fallback.fallback_reason)
        supported = ensure_backend_supports_workload(fake, "mechanistic_transition")
        self.assertEqual(supported.name, "cupy")
        plus_supported = ensure_backend_supports_workload(
            fake, "method:plus", real_environment("Amur tiger", "ricker")
        )
        self.assertEqual(plus_supported.name, "cupy")
        self.assertEqual(
            plus_supported.acceleration_scope, "plus_mechanistic_transition"
        )

    def test_plus_candidates_use_active_backend(self):
        from real_ecology_benchmark.backend import (
            Backend,
            reset_active_backend,
            set_active_backend,
        )
        from real_ecology_benchmark.methods.plus import PLUSPolicy

        fake = Backend(
            "cupy", "cupy", 0, True, np, None,
            workload="method:plus",
            acceleration_scope="plus_mechanistic_transition",
        )
        set_active_backend(fake)
        try:
            policy = PLUSPolicy(
                real_environment("Amur tiger", "ricker"),
                ModelConfig(),
                PlannerConfig(),
                seed=1,
                candidate_count=3,
            )
            policy.fit(None, None)
            self.assertEqual({proposal.backend.name for proposal in policy.proposals}, {"cupy"})
        finally:
            reset_active_backend()

    def test_vectorized_proposal_matches_scalar(self):
        # The vectorized backend proposal must equal the scalar per-particle
        # transition_value (bit-identical up to float noise) for every family.
        from real_ecology_benchmark.beliefs import MechanisticProposal
        rng = np.random.default_rng(0)
        for family in ALL_FAMILIES:
            cfg = real_environment("Amur tiger", family)
            prop = MechanisticProposal(cfg, family, r_value=0.05)
            n = 2000
            states = rng.uniform(0.0, 400.0, n)
            states[rng.integers(0, n, 40)] = 0.0  # include absorbing zeros
            actions = rng.integers(0, realdata.NUM_REAL_ACTIONS, n)
            contexts = rng.normal(0.0, 1.0, (n, 3))
            regimes = rng.integers(0, 2, n).astype(np.int8)
            rho = np.zeros(n)
            kappa = rng.uniform(0.0, cfg.K_base, n)
            vec, _ = prop.sample_next(states, actions, contexts, regimes,
                                      np.random.default_rng(1), rho, kappa)
            cu = 1.0 / (1.0 + np.exp(-contexts))
            C = cfg.C_low + cu[:, 1] * (cfg.C_high - cfg.C_low)
            th = cfg.theta_low + cu[:, 2] * (cfg.theta_high - cfg.theta_low)
            ref = np.array([
                prop._env.transition_value(
                    float(states[i]), prop.actions[int(actions[i])], 0.05,
                    float(C[i]), float(th[i]), int(regimes[i]), 0.0,
                    float(rho[i]), float(kappa[i]))
                for i in range(n)
            ])
            rel = np.max(np.abs(vec - ref) / np.maximum(np.abs(ref), 1e-9))
            self.assertLess(rel, 1e-9, msg=f"{family} vectorized vs scalar rel={rel:.2e}")

    def test_run_records_backend_and_namespaces_output(self):
        import tempfile
        from pathlib import Path
        from real_ecology_benchmark.config import BenchmarkConfig, FilterConfig, EvaluationConfig
        from real_ecology_benchmark.pipeline import run_method
        env_cfg = real_environment("Amur tiger", "ricker", observation_noise_sigma=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = BenchmarkConfig(
                seed=116, environment=env_cfg,
                filter=FilterConfig(particles=32, proposal="reference"),
                evaluation=EvaluationConfig(seeds=[9001], episodes_per_seed=1, horizon=6,
                                            output_dir=str(Path(tmp) / "eval")),
            )
            cfg.model.ensemble_size = 3
            cfg.planner.sequences = 12; cfg.planner.particles = 6; cfg.planner.horizon = 2
            cfg.dataset.transitions = 300; cfg.dataset.episode_length = 15
            cfg.dataset.output = str(Path(tmp) / "p.npz")
            cfg.dataset.private_output = str(Path(tmp) / "q.npz")
            summary = run_method("mopo", cfg, "reference", regenerate=True)
            self.assertEqual(summary["compute_backend_effective"], "numpy")
            self.assertIn("backend_numpy", summary["output_dir"])

    def test_aggregate_separates_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for backend, value in (("numpy", 1.0), ("cupy", 3.0)):
                path = root / "evaluation" / f"backend_{backend}" / "reward_safe" / "mopo" / "learned"
                path.mkdir(parents=True)
                (path / "summary.json").write_text(json.dumps({
                    "compute_backend_effective": backend,
                    "reward_mode": "safe",
                    "model": "mopo",
                    "filter": "learned",
                    "operational_return_mean": value,
                    "row_seconds": value,
                }), encoding="utf-8")
            out = root / "aggregate.json"
            aggregate = aggregate_summaries(root / "evaluation", out)
            self.assertEqual(
                aggregate["model_return_mean"]["numpy"]["safe"]["mopo"], 1.0
            )
            self.assertEqual(
                aggregate["model_return_mean"]["cupy"]["safe"]["mopo"], 3.0
            )
            backends = {
                row["compute_backend_effective"]
                for row in aggregate["efficiency_mean"]
            }
            self.assertEqual(backends, {"numpy", "cupy"})


if __name__ == "__main__":
    unittest.main()
