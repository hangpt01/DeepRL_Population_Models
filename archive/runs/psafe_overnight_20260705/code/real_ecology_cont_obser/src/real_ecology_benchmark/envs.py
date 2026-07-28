"""Continuous unbounded ecological POMDP environments."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any

import numpy as np

from .actions import ActionSpec, action_table, resolve_actions
from .config import EnvironmentConfig
from .controls import (
    advance_public_controls,
    control_fields_enabled,
    initial_controls,
    private_r_eff,
    public_controls,
)
from .observation import LogNormalObservationModel
from .reward import build_reward, safety_penalty_indicator
from .types import ResetResult, StepResult


_LOG_FLOAT_MAX = math.log(np.finfo(np.float64).max)


def real_initial_state(
    rng: np.random.Generator,
    N0: float,
    s_safe: float,
    low_probability: float,
    log_sigma: float,
) -> float:
    """Sample a real-ecology episode start.

    With ``low_probability == 0`` and ``log_sigma == 0`` this is the spec-faithful
    deterministic start ``s0 = N0`` (used by evaluation and the gate).  The
    collector passes non-zero values to induce calibration start coverage without
    changing the evaluation reset semantics.
    """

    if low_probability > 0.0 and rng.random() < low_probability:
        return float(rng.uniform(0.5 * s_safe, 1.5 * s_safe))
    if log_sigma > 0.0:
        return float(rng.lognormal(math.log(max(N0, 1.0)), log_sigma))
    return float(N0)


def _safe_mul_exp(state: float, exponent: float) -> float:
    if state == 0.0:
        return 0.0
    log_next = math.log(state) + exponent
    if log_next > _LOG_FLOAT_MAX:
        raise FloatingPointError(
            f"unbounded dynamics overflow: log(next_state)={log_next:.3f}"
        )
    if log_next < math.log(np.finfo(np.float64).tiny):
        return 0.0
    return float(math.exp(log_next))


class ContinuousEcologyEnv:
    """One simulator class with explicit dynamics dispatch and private truth."""

    def __init__(self, cfg: EnvironmentConfig):
        cfg.validate()
        self.cfg = cfg
        self.actions = resolve_actions(cfg)
        self.observation_model = LogNormalObservationModel(cfg.observation_noise_sigma)
        # Mode-aware reward model (P=0 under real "yield", else collapse_penalty).
        self.reward_model = build_reward(cfg)
        self._rngs: dict[str, np.random.Generator] = {}
        self._state = 0.0
        self._observation = 0.0
        self._r_base = 0.0
        self._C = 0.0
        self._theta = 0.0
        self._regime = 0
        self._rho = (
            initial_controls(self.cfg).rho
            if self.cfg.control_mode == "real_setpoint"
            else 0.0
        )
        self._kappa = 0.0
        self._step = 0
        self._entry_latched = False
        self._done = True

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def state(self) -> float:
        """Evaluator-only truth; policies must not call this property."""
        return self._state

    def _spawn_rngs(self, seed: int) -> None:
        streams = np.random.SeedSequence(seed).spawn(5)
        names = ("parameters", "regime", "process", "observation", "initial")
        self._rngs = {name: np.random.default_rng(ss) for name, ss in zip(names, streams)}

    def _sample_initial(self) -> float:
        rng = self._rngs["initial"]
        if self.cfg.control_mode == "real_setpoint":
            # Per the real-ecology spec the episode starts at the population's
            # published abundance N0 (E1).  The real defaults
            # (low_start_probability=0, initial_log_sigma=0) make this exact for
            # evaluation and the gate; the collector induces its own start spread
            # (it passes state_override), so calibration coverage is decoupled
            # from evaluation reset semantics.
            return real_initial_state(
                rng,
                self.cfg.N0,
                self.cfg.safety_threshold,
                self.cfg.low_start_probability,
                self.cfg.initial_log_sigma,
            )
        if rng.random() < self.cfg.low_start_probability:
            return float(rng.uniform(25.0, 150.0))
        return float(rng.lognormal(math.log(self.cfg.K_base), self.cfg.initial_log_sigma))

    def _truth(self) -> dict[str, Any]:
        truth = {
            "state": float(self._state),
            "r_base": float(self._r_base),
            "C": float(self._C),
            "theta": float(self._theta),
            "regime": int(self._regime),
            "entry_latched": bool(self._entry_latched),
            "timestep": int(self._step),
        }
        if control_fields_enabled(self.cfg):
            controls = public_controls(self.cfg, self._rho, self._kappa)
            truth.update(
                {
                    "r_eff_true": float(
                        private_r_eff(self.cfg, self._r_base, self._rho, self._theta)
                    ),
                    "rho": controls.rho,
                    "kappa": controls.kappa,
                    "K_eff": controls.K_eff,
                }
            )
        return truth

    def _public_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"timestep": int(self._step)}
        if control_fields_enabled(self.cfg):
            info.update(public_controls(self.cfg, self._rho, self._kappa).to_public_info())
        return info

    def reset(self, seed: int, state_override: float | None = None) -> ResetResult:
        self._spawn_rngs(seed)
        p = self._rngs["parameters"]
        if self.cfg.control_mode == "cumulative_capped" and self.cfg.kind == "theta":
            self._C = float(p.uniform(self.cfg.C_low, self.cfg.C_high))
            self._theta = float(p.uniform(self.cfg.theta_low, self.cfg.theta_high))
            high = min(self.cfg.r_base_high, self.cfg.theta_base_stability_margin / self._theta)
            if high < self.cfg.r_base_low:
                high = self.cfg.r_base_low
            self._r_base = float(p.uniform(self.cfg.r_base_low, high))
        else:
            self._r_base = float(p.uniform(self.cfg.r_base_low, self.cfg.r_base_high))
            self._C = float(p.uniform(self.cfg.C_low, self.cfg.C_high))
            self._theta = float(p.uniform(self.cfg.theta_low, self.cfg.theta_high))
        self._regime = int(self._rngs["regime"].integers(0, 2))
        self._rho = (
            initial_controls(self.cfg).rho
            if self.cfg.control_mode == "real_setpoint"
            else 0.0
        )
        self._kappa = 0.0
        self._state = self._sample_initial() if state_override is None else float(state_override)
        if self._state < 0:
            raise ValueError("state_override must be non-negative")
        self._step = 0
        self._entry_latched = False
        self._done = self._state == 0.0
        self._observation = self.observation_model.sample(
            self._state, self._rngs["observation"]
        )
        return ResetResult(
            observation=self._observation,
            done=self._done,
            public_info=self._public_info(),
            evaluator_info=self._truth(),
        )

    def _managed_state(self, action: ActionSpec) -> float:
        return max(self._state * (1.0 - action.harvest_fraction) + action.stocking_delta, 0.0)

    def transition_value(
        self,
        state: float,
        action: ActionSpec,
        r_base: float,
        C: float,
        theta: float,
        regime: int,
        process_noise: float = 0.0,
        rho: float = 0.0,
        kappa: float = 0.0,
    ) -> float:
        if state == 0.0:
            return 0.0
        if control_fields_enabled(self.cfg):
            rho_next, _kappa_next, K_eff_arr = advance_public_controls(
                self.cfg, action.id, rho, kappa
            )
            # Translocation (a10) is the only direct-state action: it adds
            # stocking_delta = 0.10*N0 to abundance *before* growth.  Every other
            # real action has stocking_delta == 0, so this is an identity for
            # them (and for Tier-3 cumulative, where direct-s authority is 0).
            managed = max(state + action.stocking_delta, 0.0)
            r_eff = float(private_r_eff(self.cfg, r_base, rho_next, theta))
            K_eff = float(K_eff_arr)
        else:
            managed = max(
                state * (1.0 - action.harvest_fraction) + action.stocking_delta, 0.0
            )
            r_eff = r_base + action.delta_r
            K_eff = self.cfg.K_base + action.delta_K
        if managed == 0.0:
            return 0.0
        if K_eff <= 0:
            raise FloatingPointError("effective carrying capacity must be positive")
        # Tier-3 lets sustained harvest push the effective rate negative
        # (r_min < 0).  A negative r inside the density-dependent exponent is
        # ill-posed: for s>K the factor (1-s/K)<0 (and below C the Allee factor
        # also flips sign), so a negative r turns into a POSITIVE exponent and
        # the map explodes to overflow instead of declining.  Split the rate:
        # the growth part max(r_eff,0) drives density-dependent dynamics exactly
        # as before, while the net-negative part min(r_eff,0) acts as
        # unconditional per-capita exploitation mortality exp(r_mort).  This is
        # an identity when r_eff>=0 (so tier2_one_step, where r_eff>=0.10, is
        # bit-for-bit unchanged) and makes harvest a guaranteed monotone decline.
        r_pos = max(r_eff, 0.0)
        r_mort = min(r_eff, 0.0)
        kind = self.cfg.kind
        if kind == "ricker":
            exponent = r_pos * (1.0 - managed / K_eff) + process_noise
            value = _safe_mul_exp(managed, exponent)
        elif kind == "allee":
            exponent = r_pos * (1.0 - managed / K_eff) * (managed / C - 1.0)
            value = _safe_mul_exp(managed, exponent + process_noise)
        elif kind == "theta":
            ratio = max(managed / K_eff, 0.0)
            growth = r_pos * managed * (1.0 - ratio ** theta)
            value = managed + growth + process_noise * managed
        elif kind == "regime":
            # Regime thresholds and the weak-regime multiplier are config-driven;
            # for the real setting they are scaled to the population's K_base so
            # the depensation band stays inside [0, K] for K in 31..325.
            threshold = (
                self.cfg.regime_threshold_low if regime == 0
                else self.cfg.regime_threshold_high
            )
            multiplier = 1.0 if regime == 0 else self.cfg.regime_weak_multiplier
            exponent = multiplier * r_pos * (1.0 - managed / K_eff) * (
                managed / threshold - 1.0
            )
            value = _safe_mul_exp(managed, exponent + process_noise)
        else:  # guarded by config validation
            raise ValueError(kind)
        if r_mort < 0.0:
            value *= math.exp(r_mort)
        if not np.isfinite(value):
            raise FloatingPointError("non-finite next state")
        return float(max(value, 0.0))

    def step(self, action_id: int) -> StepResult:
        if self._done:
            raise RuntimeError("step called on terminal environment; call reset")
        if not 0 <= action_id < self.num_actions:
            raise ValueError(f"invalid action {action_id}")
        action = self.actions[action_id]
        state_prev = self._state
        observation_prev = self._observation
        regime_prev = self._regime
        rho_prev = self._rho
        kappa_prev = self._kappa
        noise = 0.0
        if self.cfg.process_noise_sigma > 0:
            noise = float(
                self._rngs["process"].normal(0.0, self.cfg.process_noise_sigma)
            )
        state_next = self.transition_value(
            state_prev,
            action,
            self._r_base,
            self._C,
            self._theta,
            regime_prev,
            noise,
            rho_prev,
            kappa_prev,
        )
        rho_next = rho_prev
        kappa_next = kappa_prev
        K_eff_next = self.cfg.K_base + action.delta_K
        if control_fields_enabled(self.cfg):
            rho_arr, kappa_arr, K_eff_arr = advance_public_controls(
                self.cfg, action_id, rho_prev, kappa_prev
            )
            rho_next = float(rho_arr)
            kappa_next = float(kappa_arr)
            K_eff_next = float(K_eff_arr)
        entered_now = (
            not self._entry_latched
            and state_prev > self.cfg.safety_threshold
            and state_next <= self.cfg.safety_threshold
        )
        if entered_now:
            self._entry_latched = True
        below_safety_now = state_next <= self.cfg.safety_threshold
        safety_penalty_now = bool(
            safety_penalty_indicator(self.cfg, state_prev, state_next, entered_now)
        )
        if self.cfg.control_mode == "real_setpoint":
            # Real-ecology reward: benefit on the TRUE next state s_{t+1} (spec E6),
            # logged into the public dataset so every method consumes identical
            # (o_t, a_t, R_t, o_{t+1}); operational == true here.
            reward = self.reward_model.state_reward(
                state_next, action_id, safety_penalty_now
            )
            reward_true = reward
        else:
            reward = self.reward_model.operational(
                observation_prev, action_id, safety_penalty_now
            )
            reward_true = self.reward_model.true(
                state_prev, action_id, safety_penalty_now
            )
        if self.cfg.kind == "regime":
            if self._rngs["regime"].random() > self.cfg.regime_persistence:
                self._regime = 1 - self._regime
        self._rho = rho_next
        self._kappa = kappa_next
        self._state = state_next
        self._step += 1
        terminated = state_next == 0.0
        truncated = self._step >= self.cfg.horizon and not terminated
        self._done = terminated or truncated
        self._observation = self.observation_model.sample(
            state_next, self._rngs["observation"]
        )
        private = self._truth()
        private.update(
            {
                "state_previous": float(state_prev),
                "regime_previous": int(regime_prev),
                "entered_safety_region": bool(entered_now),
                "below_safety_region": bool(below_safety_now),
                "safety_penalty_applied": bool(safety_penalty_now),
                "below_mvp_region": bool(state_next <= self.cfg.mvp_threshold),
                "reward_true": float(reward_true),
            }
        )
        if control_fields_enabled(self.cfg):
            previous_controls = public_controls(self.cfg, rho_prev, kappa_prev)
            private.update(
                {
                    "rho_previous": previous_controls.rho,
                    "kappa_previous": previous_controls.kappa,
                    "K_eff_previous": previous_controls.K_eff,
                    "rho": float(rho_next),
                    "kappa": float(kappa_next),
                    "K_eff": float(K_eff_next),
                    "r_eff_true": float(
                        private_r_eff(self.cfg, self._r_base, rho_next, self._theta)
                    ),
                }
            )
        return StepResult(
            observation=self._observation,
            reward=reward,
            done=self._done,
            truncated=truncated,
            public_info=self._public_info(),
            evaluator_info=private,
        )

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.cfg)


def make_env(cfg: EnvironmentConfig | dict[str, Any]) -> ContinuousEcologyEnv:
    if isinstance(cfg, dict):
        cfg = EnvironmentConfig(**cfg)
    return ContinuousEcologyEnv(cfg)
