"""Known-emission particle filters, proposal ladder, and offline belief caching."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from . import realdata
from .actions import action_table, resolve_actions
from .backend import Backend, get_active_backend
from .config import EnvironmentConfig, FilterConfig
from .controls import advance_public_controls, control_fields_enabled, initial_controls, public_controls
from .dataset import TrajectoryDataset
from .envs import ContinuousEcologyEnv
from .observation import LogNormalObservationModel
from .types import BeliefState, TransitionProposal


_LOG_FLOAT_MAX = math.log(np.finfo(np.float64).max)


def real_next_states(
    cfg, backend, delta_r_dev, delta_K_dev, stock_dev,
    states, actions, C, theta, regimes, rho, kappa,
):
    """Vectorized backend-array form of ``env.transition_value`` (real_setpoint).

    Computes the next latent state for every particle at once, on the active
    backend (NumPy CPU reference or CuPy GPU).  Bit-identical to the scalar
    per-particle loop up to floating-point noise; overflow is clipped rather than
    raised (unreachable in the real setting, where r and s are bounded).  Shared
    by :class:`MechanisticProposal` and the gate's ``ExactEpisodeProposal``.
    """

    b = backend
    xp = b.xp
    s = b.asarray(states)
    a = b.asarray(actions)
    C = b.asarray(C)
    theta = b.asarray(theta)
    reg = b.asarray(regimes)
    rho = b.asarray(rho)
    kappa = b.asarray(kappa)
    dr = delta_r_dev[a]
    dK = delta_K_dev[a]
    ds = stock_dev[a]
    rho_next = (1.0 - cfg.accumulator_decay_r) * rho + dr
    kappa_next = (1.0 - cfg.accumulator_decay_K) * kappa + dK
    K_eff = xp.clip(cfg.K_base + kappa_next, cfg.K_min, cfg.K_max)
    r_eff = xp.clip(rho_next, cfg.r_min, cfg.r_max)  # real: r_base ignored
    managed = xp.maximum(s + ds, 0.0)
    r_pos = xp.maximum(r_eff, 0.0)
    r_mort = xp.minimum(r_eff, 0.0)
    kind = cfg.kind
    if kind == "theta":
        ratio = xp.maximum(managed / K_eff, 0.0)
        value = managed + r_pos * managed * (1.0 - ratio ** theta)
    else:
        if kind == "ricker":
            exponent = r_pos * (1.0 - managed / K_eff)
        elif kind == "allee":
            exponent = r_pos * (1.0 - managed / K_eff) * (managed / C - 1.0)
        elif kind == "regime":
            thr = xp.where(reg == 0, cfg.regime_threshold_low, cfg.regime_threshold_high)
            mult = xp.where(reg == 0, 1.0, cfg.regime_weak_multiplier)
            exponent = mult * r_pos * (1.0 - managed / K_eff) * (managed / thr - 1.0)
        else:  # pragma: no cover - guarded by config
            raise ValueError(kind)
        safe = managed > 0.0
        log_managed = xp.log(xp.where(safe, managed, 1.0))
        value = xp.where(
            safe, xp.exp(xp.clip(log_managed + exponent, -700.0, _LOG_FLOAT_MAX)), 0.0
        )
    value = xp.where(r_mort < 0.0, value * xp.exp(r_mort), value)
    value = xp.where(s == 0.0, 0.0, xp.maximum(value, 0.0))
    return b.to_numpy(value)


def real_action_lookups(backend, actions):
    """Return device (delta_r, delta_K, stocking_delta) lookups for a spec tuple."""
    delta_r = np.asarray([a.delta_r for a in actions], dtype=np.float64)
    delta_K = np.asarray([a.delta_K for a in actions], dtype=np.float64)
    stock = np.asarray([a.stocking_delta for a in actions], dtype=np.float64)
    return backend.asarray(delta_r), backend.asarray(delta_K), backend.asarray(stock)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def _broadcast_controls(
    states: np.ndarray,
    rho: float | np.ndarray | None,
    kappa: float | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    rho_values = np.zeros_like(states, dtype=np.float64) if rho is None else np.broadcast_to(
        np.asarray(rho, dtype=np.float64), states.shape
    )
    kappa_values = np.zeros_like(states, dtype=np.float64) if kappa is None else np.broadcast_to(
        np.asarray(kappa, dtype=np.float64), states.shape
    )
    return rho_values, kappa_values


def _next_public_controls(
    env_cfg: EnvironmentConfig,
    states: np.ndarray,
    actions: np.ndarray,
    rho: float | np.ndarray | None,
    kappa: float | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_values, kappa_values = _broadcast_controls(states, rho, kappa)
    return advance_public_controls(env_cfg, actions, rho_values, kappa_values)


class ReferenceProposal:
    """Weak log-space persistence proposal; no simulator equations."""

    def __init__(self, env_cfg: EnvironmentConfig, sigma: float = 0.18):
        self.cfg = env_cfg
        self.sigma = float(sigma)
        self.actions = resolve_actions(env_cfg)

    def sample_next(self, states, action, contexts, regimes, rng, rho=None, kappa=None):
        states = np.asarray(states, dtype=np.float64)
        actions = np.broadcast_to(np.asarray(action, dtype=int), states.shape)
        if control_fields_enabled(self.cfg):
            next_rho, next_kappa, next_K_eff = _next_public_controls(
                self.cfg, states, actions, rho, kappa
            )
        else:
            next_rho = next_kappa = next_K_eff = None
        next_states = np.empty_like(states)
        for aid in np.unique(actions):
            mask = actions == aid
            spec = self.actions[int(aid)]
            managed = np.maximum(
                states[mask] * (1.0 - spec.harvest_fraction) + spec.stocking_delta,
                0.0,
            )
            scale = np.maximum(managed, 1e-12)
            context = contexts[mask]
            if context.ndim > 1:
                context = context[:, 0]
            drift = 0.02 * np.tanh(context)
            if next_rho is not None and next_kappa is not None and next_K_eff is not None:
                drift = drift + 0.8 * next_rho[mask] + 0.12 * np.tanh(
                    (next_K_eff[mask] - self.cfg.K_base) / max(self.cfg.K_ref, 1.0)
                )
            draw = np.exp(np.log(scale) + drift + rng.normal(0.0, self.sigma, mask.sum()))
            next_states[mask] = np.where(states[mask] == 0.0, 0.0, draw)
        next_regimes = regimes.copy()
        if self.cfg.kind == "regime":
            switch = rng.random(len(states)) > self.cfg.regime_persistence
            next_regimes[switch] = 1 - next_regimes[switch]
        return np.maximum(next_states, 0.0), next_regimes


@dataclass
class LearnedLinearProposal:
    """Public-data action-conditional latent proposal in log-abundance space."""

    env_cfg: EnvironmentConfig
    coefficients: np.ndarray
    residual_sigma: np.ndarray
    uses_controls: bool = False

    @staticmethod
    def _design_dim(uses_controls: bool) -> int:
        return 8 if uses_controls else 5

    @staticmethod
    def _require_dataset_controls(dataset: TrajectoryDataset) -> None:
        missing = [
            key for key in ("rho", "kappa", "K_eff", "next_rho", "next_kappa", "next_K_eff")
            if getattr(dataset, key) is None
        ]
        if missing:
            raise ValueError(
                "cumulative learned proposal requires public control fields: "
                + ", ".join(missing)
            )

    @staticmethod
    def _design_matrix(
        log_state: np.ndarray,
        context: np.ndarray,
        uses_controls: bool,
        control_columns: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        K_ref: float = 500.0,
    ) -> np.ndarray:
        if context.ndim == 1:
            context = context[:, None]
        if context.shape[1] < 3:
            context = np.pad(context, ((0, 0), (0, 3 - context.shape[1])))
        columns = [np.ones(len(log_state)), log_state, context[:, 0], context[:, 1], context[:, 2]]
        if uses_controls:
            if control_columns is None:
                raise ValueError("control columns are required for a cumulative learned proposal")
            next_rho, next_kappa, next_K_eff = control_columns
            columns.extend([
                np.asarray(next_rho, dtype=np.float64),
                np.asarray(next_kappa, dtype=np.float64) / max(K_ref, 1.0),
                np.asarray(next_K_eff, dtype=np.float64) / max(K_ref, 1.0),
            ])
        return np.column_stack(columns)

    @classmethod
    def fit(
        cls,
        dataset: TrajectoryDataset,
        env_cfg: EnvironmentConfig,
        ridge: float = 1e-3,
    ) -> "LearnedLinearProposal":
        eps = 1e-8
        x = np.log(np.maximum(dataset.observations, eps))
        y = np.log(np.maximum(dataset.next_observations, eps))
        uses_controls = control_fields_enabled(env_cfg)
        if uses_controls:
            cls._require_dataset_controls(dataset)
        episode_context = np.zeros((len(dataset), 3), dtype=np.float64)
        for idx in dataset.episode_indices():
            residual = y[idx] - x[idx]
            trend = float(residual[-1] - residual[0]) if len(residual) > 1 else 0.0
            episode_context[idx] = [
                float(np.mean(residual)), float(np.std(residual)), trend
            ]
        dim = cls._design_dim(uses_controls)
        coefs = np.zeros((env_cfg.num_actions, dim), dtype=np.float64)
        sigmas = np.full(env_cfg.num_actions, 0.2, dtype=np.float64)
        for action in range(env_cfg.num_actions):
            mask = dataset.actions == action
            if np.sum(mask) < 4:
                coefs[action, :5] = [0.0, 1.0, 0.0, 0.0, 0.0]
                if uses_controls:
                    coefs[action, 5] = 0.8
                    coefs[action, 6] = 0.12
                continue
            controls = None
            if uses_controls:
                controls = (
                    np.asarray(dataset.next_rho[mask], dtype=np.float64),
                    np.asarray(dataset.next_kappa[mask], dtype=np.float64),
                    np.asarray(dataset.next_K_eff[mask], dtype=np.float64),
                )
            design = cls._design_matrix(
                x[mask], episode_context[mask], uses_controls, controls, env_cfg.K_ref
            )
            lhs = design.T @ design + ridge * np.eye(dim)
            coefs[action] = np.linalg.solve(lhs, design.T @ y[mask])
            residual = y[mask] - design @ coefs[action]
            sigmas[action] = max(float(np.std(residual)), 0.03)
        return cls(env_cfg, coefs, sigmas, uses_controls)

    def sample_next(self, states, action, contexts, regimes, rng, rho=None, kappa=None):
        states = np.asarray(states, dtype=np.float64)
        actions = np.broadcast_to(np.asarray(action, dtype=int), states.shape)
        log_state = np.log(np.maximum(states, 1e-12))
        controls = None
        if self.uses_controls:
            controls = _next_public_controls(self.env_cfg, states, actions, rho, kappa)
        out = np.zeros_like(states)
        for aid in np.unique(actions):
            mask = actions == aid
            control_columns = None
            if controls is not None:
                control_columns = tuple(column[mask] for column in controls)
            design = self._design_matrix(
                log_state[mask],
                contexts[mask],
                self.uses_controls,
                control_columns,
                self.env_cfg.K_ref,
            )
            mean = design @ self.coefficients[int(aid)]
            draw = np.exp(mean + rng.normal(0.0, self.residual_sigma[int(aid)], mask.sum()))
            out[mask] = np.where(states[mask] == 0.0, 0.0, draw)
        next_regimes = regimes.copy()
        if self.env_cfg.kind == "regime":
            switch = rng.random(len(states)) > self.env_cfg.regime_persistence
            next_regimes[switch] = 1 - next_regimes[switch]
        return np.maximum(out, 0.0), next_regimes

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            coefficients=self.coefficients,
            residual_sigma=self.residual_sigma,
            env_json=np.asarray(json.dumps(self.env_cfg.__dict__, sort_keys=True)),
            uses_controls=np.asarray(self.uses_controls),
            proposal_version=np.asarray(2),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedLinearProposal":
        with np.load(path, allow_pickle=False) as data:
            env_cfg = EnvironmentConfig(**json.loads(str(data["env_json"].item())))
            coefficients = data["coefficients"]
            uses_controls = bool(data["uses_controls"].item()) if "uses_controls" in data.files else (
                coefficients.shape[1] > cls._design_dim(False)
            )
            return cls(env_cfg, coefficients, data["residual_sigma"], uses_controls)

    def is_compatible_with(self, env_cfg: EnvironmentConfig) -> bool:
        expected_dim = self._design_dim(control_fields_enabled(env_cfg))
        return (
            self.env_cfg.__dict__ == env_cfg.__dict__
            and self.coefficients.shape[1] == expected_dim
            and self.uses_controls == control_fields_enabled(env_cfg)
        )


class MechanisticProposal:
    """Ricker or true-family proposal used by gate/baseline fidelity ablations."""

    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        family: str = "ricker",
        r_value: float | None = None,
        K_value: float | None = None,
        backend: Backend | None = None,
    ):
        overrides = {**env_cfg.__dict__, "kind": family}
        if env_cfg.control_mode == "real_setpoint":
            # Switching the assumed family also switches which data column the
            # set-point caps come from (Ricker vs LGM) for this known population.
            pop = realdata.pops_for(env_cfg.data_dir or realdata.DATA_DIR)[
                env_cfg.population
            ]
            _r_base, r_min, r_max = pop.caps(family)
            overrides["r_min"] = float(r_min)
            overrides["r_max"] = float(r_max)
        if K_value is not None:
            overrides["K_base"] = float(K_value)
        self.cfg = EnvironmentConfig(**overrides)
        self.family = family
        self.r_value = r_value
        self.K_value = K_value
        self._env = ContinuousEcologyEnv(self.cfg)
        self.actions = resolve_actions(self.cfg)
        self.backend = backend or get_active_backend()
        # Per-action lookups for the vectorized map (host arrays moved to device).
        self._delta_r_dev, self._delta_K_dev, self._stock_dev = real_action_lookups(
            self.backend, self.actions
        )

    def _vectorized_next(self, states, actions, C, theta, regimes, rho, kappa):
        return real_next_states(
            self.cfg, self.backend, self._delta_r_dev, self._delta_K_dev,
            self._stock_dev, states, actions, C, theta, regimes, rho, kappa,
        )

    def sample_next(self, states, action, contexts, regimes, rng, rho=None, kappa=None):
        states = np.asarray(states, dtype=np.float64)
        actions = np.broadcast_to(np.asarray(action, dtype=int), states.shape)
        context = contexts
        if context.ndim == 1:
            context = np.column_stack([context, context, context])
        context_u = 1.0 / (1.0 + np.exp(-context))
        C = self.cfg.C_low + context_u[:, 1] * (self.cfg.C_high - self.cfg.C_low)
        theta = self.cfg.theta_low + context_u[:, 2] * (self.cfg.theta_high - self.cfg.theta_low)
        rho_values = np.zeros_like(states) if rho is None else np.broadcast_to(
            np.asarray(rho, dtype=np.float64), states.shape
        )
        kappa_values = np.zeros_like(states) if kappa is None else np.broadcast_to(
            np.asarray(kappa, dtype=np.float64), states.shape
        )
        if self.cfg.control_mode == "real_setpoint":
            # Vectorized backend path (CPU NumPy reference / GPU CuPy).  This is the
            # PLUS/MOOR/gate hot path; the per-particle Python loop below is kept as
            # the reference for the (unused-here) non-real control modes.
            out = self._vectorized_next(
                states, actions, C, theta, np.asarray(regimes), rho_values, kappa_values
            )
        else:
            r = (
                np.full(len(states), self.r_value)
                if self.r_value is not None
                else self.cfg.r_base_low
                + context_u[:, 0] * (self.cfg.r_base_high - self.cfg.r_base_low)
            )
            out = np.empty_like(states)
            for i, (s, aid) in enumerate(zip(states, actions)):
                out[i] = self._env.transition_value(
                    float(s), self.actions[int(aid)], float(r[i]), float(C[i]),
                    float(theta[i]), int(regimes[i]), 0.0,
                    float(rho_values[i]), float(kappa_values[i]),
                )
        next_regimes = regimes.copy()
        if self.family == "regime":
            switch = rng.random(len(states)) > self.cfg.regime_persistence
            next_regimes[switch] = 1 - next_regimes[switch]
        return out, next_regimes


class ParticleFilter:
    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        filter_cfg: FilterConfig,
        proposal: TransitionProposal,
    ):
        self.env_cfg = env_cfg
        self.cfg = filter_cfg
        self.proposal = proposal
        self.observation_model = LogNormalObservationModel(env_cfg.observation_noise_sigma)
        self._rng = np.random.default_rng(0)

    def _initial_particles(self, n: int, rng: np.random.Generator) -> np.ndarray:
        low = rng.random(n) < self.env_cfg.low_start_probability
        if self.env_cfg.control_mode == "real_setpoint":
            # Real episodes start at the population's N0; centre the prior there
            # (not at K_base, which can be ~8x N0) with a spread that covers the
            # survey-noise scale, then let the emission re-weight.
            sigma = max(
                self.env_cfg.initial_log_sigma,
                self.env_cfg.observation_noise_sigma,
                0.3,
            )
            states = rng.lognormal(np.log(max(self.env_cfg.N0, 1.0)), sigma, n)
            s_safe = self.env_cfg.safety_threshold
            states[low] = rng.uniform(
                0.5 * s_safe, max(1.5 * s_safe, self.env_cfg.N0), int(np.sum(low))
            )
            return states
        states = rng.lognormal(
            np.log(self.env_cfg.K_base), self.env_cfg.initial_log_sigma, n
        )
        states[low] = rng.uniform(25.0, 150.0, np.sum(low))
        return states

    def _normalize(self, log_weights: np.ndarray) -> np.ndarray:
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            return np.full(len(log_weights), -np.log(len(log_weights)))
        m = np.max(log_weights[finite])
        total = np.sum(np.exp(log_weights[finite] - m))
        normalized = np.full(len(log_weights), -np.inf)
        normalized[finite] = log_weights[finite] - m - np.log(total)
        return normalized

    def reset(self, observation: float, seed: int) -> BeliefState:
        self._rng = np.random.default_rng(seed)
        n = self.cfg.particles
        if self.env_cfg.observation_noise_sigma == 0.0:
            states = np.full(n, observation, dtype=np.float64)
        else:
            states = self._initial_particles(n, self._rng)
        contexts = self._rng.normal(0.0, 1.0, (n, 3))
        regimes = self._rng.integers(0, 2, n, dtype=np.int8)
        log_weights = self._normalize(self.observation_model.log_prob(observation, states))
        kwargs = {}
        if control_fields_enabled(self.env_cfg):
            controls = initial_controls(self.env_cfg)
            kwargs = {"rho": controls.rho, "kappa": controls.kappa, "K_eff": controls.K_eff}
        belief = BeliefState(states, contexts, regimes, log_weights, observation, 0, **kwargs)
        return self._resample_if_needed(belief)

    def _recover(self, observation: float, n: int) -> np.ndarray:
        if observation == 0.0:
            return np.zeros(n)
        sigma = max(self.env_cfg.observation_noise_sigma, 0.05)
        return np.exp(np.log(observation) + self._rng.normal(0.0, sigma, n))

    def _resample_if_needed(self, belief: BeliefState) -> BeliefState:
        n = len(belief.states)
        belief.diagnostics["ess_before"] = belief.ess
        if belief.ess >= self.cfg.ess_fraction * n:
            return belief
        idx = _systematic_resample(belief.weights, self._rng)
        belief.states = belief.states[idx]
        belief.contexts = belief.contexts[idx] + self._rng.normal(
            0.0, self.cfg.context_rejuvenation, belief.contexts[idx].shape
        )
        belief.regimes = belief.regimes[idx]
        belief.log_weights = np.full(n, -np.log(n))
        belief.diagnostics["resampled"] = 1.0
        return belief

    def update(self, belief: BeliefState, action: int, observation: float) -> BeliefState:
        rho = belief.rho if belief.rho is not None else None
        kappa = belief.kappa if belief.kappa is not None else None
        states, regimes = self.proposal.sample_next(
            belief.states,
            action,
            belief.contexts,
            belief.regimes,
            self._rng,
            rho,
            kappa,
        )
        control_kwargs = {}
        if control_fields_enabled(self.env_cfg):
            next_rho, next_kappa, next_K_eff = advance_public_controls(
                self.env_cfg,
                action,
                0.0 if rho is None else rho,
                0.0 if kappa is None else kappa,
            )
            control_kwargs = {
                "rho": float(next_rho),
                "kappa": float(next_kappa),
                "K_eff": float(next_K_eff),
            }
        if self.env_cfg.observation_noise_sigma == 0.0:
            states = np.full(len(states), observation, dtype=np.float64)
            return BeliefState(
                states=states,
                contexts=belief.contexts.copy(),
                regimes=regimes,
                log_weights=np.full(len(states), -np.log(len(states))),
                observation=float(observation),
                timestep=belief.timestep + 1,
                diagnostics={"exact_observation": 1.0},
                **control_kwargs,
            )
        likelihood = self.observation_model.log_prob(observation, states)
        if not np.any(np.isfinite(likelihood)):
            states = self._recover(observation, len(states))
            likelihood = self.observation_model.log_prob(observation, states)
        log_weights = self._normalize(belief.log_weights + likelihood)
        updated = BeliefState(
            states=states,
            contexts=belief.contexts.copy(),
            regimes=regimes,
            log_weights=log_weights,
            observation=float(observation),
            timestep=belief.timestep + 1,
            **control_kwargs,
        )
        return self._resample_if_needed(updated)


class RawObservationFilter(ParticleFilter):
    """Raw-observation ablation with the same BeliefState interface."""

    def reset(self, observation: float, seed: int) -> BeliefState:
        self._rng = np.random.default_rng(seed)
        n = self.cfg.particles
        kwargs = {}
        if control_fields_enabled(self.env_cfg):
            controls = initial_controls(self.env_cfg)
            kwargs = {"rho": controls.rho, "kappa": controls.kappa, "K_eff": controls.K_eff}
        return BeliefState(
            np.full(n, observation), np.zeros((n, 3)), np.zeros(n, dtype=np.int8),
            np.full(n, -np.log(n)), observation, 0, **kwargs,
        )

    def update(self, belief: BeliefState, action: int, observation: float) -> BeliefState:
        updated = self.reset(observation, int(self._rng.integers(0, 2**31 - 1)))
        if control_fields_enabled(self.env_cfg):
            next_rho, next_kappa, next_K_eff = advance_public_controls(
                self.env_cfg,
                action,
                0.0 if belief.rho is None else belief.rho,
                0.0 if belief.kappa is None else belief.kappa,
            )
            updated.rho = float(next_rho)
            updated.kappa = float(next_kappa)
            updated.K_eff = float(next_K_eff)
            updated.timestep = belief.timestep + 1
        return updated


class OracleStateFilter(RawObservationFilter):
    def set_true_state(
        self,
        state: float,
        timestep: int = 0,
        public_info: dict[str, object] | None = None,
    ) -> BeliefState:
        n = self.cfg.particles
        kwargs = {}
        if control_fields_enabled(self.env_cfg):
            if public_info and "rho" in public_info:
                kwargs = {
                    "rho": float(public_info["rho"]),
                    "kappa": float(public_info["kappa"]),
                    "K_eff": float(public_info["K_eff"]),
                }
            else:
                controls = initial_controls(self.env_cfg)
                kwargs = {"rho": controls.rho, "kappa": controls.kappa, "K_eff": controls.K_eff}
        return BeliefState(
            np.full(n, state), np.zeros((n, 3)), np.zeros(n, dtype=np.int8),
            np.full(n, -np.log(n)), state, timestep, **kwargs,
        )


@dataclass
class BeliefCache:
    features: np.ndarray
    next_features: np.ndarray
    mean_states: np.ndarray
    next_mean_states: np.ndarray
    metadata: dict[str, object]
    rho: np.ndarray | None = None
    kappa: np.ndarray | None = None
    K_eff: np.ndarray | None = None
    next_rho: np.ndarray | None = None
    next_kappa: np.ndarray | None = None
    next_K_eff: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": self.features,
            "next_features": self.next_features,
            "mean_states": self.mean_states,
            "next_mean_states": self.next_mean_states,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for key in ("rho", "kappa", "K_eff", "next_rho", "next_kappa", "next_K_eff"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        np.savez_compressed(target, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "BeliefCache":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["features"], data["next_features"], data["mean_states"],
                data["next_mean_states"], json.loads(str(data["metadata_json"].item())),
                **{
                    key: np.asarray(data[key]) if key in data.files else None
                    for key in ("rho", "kappa", "K_eff", "next_rho", "next_kappa", "next_K_eff")
                },
            )


def cache_dataset_beliefs(
    dataset: TrajectoryDataset,
    filter_factory: Callable[[], ParticleFilter],
    seed: int,
    K_ref: float,
    safety_threshold: float,
) -> BeliefCache:
    features: list[np.ndarray] = []
    next_features: list[np.ndarray] = []
    states: list[float] = []
    next_states: list[float] = []
    rho: list[float] = []
    kappa: list[float] = []
    K_eff: list[float] = []
    next_rho: list[float] = []
    next_kappa: list[float] = []
    next_K_eff: list[float] = []
    for episode, idx in enumerate(dataset.episode_indices()):
        filt = filter_factory()
        cumulative_controls = control_fields_enabled(filt.env_cfg)
        belief = filt.reset(float(dataset.observations[idx[0]]), seed + episode)
        for i in idx:
            features.append(belief.features(K_ref, safety_threshold))
            states.append(belief.mean_state())
            if cumulative_controls:
                rho.append(float(belief.rho))
                kappa.append(float(belief.kappa))
                K_eff.append(float(belief.K_eff))
            updated = filt.update(
                belief, int(dataset.actions[i]), float(dataset.next_observations[i])
            )
            next_features.append(updated.features(K_ref, safety_threshold))
            next_states.append(updated.mean_state())
            if cumulative_controls:
                next_rho.append(float(updated.rho))
                next_kappa.append(float(updated.kappa))
                next_K_eff.append(float(updated.K_eff))
            belief = updated
    metadata = {"seed": seed, "transitions": len(dataset), "feature_dim": len(features[0])}
    payload = {}
    if rho:
        payload = {
            "rho": np.asarray(rho, dtype=np.float64),
            "kappa": np.asarray(kappa, dtype=np.float64),
            "K_eff": np.asarray(K_eff, dtype=np.float64),
            "next_rho": np.asarray(next_rho, dtype=np.float64),
            "next_kappa": np.asarray(next_kappa, dtype=np.float64),
            "next_K_eff": np.asarray(next_K_eff, dtype=np.float64),
        }
        metadata["control_fields"] = True
    return BeliefCache(
        np.vstack(features), np.vstack(next_features), np.asarray(states),
        np.asarray(next_states), metadata, **payload,
    )
