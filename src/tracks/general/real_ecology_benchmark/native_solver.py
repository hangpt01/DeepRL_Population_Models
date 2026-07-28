"""Tabular native solver machinery for discrete ecological baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import resolve_actions
from .config import (
    EnvironmentConfig,
    dummy_environment_like,
    is_setpoint_cumulative,
    real_environment_like,
    uses_dummy_data,
    uses_real_data,
)
from .controls import initial_controls
from .discretize import NativeGrid, build_native_grid
from .envs import ContinuousEcologyEnv
from .reward import build_reward, safety_penalty_indicator


# The mechanistic hypothesis class a native ecological solver can enumerate.
# MOOR-native commits to a single fitted Ricker model; PLUS-native carries a
# posterior over all four forms (the "candidate mechanistic models" of the
# adaptive-management literature).
NATIVE_FAMILIES = ("ricker", "allee", "theta", "regime")

# Only the regime-switching form has a hidden discrete mode; the others are
# fully determined by abundance and the public controls.
_FAMILY_REGIMES = {"ricker": 1, "allee": 1, "theta": 1, "regime": 2}


def assumption_config(
    cfg: EnvironmentConfig,
    family: str = "ricker",
    assumed_K_base: float | None = None,
) -> EnvironmentConfig:
    """Return a ``family``-form config carrying the current cell's public settings.

    Re-derives the per-population set-point caps for the assumed family (Ricker
    columns vs LGM columns), so a candidate model is scaled the way an ecologist
    fitting that form to this population's data would scale it.
    """

    if family not in NATIVE_FAMILIES:
        raise ValueError(f"unknown native family {family!r}; choose from {NATIVE_FAMILIES}")
    if is_setpoint_cumulative(cfg):
        if uses_real_data(cfg):
            assumed = real_environment_like(cfg, cfg.population, family)
        elif uses_dummy_data(cfg):
            assumed = dummy_environment_like(cfg, cfg.population, family)
        else:
            assumed = EnvironmentConfig(**{**cfg.__dict__, "kind": family})
    else:
        assumed = EnvironmentConfig(**{**cfg.__dict__, "kind": family})
    if assumed_K_base is not None:
        assumed = EnvironmentConfig(**{**assumed.__dict__, "K_base": float(assumed_K_base)})
    assumed.validate()
    return assumed


def ricker_assumption_config(
    cfg: EnvironmentConfig,
    assumed_K_base: float | None = None,
) -> EnvironmentConfig:
    """Return a Ricker-form config carrying the current cell's public settings."""

    return assumption_config(cfg, "ricker", assumed_K_base)


def predict_ricker_next(
    cfg: EnvironmentConfig,
    states: np.ndarray,
    actions: np.ndarray,
    kappa: np.ndarray | float | None = None,
    assumed_K_base: float | None = None,
) -> np.ndarray:
    """Predict next abundance under the native Ricker assumption."""

    assumed = ricker_assumption_config(cfg, assumed_K_base)
    env = ContinuousEcologyEnv(assumed)
    specs = resolve_actions(assumed)
    states = np.asarray(states, dtype=np.float64)
    action_ids = np.broadcast_to(np.asarray(actions, dtype=int), states.shape)
    if kappa is None:
        kappa_values = np.zeros_like(states, dtype=np.float64)
    else:
        kappa_values = np.broadcast_to(np.asarray(kappa, dtype=np.float64), states.shape)
    rho0 = float(initial_controls(assumed).rho)
    out = np.empty_like(states, dtype=np.float64)
    for i, (state, action_id, kappa_value) in enumerate(zip(states, action_ids, kappa_values)):
        out[i] = env.transition_value(
            float(state),
            specs[int(action_id)],
            0.0,
            assumed.C_low,
            assumed.theta_low,
            0,
            0.0,
            rho0,
            float(kappa_value),
        )
    return out


@dataclass
class NativeSolver:
    """QMDP policy over a discrete ``(abundance, regime)`` belief and public K level."""

    env_cfg: EnvironmentConfig
    assumed_cfg: EnvironmentConfig
    grid: NativeGrid
    transition: np.ndarray
    reward: np.ndarray
    q_values: np.ndarray
    next_kappa: np.ndarray

    @classmethod
    def build(
        cls,
        env_cfg: EnvironmentConfig,
        *,
        assumed_family: str = "ricker",
        assumed_K_base: float | None = None,
        state_bins: int = 51,
        discount: float = 0.95,
        iterations: int = 250,
        tolerance: float = 1e-7,
    ) -> "NativeSolver":
        assumed = assumption_config(env_cfg, assumed_family, assumed_K_base)
        grid = build_native_grid(env_cfg, state_bins, _FAMILY_REGIMES[assumed_family])
        transition, reward, next_kappa = _build_tables(env_cfg, assumed, grid)
        q = _value_iteration(transition, reward, next_kappa, discount, iterations, tolerance)
        return cls(env_cfg, assumed, grid, transition, reward, q, next_kappa)

    @property
    def family(self) -> str:
        return str(self.assumed_cfg.kind)

    def kappa_index(self, kappa: float | None) -> int:
        return self.grid.kappa_index(kappa)

    def initial_log_weights(self, observation: float) -> np.ndarray:
        return self.grid.initial_log_weights(float(observation))

    def action_values(self, log_weights: np.ndarray, kappa: float | None) -> np.ndarray:
        weights = np.exp(self.grid.normalize_log_weights(log_weights))
        k_idx = self.kappa_index(kappa)
        return weights @ self.q_values[k_idx]

    def act(self, log_weights: np.ndarray, kappa: float | None) -> int:
        return int(np.argmax(self.action_values(log_weights, kappa)))

    def update_log_weights(
        self,
        log_weights: np.ndarray,
        kappa: float | None,
        action: int,
        observation: float,
    ) -> tuple[np.ndarray, float, float]:
        """Discrete Bayes update on ``(a, o)`` only.

        Returns ``(log_weights, next_kappa, evidence)``.  ``evidence`` is the log
        marginal likelihood of ``observation`` under this candidate model, which
        is what PLUS-native's model posterior is scored on; it is comparable
        across candidates because each belief is normalized within its own model.
        """

        k_idx = self.kappa_index(kappa)
        action = int(action)
        next_kappa_value = float(self.grid.kappa_values[self.next_kappa[k_idx, action]])
        weights = np.exp(self.grid.normalize_log_weights(log_weights))
        predicted = weights @ self.transition[k_idx, action]
        ll = self.grid.hidden_log_likelihood(float(observation))
        values = np.log(np.maximum(predicted, 1e-300)) + ll
        finite = np.isfinite(values)
        if not np.any(finite):
            recovered = self.grid.hidden_point_mass(float(observation))
            return recovered, next_kappa_value, -1e12
        m = float(np.max(values[finite]))
        evidence = m + float(np.log(np.sum(np.exp(values[finite] - m))))
        out = np.full(self.grid.num_hidden, -np.inf, dtype=np.float64)
        out[finite] = values[finite] - evidence
        return out, next_kappa_value, evidence


def _build_tables(
    env_cfg: EnvironmentConfig,
    assumed_cfg: EnvironmentConfig,
    grid: NativeGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tabulate ``T[k, a, z, z']`` and ``R[k, z, a]`` for one candidate model.

    ``z = (abundance bin, regime)``.  The candidate's nuisance biology (the Allee
    threshold ``C``, the theta exponent) is pinned at the midpoint of the cell's
    published range: a point estimate is what a native solver fitting this form
    would use, and it keeps the candidate honest without handing it the truth.
    """

    actions = resolve_actions(assumed_cfg)
    reward_model = build_reward(assumed_cfg)
    regime_kernel = grid.regime_kernel()
    num_k = grid.num_kappa
    num_s = grid.num_states
    num_g = grid.num_regimes
    num_z = grid.num_hidden
    num_a = len(actions)
    transition = np.zeros((num_k, num_a, num_z, num_z), dtype=np.float64)
    reward = np.zeros((num_k, num_z, num_a), dtype=np.float64)
    next_kappa = np.zeros((num_k, num_a), dtype=np.int16)
    env = ContinuousEcologyEnv(assumed_cfg)
    rho0 = float(initial_controls(assumed_cfg).rho)
    C = 0.5 * (float(assumed_cfg.C_low) + float(assumed_cfg.C_high))
    theta = 0.5 * (float(assumed_cfg.theta_low) + float(assumed_cfg.theta_high))
    abundance_row = np.zeros(num_s, dtype=np.float64)
    for k_idx, kappa in enumerate(grid.kappa_values):
        for action_id, spec in enumerate(actions):
            next_kappa[k_idx, action_id] = grid.next_kappa_index(k_idx, action_id)
            for g_idx, regime in enumerate(grid.regime_values):
                for s_idx, state in enumerate(grid.state_values):
                    next_state = env.transition_value(
                        float(state),
                        spec,
                        0.0,
                        C,
                        theta,
                        int(regime),
                        0.0,
                        rho0,
                        float(kappa),
                    )
                    abundance_row.fill(0.0)
                    _add_barycentric_mass(abundance_row, grid, next_state)
                    z = g_idx * num_s + s_idx
                    for gp_idx in range(num_g):
                        probability = float(regime_kernel[g_idx, gp_idx])
                        if probability <= 0.0:
                            continue
                        block = slice(gp_idx * num_s, (gp_idx + 1) * num_s)
                        transition[k_idx, action_id, z, block] = probability * abundance_row
                    crossing = (
                        float(state) > env_cfg.safety_threshold
                        and float(next_state) <= env_cfg.safety_threshold
                    )
                    penalty = bool(
                        safety_penalty_indicator(
                            env_cfg, float(state), float(next_state), crossing
                        )
                    )
                    reward[k_idx, z, action_id] = reward_model.state_reward(
                        float(next_state), action_id, penalty
                    )
    return transition, reward, next_kappa


def _add_barycentric_mass(row: np.ndarray, grid: NativeGrid, value: float) -> None:
    states = grid.state_values
    if value <= states[0]:
        row[0] = 1.0
        return
    if value >= states[-1]:
        row[-1] = 1.0
        return
    right = int(np.searchsorted(states, value, side="left"))
    if np.isclose(value, states[right]):
        row[right] = 1.0
        return
    left = right - 1
    span = float(states[right] - states[left])
    if span <= 0.0:
        row[right] = 1.0
        return
    right_weight = float((value - states[left]) / span)
    row[left] = 1.0 - right_weight
    row[right] = right_weight


def _value_iteration(
    transition: np.ndarray,
    reward: np.ndarray,
    next_kappa: np.ndarray,
    discount: float,
    iterations: int,
    tolerance: float,
) -> np.ndarray:
    num_k, num_a, num_s, _ = transition.shape
    value = np.zeros((num_k, num_s), dtype=np.float64)
    q_values = np.zeros((num_k, num_s, num_a), dtype=np.float64)
    for _ in range(max(int(iterations), 1)):
        previous = value.copy()
        for k_idx in range(num_k):
            for action in range(num_a):
                nk = int(next_kappa[k_idx, action])
                q_values[k_idx, :, action] = (
                    reward[k_idx, :, action]
                    + discount * (transition[k_idx, action] @ previous[nk])
                )
        value = np.max(q_values, axis=2)
        if float(np.max(np.abs(value - previous))) < float(tolerance):
            break
    return q_values
