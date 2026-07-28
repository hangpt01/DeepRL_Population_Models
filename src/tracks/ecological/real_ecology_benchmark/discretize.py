"""Discrete grids for native ecological baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import resolve_actions
from .config import EnvironmentConfig
from .observation import LogNormalObservationModel


def _unique_sorted(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    arr = np.unique(np.round(arr, 12))
    return np.sort(arr.astype(np.float64))


def reachable_kappa_levels(cfg: EnvironmentConfig) -> np.ndarray:
    """Return public kappa levels, normalized by ``K_base``.

    Real capacity actions are multipliers (0.1 and 0.3 of K_base in the shipped
    table), not absolute increments.  Build the public lattice from the action
    table so non-250 populations use the right scale.
    """

    actions = resolve_actions(cfg)
    scale = max(float(cfg.K_base), 1.0)
    increments = _unique_sorted([max(0.0, a.delta_K / scale) for a in actions])
    levels = {0.0}
    frontier = {0.0}
    for _ in range(64):
        new_frontier: set[float] = set()
        for level in frontier:
            for inc in increments:
                nxt = min(1.0, round(level + float(inc), 10))
                if nxt not in levels:
                    levels.add(nxt)
                    new_frontier.add(nxt)
        if not new_frontier:
            break
        frontier = new_frontier
    return np.asarray(sorted(levels), dtype=np.float64)


@dataclass(frozen=True)
class NativeGrid:
    """Discrete hidden state for a native candidate model.

    The hidden state is ``(abundance, regime)``.  Only the regime-switching
    candidate carries a non-trivial regime dimension; every other mechanistic
    form has ``num_regimes == 1`` and the hidden index collapses to the abundance
    bin.  Hidden indices are regime-major: ``z = regime_index * num_states + state_index``.
    """

    env_cfg: EnvironmentConfig
    state_values: np.ndarray
    kappa_values: np.ndarray
    kappa_normalized: np.ndarray
    regime_values: np.ndarray

    @property
    def num_states(self) -> int:
        return int(len(self.state_values))

    @property
    def num_regimes(self) -> int:
        return int(len(self.regime_values))

    @property
    def num_hidden(self) -> int:
        return self.num_states * self.num_regimes

    @property
    def hidden_state_values(self) -> np.ndarray:
        """Abundance carried by each hidden index."""

        return np.tile(self.state_values, self.num_regimes)

    @property
    def hidden_regime_values(self) -> np.ndarray:
        """Regime label carried by each hidden index."""

        return np.repeat(self.regime_values, self.num_states)

    @property
    def num_kappa(self) -> int:
        return int(len(self.kappa_values))

    def regime_kernel(self) -> np.ndarray:
        """Markov kernel over the hidden regime.

        Mirrors ``ContinuousEcologyEnv.step``: the regime flips with probability
        ``1 - regime_persistence`` after each transition.  Degenerates to ``[[1]]``
        for the single-regime candidate models.
        """

        n = self.num_regimes
        if n == 1:
            return np.ones((1, 1), dtype=np.float64)
        stay = float(self.env_cfg.regime_persistence)
        kernel = np.full((n, n), (1.0 - stay) / max(n - 1, 1), dtype=np.float64)
        np.fill_diagonal(kernel, stay)
        return kernel

    def state_index(self, value: float | np.ndarray) -> int | np.ndarray:
        values = np.asarray(value, dtype=np.float64)
        idx = np.searchsorted(self.state_values, values)
        idx = np.clip(idx, 0, len(self.state_values) - 1)
        left = np.maximum(idx - 1, 0)
        use_left = (
            np.abs(values - self.state_values[left])
            <= np.abs(values - self.state_values[idx])
        )
        result = np.where(use_left, left, idx)
        if np.ndim(value) == 0:
            return int(result)
        return result.astype(int)

    def kappa_index(self, kappa: float | None) -> int:
        value = 0.0 if kappa is None else float(kappa)
        return int(np.argmin(np.abs(self.kappa_values - value)))

    def next_kappa_index(self, kappa_index: int, action: int) -> int:
        actions = resolve_actions(self.env_cfg)
        current = float(self.kappa_values[int(kappa_index)])
        nxt = np.clip(
            current + float(actions[int(action)].delta_K),
            self.env_cfg.K_min - self.env_cfg.K_base,
            self.env_cfg.K_max - self.env_cfg.K_base,
        )
        return self.kappa_index(float(nxt))

    def log_likelihood(self, observation: float) -> np.ndarray:
        """Observation log likelihood on the state grid.

        At sigma=0 the observation kernel is a point mass.  On a grid, route the
        point mass to the nearest bin instead of evaluating a density that would
        be all -inf unless a grid centre exactly matched the observation.
        """

        obs = float(observation)
        out = np.full(self.num_states, -np.inf, dtype=np.float64)
        if self.env_cfg.observation_noise_sigma == 0.0:
            out[self.state_index(obs)] = 0.0
            return out
        return LogNormalObservationModel(self.env_cfg.observation_noise_sigma).log_prob(
            obs, self.state_values
        )

    def hidden_log_likelihood(self, observation: float) -> np.ndarray:
        """Observation log likelihood on the hidden ``(abundance, regime)`` grid.

        The survey observes abundance only, so the likelihood is constant across
        the regime dimension.
        """

        return np.tile(self.log_likelihood(observation), self.num_regimes)

    def hidden_point_mass(self, observation: float) -> np.ndarray:
        """Degenerate log-belief on the observed abundance bin, flat over regimes."""

        out = np.full(self.num_hidden, -np.inf, dtype=np.float64)
        s_idx = int(self.state_index(float(observation)))
        for g_idx in range(self.num_regimes):
            out[g_idx * self.num_states + s_idx] = 0.0
        return self.normalize_log_weights(out)

    @staticmethod
    def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
        values = np.asarray(log_weights, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.full(len(values), -np.log(len(values)), dtype=np.float64)
        m = float(np.max(values[finite]))
        total = float(np.sum(np.exp(values[finite] - m)))
        if total <= 0.0 or not np.isfinite(total):
            return np.full(len(values), -np.log(len(values)), dtype=np.float64)
        out = np.full(len(values), -np.inf, dtype=np.float64)
        out[finite] = values[finite] - m - np.log(total)
        return out

    def initial_log_weights(self, observation: float) -> np.ndarray:
        """Prior belief at episode start: survey likelihood, flat over regimes."""

        return self.normalize_log_weights(self.hidden_log_likelihood(observation))


def build_native_grid(
    env_cfg: EnvironmentConfig,
    state_bins: int = 51,
    num_regimes: int = 1,
) -> NativeGrid:
    """Build a per-population abundance grid, public K lattice, and regime axis."""

    n = max(int(state_bins), 12)
    core = np.linspace(0.0, float(env_cfg.K_max), n, dtype=np.float64)
    step = float(core[1] - core[0]) if len(core) > 1 else max(float(env_cfg.K_max), 1.0)
    anchors = [
        0.0,
        float(env_cfg.safety_threshold),
        float(env_cfg.mvp_threshold),
        float(env_cfg.N0),
        float(env_cfg.K_base),
        float(env_cfg.K_max),
        float(env_cfg.K_max + step),
    ]
    states = _unique_sorted(np.concatenate([core, np.asarray(anchors, dtype=np.float64)]))
    states[0] = 0.0
    k_norm = reachable_kappa_levels(env_cfg)
    kappa = k_norm * float(env_cfg.K_base)
    regimes = np.arange(max(int(num_regimes), 1), dtype=np.int64)
    return NativeGrid(env_cfg, states, kappa, k_norm, regimes)
