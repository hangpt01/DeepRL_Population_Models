"""Reward channels.

Real-ecology reward (spec: 29_6 "Reward: two state-dependent settings" / E6):
the reward is computed by the simulator from the **true next state** ``s_{t+1}``
(not the survey ``o_t``) and logged, so every method consumes identical
``(o_t, a_t, R_t, o_{t+1})``.  Two settings share the same benefit and cost and
differ only in the collapse penalty weight ``P_m``:
``reward_mode="yield"`` -> ``P=0`` (baseline-style, matches PLUS/MOOR),
``reward_mode="safe"``  -> ``P=collapse_penalty`` (collapse-aware).  The
penalty indicator is configured separately: default real cells charge below-safe
occupancy, while ``safety_penalty_mode="crossing"`` preserves the old event-only
ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .actions import ActionSpec, resolve_actions

if TYPE_CHECKING:  # cfg is duck-typed at runtime; import only for type hints
    from .config import EnvironmentConfig


@dataclass(frozen=True)
class ContinuousReward:
    actions: tuple[ActionSpec, ...]
    alpha: float = 1.0
    K_ref: float = 500.0
    collapse_penalty: float = 20.0

    def utility(self, abundance: float | np.ndarray):
        x = np.asarray(abundance, dtype=np.float64)
        return self.alpha * x / (x + self.K_ref)

    def operational(self, observation: float, action: int, safety_penalty: bool) -> float:
        value = float(self.utility(observation)) - self.actions[action].cost
        if safety_penalty:
            value -= self.collapse_penalty
        return float(value)

    def true(self, state: float, action: int, safety_penalty: bool) -> float:
        value = float(self.utility(state)) - self.actions[action].cost
        if safety_penalty:
            value -= self.collapse_penalty
        return float(value)

    def state_reward(self, next_state: float, action: int, safety_penalty: bool) -> float:
        """Real-ecology reward on the true next state ``s_{t+1}`` (spec Eq. reward).

        ``R = alpha * s'/(s'+K_ref) - cost(a) - collapse_penalty * indicator``.
        The collapse-penalty weight is already baked into ``collapse_penalty``
        (0 under ``reward_mode="yield"``; see :func:`build_reward`).
        """

        value = float(self.utility(next_state)) - self.actions[action].cost
        if safety_penalty:
            value -= self.collapse_penalty
        return float(value)

    def expected(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        safety_penalty: np.ndarray,
    ) -> np.ndarray:
        costs = np.asarray([self.actions[int(a)].cost for a in actions])
        return self.utility(observations) - costs - self.collapse_penalty * safety_penalty


def safety_penalty_indicator(
    cfg: "EnvironmentConfig",
    previous_state: float | np.ndarray,
    next_state: float | np.ndarray,
    crossing: bool | np.ndarray | None = None,
) -> bool | np.ndarray:
    """Return the configured true-state safety-penalty indicator.

    ``occupancy`` charges every step whose next true state is below the
    population safety floor; this is the real-setting default because depleted
    sink populations can start below ``s_safe``.  ``crossing`` preserves the old
    downward-crossing event convention.
    """

    if cfg.safety_penalty_mode == "occupancy":
        return np.asarray(next_state) <= cfg.safety_threshold
    if crossing is not None:
        return crossing
    return (np.asarray(previous_state) > cfg.safety_threshold) & (
        np.asarray(next_state) <= cfg.safety_threshold
    )


def effective_collapse_penalty(cfg: "EnvironmentConfig") -> float:
    """Collapse-penalty weight for the cell: 0 under real ``yield``, else configured.

    ``reward_mode="yield"`` leaves collapse implicit (discounting + absorbing
    ``s=0``); ``"safe"`` uses the configured true-state safety indicator with
    ``collapse_penalty``.
    Non-real control modes keep the configured penalty unchanged.
    """

    if cfg.control_mode == "real_setpoint" and cfg.reward_mode == "yield":
        return 0.0
    return cfg.collapse_penalty


def build_reward(cfg: "EnvironmentConfig") -> ContinuousReward:
    """Single source for a cell's reward model (actions + mode-aware penalty)."""

    return ContinuousReward(
        resolve_actions(cfg), cfg.alpha, cfg.K_ref, effective_collapse_penalty(cfg)
    )
