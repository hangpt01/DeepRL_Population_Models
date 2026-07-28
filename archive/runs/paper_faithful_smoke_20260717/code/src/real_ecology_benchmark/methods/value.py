"""Continuous fitted-value helpers for mechanistic baselines."""

from __future__ import annotations

import numpy as np

from ..actions import resolve_actions
from ..beliefs import MechanisticProposal
from ..config import EnvironmentConfig
from ..reward import ContinuousReward, safety_penalty_indicator


def value_features(states: np.ndarray, K_ref: float) -> np.ndarray:
    x = np.log1p(np.maximum(states, 0.0) / K_ref)
    return np.column_stack([np.ones(len(states)), x, x * x, x * x * x])


def fit_mechanistic_q(
    proposal: MechanisticProposal,
    support_states: np.ndarray,
    env_cfg: EnvironmentConfig,
    reward: ContinuousReward,
    discount: float,
    ridge: float,
    iterations: int = 40,
) -> np.ndarray:
    states = np.asarray(support_states, dtype=np.float64)
    upper = max(float(np.quantile(states, 0.995)), 1000.0)
    states = np.unique(np.concatenate([
        states,
        np.geomspace(1.0, upper, 256),
        np.asarray([0.0, env_cfg.safety_threshold, env_cfg.K_base]),
    ]))
    X = value_features(states, env_cfg.K_ref)
    q_weights = np.zeros((env_cfg.num_actions, X.shape[1]))
    contexts = np.zeros((len(states), 3))
    regimes = np.zeros(len(states), dtype=np.int8)
    rng = np.random.default_rng(12345)
    specs = resolve_actions(env_cfg)
    for _ in range(iterations):
        for action in range(env_cfg.num_actions):
            next_states, _ = proposal.sample_next(states, action, contexts, regimes, rng)
            next_q = value_features(next_states, env_cfg.K_ref) @ q_weights.T
            crossing = (states > env_cfg.safety_threshold) & (
                next_states <= env_cfg.safety_threshold
            )
            penalty_indicator = safety_penalty_indicator(env_cfg, states, next_states, crossing)
            # Benefit on the TRUE next state s_{t+1} (spec E6), not the expected
            # survey; the collapse penalty is already mode-aware (build_reward).
            immediate = reward.utility(next_states) - specs[action].cost
            immediate -= reward.collapse_penalty * penalty_indicator
            target = immediate + discount * np.max(next_q, axis=1)
            q_weights[action] = np.linalg.solve(
                X.T @ X + ridge * np.eye(X.shape[1]), X.T @ target
            )
    return q_weights


def evaluate_q(q_weights: np.ndarray, state: float, K_ref: float) -> np.ndarray:
    return value_features(np.asarray([state]), K_ref)[0] @ q_weights.T
