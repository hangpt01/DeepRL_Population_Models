"""Small MDP helpers mirroring the R MDPtoolbox usage in this project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class FiniteHorizonResult:
    V: np.ndarray
    policy: np.ndarray


def _as_transition_array(P: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Normalize transitions to shape [S, S, A]."""
    if isinstance(P, np.ndarray):
        if P.ndim != 3:
            raise ValueError("Transition array must have shape [S, S, A].")
        return P.astype(float)

    mats = [np.asarray(m, dtype=float) for m in P]
    return np.stack(mats, axis=2)


def finite_horizon(
    P: Sequence[np.ndarray] | np.ndarray,
    R: np.ndarray,
    discount: float,
    horizon: int,
) -> FiniteHorizonResult:
    """
    Finite-horizon dynamic programming.

    Returns:
      - V: shape [S, horizon + 1], with terminal value at column horizon
      - policy: shape [S, horizon], action indices in [1..A]
    """
    tr = _as_transition_array(P)
    rew = np.asarray(R, dtype=float)
    s, _, a = tr.shape
    if rew.shape != (s, a):
        raise ValueError(f"Reward matrix must be shape {(s, a)}, got {rew.shape}.")

    V = np.zeros((s, horizon + 1), dtype=float)
    policy = np.ones((s, horizon), dtype=int)

    # Backward induction: t = horizon-1 ... 0
    for t in range(horizon - 1, -1, -1):
        q = np.zeros((s, a), dtype=float)
        for act in range(a):
            q[:, act] = rew[:, act] + discount * (tr[:, :, act] @ V[:, t + 1])
        policy[:, t] = np.argmax(q, axis=1) + 1  # keep 1-based action indexing
        V[:, t] = q[np.arange(s), policy[:, t] - 1]

    return FiniteHorizonResult(V=V, policy=policy)


def bellman_operator(
    P: Sequence[np.ndarray] | np.ndarray,
    R: np.ndarray,
    discount: float,
    V: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """One-step Bellman backup for value iteration."""
    tr = _as_transition_array(P)
    rew = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)
    s, _, a = tr.shape

    q = np.zeros((s, a), dtype=float)
    for act in range(a):
        q[:, act] = rew[:, act] + discount * (tr[:, :, act] @ V)
    policy = np.argmax(q, axis=1) + 1
    V_new = q[np.arange(s), policy - 1]
    return V_new, policy


def value_iteration(
    P: Sequence[np.ndarray] | np.ndarray,
    R: np.ndarray,
    discount: float,
    epsilon: float = 1e-4,
    max_iter: int = 50000,
) -> Dict[str, np.ndarray]:
    """Simple discounted value iteration used by read_policyx.py helper."""
    tr = _as_transition_array(P)
    s = tr.shape[0]

    V = np.zeros(s, dtype=float)
    thresh = epsilon * (1 - discount) / max(discount, 1e-12)
    for _ in range(max_iter):
        V_prev = V.copy()
        V, policy = bellman_operator(tr, R, discount, V_prev)
        if np.max(V - V_prev) < thresh:
            return {"V": V, "policy": policy}

    return {"V": V, "policy": policy}
