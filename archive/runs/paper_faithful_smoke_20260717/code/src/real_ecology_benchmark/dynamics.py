"""Auditable NumPy continuous transition ensemble."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .config import EnvironmentConfig
from .controls import advance_public_controls, control_fields_enabled
from .dataset import TrajectoryDataset
from .beliefs import BeliefCache


def _design(
    states: np.ndarray,
    actions: np.ndarray,
    num_actions: int,
    K_ref: float,
    env_cfg: EnvironmentConfig | None = None,
    rho: np.ndarray | float | None = None,
    kappa: np.ndarray | float | None = None,
):
    x = np.log1p(np.maximum(states, 0.0) / K_ref)
    onehot = np.eye(num_actions, dtype=np.float64)[actions.astype(int)]
    # Harvest is proportional to abundance while stocking is additive, so an
    # action-only intercept cannot represent the locked authority map.  Keep
    # the auditable ridge model but give each action its own linear/quadratic
    # response in log abundance.
    columns = [
        np.ones(len(states)),
        x,
        x * x,
        onehot,
        onehot * x[:, None],
        onehot * (x * x)[:, None],
    ]
    if env_cfg is not None and control_fields_enabled(env_cfg):
        rho_in = np.zeros(len(states)) if rho is None else np.broadcast_to(
            np.asarray(rho, dtype=np.float64), states.shape
        )
        kappa_in = np.zeros(len(states)) if kappa is None else np.broadcast_to(
            np.asarray(kappa, dtype=np.float64), states.shape
        )
        rho_next, kappa_next, K_eff_next = advance_public_controls(
            env_cfg, actions, rho_in, kappa_in
        )
        controls = np.column_stack(
            [
                rho_next,
                kappa_next / max(K_ref, 1.0),
                K_eff_next / max(K_ref, 1.0),
            ]
        )
        columns.extend([controls, onehot * controls[:, 1:2]])
    return np.column_stack(columns)


@dataclass
class LinearDynamicsMember:
    coefficients: np.ndarray
    residual_sigma: float
    num_actions: int
    K_ref: float
    env_cfg: EnvironmentConfig | None = None

    def mean_next(self, states: np.ndarray, actions: np.ndarray, rho=None, kappa=None) -> np.ndarray:
        z = _design(
            states, actions, self.num_actions, self.K_ref, self.env_cfg, rho, kappa
        ) @ self.coefficients
        return self.K_ref * np.expm1(np.clip(z, 0.0, 50.0))

    def sample_next(self, states, actions, rng, rho=None, kappa=None):
        mean_log = _design(
            states, actions, self.num_actions, self.K_ref, self.env_cfg, rho, kappa
        ) @ self.coefficients
        draw = mean_log + rng.normal(0.0, self.residual_sigma, len(states))
        return np.maximum(self.K_ref * np.expm1(np.clip(draw, 0.0, 50.0)), 0.0)


class ContinuousDynamicsEnsemble:
    def __init__(self, members: list[LinearDynamicsMember], seed: int = 0):
        if not members:
            raise ValueError("ensemble requires at least one member")
        self.members = members
        self.num_actions = members[0].num_actions
        self.K_ref = members[0].K_ref
        self.rng = np.random.default_rng(seed)

    @classmethod
    def fit(
        cls,
        dataset: TrajectoryDataset,
        cache: BeliefCache,
        num_actions: int,
        ensemble_size: int = 5,
        ridge: float = 1e-3,
        K_ref: float = 500.0,
        seed: int = 0,
        env_cfg: EnvironmentConfig | None = None,
    ) -> "ContinuousDynamicsEnsemble":
        rng = np.random.default_rng(seed)
        members = []
        n = len(dataset)
        target = np.log1p(np.maximum(cache.next_mean_states, 0.0) / K_ref)
        for _ in range(ensemble_size):
            idx = rng.integers(0, n, n)
            X = _design(
                cache.mean_states[idx],
                dataset.actions[idx],
                num_actions,
                K_ref,
                env_cfg,
                cache.rho[idx] if cache.rho is not None else None,
                cache.kappa[idx] if cache.kappa is not None else None,
            )
            lhs = X.T @ X + ridge * np.eye(X.shape[1])
            coef = np.linalg.solve(lhs, X.T @ target[idx])
            residual = target[idx] - X @ coef
            members.append(
                LinearDynamicsMember(
                    coef, max(float(np.std(residual)), 0.02), num_actions, K_ref, env_cfg
                )
            )
        return cls(members, seed + 1)

    def predict(self, states: np.ndarray, actions: np.ndarray, rho=None, kappa=None):
        predictions = np.vstack([
            m.mean_next(states, actions, rho, kappa) for m in self.members
        ])
        return predictions.mean(axis=0), predictions.var(axis=0), predictions

    def sample_next(self, states, action, contexts, regimes, rng, rho=None, kappa=None):
        states = np.asarray(states, dtype=np.float64)
        actions = np.broadcast_to(np.asarray(action, dtype=int), states.shape)
        member_idx = rng.integers(0, len(self.members), len(states))
        out = np.empty_like(states)
        rho_values = None if rho is None else np.broadcast_to(np.asarray(rho, dtype=np.float64), states.shape)
        kappa_values = None if kappa is None else np.broadcast_to(np.asarray(kappa, dtype=np.float64), states.shape)
        for m_idx in np.unique(member_idx):
            mask = member_idx == m_idx
            out[mask] = self.members[int(m_idx)].sample_next(
                states[mask], actions[mask], rng,
                None if rho_values is None else rho_values[mask],
                None if kappa_values is None else kappa_values[mask],
            )
        out[states == 0.0] = 0.0
        return out, regimes.copy()

    def disagreement(self, states: np.ndarray, actions: np.ndarray, rho=None, kappa=None) -> np.ndarray:
        return self.predict(states, actions, rho, kappa)[1]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            coefficients=np.vstack([m.coefficients for m in self.members]),
            residual_sigma=np.asarray([m.residual_sigma for m in self.members]),
            num_actions=np.asarray(self.num_actions),
            K_ref=np.asarray(self.K_ref),
            env_json=np.asarray(
                "{}" if self.members[0].env_cfg is None
                else json.dumps(self.members[0].env_cfg.__dict__, sort_keys=True)
            ),
        )

    @classmethod
    def load(cls, path: str | Path, seed: int = 0):
        with np.load(path, allow_pickle=False) as data:
            n_actions = int(data["num_actions"])
            K_ref = float(data["K_ref"])
            env_cfg = None
            if "env_json" in data.files and str(data["env_json"].item()) != "{}":
                env_cfg = EnvironmentConfig(**json.loads(str(data["env_json"].item())))
            members = [
                LinearDynamicsMember(c, float(s), n_actions, K_ref, env_cfg)
                for c, s in zip(data["coefficients"], data["residual_sigma"])
            ]
        return cls(members, seed)
