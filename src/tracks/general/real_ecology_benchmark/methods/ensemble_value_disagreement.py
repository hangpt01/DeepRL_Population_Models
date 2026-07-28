"""Bootstrap fitted-Q ensemble with value-disagreement pessimism.

Delphic work motivated using cross-model value disagreement pessimistically,
but this idea-level baseline does not implement latent causal/generative
models. Each member is an independently fitted conservative Q estimator whose
only source of ensemble variation is an episode bootstrap and registered seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..behavior_model import augment_features, behavior_nll, fit_reference_behavior
from ..beliefs import BeliefCache
from ..dataset import TrajectoryDataset
from ..types import BeliefState
from .base import BasePolicy


def _ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)


@dataclass(frozen=True)
class BootstrapQMember:
    """One conservative fitted-Q estimator trained on an episode bootstrap."""

    q_weights: np.ndarray
    bootstrap_episode_ids: np.ndarray
    seed: int
    bellman_mse: float

    def values(self, features: np.ndarray) -> np.ndarray:
        return augment_features(features) @ self.q_weights.T


class EnsembleValueDisagreementPolicy(BasePolicy):
    """Conservative bootstrap Q mean minus an empirical-variance penalty."""

    name = "ensemble_value_disagreement_pessimism"
    reader_label = "Ensemble value-disagreement pessimism (Delphic-motivated)"

    def __init__(
        self,
        *args,
        ensemble_size: int = 20,
        cql_alpha: float = 0.5,
        disagreement_penalty: float = 0.1,
        fit_iterations: int = 35,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if ensemble_size < 2:
            raise ValueError("Q disagreement requires at least two fitted members")
        self.ensemble_size = int(ensemble_size)
        self.cql_alpha = float(cql_alpha)
        # Q variance has squared Q units, so this registered coefficient has
        # inverse-Q units and maps the penalty back to the Q-score scale.
        self.disagreement_penalty = float(disagreement_penalty)
        self.fit_iterations = int(fit_iterations)

    def _bootstrap_rows(
        self, dataset: TrajectoryDataset, member_index: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        member_seed = self.seed + 1009 * (member_index + 1)
        rng = np.random.default_rng(member_seed)
        episodes = dataset.episode_indices()
        sampled = rng.integers(0, len(episodes), size=len(episodes))
        rows = np.concatenate([episodes[int(i)] for i in sampled])
        return rows, sampled.astype(np.int32), member_seed

    def _fit_member(
        self,
        dataset: TrajectoryDataset,
        beliefs: BeliefCache,
        member_index: int,
    ) -> BootstrapQMember:
        rows, sampled_episodes, member_seed = self._bootstrap_rows(dataset, member_index)
        X = augment_features(beliefs.features[rows])
        X_next = augment_features(beliefs.next_features[rows])
        actions = dataset.actions[rows].astype(int)
        rewards = dataset.rewards[rows]
        dones = dataset.dones[rows]
        q_weights = np.zeros((self.num_actions, X.shape[1]))
        for _ in range(self.fit_iterations):
            q_next = X_next @ q_weights.T
            target = rewards + self.planner_cfg.discount * (~dones) * np.max(q_next, axis=1)
            for action in range(self.num_actions):
                observed = actions == action
                if np.sum(observed) < 2:
                    continue
                X_fit = np.vstack([X[observed], np.sqrt(self.cql_alpha) * X])
                y_fit = np.concatenate([target[observed], np.zeros(len(X))])
                q_weights[action] = _ridge(X_fit, y_fit, self.model_cfg.ridge)
        q_current = X @ q_weights.T
        q_next = X_next @ q_weights.T
        target = rewards + self.planner_cfg.discount * (~dones) * np.max(q_next, axis=1)
        observed_q = q_current[np.arange(len(rows)), actions]
        return BootstrapQMember(
            q_weights=q_weights,
            bootstrap_episode_ids=sampled_episodes,
            seed=member_seed,
            bellman_mse=float(np.mean((observed_q - target) ** 2)),
        )

    def _statistics(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.stack([member.values(features) for member in self.q_members])
        return values.mean(axis=0), values.var(axis=0), values

    def _diagnostics(self, dataset: TrajectoryDataset, beliefs: BeliefCache) -> dict[str, float]:
        mean_q, disagreement, _ = self._statistics(beliefs.features)
        rows = np.arange(len(dataset))
        actions = dataset.actions.astype(int)
        observed = disagreement[rows, actions]
        unobserved_mask = np.ones_like(disagreement, dtype=bool)
        unobserved_mask[rows, actions] = False
        q_next = np.mean(
            np.stack([member.values(beliefs.next_features) for member in self.q_members]), axis=0
        )
        target = dataset.rewards + self.planner_cfg.discount * (~dataset.dones) * np.max(
            q_next, axis=1
        )
        observed_q = mean_q[rows, actions]
        return {
            "q_ensemble_disagreement_observed": float(np.mean(observed)),
            "q_ensemble_disagreement_unobserved": float(np.mean(disagreement[unobserved_mask])),
            "q_ensemble_mean_bellman_mse": float(np.mean((observed_q - target) ** 2)),
        }

    def fit(self, dataset: TrajectoryDataset, beliefs: BeliefCache | None = None):
        if beliefs is None:
            raise ValueError("bootstrap Q ensemble requires cached shared beliefs")
        self.behavior_model = fit_reference_behavior(
            beliefs.features, dataset.actions, self.num_actions
        )
        self.q_members = [
            self._fit_member(dataset, beliefs, member_index)
            for member_index in range(self.ensemble_size)
        ]
        self.feature_dim = beliefs.features.shape[1]
        # Mean weights are exposed only for generic Bellman diagnostics; action
        # selection always uses the member predictions and empirical variance.
        self.q_weights = np.mean(
            np.stack([member.q_weights for member in self.q_members]), axis=0
        )
        diagnostics = self._diagnostics(dataset, beliefs)
        diagnostics.update(
            {
                "q_ensemble_members": float(self.ensemble_size),
                "q_ensemble_bootstrap_unique_fraction_mean": float(np.mean([
                    len(np.unique(member.bootstrap_episode_ids))
                    / len(member.bootstrap_episode_ids)
                    for member in self.q_members
                ])),
                "q_ensemble_member_bellman_mse_mean": float(np.mean([
                    member.bellman_mse for member in self.q_members
                ])),
                "behavior_reference_nll_train": behavior_nll(
                    self.behavior_model, beliefs.features, dataset.actions
                ),
            }
        )
        if self.training_holdout_dataset is not None and self.training_holdout_beliefs is not None:
            diagnostics["behavior_reference_nll_holdout"] = behavior_nll(
                self.behavior_model,
                self.training_holdout_beliefs.features,
                self.training_holdout_dataset.actions,
            )
            diagnostics.update({
                f"holdout_{key}": value
                for key, value in self._diagnostics(
                    self.training_holdout_dataset, self.training_holdout_beliefs
                ).items()
            })
        return diagnostics

    def act(self, belief: BeliefState, observation: float) -> int:
        del observation
        features = (
            belief.public_features(self.public_context.observation_scale)[None, :]
            if self.hidden
            else belief.features(self.env_cfg.K_ref, self.env_cfg.safety_threshold)[None, :]
        )
        mean_q, disagreement, _ = self._statistics(features)
        score = mean_q[0] - self.disagreement_penalty * disagreement[0]
        action = int(np.argmax(score))
        self.last_diagnostics = {
            "q_ensemble_mean": mean_q[0].tolist(),
            "q_ensemble_disagreement": float(disagreement[0, action]),
            "q_ensemble_disagreement_all": disagreement[0].tolist(),
            "pessimistic_q_score": score.tolist(),
        }
        return action
