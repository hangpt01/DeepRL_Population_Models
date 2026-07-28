"""Delphic-CQL with compatible latent worlds and a discrete linear CQL head."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..beliefs import BeliefCache
from ..dataset import TrajectoryDataset
from ..types import BeliefState
from .base import BasePolicy


def _augment(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features)), features])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / exp.sum(axis=1, keepdims=True)


def _ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)


def _posterior_ambiguity(features: np.ndarray) -> np.ndarray:
    """Log-state ambiguity carried by the shared belief representation.

    Feature 1 is posterior log-state standard deviation and features 2/4 are
    its 10/90 percentiles.  Both collapse to zero spread when sigma_obs=0,
    which makes the no-hidden-confounder cell a genuine Delphic negative
    control rather than leaving arbitrary random-feature disagreement.
    """

    standard_deviation = np.maximum(features[:, 1], 0.0)
    quantile_scale = np.maximum(features[:, 4] - features[:, 2], 0.0) / 2.5631
    return np.maximum(standard_deviation, quantile_scale)


@dataclass
class CompatibleWorld:
    projection: np.ndarray
    behavior_weights: np.ndarray
    q_weights: np.ndarray
    latent_scale: float
    num_actions: int
    behavior_nll: float
    bellman_mse: float

    def latent(self, features: np.ndarray) -> np.ndarray:
        ambiguity = _posterior_ambiguity(features)[:, None]
        return np.tanh(features @ self.projection) * ambiguity * self.latent_scale

    def world_features(self, features: np.ndarray) -> np.ndarray:
        return _augment(np.column_stack([features, self.latent(features)]))

    def behavior_prob(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.world_features(features) @ self.behavior_weights.T)

    def behavior_q(self, features: np.ndarray) -> np.ndarray:
        return self.world_features(features) @ self.q_weights.T

    def counterfactual_q(self, features: np.ndarray, target_actions: np.ndarray) -> np.ndarray:
        q = self.behavior_q(features)
        behavior = np.clip(self.behavior_prob(features), 1e-3, 1.0)
        target = np.full_like(behavior, 0.05 / max(self.num_actions - 1, 1))
        target[np.arange(len(features)), target_actions] = 0.95
        ratio = np.clip(target / behavior, 0.1, 10.0)
        return q * ratio


class DelphicCQLPolicy(BasePolicy):
    name = "delphic"

    def __init__(
        self,
        *args,
        world_count: int = 10,
        cql_alpha: float = 0.5,
        delphic_lambda: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.world_count = world_count
        self.cql_alpha = cql_alpha
        self.delphic_lambda = delphic_lambda

    def _fit_behavior(self, X, actions):
        # A common deterministic initialization ensures that compatible-world
        # spread is caused by the latent ambiguity assumptions, not optimizer
        # initialization or ordinary bootstrap epistemic noise.
        weights = np.zeros((self.num_actions, X.shape[1]))
        onehot = np.eye(self.num_actions)[actions]
        for _ in range(120):
            probs = _softmax(X @ weights.T)
            gradient = (probs - onehot).T @ X / len(X) + 1e-3 * weights
            weights -= 0.15 * gradient
        return weights

    def _fit_world(self, dataset, cache, world_index):
        rng = np.random.default_rng(self.seed + 1009 * (world_index + 1))
        latent_dim = 1 + world_index % 3
        latent_scale = (0.5, 1.0, 1.5)[world_index % 3]
        projection = rng.normal(0.0, 1.0 / np.sqrt(cache.features.shape[1]),
                                (cache.features.shape[1], latent_dim))
        latent = (
            np.tanh(cache.features @ projection)
            * _posterior_ambiguity(cache.features)[:, None]
            * latent_scale
        )
        X = _augment(np.column_stack([cache.features, latent]))
        n = len(dataset)
        behavior_weights = self._fit_behavior(X, dataset.actions.astype(int))
        q_weights = np.zeros((self.num_actions, X.shape[1]))
        q_next = np.zeros((n, self.num_actions))
        for _ in range(25):
            target = dataset.rewards + 0.95 * (~dataset.dones) * np.max(q_next, axis=1)
            for action in range(self.num_actions):
                mask = dataset.actions == action
                # Ridge is well-defined in the underdetermined case. Requiring
                # one sample per feature would make worlds with different
                # latent dimensions fit different action subsets even when
                # posterior ambiguity (and therefore every latent) is zero.
                if np.sum(mask) >= 2:
                    q_weights[action] = _ridge(X[mask], target[mask], self.model_cfg.ridge)
            next_latent = (
                np.tanh(cache.next_features @ projection)
                * _posterior_ambiguity(cache.next_features)[:, None]
                * latent_scale
            )
            X_next = _augment(np.column_stack([cache.next_features, next_latent]))
            q_next = X_next @ q_weights.T
        behavior = np.clip(_softmax(X @ behavior_weights.T), 1e-12, 1.0)
        behavior_nll = float(-np.mean(
            np.log(behavior[np.arange(n), dataset.actions.astype(int)])
        ))
        final_target = dataset.rewards + 0.95 * (~dataset.dones) * np.max(q_next, axis=1)
        observed_q = (X @ q_weights.T)[np.arange(n), dataset.actions.astype(int)]
        bellman_mse = float(np.mean((observed_q - final_target) ** 2))
        return CompatibleWorld(
            projection,
            behavior_weights,
            q_weights,
            latent_scale,
            self.num_actions,
            behavior_nll,
            bellman_mse,
        )

    def _world_values(self, features: np.ndarray, target_actions: np.ndarray):
        return np.stack([
            world.counterfactual_q(features, target_actions) for world in self.worlds
        ])

    def _uncertainty(self, features: np.ndarray, target_actions: np.ndarray):
        values = self._world_values(features, target_actions)
        return np.var(values, axis=0)

    def _q_training_metrics(self, dataset: TrajectoryDataset, beliefs: BeliefCache):
        X = _augment(beliefs.features)
        X_next = _augment(beliefs.next_features)
        q_current = X @ self.q_weights.T
        q_next = X_next @ self.q_weights.T
        target_actions = np.argmax(q_current, axis=1)
        uncertainty = self._uncertainty(beliefs.features, target_actions)
        observed_uncertainty = uncertainty[
            np.arange(len(dataset)), dataset.actions.astype(int)
        ]
        target = (
            dataset.rewards
            + self.planner_cfg.discount * (~dataset.dones) * np.max(q_next, axis=1)
            - self.delphic_lambda * observed_uncertainty
        )
        observed_q = q_current[np.arange(len(dataset)), dataset.actions.astype(int)]
        return {
            "q_bellman_mse": float(np.mean((observed_q - target) ** 2)),
            "mean_delphic_uncertainty": float(np.mean(observed_uncertainty)),
        }

    def fit(self, dataset: TrajectoryDataset, beliefs: BeliefCache | None = None):
        if beliefs is None:
            raise ValueError("Delphic-CQL requires cached shared beliefs")
        self.worlds = [self._fit_world(dataset, beliefs, i) for i in range(self.world_count)]
        X = _augment(beliefs.features)
        X_next = _augment(beliefs.next_features)
        self.q_weights = np.zeros((self.num_actions, X.shape[1]))
        delphic_mean = 0.0
        holdout_dataset = self.training_holdout_dataset
        holdout_beliefs = self.training_holdout_beliefs
        for iteration in range(35):
            q_current = X @ self.q_weights.T
            q_next = X_next @ self.q_weights.T
            target_actions = np.argmax(q_current, axis=1)
            uncertainty = self._uncertainty(beliefs.features, target_actions)
            observed_uncertainty = uncertainty[
                np.arange(len(dataset)), dataset.actions.astype(int)
            ]
            target = (
                dataset.rewards
                + 0.95 * (~dataset.dones) * np.max(q_next, axis=1)
                - self.delphic_lambda * observed_uncertainty
            )
            for action in range(self.num_actions):
                observed = dataset.actions == action
                if np.sum(observed) < X.shape[1]:
                    continue
                # CQL-style conservative pseudo-targets suppress unsupported actions.
                X_fit = np.vstack([X[observed], np.sqrt(self.cql_alpha) * X])
                y_fit = np.concatenate([target[observed], np.zeros(len(X))])
                self.q_weights[action] = _ridge(X_fit, y_fit, self.model_cfg.ridge)
            delphic_mean = float(np.mean(observed_uncertainty))
            self.log_training(
                iteration,
                "train",
                self._q_training_metrics(dataset, beliefs),
                phase="cql",
            )
            if holdout_dataset is not None and holdout_beliefs is not None:
                self.log_training(
                    iteration,
                    "holdout",
                    self._q_training_metrics(holdout_dataset, holdout_beliefs),
                    phase="cql",
                )
        self.feature_dim = beliefs.features.shape[1]
        return {
            "worlds": float(self.world_count),
            "mean_delphic_uncertainty": delphic_mean,
            "posterior_ambiguity_mean": float(np.mean(
                _posterior_ambiguity(beliefs.features)
            )),
            "world_behavior_nll_range": float(
                max(world.behavior_nll for world in self.worlds)
                - min(world.behavior_nll for world in self.worlds)
            ),
            "world_bellman_mse_range": float(
                max(world.bellman_mse for world in self.worlds)
                - min(world.bellman_mse for world in self.worlds)
            ),
        }

    def act(self, belief: BeliefState, observation: float) -> int:
        del observation
        if self.hidden:
            features = belief.public_features(self.public_context.observation_scale)[None, :]
        else:
            features = belief.features(
                self.env_cfg.K_ref, self.env_cfg.safety_threshold
            )[None, :]
        q = _augment(features) @ self.q_weights.T
        target_action = np.argmax(q, axis=1)
        uncertainty = self._uncertainty(features, target_action)[0]
        conservative = q[0] - self.delphic_lambda * uncertainty
        action = int(np.argmax(conservative))
        self.last_diagnostics = {
            "delphic_uncertainty": float(uncertainty[action]),
            "delphic_uncertainty_all": uncertainty.tolist(),
            "q": q[0].tolist(),
        }
        return action
