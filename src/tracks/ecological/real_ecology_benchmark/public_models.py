"""Sanitized observation-space dynamics and planning for hidden-r/K methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MethodContext, ModelConfig, PlannerConfig
from .dataset import TrajectoryDataset
from .planning import PlanDiagnostics


def _design(
    observations: np.ndarray,
    actions: np.ndarray,
    num_actions: int,
    observation_scale: float,
) -> np.ndarray:
    x = np.log1p(np.maximum(observations, 0.0) / observation_scale)
    onehot = np.eye(num_actions, dtype=np.float64)[np.asarray(actions, dtype=int)]
    return np.column_stack(
        [np.ones(len(x)), x, x * x, onehot, onehot * x[:, None], onehot * (x * x)[:, None]]
    )


@dataclass(frozen=True)
class PublicDynamicsMember:
    coefficients: np.ndarray
    residual_sigma: float
    num_actions: int
    observation_scale: float

    def mean_next(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        value = _design(
            np.asarray(observations), np.asarray(actions), self.num_actions, self.observation_scale
        ) @ self.coefficients
        return np.maximum(self.observation_scale * np.expm1(np.clip(value, 0.0, 50.0)), 0.0)

    def sample_next(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mean = _design(
            np.asarray(observations), np.asarray(actions), self.num_actions, self.observation_scale
        ) @ self.coefficients
        draw = mean + rng.normal(0.0, self.residual_sigma, len(observations))
        return np.maximum(self.observation_scale * np.expm1(np.clip(draw, 0.0, 50.0)), 0.0)


class PublicDynamicsEnsemble:
    def __init__(self, members: list[PublicDynamicsMember], seed: int):
        self.members = members
        self.num_actions = members[0].num_actions
        self.observation_scale = members[0].observation_scale
        self.rng = np.random.default_rng(seed)

    @classmethod
    def fit(
        cls,
        dataset: TrajectoryDataset,
        cache,
        context: MethodContext,
        model_cfg: ModelConfig,
        seed: int,
        minimum_members: int = 1,
    ) -> "PublicDynamicsEnsemble":
        rng = np.random.default_rng(seed)
        members: list[PublicDynamicsMember] = []
        target = np.log1p(
            np.maximum(cache.next_mean_states, 0.0) / context.observation_scale
        )
        count = max(model_cfg.ensemble_size, minimum_members)
        for _ in range(count):
            indices = rng.integers(0, len(dataset), len(dataset))
            design = _design(
                cache.mean_states[indices],
                dataset.actions[indices],
                context.num_actions,
                context.observation_scale,
            )
            penalty = model_cfg.ridge * np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            coefficients = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ target[indices],
            )
            residual = target[indices] - design @ coefficients
            members.append(
                PublicDynamicsMember(
                    coefficients,
                    max(float(np.std(residual)), 0.02),
                    context.num_actions,
                    context.observation_scale,
                )
            )
        return cls(members, seed + 1)

    def predict(self, observations: np.ndarray, actions: np.ndarray):
        predictions = np.vstack(
            [member.mean_next(observations, actions) for member in self.members]
        )
        return predictions.mean(axis=0), predictions.var(axis=0), predictions

    def sample_next(self, observations, action, contexts, regimes, rng, rho=None, kappa=None):
        del contexts, rho, kappa
        observations = np.asarray(observations, dtype=np.float64)
        actions = np.broadcast_to(np.asarray(action, dtype=int), observations.shape)
        member_indices = rng.integers(0, len(self.members), len(observations))
        output = np.empty_like(observations)
        for member_index in np.unique(member_indices):
            mask = member_indices == member_index
            output[mask] = self.members[int(member_index)].sample_next(
                observations[mask], actions[mask], rng
            )
        return output, np.asarray(regimes).copy()

    def disagreement(self, observations: np.ndarray, actions: np.ndarray, *unused) -> np.ndarray:
        return self.predict(observations, actions)[1]


class PublicParticlePlanner:
    def __init__(
        self,
        context: MethodContext,
        planner_cfg: PlannerConfig,
        seed: int,
    ):
        context.validate()
        if context.surrogate is None:
            raise ValueError("hidden planning requires the shared public surrogate")
        self.context = context
        self.cfg = planner_cfg
        self.surrogate = context.surrogate
        self.rng = np.random.default_rng(seed)

    def _sequences(self) -> np.ndarray:
        count = max(self.cfg.sequences, self.context.num_actions)
        sequences = self.rng.integers(
            0,
            self.context.num_actions,
            size=(count, self.cfg.horizon),
            dtype=np.int16,
        )
        sequences[: self.context.num_actions, 0] = np.arange(self.context.num_actions)
        return sequences

    def plan(self, belief, dynamics: PublicDynamicsEnsemble, pessimism: float | None = None):
        sequences = self._sequences()
        sequence_count = len(sequences)
        particle_count = self.cfg.particles
        indices = belief.sample_indices(sequence_count * particle_count, self.rng)
        current = belief.states[indices].reshape(sequence_count, particle_count)
        if belief.contexts.ndim > 1 and belief.contexts.shape[1] > 0:
            previous = belief.contexts[indices, 0].reshape(sequence_count, particle_count)
        else:
            previous = current.copy()
        returns = np.zeros_like(current)
        cumulative_risk = np.zeros_like(current)
        gamma = 1.0
        for depth in range(sequences.shape[1]):
            actions = np.repeat(sequences[:, depth], particle_count)
            flat_current = current.reshape(-1)
            flat_previous = previous.reshape(-1)
            following, _ = dynamics.sample_next(
                flat_current,
                actions,
                np.zeros((len(actions), 1)),
                np.zeros(len(actions), dtype=np.int8),
                self.rng,
            )
            timesteps = np.full(len(actions), belief.timestep + depth, dtype=np.int32)
            pop_ids = np.full(len(actions), self.context.pop_id, dtype="U20")
            reward, risk = self.surrogate.predict(
                flat_previous,
                flat_current,
                following,
                actions,
                timesteps,
                pop_ids,
            )
            returns += gamma * reward.reshape(sequence_count, particle_count)
            step_risk = risk.reshape(sequence_count, particle_count)
            cumulative_risk = 1.0 - (1.0 - cumulative_risk) * (1.0 - step_risk)
            previous = current
            current = following.reshape(sequence_count, particle_count)
            gamma *= self.cfg.discount
        penalty = self.cfg.pessimism if pessimism is None else pessimism
        scores = returns.mean(axis=1) - penalty * returns.std(axis=1)
        best = int(np.argmax(scores))
        action_scores = np.full(self.context.num_actions, -np.inf)
        action_risk = np.full(self.context.num_actions, np.nan)
        for action in range(self.context.num_actions):
            matching = np.flatnonzero(sequences[:, 0] == action)
            if len(matching):
                selected = matching[np.argmax(scores[matching])]
                action_scores[action] = scores[selected]
                action_risk[action] = float(np.mean(cumulative_risk[selected]))
        diagnostics = PlanDiagnostics(
            action_scores,
            action_risk,
            np.zeros(self.context.num_actions),
            sequences[best].copy(),
        )
        return int(sequences[best, 0]), diagnostics
