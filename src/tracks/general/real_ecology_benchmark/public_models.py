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
        standard_normal = rng.normal(0.0, 1.0, len(observations))
        return self.sample_next_from_standard_normal(observations, actions, standard_normal)

    def sample_next_from_standard_normal(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        standard_normal: np.ndarray,
    ) -> np.ndarray:
        """Sample with an externally registered predictive-residual draw.

        Supplying the standard-normal innovation lets constrained planners use
        common random numbers across candidate actions without changing the
        fitted observation-space predictive distribution.
        """

        observations = np.asarray(observations, dtype=np.float64)
        actions = np.asarray(actions, dtype=int)
        standard_normal = np.asarray(standard_normal, dtype=np.float64)
        if observations.shape != actions.shape or observations.shape != standard_normal.shape:
            raise ValueError("public predictive inputs and residual draws must have equal shape")
        mean = _design(
            observations, actions, self.num_actions, self.observation_scale
        ) @ self.coefficients
        draw = mean + self.residual_sigma * standard_normal
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

    def _sequences(self, prior_probabilities: np.ndarray | None = None) -> np.ndarray:
        count = max(self.cfg.sequences, self.context.num_actions)
        if prior_probabilities is None:
            sequences = self.rng.integers(
                0, self.context.num_actions,
                size=(count, self.cfg.horizon), dtype=np.int16,
            )
        else:
            probabilities = np.asarray(prior_probabilities, dtype=np.float64)
            probabilities = probabilities / probabilities.sum()
            sequences = self.rng.choice(
                self.context.num_actions,
                size=(count, self.cfg.horizon),
                p=probabilities,
            ).astype(np.int16)
        # The conservative prior proposes candidates, while this coverage row
        # keeps every discrete first action alive under the epsilon floor.
        sequences[: self.context.num_actions, 0] = np.arange(self.context.num_actions)
        return sequences

    def _proposal_features(
        self,
        root_features: np.ndarray,
        previous: np.ndarray,
        current: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        """Deterministic public-history features for prior-guided proposals."""

        features = np.repeat(root_features[None, :], len(current), axis=0)
        x = np.log1p(np.maximum(current, 0.0) / self.context.observation_scale)
        features[:, 0] = x
        features[:, 1] = 0.0
        features[:, 2:5] = x[:, None]
        features[:, 5] = (current <= 0.0).astype(np.float64)
        # PublicObservationFilter contributes previous observation, current
        # observation, and timestep before the final normalized-ESS feature.
        if features.shape[1] >= 10:
            features[:, 6] = previous
            features[:, 7] = current
            features[:, 8] = float(timestep)
            features[:, 9] = 1.0
        return features

    def _history_conditioned_prior_sequences(
        self,
        belief,
        dynamics: PublicDynamicsEnsemble,
        posterior: np.ndarray,
        policy_prior,
        prior_epsilon: float,
    ) -> np.ndarray:
        """Propose fixed sequences by re-evaluating the prior after every step."""

        count = max(self.cfg.sequences, self.context.num_actions)
        sequences = np.empty((count, self.cfg.horizon), dtype=np.int16)
        root_features = belief.public_features(self.context.observation_scale)
        current = np.full(count, belief.mean_state(), dtype=np.float64)
        previous = current.copy()
        uniform = np.full(self.context.num_actions, 1.0 / self.context.num_actions)
        for depth in range(self.cfg.horizon):
            features = self._proposal_features(
                root_features, previous, current, belief.timestep + depth
            )
            learned = np.asarray(policy_prior.probabilities(features), dtype=np.float64)
            probabilities = (1.0 - prior_epsilon) * learned + prior_epsilon * uniform
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            actions = np.asarray([
                self.rng.choice(self.context.num_actions, p=row) for row in probabilities
            ], dtype=np.int16)
            if depth == 0:
                actions[: self.context.num_actions] = np.arange(self.context.num_actions)
            sequences[:, depth] = actions
            predictions = np.stack([
                member.mean_next(current, actions.astype(int))
                for member in dynamics.members
            ], axis=0)
            following = posterior @ predictions
            previous, current = current, following
        self.last_proposal_sequences = sequences.copy()
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

    def plan_marginalized(
        self,
        belief,
        dynamics: PublicDynamicsEnsemble,
        posterior,
        pessimism: float | None = None,
        policy_prior=None,
        prior_epsilon: float = 0.10,
    ):
        """Reflect-then-Plan hidden planning: marginalize the deployment model
        posterior ``b(theta)`` over ensemble members into candidate-plan scoring.

        Each shared candidate action sequence is scored under *every* ensemble
        member using common particle starts and actions (common random numbers),
        producing a ``[members, sequences]`` return matrix.  The
        posterior-weighted mean is the marginalized plan value and the posterior
        return standard deviation is the reflected uncertainty penalty (Sikchi
        et al. 2021).  This is the hidden-mode analogue of the full-mode
        per-member reflection and mirrors the paper's marginalization of the
        model belief into model-based planning (Jeong et al., ICML 2025).

        Discrete-action adaptation: a calibrated public behavior policy proposes
        candidate sequences.  It is a distinct object from the dynamics posterior
        and is mixed with a registered uniform epsilon floor.
        """
        posterior = np.asarray(posterior, dtype=np.float64)
        total = posterior.sum()
        posterior = (
            posterior / total
            if total > 0
            else np.full(len(dynamics.members), 1.0 / len(dynamics.members))
        )
        sequences = (
            self._sequences()
            if policy_prior is None
            else self._history_conditioned_prior_sequences(
                belief, dynamics, posterior, policy_prior, prior_epsilon
            )
        )
        n_seq = len(sequences)
        n_part = self.cfg.particles
        indices = belief.sample_indices(n_seq * n_part, self.rng)
        start = belief.states[indices].reshape(n_seq, n_part)
        if belief.contexts.ndim > 1 and belief.contexts.shape[1] > 0:
            start_prev = belief.contexts[indices, 0].reshape(n_seq, n_part)
        else:
            start_prev = start.copy()
        member_returns = np.empty((len(dynamics.members), n_seq))
        member_risk = np.empty((len(dynamics.members), n_seq))
        for m_idx, member in enumerate(dynamics.members):
            # Fresh but deterministic per-member stream; identical starts/actions
            # across members isolate dynamics disagreement from start noise.
            rng = np.random.default_rng(int(self.rng.integers(0, 2**63 - 1)))
            current = start.copy()
            previous = start_prev.copy()
            returns = np.zeros_like(current)
            cumulative_risk = np.zeros_like(current)
            gamma = 1.0
            for depth in range(sequences.shape[1]):
                actions = np.repeat(sequences[:, depth], n_part)
                flat_current = current.reshape(-1)
                flat_previous = previous.reshape(-1)
                following = member.sample_next(flat_current, actions, rng)
                timesteps = np.full(len(actions), belief.timestep + depth, dtype=np.int32)
                pop_ids = np.full(len(actions), self.context.pop_id, dtype="U20")
                reward, risk = self.surrogate.predict(
                    flat_previous, flat_current, following, actions, timesteps, pop_ids
                )
                returns += gamma * reward.reshape(n_seq, n_part)
                step_risk = risk.reshape(n_seq, n_part)
                cumulative_risk = 1.0 - (1.0 - cumulative_risk) * (1.0 - step_risk)
                previous = current
                current = following.reshape(n_seq, n_part)
                gamma *= self.cfg.discount
            member_returns[m_idx] = returns.mean(axis=1)
            member_risk[m_idx] = cumulative_risk.mean(axis=1)
        penalty = self.cfg.pessimism if pessimism is None else pessimism
        mean = posterior @ member_returns
        variance = posterior @ (member_returns - mean) ** 2
        reflected = mean - penalty * np.sqrt(np.maximum(variance, 0.0))
        marginal_risk = posterior @ member_risk
        best = int(np.argmax(reflected))
        action_scores = np.full(self.context.num_actions, -np.inf)
        action_risk = np.full(self.context.num_actions, np.nan)
        for action in range(self.context.num_actions):
            matching = np.flatnonzero(sequences[:, 0] == action)
            if len(matching):
                selected = matching[np.argmax(reflected[matching])]
                action_scores[action] = reflected[selected]
                action_risk[action] = float(marginal_risk[selected])
        diagnostics = PlanDiagnostics(
            action_scores,
            action_risk,
            np.zeros(self.context.num_actions),
            sequences[best].copy(),
        )
        return int(sequences[best, 0]), diagnostics
