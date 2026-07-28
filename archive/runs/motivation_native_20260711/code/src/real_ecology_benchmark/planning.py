"""Shared continuous particle-MPC engine used by methods and gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import EnvironmentConfig, PlannerConfig
from .controls import advance_public_controls, control_fields_enabled
from .observation import LogNormalObservationModel
from .reward import ContinuousReward, safety_penalty_indicator
from .types import BeliefState, TransitionProposal


@dataclass
class PlanDiagnostics:
    action_scores: np.ndarray
    action_safety_cost: np.ndarray
    action_ood_cost: np.ndarray
    best_sequence: np.ndarray


class ParticleMPC:
    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        planner_cfg: PlannerConfig,
        reward: ContinuousReward,
        observation_model: LogNormalObservationModel,
        seed: int = 0,
    ):
        self.env_cfg = env_cfg
        self.cfg = planner_cfg
        self.reward = reward
        self.observation_model = observation_model
        self.rng = np.random.default_rng(seed)

    def _sequences(self) -> np.ndarray:
        n = max(self.cfg.sequences, self.env_cfg.num_actions)
        seq = self.rng.integers(
            0, self.env_cfg.num_actions, size=(n, self.cfg.horizon), dtype=np.int16
        )
        seq[: self.env_cfg.num_actions, 0] = np.arange(self.env_cfg.num_actions)
        return seq

    def score_sequences(
        self,
        belief: BeliefState,
        proposal: TransitionProposal,
        sequences: np.ndarray | None = None,
        guardian: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        safety_lambda: float = 0.0,
        ood_lambda: float = 0.0,
        pessimism: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequences = self._sequences() if sequences is None else np.asarray(sequences)
        n_seq = len(sequences)
        n_particles = self.cfg.particles
        idx = belief.sample_indices(n_seq * n_particles, self.rng)
        states = belief.states[idx].reshape(n_seq, n_particles)
        context_values = belief.contexts[idx]
        if context_values.ndim == 1:
            contexts = context_values.reshape(n_seq, n_particles)
        else:
            contexts = context_values.reshape(n_seq, n_particles, context_values.shape[1])
        regimes = belief.regimes[idx].reshape(n_seq, n_particles)
        if control_fields_enabled(self.env_cfg):
            rho = np.full((n_seq, n_particles), 0.0 if belief.rho is None else belief.rho)
            kappa = np.full((n_seq, n_particles), 0.0 if belief.kappa is None else belief.kappa)
        else:
            rho = None
            kappa = None
        returns = np.zeros((n_seq, n_particles), dtype=np.float64)
        safety = np.zeros_like(returns)
        ood = np.zeros_like(returns)
        entered_latched = states <= self.env_cfg.safety_threshold
        gamma = 1.0
        epistemic = np.zeros_like(returns)
        for h in range(sequences.shape[1]):
            actions = sequences[:, h]
            flat_actions = np.repeat(actions, n_particles)
            flat_states = states.reshape(-1)
            flat_rho = None if rho is None else rho.reshape(-1)
            flat_kappa = None if kappa is None else kappa.reshape(-1)
            next_states, next_regimes = proposal.sample_next(
                flat_states,
                flat_actions,
                contexts.reshape(n_seq * n_particles, -1)
                if contexts.ndim == 3
                else contexts.reshape(-1),
                regimes.reshape(-1),
                self.rng,
                flat_rho,
                flat_kappa,
            )
            next_states = next_states.reshape(n_seq, n_particles)
            regimes = next_regimes.reshape(n_seq, n_particles)
            crossing = (
                (~entered_latched)
                & (states > self.env_cfg.safety_threshold)
                & (next_states <= self.env_cfg.safety_threshold)
            )
            entered_latched |= crossing
            penalty_indicator = safety_penalty_indicator(
                self.env_cfg, states, next_states, crossing
            )
            costs = np.asarray([self.reward.actions[int(a)].cost for a in actions])[:, None]
            # Benefit on the predicted TRUE next state s_{t+1} (spec E6), i.e. the
            # belief-expected next-state benefit -- consistent with the env reward
            # and free of the observation-noise bias E[o|s]=s*exp(sigma^2/2).
            step_reward = self.reward.utility(next_states) - costs
            step_reward -= self.reward.collapse_penalty * penalty_indicator
            step_safety = penalty_indicator.astype(np.float64)
            if guardian is not None:
                # Compact state/action coordinates; the guardian standardizes internally.
                g_features = np.column_stack(
                    [np.log1p(flat_states / self.env_cfg.K_ref), flat_actions]
                )
                step_ood = guardian(g_features, flat_actions).reshape(n_seq, n_particles)
            else:
                step_ood = np.zeros_like(step_safety)
            returns += gamma * (
                step_reward - safety_lambda * step_safety - ood_lambda * step_ood
            )
            safety += gamma * step_safety
            ood += gamma * step_ood
            if hasattr(proposal, "disagreement"):
                epistemic += gamma * np.asarray(
                    proposal.disagreement(
                        flat_states,
                        flat_actions,
                        flat_rho,
                        flat_kappa,
                    )
                ).reshape(n_seq, n_particles)
            if rho is not None and kappa is not None:
                next_rho, next_kappa, _next_K_eff = advance_public_controls(
                    self.env_cfg,
                    flat_actions,
                    flat_rho,
                    flat_kappa,
                )
                rho = next_rho.reshape(n_seq, n_particles)
                kappa = next_kappa.reshape(n_seq, n_particles)
            states = next_states
            gamma *= self.cfg.discount
        penalty = self.cfg.pessimism if pessimism is None else pessimism
        scores = returns.mean(axis=1) - penalty * returns.std(axis=1)
        if np.any(epistemic):
            scale = np.maximum(np.mean(states, axis=1) ** 2, 1.0)
            scores -= penalty * epistemic.mean(axis=1) / scale
        return scores, safety.mean(axis=1), ood.mean(axis=1)

    def plan(
        self,
        belief: BeliefState,
        proposal: TransitionProposal,
        guardian=None,
        safety_lambda: float = 0.0,
        ood_lambda: float = 0.0,
        pessimism: float | None = None,
    ) -> tuple[int, PlanDiagnostics]:
        sequences = self._sequences()
        scores, safety, ood = self.score_sequences(
            belief, proposal, sequences, guardian, safety_lambda, ood_lambda, pessimism
        )
        best = int(np.argmax(scores))
        action_scores = np.full(self.env_cfg.num_actions, -np.inf)
        action_safety = np.full(self.env_cfg.num_actions, np.nan)
        action_ood = np.full(self.env_cfg.num_actions, np.nan)
        for action in range(self.env_cfg.num_actions):
            idx = np.flatnonzero(sequences[:, 0] == action)
            if len(idx):
                local = idx[np.argmax(scores[idx])]
                action_scores[action] = scores[local]
                action_safety[action] = safety[local]
                action_ood[action] = ood[local]
        return int(sequences[best, 0]), PlanDiagnostics(
            action_scores, action_safety, action_ood, sequences[best].copy()
        )
