"""Continuous Reflect-then-Plan adaptation with model posterior."""

from __future__ import annotations

import math
import numpy as np

from ..beliefs import BeliefCache
from ..dataset import TrajectoryDataset
from ..dynamics import ContinuousDynamicsEnsemble, LinearDynamicsMember
from ..behavior_model import fit_reference_behavior
from ..observation import LogNormalObservationModel
from ..planning import ParticleMPC
from ..public_models import PublicDynamicsEnsemble, PublicParticlePlanner
from ..reward import build_reward
from ..types import BeliefState, PublicTransition
from .base import BasePolicy


class _MemberProposal:
    def __init__(self, member: LinearDynamicsMember):
        self.member = member

    def sample_next(self, states, action, contexts, regimes, rng, rho=None, kappa=None):
        actions = np.broadcast_to(np.asarray(action, dtype=int), np.asarray(states).shape)
        out = self.member.sample_next(np.asarray(states), actions, rng, rho, kappa)
        out[np.asarray(states) == 0.0] = 0.0
        return out, regimes.copy()


class RefPlanPolicy(BasePolicy):
    name = "refplan"
    prior_epsilon = 0.10

    def __init__(self, *args, max_planning_members: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_planning_members = max_planning_members

    def fit(self, dataset: TrajectoryDataset, beliefs: BeliefCache | None = None):
        if beliefs is None:
            raise ValueError("RefPlan requires cached shared beliefs")
        size = max(self.model_cfg.ensemble_size, 5)
        if self.hidden:
            self.dynamics = PublicDynamicsEnsemble.fit(
                dataset,
                beliefs,
                self.public_context,
                self.model_cfg,
                self.seed,
                minimum_members=size,
            )
            self.posterior = np.full(len(self.dynamics.members), 1.0 / len(self.dynamics.members))
            # Distinct public policy prior (train split only): this proposes
            # plans, while ``posterior`` remains the deployment model belief.
            self.policy_prior = fit_reference_behavior(
                beliefs.features, dataset.actions, self.num_actions
            )
            self.prior_weights = self.policy_prior.weights
            self.prior_scaler = (self.policy_prior.mean, self.policy_prior.scale)
            self.planner = PublicParticlePlanner(
                self.public_context, self.planner_cfg, self.seed
            )
            prior_probs = self.policy_prior.probabilities(beliefs.features)
            return {
                "members": float(len(self.dynamics.members)),
                "expose_rk_hidden": 1.0,
                "policy_prior_epsilon": self.prior_epsilon,
                "policy_prior_min_probability": float(
                    np.min((1.0 - self.prior_epsilon) * prior_probs
                           + self.prior_epsilon / self.num_actions)
                ),
            }
        self.dynamics = ContinuousDynamicsEnsemble.fit(
            dataset, beliefs, self.num_actions, size, self.model_cfg.ridge,
            self.env_cfg.K_ref, self.seed, self.env_cfg,
        )
        self.posterior = np.full(size, 1.0 / size)
        self.reward = build_reward(self.env_cfg)
        self.planner = ParticleMPC(
            self.env_cfg, self.planner_cfg, self.reward,
            LogNormalObservationModel(self.env_cfg.observation_noise_sigma), self.seed,
        )
        return {"members": float(size)}

    def reset(self, seed: int) -> None:
        super().reset(seed)
        if hasattr(self, "dynamics"):
            self.posterior = np.full(len(self.dynamics.members), 1.0 / len(self.dynamics.members))

    def act(self, belief: BeliefState, observation: float) -> int:
        if self.hidden:
            # Reflect-then-Plan: marginalize the deployment model posterior
            # b(theta) -- maintained and updated in observe() -- into every
            # candidate-plan score, rather than computing it as a diagnostic
            # only.  This makes the belief materially affect planning and
            # distinguishes RefPlan from the uniform-ensemble MOPO planner.
            action, diagnostics = self.planner.plan_marginalized(
                belief, self.dynamics, self.posterior,
                pessimism=self.planner_cfg.pessimism,
                policy_prior=self.policy_prior,
                prior_epsilon=self.prior_epsilon,
            )
            self.last_diagnostics = {
                "model_entropy": float(
                    -np.sum(self.posterior * np.log(self.posterior + 1e-12))
                ),
                "posterior_max": float(np.max(self.posterior)),
                "action_scores": diagnostics.action_scores.tolist(),
                "public_extinction_risk": float(diagnostics.action_safety_cost[action]),
            }
            return action
        sequences = self.planner._sequences()
        scores = []
        if len(self.dynamics.members) <= self.max_planning_members:
            selected = np.arange(len(self.dynamics.members))
        else:
            selected = self.rng.choice(
                len(self.dynamics.members), self.max_planning_members,
                replace=False, p=self.posterior,
            )
        selected_posterior = self.posterior[selected]
        selected_posterior /= selected_posterior.sum()
        for index in selected:
            member = self.dynamics.members[int(index)]
            member_scores, _, _ = self.planner.score_sequences(
                belief, _MemberProposal(member), sequences, pessimism=0.0
            )
            scores.append(member_scores)
        matrix = np.vstack(scores)
        mean = selected_posterior @ matrix
        variance = selected_posterior @ ((matrix - mean) ** 2)
        reflected = mean - self.planner_cfg.pessimism * np.sqrt(np.maximum(variance, 0.0))
        best = int(np.argmax(reflected))
        self.last_diagnostics = {
            "model_entropy": float(-np.sum(self.posterior * np.log(self.posterior + 1e-12))),
            "return_std": float(np.sqrt(variance[best])),
        }
        return int(sequences[best, 0])

    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None:
        state = np.asarray([belief.mean_state()])
        actions = np.asarray([action])
        target = result.observation
        log_likelihood = []
        obs_var = (
            self.public_context.observation_noise_sigma ** 2
            if self.hidden else self.env_cfg.observation_noise_sigma ** 2
        )
        for member in self.dynamics.members:
            mean = float(
                member.mean_next(state, actions)[0]
                if self.hidden
                else member.mean_next(state, actions, belief.rho, belief.kappa)[0]
            )
            sigma = max(member.residual_sigma + obs_var ** 0.5, 0.03)
            residual = math.log(max(target, 1e-8)) - math.log(max(mean, 1e-8))
            log_likelihood.append(-0.5 * (residual / sigma) ** 2 - math.log(sigma))
        logp = np.log(self.posterior + 1e-300) + np.asarray(log_likelihood)
        logp -= np.max(logp)
        self.posterior = np.exp(logp)
        self.posterior /= self.posterior.sum()
