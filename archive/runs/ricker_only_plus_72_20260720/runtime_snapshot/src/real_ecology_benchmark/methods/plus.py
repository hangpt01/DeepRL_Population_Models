"""PLUS: candidate Ricker models with Rao-Blackwellized state-filter bank."""

from __future__ import annotations

import numpy as np

from ..beliefs import BeliefCache, MechanisticProposal
from ..config import is_setpoint_cumulative
from ..controls import advance_public_controls, control_fields_enabled
from ..dataset import TrajectoryDataset
from ..native_fit import PUBLIC_FORM_CANDIDATES, build_fitted_solver
from ..observation import LogNormalObservationModel
from ..planning import ParticleMPC
from ..reward import build_reward
from ..types import BeliefState, PublicTransition
from .base import BasePolicy
from .value import evaluate_q, fit_mechanistic_q


class PLUSPolicy(BasePolicy):
    name = "plus"

    def __init__(self, *args, candidate_count: int = 21, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_count = candidate_count

    def fit(self, dataset: TrajectoryDataset, beliefs: BeliefCache | None = None):
        if self.hidden:
            self.candidates = list(PUBLIC_FORM_CANDIDATES)
            fitted_and_solvers = [
                build_fitted_solver(
                    dataset,
                    self.public_context,
                    family,
                    self.model_cfg,
                    self.planner_cfg,
                )
                for family in self.candidates
            ]
            self.fitted_models = [item[0] for item in fitted_and_solvers]
            self.public_solvers = [item[1] for item in fitted_and_solvers]
            count = len(self.candidates)
            self.posterior = np.full(count, 1.0 / count, dtype=np.float64)
            self.public_bank = None
            diagnostics = {
                "candidate_models": float(count),
                "expose_rk_hidden": 1.0,
            }
            diagnostics.update({
                f"fit_loss_{family}": float(model.fit_loss)
                for family, model in zip(self.candidates, self.fitted_models)
            })
            return diagnostics
        if is_setpoint_cumulative(self.env_cfg):
            # In the real setting the per-action growth rate is known (population
            # identity is observed); the candidate Ricker models instead span
            # capacity K in [K_base, K_max], the structural quantity the
            # Ricker-only baseline is uncertain about.
            self.candidates = np.geomspace(
                self.env_cfg.K_base, self.env_cfg.K_max, self.candidate_count
            )
            self.proposals = [
                MechanisticProposal(self.env_cfg, "ricker", None, float(K))
                for K in self.candidates
            ]
        else:
            self.candidates = np.linspace(
                self.env_cfg.r_base_low, self.env_cfg.r_base_high, self.candidate_count
            )
            self.proposals = [
                MechanisticProposal(self.env_cfg, "ricker", float(r)) for r in self.candidates
            ]
        self.posterior = np.full(self.candidate_count, 1.0 / self.candidate_count)
        self.reward = build_reward(self.env_cfg)
        self.observation_model = LogNormalObservationModel(
            self.env_cfg.observation_noise_sigma
        )
        self.planner = ParticleMPC(
            self.env_cfg, self.planner_cfg, self.reward, self.observation_model, self.seed
        )
        support = beliefs.mean_states if beliefs is not None else np.geomspace(1.0, 2000.0, 256)
        self.candidate_q = np.stack([
            fit_mechanistic_q(
                proposal, support, self.env_cfg, self.reward,
                self.planner_cfg.discount, self.model_cfg.ridge,
            )
            for proposal in self.proposals
        ]) if not control_fields_enabled(self.env_cfg) else None
        self.bank: list[BeliefState] | None = None
        return {"candidate_models": float(self.candidate_count)}

    def reset(self, seed: int) -> None:
        super().reset(seed)
        if hasattr(self, "candidates"):
            count = len(self.candidates)
            self.posterior = np.full(count, 1.0 / count)
        if self.hidden:
            self.public_bank = None
        self.bank = None

    def _initialize_bank(self, shared: BeliefState) -> None:
        self.bank = []
        for _ in self.candidates:
            self.bank.append(
                BeliefState(
                    shared.states.copy(), shared.contexts.copy(), shared.regimes.copy(),
                    shared.log_weights.copy(), shared.observation, shared.timestep,
                    rho=shared.rho, kappa=shared.kappa, K_eff=shared.K_eff,
                )
            )

    def act(self, belief: BeliefState, observation: float) -> int:
        if self.hidden:
            if self.public_bank is None:
                self.public_bank = [
                    solver.initial_log_weights(observation)
                    for solver in self.public_solvers
                ]
            matrix = np.vstack([
                solver.action_values(log_weights)
                for solver, log_weights in zip(self.public_solvers, self.public_bank)
            ])
            combined = self.posterior @ matrix
            entropy = float(-np.sum(self.posterior * np.log(self.posterior + 1e-300)))
            self.last_diagnostics = {
                "candidate_entropy": entropy,
                "selected_candidate_index": float(int(np.argmax(self.posterior))),
                "action_scores": combined.tolist(),
                **{
                    f"posterior_{family}": float(weight)
                    for family, weight in zip(self.candidates, self.posterior)
                },
            }
            return int(np.argmax(combined))
        if self.bank is None:
            self._initialize_bank(belief)
        if control_fields_enabled(self.env_cfg):
            sequences = self.planner._sequences()
            combined = np.zeros(len(sequences), dtype=np.float64)
            for weight, candidate_belief, proposal in zip(
                self.posterior, self.bank, self.proposals
            ):
                scores, _safety, _ood = self.planner.score_sequences(
                    candidate_belief, proposal, sequences, pessimism=0.0
                )
                combined += float(weight) * scores
            best_sequence = int(np.argmax(combined))
            entropy = float(-np.sum(self.posterior * np.log(self.posterior + 1e-300)))
            self.last_diagnostics = {
                "candidate_entropy": entropy,
                "posterior_mean_r": float(self.posterior @ self.candidates),
            }
            return int(sequences[best_sequence, 0])
        matrix = np.vstack([
            evaluate_q(weights, candidate_belief.mean_state(), self.env_cfg.K_ref)
            for weights, candidate_belief in zip(self.candidate_q, self.bank)
        ])
        combined = self.posterior @ matrix
        best = int(np.argmax(combined))
        entropy = float(-np.sum(self.posterior * np.log(self.posterior + 1e-300)))
        self.last_diagnostics = {
            "candidate_entropy": entropy,
            "posterior_mean_r": float(self.posterior @ self.candidates),
        }
        return best

    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None:
        if self.hidden:
            if self.public_bank is None:
                self.public_bank = [
                    solver.initial_log_weights(belief.observation)
                    for solver in self.public_solvers
                ]
            evidences = np.zeros(len(self.public_solvers), dtype=np.float64)
            updated = []
            for index, (solver, log_weights) in enumerate(
                zip(self.public_solvers, self.public_bank)
            ):
                next_weights, _control, evidence = solver.update_log_weights(
                    log_weights,
                    None,
                    action,
                    result.observation,
                )
                updated.append(next_weights)
                evidences[index] = evidence
            log_posterior = np.log(self.posterior + 1e-300) + evidences
            log_posterior -= np.max(log_posterior)
            posterior = np.exp(log_posterior)
            total = float(np.sum(posterior))
            self.posterior = (
                posterior / total
                if total > 0.0 and np.isfinite(total)
                else np.full(len(self.public_solvers), 1.0 / len(self.public_solvers))
            )
            self.public_bank = updated
            return
        if self.bank is None:
            self._initialize_bank(belief)
        evidences = np.zeros(self.candidate_count)
        updated_bank = []
        for k, (candidate_belief, proposal) in enumerate(zip(self.bank, self.proposals)):
            states, regimes = proposal.sample_next(
                candidate_belief.states, action, candidate_belief.contexts,
                candidate_belief.regimes, self.rng,
                candidate_belief.rho, candidate_belief.kappa,
            )
            control_kwargs = {}
            if control_fields_enabled(self.env_cfg):
                next_rho, next_kappa, next_K_eff = advance_public_controls(
                    self.env_cfg,
                    action,
                    0.0 if candidate_belief.rho is None else candidate_belief.rho,
                    0.0 if candidate_belief.kappa is None else candidate_belief.kappa,
                )
                control_kwargs = {
                    "rho": float(next_rho),
                    "kappa": float(next_kappa),
                    "K_eff": float(next_K_eff),
                }
            ll = self.observation_model.log_prob(result.observation, states)
            prior_log = candidate_belief.log_weights
            finite = np.isfinite(ll + prior_log)
            if not np.any(finite):
                log_weights = np.full(len(states), -np.log(len(states)))
                evidences[k] = -1e6
            else:
                values = ll[finite] + prior_log[finite]
                m = np.max(values)
                evidence = m + np.log(np.sum(np.exp(values - m)))
                evidences[k] = evidence
                log_weights = np.full(len(states), -np.inf)
                log_weights[finite] = values - evidence
            updated_bank.append(
                BeliefState(
                    states, candidate_belief.contexts.copy(), regimes, log_weights,
                    result.observation, candidate_belief.timestep + 1,
                    **control_kwargs,
                )
            )
        log_posterior = np.log(self.posterior + 1e-300) + evidences
        log_posterior -= np.max(log_posterior)
        self.posterior = np.exp(log_posterior)
        self.posterior /= self.posterior.sum()
        self.bank = updated_bank
