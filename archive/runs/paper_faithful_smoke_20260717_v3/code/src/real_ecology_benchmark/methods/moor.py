"""MOOR: one misspecified Ricker model fitted to shared filtered states."""

from __future__ import annotations

import numpy as np

from ..actions import resolve_actions
from ..beliefs import BeliefCache, MechanisticProposal
from ..config import is_setpoint_cumulative
from ..controls import advance_public_controls, control_fields_enabled
from ..dataset import TrajectoryDataset
from ..native_fit import build_fitted_solver
from ..observation import LogNormalObservationModel
from ..planning import ParticleMPC
from ..reward import build_reward
from ..types import BeliefState, PublicTransition
from .base import BasePolicy
from .value import evaluate_q, fit_mechanistic_q


class MOORPolicy(BasePolicy):
    name = "moor"

    def _predict(self, states, actions, r, K, rho=None, kappa=None):
        specs = resolve_actions(self.env_cfg)
        out = np.zeros_like(states, dtype=np.float64)
        for aid in range(self.num_actions):
            mask = actions == aid
            if not np.any(mask):
                continue
            spec = specs[aid]
            if control_fields_enabled(self.env_cfg):
                rho_in = np.zeros(np.sum(mask)) if rho is None else np.asarray(rho)[mask]
                kappa_in = np.zeros(np.sum(mask)) if kappa is None else np.asarray(kappa)[mask]
                rho_next, kappa_next, _K_eff_public = advance_public_controls(
                    self.env_cfg, aid, rho_in, kappa_in
                )
                managed = np.maximum(states[mask] + spec.stocking_delta, 0.0)
                if is_setpoint_cumulative(self.env_cfg):
                    # r is known (set-point); rho already carries it, so fit K only.
                    r_eff = np.clip(rho_next, self.env_cfg.r_min, self.env_cfg.r_max)
                else:
                    r_eff = np.clip(r + rho_next, self.env_cfg.r_min, self.env_cfg.r_max)
                K_eff = np.clip(K + kappa_next, self.env_cfg.K_min, self.env_cfg.K_max)
            else:
                managed = np.maximum(
                    states[mask] * (1.0 - spec.harvest_fraction) + spec.stocking_delta, 0.0
                )
                r_eff = r + spec.delta_r
                K_eff = K + spec.delta_K
            # Mirror the env's growth/mortality split (envs.transition_value): the
            # positive part drives the density-dependent Ricker term, the negative
            # part is unconditional per-capita mortality.  Identity for r_eff>=0,
            # and avoids mismodelling the negative set-points of declining stocks.
            r_pos = np.maximum(r_eff, 0.0)
            r_mort = np.minimum(r_eff, 0.0)
            exponent = r_pos * (1.0 - managed / K_eff)
            out[mask] = (
                managed * np.exp(np.clip(exponent, -50.0, 50.0)) * np.exp(r_mort)
            )
        return out

    def fit(self, dataset: TrajectoryDataset, beliefs: BeliefCache | None = None):
        if beliefs is None:
            raise ValueError("MOOR requires cached shared beliefs")
        if self.hidden:
            self.fitted_model, self.public_solver = build_fitted_solver(
                dataset,
                self.public_context,
                "ricker",
                self.model_cfg,
                self.planner_cfg,
            )
            self.public_log_weights = None
            return {
                "fit_loss": float(self.fitted_model.fit_loss),
                "fitted_form_ricker": 1.0,
                "expose_rk_hidden": 1.0,
            }
        r_grid = np.linspace(max(0.01, self.env_cfg.r_base_low * 0.5), self.env_cfg.r_base_high * 1.5, 31)
        K_grid = np.geomspace(self.env_cfg.K_base * 0.5, self.env_cfg.K_base * 2.0, 31)
        target = np.log1p(beliefs.next_mean_states / self.env_cfg.K_ref)
        best = (np.inf, float(np.mean(r_grid)), self.env_cfg.K_base)
        for r in r_grid:
            for K in K_grid:
                prediction = self._predict(
                    beliefs.mean_states,
                    dataset.actions,
                    float(r),
                    float(K),
                    beliefs.rho,
                    beliefs.kappa,
                )
                loss = float(np.mean((np.log1p(prediction / self.env_cfg.K_ref) - target) ** 2))
                if loss < best[0]:
                    best = (loss, float(r), float(K))
        self.loss, self.r_hat, self.K_hat = best
        self.proposal = MechanisticProposal(
            self.env_cfg, "ricker", self.r_hat, self.K_hat
        )
        self.reward = build_reward(self.env_cfg)
        self.planner = ParticleMPC(
            self.env_cfg, self.planner_cfg, self.reward,
            LogNormalObservationModel(self.env_cfg.observation_noise_sigma), self.seed,
        )
        self.q_weights = None
        if not control_fields_enabled(self.env_cfg):
            self.q_weights = fit_mechanistic_q(
                self.proposal, beliefs.mean_states, self.env_cfg, self.reward,
                self.planner_cfg.discount, self.model_cfg.ridge,
            )
        return {"fit_loss": self.loss, "r_hat": self.r_hat, "K_hat": self.K_hat}

    def reset(self, seed: int) -> None:
        super().reset(seed)
        if self.hidden:
            self.public_log_weights = None

    def act(self, belief: BeliefState, observation: float) -> int:
        if self.hidden:
            if self.public_log_weights is None:
                self.public_log_weights = self.public_solver.initial_log_weights(observation)
            scores = self.public_solver.action_values(self.public_log_weights)
            self.last_diagnostics = {
                "fitted_form_ricker": 1.0,
                "action_scores": scores.tolist(),
            }
            return int(np.argmax(scores))
        if control_fields_enabled(self.env_cfg):
            action, diag = self.planner.plan(belief, self.proposal, pessimism=0.0)
            self.last_diagnostics = {
                "r_hat": self.r_hat,
                "K_hat": self.K_hat,
                "action_scores": diag.action_scores.tolist(),
            }
            return action
        q = evaluate_q(self.q_weights, belief.mean_state(), self.env_cfg.K_ref)
        action = int(np.argmax(q))
        self.last_diagnostics = {
            "r_hat": self.r_hat,
            "K_hat": self.K_hat,
            "action_scores": q.tolist(),
        }
        return action

    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None:
        del belief
        if not self.hidden:
            return
        if self.public_log_weights is None:
            self.public_log_weights = self.public_solver.initial_log_weights(result.observation)
            return
        self.public_log_weights, _control, _evidence = self.public_solver.update_log_weights(
            self.public_log_weights,
            None,
            action,
            result.observation,
        )
