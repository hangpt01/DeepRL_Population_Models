"""Native PLUS baseline: posterior over candidate mechanistic models.

PLUS's mechanism is Bayesian model averaging over a bank of candidate mechanistic
models, each solved as its own discrete POMDP.  In this set-point real-ecology
setting the growth set-point is a public function of the action and ``K_base``
comes straight from the species table, so a bank spanning Ricker *parameters*
carries no signal: 6 of the 11 actions have ``r_setpoint <= 0``, which zeroes
``r_pos`` and cancels ``K`` out of the density-dependent exponent entirely.  Such
a bank leaves the posterior flat and collapses PLUS onto MOOR.

The candidates are therefore the four mechanistic *forms* (Ricker / Allee /
theta-logistic / regime-switching) — the "candidate mechanistic models" of the
adaptive-management literature, and the axis on which this setting actually
carries structural uncertainty.  Each candidate is discretized and solved
independently; the model posterior is updated from ``(a, o)`` evidence only.
"""

from __future__ import annotations

import numpy as np

from ..dataset import TrajectoryDataset
from ..native_solver import NATIVE_FAMILIES, NativeSolver
from ..native_fit import PUBLIC_FORM_CANDIDATES, build_fitted_solver
from ..types import BeliefState, PublicTransition
from .base import BasePolicy


class PLUSNativePolicy(BasePolicy):
    name = "plus_native"

    def _require_native_belief(self, belief: BeliefState) -> None:
        # PLUS carries its own per-candidate belief bank (the candidates have
        # different hidden dimensions), so only the routing needs checking.
        if self.hidden:
            if belief.diagnostics.get("public_observation_filter") != 1.0:
                raise ValueError("hidden plus_native requires the public observation filter")
            return
        if belief.diagnostics.get("discrete_grid") != 1.0:
            raise ValueError("plus_native requires filter='native_discrete'")

    def fit(self, dataset: TrajectoryDataset, beliefs=None):
        del beliefs
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
            self.solvers = [item[1] for item in fitted_and_solvers]
            count = len(self.candidates)
            self.posterior = np.full(count, 1.0 / count, dtype=np.float64)
            self.bank_log_weights = None
            self.bank_controls = None
            output = {"candidate_models": float(count), "expose_rk_hidden": 1.0}
            output.update(
                {
                    f"fit_loss_{family}": float(model.fit_loss)
                    for family, model in zip(self.candidates, self.fitted_models)
                }
            )
            return output
        del dataset
        self.candidates = list(NATIVE_FAMILIES)
        self.solvers = [
            NativeSolver.build(
                self.env_cfg,
                assumed_family=family,
                state_bins=self.model_cfg.native_state_bins,
                discount=self.planner_cfg.discount,
                iterations=self.model_cfg.native_vi_iterations,
                tolerance=self.model_cfg.native_vi_tolerance,
            )
            for family in self.candidates
        ]
        count = len(self.candidates)
        self.posterior = np.full(count, 1.0 / count, dtype=np.float64)
        self.bank_log_weights: list[np.ndarray] | None = None
        self.bank_kappa: list[float | None] | None = None
        return {"candidate_models": float(count)}

    def reset(self, seed: int) -> None:
        super().reset(seed)
        if hasattr(self, "candidates"):
            self.posterior = np.full(len(self.candidates), 1.0 / len(self.candidates))
        self.bank_log_weights = None
        if self.hidden:
            self.bank_controls = None
        else:
            self.bank_kappa = None

    def _initialize_bank(self, belief: BeliefState, observation: float) -> None:
        self._require_native_belief(belief)
        self.bank_log_weights = [
            solver.initial_log_weights(float(observation)) for solver in self.solvers
        ]
        if self.hidden:
            self.bank_controls = [None for _ in self.solvers]
        else:
            self.bank_kappa = [belief.kappa for _ in self.solvers]

    def _diagnostics(self, scores: np.ndarray) -> dict[str, float | list[float]]:
        entropy = float(-np.sum(self.posterior * np.log(self.posterior + 1e-300)))
        diagnostics: dict[str, float | list[float]] = {
            "candidate_entropy": entropy,
            "action_scores": scores.tolist(),
        }
        diagnostics[
            "selected_candidate_index" if self.hidden else "map_family"
        ] = float(int(np.argmax(self.posterior)))
        for family, weight in zip(self.candidates, self.posterior):
            diagnostics[f"posterior_{family}"] = float(weight)
        return diagnostics

    def act(self, belief: BeliefState, observation: float) -> int:
        self._require_native_belief(belief)
        bank_controls = (
            self.bank_controls if self.hidden else self.bank_kappa
        )
        if self.bank_log_weights is None or bank_controls is None:
            self._initialize_bank(belief, observation)
            bank_controls = self.bank_controls if self.hidden else self.bank_kappa
        combined = np.zeros(self.num_actions, dtype=np.float64)
        for weight, solver, log_weights, control_state in zip(
            self.posterior, self.solvers, self.bank_log_weights, bank_controls
        ):
            combined += float(weight) * solver.action_values(log_weights, control_state)
        self.last_diagnostics = self._diagnostics(combined)
        return int(np.argmax(combined))

    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None:
        bank_controls = self.bank_controls if self.hidden else self.bank_kappa
        if self.bank_log_weights is None or bank_controls is None:
            self._initialize_bank(belief, belief.observation)
            bank_controls = self.bank_controls if self.hidden else self.bank_kappa
        evidences = np.zeros(len(self.solvers), dtype=np.float64)
        next_log_weights: list[np.ndarray] = []
        next_kappa: list[float | None] = []
        for idx, (solver, log_weights, control_state) in enumerate(
            zip(self.solvers, self.bank_log_weights, bank_controls)
        ):
            updated, next_control, evidence = solver.update_log_weights(
                log_weights,
                control_state,
                int(action),
                float(result.observation),
            )
            next_log_weights.append(updated)
            next_kappa.append(next_control)
            evidences[idx] = evidence
        log_posterior = np.log(self.posterior + 1e-300) + evidences
        finite = np.isfinite(log_posterior)
        if not np.any(finite):
            self.posterior = np.full(len(self.solvers), 1.0 / len(self.solvers))
        else:
            log_posterior -= float(np.max(log_posterior[finite]))
            posterior = np.zeros_like(self.posterior)
            posterior[finite] = np.exp(log_posterior[finite])
            total = float(posterior.sum())
            self.posterior = (
                posterior / total if total > 0.0 and np.isfinite(total)
                else np.full(len(self.solvers), 1.0 / len(self.solvers))
            )
        self.bank_log_weights = next_log_weights
        if self.hidden:
            self.bank_controls = next_kappa
        else:
            self.bank_kappa = next_kappa
