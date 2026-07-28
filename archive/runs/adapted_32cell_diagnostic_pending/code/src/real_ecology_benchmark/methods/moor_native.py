"""Native MOOR/Ju baseline: fit a discrete Ricker POMDP and solve it."""

from __future__ import annotations

import numpy as np

from ..dataset import TrajectoryDataset
from ..native_solver import NativeSolver, predict_ricker_next
from ..native_fit import build_fitted_solver
from ..beliefs import FittedDiscreteGridFilter
from ..types import BeliefState
from .base import BasePolicy


class MOORNativePolicy(BasePolicy):
    name = "moor_native"

    def _require_native_belief(self, belief: BeliefState) -> None:
        if belief.diagnostics.get("discrete_grid") != 1.0:
            raise ValueError("moor_native requires filter='native_discrete'")
        if len(belief.log_weights) != self.solver.grid.num_hidden:
            raise ValueError(
                "moor_native belief grid size does not match the fitted native solver"
            )

    def fit(self, dataset: TrajectoryDataset, beliefs=None):
        del beliefs
        if self.hidden:
            self.fitted_model, self.solver = build_fitted_solver(
                dataset,
                self.public_context,
                "ricker",
                self.model_cfg,
                self.planner_cfg,
            )
            self.loss = self.fitted_model.fit_loss
            return {
                "fit_loss": self.loss,
                "fitted_form_ricker": 1.0,
                "expose_rk_hidden": 1.0,
            }
        kappa = dataset.kappa if dataset.kappa is not None else np.zeros(len(dataset))
        candidates = np.geomspace(
            max(self.env_cfg.K_base * 0.5, 1e-6),
            max(self.env_cfg.K_max, self.env_cfg.K_base),
            int(self.model_cfg.native_fit_grid),
        )
        target = np.log1p(np.maximum(dataset.next_observations, 0.0) / self.env_cfg.K_ref)
        best = (np.inf, float(self.env_cfg.K_base))
        for K in candidates:
            prediction = predict_ricker_next(
                self.env_cfg,
                dataset.observations,
                dataset.actions,
                kappa,
                assumed_K_base=float(K),
            )
            loss = float(np.mean(
                (np.log1p(np.maximum(prediction, 0.0) / self.env_cfg.K_ref) - target) ** 2
            ))
            if loss < best[0]:
                best = (loss, float(K))
        self.loss, self.K_hat = best
        self.solver = NativeSolver.build(
            self.env_cfg,
            assumed_K_base=self.K_hat,
            state_bins=self.model_cfg.native_state_bins,
            discount=self.planner_cfg.discount,
            iterations=self.model_cfg.native_vi_iterations,
            tolerance=self.model_cfg.native_vi_tolerance,
        )
        return {"fit_loss": self.loss, "K_hat": self.K_hat}

    def act(self, belief: BeliefState, observation: float) -> int:
        del observation
        self._require_native_belief(belief)
        scores = self.solver.action_values(
            belief.log_weights, None if self.hidden else belief.kappa
        )
        action = int(np.argmax(scores))
        self.last_diagnostics = {
            "action_scores": scores.tolist(),
        }
        if self.hidden:
            self.last_diagnostics["fitted_form_ricker"] = 1.0
        else:
            self.last_diagnostics["K_hat"] = float(self.K_hat)
        return action

    def hidden_filter_factory(self):
        if not self.hidden or not hasattr(self, "solver"):
            raise RuntimeError("hidden fitted solver is not ready")
        return lambda: FittedDiscreteGridFilter(self.public_context, self.solver)
