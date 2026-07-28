"""Continuous MOPO: bootstrap dynamics plus pessimistic particle MPC."""

from __future__ import annotations

from ..beliefs import BeliefCache
from ..dataset import TrajectoryDataset
from ..dynamics import ContinuousDynamicsEnsemble
from ..observation import LogNormalObservationModel
from ..planning import ParticleMPC
from ..reward import build_reward
from ..types import BeliefState
from .base import BasePolicy


class MOPOPolicy(BasePolicy):
    name = "mopo"

    def fit(self, dataset: TrajectoryDataset, beliefs: BeliefCache | None = None):
        if beliefs is None:
            raise ValueError("MOPO requires cached shared beliefs")
        self.dynamics = ContinuousDynamicsEnsemble.fit(
            dataset,
            beliefs,
            self.num_actions,
            self.model_cfg.ensemble_size,
            self.model_cfg.ridge,
            self.env_cfg.K_ref,
            self.seed,
            self.env_cfg,
        )
        reward = build_reward(self.env_cfg)
        self.planner = ParticleMPC(
            self.env_cfg,
            self.planner_cfg,
            reward,
            LogNormalObservationModel(self.env_cfg.observation_noise_sigma),
            self.seed,
        )
        mean, _, _ = self.dynamics.predict(
            beliefs.mean_states, dataset.actions, beliefs.rho, beliefs.kappa
        )
        rmse = float(((mean - beliefs.next_mean_states) ** 2).mean() ** 0.5)
        return {"dynamics_rmse": rmse}

    def act(self, belief: BeliefState, observation: float) -> int:
        action, diag = self.planner.plan(
            belief, self.dynamics, pessimism=self.planner_cfg.pessimism
        )
        self.last_diagnostics = {
            "uncertainty": float(
                self.dynamics.disagreement(
                    belief.states,
                    self.rng.integers(0, self.num_actions, len(belief.states)),
                    belief.rho,
                    belief.kappa,
                ).mean()
            ),
            "action_scores": diag.action_scores.tolist(),
        }
        return action
