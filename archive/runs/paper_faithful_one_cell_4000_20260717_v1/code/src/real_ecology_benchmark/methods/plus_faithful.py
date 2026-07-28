"""Paper-faithful PLUS adaptation with a fitted mechanistic candidate bank."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import FaithfulConfig, MethodContext
from ..dataset import TrajectoryDataset
from ..faithful_artifacts import save_fit_artifacts
from ..faithful_fit import build_candidate_bank
from ..faithful_pomdp import CandidateBelief, CandidatePOMDP
from ..planners.pbvi import PointBasedPlanner
from ..types import BeliefState, PublicTransition
from .base import BasePolicy


PLUS_FAITHFUL_VERSION = "plus_faithful_bootstrap_map_cross_form_v1"


class PLUSFaithfulPBVIPolicy(BasePolicy):
    name = "plus_faithful_pbvi"

    def __init__(self, *args, faithful_cfg: FaithfulConfig, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.hidden or not isinstance(self.public_context, MethodContext):
            raise ValueError(f"{self.name} is registered only for expose_rk='hidden'")
        faithful_cfg.validate()
        if faithful_cfg.planner.name != "pbvi":
            raise ValueError(f"{self.name} requires faithful.planner.name='pbvi'")
        self.faithful_config = faithful_cfg
        self.internal_beliefs: list[CandidateBelief] | None = None

    def fit(self, dataset: TrajectoryDataset, beliefs=None):
        del beliefs
        self.candidate_bank = build_candidate_bank(
            dataset,
            self.public_context,
            self.faithful_config.model,
            self.faithful_config.fit,
            self.seed + 11_000,
        )
        self.pomdps = [
            CandidatePOMDP(
                fit.model,
                self.public_context,
                self.faithful_config.planner,
                self.seed + 12_000 + index,
            )
            for index, fit in enumerate(self.candidate_bank.fits)
        ]
        self.planners = [
            PointBasedPlanner(
                pomdp,
                self.faithful_config.planner,
                self.planner_cfg.discount,
                self.seed + 13_000 + index,
            )
            for index, pomdp in enumerate(self.pomdps)
        ]
        self.posterior = self.candidate_bank.initial_weights.copy()
        return {
            "candidate_count": float(len(self.candidate_bank.fits)),
            "candidate_form_count": float(len(self.faithful_config.model.forms)),
            "uniform_prior": float(self.candidate_bank.prior_type == "uniform"),
            "mean_trajectory_objective": float(
                np.mean([fit.objective for fit in self.candidate_bank.fits])
            ),
            "paper_faithful_adaptation": 1.0,
            "cross_form_extension": 1.0,
            "bootstrap_map_candidates": 1.0,
        }

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.internal_beliefs = None
        if hasattr(self, "candidate_bank"):
            self.posterior = self.candidate_bank.initial_weights.copy()

    @staticmethod
    def _require_public_filter(belief: BeliefState) -> None:
        if belief.diagnostics.get("public_observation_filter") != 1.0:
            raise ValueError("faithful PLUS requires the public observation filter")

    def _initialize(self, observation: float) -> None:
        self.internal_beliefs = [pomdp.initial_belief(float(observation)) for pomdp in self.pomdps]

    def act(self, belief: BeliefState, observation: float) -> int:
        self._require_public_filter(belief)
        if self.internal_beliefs is None:
            self._initialize(observation)
        candidate_scores = np.asarray(
            [
                planner.action_values(candidate_belief)
                for planner, candidate_belief in zip(self.planners, self.internal_beliefs)
            ]
        )
        combined = self.posterior @ candidate_scores
        action = int(np.argmax(combined))
        entropy = float(-np.sum(self.posterior * np.log(self.posterior + 1e-300)))
        self.last_diagnostics = {
            "action_scores": combined.tolist(),
            "candidate_entropy": entropy,
            "candidate_weights": self.posterior.tolist(),
            "selected_candidate_index": float(int(np.argmax(self.posterior))),
            "planner_invocations": float(
                sum(planner.invocation_count for planner in self.planners)
            ),
        }
        return action

    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None:
        self._require_public_filter(belief)
        if self.internal_beliefs is None:
            self._initialize(belief.observation)
        updated = []
        evidence = []
        for pomdp, candidate_belief in zip(self.pomdps, self.internal_beliefs):
            next_belief, log_evidence = pomdp.update(
                candidate_belief, int(action), float(result.observation)
            )
            updated.append(next_belief)
            evidence.append(log_evidence)
        log_weights = np.log(self.posterior + 1e-300) + np.asarray(evidence)
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        self.posterior = weights / weights.sum()
        self.internal_beliefs = updated
        self.last_diagnostics["candidate_weights"] = self.posterior.tolist()

    def save_fit_artifacts(self, output: str | Path) -> dict[str, object]:
        return save_fit_artifacts(
            output,
            self.candidate_bank.fits,
            [planner.provenance().to_dict() for planner in self.planners],
            PLUS_FAITHFUL_VERSION,
            self.candidate_bank.initial_weights,
            self.candidate_bank.prior_type,
            self.candidate_bank.construction_type,
            pomdps=self.pomdps,
            planner_action_values=[planner.last_action_values for planner in self.planners],
        )
