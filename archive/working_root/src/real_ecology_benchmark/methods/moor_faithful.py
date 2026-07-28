"""Adapted MOOR baseline: one ordered-episode fitted Ricker POMDP."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from ..config import FaithfulConfig, MethodContext
from ..dataset import TrajectoryDataset
from ..faithful_artifacts import save_fit_artifacts
from ..faithful_fit import load_or_fit_mechanistic_model
from ..faithful_pomdp import CandidateBelief, CandidatePOMDP
from ..planners.pbvi import PointBasedPlanner
from ..types import BeliefState, PublicTransition
from .base import BasePolicy


MOOR_FAITHFUL_VERSION = "moor_adapted_ricker_misspec_pbvi_v2"


class MOORFaithfulRickerPBVIPolicy(BasePolicy):
    name = "moor_adapted_ricker_misspec_pbvi"

    def __init__(self, *args, faithful_cfg: FaithfulConfig, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.hidden or not isinstance(self.public_context, MethodContext):
            raise ValueError(f"{self.name} is registered only for expose_rk='hidden'")
        faithful_cfg.validate()
        if faithful_cfg.planner.name != "pbvi":
            raise ValueError(f"{self.name} requires faithful.planner.name='pbvi'")
        self.faithful_config = faithful_cfg
        self.internal_belief: CandidateBelief | None = None

    def fit(self, dataset: TrajectoryDataset, beliefs=None):
        del beliefs
        started = time.perf_counter()
        self.fit_result, self.fit_cache_status = load_or_fit_mechanistic_model(
            dataset,
            self.public_context,
            "ricker",
            self.faithful_config.model,
            self.faithful_config.fit,
            self.seed + 7_000,
            candidate_id="moor_ricker_00",
            cache_dir=self.faithful_config.fit_cache_dir,
        )
        dynamics_fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.pomdp = CandidatePOMDP(
            self.fit_result.model,
            self.public_context,
            self.faithful_config.planner,
            self.seed + 8_000,
        )
        pomdp_init_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.planner = PointBasedPlanner(
            self.pomdp,
            self.faithful_config.planner,
            self.planner_cfg.discount,
            self.seed + 9_000,
        )
        planner_init_seconds = time.perf_counter() - started
        return {
            "trajectory_objective": float(self.fit_result.objective),
            "selected_start": float(self.fit_result.selected_start),
            "sparse_action_count": float(len(self.fit_result.sparse_actions)),
            "ordered_episode_count": float(len(self.fit_result.fit_episode_ids)),
            "adapted_mechanistic_baseline": 1.0,
            "exact_paper_reproduction": 0.0,
            "ricker_misspecification": 1.0,
            "fit_cache_hit": float(self.fit_cache_status == "hit"),
            "fit_cache_key": self.fit_result.fit_cache_key,
            "dynamics_fit_seconds": dynamics_fit_seconds,
            "pomdp_init_seconds": pomdp_init_seconds,
            "planner_init_seconds": planner_init_seconds,
        }

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.internal_belief = None

    @staticmethod
    def _require_public_filter(belief: BeliefState) -> None:
        if belief.diagnostics.get("public_observation_filter") != 1.0:
            raise ValueError("adapted MOOR requires the public observation filter")

    def act(self, belief: BeliefState, observation: float) -> int:
        self._require_public_filter(belief)
        if self.internal_belief is None:
            self.internal_belief = self.pomdp.initial_belief(float(observation))
        scores = self.planner.action_values(self.internal_belief)
        action = int(np.argmax(scores))
        self.last_diagnostics = {
            "action_scores": scores.tolist(),
            "internal_belief_entropy": float(-np.sum(
                self.internal_belief.probabilities
                * np.log(self.internal_belief.probabilities + 1e-300)
            )),
            "planner_invocations": float(self.planner.invocation_count),
        }
        return action

    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None:
        self._require_public_filter(belief)
        if self.internal_belief is None:
            self.internal_belief = self.pomdp.initial_belief(float(belief.observation))
        self.internal_belief, evidence = self.pomdp.update(
            self.internal_belief, int(action), float(result.observation)
        )
        self.last_diagnostics["public_log_evidence"] = float(evidence)

    def save_fit_artifacts(self, output: str | Path) -> dict[str, object]:
        return save_fit_artifacts(
            output,
            [self.fit_result],
            [self.planner.provenance().to_dict()],
            MOOR_FAITHFUL_VERSION,
            pomdps=[self.pomdp],
            planner_action_values=[self.planner.last_action_values],
        )
