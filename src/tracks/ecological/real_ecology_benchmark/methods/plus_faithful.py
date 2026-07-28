"""Adapted PLUS baseline with a fitted mechanistic candidate bank."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from ..config import FaithfulConfig, MethodContext
from ..dataset import TrajectoryDataset
from ..faithful_artifacts import save_fit_artifacts
from ..faithful_fit import build_candidate_bank
from ..faithful_pomdp import CandidateBelief, CandidatePOMDP
from ..planners.pbvi import PointBasedPlanner
from ..types import BeliefState, PublicTransition
from .base import BasePolicy


PLUS_FAITHFUL_VERSION = "plus_adapted_mechanistic_fixed_pi_pbvi_v2"
PLUS_RICKER_ONLY_VERSION = "plus_adapted_ricker_only_pbvi_v1"


class PLUSFaithfulPBVIPolicy(BasePolicy):
    name = "plus_adapted_mechanistic_pbvi"
    required_forms = ("ricker", "allee", "theta", "regime")
    # The frozen cross-family implementation permits smaller candidate banks in
    # unit/smoke fixtures.  Production cardinality remains manifest-controlled.
    required_candidate_count: int | None = None
    method_impl_version = PLUS_FAITHFUL_VERSION

    def __init__(self, *args, faithful_cfg: FaithfulConfig, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.hidden or not isinstance(self.public_context, MethodContext):
            raise ValueError(f"{self.name} is registered only for expose_rk='hidden'")
        faithful_cfg.validate()
        actual_count = len(faithful_cfg.model.forms) * faithful_cfg.model.candidates_per_form
        if faithful_cfg.model.forms != self.required_forms:
            raise ValueError(
                f"{self.name} requires faithful.model.forms={self.required_forms!r}"
            )
        if (
            self.required_candidate_count is not None
            and actual_count != self.required_candidate_count
        ):
            raise ValueError(
                f"{self.name} requires exactly {self.required_candidate_count} candidates"
            )
        if faithful_cfg.planner.name != "pbvi":
            raise ValueError(f"{self.name} requires faithful.planner.name='pbvi'")
        self.faithful_config = faithful_cfg
        self.internal_beliefs: list[CandidateBelief] | None = None

    def fit(self, dataset: TrajectoryDataset, beliefs=None):
        del beliefs
        started = time.perf_counter()
        self.candidate_bank, self.fit_cache_statuses = build_candidate_bank(
            dataset,
            self.public_context,
            self.faithful_config.model,
            self.faithful_config.fit,
            self.seed + 11_000,
            self.faithful_config.fit_cache_dir,
        )
        dynamics_fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.pomdps = [
            CandidatePOMDP(
                fit.model,
                self.public_context,
                self.faithful_config.planner,
                self.seed + 12_000 + index,
            )
            for index, fit in enumerate(self.candidate_bank.fits)
        ]
        pomdp_init_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.planners = [
            PointBasedPlanner(
                pomdp,
                self.faithful_config.planner,
                self.planner_cfg.discount,
                self.seed + 13_000 + index,
            )
            for index, pomdp in enumerate(self.pomdps)
        ]
        planner_init_seconds = time.perf_counter() - started
        self.posterior = self.candidate_bank.initial_weights.copy()
        return {
            "candidate_count": float(len(self.candidate_bank.fits)),
            "candidate_form_count": float(len(self.faithful_config.model.forms)),
            "uniform_prior": float(self.candidate_bank.prior_type == "uniform"),
            "mean_trajectory_objective": float(
                np.mean([fit.objective for fit in self.candidate_bank.fits])
            ),
            "adapted_mechanistic_baseline": 1.0,
            "exact_paper_reproduction": 0.0,
            "cross_form_extension": float(len(self.faithful_config.model.forms) > 1),
            "ricker_only_candidate_bank": float(
                self.faithful_config.model.forms == ("ricker",)
            ),
            "bootstrap_map_candidates": 1.0,
            "fit_cache_hits": float(self.fit_cache_statuses.count("hit")),
            "fit_cache_misses": float(self.fit_cache_statuses.count("miss_fitted")),
            "fit_cache_keys": [fit.fit_cache_key for fit in self.candidate_bank.fits],
            "dynamics_fit_seconds": dynamics_fit_seconds,
            "pomdp_init_seconds": pomdp_init_seconds,
            "planner_init_seconds": planner_init_seconds,
        }

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.internal_beliefs = None
        if hasattr(self, "candidate_bank"):
            self.posterior = self.candidate_bank.initial_weights.copy()

    @staticmethod
    def _require_public_filter(belief: BeliefState) -> None:
        if belief.diagnostics.get("public_observation_filter") != 1.0:
            raise ValueError("adapted PLUS requires the public observation filter")

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
            self.method_impl_version,
            self.candidate_bank.initial_weights,
            self.candidate_bank.prior_type,
            self.candidate_bank.construction_type,
            pomdps=self.pomdps,
            planner_action_values=[planner.last_action_values for planner in self.planners],
        )


class PLUSRickerOnlyFaithfulPBVIPolicy(PLUSFaithfulPBVIPolicy):
    """PLUS with eight fixed Ricker candidate POMDPs and no cross-family candidates."""

    name = "plus_adapted_ricker_only_pbvi"
    required_forms = ("ricker",)
    required_candidate_count = 8
    method_impl_version = PLUS_RICKER_ONLY_VERSION
