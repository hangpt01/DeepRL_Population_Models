"""Shared policy lifecycle."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import EnvironmentConfig, ModelConfig, PlannerConfig
from ..dataset import TrajectoryDataset
from ..beliefs import BeliefCache
from ..types import BeliefState, PublicTransition


class BasePolicy:
    name = "base"

    def __init__(
        self,
        env_cfg: EnvironmentConfig,
        model_cfg: ModelConfig,
        planner_cfg: PlannerConfig,
        seed: int = 0,
    ):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.planner_cfg = planner_cfg
        self.num_actions = env_cfg.num_actions
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.last_diagnostics: dict[str, Any] = {}
        self.fit_diagnostics: dict[str, Any] = {}
        self.training_history: Any | None = None
        self.training_holdout_dataset: TrajectoryDataset | None = None
        self.training_holdout_beliefs: BeliefCache | None = None

    def fit(
        self,
        dataset: TrajectoryDataset,
        beliefs: BeliefCache | None = None,
    ) -> dict[str, float]:
        return {}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.last_diagnostics = {}

    def log_training(
        self,
        step: int,
        split: str,
        metrics: dict[str, float | int | None],
        phase: str = "fit",
    ) -> None:
        if self.training_history is not None:
            self.training_history.log(step, split, metrics, phase)

    def act(self, belief: BeliefState, observation: float) -> int:
        raise NotImplementedError

    def observe(
        self,
        belief: BeliefState,
        action: int,
        result: PublicTransition,
    ) -> None:
        return None
