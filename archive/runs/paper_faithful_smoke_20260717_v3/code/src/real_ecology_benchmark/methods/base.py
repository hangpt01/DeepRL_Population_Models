"""Shared policy lifecycle."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np

from ..config import EnvironmentConfig, MethodContext, ModelConfig, PlannerConfig, hides_rk
from ..dataset import TrajectoryDataset
from ..beliefs import BeliefCache
from ..types import BeliefState, PublicTransition


class BasePolicy:
    name = "base"

    def __init__(
        self,
        env_cfg: EnvironmentConfig | MethodContext,
        model_cfg: ModelConfig,
        planner_cfg: PlannerConfig,
        seed: int = 0,
    ):
        self.hidden = isinstance(env_cfg, MethodContext)
        if self.hidden:
            env_cfg.validate()
            self.method_context = env_cfg
        else:
            if hides_rk(env_cfg):
                raise ValueError(
                    "expose_rk='hidden' policies require a sanitized MethodContext"
                )
            # Full mode intentionally retains the exact pre-change object and path.
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

    @property
    def public_context(self) -> MethodContext:
        if not self.hidden:
            raise AttributeError("public_context is only available in hidden mode")
        return self.method_context

    def fit(
        self,
        dataset: TrajectoryDataset,
        beliefs: BeliefCache | None = None,
    ) -> dict[str, float]:
        return {}

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.last_diagnostics = {}

    def save_fit_artifacts(self, output: str | Path) -> dict[str, Any]:
        """Serialize method-specific fit state; legacy policies intentionally no-op."""

        del output
        return {}

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
