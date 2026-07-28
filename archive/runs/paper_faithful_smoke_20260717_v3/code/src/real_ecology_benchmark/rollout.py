"""Augmented rollout state for cumulative-control planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvironmentConfig
from .controls import advance_public_controls, control_fields_enabled
from .types import TransitionProposal


@dataclass
class AugmentedRolloutState:
    """Rollout state `(s, context, regime, rho, kappa)`.

    `rho/kappa` are public controls.  Hidden biology remains inside proposal
    particles or private evaluator state; this object never carries `r_eff`.
    """

    states: np.ndarray
    contexts: np.ndarray
    regimes: np.ndarray
    rho: np.ndarray | None = None
    kappa: np.ndarray | None = None

    def advance(
        self,
        env_cfg: EnvironmentConfig,
        proposal: TransitionProposal,
        actions: int | np.ndarray,
        rng: np.random.Generator,
    ) -> "AugmentedRolloutState":
        actions_arr = np.broadcast_to(np.asarray(actions, dtype=int), self.states.shape)
        next_states, next_regimes = proposal.sample_next(
            self.states,
            actions_arr,
            self.contexts,
            self.regimes,
            rng,
            self.rho,
            self.kappa,
        )
        if control_fields_enabled(env_cfg):
            next_rho, next_kappa, _next_K_eff = advance_public_controls(
                env_cfg,
                actions_arr,
                np.zeros_like(self.states) if self.rho is None else self.rho,
                np.zeros_like(self.states) if self.kappa is None else self.kappa,
            )
        else:
            next_rho = self.rho
            next_kappa = self.kappa
        return AugmentedRolloutState(
            np.asarray(next_states, dtype=np.float64),
            self.contexts,
            np.asarray(next_regimes, dtype=np.int8),
            None if next_rho is None else np.asarray(next_rho, dtype=np.float64),
            None if next_kappa is None else np.asarray(next_kappa, dtype=np.float64),
        )

