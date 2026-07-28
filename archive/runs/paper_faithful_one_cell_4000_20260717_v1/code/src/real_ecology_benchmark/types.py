"""Public interfaces and shared result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class ResetResult:
    observation: float
    done: bool = False
    public_info: dict[str, Any] = field(default_factory=dict)
    evaluator_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    observation: float
    reward: float
    done: bool
    truncated: bool
    public_info: dict[str, Any] = field(default_factory=dict)
    evaluator_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicTransition:
    """Policy-visible feedback from one environment transition.

    This deliberately carries only the observation/action-public context history.
    Realized reward is logged for training/evaluation targets, but it is not an
    online belief or policy input in the real-ecology POMDP.
    """

    observation: float
    done: bool
    truncated: bool
    terminated: bool = False
    public_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefState:
    """Policy belief over private abundance plus public management controls.

    ``rho``, ``kappa``, and ``K_eff`` are public cumulative-control variables. Hidden
    biology such as ``r_base``, ``C``, ``theta``, true abundance labels, and
    ``r_eff`` must never be placed in these public fields.
    """

    states: np.ndarray
    contexts: np.ndarray
    regimes: np.ndarray
    log_weights: np.ndarray
    observation: float
    timestep: int = 0
    diagnostics: dict[str, float] = field(default_factory=dict)
    rho: float | None = None
    kappa: float | None = None
    K_eff: float | None = None

    @property
    def weights(self) -> np.ndarray:
        shifted = self.log_weights - np.max(self.log_weights)
        weights = np.exp(shifted)
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total):
            return np.full(len(weights), 1.0 / len(weights))
        return weights / total

    @property
    def ess(self) -> float:
        weights = self.weights
        return float(1.0 / np.sum(weights * weights))

    def mean_state(self) -> float:
        return float(np.sum(self.states * self.weights))

    def sample_indices(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(len(self.states), size=n, replace=True, p=self.weights)

    def features(self, K_ref: float, safety_threshold: float) -> np.ndarray:
        if K_ref <= 0.0:
            raise ValueError("K_ref must be positive")
        if safety_threshold < 0.0:
            raise ValueError("safety_threshold must be non-negative")
        w = self.weights
        x = np.log1p(np.maximum(self.states, 0.0) / K_ref)
        order = np.argsort(x)
        cdf = np.cumsum(w[order])
        quantiles = []
        for q in (0.1, 0.5, 0.9):
            idx = min(int(np.searchsorted(cdf, q)), len(order) - 1)
            quantiles.append(float(x[order[idx]]))
        mean = float(np.sum(w * x))
        var = float(np.sum(w * (x - mean) ** 2))
        unsafe = float(np.sum(w[self.states <= safety_threshold]))
        extinct = float(np.sum(w[self.states == 0.0]))
        context = self.contexts
        if context.ndim == 1:
            context = context[:, None]
        context_mean = np.sum(w[:, None] * context, axis=0)
        regime_prob = float(np.sum(w * (self.regimes > 0)))
        values = [
            [mean, np.sqrt(max(var, 0.0)), *quantiles, unsafe, extinct,
             *context_mean.tolist(), regime_prob, self.ess / len(w)]
        ][0]
        if self.rho is not None and self.kappa is not None and self.K_eff is not None:
            values.extend(
                [
                    float(self.rho),
                    float(self.kappa) / max(K_ref, 1.0),
                    float(self.K_eff) / max(K_ref, 1.0),
                ]
            )
        return np.asarray(values, dtype=np.float64)

    def public_features(self, observation_scale: float) -> np.ndarray:
        """Belief summary derived only from observation-space particles."""

        if observation_scale <= 0.0:
            raise ValueError("observation_scale must be positive")
        w = self.weights
        x = np.log1p(np.maximum(self.states, 0.0) / observation_scale)
        order = np.argsort(x)
        cdf = np.cumsum(w[order])
        quantiles = []
        for q in (0.1, 0.5, 0.9):
            idx = min(int(np.searchsorted(cdf, q)), len(order) - 1)
            quantiles.append(float(x[order[idx]]))
        mean = float(np.sum(w * x))
        variance = float(np.sum(w * (x - mean) ** 2))
        extinct = float(np.sum(w[self.states == 0.0]))
        context = self.contexts
        if context.ndim == 1:
            context = context[:, None]
        context_mean = np.sum(w[:, None] * context, axis=0)
        return np.asarray(
            [
                mean,
                np.sqrt(max(variance, 0.0)),
                *quantiles,
                extinct,
                *context_mean.tolist(),
                self.ess / len(w),
            ],
            dtype=np.float64,
        )


class TransitionProposal(Protocol):
    def sample_next(
        self,
        states: np.ndarray,
        action: int | np.ndarray,
        contexts: np.ndarray,
        regimes: np.ndarray,
        rng: np.random.Generator,
        rho: float | np.ndarray | None = None,
        kappa: float | np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class BeliefFilter(Protocol):
    def reset(self, observation: float, seed: int) -> BeliefState: ...
    def update(self, belief: BeliefState, action: int, observation: float) -> BeliefState: ...


class BeliefPolicy(Protocol):
    name: str
    num_actions: int
    def fit(self, dataset: Any, beliefs: np.ndarray | None = None) -> dict[str, float]: ...
    def reset(self, seed: int) -> None: ...
    def act(self, belief: BeliefState, observation: float) -> int: ...
    def observe(self, belief: BeliefState, action: int, result: PublicTransition) -> None: ...
