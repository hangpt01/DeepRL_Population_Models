"""Known multiplicative log-normal observation kernel."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LogNormalObservationModel:
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    def sample(self, state: float | np.ndarray, rng: np.random.Generator):
        values = np.asarray(state, dtype=np.float64)
        if np.any(values < 0):
            raise ValueError("state must be non-negative")
        if self.sigma == 0:
            result = values.copy()
        else:
            noise = rng.lognormal(mean=0.0, sigma=self.sigma, size=values.shape)
            result = values * noise
        if np.ndim(state) == 0:
            return float(result)
        return result

    def log_prob(self, observation: float, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        if observation < 0 or np.any(states < 0):
            raise ValueError("observation and states must be non-negative")
        result = np.full(states.shape, -np.inf, dtype=np.float64)
        zero = states == 0.0
        if observation == 0.0:
            result[zero] = 0.0
            return result
        positive = states > 0.0
        if self.sigma == 0.0:
            tolerance = 1e-12 * np.maximum(1.0, states[positive])
            result[positive] = np.where(
                np.abs(states[positive] - observation) <= tolerance, 0.0, -np.inf
            )
            return result
        log_ratio = np.log(observation) - np.log(states[positive])
        result[positive] = (
            -0.5 * (log_ratio / self.sigma) ** 2
            - math.log(self.sigma)
            - 0.5 * math.log(2.0 * math.pi)
            - math.log(observation)
        )
        return result

    def expected_observation(self, state: np.ndarray | float):
        return np.asarray(state) * math.exp(0.5 * self.sigma * self.sigma)

