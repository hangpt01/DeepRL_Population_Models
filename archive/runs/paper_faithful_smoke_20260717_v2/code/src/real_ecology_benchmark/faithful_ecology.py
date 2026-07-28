"""Public-data mechanistic ecology models for the paper-faithful baselines.

This module deliberately has no route to benchmark tables, private simulator
configuration, evaluator state, or private sidecars.  All quantities are in the
public observation scale used by the fitted method.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


EQUATION_VERSION = "faithful_ecology_v1"
REGISTERED_FORMS = ("ricker", "allee", "theta", "regime")


def _array(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result


@dataclass(frozen=True)
class MechanisticModel:
    """One fixed candidate model after public-history fitting."""

    form: str
    growth: np.ndarray
    mortality: np.ndarray
    capacity_increment: np.ndarray
    stocking: np.ndarray
    reset_log_mean: float
    reset_log_scale: float
    initial_capacity: float
    capacity_ceiling: float
    process_scale: float
    observation_scale: float
    survey_scale: float
    depensation_thresholds: np.ndarray
    theta_exponent: float
    regime_multipliers: np.ndarray
    regime_matrix: np.ndarray
    candidate_id: str = "model_000"

    def __post_init__(self) -> None:
        if self.form not in REGISTERED_FORMS:
            raise ValueError(f"unknown mechanistic form {self.form!r}")
        action_count = len(np.asarray(self.growth))
        if action_count < 1:
            raise ValueError("a mechanistic model requires at least one action")
        for name in ("growth", "mortality", "capacity_increment", "stocking"):
            value = _array(getattr(self, name), (action_count,), name)
            if np.any(value < 0.0):
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        thresholds = _array(self.depensation_thresholds, (2,), "depensation_thresholds")
        multipliers = _array(self.regime_multipliers, (2,), "regime_multipliers")
        matrix = _array(self.regime_matrix, (2, 2), "regime_matrix")
        object.__setattr__(self, "depensation_thresholds", thresholds)
        object.__setattr__(self, "regime_multipliers", multipliers)
        object.__setattr__(self, "regime_matrix", matrix)
        if min(
            self.reset_log_scale,
            self.initial_capacity,
            self.capacity_ceiling,
            self.observation_scale,
            self.survey_scale,
            self.theta_exponent,
        ) <= 0.0 or self.process_scale < 0.0:
            raise ValueError("invalid positive mechanistic parameter")
        if self.capacity_ceiling < self.initial_capacity:
            raise ValueError("capacity ceiling must not be below its reset value")
        if np.any(thresholds <= 0.0) or np.any(thresholds >= self.initial_capacity):
            raise ValueError("depensation thresholds must be within reset capacity")
        if np.any(multipliers <= 0.0):
            raise ValueError("regime multipliers must be positive")
        if np.any(matrix < 0.0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("regime transition rows must be probability simplexes")

    @property
    def num_actions(self) -> int:
        return int(len(self.growth))

    @property
    def num_regimes(self) -> int:
        return 2 if self.form == "regime" else 1

    def next_capacity(self, capacity: float, action: int) -> float:
        return float(np.clip(
            float(capacity) + self.capacity_increment[int(action)],
            self.initial_capacity,
            self.capacity_ceiling,
        ))

    def noiseless_next(
        self,
        abundance: np.ndarray | float,
        capacity: float,
        action: int,
        regime: np.ndarray | int = 0,
    ) -> np.ndarray:
        x = np.maximum(np.asarray(abundance, dtype=np.float64), 0.0)
        action = int(action)
        next_capacity = self.next_capacity(capacity, action)
        managed = np.maximum(x + self.stocking[action], 0.0)
        growth = self.growth[action]
        mortality = self.mortality[action]
        if self.form == "ricker":
            exponent = growth * (1.0 - managed / next_capacity) - mortality
            result = managed * np.exp(np.clip(exponent, -40.0, 40.0))
        elif self.form == "allee":
            threshold = self.depensation_thresholds[0]
            exponent = (
                growth * (1.0 - managed / next_capacity)
                * (managed / threshold - 1.0) - mortality
            )
            result = managed * np.exp(np.clip(exponent, -40.0, 40.0))
        elif self.form == "theta":
            core = np.maximum(
                0.0,
                managed + growth * managed
                * (1.0 - (managed / next_capacity) ** self.theta_exponent),
            )
            result = core * np.exp(-mortality)
        else:
            regime_index = np.asarray(regime, dtype=np.int64)
            threshold = self.depensation_thresholds[regime_index]
            multiplier = self.regime_multipliers[regime_index]
            exponent = (
                multiplier * growth * (1.0 - managed / next_capacity)
                * (managed / threshold - 1.0) - mortality
            )
            result = managed * np.exp(np.clip(exponent, -40.0, 40.0))
        # The declared candidate can leave zero only through fitted stocking.
        return np.where((x == 0.0) & (self.stocking[action] == 0.0), 0.0, result)

    def sample_next(
        self,
        abundance: np.ndarray,
        capacity: float,
        action: int,
        regimes: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        abundance = np.asarray(abundance, dtype=np.float64)
        regimes = np.asarray(regimes, dtype=np.int64)
        next_regimes = regimes.copy()
        if self.form == "regime":
            draws = rng.random(len(regimes))
            next_regimes = (draws > self.regime_matrix[regimes, 0]).astype(np.int64)
        mean = self.noiseless_next(abundance, capacity, action, regimes)
        innovations = rng.normal(size=len(abundance))
        sampled = mean * np.exp(self.process_scale * innovations)
        sampled = np.where(mean <= 0.0, 0.0, sampled)
        return sampled, next_regimes, self.next_capacity(capacity, action)

    def observation_likelihood(
        self, observation: float, abundance: np.ndarray
    ) -> np.ndarray:
        x = np.maximum(np.asarray(abundance, dtype=np.float64), 0.0)
        y = max(float(observation) / self.survey_scale, 0.0)
        result = np.zeros_like(x)
        zero = x <= 0.0
        if y == 0.0:
            result[zero] = 1.0
            return result
        if self.observation_scale <= 1e-12:
            return np.isclose(x, y, rtol=1e-6, atol=1e-9).astype(np.float64)
        positive = ~zero
        log_error = np.log(y) - np.log(np.maximum(x[positive], 1e-300))
        scale = self.observation_scale
        result[positive] = np.exp(-0.5 * (log_error / scale) ** 2) / (
            y * scale * np.sqrt(2.0 * np.pi)
        )
        return result

    def parameter_hash(self) -> str:
        payload = {
            "equation_version": EQUATION_VERSION,
            "form": self.form,
            "scalars": [
                self.reset_log_mean,
                self.reset_log_scale,
                self.initial_capacity,
                self.capacity_ceiling,
                self.process_scale,
                self.observation_scale,
                self.survey_scale,
                self.theta_exponent,
            ],
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("ascii"))
        for name in (
            "growth", "mortality", "capacity_increment", "stocking",
            "depensation_thresholds", "regime_multipliers", "regime_matrix",
        ):
            digest.update(np.ascontiguousarray(getattr(self, name)).tobytes())
        return digest.hexdigest()


def default_model(form: str, num_actions: int, survey_scale: float, survey_noise: float) -> MechanisticModel:
    """Small deterministic model used by equation/planner tests, never real fitting."""

    return MechanisticModel(
        form=form,
        growth=np.full(num_actions, 0.2),
        mortality=np.full(num_actions, 0.05),
        capacity_increment=np.zeros(num_actions),
        stocking=np.zeros(num_actions),
        reset_log_mean=np.log(0.8),
        reset_log_scale=0.2,
        initial_capacity=1.0,
        capacity_ceiling=2.0,
        process_scale=0.1,
        observation_scale=float(survey_noise),
        survey_scale=float(survey_scale),
        depensation_thresholds=np.asarray([0.2, 0.35]),
        theta_exponent=2.0,
        regime_multipliers=np.asarray([0.7, 1.2]),
        regime_matrix=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
    )
