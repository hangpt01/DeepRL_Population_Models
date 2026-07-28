"""Discretization, filtering, and public-history reward for adapted candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np

from .config import FaithfulPlannerConfig, MethodContext
from .faithful_ecology import MechanisticModel


LIKELIHOOD_FLOOR = 1e-300


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    return values / total


@dataclass(frozen=True)
class CandidateBelief:
    probabilities: np.ndarray
    capacity: float
    previous_observation: float
    current_observation: float
    timestep: int

    def __post_init__(self) -> None:
        probabilities = _normalize(self.probabilities)
        object.__setattr__(self, "probabilities", probabilities)
        if self.capacity <= 0.0 or self.timestep < 0:
            raise ValueError("invalid adapted candidate belief context")


class CandidatePOMDP:
    """One object supplies filtering, evidence, rewards, and planner transitions."""

    def __init__(
        self,
        model: MechanisticModel,
        context: MethodContext,
        planner_cfg: FaithfulPlannerConfig,
        seed: int,
    ):
        context.validate()
        self.model = model
        self.context = context
        self.config = planner_cfg
        self.seed = int(seed)
        positive = np.geomspace(
            1e-4,
            max(model.capacity_ceiling * 1.5, 1.0),
            planner_cfg.state_bins - 1,
        )
        self.abundance_grid = np.concatenate(([0.0], positive))
        self.regime_count = model.num_regimes
        self.hidden_count = len(self.abundance_grid) * self.regime_count
        self._transition_cache: dict[tuple[int, int], np.ndarray] = {}
        self.kernel_build_seconds = 0.0
        self.kernel_build_count = 0

    def _capacity_key(self, capacity: float) -> int:
        fraction = (
            (float(capacity) - self.model.initial_capacity)
            / max(self.model.capacity_ceiling - self.model.initial_capacity, 1e-12)
        )
        return int(np.clip(round(fraction * (self.config.capacity_bins - 1)), 0,
                           self.config.capacity_bins - 1))

    def _key_capacity(self, key: int) -> float:
        if self.config.capacity_bins == 1:
            return self.model.initial_capacity
        fraction = key / (self.config.capacity_bins - 1)
        return float(
            self.model.initial_capacity
            + fraction * (self.model.capacity_ceiling - self.model.initial_capacity)
        )

    def _deposit_samples(self, samples: np.ndarray) -> np.ndarray:
        """Sparse barycentric projection of continuous draws onto abundance bins."""

        samples = np.maximum(np.asarray(samples, dtype=np.float64), 0.0)
        values = np.zeros(len(self.abundance_grid), dtype=np.float64)
        for sample in samples:
            if sample <= 0.0:
                values[0] += 1.0
                continue
            upper = int(np.searchsorted(self.abundance_grid, sample, side="left"))
            if upper >= len(self.abundance_grid):
                values[-1] += 1.0
            elif upper == 0:
                values[0] += 1.0
            else:
                lower = upper - 1
                width = self.abundance_grid[upper] - self.abundance_grid[lower]
                fraction = (sample - self.abundance_grid[lower]) / max(width, 1e-300)
                values[lower] += 1.0 - fraction
                values[upper] += fraction
        return _normalize(values)

    def transition_matrix(self, capacity: float, action: int) -> np.ndarray:
        key = (self._capacity_key(capacity), int(action))
        if key in self._transition_cache:
            return self._transition_cache[key]
        started = time.perf_counter()
        represented_capacity = self._key_capacity(key[0])
        matrix = np.zeros((self.hidden_count, self.hidden_count), dtype=np.float64)
        bins = len(self.abundance_grid)
        for regime in range(self.regime_count):
            regime_probs = (
                self.model.regime_matrix[regime]
                if self.regime_count == 2 else np.asarray([1.0])
            )
            means = self.model.noiseless_next(self.abundance_grid, represented_capacity, action, regime)
            for abundance_index, mean in enumerate(means):
                source = regime * bins + abundance_index
                rng = np.random.default_rng(
                    self.seed + key[0] * 1_000_003 + int(action) * 10_007
                    + regime * 1_009 + abundance_index
                )
                count = self.config.transition_samples
                if float(mean) <= 0.0:
                    samples = np.zeros(count, dtype=np.float64)
                else:
                    samples = float(mean) * np.exp(
                        self.model.process_scale * rng.normal(size=count)
                    )
                if self.regime_count == 1:
                    matrix[source, :bins] = self._deposit_samples(samples)
                else:
                    next_regimes = rng.choice(2, size=count, p=regime_probs)
                    for next_regime in range(2):
                        selected = samples[next_regimes == next_regime]
                        if len(selected):
                            start = next_regime * bins
                            matrix[source, start:start + bins] = (
                                len(selected) / count * self._deposit_samples(selected)
                            )
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
            raise AssertionError("adapted transition kernel is not row stochastic")
        self._transition_cache[key] = matrix
        self.kernel_build_seconds += time.perf_counter() - started
        self.kernel_build_count += 1
        return matrix

    def state_abundances(self) -> np.ndarray:
        return np.tile(self.abundance_grid, self.regime_count)

    def observation_vector(self, observation: float) -> np.ndarray:
        return np.tile(
            self.model.observation_likelihood(observation, self.abundance_grid),
            self.regime_count,
        )

    def initial_belief(self, observation: float) -> CandidateBelief:
        grid = self.abundance_grid
        prior = np.zeros(len(grid), dtype=np.float64)
        prior[1:] = np.exp(
            -0.5 * ((np.log(grid[1:]) - self.model.reset_log_mean)
                    / self.model.reset_log_scale) ** 2
        ) / np.maximum(grid[1:], 1e-300)
        prior = _normalize(prior)
        if self.regime_count == 2:
            prior = np.concatenate((0.5 * prior, 0.5 * prior))
        likelihood = self.observation_vector(observation)
        probabilities = _normalize(prior * likelihood)
        return CandidateBelief(
            probabilities,
            self.model.initial_capacity,
            float(observation),
            float(observation),
            0,
        )

    def predict(self, belief: CandidateBelief, action: int) -> np.ndarray:
        return _normalize(
            belief.probabilities @ self.transition_matrix(belief.capacity, action)
        )

    def update(
        self, belief: CandidateBelief, action: int, observation: float
    ) -> tuple[CandidateBelief, float]:
        predicted = self.predict(belief, action)
        likelihood = self.observation_vector(observation)
        evidence = float(np.dot(predicted, likelihood))
        if evidence <= LIKELIHOOD_FLOOR or not np.isfinite(evidence):
            posterior = predicted
            log_evidence = float(np.log(LIKELIHOOD_FLOOR))
        else:
            posterior = _normalize(predicted * likelihood)
            log_evidence = float(np.log(evidence))
        return (
            CandidateBelief(
                posterior,
                self.model.next_capacity(belief.capacity, action),
                belief.current_observation,
                float(observation),
                belief.timestep + 1,
            ),
            log_evidence,
        )

    def expected_public_reward(
        self,
        belief: CandidateBelief,
        action: int,
        predicted: np.ndarray | None = None,
    ) -> float:
        predicted = self.predict(belief, action) if predicted is None else predicted
        following = self.state_abundances() * self.model.survey_scale
        count = len(following)
        surrogate = self.context.surrogate
        if surrogate is None:
            return float(-self.context.action_costs[int(action)])
        rewards, _risks = surrogate.predict(
            np.full(count, belief.previous_observation),
            np.full(count, belief.current_observation),
            following,
            np.full(count, int(action), dtype=np.int64),
            np.full(count, min(belief.timestep, self.context.horizon - 1), dtype=np.int64),
            np.full(count, self.context.pop_id),
        )
        return float(np.dot(predicted, rewards))

    def representative_observations(
        self, predicted: np.ndarray, branch_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = np.zeros(len(self.abundance_grid), dtype=np.float64)
        bins = len(self.abundance_grid)
        for regime in range(self.regime_count):
            probabilities += predicted[regime * bins:(regime + 1) * bins]
        cdf = np.cumsum(_normalize(probabilities))
        quantiles = (np.arange(branch_count) + 0.5) / branch_count
        indices = np.minimum(np.searchsorted(cdf, quantiles), len(cdf) - 1)
        observations = self.abundance_grid[indices] * self.model.survey_scale
        return observations.astype(np.float64), np.full(branch_count, 1.0 / branch_count)

    def model_hash(self) -> str:
        digest = hashlib.sha256(self.model.parameter_hash().encode("ascii"))
        digest.update(np.ascontiguousarray(self.abundance_grid).tobytes())
        digest.update(str(self.config).encode("utf-8"))
        return digest.hexdigest()
