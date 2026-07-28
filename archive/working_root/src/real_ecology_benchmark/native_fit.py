"""Closed-world mechanistic-form fitting from sanitized public transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MethodContext, ModelConfig, PlannerConfig
from .dataset import TrajectoryDataset


PUBLIC_FORM_CANDIDATES = ("ricker", "allee", "theta", "regime")


def _basis(family: str, values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if family == "ricker":
        return np.column_stack([np.ones(len(x)), x])
    if family == "allee":
        return np.column_stack([np.ones(len(x)), x, x * x])
    if family == "theta":
        return np.column_stack([np.ones(len(x)), x, np.sqrt(np.maximum(x, 0.0)), x * x])
    if family == "regime":
        return np.column_stack([np.ones(len(x)), x, x * x, (x < 1.0).astype(float)])
    raise ValueError(f"unknown public form candidate {family!r}")


@dataclass(frozen=True)
class FittedPublicForm:
    form: str
    coefficients: np.ndarray
    residual_sigma: np.ndarray
    observation_scale: float
    num_actions: int
    fit_loss: float

    def predict(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float64)
        actions = np.broadcast_to(np.asarray(actions, dtype=int), observations.shape)
        x = np.log1p(np.maximum(observations, 0.0) / self.observation_scale)
        output = np.empty_like(observations)
        for action in np.unique(actions):
            mask = actions == action
            value = _basis(self.form, x[mask]) @ self.coefficients[int(action)]
            output[mask] = self.observation_scale * np.expm1(np.clip(value, 0.0, 50.0))
        return np.maximum(output, 0.0)


def fit_from_public_data(
    dataset: TrajectoryDataset,
    context: MethodContext,
    family: str,
    ridge: float = 1e-3,
) -> FittedPublicForm:
    """Fit one candidate using no simulator config, table, path, or sidecar."""

    if family not in PUBLIC_FORM_CANDIDATES:
        raise ValueError(f"unsupported candidate family {family!r}")
    context.validate()
    scale = context.observation_scale
    x = np.log1p(np.maximum(dataset.observations, 0.0) / scale)
    target = np.log1p(np.maximum(dataset.next_observations, 0.0) / scale)
    dimension = _basis(family, np.asarray([0.0])).shape[1]
    coefficients = np.zeros((context.num_actions, dimension), dtype=np.float64)
    residual_sigma = np.full(context.num_actions, 0.2, dtype=np.float64)
    predictions = np.zeros(len(dataset), dtype=np.float64)
    global_design = _basis(family, x)
    penalty = ridge * np.eye(dimension)
    penalty[0, 0] = 0.0
    global_coefficients = np.linalg.solve(
        global_design.T @ global_design + penalty,
        global_design.T @ target,
    )
    for action in range(context.num_actions):
        mask = dataset.actions == action
        if np.sum(mask) < dimension + 1:
            coefficients[action] = global_coefficients
        else:
            design = _basis(family, x[mask])
            coefficients[action] = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ target[mask],
            )
        action_prediction = _basis(family, x[mask]) @ coefficients[action]
        predictions[mask] = action_prediction
        if np.any(mask):
            residual_sigma[action] = max(
                float(np.std(target[mask] - action_prediction)), 0.02
            )
    loss = float(np.mean((predictions - target) ** 2))
    return FittedPublicForm(
        family,
        coefficients,
        residual_sigma,
        scale,
        context.num_actions,
        loss,
    )


@dataclass(frozen=True)
class PublicDiscreteGrid:
    hidden_state_values: np.ndarray

    @property
    def num_hidden(self) -> int:
        return len(self.hidden_state_values)

    @property
    def hidden_regime_values(self) -> np.ndarray:
        return np.zeros(self.num_hidden, dtype=np.int8)

    @staticmethod
    def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
        values = np.asarray(log_weights, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.full(len(values), -np.log(len(values)))
        maximum = float(np.max(values[finite]))
        total = float(np.sum(np.exp(values[finite] - maximum)))
        output = np.full(len(values), -np.inf)
        output[finite] = values[finite] - maximum - np.log(total)
        return output


class FittedNativeSolver:
    """Tabular solver whose transitions and rewards are entirely public-data fitted."""

    def __init__(
        self,
        fitted: FittedPublicForm,
        context: MethodContext,
        model_cfg: ModelConfig,
        planner_cfg: PlannerConfig,
        dataset: TrajectoryDataset,
    ):
        self.fitted = fitted
        self.context = context
        upper = max(
            float(np.quantile(np.concatenate([dataset.observations, dataset.next_observations]), 0.995)),
            context.observation_scale,
        )
        state_values = np.linspace(0.0, upper * 1.25, model_cfg.native_state_bins)
        self.grid = PublicDiscreteGrid(state_values)
        self.transition = self._transition_matrix()
        self.reward = self._reward_table()
        self.q_values = self._solve(
            planner_cfg.discount,
            model_cfg.native_vi_iterations,
            model_cfg.native_vi_tolerance,
        )

    def _transition_matrix(self) -> np.ndarray:
        states = self.grid.hidden_state_values
        count = len(states)
        matrix = np.empty((self.context.num_actions, count, count), dtype=np.float64)
        log_grid = np.log1p(states / self.context.observation_scale)
        for action in range(self.context.num_actions):
            actions = np.full(count, action, dtype=int)
            mean = self.fitted.predict(states, actions)
            mean_log = np.log1p(mean / self.context.observation_scale)
            sigma = max(float(self.fitted.residual_sigma[action]), 0.02)
            distance = (log_grid[None, :] - mean_log[:, None]) / sigma
            probability = np.exp(-0.5 * distance * distance)
            probability /= np.maximum(probability.sum(axis=1, keepdims=True), 1e-300)
            matrix[action] = probability
        return matrix

    def _reward_table(self) -> np.ndarray:
        states = self.grid.hidden_state_values
        output = np.empty((len(states), self.context.num_actions), dtype=np.float64)
        for action in range(self.context.num_actions):
            actions = np.full(len(states), action, dtype=int)
            following = self.fitted.predict(states, actions)
            reward, _risk = self.context.surrogate.predict(
                states,
                states,
                following,
                actions,
                np.zeros(len(states), dtype=int),
                np.full(len(states), self.context.pop_id),
            )
            output[:, action] = reward
        return output

    def _solve(self, discount: float, iterations: int, tolerance: float) -> np.ndarray:
        value = np.zeros(self.grid.num_hidden, dtype=np.float64)
        q_values = np.zeros((self.grid.num_hidden, self.context.num_actions))
        for _ in range(iterations):
            for action in range(self.context.num_actions):
                q_values[:, action] = self.reward[:, action] + discount * (
                    self.transition[action] @ value
                )
            updated = np.max(q_values, axis=1)
            if float(np.max(np.abs(updated - value))) < tolerance:
                value = updated
                break
            value = updated
        for action in range(self.context.num_actions):
            q_values[:, action] = self.reward[:, action] + discount * (
                self.transition[action] @ value
            )
        return q_values

    def _emission_log_prob(self, observation: float) -> np.ndarray:
        states = self.grid.hidden_state_values
        sigma = max(self.context.observation_noise_sigma, 0.02)
        if observation <= 0.0:
            return np.where(states == 0.0, 0.0, -np.inf)
        log_observation = np.log(max(observation, 1e-12))
        log_state = np.log(np.maximum(states, 1e-12))
        return -0.5 * ((log_observation - log_state) / sigma) ** 2 - np.log(sigma)

    def initial_log_weights(self, observation: float) -> np.ndarray:
        return self.grid.normalize_log_weights(self._emission_log_prob(observation))

    def update_log_weights(self, log_weights, control_state, action, observation):
        del control_state
        weights = np.exp(self.grid.normalize_log_weights(log_weights))
        predicted = weights @ self.transition[int(action)]
        prior = np.log(np.maximum(predicted, 1e-300))
        unnormalized = prior + self._emission_log_prob(float(observation))
        normalized = self.grid.normalize_log_weights(unnormalized)
        finite = np.isfinite(unnormalized)
        if not np.any(finite):
            evidence = -1e6
        else:
            maximum = float(np.max(unnormalized[finite]))
            evidence = float(
                maximum + np.log(np.sum(np.exp(unnormalized[finite] - maximum)))
            )
        return normalized, None, evidence

    def action_values(self, log_weights, control_state=None) -> np.ndarray:
        del control_state
        weights = np.exp(self.grid.normalize_log_weights(log_weights))
        return weights @ self.q_values


def build_fitted_solver(
    dataset: TrajectoryDataset,
    context: MethodContext,
    family: str,
    model_cfg: ModelConfig,
    planner_cfg: PlannerConfig,
) -> tuple[FittedPublicForm, FittedNativeSolver]:
    fitted = fit_from_public_data(dataset, context, family, ridge=model_cfg.ridge)
    return fitted, FittedNativeSolver(fitted, context, model_cfg, planner_cfg, dataset)
