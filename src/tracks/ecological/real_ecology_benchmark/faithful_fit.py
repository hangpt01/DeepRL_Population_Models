"""Ordered-episode autodiff fitting for adapted mechanistic ecology models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import FaithfulFitConfig, FaithfulModelConfig, MethodContext
from .dataset import TrajectoryDataset, dataset_sha256
from .faithful_ecology import (
    EQUATION_VERSION,
    REGIME_LAW_VERSION,
    MechanisticModel,
    REGISTERED_FORMS,
    fixed_regime_matrix,
)


FIT_SCHEMA_VERSION = "adapted_mechanistic_fit_v2"
FIT_TRANSITION_HASH_VERSION = "fit_transition_hash_v1"
CANDIDATE_CONSTRUCTION = "episode_bootstrap_map_fixed_pi_v2"
RICKER_ONLY_CANDIDATE_CONSTRUCTION = (
    "ricker_only_episode_bootstrap_map_1full_7bootstrap_v1"
)
ACTION_PARAMETER_COUNT = 14


def conditional_survey_mean_factor(sigma: float) -> float:
    """E[exp(epsilon)] for epsilon ~ Normal(0, sigma^2)."""

    if sigma < 0.0:
        raise ValueError("survey noise sigma must be non-negative")
    return float(np.exp(float(sigma) ** 2 / 2.0))


@dataclass(frozen=True)
class FitResult:
    model: MechanisticModel
    objective: float
    holdout_normalized_survey_sse: float
    selected_gradient_norm: float
    curvature_condition_estimate: float
    start_gradient_norms: tuple[float, ...]
    start_objective_traces: tuple[tuple[float, ...], ...]
    finite_start_parameter_std_mean: float
    selected_start: int
    start_objectives: tuple[float, ...]
    action_rows: tuple[int, ...]
    action_episodes: tuple[int, ...]
    sparse_actions: tuple[int, ...]
    fit_episode_ids: tuple[int, ...]
    holdout_episode_ids: tuple[int, ...]
    public_data_hash: str
    random_bank_hash: str
    optimizer: str
    iterations: int
    transition_data_hash: str = ""
    regularization_variant: str = "none_structural_v1"
    action_parameter_count: int = ACTION_PARAMETER_COUNT
    fixed_regime_persistence: float = 0.90
    fit_cache_key: str = ""

    def diagnostics(self) -> dict[str, Any]:
        return {
            "fit_schema": FIT_SCHEMA_VERSION,
            "fit_transition_hash_version": FIT_TRANSITION_HASH_VERSION,
            "equation_version": EQUATION_VERSION,
            "regime_law_version": REGIME_LAW_VERSION,
            "regime_law_hash": self.model.regime_law_hash,
            "form": self.model.form,
            "objective": float(self.objective),
            "holdout_normalized_survey_sse": (
                float(self.holdout_normalized_survey_sse)
                if np.isfinite(self.holdout_normalized_survey_sse)
                else None
            ),
            "selected_gradient_norm": float(self.selected_gradient_norm),
            "curvature_condition_estimate": float(self.curvature_condition_estimate),
            "start_gradient_norms": [
                float(value) if np.isfinite(value) else None for value in self.start_gradient_norms
            ],
            "start_objective_traces": [
                [float(value) if np.isfinite(value) else None for value in trace]
                for trace in self.start_objective_traces
            ],
            "finite_start_parameter_std_mean": float(self.finite_start_parameter_std_mean),
            "fitted_process_scale": float(self.model.process_scale),
            "public_observation_scale": float(self.model.observation_scale),
            "selected_start": int(self.selected_start),
            "start_objectives": [
                float(value) if np.isfinite(value) else None for value in self.start_objectives
            ],
            "action_rows": list(self.action_rows),
            "action_episodes": list(self.action_episodes),
            "sparse_actions": list(self.sparse_actions),
            "fit_episode_ids": list(self.fit_episode_ids),
            "holdout_episode_ids": list(self.holdout_episode_ids),
            "public_data_hash": self.public_data_hash,
            "transition_data_hash": self.transition_data_hash,
            "random_bank_hash": self.random_bank_hash,
            "optimizer": self.optimizer,
            "iterations": int(self.iterations),
            "regularization_variant": self.regularization_variant,
            "action_parameter_count": int(self.action_parameter_count),
            "fixed_regime_persistence": float(self.fixed_regime_persistence),
            "fit_cache_key": self.fit_cache_key,
            "parameter_hash": self.model.parameter_hash(),
        }


@dataclass(frozen=True)
class CandidateBank:
    fits: tuple[FitResult, ...]
    initial_weights: np.ndarray
    prior_type: str
    construction_type: str = CANDIDATE_CONSTRUCTION

    def __post_init__(self) -> None:
        weights = np.asarray(self.initial_weights, dtype=np.float64)
        if len(weights) != len(self.fits) or len(weights) == 0:
            raise ValueError("candidate prior must cover a non-empty bank")
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("candidate prior must be a probability vector")
        if self.construction_type not in {
            CANDIDATE_CONSTRUCTION,
            RICKER_ONLY_CANDIDATE_CONSTRUCTION,
        }:
            raise ValueError("unknown adapted candidate construction")
        hashes = [fit.model.parameter_hash() for fit in self.fits]
        if len(set(hashes)) != len(hashes):
            raise ValueError("candidate bank contains duplicate parameterizations")
        object.__setattr__(self, "initial_weights", weights)


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "adapted mechanistic fitting requires the optional CPU PyTorch dependency; "
            "no regression or finite-difference fallback is allowed"
        ) from exc
    return torch


def fit_transition_hash(dataset: TrajectoryDataset) -> str:
    """Hash only the seven registered public transition fields."""

    digest = hashlib.sha256(FIT_TRANSITION_HASH_VERSION.encode("ascii"))
    for name in (
        "observations",
        "actions",
        "next_observations",
        "episode_id",
        "timestep",
        "terminated",
        "truncated",
    ):
        value = getattr(dataset, name)
        if value is None:
            raise ValueError(f"fit transition field {name!r} is missing")
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _episode_ids(dataset: TrajectoryDataset) -> np.ndarray:
    return np.asarray(sorted(int(value) for value in np.unique(dataset.episode_id)))


def split_history_episodes(
    dataset: TrajectoryDataset, fraction: float, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    episodes = _episode_ids(dataset)
    if len(episodes) < 2:
        return tuple(int(x) for x in episodes), ()
    shuffled = episodes.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    cut = max(1, min(len(shuffled) - 1, int(round(fraction * len(shuffled)))))
    return (
        tuple(sorted(int(x) for x in shuffled[:cut])),
        tuple(sorted(int(x) for x in shuffled[cut:])),
    )


def _ordered_indices(dataset: TrajectoryDataset, episode_ids: Iterable[int]) -> list[np.ndarray]:
    output = []
    for episode in episode_ids:
        idx = np.flatnonzero(dataset.episode_id == int(episode))
        idx = idx[np.argsort(dataset.timestep[idx], kind="stable")]
        if len(idx):
            output.append(idx)
    return output


def _coverage(
    dataset: TrajectoryDataset, num_actions: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rows, episodes = [], []
    for action in range(num_actions):
        mask = dataset.actions.astype(int) == action
        rows.append(int(np.sum(mask)))
        episodes.append(int(len(np.unique(dataset.episode_id[mask]))))
    return tuple(rows), tuple(episodes)


def _random_bank(
    episode_lengths: list[int], paths: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    episodes = len(episode_lengths)
    width = max(episode_lengths, default=1)
    initial = rng.normal(size=(paths, episodes))
    process = rng.normal(size=(paths, episodes, width))
    regime_uniforms = rng.random(size=(paths, episodes, width + 1))
    digest = hashlib.sha256()
    for value in (initial, process, regime_uniforms):
        digest.update(np.ascontiguousarray(value).tobytes())
    return initial, process, regime_uniforms, digest.hexdigest()


def _inverse_sigmoid(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(value / (1.0 - value))


def _inverse_signed_rate(value: np.ndarray | float) -> np.ndarray:
    scaled = (np.asarray(value, dtype=np.float64) - 0.25) / 1.75
    return np.arctanh(np.clip(scaled, -1.0 + 1e-6, 1.0 - 1e-6))


def _action_layout(channels: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rate_actions = tuple(
        index for index, channel in enumerate(channels) if channel in {"rate", "rate+capacity"}
    )
    capacity_actions = tuple(
        index for index, channel in enumerate(channels) if channel in {"capacity", "rate+capacity"}
    )
    return rate_actions, capacity_actions


def action_parameter_count(channels: tuple[str, ...]) -> int:
    rate_actions, capacity_actions = _action_layout(channels)
    stocking_actions = sum(channel == "state" for channel in channels)
    return 1 + len(rate_actions) + len(capacity_actions) + stocking_actions


def _initial_raw(
    context: MethodContext,
    form: str,
    model_cfg: FaithfulModelConfig,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower_log = np.log(1e-3)
    upper_log = np.log(model_cfg.abundance_upper)
    reset_fraction = (np.log(0.8) - lower_log) / (upper_log - lower_log)
    initial_capacity_fraction = (1.0 - 0.05) / (model_cfg.capacity_upper - 0.05)
    ceiling_fraction = (2.0 - 1.0) / (model_cfg.capacity_upper - 1.0)
    rate_actions, capacity_actions = _action_layout(context.action_channels)
    rate_starts = np.asarray([0.12] + [(-0.08 if i % 2 else 0.18) for i in rate_actions])
    values = [
        np.asarray(
            [
                float(_inverse_sigmoid(reset_fraction)),
                float(_inverse_sigmoid((0.2 - 0.01) / 1.49)),
                float(_inverse_sigmoid(initial_capacity_fraction)),
                float(_inverse_sigmoid(ceiling_fraction)),
                float(_inverse_sigmoid(0.1 / model_cfg.process_scale_upper)),
            ]
        ),
        _inverse_signed_rate(rate_starts),
        _inverse_sigmoid(np.full(len(capacity_actions), 0.02 / model_cfg.action_capacity_upper)),
        _inverse_sigmoid(
            np.full(
                sum(x == "state" for x in context.action_channels),
                0.01 / model_cfg.action_stocking_upper,
            )
        ),
    ]
    if form == "allee":
        values.append(_inverse_sigmoid(np.asarray([0.2])))
    elif form == "theta":
        values.append(_inverse_sigmoid(np.asarray([0.375])))
    elif form == "regime":
        values.extend(
            (
                _inverse_sigmoid(np.asarray([0.2, 0.35])),
                _inverse_sigmoid(np.asarray([0.25, 0.55])),
            )
        )
    raw = np.concatenate(values).astype(np.float64)
    raw += rng.normal(0.0, 0.25, size=len(raw))
    return raw


def _symmetric_positive(value, torch):
    class SymmetricPositive(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_value):
            ctx.save_for_backward(input_value)
            return torch.clamp(input_value, min=0.0)

        @staticmethod
        def backward(ctx, grad_output):
            (input_value,) = ctx.saved_tensors
            slope = torch.where(
                input_value > 0.0,
                torch.ones_like(input_value),
                torch.where(
                    input_value < 0.0,
                    torch.zeros_like(input_value),
                    torch.full_like(input_value, 0.5),
                ),
            )
            return grad_output * slope

    return SymmetricPositive.apply(value)


def _decode(
    raw,
    context: MethodContext,
    form: str,
    model_cfg: FaithfulModelConfig,
    regime_persistence: float,
    torch,
):
    sigmoid = torch.sigmoid
    pos = 0
    lower_log = float(np.log(1e-3))
    upper_log = float(np.log(model_cfg.abundance_upper))
    reset_mean = lower_log + (upper_log - lower_log) * sigmoid(raw[pos])
    pos += 1
    reset_scale = 0.01 + 1.49 * sigmoid(raw[pos])
    pos += 1
    initial_capacity = 0.05 + (model_cfg.capacity_upper - 0.05) * sigmoid(raw[pos])
    pos += 1
    capacity_ceiling = initial_capacity + (model_cfg.capacity_upper - initial_capacity) * sigmoid(
        raw[pos]
    )
    pos += 1
    process_scale = model_cfg.process_scale_upper * sigmoid(raw[pos])
    pos += 1

    rate_actions, capacity_actions = _action_layout(context.action_channels)
    rate_count = 1 + len(rate_actions)
    signed_free = 0.25 + 1.75 * torch.tanh(raw[pos : pos + rate_count])
    pos += rate_count
    baseline = signed_free[0]
    specific = {action: signed_free[index + 1] for index, action in enumerate(rate_actions)}
    signed_rates = torch.stack(
        [specific.get(action, baseline) for action in range(context.num_actions)]
    )
    growth = _symmetric_positive(signed_rates, torch)
    mortality = _symmetric_positive(-signed_rates, torch)

    free_capacity = model_cfg.action_capacity_upper * sigmoid(
        raw[pos : pos + len(capacity_actions)]
    )
    pos += len(capacity_actions)
    capacity_lookup = {
        action: free_capacity[index] for index, action in enumerate(capacity_actions)
    }
    zero = raw.new_zeros(())
    capacity_increment = torch.stack(
        [capacity_lookup.get(action, zero) for action in range(context.num_actions)]
    )

    state_actions = tuple(
        index for index, channel in enumerate(context.action_channels) if channel == "state"
    )
    free_stocking = model_cfg.action_stocking_upper * sigmoid(raw[pos : pos + len(state_actions)])
    pos += len(state_actions)
    stocking_lookup = {action: free_stocking[index] for index, action in enumerate(state_actions)}
    stocking = torch.stack(
        [stocking_lookup.get(action, zero) for action in range(context.num_actions)]
    )

    thresholds = initial_capacity * raw.new_tensor([0.2, 0.35])
    theta_exponent = raw.new_tensor(2.0)
    multipliers = raw.new_tensor([0.7, 1.2])
    if form == "allee":
        threshold = (0.05 + 0.75 * sigmoid(raw[pos])) * initial_capacity
        pos += 1
        thresholds = torch.stack((threshold, threshold))
    elif form == "theta":
        theta_exponent = 0.2 + 4.8 * sigmoid(raw[pos])
        pos += 1
    elif form == "regime":
        threshold_fraction = 0.05 + 0.75 * sigmoid(raw[pos : pos + 2])
        pos += 2
        thresholds = threshold_fraction * initial_capacity
        multipliers = 0.25 + 1.75 * sigmoid(raw[pos : pos + 2])
        pos += 2
    if pos != len(raw):
        raise AssertionError("adapted parameter decoder consumed the wrong dimension")
    matrix = torch.tensor(fixed_regime_matrix(regime_persistence), dtype=torch.float64)
    return {
        "reset_mean": reset_mean,
        "reset_scale": reset_scale,
        "initial_capacity": initial_capacity,
        "capacity_ceiling": capacity_ceiling,
        "process_scale": process_scale,
        "growth": growth,
        "mortality": mortality,
        "signed_rates": signed_rates,
        "free_signed_rates": signed_free,
        "capacity_increment": capacity_increment,
        "free_capacity": free_capacity,
        "stocking": stocking,
        "thresholds": thresholds,
        "theta_exponent": theta_exponent,
        "multipliers": multipliers,
        "matrix": matrix,
    }


def _trajectory_objective(
    raw,
    dataset: TrajectoryDataset,
    ordered: list[np.ndarray],
    context: MethodContext,
    form: str,
    model_cfg: FaithfulModelConfig,
    fit_cfg: FaithfulFitConfig,
    random_bank: tuple[np.ndarray, np.ndarray, np.ndarray],
    regime_persistence: float,
    torch,
):
    p = _decode(raw, context, form, model_cfg, regime_persistence, torch)
    initial_noise, process_noise, regime_uniforms = [
        torch.as_tensor(value, dtype=torch.float64) for value in random_bank
    ]
    total = torch.zeros((), dtype=torch.float64)
    count = 0
    scale = float(context.observation_scale)
    survey_mean_factor = conditional_survey_mean_factor(context.observation_noise_sigma)
    for episode_position, indices in enumerate(ordered):
        capacity = p["initial_capacity"]
        latent = torch.exp(p["reset_mean"] + p["reset_scale"] * initial_noise[:, episode_position])
        regime = (regime_uniforms[:, episode_position, 0] >= 0.5).to(torch.int64)
        first = int(indices[0])
        target = float(dataset.observations[first]) / scale
        total = total + torch.mean((latent * survey_mean_factor - target) ** 2)
        count += 1
        for step, row in enumerate(indices):
            action = int(dataset.actions[row])
            mean, capacity = _torch_noiseless_transition(
                p, form, latent, capacity, action, regime, torch
            )
            latent = mean * torch.exp(p["process_scale"] * process_noise[:, episode_position, step])
            target = float(dataset.next_observations[row]) / scale
            total = total + torch.mean((latent * survey_mean_factor - target) ** 2)
            count += 1
            if form == "regime":
                probability_zero = p["matrix"][regime, 0]
                regime = (regime_uniforms[:, episode_position, step + 1] > probability_zero).to(
                    torch.int64
                )
            if dataset.terminated is not None and bool(dataset.terminated[row]):
                break
    objective = total / max(count, 1)
    if fit_cfg.regularization_variant == "hierarchical_weak_v1":
        rates = p["free_signed_rates"]
        capacities = p["free_capacity"]
        objective = objective + fit_cfg.hierarchical_rate_penalty * torch.mean(
            (rates - torch.mean(rates)) ** 2
        )
        if len(capacities):
            objective = objective + fit_cfg.hierarchical_capacity_penalty * torch.mean(
                (capacities - torch.mean(capacities)) ** 2
            )
    return objective


def _torch_noiseless_transition(
    parameters, form: str, latent, capacity, action: int, regime, torch
):
    """Autodiff counterpart of the deployed noiseless transition."""

    next_capacity = torch.minimum(
        parameters["capacity_ceiling"],
        capacity + parameters["capacity_increment"][action],
    )
    managed = torch.clamp(latent + parameters["stocking"][action], min=0.0)
    growth = parameters["growth"][action]
    mortality = parameters["mortality"][action]
    if form == "ricker":
        exponent = growth * (1.0 - managed / next_capacity) - mortality
        mean = managed * torch.exp(torch.clamp(exponent, -40.0, 40.0))
    elif form == "allee":
        exponent = (
            growth * (1.0 - managed / next_capacity) * (managed / parameters["thresholds"][0] - 1.0)
            - mortality
        )
        mean = managed * torch.exp(torch.clamp(exponent, -40.0, 40.0))
    elif form == "theta":
        core = torch.clamp(
            managed
            + growth * managed * (1.0 - (managed / next_capacity) ** parameters["theta_exponent"]),
            min=0.0,
        )
        mean = core * torch.exp(-mortality)
    else:
        threshold = parameters["thresholds"][regime]
        multiplier = parameters["multipliers"][regime]
        exponent = (
            multiplier * growth * (1.0 - managed / next_capacity) * (managed / threshold - 1.0)
            - mortality
        )
        mean = managed * torch.exp(torch.clamp(exponent, -40.0, 40.0))
    mean = torch.where(
        (latent == 0.0) & (parameters["stocking"][action] == 0.0),
        torch.zeros_like(mean),
        mean,
    )
    return mean, next_capacity


def _to_model(
    raw: np.ndarray,
    context: MethodContext,
    form: str,
    model_cfg: FaithfulModelConfig,
    candidate_id: str,
    regime_persistence: float,
) -> MechanisticModel:
    torch = require_torch()
    values = _decode(
        torch.as_tensor(raw, dtype=torch.float64),
        context,
        form,
        model_cfg,
        regime_persistence,
        torch,
    )

    def value(name: str) -> np.ndarray:
        return values[name].detach().cpu().numpy()

    return MechanisticModel(
        form=form,
        growth=value("growth"),
        mortality=value("mortality"),
        capacity_increment=value("capacity_increment"),
        stocking=value("stocking"),
        reset_log_mean=float(value("reset_mean")),
        reset_log_scale=float(value("reset_scale")),
        initial_capacity=float(value("initial_capacity")),
        capacity_ceiling=float(value("capacity_ceiling")),
        process_scale=float(value("process_scale")),
        observation_scale=float(context.observation_noise_sigma),
        survey_scale=float(context.observation_scale),
        depensation_thresholds=value("thresholds"),
        theta_exponent=float(value("theta_exponent")),
        regime_multipliers=value("multipliers"),
        regime_matrix=value("matrix"),
        action_channels=context.action_channels,
        candidate_id=candidate_id,
    )


def fit_mechanistic_model(
    dataset: TrajectoryDataset,
    context: MethodContext,
    form: str,
    model_cfg: FaithfulModelConfig,
    fit_cfg: FaithfulFitConfig,
    seed: int,
    candidate_id: str = "model_000",
    fit_episode_ids: tuple[int, ...] | None = None,
    holdout_episode_ids: tuple[int, ...] | None = None,
    regime_persistence: float = 0.90,
) -> FitResult:
    """Fit one fixed candidate from ordered public action-observation episodes."""

    if form not in REGISTERED_FORMS:
        raise ValueError(f"unknown mechanistic form {form!r}")
    context.validate()
    dataset.validate()
    if (
        context.num_actions == 11
        and action_parameter_count(context.action_channels) != ACTION_PARAMETER_COUNT
    ):
        raise ValueError(
            "corrected real-ecology channel map must expose exactly 14 action parameters"
        )
    if regime_persistence not in model_cfg.regime_persistence_grid:
        raise ValueError("candidate regime persistence is outside the registered grid")
    torch = require_torch()
    torch.set_default_dtype(torch.float64)
    if fit_episode_ids is None:
        fit_episode_ids, inferred_holdout = split_history_episodes(
            dataset, fit_cfg.history_fraction, seed
        )
        if holdout_episode_ids is None:
            holdout_episode_ids = inferred_holdout
    holdout_episode_ids = holdout_episode_ids or ()
    ordered = _ordered_indices(dataset, fit_episode_ids)
    if not ordered:
        raise ValueError("adapted fit received no complete training episodes")
    path_count = fit_cfg.regime_mc_paths if form == "regime" else fit_cfg.mc_paths
    initial, process, regime_uniforms, random_hash = _random_bank(
        [len(indices) for indices in ordered], path_count, seed + 17
    )
    random_values = (initial, process, regime_uniforms)
    objectives: list[float] = []
    raw_values: list[np.ndarray | None] = []
    gradient_norms: list[float] = []
    traces: list[tuple[float, ...]] = []
    curvature_conditions: list[float] = []
    for start in range(fit_cfg.starts):
        initial_raw = _initial_raw(context, form, model_cfg, seed + 1009 * start)
        raw = torch.nn.Parameter(torch.as_tensor(initial_raw, dtype=torch.float64))
        optimizer = torch.optim.LBFGS(
            [raw],
            lr=fit_cfg.learning_rate,
            max_iter=fit_cfg.iterations,
            tolerance_grad=fit_cfg.tolerance_grad,
            tolerance_change=fit_cfg.tolerance_change,
            line_search_fn="strong_wolfe",
        )
        objective_trace: list[float] = []

        def closure():
            optimizer.zero_grad()
            objective = _trajectory_objective(
                raw,
                dataset,
                ordered,
                context,
                form,
                model_cfg,
                fit_cfg,
                random_values,
                regime_persistence,
                torch,
            )
            objective.backward()
            objective_trace.append(float(objective.detach().cpu()))
            return objective

        try:
            optimizer.step(closure)
            objective = float(closure().detach().cpu())
            final_raw = raw.detach().cpu().numpy().copy()
            if not np.isfinite(objective) or not np.all(np.isfinite(final_raw)):
                raise FloatingPointError("non-finite adapted fit")
            objectives.append(objective)
            raw_values.append(final_raw)
            gradient_norms.append(float(raw.grad.detach().norm().cpu()))
            state = optimizer.state.get(raw, {})
            ratios = []
            for direction, step_vector in zip(state.get("old_dirs", ()), state.get("old_stps", ())):
                denominator = float(torch.dot(step_vector, step_vector).detach().cpu())
                numerator = float(torch.dot(direction, step_vector).detach().cpu())
                if denominator > 0.0 and numerator > 0.0:
                    ratios.append(numerator / denominator)
            curvature_conditions.append(float(max(ratios) / min(ratios)) if ratios else -1.0)
            traces.append(tuple(objective_trace))
        except (RuntimeError, FloatingPointError):
            objectives.append(float("inf"))
            raw_values.append(None)
            gradient_norms.append(float("inf"))
            curvature_conditions.append(-1.0)
            traces.append(tuple(objective_trace))
    finite = np.flatnonzero(np.isfinite(objectives))
    if len(finite) == 0:
        raise RuntimeError("all registered automatic-differentiation starts failed")
    selected = int(min(finite, key=lambda index: objectives[int(index)]))
    model = _to_model(
        raw_values[selected], context, form, model_cfg, candidate_id, regime_persistence
    )
    finite_parameters = np.asarray([raw_values[int(index)] for index in finite])
    parameter_std_mean = (
        float(np.mean(np.std(finite_parameters, axis=0))) if len(finite_parameters) > 1 else 0.0
    )
    action_rows, action_episodes = _coverage(dataset, context.num_actions)
    threshold = max(fit_cfg.sparse_action_rows, 20)
    return FitResult(
        model=model,
        objective=float(objectives[selected]),
        holdout_normalized_survey_sse=float(
            ordered_trajectory_sse(model, dataset, holdout_episode_ids)
        ),
        selected_gradient_norm=float(gradient_norms[selected]),
        curvature_condition_estimate=float(curvature_conditions[selected]),
        start_gradient_norms=tuple(float(x) for x in gradient_norms),
        start_objective_traces=tuple(traces),
        finite_start_parameter_std_mean=parameter_std_mean,
        selected_start=selected,
        start_objectives=tuple(float(x) for x in objectives),
        action_rows=action_rows,
        action_episodes=action_episodes,
        sparse_actions=tuple(i for i, rows in enumerate(action_rows) if rows < threshold),
        fit_episode_ids=tuple(int(x) for x in fit_episode_ids),
        holdout_episode_ids=tuple(int(x) for x in holdout_episode_ids),
        public_data_hash=str(dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset)),
        random_bank_hash=random_hash,
        optimizer=fit_cfg.optimizer,
        iterations=fit_cfg.iterations,
        transition_data_hash=fit_transition_hash(dataset),
        regularization_variant=fit_cfg.regularization_variant,
        action_parameter_count=action_parameter_count(context.action_channels),
        fixed_regime_persistence=float(regime_persistence),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_cache_key(
    dataset: TrajectoryDataset,
    context: MethodContext,
    form: str,
    model_cfg: FaithfulModelConfig,
    fit_cfg: FaithfulFitConfig,
    seed: int,
    candidate_id: str,
    fit_episode_ids: tuple[int, ...],
    holdout_episode_ids: tuple[int, ...],
    regime_persistence: float,
) -> str:
    """Build a reward-independent identity for one fitted dynamics candidate."""

    torch = require_torch()
    model_values = asdict(model_cfg)
    for name in (
        "forms",
        "candidates_per_form",
        "prior",
        "prior_temperature",
        "minimum_candidate_distance",
    ):
        model_values.pop(name)
    fit_values = asdict(fit_cfg)
    fit_values.pop("mc_paths")
    fit_values.pop("regime_mc_paths")
    effective_mc_paths = fit_cfg.regime_mc_paths if form == "regime" else fit_cfg.mc_paths
    payload = {
        "fit_schema": FIT_SCHEMA_VERSION,
        "equation_version": EQUATION_VERSION,
        "regime_law_version": REGIME_LAW_VERSION,
        "transition_data_hash": fit_transition_hash(dataset),
        "observation_noise_sigma": float(context.observation_noise_sigma),
        "observation_scale": float(context.observation_scale),
        "observation_protocol": "lognormal_conditional_mean_v1",
        "num_actions": int(context.num_actions),
        "action_channels": list(context.action_channels),
        "model_config": model_values,
        "fit_config": fit_values,
        "effective_mc_paths": effective_mc_paths,
        "form": form,
        "regime_persistence": float(regime_persistence),
        "candidate_id": candidate_id,
        "candidate_seed": int(seed),
        "fit_episode_multiset": list(fit_episode_ids),
        "holdout_episode_ids": list(holdout_episode_ids),
        "optimizer_runtime": f"torch-{torch.__version__}",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _fit_cache_payload(result: FitResult, key: str, array_hash: str) -> dict[str, Any]:
    fields = {
        name: getattr(result, name) for name in result.__dataclass_fields__ if name != "model"
    }
    for name, value in list(fields.items()):
        if isinstance(value, tuple):
            fields[name] = list(value)
    return {
        "cache_schema": "adapted_fit_cache_v1",
        "cache_key": key,
        "array_hash": array_hash,
        "model": {
            "form": result.model.form,
            "candidate_id": result.model.candidate_id,
            "action_channels": list(result.model.action_channels),
            "parameter_hash": result.model.parameter_hash(),
        },
        "fit": fields,
    }


def _write_fit_cache(root: Path, key: str, result: FitResult) -> None:
    array_path = root / f"{key}.npz"
    metadata_path = root / f"{key}.json"
    temporary_array = root / f".{key}.{os.getpid()}.tmp.npz"
    temporary_metadata = root / f".{key}.{os.getpid()}.tmp.json"
    model = result.model
    np.savez_compressed(
        temporary_array,
        growth=model.growth,
        mortality=model.mortality,
        capacity_increment=model.capacity_increment,
        stocking=model.stocking,
        reset_log_mean=np.asarray([model.reset_log_mean]),
        reset_log_scale=np.asarray([model.reset_log_scale]),
        initial_capacity=np.asarray([model.initial_capacity]),
        capacity_ceiling=np.asarray([model.capacity_ceiling]),
        process_scale=np.asarray([model.process_scale]),
        observation_scale=np.asarray([model.observation_scale]),
        survey_scale=np.asarray([model.survey_scale]),
        depensation_thresholds=model.depensation_thresholds,
        theta_exponent=np.asarray([model.theta_exponent]),
        regime_multipliers=model.regime_multipliers,
        regime_matrix=model.regime_matrix,
    )
    array_hash = _sha256_file(temporary_array)
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(_fit_cache_payload(result, key, array_hash), handle, indent=2, sort_keys=True)
    os.replace(temporary_array, array_path)
    os.replace(temporary_metadata, metadata_path)


def _load_fit_cache(root: Path, key: str, public_data_hash: str) -> FitResult:
    array_path, metadata_path = root / f"{key}.npz", root / f"{key}.json"
    if not array_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"partial adapted fit cache entry for key {key}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("cache_schema") != "adapted_fit_cache_v1" or payload.get("cache_key") != key:
        raise RuntimeError(f"adapted fit cache identity mismatch for key {key}")
    if _sha256_file(array_path) != payload.get("array_hash"):
        raise RuntimeError(f"adapted fit cache array hash mismatch for key {key}")
    with np.load(array_path, allow_pickle=False) as data:

        def scalar(name: str) -> float:
            return float(np.asarray(data[name]).reshape(-1)[0])

        model_metadata = payload["model"]
        model = MechanisticModel(
            form=str(model_metadata["form"]),
            growth=np.asarray(data["growth"]),
            mortality=np.asarray(data["mortality"]),
            capacity_increment=np.asarray(data["capacity_increment"]),
            stocking=np.asarray(data["stocking"]),
            reset_log_mean=scalar("reset_log_mean"),
            reset_log_scale=scalar("reset_log_scale"),
            initial_capacity=scalar("initial_capacity"),
            capacity_ceiling=scalar("capacity_ceiling"),
            process_scale=scalar("process_scale"),
            observation_scale=scalar("observation_scale"),
            survey_scale=scalar("survey_scale"),
            depensation_thresholds=np.asarray(data["depensation_thresholds"]),
            theta_exponent=scalar("theta_exponent"),
            regime_multipliers=np.asarray(data["regime_multipliers"]),
            regime_matrix=np.asarray(data["regime_matrix"]),
            action_channels=tuple(model_metadata["action_channels"]),
            candidate_id=str(model_metadata["candidate_id"]),
        )
    if model.parameter_hash() != model_metadata.get("parameter_hash"):
        raise RuntimeError(f"adapted fit cache parameter hash mismatch for key {key}")
    fields = payload["fit"]
    tuple_fields = {
        "start_gradient_norms",
        "start_objective_traces",
        "start_objectives",
        "action_rows",
        "action_episodes",
        "sparse_actions",
        "fit_episode_ids",
        "holdout_episode_ids",
    }
    for name in tuple_fields:
        fields[name] = tuple(
            tuple(value) if isinstance(value, list) else value for value in fields[name]
        )
    fields["public_data_hash"] = public_data_hash
    return FitResult(model=model, **fields)


def load_or_fit_mechanistic_model(
    dataset: TrajectoryDataset,
    context: MethodContext,
    form: str,
    model_cfg: FaithfulModelConfig,
    fit_cfg: FaithfulFitConfig,
    seed: int,
    candidate_id: str = "model_000",
    fit_episode_ids: tuple[int, ...] | None = None,
    holdout_episode_ids: tuple[int, ...] | None = None,
    regime_persistence: float = 0.90,
    cache_dir: str | Path = "",
) -> tuple[FitResult, str]:
    if fit_episode_ids is None:
        fit_episode_ids, inferred_holdout = split_history_episodes(
            dataset, fit_cfg.history_fraction, seed
        )
        if holdout_episode_ids is None:
            holdout_episode_ids = inferred_holdout
    holdout_episode_ids = holdout_episode_ids or ()
    if not cache_dir:
        return (
            fit_mechanistic_model(
                dataset,
                context,
                form,
                model_cfg,
                fit_cfg,
                seed,
                candidate_id,
                fit_episode_ids,
                holdout_episode_ids,
                regime_persistence,
            ),
            "disabled",
        )
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = fit_cache_key(
        dataset,
        context,
        form,
        model_cfg,
        fit_cfg,
        seed,
        candidate_id,
        fit_episode_ids,
        holdout_episode_ids,
        regime_persistence,
    )
    lock_path = root / f"{key}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        array_path, metadata_path = root / f"{key}.npz", root / f"{key}.json"
        full_hash = str(dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset))
        if array_path.exists() or metadata_path.exists():
            return replace(_load_fit_cache(root, key, full_hash), fit_cache_key=key), "hit"
        result = fit_mechanistic_model(
            dataset,
            context,
            form,
            model_cfg,
            fit_cfg,
            seed,
            candidate_id,
            fit_episode_ids,
            holdout_episode_ids,
            regime_persistence,
        )
        result = replace(result, fit_cache_key=key)
        _write_fit_cache(root, key, result)
        return result, "miss_fitted"


def _regime_schedule(count: int) -> tuple[tuple[float, bool], ...]:
    schedules = {
        1: ((0.90, False),),
        2: ((0.80, False), (0.90, False)),
        3: ((0.80, False), (0.90, False), (0.97, False)),
        4: ((0.80, False), (0.90, False), (0.97, False), (0.90, True)),
        8: (
            (0.80, False),
            (0.90, False),
            (0.97, False),
            (0.90, True),
            (0.80, True),
            (0.90, True),
            (0.97, True),
            (0.90, True),
        ),
    }
    return schedules[count]


def build_candidate_bank(
    dataset: TrajectoryDataset,
    context: MethodContext,
    model_cfg: FaithfulModelConfig,
    fit_cfg: FaithfulFitConfig,
    seed: int,
    cache_dir: str | Path = "",
) -> tuple[CandidateBank, tuple[str, ...]]:
    """Fit a registered fixed mechanistic episode-bootstrap MAP bank."""

    history, holdout = split_history_episodes(dataset, fit_cfg.history_fraction, seed)
    fits: list[FitResult] = []
    cache_statuses: list[str] = []
    for form_index, form in enumerate(model_cfg.forms):
        schedule = (
            _regime_schedule(model_cfg.candidates_per_form)
            if form == "regime"
            else tuple(
                (0.90, candidate_index > 0)
                for candidate_index in range(model_cfg.candidates_per_form)
            )
        )
        for candidate_index, (persistence, bootstrap) in enumerate(schedule):
            sampled = history
            if bootstrap:
                rng = np.random.default_rng(
                    seed + form_index * 100_000 + candidate_index * 10_000 + 404
                )
                sampled = tuple(
                    int(value) for value in rng.choice(history, size=len(history), replace=True)
                )
            fit, cache_status = load_or_fit_mechanistic_model(
                dataset,
                context,
                form,
                model_cfg,
                fit_cfg,
                seed + form_index * 100_000 + candidate_index * 10_000,
                candidate_id=f"candidate_{form_index:02d}_{candidate_index:02d}",
                fit_episode_ids=sampled,
                holdout_episode_ids=holdout,
                regime_persistence=persistence,
                cache_dir=cache_dir,
            )
            cache_statuses.append(cache_status)
            for existing in fits:
                if existing.model.form != form:
                    continue
                left, right = _model_vector(existing.model), _model_vector(fit.model)
                scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 0.05)
                distance = float(np.linalg.norm((left - right) / scale) / np.sqrt(len(left)))
                if distance < model_cfg.minimum_candidate_distance:
                    raise RuntimeError(
                        f"candidate form {form!r} failed the registered diversity threshold"
                    )
            fits.append(fit)
    count = len(fits)
    if model_cfg.prior == "uniform":
        weights = np.full(count, 1.0 / count, dtype=np.float64)
    else:
        losses = np.asarray([fit.objective for fit in fits])
        logits = -(losses - np.min(losses)) / model_cfg.prior_temperature
        weights = np.exp(logits - np.max(logits))
        weights /= weights.sum()
    construction = (
        RICKER_ONLY_CANDIDATE_CONSTRUCTION
        if model_cfg.forms == ("ricker",)
        else CANDIDATE_CONSTRUCTION
    )
    return (
        CandidateBank(tuple(fits), weights, model_cfg.prior, construction),
        tuple(cache_statuses),
    )


def _model_vector(model: MechanisticModel) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(
                [
                    model.reset_log_mean,
                    model.reset_log_scale,
                    model.initial_capacity,
                    model.capacity_ceiling,
                    model.process_scale,
                    model.theta_exponent,
                ]
            ),
            model.growth,
            model.mortality,
            model.capacity_increment,
            model.stocking,
            model.depensation_thresholds,
            model.regime_multipliers,
            model.regime_matrix.reshape(-1),
        )
    )


def fit_signature(result: FitResult) -> str:
    payload = json.dumps(result.diagnostics(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def ordered_trajectory_sse(
    model: MechanisticModel,
    dataset: TrajectoryDataset,
    episode_ids: Iterable[int] | None = None,
) -> float:
    """Ordered-history diagnostic using propagated latent and conditional survey means."""

    episodes = tuple(_episode_ids(dataset)) if episode_ids is None else tuple(episode_ids)
    errors = []
    survey_mean_factor = conditional_survey_mean_factor(model.observation_scale)
    for indices in _ordered_indices(dataset, episodes):
        latent = float(np.exp(model.reset_log_mean + 0.5 * model.reset_log_scale**2))
        capacity, regime = model.initial_capacity, 0
        for row in indices:
            action = int(dataset.actions[row])
            latent = float(model.noiseless_next(latent, capacity, action, regime))
            capacity = model.next_capacity(capacity, action)
            prediction = latent * survey_mean_factor * model.survey_scale
            errors.append(
                ((prediction - float(dataset.next_observations[row])) / model.survey_scale) ** 2
            )
            if model.form == "regime":
                regime = int(np.argmax(model.regime_matrix[regime]))
            if dataset.terminated is not None and bool(dataset.terminated[row]):
                break
    return float(np.mean(errors)) if errors else float("nan")
