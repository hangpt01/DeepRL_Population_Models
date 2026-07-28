"""Ordered-episode automatic-differentiation fitting for faithful ecology models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from .config import FaithfulFitConfig, FaithfulModelConfig, MethodContext
from .dataset import TrajectoryDataset, dataset_sha256
from .faithful_ecology import EQUATION_VERSION, MechanisticModel, REGISTERED_FORMS


FIT_SCHEMA_VERSION = "faithful_fit_v1"


@dataclass(frozen=True)
class FitResult:
    model: MechanisticModel
    objective: float
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

    def diagnostics(self) -> dict[str, Any]:
        return {
            "fit_schema": FIT_SCHEMA_VERSION,
            "equation_version": EQUATION_VERSION,
            "form": self.model.form,
            "objective": float(self.objective),
            "selected_start": int(self.selected_start),
            "start_objectives": list(self.start_objectives),
            "action_rows": list(self.action_rows),
            "action_episodes": list(self.action_episodes),
            "sparse_actions": list(self.sparse_actions),
            "fit_episode_ids": list(self.fit_episode_ids),
            "holdout_episode_ids": list(self.holdout_episode_ids),
            "public_data_hash": self.public_data_hash,
            "random_bank_hash": self.random_bank_hash,
            "optimizer": self.optimizer,
            "iterations": int(self.iterations),
            "parameter_hash": self.model.parameter_hash(),
        }


@dataclass(frozen=True)
class CandidateBank:
    fits: tuple[FitResult, ...]
    initial_weights: np.ndarray
    prior_type: str

    def __post_init__(self) -> None:
        weights = np.asarray(self.initial_weights, dtype=np.float64)
        if len(weights) != len(self.fits) or len(weights) == 0:
            raise ValueError("candidate prior must cover a non-empty bank")
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("candidate prior must be a probability vector")
        hashes = [fit.model.parameter_hash() for fit in self.fits]
        if len(set(hashes)) != len(hashes):
            raise ValueError("candidate bank contains duplicate parameterizations")
        object.__setattr__(self, "initial_weights", weights)


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "paper-faithful fitting requires the optional 'paper-faithful-fit' "
            "dependency (PyTorch CPU); no regression or finite-difference fallback is allowed"
        ) from exc
    return torch


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


def _coverage(dataset: TrajectoryDataset, num_actions: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rows = []
    episodes = []
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
    survey = rng.normal(size=(paths, episodes, width + 1))
    digest = hashlib.sha256()
    for value in (initial, process, survey):
        digest.update(np.ascontiguousarray(value).tobytes())
    return initial, process, survey, digest.hexdigest()


def _inverse_sigmoid(value: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(value / (1.0 - value))


def _initial_raw(num_actions: int, form: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = [
        np.asarray([np.log(0.8), -1.6, -0.2, -0.2, -2.5]),
        _inverse_sigmoid(np.full(num_actions, 0.12)),
        _inverse_sigmoid(np.full(num_actions, 0.06)),
        _inverse_sigmoid(np.full(num_actions, 0.02)),
        _inverse_sigmoid(np.full(num_actions, 0.01)),
        _inverse_sigmoid(np.asarray([0.2, 0.35])),
        _inverse_sigmoid(np.asarray([0.375])),
        _inverse_sigmoid(np.asarray([0.25, 0.55])),
        _inverse_sigmoid(np.asarray([0.82, 0.82])),
    ]
    raw = np.concatenate(values).astype(np.float64)
    raw += rng.normal(0.0, 0.25, size=len(raw))
    if form == "ricker":
        raw[-7:] = np.concatenate(values)[-7:]
    return raw


def _decode(raw, num_actions: int, model_cfg: FaithfulModelConfig, torch):
    softplus = torch.nn.functional.softplus
    sigmoid = torch.sigmoid
    pos = 0
    reset_mean = raw[pos]
    pos += 1
    reset_scale = 0.01 + softplus(raw[pos])
    pos += 1
    initial_capacity = 0.05 + softplus(raw[pos])
    pos += 1
    capacity_ceiling = initial_capacity + softplus(raw[pos])
    pos += 1
    process_scale = model_cfg.process_scale_upper * sigmoid(raw[pos])
    pos += 1
    arrays = []
    uppers = (
        model_cfg.action_growth_upper,
        model_cfg.action_mortality_upper,
        model_cfg.action_capacity_upper,
        model_cfg.action_stocking_upper,
    )
    for upper in uppers:
        arrays.append(upper * sigmoid(raw[pos:pos + num_actions]))
        pos += num_actions
    threshold_fraction = 0.05 + 0.75 * sigmoid(raw[pos:pos + 2])
    pos += 2
    thresholds = threshold_fraction * initial_capacity
    theta_exponent = 0.2 + 4.8 * sigmoid(raw[pos])
    pos += 1
    multipliers = 0.25 + 1.75 * sigmoid(raw[pos:pos + 2])
    pos += 2
    persistence = 0.5 + 0.49 * sigmoid(raw[pos:pos + 2])
    pos += 2
    matrix = torch.stack((
        torch.stack((persistence[0], 1.0 - persistence[0])),
        torch.stack((1.0 - persistence[1], persistence[1])),
    ))
    if pos != len(raw):
        raise AssertionError("faithful parameter decoder consumed the wrong dimension")
    return {
        "reset_mean": reset_mean,
        "reset_scale": reset_scale,
        "initial_capacity": initial_capacity,
        "capacity_ceiling": capacity_ceiling,
        "process_scale": process_scale,
        "growth": arrays[0],
        "mortality": arrays[1],
        "capacity_increment": arrays[2],
        "stocking": arrays[3],
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
    torch,
):
    p = _decode(raw, context.num_actions, model_cfg, torch)
    initial_noise, process_noise, survey_noise = [
        torch.as_tensor(value, dtype=torch.float64) for value in random_bank
    ]
    total = torch.zeros((), dtype=torch.float64)
    count = 0
    scale = float(context.observation_scale)
    for episode_position, indices in enumerate(ordered):
        capacity = p["initial_capacity"]
        latent = torch.exp(
            p["reset_mean"] + p["reset_scale"] * initial_noise[:, episode_position]
        )
        regime_prob = torch.full(
            (fit_cfg.mc_paths,), 0.5, dtype=torch.float64
        )
        first = int(indices[0])
        predicted = latent * torch.exp(
            float(context.observation_noise_sigma) * survey_noise[:, episode_position, 0]
        )
        target = float(dataset.observations[first]) / scale
        total = total + torch.mean((predicted - target) ** 2)
        count += 1
        for step, row in enumerate(indices):
            action = int(dataset.actions[row])
            capacity = torch.minimum(
                p["capacity_ceiling"], capacity + p["capacity_increment"][action]
            )
            managed = torch.clamp(latent + p["stocking"][action], min=0.0)
            growth = p["growth"][action]
            mortality = p["mortality"][action]
            if form == "ricker":
                exponent = growth * (1.0 - managed / capacity) - mortality
                mean = managed * torch.exp(torch.clamp(exponent, -40.0, 40.0))
            elif form == "allee":
                exponent = (
                    growth * (1.0 - managed / capacity)
                    * (managed / p["thresholds"][0] - 1.0) - mortality
                )
                mean = managed * torch.exp(torch.clamp(exponent, -40.0, 40.0))
            elif form == "theta":
                core = torch.clamp(
                    managed + growth * managed
                    * (1.0 - (managed / capacity) ** p["theta_exponent"]),
                    min=0.0,
                )
                mean = core * torch.exp(-mortality)
            else:
                threshold = (
                    (1.0 - regime_prob) * p["thresholds"][0]
                    + regime_prob * p["thresholds"][1]
                )
                multiplier = (
                    (1.0 - regime_prob) * p["multipliers"][0]
                    + regime_prob * p["multipliers"][1]
                )
                exponent = (
                    multiplier * growth * (1.0 - managed / capacity)
                    * (managed / threshold - 1.0) - mortality
                )
                mean = managed * torch.exp(torch.clamp(exponent, -40.0, 40.0))
                regime_prob = (
                    (1.0 - regime_prob) * p["matrix"][0, 1]
                    + regime_prob * p["matrix"][1, 1]
                )
            latent = mean * torch.exp(
                p["process_scale"] * process_noise[:, episode_position, step]
            )
            predicted = latent * torch.exp(
                float(context.observation_noise_sigma)
                * survey_noise[:, episode_position, step + 1]
            )
            target = float(dataset.next_observations[row]) / scale
            total = total + torch.mean((predicted - target) ** 2)
            count += 1
            if dataset.terminated is not None and bool(dataset.terminated[row]):
                break
    action_matrix = torch.stack((
        p["growth"], p["mortality"], p["capacity_increment"], p["stocking"]
    ), dim=1)
    pooled = torch.mean(action_matrix, dim=0, keepdim=True)
    shrinkage = fit_cfg.shrinkage * torch.mean((action_matrix - pooled) ** 2)
    group = fit_cfg.group_penalty * torch.mean(
        torch.sqrt(torch.sum((action_matrix - pooled) ** 2, dim=1) + 1e-12)
    )
    complementarity = fit_cfg.complementarity_penalty * torch.mean(
        p["growth"] * p["mortality"]
    )
    return total / max(count, 1) + shrinkage + group + complementarity


def _to_model(
    raw: np.ndarray,
    context: MethodContext,
    form: str,
    model_cfg: FaithfulModelConfig,
    candidate_id: str,
) -> MechanisticModel:
    torch = require_torch()
    tensor = torch.as_tensor(raw, dtype=torch.float64)
    values = _decode(tensor, context.num_actions, model_cfg, torch)

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
) -> FitResult:
    """Fit one fixed candidate from ordered public action-observation episodes."""

    if form not in REGISTERED_FORMS:
        raise ValueError(f"unknown mechanistic form {form!r}")
    context.validate()
    dataset.validate()
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
        raise ValueError("faithful fit received no complete training episodes")
    lengths = [len(indices) for indices in ordered]
    initial, process, survey, random_hash = _random_bank(
        lengths, fit_cfg.mc_paths, seed + 17
    )
    random_values = (initial, process, survey)
    objectives: list[float] = []
    raw_values: list[np.ndarray | None] = []
    for start in range(fit_cfg.starts):
        initial_raw = _initial_raw(context.num_actions, form, seed + 1009 * start)
        raw = torch.nn.Parameter(torch.as_tensor(initial_raw, dtype=torch.float64))
        optimizer = torch.optim.LBFGS(
            [raw],
            lr=fit_cfg.learning_rate,
            max_iter=fit_cfg.iterations,
            tolerance_grad=fit_cfg.tolerance_grad,
            tolerance_change=fit_cfg.tolerance_change,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            objective = _trajectory_objective(
                raw, dataset, ordered, context, form, model_cfg, fit_cfg,
                random_values, torch,
            )
            objective.backward()
            return objective

        try:
            optimizer.step(closure)
            objective = float(closure().detach().cpu())
            final_raw = raw.detach().cpu().numpy().copy()
            if not np.isfinite(objective) or not np.all(np.isfinite(final_raw)):
                raise FloatingPointError("non-finite faithful fit")
            objectives.append(objective)
            raw_values.append(final_raw)
        except (RuntimeError, FloatingPointError):
            objectives.append(float("inf"))
            raw_values.append(None)
    finite = np.flatnonzero(np.isfinite(objectives))
    if len(finite) == 0:
        raise RuntimeError("all registered automatic-differentiation starts failed")
    selected = int(min(finite, key=lambda index: objectives[int(index)]))
    model = _to_model(
        raw_values[selected], context, form, model_cfg, candidate_id
    )
    action_rows, action_episodes = _coverage(dataset, context.num_actions)
    threshold = max(fit_cfg.sparse_action_rows, 5 * 4)
    sparse = tuple(index for index, rows in enumerate(action_rows) if rows < threshold)
    public_hash = dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset)
    return FitResult(
        model=model,
        objective=float(objectives[selected]),
        selected_start=selected,
        start_objectives=tuple(float(x) for x in objectives),
        action_rows=action_rows,
        action_episodes=action_episodes,
        sparse_actions=sparse,
        fit_episode_ids=tuple(int(x) for x in fit_episode_ids),
        holdout_episode_ids=tuple(int(x) for x in holdout_episode_ids),
        public_data_hash=str(public_hash),
        random_bank_hash=random_hash,
        optimizer=fit_cfg.optimizer,
        iterations=fit_cfg.iterations,
    )


def build_candidate_bank(
    dataset: TrajectoryDataset,
    context: MethodContext,
    model_cfg: FaithfulModelConfig,
    fit_cfg: FaithfulFitConfig,
    seed: int,
) -> CandidateBank:
    """Fit and freeze the registered cross-form public-history candidate bank."""

    history, holdout = split_history_episodes(dataset, fit_cfg.history_fraction, seed)
    rng = np.random.default_rng(seed + 404)
    fits: list[FitResult] = []
    for form_index, form in enumerate(model_cfg.forms):
        for candidate_index in range(model_cfg.candidates_per_form):
            sampled = history
            if candidate_index:
                sampled = tuple(
                    int(value) for value in rng.choice(history, size=len(history), replace=True)
                )
            fit = fit_mechanistic_model(
                dataset,
                context,
                form,
                model_cfg,
                fit_cfg,
                seed + form_index * 100_000 + candidate_index * 10_000,
                candidate_id=f"candidate_{form_index:02d}_{candidate_index:02d}",
                fit_episode_ids=sampled,
                holdout_episode_ids=holdout,
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
    return CandidateBank(tuple(fits), weights, model_cfg.prior)


def fit_signature(result: FitResult) -> str:
    payload = json.dumps(result.diagnostics(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def ordered_trajectory_sse(
    model: MechanisticModel,
    dataset: TrajectoryDataset,
    episode_ids: Iterable[int] | None = None,
) -> float:
    """Deterministic ordered-history diagnostic using propagated latent means."""

    episodes = tuple(_episode_ids(dataset)) if episode_ids is None else tuple(episode_ids)
    errors = []
    for indices in _ordered_indices(dataset, episodes):
        latent = float(np.exp(model.reset_log_mean + 0.5 * model.reset_log_scale ** 2))
        capacity = model.initial_capacity
        regime = 0
        for row in indices:
            action = int(dataset.actions[row])
            latent = float(model.noiseless_next(latent, capacity, action, regime))
            capacity = model.next_capacity(capacity, action)
            prediction = latent * model.survey_scale
            errors.append((prediction - float(dataset.next_observations[row])) ** 2)
            if model.form == "regime":
                regime = int(np.argmax(model.regime_matrix[regime]))
            if dataset.terminated is not None and bool(dataset.terminated[row]):
                break
    return float(np.mean(errors)) if errors else float("nan")
