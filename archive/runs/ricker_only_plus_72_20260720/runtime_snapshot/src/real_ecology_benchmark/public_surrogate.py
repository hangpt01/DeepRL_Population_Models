"""Shared reward and genuine-termination models fitted from sanitized data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import PrivateTrajectoryData, TrajectoryDataset, dataset_sha256


SURROGATE_VERSION = 2


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(target) == 0:
        return {"rmse": float("nan"), "mae": float("nan")}
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
    }


def _episode_split(dataset: TrajectoryDataset, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit = np.zeros(len(dataset), dtype=bool)
    holdout = np.zeros(len(dataset), dtype=bool)
    pop_ids = (
        dataset.pop_ids
        if dataset.pop_ids is not None
        else np.full(len(dataset), "pop_default", dtype="U16")
    )
    for pop_id in np.unique(pop_ids):
        episodes = np.unique(dataset.episode_id[pop_ids == pop_id]).astype(int)
        rng.shuffle(episodes)
        if len(episodes) < 2:
            fit |= pop_ids == pop_id
            continue
        cut = max(1, min(len(episodes) - 1, int(round(0.8 * len(episodes)))))
        fit |= (pop_ids == pop_id) & np.isin(dataset.episode_id, episodes[:cut])
        holdout |= (pop_ids == pop_id) & np.isin(dataset.episode_id, episodes[cut:])
    if not np.any(holdout):
        holdout = ~fit
    return fit, holdout


@dataclass(frozen=True)
class PublicFeatureSpec:
    observation_scale: float
    continuous_mean: np.ndarray
    continuous_std: np.ndarray
    pop_vocabulary: tuple[str, ...]
    num_actions: int
    public_horizon: int

    @classmethod
    def fit(
        cls,
        dataset: TrajectoryDataset,
        fit_mask: np.ndarray,
    ) -> "PublicFeatureSpec":
        positive = dataset.observations[fit_mask & (dataset.observations > 0.0)]
        scale = float(np.median(positive)) if len(positive) else 1.0
        pop_ids = (
            dataset.pop_ids
            if dataset.pop_ids is not None
            else np.full(len(dataset), "pop_default", dtype="U16")
        )
        vocabulary = tuple(sorted(str(value) for value in np.unique(pop_ids)))
        num_actions = (
            len(dataset.action_costs)
            if dataset.action_costs is not None
            else int(np.max(dataset.actions)) + 1
        )
        horizon = max(int(dataset.metadata.get("horizon", 0)), int(np.max(dataset.timestep)) + 1)
        temporary = cls(
            scale,
            np.zeros(7, dtype=np.float64),
            np.ones(7, dtype=np.float64),
            vocabulary,
            num_actions,
            horizon,
        )
        continuous = temporary.continuous(dataset)
        mean = continuous[fit_mask].mean(axis=0)
        std = np.maximum(continuous[fit_mask].std(axis=0), 1e-8)
        return cls(scale, mean, std, vocabulary, num_actions, horizon)

    def _z(self, values: np.ndarray) -> np.ndarray:
        return np.log1p(np.maximum(np.asarray(values, dtype=np.float64), 0.0) / self.observation_scale)

    def continuous(self, dataset: TrajectoryDataset) -> np.ndarray:
        previous = dataset.observations.copy()
        for indices in dataset.episode_indices():
            if len(indices) > 1:
                previous[indices[1:]] = dataset.observations[indices[:-1]]
        current = self._z(dataset.observations)
        prior = self._z(previous)
        following = self._z(dataset.next_observations)
        cost = (
            dataset.costs
            if dataset.costs is not None
            else np.zeros(len(dataset), dtype=np.float64)
        )
        tau = dataset.timestep.astype(np.float64) / max(self.public_horizon - 1, 1)
        return np.column_stack(
            [prior, current, current - prior, following, following - current, tau, cost]
        )

    def design(self, dataset: TrajectoryDataset) -> np.ndarray:
        continuous = (self.continuous(dataset) - self.continuous_mean) / self.continuous_std
        actions = np.eye(self.num_actions, dtype=np.float64)[dataset.actions.astype(int)]
        pop_ids = (
            dataset.pop_ids
            if dataset.pop_ids is not None
            else np.full(len(dataset), "pop_default", dtype="U16")
        )
        lookup = {value: index for index, value in enumerate(self.pop_vocabulary)}
        try:
            pop_indices = np.asarray([lookup[str(value)] for value in pop_ids], dtype=int)
        except KeyError as exc:
            raise ValueError(f"unknown opaque population token: {exc.args[0]}") from exc
        populations = np.eye(len(self.pop_vocabulary), dtype=np.float64)[pop_indices]
        return np.column_stack(
            [
                np.ones(len(dataset)),
                continuous[:, :6],
                actions,
                continuous[:, 6],
                populations,
            ]
        )

    def transition_design(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        following: np.ndarray,
        actions: np.ndarray,
        timesteps: np.ndarray,
        pop_ids: np.ndarray,
        action_costs: np.ndarray,
    ) -> np.ndarray:
        actions = np.asarray(actions, dtype=int)
        prior_z = self._z(previous)
        current_z = self._z(current)
        following_z = self._z(following)
        tau = np.asarray(timesteps, dtype=np.float64) / max(self.public_horizon - 1, 1)
        cost = np.asarray(action_costs, dtype=np.float64)[actions]
        continuous = np.column_stack(
            [
                prior_z,
                current_z,
                current_z - prior_z,
                following_z,
                following_z - current_z,
                tau,
                cost,
            ]
        )
        continuous = (continuous - self.continuous_mean) / self.continuous_std
        action_columns = np.eye(self.num_actions, dtype=np.float64)[actions]
        lookup = {value: index for index, value in enumerate(self.pop_vocabulary)}
        try:
            pop_indices = np.asarray([lookup[str(value)] for value in pop_ids], dtype=int)
        except KeyError as exc:
            raise ValueError(f"unknown opaque population token: {exc.args[0]}") from exc
        population_columns = np.eye(len(self.pop_vocabulary), dtype=np.float64)[pop_indices]
        return np.column_stack(
            [
                np.ones(len(actions)),
                continuous[:, :6],
                action_columns,
                continuous[:, 6],
                population_columns,
            ]
        )


@dataclass(frozen=True)
class PublicRewardRiskSurrogate:
    feature_spec: PublicFeatureSpec
    reward_coefficients: np.ndarray
    risk_coefficients: np.ndarray | None
    constant_risk: float | None
    action_costs: np.ndarray
    diagnostics: dict[str, Any]
    public_data_hash: str
    split_seed: int

    @property
    def observation_scale(self) -> float:
        return self.feature_spec.observation_scale

    def predict(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        following: np.ndarray,
        actions: np.ndarray,
        timesteps: np.ndarray,
        pop_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        design = self.feature_spec.transition_design(
            previous,
            current,
            following,
            actions,
            timesteps,
            pop_ids,
            self.action_costs,
        )
        reward = design @ self.reward_coefficients
        if self.constant_risk is not None:
            risk = np.full(len(design), self.constant_risk, dtype=np.float64)
        else:
            logits = np.clip(design @ self.risk_coefficients, -50.0, 50.0)
            risk = 1.0 / (1.0 + np.exp(-logits))
        return reward, risk

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        spec = self.feature_spec
        metadata = {
            "version": SURROGATE_VERSION,
            "pop_vocabulary": list(spec.pop_vocabulary),
            "num_actions": spec.num_actions,
            "public_horizon": spec.public_horizon,
            "diagnostics": self.diagnostics,
            "public_data_hash": self.public_data_hash,
            "split_seed": self.split_seed,
        }
        np.savez_compressed(
            target,
            observation_scale=np.asarray(spec.observation_scale),
            continuous_mean=spec.continuous_mean,
            continuous_std=spec.continuous_std,
            reward_coefficients=self.reward_coefficients,
            risk_coefficients=(
                np.asarray([], dtype=np.float64)
                if self.risk_coefficients is None else self.risk_coefficients
            ),
            constant_risk=np.asarray(
                np.nan if self.constant_risk is None else self.constant_risk
            ),
            action_costs=self.action_costs,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PublicRewardRiskSurrogate":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if int(metadata["version"]) != SURROGATE_VERSION:
                raise ValueError("unsupported public surrogate version")
            spec = PublicFeatureSpec(
                float(data["observation_scale"]),
                np.asarray(data["continuous_mean"]),
                np.asarray(data["continuous_std"]),
                tuple(metadata["pop_vocabulary"]),
                int(metadata["num_actions"]),
                int(metadata["public_horizon"]),
            )
            risk = np.asarray(data["risk_coefficients"])
            constant = float(data["constant_risk"])
            return cls(
                spec,
                np.asarray(data["reward_coefficients"]),
                None if risk.size == 0 else risk,
                None if np.isnan(constant) else constant,
                np.asarray(data["action_costs"]),
                dict(metadata["diagnostics"]),
                str(metadata["public_data_hash"]),
                int(metadata["split_seed"]),
            )


def fit_public_surrogate(
    dataset: TrajectoryDataset,
    seed: int,
    ridge: float = 1e-3,
) -> PublicRewardRiskSurrogate:
    dataset.validate()
    if dataset.terminated is None or dataset.action_costs is None:
        raise ValueError("public surrogate requires sanitized event and cost fields")
    fit_mask, holdout_mask = _episode_split(dataset, seed)
    if not np.any(fit_mask):
        raise ValueError("public surrogate fit fold is empty")
    spec = PublicFeatureSpec.fit(dataset, fit_mask)
    design = spec.design(dataset)
    fit_design = design[fit_mask]
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    reward_coefficients = np.linalg.solve(
        fit_design.T @ fit_design + penalty,
        fit_design.T @ dataset.rewards[fit_mask],
    )
    reward_prediction = design @ reward_coefficients
    terminated = dataset.terminated.astype(np.float64)
    fit_target = terminated[fit_mask]
    constant_risk: float | None = None
    risk_coefficients: np.ndarray | None = None
    fallback = "none"
    if len(np.unique(fit_target)) < 2:
        constant_risk = float((np.sum(fit_target) + 1.0) / (len(fit_target) + 2.0))
        fallback = "constant_single_class"
    else:
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        logistic_penalty = ridge * np.eye(design.shape[1])
        logistic_penalty[0, 0] = 0.0
        for _ in range(200):
            logits = np.clip(fit_design @ coefficients, -50.0, 50.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            weight = np.maximum(probability * (1.0 - probability), 1e-8)
            adjusted = logits + (fit_target - probability) / weight
            lhs = fit_design.T @ (fit_design * weight[:, None]) + logistic_penalty
            rhs = fit_design.T @ (weight * adjusted)
            updated = np.linalg.solve(lhs, rhs)
            if np.max(np.abs(updated - coefficients)) < 1e-8:
                coefficients = updated
                break
            coefficients = updated
        risk_coefficients = coefficients
    tail_cutoff = float(np.quantile(dataset.rewards[fit_mask], 0.1))
    diagnostics: dict[str, Any] = {
        "surrogate_version": SURROGATE_VERSION,
        "fit_rows": int(np.sum(fit_mask)),
        "holdout_rows": int(np.sum(holdout_mask)),
        "terminated_fit": int(np.sum(terminated[fit_mask])),
        "terminated_holdout": int(np.sum(terminated[holdout_mask])),
        "nonterminated_fit": int(np.sum(fit_mask) - np.sum(terminated[fit_mask])),
        "nonterminated_holdout": int(
            np.sum(holdout_mask) - np.sum(terminated[holdout_mask])
        ),
        "terminated_prevalence": float(np.mean(terminated)),
        "risk_fallback": fallback,
        "low_reward_tail_cutoff": tail_cutoff,
    }
    if constant_risk is not None:
        risk_prediction = np.full(len(dataset), constant_risk, dtype=np.float64)
    else:
        logits = np.clip(design @ risk_coefficients, -50.0, 50.0)
        risk_prediction = 1.0 / (1.0 + np.exp(-logits))
    for prefix, mask in (("fit", fit_mask), ("holdout", holdout_mask)):
        metrics = _regression_metrics(dataset.rewards[mask], reward_prediction[mask])
        diagnostics.update({f"reward_{prefix}_{key}": value for key, value in metrics.items()})
        if np.any(mask):
            clipped = np.clip(risk_prediction[mask], 1e-6, 1.0 - 1e-6)
            target = terminated[mask]
            diagnostics[f"risk_{prefix}_brier"] = float(np.mean((clipped - target) ** 2))
            diagnostics[f"risk_{prefix}_log_loss"] = float(-np.mean(
                target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)
            ))
            diagnostics[f"risk_{prefix}_predicted_prevalence"] = float(
                np.mean(risk_prediction[mask])
            )
    pop_ids = dataset.pop_ids
    for pop_id in sorted(str(value) for value in np.unique(pop_ids)):
        token_mask = pop_ids == pop_id
        diagnostics[f"terminated_prevalence_{pop_id}"] = float(
            np.mean(terminated[token_mask])
        )
        for prefix, fold_mask in (("fit", fit_mask), ("holdout", holdout_mask)):
            mask = token_mask & fold_mask
            metrics = _regression_metrics(dataset.rewards[mask], reward_prediction[mask])
            diagnostics.update(
                {
                    f"reward_{prefix}_{pop_id}_{key}": value
                    for key, value in metrics.items()
                }
            )
    tail_mask = holdout_mask & (dataset.rewards <= tail_cutoff)
    tail_metrics = _regression_metrics(dataset.rewards[tail_mask], reward_prediction[tail_mask])
    diagnostics.update({f"reward_holdout_low_tail_{key}": value for key, value in tail_metrics.items()})
    return PublicRewardRiskSurrogate(
        spec,
        reward_coefficients,
        risk_coefficients,
        constant_risk,
        np.asarray(dataset.action_costs, dtype=np.float64),
        diagnostics,
        dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset),
        int(seed),
    )


def evaluator_only_surrogate_diagnostics(
    surrogate: PublicRewardRiskSurrogate,
    dataset: TrajectoryDataset,
    private: PrivateTrajectoryData,
) -> dict[str, float]:
    """Report private safety strata without mutating or selecting the surrogate."""

    private.validate(dataset)
    if private.safety_penalty_applied is None:
        return {}
    design = surrogate.feature_spec.design(dataset)
    prediction = design @ surrogate.reward_coefficients
    output: dict[str, float] = {}
    flags = private.safety_penalty_applied.astype(bool)
    for label, mask in (("safe", ~flags), ("unsafe", flags)):
        metrics = _regression_metrics(dataset.rewards[mask], prediction[mask])
        output.update({f"evaluator_reward_{label}_{key}": value for key, value in metrics.items()})
        output[f"evaluator_reward_{label}_rows"] = int(np.sum(mask))
    return output
