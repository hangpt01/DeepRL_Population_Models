"""Versioned, no-pickle artifacts for paper-faithful ecology methods."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .faithful_ecology import EQUATION_VERSION, MechanisticModel
from .faithful_fit import CandidateBank, FitResult, FIT_SCHEMA_VERSION
from .faithful_pomdp import CandidatePOMDP
from .privacy import assert_hidden_method_artifact


ARTIFACT_SCHEMA_VERSION = "faithful_artifacts_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    assert_hidden_method_artifact(payload, root=path.stem)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def _model_arrays(model: MechanisticModel) -> dict[str, np.ndarray]:
    return {
        "growth": model.growth,
        "mortality": model.mortality,
        "capacity_increment": model.capacity_increment,
        "stocking": model.stocking,
        "reset_log_mean": np.asarray([model.reset_log_mean]),
        "reset_log_scale": np.asarray([model.reset_log_scale]),
        "initial_capacity": np.asarray([model.initial_capacity]),
        "capacity_ceiling": np.asarray([model.capacity_ceiling]),
        "process_scale": np.asarray([model.process_scale]),
        "observation_scale": np.asarray([model.observation_scale]),
        "survey_scale": np.asarray([model.survey_scale]),
        "depensation_thresholds": model.depensation_thresholds,
        "theta_exponent": np.asarray([model.theta_exponent]),
        "regime_multipliers": model.regime_multipliers,
        "regime_matrix": model.regime_matrix,
    }


def save_model(path: str | Path, model: MechanisticModel) -> dict[str, str]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.npz")
    np.savez_compressed(temporary, **_model_arrays(model))
    temporary.replace(target)
    metadata_path = target.with_suffix(".json")
    _atomic_json(metadata_path, {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "equation_version": EQUATION_VERSION,
        "form": model.form,
        "candidate_id": model.candidate_id,
        "parameter_hash": model.parameter_hash(),
        "array_hash": sha256_file(target),
    })
    return {
        "array_path": str(target),
        "metadata_path": str(metadata_path),
        "array_hash": sha256_file(target),
        "metadata_hash": sha256_file(metadata_path),
    }


def load_model(path: str | Path) -> MechanisticModel:
    source = Path(path)
    with source.with_suffix(".json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("artifact_schema") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unknown faithful model artifact schema")
    with np.load(source, allow_pickle=False) as data:
        def scalar(name: str) -> float:
            return float(np.asarray(data[name]).reshape(-1)[0])

        model = MechanisticModel(
            form=str(metadata["form"]),
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
            candidate_id=str(metadata["candidate_id"]),
        )
    if model.parameter_hash() != metadata.get("parameter_hash"):
        raise ValueError("faithful model parameter hash mismatch")
    return model


def save_fit_artifacts(
    output: str | Path,
    fits: Iterable[FitResult],
    planner_provenance: Iterable[dict[str, Any]],
    method_impl_version: str,
    initial_weights: np.ndarray | None = None,
    prior_type: str | None = None,
    pomdps: Iterable[CandidatePOMDP] | None = None,
    planner_action_values: Iterable[np.ndarray] | None = None,
) -> dict[str, Any]:
    root = Path(output) / "faithful_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    fit_list = list(fits)
    records = []
    for index, fit in enumerate(fit_list):
        artifact = save_model(root / f"candidate_{index:03d}.npz", fit.model)
        record = {**fit.diagnostics(), **artifact}
        assert_hidden_method_artifact(record, root=f"candidate_{index:03d}")
        records.append(record)
    provenance = list(planner_provenance)
    fit_payload = {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "fit_schema": FIT_SCHEMA_VERSION,
        "method_impl_version": method_impl_version,
        "candidate_count": len(fit_list),
        "fits": records,
    }
    _atomic_json(root / "faithful_fit.json", fit_payload)
    np.savez_compressed(
        root / "faithful_fit.npz",
        objectives=np.asarray([fit.objective for fit in fit_list]),
        holdout_normalized_survey_sse=np.asarray([
            fit.holdout_normalized_survey_sse for fit in fit_list
        ]),
        selected_starts=np.asarray([fit.selected_start for fit in fit_list], dtype=np.int64),
        action_rows=np.asarray([fit.action_rows for fit in fit_list], dtype=np.int64),
        action_episodes=np.asarray([fit.action_episodes for fit in fit_list], dtype=np.int64),
    )
    bank_payload: dict[str, Any] | None = None
    if initial_weights is not None:
        weights = np.asarray(initial_weights, dtype=np.float64)
        np.savez_compressed(root / "candidate_bank.npz", initial_weights=weights)
        bank_payload = {
            "artifact_schema": ARTIFACT_SCHEMA_VERSION,
            "prior_type": str(prior_type),
            "candidate_count": len(weights),
            "candidate_ids": [fit.model.candidate_id for fit in fit_list],
            "parameter_hashes": [fit.model.parameter_hash() for fit in fit_list],
            "array_hash": sha256_file(root / "candidate_bank.npz"),
        }
        _atomic_json(root / "candidate_bank.json", bank_payload)
    planner_payload = {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "planners": provenance,
    }
    _atomic_json(root / "planner_provenance.json", planner_payload)
    pomdp_list = list(pomdps or ())
    if pomdp_list and len(pomdp_list) != len(fit_list):
        raise ValueError("POMDP artifact count must match fitted model count")
    pomdp_hashes = []
    for index, pomdp in enumerate(pomdp_list):
        capacities = np.linspace(
            pomdp.model.initial_capacity,
            pomdp.model.capacity_ceiling,
            pomdp.config.capacity_bins,
        )
        transitions = np.asarray([
            [pomdp.transition_matrix(float(capacity), action)
             for action in range(pomdp.context.num_actions)]
            for capacity in capacities
        ])
        observation_grid = pomdp.abundance_grid * pomdp.model.survey_scale
        emissions = np.column_stack([
            pomdp.observation_vector(float(observation))
            for observation in observation_grid
        ])
        emissions /= np.maximum(emissions.sum(axis=1, keepdims=True), 1e-300)
        pomdp_path = root / f"pomdp_model_{index:03d}.npz"
        np.savez_compressed(
            pomdp_path,
            abundance_grid=pomdp.abundance_grid,
            capacity_grid=capacities,
            observation_grid=observation_grid,
            transitions=transitions,
            emissions=emissions,
        )
        pomdp_metadata = {
            "artifact_schema": ARTIFACT_SCHEMA_VERSION,
            "candidate_id": pomdp.model.candidate_id,
            "model_hash": pomdp.model_hash(),
            "array_hash": sha256_file(pomdp_path),
            "state_bins": pomdp.config.state_bins,
            "capacity_bins": pomdp.config.capacity_bins,
            "observation_bins": pomdp.config.observation_bins,
            "transition_samples": pomdp.config.transition_samples,
            "observation_convention": "public_lognormal_exact_emission",
            "reward_context": "previous_current_following_timestep_action_cost_pop_token",
        }
        _atomic_json(pomdp_path.with_suffix(".json"), pomdp_metadata)
        pomdp_hashes.append(sha256_file(pomdp_path))
    action_values = list(planner_action_values or ())
    if action_values:
        if len(action_values) != len(fit_list):
            raise ValueError("planner diagnostic count must match fitted model count")
        np.savez_compressed(
            root / "pbvi_policy_diagnostics.npz",
            last_action_values=np.asarray(action_values, dtype=np.float64),
        )
    privacy_payload = {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "forbidden_name_hits": [],
        "checked_json_files": sorted(path.name for path in root.glob("*.json")),
        "status": "passed",
    }
    _atomic_json(root / "privacy_audit.json", privacy_payload)
    result = {
        "faithful_artifact_dir": str(root),
        "method_impl_version": method_impl_version,
        "equation_version": EQUATION_VERSION,
        "candidate_count": len(fit_list),
        "planner": "pbvi",
        "faithful_fit_hash": sha256_file(root / "faithful_fit.json"),
        "planner_provenance_hash": sha256_file(root / "planner_provenance.json"),
        "privacy_audit_hash": sha256_file(root / "privacy_audit.json"),
        "pomdp_model_hashes": pomdp_hashes,
    }
    if bank_payload is not None:
        result["candidate_bank_hash"] = sha256_file(root / "candidate_bank.json")
    return result


def validate_candidate_bank(bank: CandidateBank) -> None:
    if bank.prior_type not in {"uniform", "public_history_likelihood"}:
        raise ValueError("unknown candidate-bank prior")
    CandidateBank(bank.fits, bank.initial_weights.copy(), bank.prior_type)
