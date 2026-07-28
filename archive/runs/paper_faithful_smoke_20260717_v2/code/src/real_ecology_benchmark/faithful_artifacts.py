"""Versioned, no-pickle artifacts for paper-faithful ecology methods."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .faithful_ecology import EQUATION_VERSION, MechanisticModel
from .faithful_fit import CandidateBank, FitResult, FIT_SCHEMA_VERSION
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
    privacy_payload = {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "forbidden_name_hits": [],
        "checked_json_files": sorted(path.name for path in root.glob("*.json")),
        "status": "passed",
    }
    _atomic_json(root / "privacy_audit.json", privacy_payload)
    result = {
        "faithful_artifact_dir": str(root),
        "faithful_fit_hash": sha256_file(root / "faithful_fit.json"),
        "planner_provenance_hash": sha256_file(root / "planner_provenance.json"),
        "privacy_audit_hash": sha256_file(root / "privacy_audit.json"),
    }
    if bank_payload is not None:
        result["candidate_bank_hash"] = sha256_file(root / "candidate_bank.json")
    return result


def validate_candidate_bank(bank: CandidateBank) -> None:
    if bank.prior_type not in {"uniform", "public_history_likelihood"}:
        raise ValueError("unknown candidate-bank prior")
    CandidateBank(bank.fits, bank.initial_weights.copy(), bank.prior_type)
