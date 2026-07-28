"""Strict validity-only receipts for the Phase 2E general-RL canary.

The acceptance reader is intentionally not a summary/episode reader.  It only
accepts dedicated ``validity_receipt.json`` files, rejects forbidden field names
from raw bytes before JSON value deserialization, and exposes a small structural
schema containing identities, mechanism diagnostics, and resource metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "general_phase2e_validity_v1"
TOP_LEVEL_FIELDS = {
    "schema", "submitted", "identity", "dataset", "fit", "mechanism",
    "resources", "execution", "threads",
}
FORBIDDEN_FIELD_FRAGMENTS = (
    "reward", "return", "survival", "ranking", "rank", "performance",
    "collapse", "persistence", "unsafe", "true_state", "economic",
)
_JSON_KEY = re.compile(rb'"((?:\\.|[^"\\])*)"\s*:')

MECHANISM_FIELDS = {
    "refplan": {
        "members", "policy_prior_epsilon", "policy_prior_min_probability",
    },
    "ogsrl": {
        "hidden_actor_trained", "lambda_safety", "lambda_ood",
        "modeled_safety_cost", "modeled_ood_cost", "safety_channel_action_spread",
        "safety_channel_degenerate", "safety_dual_moved", "low_abundance_scale",
        "low_abundance_quantile", "low_abundance_cost_prevalence",
        "low_abundance_cost_mean", "cost_horizon", "behavior_normalized_cost",
        "behavior_cost_episode_count", "behavior_cost_standard_deviation",
        "behavior_cost_standard_error", "safety_budget", "guardian_threshold",
    },
    "bamcts": {"members", "guardian_threshold"},
    "ensemble_value_disagreement_pessimism": {
        "q_ensemble_members", "q_ensemble_bootstrap_unique_fraction_mean",
        "q_ensemble_member_bellman_mse_mean", "q_ensemble_disagreement_observed",
        "q_ensemble_disagreement_unobserved", "q_ensemble_mean_bellman_mse",
        "behavior_reference_nll_train", "behavior_reference_nll_holdout",
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float)) and (
        not isinstance(value, float) or math.isfinite(value)
    )


def build_validity_receipt(
    summary: dict[str, Any],
    manifest_row: dict[str, str],
    manifest_path: str | Path,
    source_snapshot_sha256: str,
    action_counts: list[int],
) -> dict[str, Any]:
    """Select structural fields from an in-process runner result.

    The emitted object contains no evaluator outcome fields.  Acceptance later
    reads this dedicated object and never opens evaluator summaries or episodes.
    """

    method = str(manifest_row["method"])
    diagnostics = summary.get("fit_diagnostics") or {}
    selected = {
        key: diagnostics[key]
        for key in sorted(MECHANISM_FIELDS.get(method, set()))
        if key in diagnostics and _finite_scalar(diagnostics[key])
    }
    required_counts = {
        "target_rows": int(summary.get("target_rows", 0)),
        "actual_rows": int(summary.get("actual_rows", 0)),
        "overshoot_rows": int(summary.get("overshoot_rows", 0)),
        "episode_count": int(summary.get("episode_count", 0)),
    }
    resource_names = (
        "manifest_row_seconds", "row_seconds", "dataset_seconds",
        "belief_cache_seconds", "policy_init_seconds", "fit_seconds",
        "evaluation_seconds", "summary_save_seconds", "manifest_row_peak_rss_mb",
        "peak_rss_mb",
    )
    resources = {
        key: float(summary[key])
        for key in resource_names
        if key in summary and _finite_scalar(summary[key])
    }
    receipt = {
        "schema": SCHEMA,
        "submitted": False,
        "identity": {
            "source_snapshot_sha256": str(source_snapshot_sha256),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_index": int(manifest_row["index"]),
            "method": method,
            "population": str(manifest_row["population"]),
            "environment": str(manifest_row["environment"]),
            "sigma_obs": float(manifest_row["sigma_obs"]),
            "expose_rk": str(manifest_row["expose_rk"]),
            "collection_seed": int(manifest_row.get("collection_seed", 116)),
            "evaluation_seed": int(manifest_row.get("evaluation_seed", 9001)),
        },
        "dataset": {
            "sha256": str(summary.get("dataset_sha256", "")),
            **required_counts,
            "episode_length": int(manifest_row.get("episode_length", 25)),
            "action_count": int(manifest_row.get("num_actions", 0)),
            "logged_action_counts": [int(value) for value in action_counts],
        },
        "fit": {
            "completed": True,
            "finite_selected_diagnostics": all(_finite_scalar(v) for v in selected.values()),
            "selected_diagnostic_count": len(selected),
        },
        "mechanism": selected,
        "resources": resources,
        "execution": {"exit_code": 0, "dependency_satisfied": True},
        "threads": {
            name: int(os.environ.get(name, "1"))
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    return receipt


def _reject_forbidden_raw_keys(raw: bytes) -> None:
    for match in _JSON_KEY.finditer(raw):
        # Decode only the JSON key token; values have not been deserialized.
        key = json.loads(b'"' + match.group(1) + b'"')
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
            raise ValueError(f"forbidden field in validity receipt: {key}")


def _validate_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite validity value at {path}")


def load_validity_receipt(path: str | Path) -> dict[str, Any]:
    """Read only the dedicated, forbidden-field-free validity schema."""

    target = Path(path)
    if target.name != "validity_receipt.json":
        raise ValueError("acceptance reader only opens validity_receipt.json")
    raw = target.read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError("validity receipt exceeds the structural size limit")
    _reject_forbidden_raw_keys(raw)
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_FIELDS:
        raise ValueError("validity receipt top-level schema mismatch")
    if payload.get("schema") != SCHEMA or payload.get("submitted") is not False:
        raise ValueError("invalid or submitted validity receipt")
    _validate_finite(payload)
    if payload.get("execution", {}).get("exit_code") != 0:
        raise ValueError("canary row did not exit cleanly")
    if any(value != 1 for value in payload.get("threads", {}).values()):
        raise ValueError("canary validity receipt is not one-thread")
    return payload


def summarize_validity_receipts(paths: list[str | Path]) -> dict[str, Any]:
    receipts = [load_validity_receipt(path) for path in paths]
    return {
        "schema": SCHEMA,
        "receipts": len(receipts),
        "all_completed": all(item["fit"]["completed"] for item in receipts),
        "methods": sorted({item["identity"]["method"] for item in receipts}),
        "manifest_sha256": sorted({
            item["identity"]["manifest_sha256"] for item in receipts
        }),
        "source_snapshot_sha256": sorted({
            item["identity"]["source_snapshot_sha256"] for item in receipts
        }),
        "max_peak_rss_mb": max(
            (float(item["resources"].get("peak_rss_mb", 0.0)) for item in receipts),
            default=0.0,
        ),
        "max_row_seconds": max(
            (float(item["resources"].get("manifest_row_seconds", 0.0)) for item in receipts),
            default=0.0,
        ),
    }
