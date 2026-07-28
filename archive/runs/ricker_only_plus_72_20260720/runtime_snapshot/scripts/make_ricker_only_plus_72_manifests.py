#!/usr/bin/env python3
"""Generate the fixed 72-cell Ricker-only PLUS manifests and reuse ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


POPULATIONS = (
    "Egyptian vulture",
    "Bottlenose dolphin",
    "Amur tiger",
    "Spotted turtle",
    "Asian elephant",
    "Puerto Rican parrot",
    "Iberian lynx",
    "Jaguar",
    "Crab-eating fox",
)
SINKS = {"Egyptian vulture", "Bottlenose dolphin"}
FAMILIES = ("ricker", "allee", "theta", "regime")
SIGMAS = (0.1, 0.2)
METHOD = "plus_adapted_ricker_only_pbvi"
SOURCE_METHOD = "plus_adapted_mechanistic_pbvi"
CONSTRUCTION = "ricker_only_episode_bootstrap_map_1full_7bootstrap_v1"
CONFIG_TAG = "ricker_only_8_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sigma_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def cells():
    for population in POPULATIONS:
        for family in FAMILIES:
            for sigma in SIGMAS:
                yield population, family, sigma


def row_for(
    position: int,
    population: str,
    family: str,
    sigma: float,
    stage: str,
    runtime_digest: str,
    config_hash: str,
) -> dict[str, object]:
    fit_cell = (
        f"regime_hidden/reward_safe/{slug(population)}/{family}/sigma_{sigma_slug(sigma)}"
    )
    artifact_dir = (
        f"evaluation/regime_hidden/{slug(population)}/{family}/sigma_{sigma_slug(sigma)}"
        f"/{CONFIG_TAG}/data_real/backend_numpy/regime_hidden/reward_safe/{METHOD}"
        "/faithful_internal/faithful_artifacts"
    )
    reusable = population in {"Amur tiger", "Egyptian vulture"}
    return {
        "index": position,
        "reward_mode": "safe",
        "collection_reward_mode": "safe",
        "population": population,
        "recoverable": population not in SINKS,
        "environment": family,
        "num_actions": 11,
        "sigma_obs": sigma,
        "method": METHOD,
        "filter": "faithful_internal",
        "expose_rk": "hidden",
        "target_rows": 4000,
        "method_impl_version": "plus_adapted_ricker_only_pbvi_v1",
        "planner": "pbvi",
        "candidate_family": "ricker",
        "candidate_count": 8,
        "candidate_construction": CONSTRUCTION,
        "candidate_allocation": "ricker:8",
        "prior": "uniform",
        "fit_budget_id": "registered_s8_i100_m16",
        "discretization_id": "registered_b41_c9_o41",
        "equation_version": "adapted_mechanistic_v2",
        "regularization_variant": "none_structural_v1",
        "regime_persistence_grid": "not_applicable",
        "regime_path_count": "not_applicable",
        "observation_protocol": "lognormal_conditional_mean_v1",
        "fit_cache_policy": "content_hash_verified_reuse_v1",
        "run_stage": "dynamics_fit" if stage == "fit" else "plan_evaluate",
        "reward_role": (
            "fit_only_reward_excluded_from_fit_identity"
            if stage == "fit"
            else "planner_evaluator_only"
        ),
        "config_tag": CONFIG_TAG,
        "fit_manifest_index": position,
        "fit_cell": fit_cell,
        "fit_receipt_path": f"fit_receipts/{fit_cell}/{METHOD}/fit_receipt.json",
        "fit_cache_hit_required": stage == "plan",
        "evaluation_artifact_dir": artifact_dir,
        "candidate_seed_root": 51116,
        "reusable_candidate_references": 4 if reusable else 0,
        "new_candidate_fits": 4 if reusable else 8,
        "runtime_digest_reference": runtime_digest,
        "config_sha256": config_hash,
        "return_blinding": (
            "no_planner_no_evaluator_no_return_or_action_quality_opened"
            if stage == "fit"
            else "summary_json_and_all_return_fields_quarantined"
        ),
        "run_scope": "ricker_only_plus_72cell_stress_test_v1",
    }


def write_manifest(
    path: Path, rows: list[dict[str, object]], line_terminator: str = "\r\n"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator=line_terminator
        )
        writer.writeheader()
        writer.writerows(rows)


def source_receipt(source_root: Path, population: str, family: str, sigma: float) -> Path:
    return (
        source_root
        / "fit_receipts/regime_hidden/reward_safe"
        / slug(population)
        / family
        / f"sigma_{sigma_slug(sigma)}"
        / SOURCE_METHOD
        / "fit_receipt.json"
    )


def build_reuse_ledger(source_root: Path) -> list[dict[str, object]]:
    ledger: list[dict[str, object]] = []
    for population in ("Egyptian vulture", "Amur tiger"):
        for family in FAMILIES:
            for sigma in SIGMAS:
                receipt_path = source_receipt(source_root, population, family, sigma)
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt.get("return_fields_opened") is not False:
                    raise RuntimeError(f"unsafe source receipt: {receipt_path}")
                for candidate_index, (key, parameter_hash) in enumerate(
                    zip(receipt["fit_cache_keys"][:4], receipt["parameter_hashes"][:4])
                ):
                    metadata = source_root / "fit_cache" / f"{key}.json"
                    arrays = source_root / "fit_cache" / f"{key}.npz"
                    payload = json.loads(metadata.read_text(encoding="utf-8"))
                    if payload["model"]["form"] != "ricker":
                        raise RuntimeError(f"non-Ricker reuse candidate: {metadata}")
                    if payload["model"]["parameter_hash"] != parameter_hash:
                        raise RuntimeError(f"parameter hash mismatch: {metadata}")
                    if payload["fit"]["transition_data_hash"] != receipt["transition_data_hash"]:
                        raise RuntimeError(f"transition hash mismatch: {metadata}")
                    ledger.append(
                        {
                            "population": population,
                            "environment": family,
                            "sigma_obs": sigma,
                            "candidate_index": candidate_index,
                            "candidate_id": payload["model"]["candidate_id"],
                            "candidate_seed": 51116 + 10000 * candidate_index,
                            "cache_key": key,
                            "parameter_hash": parameter_hash,
                            "transition_data_hash": receipt["transition_data_hash"],
                            "public_data_hash": payload["fit"]["public_data_hash"],
                            "random_bank_hash": payload["fit"]["random_bank_hash"],
                            "fit_episode_multiset_hash": stable_hash(
                                payload["fit"]["fit_episode_ids"]
                            ),
                            "holdout_episode_ids_hash": stable_hash(
                                payload["fit"]["holdout_episode_ids"]
                            ),
                            "metadata_sha256": sha256(metadata),
                            "arrays_sha256": sha256(arrays),
                            "source_runtime_digest": (
                                "f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333"
                            ),
                            "source_experiment": "adapted_32cell_diagnostic_pending",
                            "reuse_gate": "all_hashes_and_episode_splits_must_match",
                        }
                    )
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--runtime-digest", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--canary-config", type=Path)
    args = parser.parse_args()
    config_hash = sha256(args.config)
    fit_rows, plan_rows = [], []
    for position, (population, family, sigma) in enumerate(cells()):
        fit_rows.append(
            row_for(position, population, family, sigma, "fit", args.runtime_digest, config_hash)
        )
        plan_rows.append(
            row_for(position, population, family, sigma, "plan", args.runtime_digest, config_hash)
        )
    if len(fit_rows) != 72 or len(plan_rows) != 72:
        raise RuntimeError("fixed scope must contain exactly 72 fit and 72 plan rows")
    write_manifest(args.output_dir / "ricker_only_plus_fit_72.csv", fit_rows)
    write_manifest(args.output_dir / "ricker_only_plus_plan_72.csv", plan_rows)
    if args.canary_config:
        canary_hash = sha256(args.canary_config)
        canary_fit = row_for(
            0, "Amur tiger", "ricker", 0.1, "fit", args.runtime_digest, canary_hash
        )
        canary_plan = row_for(
            0, "Amur tiger", "ricker", 0.1, "plan", args.runtime_digest, canary_hash
        )
        for row in (canary_fit, canary_plan):
            row.update({
                "target_rows": 160,
                "fit_budget_id": "canary_s2_i12_m4",
                "discretization_id": "canary_b15_c5_o15",
                "config_tag": "ricker_only_8_structural_smoke_v1",
                "reusable_candidate_references": 0,
                "new_candidate_fits": 8,
                "fit_cache_policy": "isolated_structural_smoke_no_production_reuse_v1",
                "run_scope": "ricker_only_plus_structural_smoke_v1",
            })
        canary_cell = "regime_hidden/reward_safe/amur_tiger/ricker/sigma_0p1"
        canary_receipt = f"fit_receipts/{canary_cell}/{METHOD}/fit_receipt.json"
        canary_artifacts = (
            "evaluation/regime_hidden/amur_tiger/ricker/sigma_0p1/"
            "ricker_only_8_structural_smoke_v1/data_real/backend_numpy/regime_hidden/"
            f"reward_safe/{METHOD}/faithful_internal/faithful_artifacts"
        )
        for row in (canary_fit, canary_plan):
            row["fit_receipt_path"] = canary_receipt
            row["evaluation_artifact_dir"] = canary_artifacts
        write_manifest(
            args.output_dir / "ricker_only_plus_canary_fit_1.csv",
            [canary_fit], line_terminator="\n",
        )
        write_manifest(
            args.output_dir / "ricker_only_plus_canary_plan_1.csv",
            [canary_plan], line_terminator="\n",
        )
    ledger = build_reuse_ledger(args.source_run)
    if len(ledger) != 64:
        raise RuntimeError("reuse ledger must contain exactly 64 candidate references")
    write_manifest(args.output_dir / "verified_reuse_64.csv", ledger)


if __name__ == "__main__":
    main()
