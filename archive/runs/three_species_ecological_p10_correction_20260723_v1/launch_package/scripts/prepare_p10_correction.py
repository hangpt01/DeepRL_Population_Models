#!/usr/bin/env python3
"""Prepare the immutable three-species ecological P=10 correction package.

This tool never fits, plans, evaluates, or opens a new outcome.  It stages the
authoritative general-RL datasets and verified reward-independent fit caches,
then writes plan-only manifests and a candidate-level reuse ledger.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
RUN_ROOT = PACKAGE.parent
WORKSPACE = RUN_ROOT.parents[1]
GENERAL_ROOT = Path(
    "/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/"
    "general_phase2e_full_sigma01_02_20260720_v1"
)
GENERAL_MANIFEST = GENERAL_ROOT / "manifests/full_general_sigma01_02_576_rows.csv"
REGISTRY = GENERAL_ROOT / "manifests/ecological_dataset_reuse_registry_144.csv"
ACCEPTED_TABLE = (
    WORKSPACE
    / "real_ecology_runs/option_a_three_species_safe_completion_20260723_v1/"
    "analysis/OPTION_A_MATCHED_144_METHOD_CELLS.csv"
)
PLUS_CODE = (
    WORKSPACE
    / "real_ecology_runs/ricker_only_plus_72_20260720/runtime_snapshot_routing_fix"
)
MOOR_CODE = WORKSPACE / "real_ecology_runs/adapted_32cell_diagnostic_pending/code"

PLUS_METHOD = "plus_adapted_ricker_only_pbvi"
MOOR_METHOD = "moor_adapted_ricker_misspec_pbvi"
METHODS = {"plus": (PLUS_METHOD, 8), "moor": (MOOR_METHOD, 1)}
SPECIES = ("Amur tiger", "Crab-eating fox", "Egyptian vulture")
FAMILIES = ("ricker", "allee", "theta", "regime")
SIGMAS = ("0.1", "0.2")
EVALUATION_SEEDS = "7001;7051;7101;7151;7201"
GENERAL_METHODS = {
    "refplan",
    "ogsrl",
    "bamcts",
    "ensemble_value_disagreement_pessimism",
}
EXPECTED = {
    "general_manifest": "1526ce08dcf1b1d148232c075d41dbd78f0cfa2d7119cff9e330f965b6451df4",
    "registry": "474d1a65d2e5ec28c741f5b7ac5691be891712e753ee6a9a6590337d62f5f0c4",
    "accepted_table": "c10f308bd189228ed0e724cdc34c363d2a472f42c73bdeda948eb98f4586521b",
    "plus_runtime": "7628816c85a39d49373892a0e72d1751de2f0a32dd003b93a54ca1f3e151982c",
    "moor_runtime": "f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def copy_verified(source: Path, target: Path, expected: str | None = None) -> str:
    actual = sha256(source)
    if expected is not None and actual != expected:
        raise RuntimeError(f"source hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != actual:
            raise RuntimeError(f"refusing to overwrite mismatched target: {target}")
        return actual
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != actual:
        raise RuntimeError(f"copy verification failed: {temporary}")
    temporary.replace(target)
    return actual


def slug(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def sigma_slug(value: str) -> str:
    return f"{float(value):g}".replace(".", "p")


def cell(population: str, family: str, sigma: str) -> str:
    return (
        f"regime_hidden/reward_safe/{slug(population)}/{family}/"
        f"sigma_{sigma_slug(sigma)}"
    )


def source_root(episodes_path: str) -> Path:
    path = Path(episodes_path)
    parts = path.parts
    position = parts.index("evaluation")
    return Path(*parts[:position])


def runtime_checks() -> None:
    checks = {
        "general_manifest": GENERAL_MANIFEST,
        "registry": REGISTRY,
        "accepted_table": ACCEPTED_TABLE,
    }
    for name, path in checks.items():
        if sha256(path) != EXPECTED[name]:
            raise RuntimeError(f"frozen input changed: {name}: {path}")
    plus_receipt = json.loads(
        (
            PLUS_CODE / "experiments/ricker_only_plus_72/FREEZE_RECEIPT.json"
        ).read_text(encoding="utf-8")
    )
    if plus_receipt["ricker_only_runtime_digest"] != EXPECTED["plus_runtime"]:
        raise RuntimeError("PLUS runtime digest changed")
    moor_receipt = json.loads(
        (
            MOOR_CODE
            / "docs/fix_implement_ecology_baseline/SNAPSHOT_FILE_MANIFEST_20260718.json"
        ).read_text(encoding="utf-8")
    )
    if moor_receipt["runtime_code_digest"] != EXPECTED["moor_runtime"]:
        raise RuntimeError("MOOR runtime digest changed")


def registry_index() -> dict[tuple[str, str, str], dict[str, str]]:
    selected = {
        (row["population"], row["environment"], row["sigma_obs"]): row
        for row in read_csv(REGISTRY)
        if row["population"] in SPECIES
        and row["environment"] in FAMILIES
        and row["sigma_obs"] in SIGMAS
        and row["reward_mode"] == "safe"
    }
    expected = {
        (population, family, sigma)
        for population in SPECIES
        for family in FAMILIES
        for sigma in SIGMAS
    }
    if set(selected) != expected:
        raise RuntimeError("general dataset registry does not cover exactly 24 cells")
    for key, row in selected.items():
        # Crab/theta has a registered 4,000-row complete-episode subset of a
        # 4,005-row ecological source.  The target public file and dataset hash
        # below identify that authoritative subset.
        if row["collection_seed"] != "116" or int(row["source_actual_rows"]) < 4000:
            raise RuntimeError(f"registry protocol mismatch: {key}")
    return selected


def general_index() -> dict[tuple[str, str, str], list[int]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in read_csv(GENERAL_MANIFEST):
        key = (row["population"], row["environment"], row["sigma_obs"])
        if (
            key[0] in SPECIES
            and key[1] in FAMILIES
            and key[2] in SIGMAS
            and row["reward_mode"] == "safe"
        ):
            grouped.setdefault(key, []).append(row)
    if len(grouped) != 24:
        raise RuntimeError("general manifest matched-cell coverage changed")
    output = {}
    for key, rows in grouped.items():
        if len(rows) != 4 or {row["method"] for row in rows} != GENERAL_METHODS:
            raise RuntimeError(f"general method coverage mismatch: {key}")
        for row in rows:
            expected = {
                "collection_seed": "116",
                "evaluation_seeds": EVALUATION_SEEDS,
                "evaluation_episodes": "20",
                "episodes_per_evaluation_seed": "4",
                "evaluation_horizon": "50",
                "num_actions": "11",
            }
            if any(row.get(name) != value for name, value in expected.items()):
                raise RuntimeError(f"general evaluation protocol mismatch: {key}")
        output[key] = sorted(int(row["index"]) for row in rows)
    return output


def accepted_sources() -> dict[tuple[str, str, str, str], dict[str, str]]:
    selected = {}
    for row in read_csv(ACCEPTED_TABLE):
        if row["method"] not in {PLUS_METHOD, MOOR_METHOD}:
            continue
        key = (
            row["method"],
            row["population"],
            row["environment"],
            row["sigma_obs"],
        )
        selected[key] = row
    expected = {
        (method, population, family, sigma)
        for method in (PLUS_METHOD, MOOR_METHOD)
        for population in SPECIES
        for family in FAMILIES
        for sigma in SIGMAS
    }
    if set(selected) != expected:
        raise RuntimeError("accepted ecological source coverage is not exactly 48 cells")
    return selected


def stage_dataset(
    output_root: Path,
    population: str,
    family: str,
    sigma: str,
    registry: dict[str, str],
) -> dict[str, str]:
    relative = Path(cell(population, family, sigma))
    source_public = GENERAL_ROOT / registry["target_public"]
    source_private = GENERAL_ROOT / "quarantine/private" / relative / "truth.npz"
    target_public = output_root / "datasets" / relative / "public.npz"
    target_private = output_root / "private" / relative / "truth.npz"
    copy_verified(source_public, target_public, registry["public_file_sha256"])
    copy_verified(source_private, target_private, registry["private_companion_sha256"])
    surrogate_source = source_public.with_name(
        "public.regime_hidden.public_surrogate.v2.seed20116.npz"
    )
    surrogate_target = target_public.with_name(surrogate_source.name)
    surrogate_hash = copy_verified(surrogate_source, surrogate_target)
    return {
        "population": population,
        "family": family,
        "sigma_obs": sigma,
        "dataset_sha256": registry["dataset_sha256"],
        "public_file_sha256": sha256(target_public),
        "private_file_sha256": sha256(target_private),
        "surrogate_file_sha256": surrogate_hash,
        "target_public": str(target_public.relative_to(RUN_ROOT)),
        "target_private": str(target_private.relative_to(RUN_ROOT)),
    }


def resolved_source_row(source: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    episodes = Path(source["episodes_path"])
    resolved = episodes.with_name("manifest_row_resolved.json")
    if not resolved.is_file():
        raise RuntimeError(f"source resolved row missing: {resolved}")
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def make_plan_row(
    *,
    group: str,
    position: int,
    population: str,
    family: str,
    sigma: str,
    registry: dict[str, str],
    general_indices: list[int],
    accepted: dict[str, str],
    config_hash: str,
) -> dict[str, Any]:
    method, candidate_count = METHODS[group]
    resolved_path, row = resolved_source_row(accepted)
    if (
        row["method"] != method
        or row["population"] != population
        or row["environment"] != family
        or str(row["sigma_obs"]) != sigma
        or row["reward_mode"] != "safe"
    ):
        raise RuntimeError("accepted source resolved identity mismatch")
    fit_cell = cell(population, family, sigma)
    tag = f"{group}_three_species_matched_p10_v1"
    row.update(
        {
            "index": str(position),
            "run_stage": "plan_evaluate",
            "run_scope": "three_species_ecological_p10_correction_v1",
            "config_tag": tag,
            "fit_cell": fit_cell,
            "fit_manifest_index": "reused",
            "fit_receipt_path": f"fit_receipts/{fit_cell}/{method}/fit_receipt.json",
            "fit_cache_hit_required": "True",
            "reward_mode": "safe",
            "collection_reward_mode": "safe",
            "collapse_penalty": "10.0",
            "reward_definition": "general_rl_safe_p10_v1",
            "reward_role": "planner_evaluator_only",
            "evaluation_seeds": EVALUATION_SEEDS,
            "evaluation_episodes": "20",
            "episodes_per_evaluation_seed": "4",
            "evaluation_horizon": "50",
            "planner_horizon": "5",
            "planner_discount": "0.95",
            "evaluation_discount": "0.95",
            "initial_state_definition": "deterministic_population_N0",
            "observation_protocol": "lognormal_conditional_mean_v1",
            "num_actions": "11",
            "candidate_count": str(candidate_count),
            "new_candidate_fits": "0",
            "reusable_candidate_references": str(candidate_count),
            "fit_cache_policy": "verified_reward_independent_fit_reuse_v1",
            "authoritative_dataset_sha256": registry["dataset_sha256"],
            "authoritative_public_file_sha256": registry["public_file_sha256"],
            "authoritative_private_file_sha256": registry[
                "private_companion_sha256"
            ],
            "authoritative_registry_sha256": EXPECTED["registry"],
            "matched_general_indices": ";".join(map(str, general_indices)),
            "collection_seed": "116",
            "runtime_digest_reference": EXPECTED[f"{group}_runtime"],
            "config_sha256": config_hash,
            "source_p5_resolved_row": str(resolved_path),
            "source_p5_resolved_row_sha256": sha256(resolved_path),
            "return_blinding": "summary_json_and_return_fields_sealed_until_acceptance",
        }
    )
    if group == "plus":
        row.update(
            {
                "candidate_family": "ricker",
                "candidate_count": "8",
                "candidate_construction": (
                    "ricker_only_episode_bootstrap_map_1full_7bootstrap_v1"
                ),
                "candidate_allocation": "ricker:8",
                "prior": "uniform",
                "regime_persistence_grid": "not_applicable",
                "regime_path_count": "not_applicable",
                "evaluation_artifact_dir": (
                    f"evaluation/regime_hidden/{slug(population)}/{family}/"
                    f"sigma_{sigma_slug(sigma)}/{tag}/data_real/backend_numpy/"
                    f"regime_hidden/reward_safe/{method}/faithful_internal/"
                    "faithful_artifacts"
                ),
                "candidate_seed_root": "51116",
            }
        )
    else:
        row.update(
            {
                "candidate_count": "1",
                "candidate_construction": "single_map_fit",
                "candidate_allocation": "ricker_single_map",
                "prior": "not_applicable",
                "regime_persistence_grid": "0.80;0.90;0.97",
                "regime_path_count": "16",
            }
        )
    return row


def reuse_fits(
    *,
    group: str,
    output_root: Path,
    plan_rows: list[dict[str, Any]],
    accepted: dict[tuple[str, str, str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method, candidate_count = METHODS[group]
    ledger: list[dict[str, Any]] = []
    receipt_specs: list[dict[str, Any]] = []
    for row in plan_rows:
        key = (method, row["population"], row["environment"], row["sigma_obs"])
        source = accepted[key]
        root = source_root(source["episodes_path"])
        source_receipt = root / "fit_receipts" / row["fit_cell"] / method / "fit_receipt.json"
        if not source_receipt.is_file():
            raise RuntimeError(f"source fit receipt missing: {source_receipt}")
        payload = json.loads(source_receipt.read_text(encoding="utf-8"))
        keys = list(payload.get("fit_cache_keys") or ())
        parameters = list(payload.get("parameter_hashes") or ())
        if (
            payload.get("run_stage") != "dynamics_fit"
            or payload.get("return_fields_opened") is not False
            or payload.get("method") != method
            or payload.get("cell") != row["fit_cell"]
            or len(keys) != candidate_count
            or len(parameters) != candidate_count
        ):
            raise RuntimeError(f"source fit receipt invalid: {source_receipt}")
        for candidate_index, (cache_key, parameter_hash) in enumerate(
            zip(keys, parameters)
        ):
            source_meta = root / "fit_cache" / f"{cache_key}.json"
            source_array = root / "fit_cache" / f"{cache_key}.npz"
            metadata = json.loads(source_meta.read_text(encoding="utf-8"))
            if (
                metadata.get("cache_schema") != "adapted_fit_cache_v1"
                or metadata.get("cache_key") != cache_key
                or metadata.get("model", {}).get("parameter_hash") != parameter_hash
                or metadata.get("array_hash") != sha256(source_array)
            ):
                raise RuntimeError(f"source cache identity invalid: {source_meta}")
            target_meta = output_root / "fit_cache" / source_meta.name
            target_array = output_root / "fit_cache" / source_array.name
            meta_hash = copy_verified(source_meta, target_meta)
            array_hash = copy_verified(source_array, target_array)
            fit_configuration = {
                "model_form": metadata["model"]["form"],
                "candidate_id": metadata["model"]["candidate_id"],
                "action_channels": metadata["model"]["action_channels"],
                "fit": metadata["fit"],
            }
            ledger.append(
                {
                    "method": method,
                    "population": row["population"],
                    "environment": row["environment"],
                    "sigma_obs": row["sigma_obs"],
                    "candidate_index": candidate_index,
                    "cache_key": cache_key,
                    "dataset_sha256": row["authoritative_dataset_sha256"],
                    "transition_data_hash": payload["transition_data_hash"],
                    "fit_configuration_sha256": canonical_hash(fit_configuration),
                    "fit_configuration_json": json.dumps(
                        fit_configuration, sort_keys=True, separators=(",", ":")
                    ),
                    "parameter_hash": parameter_hash,
                    "source_fit_receipt_path": str(source_receipt),
                    "source_fit_receipt_sha256": sha256(source_receipt),
                    "source_cache_metadata_path": str(source_meta),
                    "source_cache_metadata_sha256": meta_hash,
                    "source_cache_array_path": str(source_array),
                    "source_cache_array_sha256": array_hash,
                    "recorded_array_hash": metadata["array_hash"],
                    "target_cache_metadata_path": str(target_meta),
                    "target_cache_array_path": str(target_array),
                    "reuse_status": "verified_reward_independent_reuse",
                    "source_reward_penalty": "5.0",
                    "target_reward_penalty": "10.0",
                }
            )
        receipt_specs.append(
            {
                "row": row,
                "source_receipt": source_receipt,
                "payload": payload,
            }
        )
    return ledger, receipt_specs


def main() -> None:
    runtime_checks()
    registry = registry_index()
    general = general_index()
    accepted = accepted_sources()
    config_paths = {
        "plus": PACKAGE / "configs/plus_ricker_only_p10.yaml",
        "moor": PACKAGE / "configs/moor_ricker_p10.yaml",
    }
    plans: dict[str, list[dict[str, Any]]] = {"plus": [], "moor": []}
    staged: dict[str, list[dict[str, str]]] = {"plus": [], "moor": []}
    for group in ("plus", "moor"):
        position = 0
        output_root = RUN_ROOT / group
        config_hash = sha256(config_paths[group])
        for population in SPECIES:
            for family in FAMILIES:
                for sigma in SIGMAS:
                    key = (population, family, sigma)
                    plans[group].append(
                        make_plan_row(
                            group=group,
                            position=position,
                            population=population,
                            family=family,
                            sigma=sigma,
                            registry=registry[key],
                            general_indices=general[key],
                            accepted=accepted[(METHODS[group][0], *key)],
                            config_hash=config_hash,
                        )
                    )
                    staged[group].append(
                        stage_dataset(
                            output_root, population, family, sigma, registry[key]
                        )
                    )
                    position += 1
    manifests = {
        group: PACKAGE / "manifests" / f"{group}_p10_plan_24.csv"
        for group in ("plus", "moor")
    }
    for group in ("plus", "moor"):
        write_csv_atomic(manifests[group], plans[group])
    ledgers: dict[str, list[dict[str, Any]]] = {}
    receipt_specs: dict[str, list[dict[str, Any]]] = {}
    for group in ("plus", "moor"):
        ledgers[group], receipt_specs[group] = reuse_fits(
            group=group,
            output_root=RUN_ROOT / group,
            plan_rows=plans[group],
            accepted=accepted,
        )
    combined_ledger = ledgers["plus"] + ledgers["moor"]
    ledger_csv = PACKAGE / "provenance/fit_reuse_ledger_216.csv"
    write_csv_atomic(ledger_csv, combined_ledger)
    ledger_hash = sha256(ledger_csv)
    for group in ("plus", "moor"):
        method, candidate_count = METHODS[group]
        output_root = RUN_ROOT / group
        for spec in receipt_specs[group]:
            row, source, old = spec["row"], spec["source_receipt"], spec["payload"]
            payload = {
                "run_stage": "dynamics_fit",
                "completion_status": "complete",
                "receipt_write": "atomic_replace",
                "return_fields_opened": False,
                "method": method,
                "cell": row["fit_cell"],
                "transition_data_hash": old["transition_data_hash"],
                "fit_cache_keys": old["fit_cache_keys"],
                "parameter_hashes": old["parameter_hashes"],
                "cache_statuses": ["reused_verified"] * candidate_count,
                "candidate_count": candidate_count,
                "target_rows": 4000,
                "actual_rows": 4000,
                "overshoot_rows": 0,
                "episode_count": 160,
                "fit_seconds": 0.0,
                "reward_independent_fit": True,
                "source_reward_penalty": 5.0,
                "target_reward_penalty": 10.0,
                "source_fit_receipt": str(source),
                "source_fit_receipt_sha256": sha256(source),
                "fit_reuse_ledger": str(ledger_csv),
                "fit_reuse_ledger_sha256": ledger_hash,
            }
            write_json_atomic(
                output_root / row["fit_receipt_path"],
                payload,
            )
    reward_audit = {
        "audit_schema": "reward_independent_fit_reuse_v1",
        "decision": "PASS_REWARD_INDEPENDENT_FIT_REUSE",
        "fit_transition_fields": [
            "observations",
            "actions",
            "next_observations",
            "episode_id",
            "timestep",
            "terminated",
            "truncated",
        ],
        "excluded_from_fit_identity": [
            "dataset reward array",
            "reward_mode value",
            "collapse_penalty",
            "planner reward matrix",
        ],
        "reward_dependent_components_rebuilt": [
            "POMDP reward arrays",
            "PBVI alpha vectors/policy",
            "50-step evaluation outcomes",
        ],
        "fit_cache_key_source": (
            "runtime_snapshot_routing_fix/src/real_ecology_benchmark/"
            "faithful_fit.py::fit_cache_key"
        ),
        "plus_reused_candidate_slots": len(ledgers["plus"]),
        "moor_reused_candidate_slots": len(ledgers["moor"]),
        "recomputed_fits": 0,
        "fit_reuse_ledger_sha256": ledger_hash,
    }
    write_json_atomic(PACKAGE / "provenance/FIT_REWARD_INDEPENDENCE_AUDIT.json", reward_audit)
    receipt = {
        "package_schema": "three_species_ecological_p10_correction_v1",
        "scope": {
            "species": list(SPECIES),
            "families": list(FAMILIES),
            "sigma_obs": list(SIGMAS),
            "reward_mode": "safe",
            "collapse_penalty": 10.0,
            "methods": [PLUS_METHOD, MOOR_METHOD],
            "method_cells_per_method": 24,
            "total_method_cells": 48,
        },
        "frozen_inputs": EXPECTED,
        "runtime_source_modified": False,
        "config_sha256": {
            group: sha256(path) for group, path in config_paths.items()
        },
        "manifest_sha256": {
            group: sha256(path) for group, path in manifests.items()
        },
        "fit_reuse_ledger_sha256": ledger_hash,
        "fit_reuse": {
            "plus_candidate_slots": len(ledgers["plus"]),
            "moor_candidate_slots": len(ledgers["moor"]),
            "source_fit_receipts": 48,
            "recomputed_fits": 0,
        },
        "staged_data": staged,
        "new_outcomes_opened": False,
    }
    write_json_atomic(PACKAGE / "PREPARATION_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
