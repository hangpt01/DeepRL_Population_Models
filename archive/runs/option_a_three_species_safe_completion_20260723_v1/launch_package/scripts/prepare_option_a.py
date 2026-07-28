#!/usr/bin/env python3
"""Build and verify the immutable Option A manifests without opening returns."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
RUN_ROOT = PACKAGE.parent
WORKSPACE = RUN_ROOT.parents[1]
GENERAL_ROOT = Path(
    "/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/"
    "general_phase2e_full_sigma01_02_20260720_v1"
)
REGISTRY = GENERAL_ROOT / "manifests/ecological_dataset_reuse_registry_144.csv"
GENERAL_MANIFEST = GENERAL_ROOT / "manifests/full_general_sigma01_02_576_rows.csv"
PLUS_CODE = (
    WORKSPACE
    / "real_ecology_runs/ricker_only_plus_72_20260720/runtime_snapshot_routing_fix"
)
PLUS_SOURCE = PLUS_CODE / "experiments/ricker_only_plus_72/manifests"
MOOR_RUN = WORKSPACE / "real_ecology_runs/adapted_32cell_diagnostic_pending"
MOOR_CODE = MOOR_RUN / "code"
MOOR_SOURCE = MOOR_RUN / "manifests"
PYTHON = WORKSPACE / ".venv-paper-faithful/bin/python"

PLUS_METHOD = "plus_adapted_ricker_only_pbvi"
MOOR_METHOD = "moor_adapted_ricker_misspec_pbvi"
FAMILIES = ("ricker", "allee", "theta", "regime")
SIGMAS = ("0.1", "0.2")
EVALUATION_SEEDS = "7001;7051;7101;7151;7201"
EXPECTED = {
    "registry": "474d1a65d2e5ec28c741f5b7ac5691be891712e753ee6a9a6590337d62f5f0c4",
    "general_manifest": "1526ce08dcf1b1d148232c075d41dbd78f0cfa2d7119cff9e330f965b6451df4",
    "plus_fit": "cb478f860b5c523407f4f11f3731904285d2a23164103ed667180ff56b9abdea",
    "plus_plan": "398780b8267af1b53f4092994eb001caa335537910905190dc58acad23595cc2",
    "plus_config": "b871668bae1c971795267bc402c3450e1fe6a628b4e8fee91b1300cfd73fbc26",
    "plus_runtime": "7628816c85a39d49373892a0e72d1751de2f0a32dd003b93a54ca1f3e151982c",
    "plus_freeze": "3285a9cc512576fe6648558ecafa5d5715de6fddd91678fc33e2bbf641286dd4",
    "moor_fit": "a8f39d83eb70a32be2c42fabecc56ca4af4d3d40edb97f87242934085fd7ee86",
    "moor_plan": "f0322718a78d485827c39661e2f1ded4cbde1d8bccd4cd053f4d1fe93dec087d",
    "moor_config": "1b45850cfad0a2b08ea2021c19169c8c0935c3baa95ef6350a7fd1f8a3f617ac",
    "moor_runtime": "f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
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


def sigma_slug(sigma: str) -> str:
    return f"{float(sigma):g}".replace(".", "p")


def fit_cell(family: str, sigma: str) -> str:
    return (
        "regime_hidden/reward_safe/crab_eating_fox/"
        f"{family}/sigma_{sigma_slug(sigma)}"
    )


def registry_rows() -> dict[tuple[str, str], dict[str, str]]:
    selected = {
        (row["environment"], row["sigma_obs"]): row
        for row in read_csv(REGISTRY)
        if row["population"] == "Crab-eating fox" and row["reward_mode"] == "safe"
    }
    if set(selected) != {(family, sigma) for family in FAMILIES for sigma in SIGMAS}:
        raise RuntimeError("authoritative Crab safe registry coverage is not exactly 8 cells")
    return selected


def general_identity(family: str, sigma: str) -> list[int]:
    rows = [
        row
        for row in read_csv(GENERAL_MANIFEST)
        if row["population"] == "Crab-eating fox"
        and row["environment"] == family
        and row["sigma_obs"] == sigma
        and row["reward_mode"] == "safe"
    ]
    expected_methods = {
        "refplan",
        "ogsrl",
        "bamcts",
        "ensemble_value_disagreement_pessimism",
    }
    if len(rows) != 4 or {row["method"] for row in rows} != expected_methods:
        raise RuntimeError(f"general identity is not a four-method cell: {family}/{sigma}")
    for row in rows:
        if (
            row["collection_seed"] != "116"
            or row["evaluation_seeds"] != EVALUATION_SEEDS
            or row["evaluation_episodes"] != "20"
            or row["episodes_per_evaluation_seed"] != "4"
            or row["evaluation_horizon"] != "50"
            or row["horizon"] != "5"
        ):
            raise RuntimeError(f"general protocol mismatch: {family}/{sigma}")
    return [int(row["index"]) for row in rows]


def add_authoritative_fields(
    row: dict[str, str], registry: dict[str, str], general_indices: list[int]
) -> dict[str, str]:
    result = dict(row)
    result.update(
        {
            "authoritative_dataset_sha256": registry["dataset_sha256"],
            "authoritative_public_file_sha256": registry["public_file_sha256"],
            "authoritative_private_file_sha256": registry[
                "private_companion_sha256"
            ],
            "authoritative_registry_sha256": EXPECTED["registry"],
            "matched_general_indices": ";".join(str(value) for value in general_indices),
            "collection_seed": "116",
            "evaluation_seeds": EVALUATION_SEEDS,
            "evaluation_episodes": "20",
            "episodes_per_evaluation_seed": "4",
            "evaluation_horizon": "50",
            "planner_horizon": "5",
            "planner_discount": "0.95",
            "evaluation_discount": "0.95",
            "environment_randomness": "frozen_evaluator_seeded_by_registered_evaluation_seeds",
        }
    )
    return result


def make_plus(registry: dict[tuple[str, str], dict[str, str]]) -> tuple[list, list]:
    source_fit = read_csv(PLUS_SOURCE / "ricker_only_plus_fit_72.csv")
    source_plan = read_csv(PLUS_SOURCE / "ricker_only_plus_plan_72.csv")
    fit_rows, plan_rows = [], []
    for position, sigma in enumerate(SIGMAS):
        source_index = 68 + position
        fit = dict(source_fit[source_index])
        plan = dict(source_plan[source_index])
        if (
            fit["population"] != "Crab-eating fox"
            or fit["environment"] != "theta"
            or fit["sigma_obs"] != sigma
            or fit["reward_mode"] != "safe"
            or fit["method"] != PLUS_METHOD
        ):
            raise RuntimeError("PLUS source row identity changed")
        cell = fit_cell("theta", sigma)
        tag = "ricker_only_8_exact_general_dataset_alignment_v1"
        artifact = (
            "evaluation/regime_hidden/crab_eating_fox/theta/"
            f"sigma_{sigma_slug(sigma)}/{tag}/data_real/backend_numpy/"
            f"regime_hidden/reward_safe/{PLUS_METHOD}/faithful_internal/faithful_artifacts"
        )
        for row, stage in ((fit, "dynamics_fit"), (plan, "plan_evaluate")):
            row.update(
                {
                    "index": str(position),
                    "config_tag": tag,
                    "fit_manifest_index": str(position),
                    "fit_cell": cell,
                    "fit_receipt_path": (
                        f"fit_receipts/{cell}/{PLUS_METHOD}/fit_receipt.json"
                    ),
                    "evaluation_artifact_dir": artifact,
                    "reusable_candidate_references": "0",
                    "new_candidate_fits": "8",
                    "run_stage": stage,
                    "run_scope": "option_a_three_species_safe_completion_v1",
                    "source_preserved_manifest_index": str(source_index),
                }
            )
        fit["fit_cache_hit_required"] = "False"
        plan["fit_cache_hit_required"] = "True"
        general = general_identity("theta", sigma)
        fit_rows.append(add_authoritative_fields(fit, registry[("theta", sigma)], general))
        plan_rows.append(add_authoritative_fields(plan, registry[("theta", sigma)], general))
    return fit_rows, plan_rows


def moor_template(rows: list[dict[str, str]], family: str, sigma: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["method"] == MOOR_METHOD
        and row["environment"] == family
        and row["sigma_obs"] == sigma
        and row["reward_mode"] == "safe"
    ]
    if len(matches) != 2:
        raise RuntimeError(f"MOOR template multiplicity changed: {family}/{sigma}")
    return dict(matches[0])


def make_moor(registry: dict[tuple[str, str], dict[str, str]]) -> tuple[list, list]:
    source_fit = read_csv(MOOR_SOURCE / "diagnostic_fit_64.csv")
    source_plan = read_csv(MOOR_SOURCE / "diagnostic_plan_safe_64.csv")
    fit_rows, plan_rows = [], []
    position = 0
    for family in FAMILIES:
        for sigma in SIGMAS:
            fit = moor_template(source_fit, family, sigma)
            plan = moor_template(source_plan, family, sigma)
            cell = fit_cell(family, sigma)
            tag = "moor_crab_exact_general_dataset_v1"
            for row, stage in ((fit, "dynamics_fit"), (plan, "plan_evaluate")):
                row.update(
                    {
                        "index": str(position),
                        "population": "Crab-eating fox",
                        "recoverable": "True",
                        "actual_rows": "4000",
                        "overshoot_rows": "0",
                        "episode_count": "160",
                        "config_tag": tag,
                        "fit_manifest_index": str(position),
                        "fit_cell": cell,
                        "fit_receipt_path": (
                            f"fit_receipts/{cell}/{MOOR_METHOD}/fit_receipt.json"
                        ),
                        "fit_cache_key_reference": "fit_receipt_json.fit_cache_keys",
                        "model_hash_reference": "fit_receipt_json.parameter_hashes",
                        "run_stage": stage,
                        "run_scope": "option_a_three_species_safe_completion_v1",
                        "source_template_manifest_index": row["index"],
                    }
                )
            fit["fit_cache_hit_required"] = "False"
            plan["fit_cache_hit_required"] = "True"
            general = general_identity(family, sigma)
            fit_rows.append(add_authoritative_fields(fit, registry[(family, sigma)], general))
            plan_rows.append(add_authoritative_fields(plan, registry[(family, sigma)], general))
            position += 1
    return fit_rows, plan_rows


def copy_verified(source: Path, target: Path, expected_hash: str) -> None:
    if sha256(source) != expected_hash:
        raise RuntimeError(f"source hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != expected_hash:
            raise RuntimeError(f"refusing to overwrite mismatched staged data: {target}")
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != expected_hash:
        raise RuntimeError(f"staged hash mismatch: {temporary}")
    temporary.replace(target)


def stage_cell(
    output_root: Path, family: str, sigma: str, row: dict[str, str]
) -> dict[str, str]:
    relative = Path(fit_cell(family, sigma))
    source_public = GENERAL_ROOT / row["target_public"]
    source_private = (
        GENERAL_ROOT
        / "quarantine/private"
        / relative
        / "truth.npz"
    )
    target_public = output_root / "datasets" / relative / "public.npz"
    target_private = output_root / "private" / relative / "truth.npz"
    copy_verified(source_public, target_public, row["public_file_sha256"])
    copy_verified(source_private, target_private, row["private_companion_sha256"])
    surrogate_source = source_public.with_name(
        "public.regime_hidden.public_surrogate.v2.seed20116.npz"
    )
    surrogate_target = target_public.with_name(surrogate_source.name)
    if not surrogate_source.is_file():
        raise RuntimeError(f"authoritative public surrogate is missing: {surrogate_source}")
    copy_verified(surrogate_source, surrogate_target, sha256(surrogate_source))
    return {
        "family": family,
        "sigma_obs": sigma,
        "dataset_sha256": row["dataset_sha256"],
        "public_file_sha256": sha256(target_public),
        "private_file_sha256": sha256(target_private),
        "surrogate_file_sha256": sha256(surrogate_target),
        "staged_public": str(target_public.relative_to(RUN_ROOT)),
        "staged_private": str(target_private.relative_to(RUN_ROOT)),
        "staged_surrogate": str(surrogate_target.relative_to(RUN_ROOT)),
    }


def runtime_identity_checks() -> None:
    checks = {
        "registry": REGISTRY,
        "general_manifest": GENERAL_MANIFEST,
        "plus_fit": PLUS_SOURCE / "ricker_only_plus_fit_72.csv",
        "plus_plan": PLUS_SOURCE / "ricker_only_plus_plan_72.csv",
        "plus_config": PLUS_CODE / "configs/paper_faithful_hidden_ricker_only_plus_v1.yaml",
        "plus_freeze": PLUS_CODE / "experiments/ricker_only_plus_72/FREEZE_RECEIPT.json",
        "moor_fit": MOOR_SOURCE / "diagnostic_fit_64.csv",
        "moor_plan": MOOR_SOURCE / "diagnostic_plan_safe_64.csv",
        "moor_config": MOOR_CODE / "configs/paper_faithful_hidden.yaml",
    }
    for key, path in checks.items():
        if sha256(path) != EXPECTED[key]:
            raise RuntimeError(f"frozen identity mismatch for {key}: {path}")
    plus_receipt = json.loads(checks["plus_freeze"].read_text(encoding="utf-8"))
    if plus_receipt["ricker_only_runtime_digest"] != EXPECTED["plus_runtime"]:
        raise RuntimeError("PLUS runtime digest mismatch")
    moor_snapshot = json.loads(
        (
            MOOR_CODE
            / "docs/fix_implement_ecology_baseline/SNAPSHOT_FILE_MANIFEST_20260718.json"
        ).read_text(encoding="utf-8")
    )
    if moor_snapshot["runtime_code_digest"] != EXPECTED["moor_runtime"]:
        raise RuntimeError("MOOR runtime digest mismatch")
    if not PYTHON.is_file():
        raise RuntimeError(f"registered Python is unavailable: {PYTHON}")


def cache_reuse_audit(
    output_root: Path, rows: list[dict[str, str]], expected_per_row: int
) -> dict[str, Any]:
    cache = output_root / "fit_cache"
    keys = []
    if cache.is_dir():
        keys = [path.stem for path in cache.glob("*.json")]
    # New output roots are deliberately isolated. Existing caches are reusable only
    # after exact scientific keys are derived by the frozen fit runner, so an empty
    # root is the conservative, non-relabeling result of the prelaunch audit.
    return {
        "output_root": str(output_root.relative_to(RUN_ROOT)),
        "existing_cache_metadata_files": len(keys),
        "rows": len(rows),
        "candidate_slots": len(rows) * expected_per_row,
        "verified_reusable_candidate_slots": 0,
        "new_candidate_fits": len(rows) * expected_per_row,
    }


def main() -> None:
    runtime_identity_checks()
    registry = registry_rows()
    plus_fit, plus_plan = make_plus(registry)
    moor_fit, moor_plan = make_moor(registry)
    manifests = PACKAGE / "manifests"
    targets = {
        "plus_fit": (manifests / "plus_alignment_fit_2.csv", plus_fit),
        "plus_plan": (manifests / "plus_alignment_plan_2.csv", plus_plan),
        "moor_fit": (manifests / "moor_crab_fit_8.csv", moor_fit),
        "moor_plan": (manifests / "moor_crab_plan_8.csv", moor_plan),
    }
    for path, rows in targets.values():
        write_csv_atomic(path, rows)
    staged = []
    for sigma in SIGMAS:
        staged.append(
            stage_cell(RUN_ROOT / "plus_alignment", "theta", sigma, registry[("theta", sigma)])
        )
    for family in FAMILIES:
        for sigma in SIGMAS:
            staged.append(
                stage_cell(
                    RUN_ROOT / "moor_crab", family, sigma, registry[(family, sigma)]
                )
            )
    manifest_hashes = {key: sha256(path) for key, (path, _rows) in targets.items()}
    receipt = {
        "package_schema": "option_a_three_species_safe_completion_v1",
        "scope": {
            "species": [
                "Amur tiger",
                "Crab-eating fox",
                "Egyptian vulture",
            ],
            "families": list(FAMILIES),
            "sigma_obs": list(SIGMAS),
            "reward_mode": "safe",
            "methods": 6,
            "final_method_cells": 144,
            "new_scientific_rows": 10,
        },
        "frozen_inputs": EXPECTED,
        "manifest_sha256": manifest_hashes,
        "staged_data": staged,
        "cache_audit": {
            "plus": cache_reuse_audit(RUN_ROOT / "plus_alignment", plus_fit, 8),
            "moor": cache_reuse_audit(RUN_ROOT / "moor_crab", moor_fit, 1),
        },
        "reuse": {
            "general_rl_method_cells": 96,
            "exact_plus_method_cells": 22,
            "exact_moor_method_cells": 16,
            "new_plus_method_cells": 2,
            "new_moor_method_cells": 8,
        },
        "return_fields_opened": False,
    }
    write_json_atomic(PACKAGE / "PREPARATION_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
