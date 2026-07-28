#!/usr/bin/env python3
"""Return-blind launch validation and positional dispatch for the frozen 32-cell run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


FIT_SHA256 = "a8f39d83eb70a32be2c42fabecc56ca4af4d3d40edb97f87242934085fd7ee86"
PLAN_SHA256 = "f0322718a78d485827c39661e2f1ded4cbde1d8bccd4cd053f4d1fe93dec087d"
RUNTIME_DIGEST = "f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333"
SNAPSHOT_COMMIT = "fd50c38c9fad9ca113c224826b9bebddedea67d0"
METHODS = {
    "plus_adapted_mechanistic_pbvi": "plus_adapted_mechanistic_fixed_pi_pbvi_v2",
    "moor_adapted_ricker_misspec_pbvi": "moor_adapted_ricker_misspec_pbvi_v2",
}
METHOD_GROUPS = {
    "all": None,
    "plus": "plus_adapted_mechanistic_pbvi",
    "moor": "moor_adapted_ricker_misspec_pbvi",
}
PAIR_FIELDS = (
    "population",
    "environment",
    "sigma_obs",
    "method",
    "filter",
    "expose_rk",
    "target_rows",
    "candidate_count",
    "candidate_construction",
    "fit_budget_id",
    "discretization_id",
    "equation_version",
    "regularization_variant",
    "regime_persistence_grid",
    "regime_path_count",
    "candidate_allocation",
    "action_channel_schema_hash",
    "observation_protocol",
    "fit_cache_policy",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def slug(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sigma_slug(value: str) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def fit_cell(row: dict[str, str]) -> str:
    reward = row.get("collection_reward_mode") or row["reward_mode"]
    return f"regime_{row['expose_rk']}/reward_{reward}/{slug(row['population'])}/{row['environment']}/sigma_{sigma_slug(row['sigma_obs'])}"


def eval_key(row: dict[str, str]) -> str:
    return f"regime_{row['expose_rk']}/{slug(row['population'])}/{row['environment']}/sigma_{sigma_slug(row['sigma_obs'])}/{row['method']}/{row['filter']}"


def identity(row: dict[str, str], config_hash: str) -> dict[str, str]:
    data_spec = {
        k: row.get(k, "")
        for k in (
            "population",
            "environment",
            "sigma_obs",
            "expose_rk",
            "target_rows",
            "collection_reward_mode",
        )
    }
    transition_spec = {
        **data_spec,
        **{
            k: row.get(k, "")
            for k in ("observation_protocol", "action_channel_schema_hash", "fit_budget_id")
        },
    }
    seed_spec = {
        "base_seed": 116,
        "method_fit_offset": 51000 if row["method"].startswith("plus_") else 47000,
        "fit_budget_id": row["fit_budget_id"],
        "configuration_sha256": config_hash,
        "cell": fit_cell(row),
    }
    return {
        "data_identity_hash": stable_hash(data_spec),
        "transition_identity_hash": stable_hash(transition_spec),
        "seed_identity_hash": stable_hash(seed_spec),
    }


def verify_runtime(code_root: Path) -> dict[str, Any]:
    manifest_path = (
        code_root / "docs/fix_implement_ecology_baseline/SNAPSHOT_FILE_MANIFEST_20260718.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("commit") != SNAPSHOT_COMMIT
        or payload.get("runtime_code_digest") != RUNTIME_DIGEST
    ):
        raise ValueError("snapshot manifest identity mismatch")
    checked = 0
    for entry in payload["files"]:
        relative = str(entry["path"])
        if relative.startswith("src/real_ecology_benchmark/") and relative.endswith(".py"):
            path = code_root / relative
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"frozen runtime file mismatch: {relative}")
            checked += 1
    if checked == 0:
        raise ValueError("snapshot manifest contains no runtime files")
    return {"runtime_digest": RUNTIME_DIGEST, "runtime_files_verified": checked}


def validate(
    fit_manifest: Path, plan_manifest: Path, code_root: Path, python_bin: Path
) -> dict[str, Any]:
    if sha256_file(fit_manifest) != FIT_SHA256 or sha256_file(plan_manifest) != PLAN_SHA256:
        raise ValueError("frozen manifest hash mismatch")
    runtime = verify_runtime(code_root)
    config = code_root / "configs/paper_faithful_hidden.yaml"
    config_hash = sha256_file(config)
    fit_rows, plan_rows = read_rows(fit_manifest), read_rows(plan_manifest)
    if len(fit_rows) != 64 or len(plan_rows) != 64:
        raise ValueError("expected exactly 64 fit and 64 plan rows")
    pairs = []
    fit_outputs: set[str] = set()
    plan_outputs: set[str] = set()
    for position, (fit, plan) in enumerate(zip(fit_rows, plan_rows)):
        if fit.get("run_stage") != "dynamics_fit" or plan.get("run_stage") != "plan_evaluate":
            raise ValueError(f"unsupported stage at position {position}")
        if fit["method"] not in METHODS or plan["method"] not in METHODS:
            raise ValueError(f"unsupported method at position {position}")
        if any(fit.get(field, "") != plan.get(field, "") for field in PAIR_FIELDS):
            raise ValueError(f"fit/plan identity mismatch at position {position}")
        left, right = identity(fit, config_hash), identity(plan, config_hash)
        if left != right:
            raise ValueError(f"data/transition/seed identity mismatch at position {position}")
        receipt = f"fit_receipts/{fit_cell(fit)}/{fit['method']}/fit_receipt.json"
        output = eval_key(plan)
        if receipt in fit_outputs or output in plan_outputs:
            raise ValueError(f"output collision at position {position}")
        fit_outputs.add(receipt)
        plan_outputs.add(output)
        pairs.append(
            {
                "position": position,
                "fit_manifest_index": int(fit["index"]),
                "plan_manifest_index": int(plan["index"]),
                "dependency_key": position,
                **left,
            }
        )
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise ValueError(f"Python executable unavailable: {python_bin}")
    for runner in ("scripts/run_adapted_fit_row.py", "scripts/run_real_manifest_row.py"):
        if not (code_root / runner).is_file():
            raise ValueError(f"frozen runner unavailable: {runner}")
    return {
        **runtime,
        "fit_manifest_sha256": FIT_SHA256,
        "plan_manifest_sha256": PLAN_SHA256,
        "config_sha256": config_hash,
        "pair_count": len(pairs),
        "unique_fit_outputs": len(fit_outputs),
        "unique_plan_outputs": len(plan_outputs),
        "pairs": pairs,
    }


def derived_plan_row(row: dict[str, str], position: int) -> dict[str, str]:
    result = dict(row)
    result["fit_manifest_index"] = str(position)
    result["fit_cell"] = fit_cell(row)
    result["fit_receipt_path"] = f"fit_receipts/{fit_cell(row)}/{row['method']}/fit_receipt.json"
    result["fit_cache_key_reference"] = "fit_receipt_json.fit_cache_keys"
    result["model_hash_reference"] = "fit_receipt_json.parameter_hashes"
    result["fit_cache_hit_required"] = "True"
    return result


def write_derived(plan_manifest: Path, derived_root: Path) -> list[dict[str, str]]:
    rows = read_rows(plan_manifest)
    derived_root.mkdir(parents=True, exist_ok=True)
    records = []
    for position, row in enumerate(rows):
        derived = derived_plan_row(row, position)
        target = derived_root / f"plan_position_{position:02d}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(derived))
            writer.writeheader()
            writer.writerow(derived)
        records.append(
            {
                "position": position,
                "stored_manifest_index": int(row["index"]),
                "path": str(target),
                "sha256": sha256_file(target),
            }
        )
    return records


def positions_for_method(rows: list[dict[str, str]], method_group: str) -> list[int]:
    if method_group not in METHOD_GROUPS:
        raise ValueError(f"unsupported method group: {method_group}")
    method = METHOD_GROUPS[method_group]
    return [
        position for position, row in enumerate(rows) if method is None or row["method"] == method
    ]


def resolved_command(
    stage: str,
    position: int,
    fit_manifest: Path,
    plan_manifest: Path,
    derived_root: Path,
    run_root: Path,
    code_root: Path,
    python_bin: Path,
    threads: int = 2,
) -> dict[str, Any]:
    source = fit_manifest if stage == "fit" else plan_manifest
    rows = read_rows(source)
    if stage not in {"fit", "plan"} or not 0 <= position < len(rows):
        raise ValueError("stage must be fit/plan and position must be 0..63")
    row = rows[position]
    if stage == "fit":
        if row["run_stage"] != "dynamics_fit":
            raise ValueError("fit dispatch received unsupported stage")
        manifest = fit_manifest
        runner = code_root / "scripts/run_adapted_fit_row.py"
    else:
        if row["run_stage"] != "plan_evaluate":
            raise ValueError("plan dispatch received unsupported stage")
        manifest = derived_root / f"plan_position_{position:02d}.csv"
        runner = code_root / "scripts/run_real_manifest_row.py"
    command = [
        str(python_bin),
        str(code_root / "scripts/phase2_thread_entry.py"),
        "--threads",
        str(threads),
        str(runner),
        str(manifest),
        row["index"],
        "--config",
        str(code_root / "configs/paper_faithful_hidden.yaml"),
        "--output-root",
        str(run_root),
    ]
    if stage == "plan":
        command.append("--allow-uncalibrated")
    return {
        "stage": stage,
        "position": position,
        "stored_manifest_index": int(row["index"]),
        "method": row["method"],
        "run_stage": row["run_stage"],
        "output_key": (
            f"fit_receipts/{fit_cell(row)}/{row['method']}" if stage == "fit" else eval_key(row)
        ),
        "dependency_key": position,
        "threads": threads,
        "manifest_path": str(manifest),
        "return_fields_opened": False,
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fit-manifest", type=Path, required=True)
    common.add_argument("--plan-manifest", type=Path, required=True)
    common.add_argument("--derived-root", type=Path, required=True)
    common.add_argument("--run-root", type=Path, required=True)
    common.add_argument("--code-root", type=Path, required=True)
    common.add_argument("--python-bin", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("validate", parents=[common]).add_argument(
        "--write-derived", action="store_true"
    )
    dry = sub.add_parser("dry-run", parents=[common])
    dry.add_argument("--stage", choices=["fit", "plan", "all"], default="all")
    dry.add_argument("--position", type=int)
    dry.add_argument("--method-group", choices=sorted(METHOD_GROUPS), default="all")
    dry.add_argument("--threads", type=int, choices=(1, 2), default=2)
    execute = sub.add_parser("execute-row", parents=[common])
    execute.add_argument("--stage", required=True)
    execute.add_argument("--position", type=int, default=None)
    execute.add_argument("--method-group", choices=sorted(METHOD_GROUPS), default="all")
    execute.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    report = validate(args.fit_manifest, args.plan_manifest, args.code_root, args.python_bin)
    if args.action == "validate":
        if args.write_derived:
            report["derived_plan_rows"] = write_derived(args.plan_manifest, args.derived_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.action == "dry-run":
        source_rows = read_rows(args.fit_manifest)
        selected = positions_for_method(source_rows, args.method_group)
        positions = [selected[args.position]] if args.position is not None else selected
        stages = [args.stage] if args.stage != "all" else ["fit", "plan"]
        commands = [
            resolved_command(
                stage,
                pos,
                args.fit_manifest,
                args.plan_manifest,
                args.derived_root,
                args.run_root,
                args.code_root,
                args.python_bin,
                args.threads,
            )
            for stage in stages
            for pos in positions
        ]
        print(
            json.dumps(
                {
                    "validation": {k: v for k, v in report.items() if k != "pairs"},
                    "dry_run_count": len(commands),
                    "return_fields_opened": False,
                    "commands": commands,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    array_position = (
        args.position if args.position is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    )
    selected = positions_for_method(read_rows(args.fit_manifest), args.method_group)
    if not 0 <= array_position < len(selected):
        raise SystemExit("array position is outside the selected method group")
    position = selected[array_position]
    spec = resolved_command(
        args.stage,
        position,
        args.fit_manifest,
        args.plan_manifest,
        args.derived_root,
        args.run_root,
        args.code_root,
        args.python_bin,
        args.threads,
    )
    if args.stage == "plan" and not Path(spec["manifest_path"]).is_file():
        raise SystemExit("derived plan manifest is missing; validation/prepare gate was not run")
    os.execv(spec["command"][0], spec["command"])


if __name__ == "__main__":
    main()
