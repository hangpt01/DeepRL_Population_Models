#!/usr/bin/env python3
"""Run one method row from a real-ecology reward-mode manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.collector import calibration_summary
from real_ecology_benchmark.backend import resolve_backend_for_workload
from real_ecology_benchmark.config import load_config, real_environment_like
from real_ecology_benchmark.dataset import load_private
from real_ecology_benchmark.general_canary_acceptance import build_validity_receipt
from real_ecology_benchmark.pipeline import ensure_dataset, run_method
from real_ecology_benchmark.telemetry import elapsed_since, peak_rss_mb, perf_seconds


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sigma_slug(value: float | str) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def read_manifest_row(path: str | Path, index: int) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["index"]) == index]
    if len(matches) != 1:
        raise SystemExit(f"manifest index {index} matched {len(matches)} rows")
    return matches[0]


# Optional manifest columns that override the planner's search budget.  A blank cell
# means "leave the config default alone".  Used by the general-method search-budget
# audit; ordinary manifests carry none of these and are unaffected.
TUNING_COLUMNS: dict[str, type] = {
    "horizon": int,
    "sequences": int,
    "pessimism": float,
    "bamcts_depth": int,
    "bamcts_simulations": int,
    "ogsrl_cost_horizon": int,
    "ogsrl_deployment_rollouts": int,
}


def planner_overrides(row: dict[str, str]) -> dict[str, object]:
    """Read search-budget overrides from a manifest row (blank/absent -> no override)."""

    return {
        key: cast(row[key])
        for key, cast in TUNING_COLUMNS.items()
        if str(row.get(key, "")).strip() != ""
    }


def apply_row_config(
    cfg,
    row: dict[str, str],
    output_root: str | Path,
    dataset_root: str | Path | None = None,
):
    population = row["population"]
    family = row["environment"]
    reward_mode = row["reward_mode"] or row.get("collection_reward_mode", "safe")
    sigma = float(row["sigma_obs"])
    cfg.environment = real_environment_like(cfg.environment, population, family)
    cfg.environment = replace(
        cfg.environment,
        observation_noise_sigma=sigma,
        reward_mode=reward_mode,
        expose_rk=str(row.get("expose_rk", "hidden") or "hidden"),
    )
    overrides = planner_overrides(row)
    if overrides:
        cfg.planner = replace(cfg.planner, **overrides)
    if row.get("method") == "plus_adapted_mechanistic_pbvi":
        candidate_count = int(row.get("candidate_count", 16))
        if candidate_count % 4:
            raise ValueError("adapted PLUS candidate count must divide evenly across four forms")
        cfg.faithful.model = replace(
            cfg.faithful.model, candidates_per_form=candidate_count // 4
        )
    if str(row.get("regime_path_count", "")).strip():
        cfg.faithful.fit = replace(
            cfg.faithful.fit,
            regime_mc_paths=int(row["regime_path_count"]),
        )
    cfg.validate()

    root = Path(output_root)
    # Datasets may live under a DIFFERENT root than the outputs: the search-budget audit
    # reuses the completed motivation run's datasets rather than regenerating them, so the
    # two sides provably see identical offline data.  Defaults to output_root, so every
    # existing caller is unchanged.
    data_root = Path(dataset_root) if dataset_root else root
    regime = cfg.environment.expose_rk
    cell = (
        Path(f"regime_{regime}")
        / f"reward_{reward_mode}"
        / slug(population)
        / family
        / f"sigma_{sigma_slug(sigma)}"
    )
    eval_cell = (
        Path(f"regime_{regime}")
        / slug(population)
        / family
        / f"sigma_{sigma_slug(sigma)}"
    )
    # The output path MUST carry the config identity.  Without it, every tuning config for
    # a cell resolves to the same directory and they silently overwrite each other -- the
    # run "succeeds" and leaves one arbitrary config's results per cell.
    tag = str(row.get("config_tag", "")).strip()
    if tag:
        eval_cell = eval_cell / slug(tag)
    cfg.dataset.output = str(data_root / "datasets" / cell / "public.npz")
    cfg.dataset.private_output = str(data_root / "private" / cell / "truth.npz")
    cfg.evaluation.output_dir = str(root / "evaluation" / eval_cell)
    if row.get("method") in {
        "plus_adapted_mechanistic_pbvi",
        "moor_adapted_ricker_misspec_pbvi",
    }:
        cfg.faithful.fit_cache_dir = str(root / "fit_cache")
    return cfg, cell, eval_cell


def _resolve_run_relative(path: str | Path, output_root: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path(output_root) / value


def validate_adapted_fit_receipt_gate(
    row: dict[str, str],
    output_root: str | Path,
) -> dict[str, object] | None:
    """Fail before evaluation unless a plan row's fitted dynamics cache is ready."""

    receipt_value = str(row.get("fit_receipt_path", "")).strip()
    if not receipt_value:
        return None
    receipt_path = _resolve_run_relative(receipt_value, output_root)
    if not receipt_path.exists():
        raise SystemExit(f"required fit receipt is missing: {receipt_path}")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("run_stage") != "dynamics_fit":
        raise SystemExit(f"fit receipt is not a dynamics-fit receipt: {receipt_path}")
    if receipt.get("return_fields_opened") is not False:
        raise SystemExit(f"fit receipt is not return-blind: {receipt_path}")
    if receipt.get("method") != row.get("method"):
        raise SystemExit(
            f"fit receipt method {receipt.get('method')!r} does not match "
            f"plan row method {row.get('method')!r}"
        )
    expected_cell = str(row.get("fit_cell", "")).strip()
    if expected_cell and receipt.get("cell") != expected_cell:
        raise SystemExit(
            f"fit receipt cell {receipt.get('cell')!r} does not match "
            f"plan row fit_cell {expected_cell!r}"
        )
    keys = list(receipt.get("fit_cache_keys") or ())
    hashes = list(receipt.get("parameter_hashes") or ())
    if not keys or len(keys) != len(hashes):
        raise SystemExit(f"fit receipt has invalid cache key/hash lists: {receipt_path}")
    cache_root = Path(output_root) / "fit_cache"
    for key, parameter_hash in zip(keys, hashes):
        metadata_path = cache_root / f"{key}.json"
        array_path = cache_root / f"{key}.npz"
        if not metadata_path.exists() or not array_path.exists():
            raise SystemExit(f"required adapted fit cache entry is missing for key {key}")
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("cache_schema") != "adapted_fit_cache_v1":
            raise SystemExit(f"adapted fit cache schema mismatch for key {key}")
        if metadata.get("cache_key") != key:
            raise SystemExit(f"adapted fit cache key mismatch for key {key}")
        if metadata.get("model", {}).get("parameter_hash") != parameter_hash:
            raise SystemExit(f"adapted fit cache model hash mismatch for key {key}")
    return {
        "fit_receipt_path": str(receipt_path),
        "fit_cache_keys": keys,
        "parameter_hashes": hashes,
    }


def validate_adapted_plan_cache_hit(
    summary: dict[str, object],
    receipt_gate: dict[str, object] | None,
) -> dict[str, object]:
    if receipt_gate is None:
        return {}
    fit = summary.get("fit_diagnostics") or {}
    receipt_keys = list(receipt_gate["fit_cache_keys"])
    if "fit_cache_keys" in fit:
        observed_keys = list(fit["fit_cache_keys"])
        misses = float(fit.get("fit_cache_misses", 0.0))
        hits = float(fit.get("fit_cache_hits", 0.0))
        if observed_keys != receipt_keys or misses != 0.0 or hits != float(len(receipt_keys)):
            raise SystemExit("adapted PLUS plan row did not reuse every canary fit cache entry")
    else:
        observed_key = str(fit.get("fit_cache_key", ""))
        hit = float(fit.get("fit_cache_hit", 0.0))
        if [observed_key] != receipt_keys or hit != 1.0:
            raise SystemExit("adapted MOOR plan row did not reuse the canary fit cache entry")
    return {
        "fit_cache_reuse_gate": "passed",
        "validated_fit_receipt_path": str(receipt_gate["fit_receipt_path"]),
        "validated_fit_cache_keys": receipt_keys,
        "validated_parameter_hashes": list(receipt_gate["parameter_hashes"]),
    }


def write_json_atomic(path: str | Path, payload: dict[str, object], suffix: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temporary.replace(target)


def apply_backend_overrides(cfg, args):
    """Apply backend CLI/env overrides using the same precedence for all runners."""

    # Backend override precedence: --backend > $BACKEND env > config.
    backend = args.backend or os.environ.get("BACKEND")
    updates = {}
    if backend:
        updates["backend"] = backend
    if args.device is not None:
        updates["device"] = args.device
    if args.backend_strict is not None:
        updates["strict"] = args.backend_strict == "true"
    if updates:
        cfg.compute = replace(cfg.compute, **updates)
        cfg.validate()
    return cfg


def gate_path_for(output_root: str | Path, cell: Path, backend_name: str) -> Path:
    return Path(output_root) / "gates" / f"backend_{backend_name}" / cell.with_suffix(".json")


def maybe_require_gate(args, row: dict[str, str], output_root: str | Path, cell: Path, cfg) -> None:
    if not args.require_gate:
        return
    gate_backend = resolve_backend_for_workload(cfg.compute, "gate", cfg.environment)
    gate_path = gate_path_for(output_root, cell, gate_backend.name)
    if not gate_path.exists():
        raise SystemExit(f"required gate artifact is missing: {gate_path}")
    with gate_path.open("r", encoding="utf-8") as handle:
        gate = json.load(handle)
    if not gate.get("passed", False):
        raise SystemExit(
            "required gate failed for "
            f"{row['reward_mode']} {row['population']} {row['environment']} "
            f"sigma={row['sigma_obs']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("index", type=int)
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_experiment.yaml"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "real_reward_modes_20260703"))
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="read datasets from here instead of --output-root (nothing is regenerated); "
             "used by the search-budget audit to reuse the motivation run's datasets",
    )
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--require-gate", action="store_true")
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="accepted for launch compatibility; calibration is diagnostic by default",
    )
    parser.add_argument(
        "--require-band",
        action="store_true",
        help="abort if healthy-start incident collapse is outside [0.15, 0.24]",
    )
    parser.add_argument("--backend", choices=["numpy", "cupy"], default=None,
                        help="compute backend (default: config, or $BACKEND env)")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--backend-strict", choices=["true", "false"], default=None)
    args = parser.parse_args()

    row_start = perf_seconds()
    row = read_manifest_row(args.manifest, args.index)
    if row.get("job_kind") == "gate":
        raise SystemExit("gate rows must be run with run_real_gate_row.py")
    if row.get("run_stage") == "dynamics_fit":
        raise SystemExit("dynamics-fit rows must use run_adapted_fit_row.py")
    if not row.get("method") or not row.get("filter"):
        raise SystemExit(f"method/filter missing from manifest row {args.index}: {row}")

    cfg = load_config(args.config)
    cfg, cell, _eval_cell = apply_row_config(cfg, row, args.output_root, args.dataset_root)
    receipt_gate = validate_adapted_fit_receipt_gate(row, args.output_root)
    target_rows = str(row.get("target_rows", "")).strip()
    if target_rows and int(target_rows) != cfg.dataset.transitions:
        raise SystemExit(
            f"manifest target_rows={target_rows} disagrees with config "
            f"dataset.transitions={cfg.dataset.transitions}"
        )
    cfg = apply_backend_overrides(cfg, args)
    maybe_require_gate(args, row, args.output_root, cell, cfg)

    calibration_start = perf_seconds()
    dataset = ensure_dataset(cfg, regenerate=args.regenerate)
    for key in ("actual_rows", "overshoot_rows", "episode_count"):
        registered = str(row.get(key, "")).strip()
        if registered and int(registered) != int(dataset.metadata[key]):
            raise SystemExit(
                f"manifest {key}={registered} disagrees with dataset "
                f"{key}={dataset.metadata[key]}"
            )
    private = load_private(cfg.dataset.private_output)
    calibration = calibration_summary(dataset, private, cfg.environment.safety_threshold)
    calibration_path = Path(args.output_root) / "calibration" / cell.with_suffix(".json")
    write_json_atomic(calibration_path, calibration, f"{args.index}.{os.getpid()}")
    calibration_seconds = elapsed_since(calibration_start)
    if args.require_band and not calibration["collapse_band_pass"]:
        raise SystemExit(
            "calibration failed for "
            f"{row['reward_mode']} {row['population']} {row['environment']} "
            f"sigma={row['sigma_obs']}: "
            f"{calibration['incident_collapse_rate_healthy_starts']:.4f}"
        )

    summary = run_method(row["method"], cfg, row["filter"], regenerate=False)
    summary.update(validate_adapted_plan_cache_hit(summary, receipt_gate))
    summary.update({
        "manifest_calibration_seconds": calibration_seconds,
        "manifest_row_seconds": elapsed_since(row_start),
        "manifest_row_peak_rss_mb": peak_rss_mb(),
        # Stamp the config identity and the RESOLVED planner into the summary, so the
        # analysis reads what actually ran rather than what the manifest intended.
        "config_tag": str(row.get("config_tag", "")).strip() or "as_run",
        "planner_horizon": cfg.planner.horizon,
        "planner_sequences": cfg.planner.sequences,
        "planner_pessimism": cfg.planner.pessimism,
        "planner_bamcts_depth": cfg.planner.bamcts_depth,
        "planner_bamcts_simulations": cfg.planner.bamcts_simulations,
        "planner_ogsrl_cost_horizon": cfg.planner.ogsrl_cost_horizon,
        "planner_ogsrl_deployment_rollouts": cfg.planner.ogsrl_deployment_rollouts,
        "expose_rk": cfg.environment.expose_rk,
    })
    output_dir = summary.get("output_dir")
    if output_dir:
        resolved_manifest_path = Path(str(output_dir)) / "manifest_row_resolved.json"
        resolved_manifest = {
            **row,
            "expose_rk": cfg.environment.expose_rk,
            "target_rows": int(dataset.metadata["target_rows"]),
            "actual_rows": int(dataset.metadata["actual_rows"]),
            "overshoot_rows": int(dataset.metadata["overshoot_rows"]),
            "episode_count": int(dataset.metadata["episode_count"]),
        }
        write_json_atomic(
            resolved_manifest_path,
            resolved_manifest,
            f"{args.index}.{os.getpid()}.manifest",
        )
        summary["resolved_manifest_path"] = str(resolved_manifest_path)
        write_json_atomic(
            Path(str(output_dir)) / "summary.json",
            summary,
            f"{args.index}.{os.getpid()}.resources",
        )
        if str(row.get("package_role", "")).startswith("phase2e_"):
            validity = build_validity_receipt(
                summary,
                row,
                args.manifest,
                os.environ.get("GENERAL_CANARY_SNAPSHOT_SHA256", "unregistered"),
                [
                    int((dataset.actions == action).sum())
                    for action in range(cfg.environment.num_actions)
                ],
            )
            write_json_atomic(
                Path(str(output_dir)) / "validity_receipt.json",
                validity,
                f"{args.index}.{os.getpid()}.validity",
            )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
