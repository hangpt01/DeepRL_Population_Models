#!/usr/bin/env python3
"""Create registered validation-smoke or blinded-runtime faithful manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


METHODS = (
    "plus_adapted_mechanistic_pbvi",
    "moor_adapted_ricker_misspec_pbvi",
)


ACTION_CHANNELS = (
    "none",
    "rate",
    "rate",
    "rate",
    "rate",
    "capacity",
    "capacity",
    "rate+capacity",
    "rate+capacity",
    "rate+capacity",
    "state",
)
CHANNEL_SCHEMA_HASH = hashlib.sha256("\0".join(ACTION_CHANNELS).encode("ascii")).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sigma_slug(value: float | str) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def cell_path(
    population: str,
    form: str,
    sigma: float,
    reward_mode: str,
    expose_rk: str = "hidden",
) -> str:
    return str(
        Path(f"regime_{expose_rk}")
        / f"reward_{reward_mode}"
        / slug(population)
        / form
        / f"sigma_{sigma_slug(sigma)}"
    )


def canary_dynamics_cells():
    return (
        ("Amur tiger", "ricker", 0.1),
        ("Egyptian vulture", "regime", 0.4),
    )


def scientific_cells(mode: str):
    if mode in {"smoke", "one_cell_4000"}:
        return [("Amur tiger", "ricker", 0.1, "safe")]
    if mode == "canary_fit":
        return [
            (population, form, sigma, "")
            for population, form, sigma in canary_dynamics_cells()
        ]
    if mode == "canary_plan":
        return [
            (population, form, sigma, reward_mode)
            for population, form, sigma in canary_dynamics_cells()
            for reward_mode in ("safe", "yield")
        ]
    populations = ("Amur tiger", "Egyptian vulture")
    return [
        (population, form, sigma, reward_mode)
        for population in populations
        for form in ("ricker", "allee", "theta", "regime")
        for sigma in (0.0, 0.1, 0.2, 0.4)
        for reward_mode in ("safe", "yield")
    ]


def write_manifest(path: Path, mode: str) -> int:
    rows = []
    cells = scientific_cells(mode)
    if mode == "diagnostic_fit":
        cells = [cell for cell in cells if cell[3] == "safe"]
    canary_fit_index = {
        (population, form, sigma, method): index
        for index, (population, form, sigma, method) in enumerate(
            (
                (population, form, sigma, method)
                for population, form, sigma in canary_dynamics_cells()
                for method in METHODS
            )
        )
    }
    sensitivity_arms = (
        ("bank12", 12, 16),
        ("primary16", 16, 16),
        ("bank32", 32, 16),
        ("regime_paths8", 16, 8),
        ("regime_paths32", 16, 32),
    )
    for population, form, sigma, reward_mode in cells:
        arms = sensitivity_arms if mode == "sensitivity" else (("primary16", 16, 16),)
        methods = ("plus_adapted_mechanistic_pbvi",) if mode == "sensitivity" else METHODS
        for arm, plus_count, regime_paths in arms:
            for method in methods:
                plus = method == "plus_adapted_mechanistic_pbvi"
                run_stage = (
                    "dynamics_fit"
                    if mode in {"diagnostic_fit", "canary_fit"}
                    else "plan_evaluate"
                )
                collection_reward_mode = "safe" if mode == "canary_fit" else reward_mode
                fit_cell = (
                    cell_path(population, form, sigma, "safe")
                    if mode == "canary_plan"
                    else ""
                )
                fit_manifest_index = (
                    canary_fit_index[(population, form, sigma, method)]
                    if mode == "canary_plan"
                    else ""
                )
                fit_receipt_path = (
                    str(Path("fit_receipts") / fit_cell / method / "fit_receipt.json")
                    if mode == "canary_plan"
                    else ""
                )
                rows.append(
                    {
                        "index": len(rows),
                        "reward_mode": reward_mode,
                        "collection_reward_mode": collection_reward_mode,
                        "population": population,
                        "recoverable": population != "Egyptian vulture",
                        "environment": form,
                        "num_actions": 11,
                        "sigma_obs": sigma,
                        "method": method,
                        "filter": "faithful_internal",
                        "expose_rk": "hidden",
                        "target_rows": 160 if mode == "smoke" else 4000,
                        "actual_rows": "",
                        "overshoot_rows": "",
                        "episode_count": "",
                        "method_impl_version": (
                            "plus_adapted_mechanistic_fixed_pi_pbvi_v2"
                            if method == "plus_adapted_mechanistic_pbvi"
                            else "moor_adapted_ricker_misspec_pbvi_v2"
                        ),
                        "planner": "pbvi",
                        "candidate_count": (4 if mode == "smoke" else plus_count) if plus else 1,
                        "candidate_construction": (
                            "episode_bootstrap_map_fixed_pi_v2"
                            if method == "plus_adapted_mechanistic_pbvi"
                            else "single_map_fit"
                        ),
                        "prior": "uniform"
                        if method == "plus_adapted_mechanistic_pbvi"
                        else "not_applicable",
                        "fit_budget_id": "smoke_s2_i12_m4"
                        if mode == "smoke"
                        else "registered_s8_i100_m16",
                        "discretization_id": "smoke_b15_c5_o15"
                        if mode == "smoke"
                        else "registered_b41_c9_o41",
                        "equation_version": "adapted_mechanistic_v2",
                        "regularization_variant": "none_structural_v1",
                        "regime_persistence_grid": "0.80;0.90;0.97",
                        "regime_path_count": regime_paths,
                        "candidate_allocation": (
                            {12: "3;3;3;1,1,1", 16: "4;4;4;1,2,1", 32: "8;8;8;2,4,2"}.get(
                                plus_count, "1;1;1;0,1,0"
                            )
                            if plus
                            else "ricker_single_map"
                        ),
                        "action_channel_schema_hash": CHANNEL_SCHEMA_HASH,
                        "observation_protocol": "lognormal_conditional_mean_v1",
                        "fit_cache_policy": "fit_transition_hash_v1_reward_excluded",
                        "run_stage": run_stage,
                        "reward_role": (
                            "fit_only_reward_excluded_from_fit_identity"
                            if run_stage == "dynamics_fit"
                            else "planner_evaluator_only"
                        ),
                        "config_tag": arm if mode == "sensitivity" else "",
                        "fit_manifest_index": fit_manifest_index,
                        "fit_cell": fit_cell,
                        "fit_receipt_path": fit_receipt_path,
                        "fit_cache_key_reference": (
                            "fit_receipt_json.fit_cache_keys"
                            if mode == "canary_plan"
                            else ""
                        ),
                        "model_hash_reference": (
                            "fit_receipt_json.parameter_hashes"
                            if mode == "canary_plan"
                            else ""
                        ),
                        "fit_cache_hit_required": mode == "canary_plan",
                        "first_stage_cpu_ceiling_hours": (
                            24 if mode in {"canary_fit", "canary_plan"} else ""
                        ),
                        "cumulative_cpu_ceiling_hours": (
                            48 if mode in {"canary_fit", "canary_plan"} else ""
                        ),
                        "requested_cpu_ceiling_hours": (
                            600
                            if mode == "sensitivity"
                            else 48 if mode in {"canary_fit", "canary_plan"} else ""
                        ),
                        "headline_sweep_authorized": False,
                        "return_blinding": (
                            "no_planner_no_evaluator_no_return_or_action_quality_opened"
                            if run_stage == "dynamics_fit"
                            else "timing_convergence_cache_only_until_scope_frozen"
                        ),
                        "run_scope": (
                            "registered_diagnostic_subset"
                            if mode in {"diagnostic_fit", "diagnostic_plan", "sensitivity"}
                            else "corrected_canary"
                            if mode in {"canary_fit", "canary_plan"}
                            else mode
                        ),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "one_cell_4000",
            "canary_fit",
            "canary_plan",
            "diagnostic_fit",
            "diagnostic_plan",
            "sensitivity",
        ),
        default="smoke",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    rows = write_manifest(output, args.mode)
    print(json.dumps({"output": str(output), "rows": rows, "mode": args.mode}, indent=2))


if __name__ == "__main__":
    main()
