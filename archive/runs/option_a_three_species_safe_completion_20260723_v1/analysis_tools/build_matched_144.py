#!/usr/bin/env python3
"""Build the Option A 144-method-cell table only after both acceptance gates pass."""

from __future__ import annotations

import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import fmean, pstdev
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
RUN_ROOT = HERE.parents[1]
WORKSPACE = RUN_ROOT.parents[1]
GENERAL_ROOT = Path(
    "/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/"
    "general_phase2e_full_sigma01_02_20260720_v1"
)
PLUS_OLD = WORKSPACE / "real_ecology_runs/ricker_only_plus_72_20260720/production"
MOOR_OLD = WORKSPACE / "real_ecology_runs/adapted_32cell_diagnostic_pending"
GENERAL_METHODS = (
    "refplan",
    "ogsrl",
    "bamcts",
    "ensemble_value_disagreement_pessimism",
)
PLUS = "plus_adapted_ricker_only_pbvi"
MOOR = "moor_adapted_ricker_misspec_pbvi"
METHODS = (*GENERAL_METHODS, PLUS, MOOR)
SPECIES = (
    ("Amur tiger", "declining_recoverable"),
    ("Crab-eating fox", "strong_growth_recoverable"),
    ("Egyptian vulture", "sink"),
)
FAMILIES = ("ricker", "allee", "theta", "regime")
SIGMAS = ("0.1", "0.2")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sigma_slug(value: str) -> str:
    return f"{float(value):g}".replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_acceptance(path: Path, method: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("decision") != "PASS_LIMITED_STRUCTURAL_ACCEPTANCE"
        or payload.get("return_fields_opened") is not False
        or payload.get("method") != method
        or payload.get("completed_rows") != payload.get("expected_rows")
    ):
        raise RuntimeError(f"acceptance gate did not pass: {path}")
    return payload


def unique_episode_path(
    population: str, family: str, sigma: str, method: str
) -> Path:
    stem = (
        Path("evaluation/regime_hidden")
        / slug(population)
        / family
        / f"sigma_{sigma_slug(sigma)}"
    )
    if method in GENERAL_METHODS:
        base = GENERAL_ROOT / "quarantine" / stem
        matches = list(
            base.glob(
                f"**/regime_hidden/reward_safe/{method}/learned/episodes.csv"
            )
        )
    elif method == PLUS:
        base = (
            RUN_ROOT / "plus_alignment"
            if population == "Crab-eating fox" and family == "theta"
            else PLUS_OLD
        ) / stem
        matches = list(
            base.glob(
                f"**/regime_hidden/reward_safe/{PLUS}/faithful_internal/episodes.csv"
            )
        )
    elif method == MOOR:
        base = (
            RUN_ROOT / "moor_crab"
            if population == "Crab-eating fox"
            else MOOR_OLD
        ) / stem
        matches = list(
            base.glob(
                f"**/regime_hidden/reward_safe/{MOOR}/faithful_internal/episodes.csv"
            )
        )
    else:
        raise ValueError(method)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one episode table for {population}/{family}/{sigma}/{method}; "
            f"found {len(matches)}"
        )
    return matches[0]


def read_episodes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 episodes: {path}")
    if any(
        row["reward_mode"] != "safe"
        or row["expose_rk"] != "hidden"
        or int(row["n_steps"]) > 50
        for row in rows
    ):
        raise RuntimeError(f"episode protocol mismatch: {path}")
    return rows


def number(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {field}")
    return value


def boolean(row: dict[str, str], field: str) -> float:
    return 1.0 if row[field].strip().lower() in {"1", "true", "yes"} else 0.0


def mean(rows: list[dict[str, str]], field: str) -> float:
    return fmean(number(row, field) for row in rows)


def rate(rows: list[dict[str, str]], field: str) -> float:
    return fmean(boolean(row, field) for row in rows)


def pbvi_diagnostics(episodes: Path, method: str) -> dict[str, Any]:
    if method not in {PLUS, MOOR}:
        return {
            "pbvi_argmax_actions": "",
            "pbvi_action_margin_min": "",
            "pbvi_action_margin_mean": "",
        }
    path = episodes.parent / "faithful_artifacts/pbvi_policy_diagnostics.npz"
    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(payload["last_action_values"], dtype=float)
    if values.ndim != 2 or values.shape[1] != 11 or not np.isfinite(values).all():
        raise RuntimeError(f"invalid PBVI action diagnostics: {path}")
    ordered = np.sort(values, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    return {
        "pbvi_argmax_actions": ";".join(str(int(x)) for x in np.argmax(values, axis=1)),
        "pbvi_action_margin_min": float(np.min(margins)),
        "pbvi_action_margin_mean": float(np.mean(margins)),
    }


def aggregate_cell(
    population: str, population_class: str, family: str, sigma: str, method: str
) -> dict[str, Any]:
    path = unique_episode_path(population, family, sigma, method)
    rows = read_episodes(path)
    entropies = [number(row, "action_entropy") for row in rows]
    result: dict[str, Any] = {
        "population": population,
        "population_class": population_class,
        "environment": family,
        "sigma_obs": sigma,
        "reward_mode": "safe",
        "method": method,
        "episodes": 20,
        "evaluation_horizon": 50,
        "operational_return_mean": mean(rows, "operational_return"),
        "operational_return_sd": pstdev(
            number(row, "operational_return") for row in rows
        ),
        "true_return_mean": mean(rows, "true_return"),
        "collapse_rate": rate(rows, "collapse_entry"),
        "unsafe_fraction": mean(rows, "unsafe_fraction"),
        "mvp_fraction": mean(rows, "mvp_fraction"),
        "mvp_breach_rate": rate(rows, "mvp_breach"),
        "persistence_mean": mean(rows, "persistence"),
        "min_population_mean": mean(rows, "min_true_state"),
        "mean_population": mean(rows, "mean_true_state"),
        "final_population_mean": mean(rows, "final_true_state"),
        "economic_cost_mean": mean(rows, "economic_cost"),
        "action_entropy_mean": fmean(entropies),
        "all_episode_action_entropy_zero": all(abs(value) <= 1e-12 for value in entropies),
        "constant_policy_status": (
            "constant_within_every_episode"
            if all(abs(value) <= 1e-12 for value in entropies)
            else "nonconstant_observed"
        ),
        "episodes_sha256": sha256(path),
        "episodes_path": str(path),
    }
    for action in range(11):
        result[f"danger_action_{action}_fraction"] = mean(
            rows, f"danger_action_{action}_fraction"
        )
    result.update(pbvi_diagnostics(path, method))
    return result


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def markdown(rows: list[dict[str, Any]], receipt: dict[str, Any]) -> str:
    lines = [
        "# Option A matched general-RL and ecological safe comparison",
        "",
        "This table keeps all species, hidden families, noise levels, and methods separate.",
        "Amur tiger is labelled declining/recoverable, Crab-eating fox strong-growth/recoverable,",
        "and Egyptian vulture a demographic sink. No yield-mode claim is made.",
        "",
        "## Acceptance and outcome-access chronology",
        "",
        "- Original gates recorded `return_fields_opened=false`.",
        "- Existing PLUS and scoped MOOR outcomes were subsequently opened under authorization.",
        "- The two new method-specific Option A acceptances passed before this builder opened",
        "  the new episode-level outcomes.",
        f"- Outcome opening time: `{receipt['outcomes_opened_at']}`.",
        "",
    ]
    columns = (
        "environment",
        "sigma_obs",
        "method",
        "operational_return_mean",
        "collapse_rate",
        "unsafe_fraction",
        "mvp_breach_rate",
        "persistence_mean",
        "min_population_mean",
        "final_population_mean",
        "economic_cost_mean",
        "action_entropy_mean",
        "constant_policy_status",
    )
    for population, population_class in SPECIES:
        lines.extend(
            [
                f"## {population} — {population_class.replace('_', ' ')}",
                "",
                "| " + " | ".join(columns) + " |",
                "|" + "|".join("---" for _ in columns) + "|",
            ]
        )
        for row in rows:
            if row["population"] != population:
                continue
            values = []
            for column in columns:
                value = row[column]
                values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        [
            "## Claim limits",
            "",
            "- Ricker-only PLUS has a correct-form inductive bias only on Ricker truth and is",
            "  deliberately misspecified on Allee, theta-logistic, and regime-switching truth.",
            "- Corrected MOOR commits to one fitted Ricker model.",
            "- Constant-policy status and action entropy must accompany return comparisons;",
            "  exact ties can reflect policy degeneracy rather than algorithmic equivalence.",
            "- Families are not pooled as the only headline.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    plus_accept = require_acceptance(
        RUN_ROOT / "acceptance_plus_alignment.json", PLUS
    )
    moor_accept = require_acceptance(RUN_ROOT / "acceptance_moor_crab.json", MOOR)
    opened = datetime.now().astimezone().isoformat()
    output_rows = [
        aggregate_cell(population, population_class, family, sigma, method)
        for population, population_class in SPECIES
        for family in FAMILIES
        for sigma in SIGMAS
        for method in METHODS
    ]
    if len(output_rows) != 144:
        raise RuntimeError("final table is not exactly 144 method-cells")
    keys = {
        (
            row["population"],
            row["environment"],
            row["sigma_obs"],
            row["reward_mode"],
            row["method"],
        )
        for row in output_rows
    }
    if len(keys) != 144:
        raise RuntimeError("final method-cell identities are not unique")
    analysis = RUN_ROOT / "analysis"
    table = analysis / "OPTION_A_MATCHED_144_METHOD_CELLS.csv"
    receipt_path = analysis / "OPTION_A_MATCHED_144_RECEIPT.json"
    report = analysis / "OPTION_A_MATCHED_144_COMPARISON.md"
    write_csv_atomic(table, output_rows)
    receipt = {
        "receipt_schema": "option_a_matched_144_results_v1",
        "method_cells": 144,
        "species": [item[0] for item in SPECIES],
        "families": list(FAMILIES),
        "sigma_obs": list(SIGMAS),
        "reward_mode": "safe",
        "methods": list(METHODS),
        "plus_acceptance_sha256": sha256(
            RUN_ROOT / "acceptance_plus_alignment.json"
        ),
        "moor_acceptance_sha256": sha256(RUN_ROOT / "acceptance_moor_crab.json"),
        "plus_acceptance_decision": plus_accept["decision"],
        "moor_acceptance_decision": moor_accept["decision"],
        "returns_opened_at_acceptance_gate": False,
        "outcomes_opened_after_acceptance": True,
        "outcomes_opened_at": opened,
        "table_sha256": sha256(table),
    }
    temporary = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(receipt_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report.with_name(f".{report.name}.{os.getpid()}.tmp")
    temporary_report.write_text(markdown(output_rows, receipt), encoding="utf-8")
    temporary_report.replace(report)
    print(json.dumps({**receipt, "report": str(report)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
