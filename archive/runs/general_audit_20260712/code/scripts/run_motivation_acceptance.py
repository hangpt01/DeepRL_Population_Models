#!/usr/bin/env python3
"""Acceptance battery for the motivation-experiment native baselines.

Runs the eight checks in SERVER_HANDOFF_motivation_experiment.md section 7 over a
representative slice of the run grid (not the full 1440-cell sweep) and reports
pass/fail per check with the evidence each one turned on.

Usage:
    PYTHONPATH=src python scripts/run_motivation_acceptance.py --output-root outputs/acceptance
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark import realdata
from real_ecology_benchmark.config import load_config, real_environment_like
from real_ecology_benchmark.dataset import load_public
from real_ecology_benchmark.methods import METHODS, NATIVE_METHODS
from real_ecology_benchmark.pipeline import ensure_dataset, run_method
from real_ecology_benchmark.reward import effective_collapse_penalty
from real_ecology_benchmark.types import PublicTransition

# A slice chosen to exercise every axis the acceptance tests care about:
#   Amur tiger      recoverable; the Ricker "assumption is correct" cell
#   Iberian lynx    recoverable; the four mechanistic forms disagree most (245/385 states)
#   Egyptian vulture demographic sink; NO action has r_setpoint > 0
#   Spotted turtle  the one population where all four forms agree everywhere
POPULATIONS = ("Amur tiger", "Iberian lynx", "Egyptian vulture", "Spotted turtle")
FAMILIES = ("ricker", "allee", "theta", "regime")
SIGMAS = (0.0, 0.4)
REWARD_MODES = ("safe", "yield")

GENERAL = ("refplan", "learned")
NATIVES = (("moor_native", "native_discrete"), ("plus_native", "native_discrete"))
ADAPTED = (("moor", "ricker"), ("plus", "ricker"))

COARSE_BINS = 31
FINE_BINS = 91


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sigma_slug(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def cell_config(base: str, population: str, family: str, sigma: float, reward_mode: str,
                output_root: Path, bins: int | None = None):
    cfg = load_config(base)
    cfg.environment = real_environment_like(cfg.environment, population, family)
    cfg.environment = replace(
        cfg.environment, observation_noise_sigma=sigma, reward_mode=reward_mode
    )
    if bins is not None:
        cfg.model = replace(cfg.model, native_state_bins=bins)
    cfg.validate()
    cell = Path(f"reward_{reward_mode}") / slug(population) / family / f"sigma_{sigma_slug(sigma)}"
    # One dataset per (population, family, sigma, reward_mode) cell, shared by every
    # method in that cell -- this is what the "same offline dataset" check verifies.
    cfg.dataset.output = str(output_root / "datasets" / cell / "public.npz")
    cfg.dataset.private_output = str(output_root / "private" / cell / "truth.npz")
    # Resolution rows re-run the same (method, filter) on the same cell at a different
    # grid, so they need their own evaluation directory or they clobber the main grid.
    evaluation = output_root / "evaluation" / cell
    if bins is not None:
        evaluation = output_root / "resolution" / f"bins_{bins}" / cell
    cfg.evaluation.output_dir = str(evaluation)
    return cfg, cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "motivation_native.yaml"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "acceptance"))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[str] = []
    started = time.perf_counter()

    def execute(method, filter_mode, population, family, sigma, reward_mode, bins=None, tag=""):
        cfg, cell = cell_config(
            args.config, population, family, sigma, reward_mode, output_root, bins
        )
        label = f"{method}/{filter_mode}{tag} {population}/{family} sigma={sigma} {reward_mode}"
        try:
            summary = run_method(method, cfg, filter_mode, regenerate=False)
        except Exception as exc:  # a failed row is itself an acceptance failure (A8)
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"  FAIL {label}: {exc}", flush=True)
            return None
        record = {
            "method": method, "filter": filter_mode, "population": population,
            "family": family, "sigma": sigma, "reward_mode": reward_mode,
            "bins": bins, "tag": tag, "cell": str(cell),
            "recoverable": population in set(realdata.recoverable_population_names()),
            **{k: summary[k] for k in (
                "operational_return_mean", "true_return_mean", "collapse_entry_mean",
                "unsafe_fraction_mean", "min_true_state_mean", "final_true_state_mean",
                "persistence_mean", "economic_cost_mean", "fallback_count_mean",
            )},
            "dataset_sha256": summary.get("dataset_sha256"),
            "summary_keys": sorted(summary),
            "output_dir": summary.get("output_dir"),
        }
        rows.append(record)
        print(f"  ok   {label}: return={record['operational_return_mean']:.3f} "
              f"fallback={record['fallback_count_mean']:.1f}", flush=True)
        return record

    # ---- main grid: general + both natives, every family/sigma/reward_mode -------
    print("[1/4] main grid (general + native, full eval protocol)", flush=True)
    for reward_mode in REWARD_MODES:
        for population in POPULATIONS:
            for family in FAMILIES:
                for sigma in SIGMAS:
                    for method, filter_mode in (GENERAL, *NATIVES):
                        execute(method, filter_mode, population, family, sigma, reward_mode)

    # ---- A2: not-crippled -- native vs adapted on the Ricker cells --------------
    print("[2/4] not-crippled check: adapted PLUS/MOOR on Ricker cells", flush=True)
    for reward_mode in REWARD_MODES:
        for population in POPULATIONS:
            for sigma in SIGMAS:
                for method, filter_mode in ADAPTED:
                    execute(method, filter_mode, population, "ricker", sigma, reward_mode)

    # ---- A6: resolution sensitivity -- natives at coarse and fine grids ---------
    print("[3/4] resolution sensitivity (coarse vs fine native grid)", flush=True)
    for bins in (COARSE_BINS, FINE_BINS):
        for population in POPULATIONS:
            for family in FAMILIES:
                for method, filter_mode in NATIVES:
                    execute(method, filter_mode, population, family, 0.4, "safe",
                            bins=bins, tag=f"@{bins}")

    # ---- A7: reward-leakage guard (runtime proof) -------------------------------
    print("[4/4] reward-leakage guard", flush=True)
    leakage: dict[str, object] = {
        "public_transition_has_no_reward_field": "reward" not in PublicTransition.__annotations__,
    }
    cfg, _ = cell_config(args.config, "Iberian lynx", "allee", 0.2, "safe", output_root)
    dataset = ensure_dataset(cfg, regenerate=False)
    from real_ecology_benchmark.beliefs import DiscreteGridFilter
    from real_ecology_benchmark.native_solver import NativeSolver

    solver = NativeSolver.build(cfg.environment, state_bins=cfg.model.native_state_bins)
    belief = DiscreteGridFilter(cfg.environment, solver).reset(float(cfg.environment.N0), 0)
    for name in ("moor_native", "plus_native"):
        honest = METHODS[name](cfg.environment, cfg.model, cfg.planner, seed=3)
        honest.fit(dataset, None)
        honest.reset(0)
        baseline_actions = [honest.act(belief, float(cfg.environment.N0)) for _ in range(3)]
        # Corrupt the logged rewards only.  A method that reads them will change behaviour.
        poisoned_data = load_public(cfg.dataset.output)
        poisoned_data.rewards = poisoned_data.rewards * -137.0 + 991.0
        poisoned = METHODS[name](cfg.environment, cfg.model, cfg.planner, seed=3)
        poisoned.fit(poisoned_data, None)
        poisoned.reset(0)
        poisoned_actions = [poisoned.act(belief, float(cfg.environment.N0)) for _ in range(3)]
        leakage[f"{name}_ignores_logged_rewards"] = baseline_actions == poisoned_actions

    # ---- evaluate the eight checks ---------------------------------------------
    def get(method, population, family, sigma, reward_mode, tag=""):
        for row in rows:
            if (row["method"] == method and row["population"] == population
                    and row["family"] == family and row["sigma"] == sigma
                    and row["reward_mode"] == reward_mode and row["tag"] == tag):
                return row
        return None

    checks: dict[str, dict] = {}

    # A1 gate
    checks["A1_gate_resolved"] = {
        "pass": set(NATIVE_METHODS) == {"moor_native", "plus_native"}
        and all(m in METHODS for m in NATIVE_METHODS),
        "evidence": "build-required; native solver added (see SERVER_CONTEXT doc section 1)",
    }

    # A2 not crippled on Ricker cells
    a2_detail, a2_pass = [], True
    for reward_mode in REWARD_MODES:
        for population in POPULATIONS:
            for sigma in SIGMAS:
                for native, adapted in (("moor_native", "moor"), ("plus_native", "plus")):
                    n = get(native, population, "ricker", sigma, reward_mode)
                    a = get(adapted, population, "ricker", sigma, reward_mode)
                    if n is None or a is None:
                        continue
                    ok = n["operational_return_mean"] >= a["operational_return_mean"]
                    a2_pass &= ok
                    a2_detail.append({
                        "population": population, "sigma": sigma, "reward_mode": reward_mode,
                        "pair": f"{native} vs {adapted}",
                        "native": round(n["operational_return_mean"], 3),
                        "adapted": round(a["operational_return_mean"], 3),
                        "pass": ok,
                    })
    checks["A2_native_not_crippled_on_ricker"] = {"pass": a2_pass, "detail": a2_detail}

    # A3 same dataset hash across methods within a cell
    by_cell: dict[str, set] = {}
    for row in rows:
        if row["tag"]:
            continue
        by_cell.setdefault(row["cell"], set()).add(row["dataset_sha256"])
    a3_pass = all(len(h) == 1 and None not in h for h in by_cell.values())
    checks["A3_same_offline_dataset_hash"] = {
        "pass": a3_pass,
        "cells": len(by_cell),
        "detail": [{"cell": c, "hashes": sorted(h)} for c, h in by_cell.items() if len(h) != 1],
    }

    # A4 both reward modes, separate agents, yield => P=0
    a4_detail = []
    for reward_mode in REWARD_MODES:
        cfg_m, _ = cell_config(args.config, "Amur tiger", "ricker", 0.0, reward_mode, output_root)
        a4_detail.append({
            "reward_mode": reward_mode,
            "effective_collapse_penalty": effective_collapse_penalty(cfg_m.environment),
            "rows": sum(1 for r in rows if r["reward_mode"] == reward_mode and not r["tag"]),
        })
    safe_dirs = {r["output_dir"] for r in rows if r["reward_mode"] == "safe"}
    yield_dirs = {r["output_dir"] for r in rows if r["reward_mode"] == "yield"}
    checks["A4_both_reward_modes_separate_agents"] = {
        "pass": (
            all(d["rows"] > 0 for d in a4_detail)
            and dict((d["reward_mode"], d["effective_collapse_penalty"]) for d in a4_detail)
            == {"safe": 5.0, "yield": 0.0}
            and not (safe_dirs & yield_dirs)
        ),
        "detail": a4_detail,
        "output_dirs_disjoint": not (safe_dirs & yield_dirs),
    }

    # A5 schema: native rows carry the same metric schema as the general rows
    general_keys = {tuple(r["summary_keys"]) for r in rows if r["method"] == "refplan"}
    native_keys = {tuple(r["summary_keys"]) for r in rows if r["method"] in NATIVE_METHODS}
    metric_suffixes = ("_mean", "_std")
    general_metrics = {k for ks in general_keys for k in ks if k.endswith(metric_suffixes)}
    native_metrics = {k for ks in native_keys for k in ks if k.endswith(metric_suffixes)}
    episode_fields: set[str] = set()
    for row in rows:
        if row["method"] in NATIVE_METHODS and row["output_dir"]:
            path = Path(row["output_dir"]) / "episodes.csv"
            if path.exists():
                with path.open() as handle:
                    episode_fields = set(next(csv.reader(handle)))
                break
    required = {"sigma_obs", "environment", "population", "reward_mode", "filter",
                "collapse_entry", "unsafe_fraction", "min_true_state", "final_true_state",
                "persistence", "economic_cost", "true_return", "operational_return"}
    checks["A5_metrics_schema_matches_general"] = {
        "pass": general_metrics == native_metrics and required <= episode_fields,
        "missing_from_native": sorted(general_metrics - native_metrics),
        "extra_in_native": sorted(native_metrics - general_metrics),
        "missing_episode_fields": sorted(required - episode_fields),
        "recoverable_split_available": bool(realdata.recoverable_population_names()),
    }

    # A6 resolution: does coarse-vs-fine flip the general-vs-native headline?
    a6_detail, a6_pass = [], True
    for population in POPULATIONS:
        for family in FAMILIES:
            general = get("refplan", population, family, 0.4, "safe")
            if general is None:
                continue
            for native, _f in NATIVES:
                coarse = get(native, population, family, 0.4, "safe", tag=f"@{COARSE_BINS}")
                fine = get(native, population, family, 0.4, "safe", tag=f"@{FINE_BINS}")
                if coarse is None or fine is None:
                    continue
                g = general["operational_return_mean"]
                d_coarse = g - coarse["operational_return_mean"]
                d_fine = g - fine["operational_return_mean"]
                # A flip is a REVERSAL (win at one grid, loss at the other).  An exact
                # tie is not a reversal: at sigma=0 the environment is deterministic, so
                # a general method and a native that find the same policy return exactly
                # the same value.  np.sign()==0 would report that tie as a phantom flip.
                ok = not (max(d_coarse, d_fine) > 0.0 > min(d_coarse, d_fine))
                a6_pass &= ok
                a6_detail.append({
                    "population": population, "family": family, "native": native,
                    "general": round(g, 3),
                    f"native@{COARSE_BINS}": round(coarse["operational_return_mean"], 3),
                    f"native@{FINE_BINS}": round(fine["operational_return_mean"], 3),
                    "headline_sign_stable": ok,
                })
    checks["A6_resolution_does_not_flip_headline"] = {"pass": a6_pass, "detail": a6_detail}

    # A7 reward leakage
    checks["A7_reward_leakage_guard"] = {
        "pass": all(bool(v) for v in leakage.values()),
        "detail": leakage,
    }

    # A8 zero failures, zero silent fallbacks
    fallbacks = [
        {"method": r["method"], "cell": r["cell"], "fallback": r["fallback_count_mean"]}
        for r in rows if r["fallback_count_mean"] > 0
    ]
    checks["A8_zero_failures_zero_fallbacks"] = {
        "pass": not failures and not fallbacks,
        "rows_run": len(rows),
        "failures": failures,
        "rows_with_fallbacks": fallbacks,
    }

    elapsed = time.perf_counter() - started
    report = {
        "rows_run": len(rows),
        "elapsed_seconds": round(elapsed, 1),
        "populations": list(POPULATIONS),
        "families": list(FAMILIES),
        "sigmas": list(SIGMAS),
        "reward_modes": list(REWARD_MODES),
        "checks": checks,
        "rows": rows,
    }
    target = output_root / "acceptance_report.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)

    print()
    print("=" * 72)
    print(f"ACCEPTANCE BATTERY  ({len(rows)} rows, {elapsed / 60:.1f} min)")
    print("=" * 72)
    for name, result in checks.items():
        print(f"  [{'PASS' if result['pass'] else 'FAIL'}]  {name}")
    print()
    print(f"report: {target}")
    if not all(r["pass"] for r in checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
