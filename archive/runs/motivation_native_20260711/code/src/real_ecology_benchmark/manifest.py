"""Manifest generation and result aggregation without pandas."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from . import realdata


def make_manifest(
    path: str | Path,
    methods: Iterable[str] | None = None,
    data_mode: str = "synthetic",
    populations: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
    sigmas: Iterable[float] | None = None,
    reward_modes: Iterable[str] | None = None,
    method_filters: dict[str, Iterable[str]] | None = None,
) -> int:
    """Write a synthetic or real manifest.

    ``data_mode="synthetic"`` preserves the original continuous-state synthetic manifest
    contract.  ``data_mode="real"`` writes the reward-mode x population real
    grid used by the real-ecology setting.
    """

    methods = list(methods or ["mopo", "refplan", "bamcts", "plus", "moor", "delphic", "ogsrl"])
    if data_mode == "synthetic":
        rows = []
        index = 0
        for env in ("allee", "theta", "regime"):
            for actions in (5, 10):
                for sigma in (0.0, 0.1, 0.2, 0.4):
                    for method in methods:
                        filters = (
                            tuple(method_filters.get(method, ()))
                            if method_filters is not None else ("learned", "raw")
                        )
                        for filter_mode in filters:
                            rows.append(
                                {
                                    "index": index,
                                    "environment": env,
                                    "num_actions": actions,
                                    "sigma_obs": sigma,
                                    "method": method,
                                    "filter": filter_mode,
                                }
                            )
                            index += 1
                    for method in ("plus", "moor"):
                        if method not in methods or method_filters is not None:
                            continue
                        rows.append(
                            {
                                "index": index,
                                "environment": env,
                                "num_actions": actions,
                                "sigma_obs": sigma,
                                "method": method,
                                "filter": "ricker",
                            }
                        )
                        index += 1
    elif data_mode == "real":
        rows = []
        index = 0
        populations = list(populations or realdata.population_names())
        families = list(families or ["ricker", "allee", "theta", "regime"])
        sigmas = list(sigmas or (0.0, 0.1, 0.2, 0.4))
        reward_modes = list(reward_modes or ("safe", "yield"))
        recoverable = set(realdata.recoverable_population_names())
        for reward_mode in reward_modes:
            for population in populations:
                is_recoverable = population in recoverable
                for env in families:
                    for sigma in sigmas:
                        for method in methods:
                            filters = (
                                tuple(method_filters.get(method, ()))
                                if method_filters is not None else ("learned", "raw")
                            )
                            for filter_mode in filters:
                                rows.append(
                                    {
                                        "index": index,
                                        "reward_mode": reward_mode,
                                        "population": population,
                                        "recoverable": is_recoverable,
                                        "environment": env,
                                        "num_actions": realdata.NUM_REAL_ACTIONS,
                                        "sigma_obs": sigma,
                                        "method": method,
                                        "filter": filter_mode,
                                    }
                                )
                                index += 1
                        # Baseline-fidelity ablations on every cell.
                        for method in ("plus", "moor"):
                            if method not in methods or method_filters is not None:
                                continue
                            rows.append(
                                {
                                    "index": index,
                                    "reward_mode": reward_mode,
                                    "population": population,
                                    "recoverable": is_recoverable,
                                    "environment": env,
                                    "num_actions": realdata.NUM_REAL_ACTIONS,
                                    "sigma_obs": sigma,
                                    "method": method,
                                    "filter": "ricker",
                                }
                            )
                            index += 1
    else:
        raise ValueError("data_mode must be 'synthetic' or 'real'")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def make_motivation_native_manifest(path: str | Path) -> int:
    return make_manifest(
        path,
        methods=("refplan", "bamcts", "ogsrl", "plus_native", "moor_native"),
        data_mode="real",
        method_filters={
            "refplan": ("learned",),
            "bamcts": ("learned",),
            "ogsrl": ("learned",),
            "plus_native": ("native_discrete",),
            "moor_native": ("native_discrete",),
        },
    )


def aggregate_summaries(
    root: str | Path,
    output: str | Path,
    challenger_filter: str | None = None,
    baselines: Iterable[tuple[str, str]] | None = None,
) -> dict[str, object]:
    summaries = []
    for path in Path(root).rglob("summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["path"] = str(path)
        summaries.append(row)
    result: dict[str, object] = {"runs": len(summaries), "summaries": summaries}
    by_model: dict[tuple[str, str, str], list[float]] = {}
    for row in summaries:
        if "operational_return_mean" in row:
            key = (
                str(row.get("compute_backend_effective", "unknown")),
                str(row.get("reward_mode", "n/a")),
                str(row["model"]),
            )
            by_model.setdefault(key, []).append(row["operational_return_mean"])
    result["model_return_mean"] = {
        backend: {
            reward_mode: {
                model: float(np.mean(by_model[(backend, reward_mode, model)]))
                for b, mode, model in sorted(by_model)
                if b == backend and mode == reward_mode
            }
            for reward_mode in sorted({mode for b, mode, _model in by_model if b == backend})
        }
        for backend in sorted({backend for backend, _mode, _model in by_model})
    }
    efficiency_metrics = (
        "manifest_row_seconds",
        "row_seconds",
        "dataset_seconds",
        "filter_factory_seconds",
        "belief_cache_seconds",
        "policy_init_seconds",
        "fit_seconds",
        "evaluation_seconds",
        "summary_save_seconds",
        "planner_seconds_mean",
        "filter_seconds_mean",
        "manifest_row_peak_rss_mb",
        "peak_rss_mb",
    )
    efficiency: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}
    for row in summaries:
        key = (
            str(row.get("compute_backend_effective", "unknown")),
            str(row.get("reward_mode", "n/a")),
            str(row.get("model", "unknown")),
            str(row.get("filter", "unknown")),
        )
        bucket = efficiency.setdefault(key, {metric: [] for metric in efficiency_metrics})
        for metric in efficiency_metrics:
            if metric in row:
                bucket[metric].append(float(row[metric]))
    result["efficiency_mean"] = [
        {
            "compute_backend_effective": backend,
            "reward_mode": reward_mode,
            "model": model,
            "filter": filter_name,
            "runs": max((len(values) for values in metrics.values()), default=0),
            **{
                metric: float(np.mean(values))
                for metric, values in metrics.items()
                if values
            },
        }
        for (backend, reward_mode, model, filter_name), metrics in sorted(efficiency.items())
    ]
    seed_values: dict[tuple, list[float]] = {}
    for path in Path(root).rglob("episodes.csv"):
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row.get("compute_backend_effective", "unknown"),
                    row.get("reward_mode", "n/a"), row.get("population", "n/a"),
                    row["environment"], int(row["num_actions"]), float(row["sigma_obs"]),
                    row.get("filter", "unknown"), int(row["block_seed"]), row["model"],
                )
                seed_values.setdefault(key, []).append(float(row["operational_return"]))
    seed_means = {key: float(np.mean(values)) for key, values in seed_values.items()}
    recoverable = set(realdata.recoverable_population_names())
    grouped_wins: dict[tuple, list[bool]] = {}
    cell_rows = []

    if challenger_filter is None and baselines is None:
        # Legacy same-filter path for the adapted-baseline reports.
        scenarios = sorted({key[:7] for key in seed_means})
        cell_differences: dict[tuple, list[float]] = {}
        for scenario in scenarios:
            seeds = sorted({key[7] for key in seed_means if key[:7] == scenario})
            methods = sorted({key[8] for key in seed_means if key[:7] == scenario})
            for method in methods:
                if method in {"plus", "moor"}:
                    continue
                differences = []
                for seed in seeds:
                    method_key = (*scenario, seed, method)
                    plus_key = (*scenario, seed, "plus")
                    moor_key = (*scenario, seed, "moor")
                    if (
                        method_key not in seed_means
                        or plus_key not in seed_means
                        or moor_key not in seed_means
                    ):
                        continue
                    baseline = max(seed_means[plus_key], seed_means[moor_key])
                    differences.append(seed_means[method_key] - baseline)
                if differences:
                    cell_differences[(*scenario, method)] = differences
        for key, differences in cell_differences.items():
            backend, reward_mode, population, env, actions, sigma, filter_name, method = key
            mean_difference = float(np.mean(differences))
            win = mean_difference > 0.0
            is_recoverable = population in recoverable
            if is_recoverable:
                grouped_wins.setdefault(
                    (backend, reward_mode, method, sigma, filter_name), []
                ).append(win)
            cell_rows.append(
                {
                    "compute_backend_effective": backend,
                    "reward_mode": reward_mode,
                    "population": population,
                    "recoverable": is_recoverable,
                    "environment": env,
                    "num_actions": actions,
                    "sigma_obs": sigma,
                    "filter": filter_name,
                    "method": method,
                    "paired_mean_difference": mean_difference,
                    "paired_seed_std": float(np.std(differences, ddof=1))
                    if len(differences) > 1 else 0.0,
                    "beats_both": win,
                    "paired_seeds": len(differences),
                }
            )
    else:
        # Motivation-native path: compare challengers and baselines across filters.
        challenger_filter = challenger_filter or "learned"
        baseline_specs = tuple(
            baselines or (("plus_native", "native_discrete"), ("moor_native", "native_discrete"))
        )
        baseline_names = {method for method, _filter in baseline_specs}
        baseline_label = ",".join(f"{method}:{filter_name}" for method, filter_name in baseline_specs)
        scenarios = sorted({key[:6] for key in seed_means})
        for scenario in scenarios:
            seeds = sorted({key[7] for key in seed_means if key[:6] == scenario})
            methods = sorted({
                key[8]
                for key in seed_means
                if key[:6] == scenario and key[6] == challenger_filter
            })
            for method in methods:
                if method in baseline_names:
                    continue
                differences = []
                for seed in seeds:
                    method_key = (*scenario, challenger_filter, seed, method)
                    baseline_keys = [
                        (*scenario, filter_name, seed, baseline_method)
                        for baseline_method, filter_name in baseline_specs
                    ]
                    if method_key not in seed_means or any(
                        key not in seed_means for key in baseline_keys
                    ):
                        continue
                    baseline = max(seed_means[key] for key in baseline_keys)
                    differences.append(seed_means[method_key] - baseline)
                if not differences:
                    continue
                backend, reward_mode, population, env, actions, sigma = scenario
                mean_difference = float(np.mean(differences))
                win = mean_difference > 0.0
                is_recoverable = population in recoverable
                if is_recoverable:
                    grouped_wins.setdefault(
                        (backend, reward_mode, method, sigma, challenger_filter, baseline_label),
                        [],
                    ).append(win)
                cell_rows.append(
                    {
                        "compute_backend_effective": backend,
                        "reward_mode": reward_mode,
                        "population": population,
                        "recoverable": is_recoverable,
                        "environment": env,
                        "num_actions": actions,
                        "sigma_obs": sigma,
                        "filter": challenger_filter,
                        "method": method,
                        "baseline_specs": baseline_label,
                        "paired_mean_difference": mean_difference,
                        "paired_seed_std": float(np.std(differences, ddof=1))
                        if len(differences) > 1 else 0.0,
                        "beats_both": win,
                        "paired_seeds": len(differences),
                    }
                )

    result["beats_both_cells"] = cell_rows
    rate_rows = []
    rng = np.random.default_rng(116)
    for key, wins in sorted(grouped_wins.items()):
        values = np.asarray(wins, dtype=float)
        bootstrap = np.asarray([
            np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(2000)
        ])
        row = {
            "compute_backend_effective": key[0],
            "reward_mode": key[1],
            "method": key[2],
            "sigma_obs": key[3],
            "filter": key[4],
            "rate": float(np.mean(values)),
            "cells": len(values),
            "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        }
        if len(key) > 5:
            row["baseline_specs"] = key[5]
        rate_rows.append(row)
    result["beats_both_rate"] = rate_rows
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result
