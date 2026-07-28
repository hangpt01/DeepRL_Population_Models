#!/usr/bin/env python3
"""Aggregate figures for the motivation native-baseline experiment.

This script is intentionally read-only with respect to the run directory: it
uses completed ``summary.json``, ``episodes.csv``, and ``training_history.csv``
artifacts and writes only paper-facing figures under ``docs/benchmark``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

_mpl_config = Path(tempfile.gettempdir()) / f"ecology_mplconfig_{os.getuid()}"
_mpl_config.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


GENERAL = ("refplan", "bamcts", "ogsrl")
NATIVE = ("plus_native", "moor_native")
METHOD_ORDER = ("refplan", "bamcts", "ogsrl", "plus_native", "moor_native")
METHOD_COLORS = {
    "refplan": "#5f83cf",
    "bamcts": "#77a8c8",
    "ogsrl": "#8cc08b",
    "moor_native": "#c9853a",
    "plus_native": "#d35f5f",
}
RECOVERABLE = {
    "Amur tiger",
    "Asian elephant",
    "Crab-eating fox",
    "Iberian lynx",
    "Jaguar",
    "Puerto Rican parrot",
    "Spotted turtle",
}


def read_summaries(run: Path) -> list[dict[str, object]]:
    rows = []
    for path in (run / "outputs" / "main").glob("**/summary.json"):
        obj = json.loads(path.read_text())
        obj["_path"] = str(path)
        rows.append(obj)
    return rows


def read_episode_rows(run: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in (run / "outputs" / "main").glob("**/episodes.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def read_training_rows(run: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in (run / "outputs" / "main").glob("**/training_history.csv"):
        # .../reward_<mode>/<method>/<filter>/training_history.csv
        method = path.parts[-3]
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_method"] = method
                rows.append(row)
    return rows


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _cell_key(row: dict[str, object]) -> tuple[str, str, float, str]:
    return (
        str(row["population"]),
        str(row["environment"]),
        float(row["sigma_obs"]),
        str(row["reward_mode"]),
    )


def plot_all_methods_by_family(rows: list[dict[str, object]], out_dir: Path) -> None:
    family_methods: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["reward_mode"] != "safe" or row["population"] not in RECOVERABLE:
            continue
        method = str(row["model"])
        if method not in METHOD_ORDER:
            continue
        family_methods[str(row["environment"])][method].append(
            float(row["operational_return_mean"])
        )

    families = ["ricker", "allee", "theta", "regime"]
    x = np.arange(len(families))
    width = 0.15
    offsets = np.linspace(-2 * width, 2 * width, len(METHOD_ORDER))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for offset, method in zip(offsets, METHOD_ORDER):
        means = [
            np.mean(family_methods[family][method])
            if family_methods[family][method]
            else np.nan
            for family in families
        ]
        ax.bar(
            x + offset,
            means,
            width,
            label=method,
            color=METHOD_COLORS.get(method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("mean operational return")
    ax.set_title("All methods on recoverable safe cells")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir / "all_methods_returns_by_family")


def plot_headline_returns(rows: list[dict[str, object]], out_dir: Path) -> dict[str, tuple[float, float]]:
    by_cell: dict[tuple[str, str, float, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_cell[_cell_key(row)][str(row["model"])] = float(row["operational_return_mean"])

    family_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    moor_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    sigma_values: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for (population, family, sigma, reward_mode), vals in by_cell.items():
        if reward_mode != "safe" or population not in RECOVERABLE:
            continue
        if not all(m in vals for m in GENERAL) or not all(m in vals for m in NATIVE):
            continue
        general = max(vals[m] for m in GENERAL)
        native = max(vals[m] for m in NATIVE)
        family_values[family].append((general, native))
        moor_values[family].append((general, vals["moor_native"]))
        sigma_values[sigma].append((general, native))

    families = ["ricker", "allee", "theta", "regime"]
    x = np.arange(len(families))
    moor_general = [np.mean([g for g, _ in moor_values[f]]) for f in families]
    moor_native = [np.mean([m for _, m in moor_values[f]]) for f in families]
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - width / 2, moor_general, width, label="best general", color="#6b8fd6")
    ax.bar(x + width / 2, moor_native, width, label="moor_native (single Ricker)", color="#c9853a")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("mean operational return")
    ax.set_title("Misspecified single-Ricker native beats general methods")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for xi, g, n in zip(x, moor_general, moor_native):
        ax.text(xi, max(g, n) + 0.08, f"{g - n:+.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, out_dir / "moor_native_returns_by_family")

    general_means = [np.mean([g for g, _ in family_values[f]]) for f in families]
    native_means = [np.mean([n for _, n in family_values[f]]) for f in families]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - width / 2, general_means, width, label="best general", color="#6b8fd6")
    ax.bar(x + width / 2, native_means, width, label="best native", color="#d6864b")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("mean operational return")
    ax.set_title("Recoverable safe cells: native ecological baselines dominate")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for xi, g, n in zip(x, general_means, native_means):
        ax.text(xi, max(g, n) + 0.08, f"{g - n:+.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, out_dir / "headline_returns_by_family")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    sigmas = sorted(sigma_values)
    gaps = [np.mean([g - n for g, n in sigma_values[s]]) for s in sigmas]
    ax.plot(sigmas, gaps, marker="o", color="#4d4d4d", lw=2)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel(r"observation noise $\sigma_{\mathrm{obs}}$")
    ax.set_ylabel("best general - best native")
    ax.set_title("Gap remains negative at every noise level")
    ax.grid(alpha=0.3)
    save(fig, out_dir / "native_gap_by_sigma")

    return {
        f: (
            float(np.mean([g for g, _ in family_values[f]])),
            float(np.mean([n for _, n in family_values[f]])),
        )
        for f in families
    } | {
        f"{f}_moor": (
            float(np.mean([g for g, _ in moor_values[f]])),
            float(np.mean([m for _, m in moor_values[f]])),
        )
        for f in families
    }


def plot_budget_audit(audit_path: Path, out_dir: Path) -> dict[str, float] | None:
    if not audit_path.exists():
        return None
    obj = json.loads(audit_path.read_text())
    verdict = obj["verdict"]
    gap_as_run = float(verdict["mean_gap_as_run"])
    gap_tuned = float(verdict["mean_gap_tuned"])
    cells = int(verdict["cells_used"])
    as_run_wins = int(verdict["as_run_general_beats_native"])
    tuned_wins = int(verdict["tuned_general_beats_native"])
    closure = (gap_tuned - gap_as_run) / max(-gap_as_run, 1e-12)

    labels = ["as-run\nbest general", "best tuned\ngeneral"]
    values = [gap_as_run, gap_tuned]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    bars = ax.bar(labels, values, color=["#6b8fd6", "#8cc08b"], width=0.55)
    ax.axhline(0.0, color="black", lw=1)
    ax.set_ylabel("best general - best native")
    ax.set_title("Search-budget audit: tuning does not close the gap")
    ax.grid(axis="y", alpha=0.25)
    for bar, value, wins in zip(bars, values, [as_run_wins, tuned_wins]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 0.5,
            f"{value:+.2f}\n{wins}/{cells} wins",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    ax.text(
        0.5,
        0.92,
        f"gap closure: {closure:.0%}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    save(fig, out_dir / "search_budget_gap")
    return {
        "gap_as_run": gap_as_run,
        "gap_tuned": gap_tuned,
        "closure": float(closure),
        "cells": float(cells),
        "as_run_wins": float(as_run_wins),
        "tuned_wins": float(tuned_wins),
    }


def plot_training(rows: list[dict[str, object]], out_dir: Path) -> None:
    # Final dynamics log-MSE: final train/holdout values exist for the learned
    # model methods.  This is a generalization diagnostic, not a convergence
    # trace for RefPlan/BA-MCTS.
    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        method = str(row["_method"])
        metric = str(row["metric"])
        split = str(row["split"])
        if method in {"refplan", "bamcts", "ogsrl"} and metric == "dynamics_log_mse":
            vals[(method, split)].append(float(row["value"]))
    methods = ["refplan", "bamcts", "ogsrl"]
    x = np.arange(len(methods))
    width = 0.36
    train = [np.mean(vals[(m, "train")]) for m in methods]
    holdout = [np.mean(vals[(m, "holdout")]) for m in methods]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - width / 2, train, width, label="train", color="#78a4d4")
    ax.bar(x + width / 2, holdout, width, label="holdout", color="#e3a05f")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("dynamics log-MSE")
    ax.set_title("Learned dynamics fit: train and holdout diagnostics")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir / "training_dynamics_log_mse")

    # OGSRL has an actual actor-training trace.
    trace: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["_method"] != "ogsrl" or row["metric"] != "surrogate_loss":
            continue
        trace[(str(row["split"]), int(row["step"]))].append(float(row["value"]))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for split, color in (("train", "#4777aa"), ("holdout", "#cc7842")):
        steps = sorted(step for s, step in trace if s == split)
        means = [np.mean(trace[(split, step)]) for step in steps]
        ax.plot(steps, means, marker="o", ms=3, lw=1.8, color=color, label=split)
    ax.set_xlabel("actor update step")
    ax.set_ylabel("surrogate loss")
    ax.set_title("OGSRL actor loss is stable, but this does not rescue return")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    save(fig, out_dir / "ogsrl_surrogate_loss")


def action_labels(run: Path) -> list[str]:
    path = run / "code" / "real_ecology_data" / "actions.csv"
    labels = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            idx = int(str(row["action"]).lstrip("a"))
            label = str(row["name_original"]).replace(" / ", "/")
            labels.append((idx, f"a{idx}: {label}"))
    return [label for _, label in sorted(labels)]


def plot_danger_actions(run: Path, rows: list[dict[str, str]], out_dir: Path) -> dict[str, tuple[int, int]]:
    labels = action_labels(run)
    values: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["reward_mode"] != "safe":
            continue
        group = "recoverable" if row["population"] in RECOVERABLE else "sink"
        arr = np.asarray([float(row.get(f"danger_action_{a}_fraction", 0.0)) for a in range(11)])
        counts[group][0] += 1
        if arr.sum() > 0:
            counts[group][1] += 1
            # The evaluator already emits conditional fractions when danger_steps>0.
            # Renormalize defensively because rows with no danger use all zeros.
            values[(group, row["model"])].append(arr / arr.sum())

    panels = [
        ("recoverable", "Recoverable populations"),
        ("sink", "Demographic sinks"),
    ]
    matrices = {}
    for group, _title in panels:
        matrices[group] = np.vstack(
            [
                np.mean(values[(group, m)], axis=0)
                if values[(group, m)]
                else np.zeros(len(labels))
                for m in METHOD_ORDER
            ]
        )
    vmax = max(float(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 6.6), sharex=True)
    cmap = "mako" if "mako" in plt.colormaps() else "viridis"
    image = None
    for ax, (group, title) in zip(axes, panels):
        total, danger = counts[group]
        image = ax.imshow(matrices[group], aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_yticks(np.arange(len(METHOD_ORDER)))
        ax.set_yticklabels(METHOD_ORDER)
        ax.set_ylabel("method")
        ax.set_title(f"{title}: {danger:,}/{total:,} safe-mode episodes enter danger band")
    axes[-1].set_xticks(np.arange(len(labels)))
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[-1].set_xlabel("action")
    fig.suptitle("Action composition when previous state is in the danger band", y=0.995)
    cb = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("mean conditional fraction")
    save(fig, out_dir / "danger_zone_action_composition")
    return {group: (total, danger) for group, (total, danger) in counts.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="real_ecology_runs/motivation_native_20260711",
        help="motivation run directory",
    )
    parser.add_argument(
        "--out",
        default="docs/benchmark/figures/motivation_native",
        help="figure output directory",
    )
    args = parser.parse_args()
    run = Path(args.run)
    out = Path(args.out)
    summaries = read_summaries(run)
    episodes = read_episode_rows(run)
    training = read_training_rows(run)
    plot_all_methods_by_family(summaries, out)
    family = plot_headline_returns(summaries, out)
    audit = plot_budget_audit(
        Path("real_ecology_runs/general_audit_20260712/analysis/budget_verdict_clean.json"),
        out,
    )
    plot_training(training, out)
    danger = plot_danger_actions(run, episodes, out)
    print(f"wrote figures to {out}")
    for family_name, (general, native) in family.items():
        print(f"{family_name}: general={general:.3f} native={native:.3f} gap={general-native:.3f}")
    if audit is not None:
        print(
            "budget audit: "
            f"gap_as_run={audit['gap_as_run']:.3f} "
            f"gap_tuned={audit['gap_tuned']:.3f} "
            f"closure={audit['closure']:.1%} "
            f"wins={int(audit['tuned_wins'])}/{int(audit['cells'])}"
        )
    for group in ("recoverable", "sink"):
        total, in_danger = danger[group]
        print(f"{group}: danger episodes={in_danger}/{total} ({in_danger / total:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
