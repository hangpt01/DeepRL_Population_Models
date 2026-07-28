#!/usr/bin/env python
"""Generate paper-facing figures from existing P_safe artifacts.

This script does not run experiments. It only reads `all_metrics.csv`,
`episodes.csv`, and `training_history.csv` files already present in the run
directory.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUN = Path(__file__).resolve().parents[1]
ANALYSIS = RUN / "analysis"
FIGS = ANALYSIS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

POLICY_FILTER = {
    "ogsrl": "learned",
    "mopo": "learned",
    "bamcts": "learned",
    "refplan": "learned",
    "delphic": "learned",
    "plus": "ricker",
    "moor": "ricker",
}
GENERAL_METHODS = ["ogsrl", "bamcts", "refplan", "mopo", "delphic"]
BASELINES = ["plus", "moor"]
ALL_METHODS = ["ogsrl", "bamcts", "refplan", "mopo", "delphic", "plus", "moor"]
SINKS = {"Egyptian vulture", "Bottlenose dolphin"}


def read_metrics() -> list[dict]:
    rows: list[dict] = []
    with (ANALYSIS / "all_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["penalty"] = int(row["penalty"])
            row["sigma_obs"] = float(row["sigma_obs"])
            for key in (
                "unsafe_fraction_mean",
                "mvp_fraction_mean",
                "collapse_entry_mean",
                "true_return_mean",
                "final_true_state_mean",
                "min_true_state_mean",
                "economic_cost_mean",
            ):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def policy_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["filter"] == POLICY_FILTER.get(r["method"])]


def save_psafe_decision_returns(rows: list[dict]) -> None:
    selected = policy_rows(rows)
    pops = ["Amur tiger", "Puerto Rican parrot", "Egyptian vulture"]
    penalties = [2, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for pop in pops:
        ys = []
        for p in penalties:
            vals = [
                r["true_return_mean"]
                for r in selected
                if r["reward_mode"] == "safe" and r["penalty"] == p and r["population"] == pop
            ]
            ys.append(mean(vals))
        ax.plot(penalties, ys, marker="o", label=pop)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(5, color="#555555", linewidth=0.8, linestyle=":")
    ax.set_xlabel("collapse_penalty")
    ax.set_ylabel("safe true_return_mean")
    ax.set_title("P_safe decision populations")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_psafe_decision_returns.png", dpi=180)
    plt.close(fig)


def save_p5_return_by_method(rows: list[dict]) -> None:
    selected = [
        r
        for r in policy_rows(rows)
        if r["penalty"] == 5 and r["reward_mode"] == "safe"
    ]
    scopes = [
        ("All", lambda r: True),
        ("Recoverable", lambda r: r["population"] not in SINKS),
        ("Sinks", lambda r: r["population"] in SINKS),
    ]
    x = np.arange(len(ALL_METHODS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for j, (label, predicate) in enumerate(scopes):
        vals = []
        for method in ALL_METHODS:
            vals.append(mean([r["true_return_mean"] for r in selected if r["method"] == method and predicate(r)]))
        ax.bar(x + (j - 1) * width, vals, width=width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() if m == "ogsrl" else m for m in ALL_METHODS], rotation=25, ha="right")
    ax.set_ylabel("true_return_mean")
    ax.set_title("P5 safe return by method and population scope")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_p5_return_by_method.png", dpi=180)
    plt.close(fig)


def save_learned_vs_raw_sigma(rows: list[dict]) -> None:
    sigmas = [0.0, 0.1, 0.2, 0.4]
    learned = []
    raw = []
    for sigma in sigmas:
        learned.append(mean([
            r["true_return_mean"]
            for r in rows
            if r["penalty"] == 5
            and r["reward_mode"] == "safe"
            and r["filter"] == "learned"
            and r["sigma_obs"] == sigma
        ]))
        raw.append(mean([
            r["true_return_mean"]
            for r in rows
            if r["penalty"] == 5
            and r["reward_mode"] == "safe"
            and r["filter"] == "raw"
            and r["sigma_obs"] == sigma
        ]))
    delta = np.asarray(learned) - np.asarray(raw)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar([str(s) for s in sigmas], delta, color="#4C78A8")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("observation noise sigma")
    ax.set_ylabel("learned - raw true_return_mean")
    ax.set_title("P5 learned-vs-raw control ablation")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_learned_vs_raw_sigma.png", dpi=180)
    plt.close(fig)


def read_policy_episodes() -> list[dict]:
    rows: list[dict] = []
    root = RUN / "outputs" / "p5" / "evaluation"
    for path in root.rglob("episodes.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method = row["model"]
                if row.get("reward_mode") != "safe":
                    continue
                if row.get("filter") != POLICY_FILTER.get(method):
                    continue
                for key in ["action_entropy", *[f"danger_action_{i}_fraction" for i in range(11)]]:
                    row[key] = float(row.get(key) or 0.0)
                rows.append(row)
    return rows


def save_action_profile(episodes: list[dict]) -> None:
    matrix = []
    for method in ALL_METHODS:
        method_rows = [r for r in episodes if r["model"] == method]
        matrix.append([
            mean([r[f"danger_action_{a}_fraction"] for r in method_rows])
            for a in range(11)
        ])
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(ALL_METHODS)))
    ax.set_yticklabels([m.upper() if m == "ogsrl" else m for m in ALL_METHODS])
    ax.set_xticks(np.arange(11))
    ax.set_xticklabels([f"a{i}" for i in range(11)])
    ax.set_xlabel("action id")
    ax.set_title("P5 aggregate danger-zone action fractions")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("mean fraction")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_p5_danger_action_profile.png", dpi=180)
    plt.close(fig)


def save_holdout_rmse() -> None:
    by_method: dict[str, list[float]] = {m: [] for m in ALL_METHODS}
    root = RUN / "outputs" / "p5" / "evaluation"
    for path in root.rglob("training_history.csv"):
        parts = path.parts
        try:
            method = parts[-3]
            filter_name = parts[-2]
        except IndexError:
            continue
        if filter_name != POLICY_FILTER.get(method):
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") == "holdout" and row.get("metric") == "dynamics_rmse":
                    by_method.setdefault(method, []).append(float(row["value"]))
    methods = [m for m in ALL_METHODS if by_method.get(m)]
    vals = [mean(by_method[m]) for m in methods]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(np.arange(len(methods)), vals, color="#F58518")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([m.upper() if m == "ogsrl" else m for m in methods], rotation=25, ha="right")
    ax.set_ylabel("holdout dynamics RMSE")
    ax.set_title("P5 final validation error by method")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_p5_holdout_rmse_by_method.png", dpi=180)
    plt.close(fig)


def main() -> None:
    rows = read_metrics()
    episodes = read_policy_episodes()
    save_psafe_decision_returns(rows)
    save_p5_return_by_method(rows)
    save_learned_vs_raw_sigma(rows)
    save_action_profile(episodes)
    save_holdout_rmse()
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
