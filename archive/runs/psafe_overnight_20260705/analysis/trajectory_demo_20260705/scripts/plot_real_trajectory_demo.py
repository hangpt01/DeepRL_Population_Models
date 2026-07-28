#!/usr/bin/env python3
"""Plot representative real-ecology trajectory traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
import numpy as np  # noqa: E402


DEMO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = DEMO_ROOT.parents[1]
DATA_DIR = RUN_ROOT / "code" / "real_ecology_data"

METHOD_ORDER = ["ogsrl", "mopo", "plus", "moor"]
COLORS = {
    "ogsrl": "#1f77b4",
    "mopo": "#2ca02c",
    "plus": "#d62728",
    "moor": "#9467bd",
}
COST_CMAP = LinearSegmentedColormap.from_list(
    "action_cost_blue_to_red",
    ["#08306b", "#2171b5", "#6baed6", "#fdae6b", "#ef3b2c", "#7f0000"],
)


def load_action_costs(num_actions: int) -> list[float]:
    path = DATA_DIR / "actions.csv"
    costs = [float(i) for i in range(num_actions)]
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            action = row.get("action", "")
            if action.startswith("a") and action[1:].isdigit():
                idx = int(action[1:])
                if 0 <= idx < num_actions:
                    costs[idx] = float(row["cost_step"])
    return costs


def cost_colors(costs: list[float]):
    norm = Normalize(vmin=min(costs), vmax=max(costs))
    return [COST_CMAP(norm(cost)) for cost in costs]


def load_traces(root: Path):
    rows = []
    for path in sorted(root.glob("*/*/*/*/*/trajectory.npz")):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        rows.append((path, meta, data))
    return rows


def median_with_iqr(values: np.ndarray):
    return (
        np.nanmedian(values, axis=0),
        np.nanpercentile(values, 25, axis=0),
        np.nanpercentile(values, 75, axis=0),
    )


def plot_cell(cell_rows, out_dir: Path):
    first_meta = cell_rows[0][1]
    title = (
        f"{first_meta['population']} / {first_meta['environment']} / "
        f"sigma={first_meta['sigma_obs']} / P_safe={first_meta['collapse_penalty']}"
    )
    horizon = int(first_meta["horizon"])
    t = np.arange(horizon)
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    method_rows = {meta["method"]: (path, meta, data) for path, meta, data in cell_rows}
    for method in METHOD_ORDER:
        if method not in method_rows:
            continue
        _path, meta, data = method_rows[method]
        color = COLORS.get(method, None)
        state_med, state_lo, state_hi = median_with_iqr(data["state_post"])
        reward_med, reward_lo, reward_hi = median_with_iqr(data["reward_true"])
        cumulative = np.cumsum(np.nan_to_num(data["reward_true"], nan=0.0), axis=1)
        cum_med, cum_lo, cum_hi = median_with_iqr(cumulative)
        action_med, _action_lo, _action_hi = median_with_iqr(data["action"])

        axes[0].plot(t, state_med, label=method, color=color, lw=2)
        axes[0].fill_between(t, state_lo, state_hi, color=color, alpha=0.14, linewidth=0)
        axes[1].plot(t, reward_med, label=method, color=color, lw=2)
        axes[1].fill_between(t, reward_lo, reward_hi, color=color, alpha=0.14, linewidth=0)
        axes[2].plot(t, cum_med, label=method, color=color, lw=2)
        axes[2].fill_between(t, cum_lo, cum_hi, color=color, alpha=0.14, linewidth=0)
        axes[3].step(t, action_med, label=method, color=color, lw=1.8, where="mid")

    axes[0].axhline(first_meta["safety_threshold"], color="black", ls="--", lw=1, alpha=0.6)
    axes[0].axhline(first_meta["mvp_threshold"], color="gray", ls=":", lw=1, alpha=0.7)
    axes[0].set_ylabel("true state")
    axes[1].set_ylabel("true reward")
    axes[2].set_ylabel("cumulative true reward")
    axes[3].set_ylabel("median action id")
    axes[3].set_xlabel("timestep")
    axes[3].set_yticks(range(len(first_meta["action_names"])))
    axes[0].set_title(title + " (median over paired demo episodes; band = IQR)")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = (
        f"{first_meta['population_slug']}_{first_meta['environment']}_"
        f"sigma_{str(first_meta['sigma_obs']).replace('.', 'p')}"
    )
    fig.savefig(out_dir / f"{slug}_trajectory.png", dpi=160)
    plt.close(fig)

    plot_actions(cell_rows, out_dir / f"{slug}_actions.png")
    return slug


def plot_actions(cell_rows, out_path: Path):
    first_meta = cell_rows[0][1]
    method_rows = {meta["method"]: (path, meta, data) for path, meta, data in cell_rows}
    horizon = int(first_meta["horizon"])
    num_actions = len(first_meta["action_names"])
    action_costs = load_action_costs(num_actions)
    t = np.arange(horizon)
    fig, axes = plt.subplots(len(METHOD_ORDER), 1, figsize=(11, 8), sharex=True)
    cmap = cost_colors(action_costs)
    for ax, method in zip(axes, METHOD_ORDER):
        if method not in method_rows:
            ax.axis("off")
            continue
        _path, _meta, data = method_rows[method]
        actions = data["action"]
        freq = np.zeros((num_actions, horizon))
        for step in range(horizon):
            col = actions[:, step]
            col = col[~np.isnan(col)].astype(int)
            if len(col):
                for action_id in range(num_actions):
                    freq[action_id, step] = float(np.mean(col == action_id))
        ax.stackplot(
            t,
            *[freq[i] for i in range(num_actions)],
            colors=cmap,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel(method, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("timestep")
    action_order = sorted(range(num_actions), key=lambda i: (action_costs[i], i))
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=cmap[i],
            lw=6,
            label=f"a{i} cost={action_costs[i]:g}: {name}",
        )
        for i in action_order
        for name in [first_meta["action_names"][i]]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle(
        f"{first_meta['population']} action composition over time "
        "(blue=low/revenue cost, red=high cost)",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0.13, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_index(rows, out_dir: Path):
    with (out_dir / "trajectory_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "population", "environment", "sigma_obs", "method", "filter",
            "episodes", "seconds", "trace",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path, meta, _data in rows:
            writer.writerow({
                "population": meta["population"],
                "environment": meta["environment"],
                "sigma_obs": meta["sigma_obs"],
                "method": meta["method"],
                "filter": meta["filter"],
                "episodes": meta["episodes"],
                "seconds": f"{float(meta['seconds']):.3f}",
                "trace": str(path),
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, default=Path(__file__).resolve().parents[1] / "traces")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = parser.parse_args()
    rows = load_traces(args.trace_root)
    if not rows:
        raise SystemExit(f"no trajectory files found under {args.trace_root}")
    grouped = {}
    for item in rows:
        _path, meta, _data = item
        key = (meta["population"], meta["environment"], float(meta["sigma_obs"]))
        grouped.setdefault(key, []).append(item)
    args.out.mkdir(parents=True, exist_ok=True)
    plotted = []
    for key in sorted(grouped):
        methods = {meta["method"] for _path, meta, _data in grouped[key]}
        missing = [m for m in METHOD_ORDER if m not in methods]
        if missing:
            print(f"warning: {key} missing methods: {missing}", flush=True)
        plotted.append(plot_cell(grouped[key], args.out))
    write_index(rows, args.out)
    print(json.dumps({
        "status": "ok",
        "figures": str(args.out),
        "cells": plotted,
        "trace_files": len(rows),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
