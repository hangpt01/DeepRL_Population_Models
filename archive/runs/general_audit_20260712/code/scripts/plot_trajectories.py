#!/usr/bin/env python3
"""Per-cell trajectory plots from outputs/trajectories/*.npz.

For each cell, writes:
  outputs/plots/<cell>.png            three panels (post-transition true state,
                                       median action id, immediate reward) with one
                                       median line per method, faint per-episode
                                       traces, collapse markers, and per-method
                                       collapse counts in the legend;
  outputs/plots/actions/<cell>_actions.png  per-method action-frequency heatmaps;
  outputs/plots/_contact_sheet_abundance.png  all-cell overview (median lines).

Note on the time axis: the captured abundance at index t is the POST-transition
true state s_{t+1} (the state after action a_t), while action/reward at index t are
a_t / R_t.  The abundance axis is labelled accordingly.  Lines are medians over the
captured paired episodes (5); they are illustrative, not the statistical result
(that is the 250-episode evaluation in the report tables).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def _smooth(y, w):
    """nan-aware centered moving average (w=1 -> unchanged)."""
    y = np.asarray(y, dtype=float)
    if w is None or w <= 1:
        return y
    half = w // 2
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - half):min(len(y), i + half + 1)]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def _collapse_count(ab, thr):
    """episodes (rows) that reach <= threshold at any timestep / total episodes."""
    hit = np.array([np.any(row[~np.isnan(row)] <= thr) for row in ab])
    return int(hit.sum()), len(ab)


def plot_cell(path: Path, out_dir: Path, smooth: int = 1) -> str:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    methods, H, thr = meta["methods"], meta["horizon"], meta["safety_threshold"]
    t = np.arange(H)
    fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for i, m in enumerate(methods):
        c = COLORS[i]
        ab, ac, rw = data[f"{m}_abundance"], data[f"{m}_action"], data[f"{m}_reward"]
        nc, ne = _collapse_count(ab, thr)
        # faint per-episode abundance traces (raw) + smoothed bold median
        for ep in range(ab.shape[0]):
            ax[0].plot(t, ab[ep], color=c, lw=0.5, alpha=0.12)
        ax[0].plot(t, _smooth(np.nanmedian(ab, axis=0), smooth), color=c, lw=2.0,
                   label=f"{m} (c={nc}/{ne})")
        # collapse markers: first timestep <= threshold per episode (raw positions)
        for ep in range(ab.shape[0]):
            row = ab[ep]
            below = np.where(row <= thr)[0]
            if len(below):
                ax[0].scatter(below[0], row[below[0]], color=c, marker="x", s=28, zorder=5)
        ax[1].plot(t, _smooth(np.nanmedian(ac, axis=0), smooth), color=c, lw=1.8, label=m)
        ax[2].plot(t, _smooth(np.nanmedian(rw, axis=0), smooth), color=c, lw=1.8, label=m)
    ax[0].axhline(thr, ls="--", color="k", lw=1.0, alpha=0.6, label=f"safety={thr:.0f}")
    ax[0].axhline(meta["K_base"], ls=":", color="grey", lw=1.0, alpha=0.6, label=f"K={meta['K_base']:.0f}")
    ax[0].set_ylabel("true post-transition state $s_{t+1}$")
    sm = f", smoothed w={smooth}" if smooth and smooth > 1 else ""
    ax[0].set_title(f"{meta['cell']} — median over {meta['episodes']} paired episodes "
                    f"(learned filter{sm}); x = first-collapse, (c=collapsed/total)")
    ax[1].set_ylabel("median action id (categorical;\nsee action-frequency figure)")
    ax[1].set_yticks(range(meta["num_actions"]))
    ax[2].set_ylabel("immediate reward $R_t$")
    ax[2].set_xlabel("timestep $t$")
    ax[0].legend(ncol=5, fontsize=7.5, loc="best")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{meta['cell']}.png", dpi=120)
    plt.close(fig)
    plot_action_composition(data, meta, out_dir / "actions", smooth)
    return meta["cell"]


# Semantic short names for the locked action tables (id -> meaning).
ACTION_NAMES = {
    5: ["0 do-nothing", "1 harvest 50%", "2 support +60", "3 restore +100",
        "4 flagship +160"],
    10: ["0 do-nothing", "1 harvest 25%", "2 harvest 50%", "3 support +40",
         "4 support +60", "5 restore +80", "6 restore +100", "7 integrated +80",
         "8 adaptive +120", "9 flagship +160"],
}


def _action_colors(A):
    """Semantic palette: grey=do-nothing, reds=harvest, greens->blues=support."""
    harvest = {1} if A == 5 else {1, 2}
    support = [a for a in range(A) if a != 0 and a not in harvest]
    colors = [None] * A
    colors[0] = "#9e9e9e"
    for col, a in zip(plt.cm.Reds(np.linspace(0.55, 0.88, len(harvest))), sorted(harvest)):
        colors[a] = col
    for col, a in zip(plt.cm.GnBu(np.linspace(0.35, 0.95, len(support))), support):
        colors[a] = col
    return colors


def plot_action_composition(data, meta, out_dir: Path, smooth: int = 1):
    """Stacked action-fraction over time, one panel per method (readable colors)."""
    methods, H, A = meta["methods"], meta["horizon"], meta["num_actions"]
    t = np.arange(H)
    colors = _action_colors(A)
    names = ACTION_NAMES.get(A, [str(a) for a in range(A)])
    fig, axes = plt.subplots(len(methods), 1, figsize=(11, 1.45 * len(methods)), sharex=True)
    for ax, m in zip(axes, methods):
        ac = data[f"{m}_action"]
        freq = np.zeros((A, H))
        for tt in range(H):
            col = ac[:, tt][~np.isnan(ac[:, tt])]
            if len(col):
                for a in range(A):
                    freq[a, tt] = np.mean(col == a)
        if smooth and smooth > 1:
            freq = np.vstack([_smooth(freq[a], smooth) for a in range(A)])
            tot = freq.sum(axis=0); tot[tot == 0] = 1.0; freq = freq / tot
        ax.stackplot(t, *[freq[a] for a in range(A)], colors=colors, labels=names,
                     edgecolor="white", linewidth=0.15)
        ax.set_ylim(0, 1); ax.set_xlim(0, H - 1)
        ax.set_ylabel(m, fontsize=9, rotation=0, ha="right", va="center")
        ax.set_yticks([0, 1]); ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("timestep $t$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(A, 5), fontsize=8, frameon=False)
    fig.suptitle(f"{meta['cell']} — action composition over time "
                 f"(fraction of {meta['episodes']} episodes; learned filter)", y=0.998)
    fig.tight_layout(rect=[0.02, 0.07 if A > 5 else 0.05, 1, 0.98])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{meta['cell']}_actions.png", dpi=120)
    plt.close(fig)


def contact_sheet(out_dir: Path, smooth: int = 1):
    npzs = sorted(glob.glob(str(ROOT / "outputs/trajectories/*.npz")))
    n = len(npzs); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
    methods0 = json.loads(str(np.load(npzs[0], allow_pickle=False)["meta"]))["methods"]
    for k, path in enumerate(npzs):
        data = np.load(path, allow_pickle=False); meta = json.loads(str(data["meta"]))
        ax = axes[k // cols][k % cols]; t = np.arange(meta["horizon"])
        for i, m in enumerate(meta["methods"]):
            ax.plot(t, _smooth(np.nanmedian(data[f"{m}_abundance"], axis=0), smooth), color=COLORS[i], lw=1.0)
        ax.axhline(meta["safety_threshold"], ls="--", color="k", lw=0.7, alpha=0.5)
        ax.set_title(meta["cell"], fontsize=8); ax.tick_params(labelsize=6)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    handles = [plt.Line2D([], [], color=COLORS[i], label=m) for i, m in enumerate(methods0)]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8)
    fig.suptitle("Median post-transition true state over time — all cells (learned filter)", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    fig.savefig(out_dir / "_contact_sheet_abundance.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default=str(ROOT / "outputs/trajectories"))
    ap.add_argument("--out", default=str(ROOT / "outputs/plots"))
    ap.add_argument("--smooth", type=int, default=1,
                    help="centered moving-average window for median lines (1 = none)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    files = sorted(glob.glob(str(Path(args.trajectories) / "*.npz")))
    for f in files:
        print("plotted", plot_cell(Path(f), out_dir, args.smooth), flush=True)
    if files:
        contact_sheet(out_dir, args.smooth)
        print(f"contact sheet + {len(files)} cell figures + action heatmaps in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
