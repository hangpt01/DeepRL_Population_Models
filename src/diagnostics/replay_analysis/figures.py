"""The four figures that carry the story.  Each returns a matplotlib Figure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constants import COLLAPSE_PENALTY, GAMMA

METHOD_ORDER = ["plus", "moor", "refplan", "ogsrl", "bamcts", "evd"]
METHOD_COLOR = {"plus": "#1b5e20", "moor": "#4caf50", "refplan": "#1565c0",
                "ogsrl": "#6a1b9a", "bamcts": "#ef6c00", "evd": "#b71c1c"}


def fig_headroom(headroom_rows, outfile=None):
    """Each method vs its best constant-action baseline, per cell.

    headroom_rows: iterable of dicts with cell_id, method, method_return,
    best_constant_return, best_constant_action.
    """
    import pandas as pd
    df = pd.DataFrame(list(headroom_rows))
    df["headroom"] = df["method_return"] - df["best_constant_return"]
    cells = sorted(df["cell_id"].unique())
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(cells)), 5))
    width = 0.8 / max(len(METHOD_ORDER), 1)
    for i, m in enumerate(METHOD_ORDER):
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        xs = [cells.index(c) + i * width - 0.4 for c in sub["cell_id"]]
        ax.bar(xs, sub["headroom"], width=width, label=m,
               color=METHOD_COLOR.get(m, "grey"))
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("return − best constant-action return")
    ax.set_title("Headroom over the best constant action\n"
                 "(bars at or below zero: the method does not beat doing "
                 "something fixed)", fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
    return fig


def fig_posterior_vs_switching(log, m1, m2, outfile=None):
    """Entropy trajectory with the action-switch fraction annotated.

    The 'inference succeeds, decisions unaffected' figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if "w_t" in log.vectors:
        w = log.vectors["w_t"]
        J = w.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.nansum(np.where(w > 0, w * np.log(w), 0.0), axis=1)
        for _, mask in log.episode_slices():
            ax.plot(np.arange(mask.sum()), H[mask], color="#1565c0",
                    alpha=0.35, lw=1)
        ax.axhline(np.log(J), ls="--", color="grey",
                   label=f"uniform ({np.log(J):.2f} nats)")
        ax.set_ylim(0, np.log(J) * 1.08)
    ax.set_xlabel("step")
    ax.set_ylabel("candidate-posterior entropy (nats)")
    txt = ("switch vs MAP:     {:.3%}\nswitch vs uniform: {:.3%}".format(
        m2.get("switch_vs_MAP", float("nan")),
        m2.get("switch_vs_uniform", float("nan")))
        if m2.get("available") else "switch fractions unavailable")
    ax.text(0.98, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", fc="#fffde7", ec="grey"))
    ax.set_title(f"{log.cell_id} · {log.method}: does belief movement change "
                 f"decisions?", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
    return fig


def fig_raw_vs_centred(m5_by_cell, outfile=None):
    """Cross-candidate disagreement before/after removing the
    action-independent component."""
    cells = list(m5_by_cell)
    raw = [m5_by_cell[c]["raw_mean"] for c in cells]
    cen = [m5_by_cell[c]["centred_mean"] for c in cells]
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(cells)), 4.2))
    ax.bar(x - 0.2, raw, 0.4, label="raw disagreement", color="#90a4ae")
    ax.bar(x + 0.2, cen, 0.4, label="centred (decision-relevant)",
           color="#1b5e20")
    ax.set_xticks(x)
    ax.set_xticklabels(cells, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean variance across candidates")
    ax.set_title("Candidate disagreement: how much of it can change an "
                 "action?", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
    return fig


def fig_return_decomposition(m6_by_method, outfile=None):
    """Stacked utility / cost / penalty per method."""
    methods = [m for m in METHOD_ORDER if m in m6_by_method]
    util = [m6_by_method[m]["utility_disc_mean"] for m in methods]
    cost = [-m6_by_method[m]["cost_disc_mean"] for m in methods]
    pen = [-m6_by_method[m]["penalty_disc_mean"] for m in methods]
    ret = [m6_by_method[m]["return_mean"] for m in methods]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(methods)), 4.6))
    ax.bar(x, util, 0.6, label="abundance utility", color="#2e7d32")
    ax.bar(x, cost, 0.6, bottom=util, label="− action cost", color="#f9a825")
    ax.bar(x, pen, 0.6, bottom=np.array(util) + np.array(cost),
           label="− safety penalty", color="#c62828")
    ax.plot(x, ret, "ko", ms=6, label="total return")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("discounted contribution")
    ax.set_title("Return decomposition: what the score is actually made of",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
    return fig
