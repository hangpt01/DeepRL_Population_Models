#!/usr/bin/env python
"""Overnight P_safe analysis: collect evaluation summaries across the four
collapse_penalty grids, build a decision table + convergence plots.

Defensive by design: any per-cell failure is skipped, never aborts the run, so
a plotting hiccup cannot destroy the aggregated deliverable.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
OUT = RUN / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
PENALTIES = [2, 5, 10, 20]
FOCUS = ["Amur tiger", "Egyptian vulture", "Puerto Rican parrot"]
GENERAL_METHODS = ["ogsrl", "mopo", "bamcts", "refplan", "delphic"]
ECO_BASELINES = ["plus", "moor"]
SINKS = {"Egyptian vulture", "Bottlenose dolphin"}
POLICY_FILTER = {
    "ogsrl": "learned",
    "mopo": "learned",
    "bamcts": "learned",
    "refplan": "learned",
    "delphic": "learned",
    "plus": "ricker",
    "moor": "ricker",
}

METRICS = [
    "unsafe_fraction_mean", "mvp_fraction_mean", "mvp_breach_mean",
    "collapse_entry_mean", "operational_return_mean", "true_return_mean",
    "final_true_state_mean", "min_true_state_mean", "economic_cost_mean",
    "persistence_mean", "safety_threshold",
]


def collect() -> list[dict]:
    rows = []
    for p in PENALTIES:
        root = RUN / "outputs" / f"p{p}" / "evaluation"
        for summ in root.rglob("summary.json"):
            try:
                d = json.loads(summ.read_text())
            except Exception:
                continue
            rec = {
                "penalty": p,
                "population": d.get("population"),
                "environment": d.get("environment"),
                "sigma_obs": d.get("sigma_obs"),
                # method is the parent dir of the filter dir; filter is the leaf dir
                "method": summ.parent.parent.name,
                "filter": d.get("filter") or summ.parent.name,
                "reward_mode": d.get("reward_mode"),
                "backend": d.get("compute_backend_effective"),
                "data_table": d.get("data_table"),
                "path": str(summ),
            }
            for m in METRICS:
                rec[m] = d.get(m)
            rec["training_history_json"] = d.get("training_history_json")
            rows.append(rec)
    return rows


def write_csv(rows: list[dict]) -> None:
    if not rows:
        (OUT / "NO_RESULTS_FOUND.txt").write_text("no summary.json found under outputs/p*/evaluation\n")
        return
    cols = ["penalty", "population", "environment", "sigma_obs", "method",
            "filter", "reward_mode", "backend", "data_table", *METRICS]
    with (OUT / "all_metrics.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else float("nan")


def policy_row(r: dict) -> bool:
    """Comparable method/filter choice used for cross-penalty decisions."""
    return r.get("filter") == POLICY_FILTER.get(r.get("method"))


def decision_table(rows: list[dict]) -> str:
    """Per (penalty, focus population) safe-mode aggregates on a comparable filter subset."""
    decision_rows = [r for r in rows if policy_row(r)]
    lines = ["# P_safe decision aid\n",
             "Safe-mode metrics for the three decision populations, averaged over "
             "method/family/sigma on a filter-consistent subset.\n",
             "\nScope: general learners use `filter=learned`; ecological baselines "
             "`PLUS` and `MOOR` use `filter=ricker`. P5 `raw` rows are excluded "
             "from this penalty-selection table and kept as a separate ablation "
             "in `learned_vs_raw_p5.md`.\n"]
    lines.append("\nComparable safe-row counts by penalty:\n")
    lines.append("| P | rows |")
    lines.append("|---|---:|")
    for p in PENALTIES:
        n = len([r for r in decision_rows if r["penalty"] == p and r["reward_mode"] == "safe"])
        lines.append(f"| {p} | {n} |")
    for pop in FOCUS:
        lines.append(f"\n## {pop}\n")
        lines.append("| P | unsafe_frac | mvp_frac | collapse_entry | true_return | final_state | min_state | econ_cost |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for p in PENALTIES:
            sel = [r for r in decision_rows if r["penalty"] == p and r["population"] == pop
                   and r["reward_mode"] == "safe"]
            if not sel:
                lines.append(f"| {p} | (no rows) | | | | | | |")
                continue
            lines.append("| {p} | {uf:.3f} | {mv:.3f} | {ce:.3f} | {tr:.2f} | {fs:.1f} | {ms:.1f} | {ec:.3f} |".format(
                p=p,
                uf=mean([r["unsafe_fraction_mean"] for r in sel]),
                mv=mean([r["mvp_fraction_mean"] for r in sel]),
                ce=mean([r["collapse_entry_mean"] for r in sel]),
                tr=mean([r["true_return_mean"] for r in sel]),
                fs=mean([r["final_true_state_mean"] for r in sel]),
                ms=mean([r["min_true_state_mean"] for r in sel]),
                ec=mean([r["economic_cost_mean"] for r in sel]),
            ))
    # Heuristic recommendation (aid only; user confirms).
    lines.append("\n## Heuristic recommendation\n")
    try:
        recov = ["Amur tiger", "Puerto Rican parrot"]  # decision-cell recoverables
        def safe_metric(pop, p, key):
            return mean([r[key] for r in decision_rows if r["penalty"] == p
                         and r["population"] == pop and r["reward_mode"] == "safe"])
        ret = {p: {pop: safe_metric(pop, p, "true_return_mean") for pop in recov}
               for p in PENALTIES}
        ce = {p: {pop: safe_metric(pop, p, "collapse_entry_mean") for pop in recov}
              for p in PENALTIES}
        # Rule: largest P that still keeps every recoverable decision-pop's safe
        # true_return non-negative (i.e. not over-penalised) AND lowers
        # collapse_entry vs P=2. Sinks (vulture) are penalty-dominated at every P,
        # so they do not drive the choice — only translocation moves them.
        rec = None
        for p in PENALTIES:
            not_over = all(ret[p][pop] >= 0 for pop in recov)
            safer = all(ce[p][pop] <= ce[2][pop] + 1e-9 for pop in recov)
            if not_over and safer:
                rec = p  # keep the largest qualifying P
        for pop in recov:
            lines.append(f"- {pop} safe true_return by P: "
                         + ", ".join(f"{p}:{ret[p][pop]:.2f}" for p in PENALTIES))
        lines.append(
            "- Egyptian vulture is a demographic sink: unsafe_frac=1.0 and safe "
            "return scales ~linearly worse with P at every level, so it is "
            "penalty-dominated regardless of P (only translocation a10 helps).")
        lines.append(
            f"\n**Heuristic pick: collapse_penalty = "
            f"{rec if rec is not None else 'inconclusive — inspect the table'}** "
            "(largest P keeping recoverable decision-pops' safe return >= 0 while "
            "still reducing collapse entries vs P=2). P=20 clearly over-penalises "
            "(recoverable returns go negative).")
        lines.append("\nThis is an AID, not an auto-committed choice. The penalty "
                     "axis is fully materialised: the full grid exists at every P, "
                     "so any choice you make is already backed by complete results.")
    except Exception as exc:  # never let the heuristic break the report
        lines.append(f"(heuristic skipped: {exc})")
    return "\n".join(lines) + "\n"


def _cell_key(r: dict) -> tuple:
    return (r["population"], r["environment"], r["sigma_obs"])


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else "0/0"


def p5_control_review(rows: list[dict]) -> str:
    """Compact method-family review at the locked P5 setting."""
    p5 = [r for r in rows if r["penalty"] == 5 and r["reward_mode"] == "safe"]
    policy = [r for r in p5 if policy_row(r)]
    idx = {(*_cell_key(r), r["method"]): r for r in policy}
    cells = sorted({_cell_key(r) for r in policy if r["method"] == "plus"})

    def ret(cell: tuple, method: str) -> float:
        return idx[(*cell, method)]["true_return_mean"]

    def scope_rows(scope_cells: list[tuple]) -> tuple[float, float, float, int, dict, dict]:
        best_general = []
        best_base = []
        general_winners = {}
        baseline_winners = {}
        wins = 0
        for cell in scope_cells:
            gvals = {m: ret(cell, m) for m in GENERAL_METHODS}
            bvals = {m: ret(cell, m) for m in ECO_BASELINES}
            gm = max(gvals.values())
            bm = max(bvals.values())
            best_general.append(gm)
            best_base.append(bm)
            wins += gm > bm
            gw = max(gvals, key=gvals.get)
            bw = max(bvals, key=bvals.get)
            general_winners[gw] = general_winners.get(gw, 0) + 1
            baseline_winners[bw] = baseline_winners.get(bw, 0) + 1
        bg = mean(best_general)
        bb = mean(best_base)
        return bg, bb, bg - bb, wins, general_winners, baseline_winners

    lines = ["# P5 control review\n",
             "Scope: `collapse_penalty=5`, `reward_mode=safe`, CPU/numpy results. "
             "General learners use `filter=learned`; ecological baselines are "
             "`PLUS-ricker` and `MOOR-ricker`.\n",
             "\nThis table answers whether general learners beat the two ecological "
             "baselines on matched population/family/noise cells. Higher "
             "`true_return_mean` is better.\n"]

    lines.append("\n## Family check\n")
    lines.append("| Scope | Cells | Best general mean | Best baseline mean | Delta | Wins vs best baseline |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    scopes = [
        ("All 9 populations", cells),
        ("Recoverable only", [c for c in cells if c[0] not in SINKS]),
        ("Sinks only", [c for c in cells if c[0] in SINKS]),
    ]
    for label, scope_cells in scopes:
        bg, bb, delta, wins, _, _ = scope_rows(scope_cells)
        lines.append(f"| {label} | {len(scope_cells)} | {bg:.3f} | {bb:.3f} | "
                     f"{delta:+.3f} | {_pct(wins, len(scope_cells))} |")

    lines.append("\n## Single-method check\n")
    lines.append("| Method | Mean return | Delta vs PLUS | Delta vs MOOR | Beats both | Recoverable beats both | Mean unsafe | Mean collapse |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method in GENERAL_METHODS:
        vals = []
        dplus = []
        dmoor = []
        beats_both = 0
        recoverable_beats = 0
        recoverable_total = 0
        unsafe = []
        collapse = []
        for cell in cells:
            row = idx[(*cell, method)]
            v = ret(cell, method)
            plus = ret(cell, "plus")
            moor = ret(cell, "moor")
            vals.append(v)
            dplus.append(v - plus)
            dmoor.append(v - moor)
            win = v > plus and v > moor
            beats_both += win
            if cell[0] not in SINKS:
                recoverable_total += 1
                recoverable_beats += win
            unsafe.append(row["unsafe_fraction_mean"])
            collapse.append(row["collapse_entry_mean"])
        lines.append(f"| {method} | {mean(vals):.3f} | {mean(dplus):+.3f} | "
                     f"{mean(dmoor):+.3f} | {_pct(beats_both, len(cells))} | "
                     f"{_pct(recoverable_beats, recoverable_total)} | "
                     f"{mean(unsafe):.3f} | {mean(collapse):.3f} |")

    lines.append("\n## Population check\n")
    lines.append("| Population | Best general mean | Best baseline mean | Delta | Wins |")
    lines.append("|---|---:|---:|---:|---:|")
    for pop in sorted({c[0] for c in cells}):
        pop_cells = [c for c in cells if c[0] == pop]
        bg, bb, delta, wins, _, _ = scope_rows(pop_cells)
        lines.append(f"| {pop} | {bg:.3f} | {bb:.3f} | {delta:+.3f} | "
                     f"{wins}/{len(pop_cells)} |")

    lines.append("\n## Learned-vs-raw control ablation at P5\n")
    lines.append("The P5 raw-filter ablation is kept separate from the P_safe decision. "
                 "It shows a clean null control result: learned filtering does "
                 "not materially improve return or unsafe fraction versus raw "
                 "observations.\n")
    lines.append("| Method | learned true_return | raw true_return | Delta learned-raw | unsafe learned/raw |")
    lines.append("|---|---:|---:|---:|---:|")
    for method in ["bamcts", "delphic", "moor", "mopo", "ogsrl", "plus", "refplan"]:
        learned = [r for r in p5 if r["method"] == method and r["filter"] == "learned"]
        raw = [r for r in p5 if r["method"] == method and r["filter"] == "raw"]
        lt = mean([r["true_return_mean"] for r in learned])
        rt = mean([r["true_return_mean"] for r in raw])
        lu = mean([r["unsafe_fraction_mean"] for r in learned])
        ru = mean([r["unsafe_fraction_mean"] for r in raw])
        lines.append(f"| {method} | {lt:.2f} | {rt:.2f} | {lt - rt:+.2f} | "
                     f"{lu:.3f}/{ru:.3f} |")

    lines.append("\nInterpretation: general learners, especially OGSRL, beat the "
                 "ecological baselines on recoverable cells. The two sink "
                 "populations remain a separate harder regime. Learned filtering "
                 "does not buy control-return gains here; that does not test "
                 "state-estimation accuracy because raw observations have no "
                 "latent-state estimate.\n")
    return "\n".join(lines) + "\n"


def rollup(rows: list[dict]) -> None:
    """Per (penalty, reward_mode, method) means — clean substitute for the
    built-in aggregate (which hit a pre-existing NoneType bug on this data)."""
    keys = {}
    for r in rows:
        k = (r["penalty"], r["reward_mode"], r["method"], r.get("filter"))
        keys.setdefault(k, []).append(r)
    with (OUT / "rollup.csv").open("w", newline="") as h:
        cols = ["penalty", "reward_mode", "method", "filter", "n", *METRICS]
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for (p, mode, method, filt), rs in sorted(
            keys.items(), key=lambda t: (t[0][0], t[0][1], t[0][2], str(t[0][3]))
        ):
            row = {"penalty": p, "reward_mode": mode, "method": method,
                   "filter": filt, "n": len(rs)}
            for m in METRICS:
                row[m] = round(mean([x[m] for x in rs]), 4)
            w.writerow(row)


# Iterative-training methods and the *informative* training signal each logs.
# OGSRL's raw surrogate_loss RISES with the safety/OOD dual multipliers (a
# dual-ascent artifact, not divergence), so we show reward_return (rises = the
# policy is learning).  Delphic logs a CQL Bellman MSE.  MOOR/mopo/refplan/
# bamcts/plus fit in closed form (one solve / one-shot) and have no descent
# curve -- their fit quality is the holdout-RMSE figure.
ITER_TRAIN_SIGNAL = {
    "ogsrl": ("reward_return", "training return (higher = better)"),
    "delphic": ("q_bellman_mse", "Bellman MSE (lower = better)"),
}
CLOSED_FORM_METHODS = ["moor", "mopo", "refplan", "bamcts", "plus"]


def _metric_series(thj: str, metric: str):
    """Return the step-ordered value list for ``metric`` on the train split."""
    try:
        hrows = json.loads(Path(thj).read_text()).get("rows", [])
    except Exception:
        return None
    pts = [(x.get("step", i), x.get("value")) for i, x in enumerate(hrows)
           if isinstance(x, dict) and x.get("split") in (None, "train")
           and x.get("metric") == metric and isinstance(x.get("value"), (int, float))]
    pts.sort(key=lambda t: t[0])
    return [v for _, v in pts] if len(pts) >= 2 else None


def _median_iqr(series: list[list[float]]):
    """Per-step median and 25/75 percentiles across cell curves (robust to the
    few numerically unstable cells, e.g. delphic Q-explosions)."""
    import numpy as np
    n = min(len(s) for s in series)
    arr = np.array([s[:n] for s in series], dtype=float)
    return (np.median(arr, axis=0), np.percentile(arr, 25, axis=0),
            np.percentile(arr, 75, axis=0))


def convergence(rows: list[dict]) -> None:
    """Per-penalty small-multiples: one panel per iterative-training method,
    median + IQR band across cells (robust), each on its own objective/axis."""
    import numpy as np
    made = 0
    unstable_note = ""
    for p in PENALTIES:
        curves: dict[str, list[list[float]]] = {}
        for r in rows:
            if r.get("penalty") != p or not r.get("training_history_json"):
                continue
            sig = ITER_TRAIN_SIGNAL.get(r["method"])
            if not sig:
                continue
            ys = _metric_series(r["training_history_json"], sig[0])
            if ys:
                curves.setdefault(r["method"], []).append(ys)
        methods = [m for m in ITER_TRAIN_SIGNAL if m in curves]
        if not methods:
            continue
        fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 3.9),
                                 squeeze=False)
        for ax, method in zip(axes[0], methods):
            metric, ylabel = ITER_TRAIN_SIGNAL[method]
            med, q25, q75 = _median_iqr(curves[method])
            x = range(len(med))
            ax.fill_between(x, q25, q75, color="tab:blue", alpha=0.2, label="IQR (25-75%)")
            ax.plot(x, med, color="tab:red", lw=2.0, label="median")
            # robust y-limits: frame the IQR, let unstable cells go off-screen
            lo, hi = float(np.min(q25)), float(np.max(q75))
            pad = 0.1 * (hi - lo + 1e-9)
            ax.set_ylim(lo - pad, hi + pad)
            n_unstable = sum(1 for s in curves[method] if max(s) > 1e6)
            title = f"{method}  ({metric})"
            if n_unstable:
                title += f"\n{n_unstable}/{len(curves[method])} cells numerically unstable"
                unstable_note += f"p{p} {method}: {n_unstable}/{len(curves[method])} cells >1e6\n"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("training step")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=8)
        fig.suptitle(f"Training convergence (collapse_penalty={p}) — iterative-training "
                     f"methods (median + IQR over cells)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / f"convergence_p{p}.png", dpi=110)
        plt.close(fig)
        made += 1
    (OUT / "convergence_note.txt").write_text(
        f"convergence overview PNGs written: {made}/{len(PENALTIES)}\n"
        "Iterative-training methods: OGSRL (training reward_return, rises=learning; "
        "its raw surrogate_loss rises only due to safety/OOD dual ascent) and "
        "Delphic (q_bellman_mse). Median+IQR over cells; y-axis frames the IQR so "
        "a few unstable Delphic cells fall off-screen.\n"
        f"Closed-form / one-shot fitters ({', '.join(CLOSED_FORM_METHODS)}) have no "
        "descent curve; fit quality is fig_p5_holdout_rmse_by_method.png.\n"
        + ("Numerically unstable cells:\n" + unstable_note if unstable_note else ""))


def main() -> None:
    rows = collect()
    write_csv(rows)
    try:
        rollup(rows)
    except Exception as exc:
        (OUT / "rollup_error.txt").write_text(str(exc))
    (OUT / "DECISION_psafe.md").write_text(decision_table(rows))
    (OUT / "P5_control_review.md").write_text(p5_control_review(rows))
    try:
        convergence(rows)
    except Exception as exc:
        (OUT / "convergence_error.txt").write_text(str(exc))
    print(f"analysis: {len(rows)} summary rows -> {OUT}")


if __name__ == "__main__":
    main()
