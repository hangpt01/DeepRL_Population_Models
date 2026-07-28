#!/usr/bin/env python3
"""S2 81/161-bin convergence probe for Ricker transfer regret and VPI."""
from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_s2 as s2  # noqa: E402

OUT = s2.OUT
RESOLUTIONS = (81, 161, 321)


def evaluate_resolution(pop, bins, reference_traces=None):
    sigma = s2.ORACLE_SIGMA[pop]
    cfgs = {family: s2.config(pop, family, sigma) for family in s2.FAMILIES}
    policies = {
        (family, seed): s2.backward_oracle(cfgs[family], seed, bins)
        for family in s2.FAMILIES for seed in s2.SEEDS
    }
    values = np.empty((len(s2.SEEDS), 4, 4))
    majority = np.empty((len(s2.SEEDS), 4))
    for si, seed in enumerate(s2.SEEDS):
        for gi, G in enumerate(s2.FAMILIES):
            for fi, F in enumerate(s2.FAMILIES):
                values[si, fi, gi] = s2.evaluate(
                    cfgs[G], seed, [policies[F, seed]])
            majority[si, gi] = s2.evaluate(
                cfgs[G], seed, [policies[f, seed] for f in s2.FAMILIES],
                majority=True)

    diag = np.asarray([values[:, i, i].mean() for i in range(4)])
    perfect = float(diag.mean())
    candidate_values = list(values.mean(axis=0).mean(axis=1))
    candidate_values.append(float(majority.mean()))
    common = float(max(candidate_values))
    raw_vpi = perfect - common
    envelope = float(np.mean(np.max(values.mean(axis=0), axis=0)))

    ricker = []
    for gi, G in enumerate(s2.FAMILIES):
        paired = values[:, 0, gi] - values[:, gi, gi]
        if gi == 0:
            paired[:] = 0.0
        lo, hi = s2.paired_ci(paired)
        ricker.append({
            "population": pop, "bins": bins, "policy_family_F": "ricker",
            "evaluation_family_G": G, "mean_regret": float(paired.mean()),
            "ci95_low": lo, "ci95_high": hi,
        })

    traces = {}
    flip_count = flip_total = 0
    for family in s2.FAMILIES:
        for seed in s2.SEEDS:
            if reference_traces is not None:
                for t, state, K, old_action in reference_traces[family, seed]:
                    new_action = int(np.argmax(s2.q_at(
                        policies[family, seed], t, state, K)))
                    flip_count += new_action != old_action
                    flip_total += 1
            env = s2.make_env(cfgs[family].environment)
            env.reset(seed)
            trace = []
            for t in range(s2.HORIZON):
                if env._done:
                    break
                K = float(np.clip(env.cfg.K_base + env._kappa,
                                  env.cfg.K_min, env.cfg.K_max))
                action = int(np.argmax(s2.q_at(
                    policies[family, seed], t, env.state, K)))
                trace.append((t, float(env.state), K, action))
                env.step(action)
            traces[family, seed] = trace
    result = {
        "population": pop, "bins": bins,
        "perfect_family_information_value": perfect,
        "best_common_policy_value": common,
        "best_common_policy": (list(s2.FAMILIES) + ["majority"])[
            int(np.argmax(candidate_values))],
        "VPI_raw": raw_vpi,
        "available_oracle_envelope_value": envelope,
        "own_oracle_gap_to_envelope": perfect - envelope,
        "flip_fraction_vs_previous_grid": (
            None if reference_traces is None else flip_count / max(flip_total, 1)),
    }
    del policies
    gc.collect()
    return result, ricker, traces


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, regrets = [], []
    for pop in s2.SPECIES:
        previous = None
        for bins in RESOLUTIONS:
            metric, ricker, traces = evaluate_resolution(pop, bins, previous)
            metrics.append(metric)
            regrets.extend(ricker)
            previous = traces
        del previous
        gc.collect()

    with (OUT / "grid_convergence_vpi.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (OUT / "grid_convergence_ricker_row.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regrets[0]))
        writer.writeheader()
        writer.writerows(regrets)

    # Include the already-computed 41-bin result without rewriting it.
    old_vpi = {
        row["population"]: float(row["VPI"])
        for row in csv.DictReader((OUT / "vpi.csv").open())
        if row["prior"] == "uniform"
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    for pop in s2.SPECIES:
        ys = [old_vpi[pop]] + [
            r["VPI_raw"] for r in metrics if r["population"] == pop]
        ax.plot((41,) + RESOLUTIONS, ys, marker="o", label=pop)
    ax.axhline(0, color="black", lw=.8)
    ax.axhline(s2.THRESHOLDS["delta_VPI"], color="black", ls="--", lw=.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks((41,) + RESOLUTIONS, tuple(map(str, (41,) + RESOLUTIONS)))
    ax.set_xlabel("abundance bins")
    ax.set_ylabel("raw uniform-prior VPI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "grid_convergence_vpi.png", dpi=180)
    plt.close(fig)

    fox = [old_vpi["Crab-eating fox"]] + [
        r["VPI_raw"] for r in metrics if r["population"] == "Crab-eating fox"]
    tiger = [old_vpi["Amur tiger"]] + [
        r["VPI_raw"] for r in metrics if r["population"] == "Amur tiger"]
    receipt = {
        "schema": "S2_grid_convergence_v1",
        "resolutions": [41] + list(RESOLUTIONS),
        "fox_VPI_raw": fox, "tiger_VPI_raw": tiger,
        "fox_last_grid_abs_change": abs(fox[-1] - fox[-2]),
        "tiger_last_grid_abs_change": abs(tiger[-1] - tiger[-2]),
        "theoretical_nonnegative_VPI_enforced": False,
        "raw_values_retained": True,
        "S2_decided": (
            fox[-1] >= s2.THRESHOLDS["delta_VPI"]
            and fox[-2] >= s2.THRESHOLDS["delta_VPI"]
            and abs(fox[-1] - fox[-2]) < 0.01
            and tiger[-1] >= 0.0 and tiger[-2] >= 0.0
            and abs(tiger[-1] - tiger[-2]) < 0.001
        ),
        "decision_note": (
            "Decided only if the last two fox grids both clear delta_VPI and "
            "change by <0.01, while tiger stays nonnegative and changes by <0.001."),
        "recomputed_fits": 0, "reranked": False,
        "accepted_values_read": False, "accepted_artifacts_modified": False,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "S2_GRID_CONVERGENCE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": receipt, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
