#!/usr/bin/env python3
"""S2 cross-family identifiability, oracle regret, and VPI analysis.

This is a new evaluator-only analysis solver.  It does not fit or alter any
accepted policy and never reads accepted result values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import (  # noqa: E402
    ECOLOGICAL_SCRIPTS,
    ECOLOGICAL_SRC,
    FOLLOWUPS_OUTPUT,
    P10_DATA_ROOT,
    P10_PACKAGE,
    REPLAY_OUTPUT,
    require_scientific_tables,
)

P10 = P10_DATA_ROOT
REPLAY = REPLAY_OUTPUT
OUT = Path(os.environ.get("DEEPRL_S2_OUTPUT", FOLLOWUPS_OUTPUT / "S2"))
sys.path[:0] = [str(ECOLOGICAL_SCRIPTS), str(ECOLOGICAL_SRC)]

import run_real_manifest_row as rrmr  # noqa: E402
from real_ecology_benchmark.config import load_config  # noqa: E402
from real_ecology_benchmark.envs import make_env  # noqa: E402

FAMILIES = ("ricker", "allee", "theta", "regime")
SPECIES = ("Crab-eating fox", "Amur tiger")
SIGMAS = {"Crab-eating fox": ("0.2", "0.1"), "Amur tiger": ("0.1",)}
ORACLE_SIGMA = {"Crab-eating fox": "0.2", "Amur tiger": "0.1"}
SEEDS = [b + i for b in (7001, 7051, 7101, 7151, 7201) for i in range(4)]
GAMMA, HORIZON = 0.95, 50
THRESHOLDS = {"rho_star": 1.0, "lnB_star": 4.6, "delta_R": 0.10,
              "k_cells": 2, "delta_VPI": 0.10}


def slug(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")


def manifest_row(pop: str, family: str, sigma: str):
    path = P10_PACKAGE / "manifests" / "plus_p10_plan_24.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["population"], row["environment"], row["sigma_obs"]) == (
                pop, family, sigma
            ):
                return row
    raise FileNotFoundError(f"NOT FOUND: manifest identity {pop}/{family}/{sigma}")


def config(pop: str, family: str, sigma: str):
    base = load_config(str(P10_PACKAGE / "configs" / "plus_ricker_only_p10.yaml"))
    cfg, _, _ = rrmr.apply_row_config(base, manifest_row(pop, family, sigma), P10 / "plus")
    cfg.validate()
    return cfg


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def lognormal_ll(y, means, sigma):
    y = max(float(y), np.finfo(float).tiny)
    means = np.maximum(np.asarray(means, float), np.finfo(float).tiny)
    z = (math.log(y) - np.log(means) + 0.5 * sigma * sigma) / sigma
    ll = -0.5 * z * z - math.log(y * sigma * math.sqrt(2 * math.pi))
    top = float(np.max(ll))
    return top + math.log(float(np.mean(np.exp(ll - top))))


def marginal_predictions(cfg, x, action, k_previous):
    """Family marginal predictive support (C/theta quadrature, z mixture)."""
    env = make_env(cfg.environment)
    e = cfg.environment
    action_spec = env.actions[int(action)]
    Cs = np.linspace(0.18 * e.K_ref, 0.30 * e.K_ref, 9)
    thetas = np.linspace(3.0, 6.0, 9)
    regimes = (0, 1)
    if e.kind == "allee":
        draws = [(c, 4.5, 0) for c in Cs]
    elif e.kind == "theta":
        draws = [(0.24 * e.K_ref, th, 0) for th in thetas]
    elif e.kind == "regime":
        draws = [(0.24 * e.K_ref, 4.5, z) for z in regimes]
    else:
        draws = [(0.24 * e.K_ref, 4.5, 0)]
    kappa = float(k_previous) - e.K_base
    return np.asarray([
        env.transition_value(float(x), action_spec, 0.0, C, theta, z, 0.0,
                             action_spec.delta_r, kappa)
        for C, theta, z in draws
    ])


def public_support(pop, family, sigma):
    path = (P10 / "plus/datasets/regime_hidden/reward_safe" / slug(pop) /
            family / f"sigma_{sigma.replace('.', 'p')}" / "public.npz")
    if not path.exists():
        raise FileNotFoundError(f"NOT FOUND: {path}")
    with np.load(path, allow_pickle=False) as z:
        obs, nxt, actions = z["observations"], z["next_observations"], z["actions"]
        episode, timestep = z["episode_id"], z["timestep"]
    cfg = config(pop, family, sigma)
    actions_table = make_env(cfg.environment).actions
    K = cfg.environment.K_base
    rows = []
    previous_episode = None
    for x, y, a, ep, t in zip(obs, nxt, actions, episode, timestep):
        if previous_episode != int(ep) or int(t) == 0:
            K = cfg.environment.K_base
        rows.append((float(x), float(y), int(a), float(K), int(ep)))
        K = float(np.clip(
            K + actions_table[int(a)].delta_K,
            cfg.environment.K_min, cfg.environment.K_max,
        ))
        previous_episode = int(ep)
    return rows


def visitation_support(pop, family, sigma):
    token = slug(pop)
    rows = []
    episode_offset = 0
    for path in sorted((REPLAY / "logs").glob(f"*_{token}_*__*diagnostic_replay.npz")):
        if f"_s{sigma.replace('.', 'p')}__" not in path.name:
            continue
        if f"_{family}_" not in path.name:
            continue
        cfg = config(pop, family, sigma)
        actions_table = make_env(cfg.environment).actions
        with np.load(path, allow_pickle=False) as z:
            required = {"x_true_t", "x_true_next", "action_t", "seed", "k_t"}
            if not required.issubset(z.files):
                continue
            for x, y, a, seed, K_after in zip(
                z["x_true_t"], z["x_true_next"], z["action_t"], z["seed"], z["k_t"]
            ):
                K_before = float(np.clip(
                    float(K_after) - actions_table[int(a)].delta_K,
                    cfg.environment.K_min, cfg.environment.K_max,
                ))
                rows.append((float(x), float(y), int(a), K_before,
                             episode_offset + int(seed)))
        episode_offset += 100000
    return rows


def g1_identifiability():
    output = []
    for pop in SPECIES:
        for sigma in SIGMAS[pop]:
            cfg_by_f = {f: config(pop, f, sigma) for f in FAMILIES}
            for true_family in FAMILIES:
                sources = {
                    "logged_support": public_support(pop, true_family, sigma),
                    "accepted_policy_visitation": visitation_support(
                        pop, true_family, sigma),
                }
                for source, samples in sources.items():
                    if not samples:
                        continue
                    # Deterministic, evenly spaced cap avoids overweighting duplicate
                    # accepted trajectories while retaining the full state range.
                    if len(samples) > 4000:
                        idx = np.linspace(0, len(samples) - 1, 4000).astype(int)
                        samples = [samples[i] for i in idx]
                    predictions = {f: [] for f in FAMILIES}
                    ll = {f: [] for f in FAMILIES}
                    episodes = []
                    for x, y, action, K, episode in samples:
                        episodes.append(episode)
                        for f in FAMILIES:
                            pred = marginal_predictions(cfg_by_f[f], x, action, K)
                            predictions[f].append(float(np.exp(np.mean(np.log(
                                np.maximum(pred, np.finfo(float).tiny))))))
                            ll[f].append(lognormal_ll(y, pred, float(sigma)))
                    episodes = np.asarray(episodes)
                    for F in FAMILIES:
                        for G in FAMILIES:
                            if F >= G:
                                continue
                            d = np.abs(np.log(np.maximum(predictions[F], 1e-300)) -
                                       np.log(np.maximum(predictions[G], 1e-300)))
                            rho = d / float(sigma)
                            episode_lnB = []
                            for ep in np.unique(episodes):
                                mask = episodes == ep
                                # Evidence in favor of the data-generating family
                                # against the paired alternative, capped at 50 steps.
                                alt = G if F == true_family else F
                                if true_family not in (F, G):
                                    alt = G
                                indices = np.flatnonzero(mask)[:50]
                                episode_lnB.append(float(np.sum(
                                    np.asarray(ll[true_family])[indices] -
                                    np.asarray(ll[alt])[indices]
                                )))
                            output.append({
                                "population": pop, "sigma_obs": sigma,
                                "true_family": true_family, "source": source,
                                "family_F": F, "family_G": G, "n": len(rho),
                                "rho_median": float(np.median(rho)),
                                "rho_q25": float(np.quantile(rho, .25)),
                                "rho_q75": float(np.quantile(rho, .75)),
                                "rho_frac_lt_1": float(np.mean(rho < 1)),
                                "achieved_lnB_50_mean": float(np.mean(episode_lnB)),
                                "heldout_predictive_LL_F": float(np.mean(ll[F])),
                                "heldout_predictive_LL_G": float(np.mean(ll[G])),
                                "rho_threshold": THRESHOLDS["rho_star"],
                                "lnB_threshold": THRESHOLDS["lnB_star"],
                            })
    return output


def regime_path(env, horizon=HORIZON):
    z = []
    current = int(env._regime)
    rng = env._rngs["regime"]
    for _ in range(horizon):
        z.append(current)
        if rng.random() > env.cfg.regime_persistence:
            current = 1 - current
    return np.asarray(z, np.int8)


def realized(cfg, seed):
    env = make_env(cfg.environment)
    env.reset(seed)
    return {
        "r_base": float(env._r_base), "C": float(env._C),
        "theta": float(env._theta), "regime_path": regime_path(env),
    }


def state_grid(e, bins):
    maximum = max(float(e.K_max) * 1.5, 1.0)
    return np.concatenate(([0.0], np.geomspace(1e-4, maximum, bins - 1)))


def next_values(e, actions, xgrid, Kgrid, draw, regime):
    """Vectorized frozen env transition, shape actions × x × K."""
    out = np.empty((len(actions), len(xgrid), len(Kgrid)))
    for ai, a in enumerate(actions):
        Knext = np.clip(Kgrid + a.delta_K, e.K_min, e.K_max)
        managed = np.maximum(xgrid[:, None] + a.stocking_delta, 0.0)
        r_eff = float(np.clip(a.delta_r, e.r_min, e.r_max))
        r_pos, r_mort = max(r_eff, 0.0), min(r_eff, 0.0)
        if e.kind == "ricker":
            val = managed * np.exp(np.clip(r_pos * (1 - managed / Knext), -745, 709))
        elif e.kind == "allee":
            val = managed * np.exp(np.clip(
                r_pos * (1 - managed / Knext) * (managed / draw["C"] - 1),
                -745, 709))
        elif e.kind == "theta":
            val = managed + r_pos * managed * (
                1 - np.maximum(managed / Knext, 0) ** draw["theta"])
        else:
            threshold = (e.regime_threshold_low if regime == 0
                         else e.regime_threshold_high)
            mult = 1.0 if regime == 0 else e.regime_weak_multiplier
            val = managed * np.exp(np.clip(
                mult * r_pos * (1 - managed / Knext) *
                (managed / threshold - 1), -745, 709))
        if r_mort < 0:
            val *= math.exp(r_mort)
        out[ai] = np.maximum(val, 0.0)
    return out


def interp2(V, xgrid, Kgrid, x, K):
    K = float(np.clip(K, Kgrid[0], Kgrid[-1]))
    hi = int(np.searchsorted(Kgrid, K, side="right"))
    hi = min(max(hi, 1), len(Kgrid) - 1)
    lo = hi - 1
    w = (K - Kgrid[lo]) / max(Kgrid[hi] - Kgrid[lo], 1e-300)
    return ((1 - w) * np.interp(x, xgrid, V[:, lo]) +
            w * np.interp(x, xgrid, V[:, hi]))


def backward_oracle(cfg, seed, bins):
    e = cfg.environment
    actions = make_env(e).actions
    xgrid = state_grid(e, bins)
    Kgrid = np.linspace(e.K_min, e.K_max, 9)
    draw = realized(cfg, seed)
    # Keep convergence probes in float64: VPI is a difference of policy values
    # and can sit close to the decision threshold.
    q = np.empty((HORIZON, bins, len(Kgrid), len(actions)), np.float64)
    Vnext = np.zeros((bins, len(Kgrid)))
    for t in range(HORIZON - 1, -1, -1):
        xn = next_values(e, actions, xgrid, Kgrid, draw, int(draw["regime_path"][t]))
        for ai, a in enumerate(actions):
            future = np.empty((bins, len(Kgrid)))
            Kn = np.clip(Kgrid + a.delta_K, e.K_min, e.K_max)
            for ki, kval in enumerate(Kn):
                future[:, ki] = interp2(Vnext, xgrid, Kgrid, xn[ai, :, ki], kval)
            reward = (xn[ai] / (xn[ai] + e.K_ref) - a.cost -
                      e.collapse_penalty * (xn[ai] <= e.safety_threshold))
            reward[xgrid == 0, :] = 0.0
            future[xn[ai] == 0] = 0.0
            q[t, :, :, ai] = reward + GAMMA * future
        Vnext = np.max(q[t].astype(float), axis=-1)
    return {"q": q, "xgrid": xgrid, "Kgrid": Kgrid}


def q_at(policy, t, x, K):
    return np.asarray([
        interp2(policy["q"][t, :, :, a], policy["xgrid"], policy["Kgrid"], x, K)
        for a in range(policy["q"].shape[-1])
    ])


def evaluate(cfg, seed, policies, majority=False):
    env = make_env(cfg.environment)
    reset = env.reset(seed)
    total = 0.0
    for t in range(HORIZON):
        if env._done:
            break
        K = float(np.clip(env.cfg.K_base + env._kappa, env.cfg.K_min, env.cfg.K_max))
        qs = np.asarray([q_at(p, t, env.state, K) for p in policies])
        if majority:
            votes = np.argmax(qs, axis=1)
            action = int(np.argmax(np.bincount(votes, minlength=env.num_actions)))
        else:
            action = int(np.argmax(qs[0]))
        result = env.step(action)
        total += (GAMMA ** t) * float(result.reward)
    return float(total)


def paired_ci(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(20260727)
    means = np.mean(rng.choice(x, size=(4000, len(x)), replace=True), axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def g2_g3():
    regret_rows, stability_rows, vpi_rows = [], [], []
    matrices = {}
    for pop in SPECIES:
        sigma = ORACLE_SIGMA[pop]
        cfgs = {f: config(pop, f, sigma) for f in FAMILIES}
        policies41, policies81 = {}, {}
        for f in FAMILIES:
            for seed in SEEDS:
                policies41[f, seed] = backward_oracle(cfgs[f], seed, 41)
                policies81[f, seed] = backward_oracle(cfgs[f], seed, 81)

        values = np.empty((len(SEEDS), 4, 4))
        majority_values = np.empty((len(SEEDS), 4))
        for si, seed in enumerate(SEEDS):
            for gi, G in enumerate(FAMILIES):
                for fi, F in enumerate(FAMILIES):
                    values[si, fi, gi] = evaluate(
                        cfgs[G], seed, [policies41[F, seed]])
                majority_values[si, gi] = evaluate(
                    cfgs[G], seed, [policies41[f, seed] for f in FAMILIES],
                    majority=True)

        # Regret is transferred-family policy minus the own-family oracle in G.
        matrix = np.empty((4, 4))
        for fi, F in enumerate(FAMILIES):
            for gi, G in enumerate(FAMILIES):
                paired = values[:, fi, gi] - values[:, gi, gi]
                if F == G:
                    paired[:] = 0.0
                lo, hi = paired_ci(paired)
                matrix[fi, gi] = paired.mean()
                regret_rows.append({
                    "population": pop, "sigma_obs": sigma,
                    "policy_family_F": F, "evaluation_family_G": G,
                    "mean_regret": float(paired.mean()), "ci95_low": lo,
                    "ci95_high": hi, "n_seeds": len(SEEDS),
                    "is_ricker_row": F == "ricker",
                    "abs_exceeds_delta_R": abs(float(paired.mean())) >= THRESHOLDS["delta_R"],
                })
        matrices[pop] = matrix

        # Own-family oracle must dominate the freshly evaluated constants.
        for fi, family in enumerate(FAMILIES):
            # Direct constant evaluation avoids constructing synthetic q arrays.
            constant_means = []
            for action in range(11):
                vals = []
                for seed in SEEDS:
                    env = make_env(cfgs[family].environment)
                    env.reset(seed)
                    total = 0.0
                    for t in range(HORIZON):
                        if env._done:
                            break
                        total += GAMMA ** t * env.step(action).reward
                    vals.append(total)
                constant_means.append(np.mean(vals))
            own = float(np.mean(values[:, fi, fi]))
            best_constant = float(np.max(constant_means))
            if own + 1e-10 < best_constant:
                raise AssertionError((pop, family, own, best_constant))

        flips = []
        for f in FAMILIES:
            for seed in SEEDS:
                env = make_env(cfgs[f].environment)
                env.reset(seed)
                different = total = 0
                for t in range(HORIZON):
                    if env._done:
                        break
                    K = float(np.clip(env.cfg.K_base + env._kappa,
                                      env.cfg.K_min, env.cfg.K_max))
                    a41 = int(np.argmax(q_at(policies41[f, seed], t, env.state, K)))
                    a81 = int(np.argmax(q_at(policies81[f, seed], t, env.state, K)))
                    different += a41 != a81
                    total += 1
                    env.step(a41)
                frac = different / max(total, 1)
                flips.append(frac)
                stability_rows.append({
                    "population": pop, "family": f, "seed": seed,
                    "grid_41": 41, "grid_81": 81,
                    "visited_argmax_flip_fraction": frac,
                })

        priors = {
            "uniform": np.full(4, .25),
            "ricker_0p5": np.asarray([.5, 1/6, 1/6, 1/6]),
            "ricker_0p7": np.asarray([.7, .1, .1, .1]),
            "nonricker": np.asarray([.1, .3, .3, .3]),
        }
        diag = np.asarray([values[:, i, i].mean() for i in range(4)])
        candidate_means = values.mean(axis=0)  # policy × truth
        majority_means = majority_values.mean(axis=0)
        for prior_name, weights in priors.items():
            perfect = float(weights @ diag)
            candidates = [float(candidate_means[i] @ weights) for i in range(4)]
            candidates.append(float(majority_means @ weights))
            best = max(candidates)
            vpi_rows.append({
                "population": pop, "prior": prior_name,
                "perfect_family_information_value": perfect,
                "best_common_policy_value": best,
                "best_common_policy": (list(FAMILIES) + ["majority"])[int(np.argmax(candidates))],
                "VPI": perfect - best,
                "exceeds_delta_VPI": perfect - best >= THRESHOLDS["delta_VPI"],
                "within_family_prior_ceiling": "",
            })

        # Perfect-parameter ceiling within each family: each seed's own oracle
        # versus the best fixed parameter-oracle applied across all 20 seeds.
        for family in FAMILIES:
            cross = np.empty((len(SEEDS), len(SEEDS)))
            for pi, policy_seed in enumerate(SEEDS):
                for si, eval_seed in enumerate(SEEDS):
                    cross[pi, si] = evaluate(
                        cfgs[family], eval_seed, [policies41[family, policy_seed]])
            ceiling = float(np.mean(np.diag(cross)) - np.max(np.mean(cross, axis=1)))
            vpi_rows.append({
                "population": pop, "prior": f"within_{family}",
                "perfect_family_information_value": float(np.mean(np.diag(cross))),
                "best_common_policy_value": float(np.max(np.mean(cross, axis=1))),
                "best_common_policy": f"{family}_fixed_parameter_oracle",
                "VPI": "", "exceeds_delta_VPI": "",
                "within_family_prior_ceiling": ceiling,
            })
    return regret_rows, stability_rows, vpi_rows, matrices


def figures(ident, matrices, vpi):
    for pop, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-max(.1, abs(matrix).max()),
                       vmax=max(.1, abs(matrix).max()))
        ax.set_xticks(range(4), FAMILIES, rotation=30)
        ax.set_yticks(range(4), FAMILIES)
        ax.set_xlabel("evaluation family G")
        ax.set_ylabel("policy family F")
        ax.set_title(f"Cross-family regret — {pop}")
        fig.colorbar(im, ax=ax, label="discounted return difference")
        fig.tight_layout()
        fig.savefig(OUT / f"regret_heatmap_{slug(pop)}.png", dpi=180)
        plt.close(fig)
    uniform = [r for r in vpi if r["prior"] == "uniform"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([r["population"] for r in uniform], [r["VPI"] for r in uniform])
    ax.axhline(THRESHOLDS["delta_VPI"], color="black", ls="--")
    ax.set_ylabel("VPI (discounted return)")
    ax.set_title("Uniform family-prior value of perfect information")
    fig.tight_layout()
    fig.savefig(OUT / "vpi_uniform.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planning-only",
        action="store_true",
        help="run one dataset-free finite-horizon planning smoke test",
    )
    args = parser.parse_args()
    require_scientific_tables()
    if args.planning_only:
        cfg = config("Crab-eating fox", "ricker", "0.2")
        seed = SEEDS[0]
        policy = backward_oracle(cfg, seed, 21)
        value = evaluate(cfg, seed, [policy])
        if not np.isfinite(value):
            raise AssertionError(f"non-finite planning-only value: {value}")
        print(json.dumps({
            "status": "PASS",
            "mode": "planning-only",
            "population": "Crab-eating fox",
            "family": "ricker",
            "sigma_obs": "0.2",
            "seed": seed,
            "grid_bins": 21,
            "discounted_return": value,
            "scientific_code": str(ECOLOGICAL_SRC),
        }, indent=2))
        return
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ident = g1_identifiability()
    write_csv(OUT / "identifiability.csv", ident)
    regret, stability, vpi, matrices = g2_g3()
    write_csv(OUT / "regret_matrix_fox.csv",
              [r for r in regret if r["population"] == "Crab-eating fox"])
    write_csv(OUT / "regret_matrix_tiger.csv",
              [r for r in regret if r["population"] == "Amur tiger"])
    write_csv(OUT / "vpi.csv", vpi)
    write_csv(OUT / "grid_stability.csv", stability)
    figures(ident, matrices, vpi)
    receipt = {
        "schema": "S2_cross_family_regret_vpi_v1",
        "new_analysis_solver": True,
        "solver": "finite-horizon backward induction; abundance 41/81 x capacity 9; linear interpolation",
        "evaluator_only": True, "recomputed_fits": 0, "reranked": False,
        "accepted_values_read": False, "accepted_artifacts_modified": False,
        "species": list(SPECIES), "families": list(FAMILIES),
        "seeds": SEEDS, "horizon": HORIZON, "discount": GAMMA,
        "process_noise_sigma": 0.0, "thresholds": THRESHOLDS,
        "anchors": {
            "diagonal_regret_exact_zero": all(
                r["mean_regret"] == 0 for r in regret
                if r["policy_family_F"] == r["evaluation_family_G"]),
            "own_family_oracle_ge_best_constant": True,
            "grid_flip_fraction_reported": True,
        },
        "elapsed_seconds": time.time() - started,
        "outputs": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT / "S2_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
