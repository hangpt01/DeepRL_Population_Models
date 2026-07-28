#!/usr/bin/env python3
"""Evaluator-only constant-policy reward screening with S1 parity anchor."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

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
OUT = Path(
    os.environ.get("DEEPRL_REWARD_SCREEN_OUTPUT", FOLLOWUPS_OUTPUT / "reward_screen")
)
sys.path[:0] = [str(ECOLOGICAL_SCRIPTS), str(ECOLOGICAL_SRC)]

import run_real_manifest_row as runner  # noqa: E402
from real_ecology_benchmark.config import load_config  # noqa: E402
from real_ecology_benchmark.dataset import load_private  # noqa: E402
from real_ecology_benchmark.envs import make_env  # noqa: E402
from real_ecology_benchmark.pipeline import ensure_dataset  # noqa: E402
from real_ecology_benchmark.public_surrogate import (  # noqa: E402
    _episode_split,
    fit_public_surrogate,
)

SEEDS = [b + i for b in (7001, 7051, 7101, 7151, 7201) for i in range(4)]
GAMMA = 0.95
HARVEST_ACTIONS = {1, 2}


def identities():
    manifest = P10_PACKAGE / "manifests" / "moor_p10_plan_24.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        cfg = load_config(str(P10_PACKAGE / "configs" / "moor_ricker_p10.yaml"))
        cfg, _cell, _eval = runner.apply_row_config(cfg, row, P10 / "moor")
        cfg.validate()
        yield row, cfg


def trajectory(cfg, action):
    rows = []
    for seed in SEEDS:
        env = make_env(cfg.environment)
        reset = env.reset(seed)
        state = float(reset.evaluator_info["state"])
        for t in range(50):
            result = env.step(action)
            following = float(result.evaluator_info["state"])
            rows.append((seed, t, state, following, float(env.actions[action].cost), action))
            state = following
    return np.asarray(rows, dtype=np.float64)


def score(states, k_ref, s_safe):
    seed = states[:, 0].astype(int)
    t = states[:, 1].astype(int)
    current_state = states[:, 2]
    following = states[:, 3]
    cost = states[:, 4]
    phi_now = current_state / (current_state + k_ref)
    phi_next = following / (following + k_ref)
    current = phi_next - cost - 10.0 * (following <= s_safe)
    option_a = phi_next - cost - 10.0 * np.maximum(0.0, (s_safe - following) / s_safe)
    option_c = current + GAMMA * phi_next - phi_now
    output = {}
    for name, reward in (("Current", current), ("OptionA", option_a), ("OptionC", option_c)):
        per_episode = [
            float(np.sum((GAMMA ** t[seed == value]) * reward[seed == value]))
            for value in np.unique(seed)
        ]
        output[name] = {
            "return_mean": float(np.mean(per_episode)),
            "return_sd": float(np.std(per_episode, ddof=1)),
        }
    return output


def surrogate_option_a(cfg):
    dataset = ensure_dataset(cfg, regenerate=False)
    private = load_private(cfg.dataset.private_output)
    private.validate(dataset)
    k_ref = float(cfg.environment.K_ref)
    s_safe = float(cfg.environment.safety_threshold)
    rewards = (
        private.next_states / (private.next_states + k_ref)
        - dataset.costs
        - 10.0 * np.maximum(0.0, (s_safe - private.next_states) / s_safe)
    )
    modified = replace(dataset, rewards=np.asarray(rewards, dtype=np.float64))
    seed = cfg.seed + 20_000
    surrogate = fit_public_surrogate(modified, seed=seed)
    fit_mask, holdout = _episode_split(modified, seed)
    prediction = surrogate.feature_spec.design(modified) @ surrogate.reward_coefficients
    y = rewards[holdout]
    p = prediction[holdout]
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((p - y) ** 2)) / denom if denom > 0 else None
    unsafe = private.next_states <= s_safe

    def err(mask):
        e = prediction[mask] - rewards[mask]
        return {
            "n": int(mask.sum()),
            "mean_signed_error": float(e.mean()) if mask.any() else None,
            "rmse": float(np.sqrt(np.mean(e * e))) if mask.any() else None,
        }

    return {"holdout_R2": r2, "safe": err(~unsafe), "unsafe": err(unsafe)}


def main():
    require_scientific_tables()
    OUT.mkdir(parents=True, exist_ok=True)
    s1_path = REPLAY / "derived/S1_constant_action_sweep.csv"
    if not s1_path.exists():
        raise FileNotFoundError(f"NOT FOUND: {s1_path}")
    with s1_path.open(newline="", encoding="utf-8") as handle:
        s1 = {
            (r["population"], r["environment"], r["sigma_obs"], int(r["action"])): float(r["return_mean"])
            for r in csv.DictReader(handle)
        }
    all_states = {}
    scored = []
    cfgs = []
    # Phase 1: regenerate and validate Current before any alternative reward verdict.
    for row, cfg in identities():
        key3 = (row["population"], row["environment"], row["sigma_obs"])
        cfgs.append((row, cfg))
        for action in range(11):
            states = trajectory(cfg, action)
            all_states[(*key3, action)] = states
            values = score(
                states, float(cfg.environment.K_ref), float(cfg.environment.safety_threshold)
            )
            expected = s1.get((*key3, action))
            if expected is None:
                raise FileNotFoundError(f"NOT FOUND: S1 identity {(*key3, action)}")
            delta = abs(values["Current"]["return_mean"] - expected)
            if delta > 1e-9:
                raise AssertionError(f"Current reward parity failed {(*key3, action)} delta={delta}")
            scored.append({**dict(zip(("population", "environment", "sigma_obs"), key3)),
                           "action": action, "current_parity_delta": delta, **values})
    # Raw state archive is a new artifact, never an accepted input.
    np.savez_compressed(
        OUT / "constant_true_state_trajectories.npz",
        **{
            f"{r[0].lower().replace(' ', '_')}__{r[1]}__s{r[2].replace('.', 'p')}__a{r[3]}": arr
            for r, arr in all_states.items()
        },
    )
    flat = []
    for r in scored:
        for reward in ("Current", "OptionA", "OptionC"):
            flat.append({
                "population": r["population"], "environment": r["environment"],
                "sigma_obs": r["sigma_obs"], "action": r["action"], "reward": reward,
                **r[reward], "current_parity_delta": r["current_parity_delta"],
            })
    with (OUT / "constant_reward_screen.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    verdicts = []
    for row, cfg in cfgs:
        subset = [r for r in scored if r["population"] == row["population"]
                  and r["environment"] == row["environment"]
                  and r["sigma_obs"] == row["sigma_obs"]]
        best = {
            name: max(subset, key=lambda x: x[name]["return_mean"])["action"]
            for name in ("Current", "OptionA", "OptionC")
        }
        spread = {
            name: max(x[name]["return_mean"] for x in subset)
            - min(x[name]["return_mean"] for x in subset)
            for name in ("Current", "OptionA", "OptionC")
        }
        verdicts.append({
            "population": row["population"], "environment": row["environment"],
            "sigma_obs": row["sigma_obs"], "argmax": best,
            "argmax_moves_Current_to_A": best["Current"] != best["OptionA"],
            "FLAG_option_C_moves_argmax": best["Current"] != best["OptionC"],
            "best_option_A_is_harvest": best["OptionA"] in HARVEST_ACTIONS,
            "spread": spread,
            "surrogate_option_A": surrogate_option_a(cfg),
        })
    receipt = {
        "schema": "reward_screening_v1",
        "reference_policies_only": True,
        "current_parity_max_abs_diff": max(r["current_parity_delta"] for r in scored),
        "current_parity_pass": True,
        "cells": verdicts,
        "option_C_any_argmax_move": any(v["FLAG_option_C_moves_argmax"] for v in verdicts),
        "vulture_option_A_all_nonharvest": all(
            not v["best_option_A_is_harvest"]
            for v in verdicts if v["population"] == "Egyptian vulture"
        ),
        "accepted_artifacts_modified": False,
        "reranked": False,
    }
    (OUT / "JOB2_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
