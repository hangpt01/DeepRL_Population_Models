#!/usr/bin/env python3
"""H14 parity-gated Tier-B replay with visited-state surrogate reward logging."""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import FOLLOWUPS_OUTPUT, REPO_ROOT, require_scientific_tables  # noqa: E402

BASE_DIR = REPO_ROOT / "scripts" / "diagnostics" / "replay"
OUT = Path(os.environ.get("DEEPRL_H14_OUTPUT", FOLLOWUPS_OUTPUT / "H14"))
sys.path.insert(0, str(BASE_DIR))
import run_tier_b_replay as base  # noqa: E402

base.REPLAY = OUT
METHODS = ("refplan", "ogsrl", "bamcts")
_SURROGATE = None
_POP_ID = None

_original_build = base.build_policy


def build_policy(cell_key: str, method: str):
    global _SURROGATE, _POP_ID
    result = _original_build(cell_key, method)
    cfg, _policy, _factory, dataset, expected_path, *_rest = result
    surrogate, loaded_path, status = base._load_or_fit_public_surrogate(cfg, dataset)
    if status != "loaded" or Path(loaded_path) != Path(expected_path):
        raise AssertionError((status, loaded_path, expected_path))
    context = base._hidden_method_context(cfg, dataset, surrogate)
    _SURROGATE = surrogate
    _POP_ID = context.pop_id
    return result


base.build_policy = build_policy
_original_reset = base.TierBLogger.on_reset
_original_step = base.TierBLogger.on_step


def on_reset(self, seed: int) -> None:
    _original_reset(self, seed)
    self._h14_previous_observation = None


def _env_rng_states(env):
    return {
        name: copy.deepcopy(rng.bit_generator.state)
        for name, rng in getattr(env, "_rngs", {}).items()
        if isinstance(rng, np.random.Generator)
    }


def on_step(self, env, result) -> None:
    _original_step(self, env, result)
    row = self.rows[-1]
    policy_before = base._rng_states(self.policy)
    env_before = _env_rng_states(env)
    previous = (
        float(row["obs_t"])
        if self._h14_previous_observation is None
        else float(self._h14_previous_observation)
    )
    current = float(row["obs_t"])
    following = float(result.observation)
    reward, _risk = _SURROGATE.predict(
        np.asarray([previous], dtype=np.float64),
        np.asarray([current], dtype=np.float64),
        np.asarray([following], dtype=np.float64),
        np.asarray([int(row["action_t"])], dtype=np.int64),
        np.asarray(
            [min(int(row["t"]), _SURROGATE.feature_spec.public_horizon - 1)],
            dtype=np.int64,
        ),
        np.asarray([_POP_ID]),
    )
    row["surrogate_reward_t"] = float(reward[0])
    self._h14_previous_observation = following
    if base._rng_states(self.policy) != policy_before or _env_rng_states(env) != env_before:
        raise AssertionError("H14 surrogate logging consumed RNG")


base.TierBLogger.on_reset = on_reset
base.TierBLogger.on_step = on_step


def _stratified_m13(rows):
    err = np.asarray(
        [r["surrogate_reward_t"] - r["reward_t"] for r in rows], dtype=np.float64
    )
    unsafe = np.asarray([bool(r["unsafe_next"]) for r in rows])

    def part(mask):
        x = err[mask]
        return {
            "n": int(len(x)),
            "mean_signed_error": float(x.mean()) if len(x) else None,
            "rmse": float(np.sqrt(np.mean(x * x))) if len(x) else None,
        }

    return {"overall": part(np.ones(len(err), bool)), "safe": part(~unsafe), "unsafe": part(unsafe)}


def write_outputs(result):
    label = base.write_outputs(result)
    rows = result["logger"].rows
    values = np.asarray([r["surrogate_reward_t"] for r in rows], dtype=np.float64)
    npz_path = OUT / "logs" / f"{label}.npz"
    with np.load(npz_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["surrogate_reward_t"] = values
    np.savez_compressed(npz_path, **arrays)

    csv_path = OUT / "logs" / f"{label}.scalars.csv.gz"
    with gzip.open(csv_path, "rt", newline="") as handle:
        data = list(csv.reader(handle))
    data[0].append("surrogate_reward_t")
    for row, value in zip(data[1:], values):
        row.append(repr(float(value)))
    with gzip.open(csv_path, "wt", newline="") as handle:
        csv.writer(handle).writerows(data)

    derived_path = OUT / "derived" / f"{label}.json"
    derived = json.loads(derived_path.read_text())
    derived["M13_surrogate_error_visited"] = _stratified_m13(rows)
    derived_path.write_text(json.dumps(derived, indent=2) + "\n")

    parity_path = OUT / "parity" / f"PARITY_{label}.json"
    receipt = json.loads(parity_path.read_text())
    receipt["logging"]["surrogate_reward_t"] = "present"
    receipt["logging"]["surrogate_predict_rng_unchanged"] = True
    receipt["M13_surrogate_error_visited"] = derived["M13_surrogate_error_visited"]
    parity_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=tuple(base.CELLS))
    parser.add_argument("method", choices=METHODS)
    args = parser.parse_args()
    require_scientific_tables()
    if base.sha(base.ACCEPTED_CSV) != base.ACCEPTED_SHA:
        raise AssertionError("accepted controlling CSV hash changed")
    result = base.run_cell(args.cell, args.method)
    if result["parity"]["max_abs_diff"] != 0.0:
        raise AssertionError(f"H14 parity anchor failed: {result['parity']}")
    label = write_outputs(result)
    print(json.dumps({
        "label": label,
        "cell": args.cell,
        "method": args.method,
        "parity": result["parity"]["verdict"],
        "max_abs_diff": result["parity"]["max_abs_diff"],
        "elapsed_seconds": result["elapsed_seconds"],
        "recomputed_fits": 0,
        "M13": _stratified_m13(result["logger"].rows),
    }, indent=2))


if __name__ == "__main__":
    main()
