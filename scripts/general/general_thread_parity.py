#!/usr/bin/env python3
"""Return-blind default-vs-one-thread numerical parity canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tracks" / "general"))
sys.path.insert(0, str(ROOT))

from real_ecology_benchmark.config import ModelConfig, PlannerConfig
from real_ecology_benchmark.methods.bamcts import BAMCTSPolicy
from real_ecology_benchmark.methods.ensemble_value_disagreement import (
    EnsembleValueDisagreementPolicy,
)
from real_ecology_benchmark.methods.ogsrl import OGSRLPolicy
from real_ecology_benchmark.methods.refplan import RefPlanPolicy
from tests.general.real.test_general_paper_mechanisms import _hidden_fixture


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _update_hash(digest, value, seen=None):
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    elif isinstance(value, (str, int, float, bool, np.number, type(None))):
        digest.update(repr(value).encode())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            if key not in {"rng", "training_history", "last_diagnostics"}:
                _update_hash(digest, key, seen)
                _update_hash(digest, value[key], seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _update_hash(digest, item, seen)
    elif hasattr(value, "__dict__"):
        _update_hash(digest, vars(value), seen)


def _numeric_vector(value, values=None, seen=None):
    if values is None:
        values = []
    if seen is None:
        seen = set()
    if id(value) in seen:
        return values
    seen.add(id(value))
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        values.extend(np.asarray(value, dtype=np.float64).reshape(-1).tolist())
    elif isinstance(value, (int, float, bool, np.number)):
        values.append(float(value))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            if key not in {"rng", "training_history", "last_diagnostics"}:
                _numeric_vector(value[key], values, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _numeric_vector(item, values, seen)
    elif hasattr(value, "__dict__"):
        _numeric_vector(vars(value), values, seen)
    return values


def worker(artifact_dir: str | None = None) -> dict:
    _env, dataset, _private, context, _filter_cfg, factory, cache = _hidden_fixture()
    model = ModelConfig(ensemble_size=5)
    planner = PlannerConfig(horizon=3, sequences=24, particles=8, ogsrl_cost_horizon=4)
    policies = {
        "refplan": RefPlanPolicy(context, model, planner, seed=7),
        "ogsrl": OGSRLPolicy(context, model, planner, seed=7),
        "bamcts": BAMCTSPolicy(context, model, planner, seed=7, simulations=24, depth=3),
        "ensemble_value_disagreement_pessimism": EnsembleValueDisagreementPolicy(
            context, model, planner, seed=7, ensemble_size=6
        ),
    }
    result = {"threads": {name: os.environ.get(name, "unset") for name in THREAD_VARS}}
    belief = factory().reset(float(dataset.observations[0]), 55)
    for name, policy in policies.items():
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        policy.fit(dataset, cache)
        action = int(policy.act(belief, belief.observation))
        digest = hashlib.sha256()
        _update_hash(digest, policy)
        result[name] = {
            "hash": digest.hexdigest(),
            "action": action,
            "wall_seconds": time.perf_counter() - start_wall,
            "cpu_seconds": time.process_time() - start_cpu,
        }
        if artifact_dir is not None:
            np.save(Path(artifact_dir) / f"{name}.npy", np.asarray(_numeric_vector(policy)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--artifact-dir")
    parser.add_argument(
        "--out",
        default=str(ROOT / "real_ecology_runs" / "general_adequacy_phase2d" / "thread_parity.json"),
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.artifact_dir)))
        return
    base = os.environ.copy()
    base["PYTHONPATH"] = str(ROOT / "src" / "tracks" / "general")
    default_env = base.copy()
    for name in THREAD_VARS:
        default_env.pop(name, None)
    one_env = base.copy()
    for name in THREAD_VARS:
        one_env[name] = "1"
    with tempfile.TemporaryDirectory() as temporary:
        default_dir = Path(temporary) / "default"; default_dir.mkdir()
        one_dir = Path(temporary) / "one"; one_dir.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--artifact-dir"]
        default = json.loads(subprocess.check_output(
            command + [str(default_dir)], env=default_env, text=True
        ))
        one = json.loads(subprocess.check_output(
            command + [str(one_dir)], env=one_env, text=True
        ))
        comparisons = {}
        for name in (
            "refplan", "ogsrl", "bamcts", "ensemble_value_disagreement_pessimism"
        ):
            left = np.load(default_dir / f"{name}.npy")
            right = np.load(one_dir / f"{name}.npy")
            delta = np.abs(left - right)
            comparisons[name] = {
                "hash_equal": default[name]["hash"] == one[name]["hash"],
                "action_equal": default[name]["action"] == one[name]["action"],
                "max_abs_numeric_difference": float(np.max(delta)) if len(delta) else 0.0,
                "numeric_byte_equal": bool(np.array_equal(left, right)),
                "allclose_atol_1e-12": bool(np.allclose(left, right, rtol=1e-12, atol=1e-12)),
            }
    result = {
        "default": default,
        "one_thread": one,
        "comparisons": comparisons,
        "passes": all(v["allclose_atol_1e-12"] and v["action_equal"] for v in comparisons.values()),
        "resource_accounting": {
            "task_hours": "sum task wall time",
            "actual_cpu_hours": "sum measured process CPU time / 3600",
            "allocated_core_hours": "sum task wall time * 1 allocated core / 3600",
            "elapsed_wall": "schedule makespan, queue delay separate",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
