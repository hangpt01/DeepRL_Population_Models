#!/usr/bin/env python3
"""Run one registered Arm O canary task on the pinned CPU.

The runner points at a local public-data copy and a non-truth existence sentinel,
patches evaluator-only private loading to fail closed, and records deployed actions
and per-step rewards externally.  It never imports or opens a derived Arm T file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DOC_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs/i2b_fasttrack_integration_canary_20260808"
REGISTRATION = DOC_ROOT / "ARM_O_CANARY_REGISTRATION.json"
EXPECTED_SEEDS = [7001, 7051, 7101, 7151, 7201]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


class RecordingEnv:
    def __init__(self, wrapped: Any, recorder: list[dict[str, Any]]):
        self._wrapped = wrapped
        self._recorder = recorder
        self._current: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def reset(self, seed: int):
        self._current = {"seed": int(seed), "actions": [], "rewards": [], "true_rewards": []}
        self._recorder.append(self._current)
        return self._wrapped.reset(seed)

    def step(self, action: int):
        if self._current is None:
            raise RuntimeError("step before reset")
        result = self._wrapped.step(action)
        self._current["actions"].append(int(action))
        self._current["rewards"].append(float(result.reward))
        self._current["true_rewards"].append(float(result.evaluator_info["reward_true"]))
        return result


def compare_episodes(
    new_path: Path,
    accepted_path: Path,
    trajectories: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    with new_path.open(newline="", encoding="utf-8") as stream:
        new_rows = list(csv.DictReader(stream))
    with accepted_path.open(newline="", encoding="utf-8") as stream:
        accepted_rows = list(csv.DictReader(stream))
    if len(new_rows) != 20 or len(accepted_rows) != 20 or len(trajectories) != 20:
        raise RuntimeError("canary/accepted/recorded episode cardinality is not 20")
    identity_fields = ("episode", "seed", "block_seed")
    timing_fields = frozenset({"filter_seconds", "planner_seconds"})
    path_fields = frozenset({"data_table"})
    action_fields = ("action_entropy", "economic_cost") + tuple(
        f"danger_action_{index}_fraction" for index in range(11)
    )
    event_fields = (
        "collapse_entry", "collapse_entries", "collapse_entry_timestep",
        "unsafe_fraction", "mvp_fraction", "mvp_breach", "min_true_state",
        "final_true_state", "persistence", "n_steps",
    )
    return_fields = ("operational_return", "true_return")
    numeric_deltas: dict[str, float] = {}
    exact_mismatches: list[dict[str, Any]] = []
    for index, (new, accepted, trajectory) in enumerate(zip(new_rows, accepted_rows, trajectories)):
        for field in identity_fields:
            if new[field] != accepted[field]:
                exact_mismatches.append({"episode": index, "field": field, "new": new[field], "accepted": accepted[field]})
        for field in new:
            if field in timing_fields or field in path_fields:
                continue
            if field not in accepted:
                exact_mismatches.append({"episode": index, "field": field, "reason": "missing accepted field"})
                continue
            try:
                delta = abs(float(new[field]) - float(accepted[field]))
            except ValueError:
                if new[field] != accepted[field]:
                    exact_mismatches.append({"episode": index, "field": field, "new": new[field], "accepted": accepted[field]})
            else:
                numeric_deltas[field] = max(numeric_deltas.get(field, 0.0), delta)
        discount = 0.95
        reconstructed = float(sum((discount ** t) * r for t, r in enumerate(trajectory["rewards"])))
        reconstructed_true = float(sum((discount ** t) * r for t, r in enumerate(trajectory["true_rewards"])))
        for field, value in (("operational_return", reconstructed), ("true_return", reconstructed_true)):
            delta = abs(value - float(new[field]))
            numeric_deltas[f"reconstructed_{field}"] = max(
                numeric_deltas.get(f"reconstructed_{field}", 0.0), delta
            )
    parity_fields = set(numeric_deltas) - {
        "reconstructed_operational_return", "reconstructed_true_return"
    }
    failed_numeric = {field: numeric_deltas[field] for field in sorted(parity_fields) if numeric_deltas[field] > tolerance}
    failed_reconstruction = {
        field: delta for field, delta in numeric_deltas.items()
        if field.startswith("reconstructed_") and delta > tolerance
    }
    return {
        "episode_count": 20,
        "identity_fields": list(identity_fields),
        "return_fields": list(return_fields),
        "action_summary_fields": list(action_fields),
        "event_fields": list(event_fields),
        "excluded_nonscientific_fields": sorted(timing_fields | path_fields),
        "historical_full_action_sequences_available": False,
        "new_full_action_sequences_recorded": True,
        "tolerance": tolerance,
        "maximum_absolute_deltas": dict(sorted(numeric_deltas.items())),
        "exact_mismatches": exact_mismatches,
        "failed_numeric_fields": failed_numeric,
        "failed_return_reconstruction": failed_reconstruction,
        "result": "PASS" if not exact_mismatches and not failed_numeric and not failed_reconstruction else "FAIL",
    }


def run(task_index: int) -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    tasks = registration["tasks"]
    if task_index < 0 or task_index >= len(tasks):
        raise RuntimeError("task index outside registered 0..11 range")
    task = tasks[task_index]
    task_root = OUTPUT_ROOT / "arm_o_tasks" / task["task_id"]
    if task_root.exists():
        raise RuntimeError(f"task namespace already exists: {task_root}")
    task_root.mkdir(parents=True)
    started = time.time()
    receipt_path = task_root / "task_receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": "i2b_arm_o_task_receipt_v1",
        "task": task,
        "started_unix": started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": platform.node(),
        "cpu_model": cpu_model(),
        "python": sys.version,
        "environment": {name: os.environ.get(name) for name in (
            "LC_ALL", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "arm": "O",
        "derived_artifact_access": False,
        "original_truth_archive_access": False,
    }
    try:
        if "8452Y" not in receipt["cpu_model"]:
            raise RuntimeError(f"pinned CPU unavailable on allocated node: {receipt['cpu_model']}")
        shared = Path(task["shared_input_root"])
        public_path = shared / "public.npz"
        sentinel = shared / "private_archive_access_disabled.json"
        if sha256_file(public_path) != task["public_file_sha256"]:
            raise RuntimeError("task-local public input hash mismatch")
        if sha256_file(Path(task["accepted_episodes_path"])) != task["accepted_episodes_sha256"]:
            raise RuntimeError("accepted episode anchor hash mismatch")
        expected_config_sha = registration["config_identities"].get(task["config_path"])
        if not expected_config_sha or sha256_file(Path(task["config_path"])) != expected_config_sha:
            raise RuntimeError("canary configuration hash mismatch")
        if not sentinel.exists():
            raise RuntimeError("private-access sentinel missing")
        track_src = ROOT / task["track_source"]
        sys.path.insert(0, str(track_src))
        from real_ecology_benchmark.config import load_config, real_environment_like
        from real_ecology_benchmark import evaluator as evaluator_module
        from real_ecology_benchmark import pipeline

        if sha256_file(ROOT / task["source_file"]) != task["source_file_sha256"]:
            raise RuntimeError("method source hash mismatch")
        cfg = load_config(task["config_path"])
        cfg.environment = real_environment_like(cfg.environment, task["population"], "allee")
        cfg.environment = replace(
            cfg.environment,
            observation_noise_sigma=0.2,
            reward_mode="safe",
            expose_rk="hidden",
        )
        cfg.dataset.output = str(public_path)
        cfg.dataset.private_output = str(sentinel)
        cfg.evaluation.output_dir = str(task_root / "evaluation")
        if task["fit_cache_dir"]:
            cfg.faithful.fit_cache_dir = task["fit_cache_dir"]
        cfg.validate()
        if cfg.evaluation.seeds != EXPECTED_SEEDS or cfg.evaluation.episodes_per_seed != 4:
            raise RuntimeError("evaluation identity mismatch")
        if cfg.evaluation.horizon != 50 or cfg.evaluation.discount != 0.95:
            raise RuntimeError("evaluation horizon/discount mismatch")
        if cfg.environment.observation_noise_sigma != 0.2:
            raise RuntimeError("Arm O sigma mismatch")

        def deny_private(*_args: Any, **_kwargs: Any):
            raise FileNotFoundError("I2B Arm O canary denies all private archive loading")

        trajectories: list[dict[str, Any]] = []
        original_make_env = evaluator_module.make_env

        class RecordingEvaluator(evaluator_module.ContinuousEvaluator):
            def run(self, policy):
                evaluator_module.make_env = lambda config: RecordingEnv(original_make_env(config), trajectories)
                try:
                    return super().run(policy)
                finally:
                    evaluator_module.make_env = original_make_env

        pipeline.load_private = deny_private
        pipeline.ContinuousEvaluator = RecordingEvaluator
        summary = pipeline.run_method(task["method"], cfg, task["filter"], regenerate=False)
        new_episodes = Path(summary["output_dir"]) / "episodes.csv"
        trajectories_path = task_root / "deployed_trajectories.json"
        strict_write_json(trajectories_path, {
            "arm": "O", "task_id": task["task_id"], "episodes": trajectories,
            "contains_exact_state": False, "contains_private_truth_archive_fields": False,
        })
        parity = compare_episodes(
            new_episodes,
            Path(task["accepted_episodes_path"]),
            trajectories,
            tolerance=float(registration["parity"]["absolute_tolerance"]),
        )
        strict_write_json(task_root / "parity.json", parity)
        receipt.update({
            "ended_unix": time.time(), "duration_seconds": time.time() - started,
            "exit_status": 0 if parity["result"] == "PASS" else 2,
            "summary_path": str(Path(summary["output_dir"]) / "summary.json"),
            "episodes_path": str(new_episodes),
            "episodes_sha256": sha256_file(new_episodes),
            "accepted_episodes_sha256_observed": sha256_file(Path(task["accepted_episodes_path"])),
            "config_sha256": expected_config_sha,
            "deployed_trajectories_path": str(trajectories_path),
            "parity": parity,
            "fit_cache_hits": summary.get("fit_diagnostics", {}).get("fit_cache_hits"),
            "fit_cache_misses": summary.get("fit_diagnostics", {}).get("fit_cache_misses"),
            "public_surrogate_cache_status": summary.get("public_surrogate_cache_status"),
        })
        strict_write_json(receipt_path, receipt)
        if parity["result"] != "PASS":
            raise SystemExit(2)
    except BaseException as exc:
        if not receipt_path.exists():
            receipt.update({
                "ended_unix": time.time(), "duration_seconds": time.time() - started,
                "exit_status": int(getattr(exc, "code", 1) or 1),
                "failure": str(exc), "traceback": traceback.format_exc(),
            })
            strict_write_json(receipt_path, receipt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_index", type=int)
    args = parser.parse_args()
    run(args.task_index)


if __name__ == "__main__":
    main()
