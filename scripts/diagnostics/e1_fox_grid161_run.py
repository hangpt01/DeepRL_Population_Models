#!/usr/bin/env python3
"""Registered fox-only E1 grid-161 scientific execution and audit tooling."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
ECOLOGY_SRC = REPO_ROOT / "src" / "tracks" / "ecological"
for entry in (str(SCRIPT_DIR), str(REPO_ROOT), str(ECOLOGY_SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import e1_phase2_core_validation as core  # noqa: E402
import e1_phase5_remediated_validation as p5  # noqa: E402
from e1_phase1_parity import LIKELIHOOD_FLOOR  # noqa: E402
from real_ecology_benchmark.beliefs import OracleStateFilter  # noqa: E402
from real_ecology_benchmark.faithful_pomdp import CandidateBelief  # noqa: E402
from real_ecology_benchmark.types import BeliefState, PublicTransition  # noqa: E402


SCHEMA = "e1_fox_only_grid161_scientific_v1"
FOX = "Crab-eating fox"
SIGMAS = (0.1, 0.2)
ARMS = ("A1", "A2", "A3", "A4")
BLOCK_SEEDS = (7001, 7051, 7101, 7151, 7201)
EPISODES_PER_SEED = 4
DERIVED_EPISODES = tuple(seed + local for seed in BLOCK_SEEDS for local in range(EPISODES_PER_SEED))
STATE_BINS = 161
OBSERVATION_BINS = 41
CAPACITY_BINS = 9
OBSERVATION_BRANCHES = 7
HORIZON = 5
BELIEF_POINTS = 32
EVALUATION_HORIZON = 50
GAMMA = 0.95
EXPECTED_PYTHON = "3.10.14"
EXPECTED_NUMPY = "2.2.6"
EXPECTED_PARTITION = "comp"
EXPECTED_CONSTRAINT = "EPYC9534"
BASE_OUTPUT = REPO_ROOT / "outputs" / "e1_fox_only_grid161"
RUN_CONTROL_NAME = "RUN_CONTROL.json"
REGISTRATION_PREFIX = "E1_RUN_REGISTRATION_"
TASKS = tuple(
    (core.cell_id(FOX, sigma), arm)
    for sigma in SIGMAS
    for arm in ARMS
)
REGISTERED_DEVIATION = {
    "state_bins": STATE_BINS,
    "replaces_frozen_state_bins": 41,
    "reason": (
        "The independently audited V6 diagnostic demonstrated genuine policy "
        "sensitivity and inadequate state resolution at 41 bins. The finest predefined "
        "diagnostic grid, 161 bins, is selected prospectively before scientific returns."
    ),
    "conditions": [
        "E1 is conditional on the 161-bin numerical convention.",
        "It is not directly comparable with accepted PLUS/MOOR on planner budget.",
        "It does not establish grid convergence.",
        "The higher-dimensional belief representation may materially increase compute.",
    ],
}
PREREGISTERED_STATEMENTS = [
    "No directional prediction is made between A2−A4 and A1−A2; both are preregistered estimands.",
    "A1−A3 is predicted to have magnitude below 0.05 because registered dynamics are deterministic and the initial state is known.",
    "All conclusions will be conditional on fox, Ricker, state_bins=161 and PBVI horizon-5/32-belief planning.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_bytes_new(path: Path, data: bytes, mode: int = 0o644) -> None:
    """Create a file without replacement, using a same-directory atomic hard link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output collision: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    os.chmod(path, mode)


def write_json_new(path: Path, value: Any, mode: int = 0o644) -> None:
    write_bytes_new(path, _json_bytes(value), mode)


def write_text_new(path: Path, value: str, mode: int = 0o644) -> None:
    write_bytes_new(path, value.encode("utf-8"), mode)


def command_output(args: list[str], check: bool = True) -> str:
    result = subprocess.run(
        args, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {args}\n{result.stdout}")
    return result.stdout


def assert_environment() -> dict[str, Any]:
    actual = (platform.python_version(), np.__version__)
    if actual != (EXPECTED_PYTHON, EXPECTED_NUMPY):
        raise AssertionError(f"wrong environment: {actual}")
    packages = sorted(
        (distribution.metadata.get("Name", ""), distribution.version)
        for distribution in importlib.metadata.distributions()
    )
    numpy_build = np.__config__.show(mode="dicts")
    manifest = {
        "python": actual[0],
        "numpy": actual[1],
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "numpy_build": numpy_build,
    }
    manifest["environment_digest"] = canonical_digest(manifest)
    return manifest


def cpu_info() -> dict[str, Any]:
    models = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                models.append(line.split(":", 1)[1].strip())
    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "cpu_models": sorted(set(models)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_constraint": os.environ.get("SLURM_JOB_CONSTRAINTS"),
    }


def git_snapshot() -> dict[str, Any]:
    status = command_output(["git", "status", "--short", "--untracked-files=all"])
    modified = [line for line in status.splitlines() if line]
    tracked_src = command_output(["git", "ls-files", "src/tracks"]).splitlines()
    tracked_hashes = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in tracked_src
        if (REPO_ROOT / relative).is_file()
    }
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"]).strip(),
        "status_lines": modified,
        "modified_or_untracked_files": [line[3:] for line in modified],
        "src_tracks_git_diff": command_output(
            ["git", "diff", "--name-only", "--", "src/tracks"]
        ).splitlines(),
        "tracked_src_tracks_digest": canonical_digest(tracked_hashes),
        "tracked_src_tracks_files": tracked_hashes,
    }


def planner_config():
    config = replace(
        core.planner_config(),
        state_bins=STATE_BINS,
        capacity_bins=CAPACITY_BINS,
        observation_bins=OBSERVATION_BINS,
        observation_branches=OBSERVATION_BRANCHES,
        horizon=HORIZON,
        belief_points=BELIEF_POINTS,
    )
    config.validate()
    return config


def _new_science_metrics() -> dict[str, Any]:
    return {
        "predict_calls": 0,
        "transition_matrix_calls": 0,
        "representative_observation_calls": 0,
        "distinct_branch_histogram": Counter(),
        "returned_branch_histogram": Counter(),
    }


class ScientificTruePOMDP(p5.RemediatedTrueStatePOMDP):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.science = _new_science_metrics()

    def transition_matrix(self, capacity: float, action: int) -> np.ndarray:
        self.science["transition_matrix_calls"] += 1
        return super().transition_matrix(capacity, action)

    def predict(self, belief: CandidateBelief, action: int) -> np.ndarray:
        self.science["predict_calls"] += 1
        return super().predict(belief, action)

    def representative_observations(
        self, predicted: np.ndarray, branch_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        observations, weights = super().representative_observations(predicted, branch_count)
        distinct = len(np.unique(np.asarray(observations, dtype=np.float64)))
        self.science["representative_observation_calls"] += 1
        self.science["distinct_branch_histogram"][distinct] += 1
        self.science["returned_branch_histogram"][len(observations)] += 1
        return observations, weights


class ScientificNoisyPOMDP(p5.RemediatedNoisyStatePOMDP):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.science = _new_science_metrics()

    def transition_matrix(self, capacity: float, action: int) -> np.ndarray:
        self.science["transition_matrix_calls"] += 1
        return super().transition_matrix(capacity, action)

    def predict(self, belief: CandidateBelief, action: int) -> np.ndarray:
        self.science["predict_calls"] += 1
        return super().predict(belief, action)

    def representative_observations(
        self, predicted: np.ndarray, branch_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        observations, weights = super().representative_observations(predicted, branch_count)
        distinct = len(np.unique(np.asarray(observations, dtype=np.float64)))
        self.science["representative_observation_calls"] += 1
        self.science["distinct_branch_histogram"][distinct] += 1
        self.science["returned_branch_histogram"][len(observations)] += 1
        return observations, weights


def build_cell(sigma: float, requested_arms: tuple[str, ...] = ARMS) -> dict[str, Any]:
    cid = core.cell_id(FOX, sigma)
    config = planner_config()
    env_cfg = core.real_environment(
        FOX,
        "ricker",
        observation_noise_sigma=sigma,
        process_noise_sigma=0.0,
        initial_log_sigma=0.0,
        low_start_probability=0.0,
        horizon=EVALUATION_HORIZON,
        expose_rk="full",
        reward_mode="safe",
        collapse_penalty=10.0,
        safety_penalty_mode="occupancy",
        alpha=1.0,
    )
    actions = core.real_action_table(FOX, "ricker", env_cfg.data_dir)
    registered = core.registered_model(env_cfg, actions)
    fitted, fitted_provenance = core.load_cached_model(core.FIT_CACHE_KEYS[(FOX, sigma)])
    registered_context = core.method_context(actions, sigma, registered.survey_scale)
    fitted_context = core.method_context(actions, sigma, fitted.survey_scale)
    planning_seed = core.PLANNING_SEEDS[cid]
    policies: dict[str, Any] = {}
    for arm in requested_arms:
        if arm == "A1":
            pomdp = ScientificTruePOMDP(
                registered, registered_context, config, planning_seed, env_cfg, arm
            )
            policy = p5.RemediatedTrueStatePolicy(arm, pomdp, planning_seed)
        elif arm == "A2":
            pomdp = ScientificTruePOMDP(
                fitted, fitted_context, config, planning_seed, env_cfg, arm
            )
            policy = p5.RemediatedTrueStatePolicy(arm, pomdp, planning_seed)
        elif arm == "A3":
            pomdp = ScientificNoisyPOMDP(
                registered, registered_context, config, planning_seed, env_cfg, arm, True
            )
            policy = p5.RemediatedNoisyStatePolicy(arm, pomdp, planning_seed)
        elif arm == "A4":
            pomdp = ScientificNoisyPOMDP(
                fitted, fitted_context, config, planning_seed, env_cfg, arm, False
            )
            policy = p5.RemediatedNoisyStatePolicy(arm, pomdp, planning_seed)
        else:
            raise ValueError(arm)
        policies[arm] = policy
    if "A2" in policies and "A4" in policies:
        if policies["A2"].pomdp.model is not policies["A4"].pomdp.model:
            raise AssertionError("A2/A4 do not share the same fitted object in precheck")
    return {
        "cell": cid,
        "sigma": sigma,
        "env_cfg": env_cfg,
        "actions": actions,
        "planning_seed": planning_seed,
        "policies": policies,
        "fitted_provenance": {
            **fitted_provenance,
            **core.fit_metadata(core.FIT_CACHE_KEYS[(FOX, sigma)]),
            "fit_seed": core.FIT_SEED,
            "fit_config_digest": core.FIT_CONFIG_DIGEST,
            "fits_executed": 0,
        },
    }


def action_hash(actions: tuple[Any, ...]) -> str:
    return canonical_digest([asdict(action) for action in actions])


def reward_hash(cell: dict[str, Any]) -> str:
    reward = core.build_reward(cell["env_cfg"])
    return canonical_digest(
        {
            "environment": asdict(cell["env_cfg"]),
            "reward_class": f"{reward.__class__.__module__}.{reward.__class__.__name__}",
            "reward_parameters": asdict(reward),
            "reward_source": inspect.getsource(core.build_reward),
            "penalty_source": inspect.getsource(core.safety_penalty_indicator),
        }
    )


def code_hashes() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        SCRIPT_DIR / "e1_fox_grid161_aggregate.py",
        SCRIPT_DIR / "e1_phase1_parity.py",
        SCRIPT_DIR / "e1_phase2_core_validation.py",
        SCRIPT_DIR / "e1_phase5_remediated_validation.py",
        ECOLOGY_SRC / "real_ecology_benchmark" / "planners" / "pbvi.py",
        ECOLOGY_SRC / "real_ecology_benchmark" / "faithful_pomdp.py",
        ECOLOGY_SRC / "real_ecology_benchmark" / "envs.py",
        ECOLOGY_SRC / "real_ecology_benchmark" / "reward.py",
    ]
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def pomdp_hashes(cell: dict[str, Any], arm: str) -> dict[str, Any]:
    policy = cell["policies"][arm]
    pomdp = policy.pomdp
    capacity = core.capacity_grid(pomdp)
    hashes = {
        "model_parameter_hash": pomdp.model.parameter_hash(),
        "kernel_hash": core.full_transition_kernel_hash(pomdp),
        "grid_hash": array_hash(pomdp.abundance_grid),
        "capacity_grid_hash": array_hash(capacity),
        "action_hash": action_hash(cell["actions"]),
        "reward_hash": reward_hash(cell),
        "wrapper_hash": core.diagnostic_wrapper_code_digest(),
        "planner_config_hash": canonical_digest(asdict(pomdp.config)),
        "planning_seed": cell["planning_seed"],
        "survey_scale": pomdp.model.survey_scale,
        "fit_array_hash": (
            cell["fitted_provenance"]["array_hash"] if arm in ("A2", "A4") else None
        ),
        "fit_cache_key": (
            cell["fitted_provenance"]["cache_key"] if arm in ("A2", "A4") else None
        ),
    }
    hashes["policy_hash"] = canonical_digest({"cell": cell["cell"], "arm": arm, **hashes})
    return hashes


def true_state_belief(policy: Any, cell: dict[str, Any]) -> CandidateBelief:
    reset = core.ContinuousEcologyEnv(cell["env_cfg"]).reset(DERIVED_EPISODES[0])
    external = core.oracle_filter(cell["env_cfg"]).set_true_state(
        float(reset.evaluator_info["state"]), 0, reset.public_info
    )
    original = policy.planner.action_values
    policy.planner.action_values = lambda _belief: np.zeros(len(cell["actions"]), dtype=np.float64)
    try:
        policy.act(external, float(reset.observation))
    finally:
        policy.planner.action_values = original
    if policy.internal_belief is None:
        raise AssertionError("true-state policy failed to construct a belief")
    return policy.internal_belief


def check_belief(belief: CandidateBelief, label: str) -> dict[str, Any]:
    probabilities = np.asarray(belief.probabilities, dtype=np.float64)
    finite = bool(np.all(np.isfinite(probabilities)))
    error = abs(float(probabilities.sum()) - 1.0)
    if not finite or error > 1.0e-12 or np.any(probabilities < 0.0):
        raise AssertionError(f"invalid belief: {label}")
    return {"finite": finite, "normalization_error": error, "support": int(np.count_nonzero(probabilities))}


def episode_identity_payload() -> dict[str, Any]:
    rows = []
    for block in BLOCK_SEEDS:
        for local in range(EPISODES_PER_SEED):
            derived = block + local
            rows.append(
                {"block_seed": block, "local_episode_index": local, "derived_episode_identity": derived}
            )
    if len(rows) != 20 or [row["derived_episode_identity"] for row in rows] != list(DERIVED_EPISODES):
        raise AssertionError("derived episode identity construction failed")
    return {"rows": rows, "sha256": canonical_digest(rows)}


def canonical_config() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "species": FOX,
        "population_model": "ricker",
        "cells": [core.cell_id(FOX, sigma) for sigma in SIGMAS],
        "arms": {
            "A1": "registered transition model + true state",
            "A2": "fitted transition model + true state",
            "A3": "registered transition model + noisy observation",
            "A4": "fitted transition model + noisy observation",
        },
        "planner": {
            "name": "PBVI",
            "state_bins": STATE_BINS,
            "capacity_bins": CAPACITY_BINS,
            "observation_bins": OBSERVATION_BINS,
            "horizon": HORIZON,
            "belief_points": BELIEF_POINTS,
            "noisy_observation_branches": OBSERVATION_BRANCHES,
            "true_state_branch_convention": "exact predictive support and weights",
            "gamma": GAMMA,
        },
        "reward": {"route": "explicit source true reward", "mode": "safe", "penalty": "occupancy", "alpha": 1.0, "P": 10.0},
        "actions": 11,
        "environment": {"horizon": EVALUATION_HORIZON, "process_noise": 0.0},
        "block_seeds": list(BLOCK_SEEDS),
        "episodes_per_seed": EPISODES_PER_SEED,
        "derived_episode_identities": list(DERIVED_EPISODES),
        "planning_seeds_by_cell": {
            core.cell_id(FOX, sigma): core.PLANNING_SEEDS[core.cell_id(FOX, sigma)]
            for sigma in SIGMAS
        },
        "slurm": {"partition": EXPECTED_PARTITION, "constraint": EXPECTED_CONSTRAINT, "tasks": 8},
        "registered_deviation": REGISTERED_DEVIATION,
    }


def task_command(run_dir: Path, index: int) -> str:
    return (
        f"PYTHONDONTWRITEBYTECODE=1 {Path(sys.executable).resolve()} "
        f"{Path(__file__).resolve()} task --run-dir {run_dir.resolve()} --task-index {index}"
    )


def slurm_script_text() -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=e1fox161
#SBATCH --partition={EXPECTED_PARTITION}
#SBATCH --constraint={EXPECTED_CONSTRAINT}
#SBATCH --array=0-7%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7-00:00:00
#SBATCH --requeue

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec {Path(sys.executable).resolve()} {Path(__file__).resolve()} task --run-dir "$1" --task-index "$SLURM_ARRAY_TASK_ID"
"""


def choose_run_dir() -> Path:
    candidate = BASE_OUTPUT
    if candidate.exists():
        candidate = BASE_OUTPUT.with_name(f"{BASE_OUTPUT.name}_{stamp()}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate.resolve()


def prepare() -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = choose_run_dir()
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    environment = assert_environment()
    git_before = git_snapshot()
    episodes = episode_identity_payload()
    config = canonical_config()
    config_digest = canonical_digest(config)
    all_hashes: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    fit_pairing: dict[str, Any] = {}
    for sigma in SIGMAS:
        cell = build_cell(sigma)
        cid = cell["cell"]
        if cell["policies"]["A2"].pomdp.model is not cell["policies"]["A4"].pomdp.model:
            raise AssertionError(f"{cid}: A2/A4 fitted object identity failure")
        if len(cell["actions"]) != 11:
            raise AssertionError(f"{cid}: wrong action count")
        fit_pairing[cid] = {
            "same_object_in_precheck": True,
            "cache_key": cell["fitted_provenance"]["cache_key"],
            "array_hash": cell["fitted_provenance"]["array_hash"],
            "fits_executed": 0,
        }
        checks[cid] = {}
        all_hashes[cid] = {}
        for arm in ARMS:
            policy = cell["policies"][arm]
            pomdp = policy.pomdp
            if pomdp.config.state_bins != STATE_BINS:
                raise AssertionError(f"{cid}/{arm}: state_bins drift")
            if pomdp.objective_flag != "explicit_source_true_reward":
                raise AssertionError(f"{cid}/{arm}: true reward route absent")
            if arm in ("A1", "A2"):
                belief = true_state_belief(policy, cell)
                repaired_route = False
                branch_convention = "exact_predictive_support"
            else:
                belief = pomdp.initial_belief(float(cell["env_cfg"].N0))
                updated, _ = pomdp.runtime_update(belief, 0, float(cell["env_cfg"].N0))
                check_belief(updated, f"{cid}/{arm}/repaired_runtime")
                updated_lookahead, _ = pomdp.update(belief, 0, float(cell["env_cfg"].N0))
                check_belief(updated_lookahead, f"{cid}/{arm}/repaired_lookahead")
                repaired_route = isinstance(pomdp, p5.RemediatedNoisyStatePOMDP)
                if not repaired_route:
                    raise AssertionError(f"{cid}/{arm}: repaired filter class absent")
                branch_convention = OBSERVATION_BRANCHES
            belief_check = check_belief(belief, f"{cid}/{arm}/initial")
            all_hashes[cid][arm] = pomdp_hashes(cell, arm)
            checks[cid][arm] = {
                "policy_constructed": True,
                "state_bins": pomdp.config.state_bins,
                "belief": belief_check,
                "repaired_logspace_filter": repaired_route,
                "true_reward_route": pomdp.objective_flag,
                "branch_convention": branch_convention,
                "planning_seed": cell["planning_seed"],
            }
    for cid in all_hashes:
        if all_hashes[cid]["A2"]["fit_array_hash"] != all_hashes[cid]["A4"]["fit_array_hash"]:
            raise AssertionError(f"{cid}: A2/A4 cache hash mismatch")
        if all_hashes[cid]["A1"]["kernel_hash"] != all_hashes[cid]["A3"]["kernel_hash"]:
            raise AssertionError(f"{cid}: A1/A3 kernel mismatch")
        if all_hashes[cid]["A2"]["kernel_hash"] != all_hashes[cid]["A4"]["kernel_hash"]:
            raise AssertionError(f"{cid}: A2/A4 kernel mismatch")
    planning_seeds = set(config["planning_seeds_by_cell"].values())
    if planning_seeds.intersection(DERIVED_EPISODES):
        raise AssertionError("planning and evaluation seeds overlap")
    architecture_precheck = {
        "partition": EXPECTED_PARTITION,
        "constraint": EXPECTED_CONSTRAINT,
        "one_architecture_enforced_by_all_task_commands": True,
        "runtime_cpu_assertion_required": "AMD EPYC 9534",
    }
    script_path = logs / "e1_fox_grid161.slurm"
    write_text_new(script_path, slurm_script_text(), 0o755)
    submit_command = (
        f"sbatch --parsable --partition={EXPECTED_PARTITION} --constraint={EXPECTED_CONSTRAINT} "
        f"--array=0-7%8 --output={run_dir}/logs/slurm-%A_%a.out "
        f"--error={run_dir}/logs/slurm-%A_%a.err {script_path} {run_dir}"
    )
    task_map = {
        str(index): {
            "cell": cid,
            "arm": arm,
            "command": task_command(run_dir, index),
            "expected_hashes": all_hashes[cid][arm],
        }
        for index, (cid, arm) in enumerate(TASKS)
    }
    registration = {
        "schema": f"{SCHEMA}_registration",
        "timestamp_utc": utc_now(),
        "result_directory": str(run_dir),
        "repository_commit": git_before["commit"],
        "git_status_before": git_before,
        "environment": environment,
        "environment_digest": environment["environment_digest"],
        "code_hashes": code_hashes(),
        "canonical_configuration": config,
        "configuration_digest": config_digest,
        "derived_episode_identities": episodes,
        "planning_rng_independent_of_evaluation_rng": True,
        "planning_seeds": config["planning_seeds_by_cell"],
        "evaluation_seeds": list(DERIVED_EPISODES),
        "fit_pairing": fit_pairing,
        "pre_return_checks": checks,
        "architecture_precheck": architecture_precheck,
        "task_map": task_map,
        "complete_submission_command": submit_command,
        "exact_task_commands": {key: value["command"] for key, value in task_map.items()},
        "preregistered_statements": PREREGISTERED_STATEMENTS,
        "registered_deviation": REGISTERED_DEVIATION,
        "returns_computed_or_opened_before_registration": False,
    }
    registration_path = run_dir / f"{REGISTRATION_PREFIX}{stamp()}.json"
    write_json_new(registration_path, registration, 0o444)
    registration_hash = sha256_file(registration_path)
    control = {
        "schema": f"{SCHEMA}_control",
        "registration_path": str(registration_path),
        "registration_sha256": registration_hash,
        "configuration_digest": config_digest,
        "episode_identity_sha256": episodes["sha256"],
        "result_directory": str(run_dir),
        "submit_command": submit_command,
        "task_map": task_map,
    }
    write_json_new(run_dir / RUN_CONTROL_NAME, control, 0o444)
    write_json_new(logs / "pre_run_check.json", {
        "status": "PASS", "checks": checks, "fit_pairing": fit_pairing,
        "architecture": architecture_precheck, "elapsed_seconds": time.perf_counter() - started,
    })
    write_json_new(logs / "episode_identities.json", episodes)
    write_text_new(logs / "environment_version.txt", json.dumps(environment, indent=2, sort_keys=True) + "\n")
    write_text_new(logs / "git_status_before.txt", "\n".join(git_before["status_lines"]) + "\n")
    output = {
        "status": "PASS",
        "run_dir": str(run_dir),
        "registration_path": str(registration_path),
        "registration_sha256": registration_hash,
        "derived_episode_identities": list(DERIVED_EPISODES),
        "episode_identity_sha256": episodes["sha256"],
        "submit_command": submit_command,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def resolve_task(run_dir: Path, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    control = load_json(run_dir / RUN_CONTROL_NAME)
    registration_path = Path(control["registration_path"])
    if sha256_file(registration_path) != control["registration_sha256"]:
        raise AssertionError("registration hash mismatch")
    registration = load_json(registration_path)
    if registration["configuration_digest"] != control["configuration_digest"]:
        raise AssertionError("configuration digest mismatch")
    task = control["task_map"].get(str(index))
    if task is None:
        raise ValueError(f"task index outside 0..7: {index}")
    return control, task


def cell_sigma(cid: str) -> float:
    for sigma in SIGMAS:
        if core.cell_id(FOX, sigma) == cid:
            return sigma
    raise ValueError(cid)


def public_transition(result: Any) -> PublicTransition:
    return PublicTransition(
        observation=float(result.observation),
        done=bool(result.done),
        truncated=bool(result.truncated),
        terminated=bool(result.done and not result.truncated),
        public_info=result.public_info.copy(),
    )


def counter_snapshot(policy: Any) -> dict[str, Any]:
    pomdp = policy.pomdp
    return {
        "phase5": pomdp.phase5.as_dict(),
        "science": {
            "predict_calls": pomdp.science["predict_calls"],
            "transition_matrix_calls": pomdp.science["transition_matrix_calls"],
            "representative_observation_calls": pomdp.science["representative_observation_calls"],
        },
        "old_fallback_calls": getattr(pomdp, "old_fallback_calls", 0),
        "true_reward_calls": pomdp.true_reward_calls,
        "planner_invocations": policy.planner.invocation_count,
        "planner_elapsed_seconds": policy.planner.elapsed_seconds,
    }


def numeric_delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in after.items():
        prior = before[key]
        if isinstance(value, dict):
            result[key] = numeric_delta(value, prior)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if key.startswith("maximum_"):
                result[key] = value
            else:
                result[key] = value - prior
        else:
            result[key] = value
    return result


def branch_delta(policy: Any, start: tuple[Counter, Counter]) -> dict[str, Any]:
    histogram = Counter(policy.pomdp.science["distinct_branch_histogram"])
    returned = Counter(policy.pomdp.science["returned_branch_histogram"])
    histogram.subtract(start[0])
    returned.subtract(start[1])
    histogram = +histogram
    returned = +returned
    calls = sum(histogram.values())
    return {
        "calls": calls,
        "distinct_branch_histogram": {str(key): value for key, value in sorted(histogram.items())},
        "returned_branch_histogram": {str(key): value for key, value in sorted(returned.items())},
        "minimum_distinct": min(histogram, default=0),
        "maximum_distinct": max(histogram, default=0),
    }


def stream_descriptor(seed: int, sigma: float) -> dict[str, Any]:
    streams = np.random.SeedSequence(seed).spawn(5)
    payload = {
        "seed": seed,
        "stream_names": ["parameters", "regime", "process", "observation", "initial"],
        "spawn_keys": [list(stream.spawn_key) for stream in streams],
        "observation_sigma": sigma,
        "process_noise_sigma": 0.0,
    }
    payload["hash"] = canonical_digest(payload)
    return payload


def evaluate_episode(cell: dict[str, Any], arm: str, seed: int, local: int, block: int) -> dict[str, Any]:
    policy = cell["policies"][arm]
    pomdp = policy.pomdp
    before = counter_snapshot(policy)
    branch_start = (
        Counter(pomdp.science["distinct_branch_histogram"]),
        Counter(pomdp.science["returned_branch_histogram"]),
    )
    env = core.ContinuousEcologyEnv(cell["env_cfg"])
    reset = env.reset(seed)
    filt = core.oracle_filter(cell["env_cfg"]) if arm in ("A1", "A2") else core.raw_filter(cell["env_cfg"])
    if isinstance(filt, OracleStateFilter):
        belief: BeliefState = filt.set_true_state(
            float(reset.evaluator_info["state"]), 0, reset.public_info
        )
    else:
        belief = filt.reset(float(reset.observation), seed + 10_000)
    policy.reset(seed + 20_000)
    observation = float(reset.observation)
    states = [float(reset.evaluator_info["state"])]
    observations = [observation]
    capacities = [float(reset.evaluator_info.get("K_eff", cell["env_cfg"].K_base))]
    public_context = [reset.public_info.copy()]
    actions: list[int] = []
    collapse_steps: list[int] = []
    unsafe_steps: list[int] = [0] if states[0] <= cell["env_cfg"].safety_threshold else []
    mvp_steps: list[int] = [0] if states[0] <= cell["env_cfg"].mvp_threshold else []
    safety_penalty_steps: list[int] = []
    discounted = 0.0
    undiscounted = 0.0
    discount = 1.0
    planner_wall = 0.0
    for step in range(EVALUATION_HORIZON):
        t0 = time.perf_counter()
        action = int(policy.act(belief, observation))
        planner_wall += time.perf_counter() - t0
        if not 0 <= action < len(cell["actions"]):
            raise AssertionError(f"invalid selected action {action}")
        result = env.step(action)
        reward_true = float(result.evaluator_info["reward_true"])
        if not math.isfinite(reward_true):
            raise AssertionError("non-finite true reward")
        discounted += discount * reward_true
        undiscounted += reward_true
        actions.append(action)
        states.append(float(result.evaluator_info["state"]))
        observations.append(float(result.observation))
        capacities.append(float(result.evaluator_info.get("K_eff", cell["env_cfg"].K_base)))
        public_context.append(result.public_info.copy())
        if bool(result.evaluator_info["entered_safety_region"]):
            collapse_steps.append(step + 1)
        if bool(result.evaluator_info["below_safety_region"]):
            unsafe_steps.append(step + 1)
        if bool(result.evaluator_info["below_mvp_region"]):
            mvp_steps.append(step + 1)
        if bool(result.evaluator_info["safety_penalty_applied"]):
            safety_penalty_steps.append(step + 1)
        policy.observe(belief, action, public_transition(result))
        if isinstance(filt, OracleStateFilter):
            belief = filt.set_true_state(states[-1], step + 1, result.public_info)
        else:
            belief = filt.update(belief, action, float(result.observation))
        observation = float(result.observation)
        discount *= GAMMA
        if result.done:
            break
    after = counter_snapshot(policy)
    delta = numeric_delta(after, before)
    branches = branch_delta(policy, branch_start)
    if not math.isfinite(discounted) or not math.isfinite(undiscounted):
        raise AssertionError("non-finite episode return")
    if pomdp.phase5.runtime["non_normalized_posterior"] != 0 or pomdp.phase5.lookahead["non_normalized_posterior"] != 0:
        raise AssertionError("posterior normalization failure")
    pairing_key = f"{cell['cell']}|episode_{seed}"
    route_counters = {
        "runtime": delta["phase5"]["runtime_repaired_update"],
        "pbvi_lookahead": delta["phase5"]["pbvi_lookahead_repaired_update"],
        "old_fallback_call_count": delta["old_fallback_calls"],
        "counterfactual_pre_repair_density_fallback_count": (
            delta["phase5"]["runtime_repaired_update"]["old_density_evidence_at_or_below_floor"]
            + delta["phase5"]["pbvi_lookahead_repaired_update"]["old_density_evidence_at_or_below_floor"]
        ),
        "representative_observation_branches": branches,
    }
    return {
        "schema": f"{SCHEMA}_episode",
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "cell": cell["cell"],
        "arm": arm,
        "block_seed": block,
        "local_episode_index": local,
        "derived_episode_identity": seed,
        "pairing_key": pairing_key,
        "discounted_return": discounted,
        "undiscounted_return": undiscounted,
        "action_sequence": actions,
        "trajectory": {
            "state": states,
            "observation": observations,
            "capacity": capacities,
            "public_context": public_context,
        },
        "events": {
            "collapse_entry_steps": collapse_steps,
            "safety_occupancy_steps": unsafe_steps,
            "mvp_occupancy_steps": mvp_steps,
            "safety_penalty_steps": safety_penalty_steps,
        },
        "planner_seconds": planner_wall,
        "planner_internal_seconds": delta["planner_elapsed_seconds"],
        "model_evaluation_count": delta["science"]["predict_calls"],
        "transition_matrix_call_count": delta["science"]["transition_matrix_calls"],
        "true_reward_call_count": delta["true_reward_calls"],
        "filter_counters": route_counters,
        "planning_seed": cell["planning_seed"],
        "evaluation_seed": seed,
        "policy_reset_seed_ignored_by_fixed_planner": seed + 20_000,
        "external_filter_seed": None if arm in ("A1", "A2") else seed + 10_000,
        "environment_random_stream": stream_descriptor(seed, cell["sigma"]),
        "reward_source": "ContinuousEcologyEnv evaluator_info.reward_true",
        "episode_length": len(actions),
    }


def aggregate_task_counters(records: list[dict[str, Any]]) -> dict[str, Any]:
    def sum_route(name: str) -> dict[str, Any]:
        rows = [record["filter_counters"][name] for record in records]
        keys = rows[0].keys()
        return {
            key: (max(float(row[key]) for row in rows) if key.startswith("maximum_") else sum(row[key] for row in rows))
            for key in keys
        }
    hist = Counter()
    for record in records:
        for key, value in record["filter_counters"]["representative_observation_branches"]["distinct_branch_histogram"].items():
            hist[int(key)] += int(value)
    return {
        "runtime_repaired_logspace": sum_route("runtime"),
        "pbvi_lookahead_repaired_logspace": sum_route("pbvi_lookahead"),
        "old_fallback_call_count": sum(record["filter_counters"]["old_fallback_call_count"] for record in records),
        "counterfactual_pre_repair_density_fallback_count": sum(
            record["filter_counters"]["counterfactual_pre_repair_density_fallback_count"] for record in records
        ),
        "distinct_representative_observation_branch_histogram": {str(key): value for key, value in sorted(hist.items())},
        "model_evaluation_count": sum(record["model_evaluation_count"] for record in records),
        "planner_seconds": sum(record["planner_seconds"] for record in records),
    }


def checkpoint(run_dir: Path, task_dir: Path, control: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    episodes = sorted((task_dir / "episodes").glob("episode_*.json"))
    files = {str(path.resolve()): sha256_file(path) for path in episodes}
    value = {
        "schema": f"{SCHEMA}_checkpoint",
        "created_utc": utc_now(),
        "configuration_digest": control["configuration_digest"],
        "registration_sha256": control["registration_sha256"],
        "cell": task["cell"],
        "arm": task["arm"],
        "completed_episode_count": len(episodes),
        "completed_files": files,
        "slurm": cpu_info(),
    }
    path = task_dir / "checkpoints" / f"checkpoint_{len(episodes):02d}_{stamp()}_{os.getpid()}.json"
    write_json_new(path, value)
    return value


def validate_existing_episode(path: Path, control: dict[str, Any], task: dict[str, Any], seed: int) -> dict[str, Any]:
    value = load_json(path)
    if value.get("status") != "COMPLETE" or value.get("cell") != task["cell"] or value.get("arm") != task["arm"]:
        raise AssertionError(f"invalid existing episode file: {path}")
    if value.get("derived_episode_identity") != seed:
        raise AssertionError(f"episode identity mismatch: {path}")
    hashes = value.get("hashes", {})
    if hashes.get("registration_sha256") != control["registration_sha256"]:
        raise AssertionError(f"episode registration mismatch: {path}")
    if hashes.get("policy_hash") != task["expected_hashes"]["policy_hash"]:
        raise AssertionError(f"episode policy hash mismatch: {path}")
    return value


def assert_slurm_cpu() -> dict[str, Any]:
    info = cpu_info()
    if info["slurm_partition"] != EXPECTED_PARTITION:
        raise AssertionError(f"wrong Slurm partition: {info['slurm_partition']}")
    if not info["slurm_job_id"] or info["slurm_array_task_id"] is None:
        raise AssertionError("scientific task must run inside the registered Slurm array")
    if not info["cpu_models"] or not all("EPYC 9534" in model for model in info["cpu_models"]):
        raise AssertionError(f"wrong CPU architecture: {info['cpu_models']}")
    return info


def run_task(run_dir: Path, index: int) -> dict[str, Any]:
    environment = assert_environment()
    cpu = assert_slurm_cpu()
    control, task = resolve_task(run_dir, index)
    cid, arm = task["cell"], task["arm"]
    if (cid, arm) != TASKS[index]:
        raise AssertionError("task mapping drift")
    task_dir = run_dir / "tasks" / cid / arm
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (task_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (task_dir / "resume_events").mkdir(parents=True, exist_ok=True)
    final_path = task_dir / "TASK_RECEIPT.json"
    if final_path.exists():
        value = load_json(final_path)
        if value.get("status") != "COMPLETE" or value.get("registration_sha256") != control["registration_sha256"]:
            raise AssertionError("completed task receipt is incompatible")
        print(json.dumps({"status": "ALREADY_COMPLETE", "path": str(final_path)}, indent=2))
        return value
    write_json_new(
        task_dir / "resume_events" / f"start_{stamp()}_{cpu['slurm_job_id']}_{os.getpid()}.json",
        {
            "created_utc": utc_now(), "cell": cid, "arm": arm, "task_index": index,
            "registration_sha256": control["registration_sha256"], "cpu": cpu,
            "command": task_command(run_dir, index),
        },
    )
    sigma = cell_sigma(cid)
    cell = build_cell(sigma, (arm,))
    policy = cell["policies"][arm]
    actual_hashes = pomdp_hashes(cell, arm)
    if actual_hashes != task["expected_hashes"]:
        raise AssertionError("task policy/model/kernel/grid hash drift")
    records: list[dict[str, Any]] = []
    with p5.FrozenBaseSpies():
        for block in BLOCK_SEEDS:
            for local in range(EPISODES_PER_SEED):
                seed = block + local
                episode_path = task_dir / "episodes" / f"episode_{seed}.json"
                if episode_path.exists():
                    records.append(validate_existing_episode(episode_path, control, task, seed))
                    continue
                record = evaluate_episode(cell, arm, seed, local, block)
                record["hashes"] = {
                    **actual_hashes,
                    "registration_sha256": control["registration_sha256"],
                    "configuration_digest": control["configuration_digest"],
                    "code_hashes": code_hashes(),
                }
                write_json_new(episode_path, record)
                checkpoint(run_dir, task_dir, control, task)
                records.append(record)
    records.sort(key=lambda row: row["derived_episode_identity"])
    if [row["derived_episode_identity"] for row in records] != list(DERIVED_EPISODES):
        raise AssertionError("task episode completeness failure")
    receipt = {
        "schema": f"{SCHEMA}_task_receipt",
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "cell": cid,
        "arm": arm,
        "task_index": index,
        "policy_build_count_this_invocation": 1,
        "expected_episode_count": 20,
        "actual_episode_count": len(records),
        "episode_files": {
            str((task_dir / "episodes" / f"episode_{row['derived_episode_identity']}.json").resolve()):
                sha256_file(task_dir / "episodes" / f"episode_{row['derived_episode_identity']}.json")
            for row in records
        },
        "episode_identities": [row["derived_episode_identity"] for row in records],
        "pairing_keys": [row["pairing_key"] for row in records],
        "aggregate_counters": aggregate_task_counters(records),
        "hashes": actual_hashes,
        "registration_sha256": control["registration_sha256"],
        "configuration_digest": control["configuration_digest"],
        "environment": environment,
        "cpu": cpu,
        "scheduler": {
            "job_id": cpu["slurm_job_id"],
            "array_job_id": cpu["slurm_array_job_id"],
            "array_task_id": cpu["slurm_array_task_id"],
            "partition": cpu["slurm_partition"],
            "required_constraint": EXPECTED_CONSTRAINT,
        },
        "fits_executed": 0,
        "completed_episode_files_overwritten": False,
    }
    write_json_new(final_path, receipt)
    print(json.dumps({
        "status": "COMPLETE", "cell": cid, "arm": arm,
        "episodes": len(records), "task_receipt": str(final_path.resolve()),
    }, indent=2, sort_keys=True))
    return receipt


def failure_record(run_dir: Path, index: int, error: BaseException) -> None:
    try:
        _control, task = resolve_task(run_dir, index)
        root = run_dir / "tasks" / task["cell"] / task["arm"] / "failures"
        write_json_new(root / f"failure_{stamp()}_{os.getpid()}.json", {
            "created_utc": utc_now(), "task_index": index, "cell": task["cell"], "arm": task["arm"],
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "cpu": cpu_info(),
        })
    except Exception:
        pass


def scheduler_status(submission: dict[str, Any], logs: Path) -> dict[str, Any]:
    job_id = str(submission["job_id"])
    output = command_output(
        ["sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State,ExitCode,NodeList,Elapsed"],
        check=False,
    )
    path = logs / "scheduler_exit_status.txt"
    if not path.exists():
        write_text_new(path, output)
    rows = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 5:
            rows.append({"job_id": fields[0], "state": fields[1], "exit_code": fields[2], "nodes": fields[3], "elapsed": fields[4]})
    return {"raw_log": str(path.resolve()), "rows": rows}


def reproduce_markdown(run_dir: Path, control: dict[str, Any]) -> str:
    script = Path(__file__).resolve()
    py = Path(sys.executable).resolve()
    return f"""# E1 fox-only grid-161 reproduction commands

These commands are non-destructive. Existing immutable episode files are never replaced.

## Reconstruct each policy (pre-return checks only)

```bash
PYTHONDONTWRITEBYTECODE=1 {py} {script} check --run-dir {run_dir}
```

## Reproduce one specified episode

Use a new result directory or an empty compatible task namespace; never point this command at a completed task:

```bash
sbatch --parsable --partition=comp --constraint=EPYC9534 --array=0 --output=/tmp/e1-reproduce-%A_%a.out --error=/tmp/e1-reproduce-%A_%a.err {run_dir}/logs/e1_fox_grid161.slurm {run_dir}
```

Task index 0 is `{TASKS[0][0]}/{TASKS[0][1]}`. The immutable record for episode 7001 is `{run_dir}/tasks/{TASKS[0][0]}/{TASKS[0][1]}/episodes/episode_7001.json`.

## Reproduce or resume every episode

```bash
{control['submit_command']}
```

## Recompute file hashes

```bash
find {run_dir} -type f -print0 | sort -z | xargs -0 sha256sum
```

## Verify pairing and completeness

```bash
PYTHONDONTWRITEBYTECODE=1 {py} {script} verify --run-dir {run_dir}
```

## Aggregate and statistically analyse later

```bash
PYTHONDONTWRITEBYTECODE=1 {py} {SCRIPT_DIR / 'e1_fox_grid161_aggregate.py'} --run-dir {run_dir} --output {run_dir}/analysis_later
```
"""


def verify(run_dir: Path, write_outputs: bool = False) -> dict[str, Any]:
    control = load_json(run_dir / RUN_CONTROL_NAME)
    receipts: dict[str, Any] = {}
    episode_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    task_paths: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    for cid, arm in TASKS:
        receipt_path = run_dir / "tasks" / cid / arm / "TASK_RECEIPT.json"
        if not receipt_path.exists():
            raise AssertionError(f"missing task receipt: {cid}/{arm}")
        receipt = load_json(receipt_path)
        if receipt.get("status") != "COMPLETE" or receipt.get("actual_episode_count") != 20:
            raise AssertionError(f"incomplete task: {cid}/{arm}")
        if receipt.get("registration_sha256") != control["registration_sha256"]:
            raise AssertionError(f"registration mismatch: {cid}/{arm}")
        key = f"{cid}/{arm}"
        receipts[key] = receipt
        task_paths[key] = str(receipt_path.resolve())
        episode_records.setdefault(cid, {})[arm] = []
        for seed in DERIVED_EPISODES:
            path = run_dir / "tasks" / cid / arm / "episodes" / f"episode_{seed}.json"
            record = validate_existing_episode(path, control, control["task_map"][str(TASKS.index((cid, arm)))], seed)
            if not math.isfinite(float(record["discounted_return"])) or not math.isfinite(float(record["undiscounted_return"])):
                raise AssertionError(f"non-finite return: {path}")
            episode_records[cid][arm].append(record)
            file_hashes[str(path.resolve())] = sha256_file(path)
        file_hashes[str(receipt_path.resolve())] = sha256_file(receipt_path)
    pairing = {}
    for cid, arms in episode_records.items():
        expected = [f"{cid}|episode_{seed}" for seed in DERIVED_EPISODES]
        pairing[cid] = {}
        for arm in ARMS:
            keys = [row["pairing_key"] for row in arms[arm]]
            identities = [row["derived_episode_identity"] for row in arms[arm]]
            if keys != expected or identities != list(DERIVED_EPISODES) or len(set(keys)) != 20:
                raise AssertionError(f"pairing failure: {cid}/{arm}")
            pairing[cid][arm] = {"count": len(keys), "keys_hash": canonical_digest(keys)}
        if len({pairing[cid][arm]["keys_hash"] for arm in ARMS}) != 1:
            raise AssertionError(f"cross-arm pairing mismatch: {cid}")
        stream_by_seed = {
            seed: {
                arms[arm][i]["environment_random_stream"]["hash"]
                for arm in ARMS
            }
            for i, seed in enumerate(DERIVED_EPISODES)
        }
        if any(len(values) != 1 for values in stream_by_seed.values()):
            raise AssertionError(f"environment stream mismatch: {cid}")
    cpu_models = {
        model
        for receipt in receipts.values()
        for model in receipt["cpu"]["cpu_models"]
    }
    if len(cpu_models) != 1 or not all("EPYC 9534" in model for model in cpu_models):
        raise AssertionError(f"mixed or wrong CPU architectures: {cpu_models}")
    result = {
        "status": "PASS",
        "all_eight_tasks_complete": len(receipts) == 8,
        "expected_episode_count_per_arm": 20,
        "actual_episode_count_per_arm": {key: value["actual_episode_count"] for key, value in receipts.items()},
        "pairing": pairing,
        "cpu_models": sorted(cpu_models),
        "task_receipts": task_paths,
        "aggregate_counters": {key: value["aggregate_counters"] for key, value in receipts.items()},
        "result_file_hashes": file_hashes,
        "returns_interpreted_or_compared": False,
        "duplicates_or_missing": False,
        "completed_result_overwritten": False,
    }
    if write_outputs:
        path = run_dir / "logs" / "final_completeness_check.json"
        write_json_new(path, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


def finalize(run_dir: Path) -> dict[str, Any]:
    control = load_json(run_dir / RUN_CONTROL_NAME)
    verification = verify(run_dir, write_outputs=True)
    logs = run_dir / "logs"
    registration = load_json(Path(control["registration_path"]))
    submission_path = run_dir / "SUBMISSION.json"
    if not submission_path.exists():
        raise AssertionError("missing submission record")
    submission = load_json(submission_path)
    scheduler = scheduler_status(submission, logs)
    git_after = git_snapshot()
    before = registration["git_status_before"]
    if git_after["tracked_src_tracks_digest"] != before["tracked_src_tracks_digest"]:
        raise AssertionError("src/tracks tracked content changed")
    if git_after["src_tracks_git_diff"]:
        raise AssertionError("src/tracks git diff is non-empty")
    reproduce_path = run_dir / "REPRODUCE.md"
    write_text_new(reproduce_path, reproduce_markdown(run_dir, control))
    claude_path = run_dir / "CLAUDE_READ_ONLY_AUDIT_PROMPT.md"
    claude = f"""Perform an independent READ-ONLY audit of the completed E1 fox-only grid-161 run.

Repository: {REPO_ROOT}
Audit manifest: {run_dir / 'AUDIT_MANIFEST.json'}
Run receipt: {run_dir / 'RUN_RECEIPT.json'}

Do not modify files, rerun the experiment, or aggregate/interpret which arm wins.

1. Verify AUDIT_MANIFEST.json and every listed SHA-256.
2. Verify the immutable registration hash is {control['registration_sha256']} and predates every episode.
3. Independently reconstruct the policy/model/grid/reward/action hashes for all eight cell/arm tasks.
4. Independently verify at least one episode per arm/cell (eight episodes total) from its seed, action sequence, true-state reward route, trajectories and discounted return.
5. Verify all twenty pairing keys match across A1–A4 within each cell and the environmental stream descriptor is shared by paired episodes.
6. Verify Python 3.10.14, NumPy 2.2.6, comp/EPYC9534 for every task, identical architecture, zero refits, and A2/A4 fitted-cache identity by hash.
7. Verify repaired runtime/lookahead filter counters, zero-evidence/fallback fields, append-only checkpoints, scheduler mapping, and that no completed episode was overwritten.
8. Verify src/tracks/** and controlling documents were unchanged.

Report every mismatch before any later aggregation. Give PASS only if all checks independently reproduce.
"""
    write_text_new(claude_path, claude)
    result_files = dict(verification["result_file_hashes"])
    for path in (
        Path(control["registration_path"]), run_dir / RUN_CONTROL_NAME, submission_path,
        reproduce_path, claude_path, logs / "pre_run_check.json",
        logs / "episode_identities.json", logs / "final_completeness_check.json",
        logs / "scheduler_exit_status.txt", logs / "e1_fox_grid161.slurm",
    ):
        result_files[str(path.resolve())] = sha256_file(path)
    run_receipt = {
        "schema": f"{SCHEMA}_run_receipt",
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "result_directory": str(run_dir.resolve()),
        "registration_path": control["registration_path"],
        "registration_sha256": control["registration_sha256"],
        "canonical_configuration": registration["canonical_configuration"],
        "registered_deviation": REGISTERED_DEVIATION,
        "task_completion": {key: "COMPLETE" for key in verification["task_receipts"]},
        "episode_counts": verification["actual_episode_count_per_arm"],
        "pairing_verification": verification["pairing"],
        "aggregate_filter_counters": verification["aggregate_counters"],
        "result_file_hashes": result_files,
        "scheduler": scheduler,
        "git_status_before": before,
        "git_status_after": git_after,
        "src_tracks_unchanged": True,
        "returns_interpreted_or_compared": False,
        "deviations_or_failures": [],
        "no_completed_result_file_overwritten": True,
    }
    run_receipt_path = run_dir / "RUN_RECEIPT.json"
    write_json_new(run_receipt_path, run_receipt)
    run_receipt_hash = sha256_file(run_receipt_path)
    output_files = dict(result_files)
    output_files[str(run_receipt_path.resolve())] = run_receipt_hash
    task_mapping = {
        key: {
            "job_id": value["scheduler"]["job_id"],
            "array_job_id": value["scheduler"]["array_job_id"],
            "array_task_id": value["scheduler"]["array_task_id"],
            "cell": value["cell"], "arm": value["arm"],
        }
        for key, value in ((key, load_json(Path(path))) for key, path in verification["task_receipts"].items())
    }
    audit_manifest = {
        "schema": f"{SCHEMA}_audit_manifest",
        "created_utc": utc_now(),
        "absolute_result_directory": str(run_dir.resolve()),
        "preregistration": {
            "path": control["registration_path"],
            "timestamp": registration["timestamp_utc"],
            "sha256": control["registration_sha256"],
        },
        "repository_commit": registration["repository_commit"],
        "git_status_before": before,
        "git_status_after": git_after,
        "complete_modified_and_untracked_files_before": before["modified_or_untracked_files"],
        "complete_modified_and_untracked_files_after": git_after["modified_or_untracked_files"],
        "src_tracks_unchanged": {
            "confirmed": True,
            "before_digest": before["tracked_src_tracks_digest"],
            "after_digest": git_after["tracked_src_tracks_digest"],
            "git_diff": git_after["src_tracks_git_diff"],
        },
        "environment": registration["environment"],
        "environment_digest": registration["environment_digest"],
        "cpu_architecture": verification["cpu_models"],
        "EPYC9534_confirmation_for_every_task": True,
        "canonical_full_experiment_configuration": registration["canonical_configuration"],
        "planning_seeds": registration["planning_seeds"],
        "evaluation_seeds": registration["evaluation_seeds"],
        "twenty_derived_episode_identities": registration["derived_episode_identities"],
        "slurm_job_to_cell_arm": task_mapping,
        "exact_task_commands": registration["exact_task_commands"],
        "complete_submission_command": registration["complete_submission_command"],
        "scheduler_exit_status": scheduler,
        "task_hashes": {
            key: value["hashes"]
            for key, value in ((key, load_json(Path(path))) for key, path in verification["task_receipts"].items())
        },
        "expected_episode_count_per_arm": 20,
        "actual_episode_count_per_arm": verification["actual_episode_count_per_arm"],
        "pairing_verification": verification["pairing"],
        "every_output_file_path_and_sha256": output_files,
        "checkpoint_resume_history": {
            key: {
                "checkpoints": sorted(str(path.resolve()) for path in (Path(receipt_path).parent / "checkpoints").glob("*.json")),
                "resume_events": sorted(str(path.resolve()) for path in (Path(receipt_path).parent / "resume_events").glob("*.json")),
            }
            for key, receipt_path in verification["task_receipts"].items()
        },
        "filter_counters": verification["aggregate_counters"],
        "warnings_failures_deviations": [],
        "no_file_was_overwritten": True,
        "run_receipt_path": str(run_receipt_path.resolve()),
        "run_receipt_sha256": run_receipt_hash,
        "reproduce_path": str(reproduce_path.resolve()),
        "claude_read_only_audit_prompt": str(claude_path.resolve()),
        "interpretation_or_conclusions_in_audit_package": False,
    }
    audit_path = run_dir / "AUDIT_MANIFEST.json"
    write_json_new(audit_path, audit_manifest)
    audit_hash = sha256_file(audit_path)
    write_text_new(run_dir / "AUDIT_MANIFEST.sha256", f"{audit_hash}  {audit_path.name}\n")
    output = {
        "status": "COMPLETE",
        "run_dir": str(run_dir.resolve()),
        "registration_sha256": control["registration_sha256"],
        "run_receipt": str(run_receipt_path.resolve()),
        "run_receipt_sha256": run_receipt_hash,
        "audit_manifest": str(audit_path.resolve()),
        "audit_manifest_sha256": audit_hash,
        "task_receipts": verification["task_receipts"],
        "episode_counts": verification["actual_episode_count_per_arm"],
        "pairing": verification["pairing"],
        "filter_counters": verification["aggregate_counters"],
        "later_aggregation_command": (
            f"PYTHONDONTWRITEBYTECODE=1 {Path(sys.executable).resolve()} "
            f"{SCRIPT_DIR / 'e1_fox_grid161_aggregate.py'} --run-dir {run_dir.resolve()} "
            f"--output {run_dir.resolve() / 'analysis_later'}"
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def check_existing(run_dir: Path) -> dict[str, Any]:
    control = load_json(run_dir / RUN_CONTROL_NAME)
    registration = load_json(Path(control["registration_path"]))
    if sha256_file(Path(control["registration_path"])) != control["registration_sha256"]:
        raise AssertionError("registration mismatch")
    result = {
        "status": "PASS",
        "registration_sha256": control["registration_sha256"],
        "configuration_digest": control["configuration_digest"],
        "task_map": registration["task_map"],
        "pre_return_checks": registration["pre_return_checks"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    for name in ("task", "verify", "finalize", "check"):
        child = sub.add_parser(name)
        child.add_argument("--run-dir", type=Path, required=True)
        if name == "task":
            child.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "task":
        try:
            run_task(args.run_dir.resolve(), args.task_index)
        except BaseException as error:
            failure_record(args.run_dir.resolve(), args.task_index, error)
            raise
    elif args.command == "verify":
        verify(args.run_dir.resolve())
    elif args.command == "finalize":
        finalize(args.run_dir.resolve())
    else:
        check_existing(args.run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
