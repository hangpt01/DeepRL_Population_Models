#!/usr/bin/env python3
"""E1 Phase 3 remainder: environment comparison and no-return V6--V8 probes."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
ECOLOGY_SRC = REPO_ROOT / "src" / "tracks" / "ecological"
for entry in (str(SCRIPT_DIR), str(REPO_ROOT), str(ECOLOGY_SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import e1_phase2_core_validation as core  # noqa: E402
from real_ecology_benchmark.config import FaithfulPlannerConfig  # noqa: E402
from real_ecology_benchmark.envs import ContinuousEcologyEnv  # noqa: E402
from real_ecology_benchmark.faithful_pomdp import CandidateBelief  # noqa: E402
from real_ecology_benchmark.planners.pbvi import PointBasedPlanner  # noqa: E402
from real_ecology_benchmark.types import PublicTransition  # noqa: E402


DECLARED_PYTHON = (3, 10, 14)
DECLARED_NUMPY = "2.2.6"
OLD_BUILD_DIGEST = "b090d6fa79d5dd1453d102cc79f18d51a6d0c3d89d58442ce937544a60461fd0"
CONTROLLING_BUILD_DIGEST = "c44bc6d514c346fd190a0c60b1bf107b6f250fe0834a6fa7a45674c845983f49"
OLD_PHASE1_RECEIPT_SHA256 = "29a5a7301aa05cf7806ff787415725a0ece710c66455a51b92aa6f8cff188f88"
OLD_PHASE2_RECEIPT_SHA256 = "f146f201b6022add1c0d3b86d59949adc22122018f0f90b9d46bc23e1c3c16dd"
PHASE3_ROOT = REPO_ROOT / ".verification" / "e1_phase3_remainder"
NEW_PHASE1 = REPO_ROOT / ".verification" / "e1_py310_numpy226" / "phase1" / "E1_RECEIPT.json"
NEW_PHASE2 = REPO_ROOT / ".verification" / "e1_py310_numpy226" / "phase2_core" / "PHASE2_CORE_PHASE3_RECEIPT.json"
OLD_PHASE1 = REPO_ROOT / ".verification" / "e1_phase1" / "E1_RECEIPT.json"
OLD_PHASE2 = REPO_ROOT / ".verification" / "e1_phase2_core" / "PHASE2_CORE_PHASE3_RECEIPT.json"
GAMMA = 0.95
Q_TOLERANCE = 1.0e-5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def assert_declared_environment() -> dict[str, Any]:
    actual_python = sys.version_info[:3]
    if actual_python != DECLARED_PYTHON or np.__version__ != DECLARED_NUMPY:
        raise AssertionError(
            f"declared environment required, got Python {platform.python_version()} "
            f"NumPy {np.__version__}"
        )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "executable": str(Path(sys.executable).resolve()),
        "digest": canonical_digest(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "executable": str(Path(sys.executable).resolve()),
            }
        ),
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_baselines() -> dict[str, Any]:
    environment = assert_declared_environment()
    if sha256_file(OLD_PHASE1) != OLD_PHASE1_RECEIPT_SHA256:
        raise AssertionError("historical Phase 1 receipt changed")
    if sha256_file(OLD_PHASE2) != OLD_PHASE2_RECEIPT_SHA256:
        raise AssertionError("historical V1--V5 receipt changed")
    old1, new1 = load_json(OLD_PHASE1), load_json(NEW_PHASE1)
    old2, new2 = load_json(OLD_PHASE2), load_json(NEW_PHASE2)
    if old1["parity_digest"] != new1["parity_digest"]:
        raise AssertionError("Phase 1 semantic parity digest changed")
    if old1["registration"] != new1["registration"] or old1["components"] != new1["components"]:
        raise AssertionError("Phase 1 posterior/transition payload changed")
    if new2["build_validation_digest"] != CONTROLLING_BUILD_DIGEST:
        raise AssertionError("unexpected declared-environment build digest")

    cells: dict[str, Any] = {}
    for cid in sorted(old2["validation_episodes"]):
        cells[cid] = {}
        for arm in core.ARMS:
            old_episode = old2["validation_episodes"][cid][arm]
            new_episode = new2["validation_episodes"][cid][arm]
            if old_episode["actions_hash"] != new_episode["actions_hash"]:
                raise AssertionError(f"selected-action change at {cid}/{arm}")
            old_row = old2["hash_and_counter_table"][cid][arm]
            new_row = new2["hash_and_counter_table"][cid][arm]
            keys = (
                "model_parameter_hash", "transition_kernel_hash", "abundance_grid_hash",
                "capacity_grid_hash", "base_initial_belief_calls",
                "override_initial_belief_calls", "planner_invocations", "true_reward_calls",
            )
            mismatches = [key for key in keys if old_row.get(key) != new_row.get(key)]
            if mismatches:
                raise AssertionError(f"held-fixed drift {cid}/{arm}: {mismatches}")
            cells[cid][arm] = {
                "actions_hash": new_episode["actions_hash"],
                **{key: new_row.get(key) for key in keys},
            }
        if old2["hash_and_counter_table"][cid]["fit_artifact"] != new2["hash_and_counter_table"][cid]["fit_artifact"]:
            raise AssertionError(f"fit artifact drift at {cid}")
        if old2["validations"]["V4"]["cells"][cid] != new2["validations"]["V4"]["cells"][cid]:
            raise AssertionError(f"fit reuse drift at {cid}")

    for validation in ("V2", "V4", "V5"):
        if old2["validations"][validation] != new2["validations"][validation]:
            raise AssertionError(f"{validation} route/counter drift")
    if old2["actual_registered_transition_provenance"] != new2["actual_registered_transition_provenance"]:
        raise AssertionError("registered kernel provenance drift")

    result = {
        "schema": "e1_environment_baseline_comparison_v1",
        "status": "PASS",
        "created_utc": now(),
        "environment": environment,
        "historical_baseline": {
            "status": "SUPERSEDED_FOR_SUBSEQUENT_EXECUTION_NOT_INVALID",
            "build_validation_digest": OLD_BUILD_DIGEST,
            "phase1_receipt_sha256": OLD_PHASE1_RECEIPT_SHA256,
            "phase2_receipt_sha256": OLD_PHASE2_RECEIPT_SHA256,
            "python": old2["execution"]["python"],
            "numpy": old2["execution"]["numpy"],
        },
        "controlling_baseline": {
            "build_validation_digest": CONTROLLING_BUILD_DIGEST,
            "phase1_parity_digest": new1["parity_digest"],
            "phase1_receipt_sha256": sha256_file(NEW_PHASE1),
            "phase2_receipt_sha256": sha256_file(NEW_PHASE2),
        },
        "comparison": {
            "transition_matrices": "byte-identical full-kernel hashes; therefore numerical max error 0",
            "phase1_posterior_payload_identical": True,
            "validation_action_hashes_identical": True,
            "fit_artifact_identity_and_reuse_identical": True,
            "registered_and_fitted_kernel_provenance_identical": True,
            "initial_prior_routes_identical": True,
            "observation_update_routes_identical": True,
            "fallback_and_update_counters_identical": True,
            "semantic_changes": [],
            "expected_environment_only_changes": [
                "build_validation_digest", "execution runtime", "runtime object identities",
            ],
        },
        "cells": cells,
    }
    result["comparison_digest"] = canonical_digest(result)
    atomic_json(PHASE3_ROOT / "ENVIRONMENT_BASELINE_COMPARISON.json", result)
    return result


class AuditedPointBasedPlanner(PointBasedPlanner):
    """The deployed PBVI with passive instrumentation of its actual method calls."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.last_audit: dict[str, Any] = {}
        self._distance_group_size = 0
        self._distance_group_count = 0
        self._distance_group_min = math.inf
        self._nearest_minima: list[float] = []

    def _distance(self, left: CandidateBelief, right: CandidateBelief) -> float:
        distance = PointBasedPlanner._distance(left, right)
        self._distance_group_min = min(self._distance_group_min, distance)
        self._distance_group_count += 1
        if self._distance_group_count == self._distance_group_size:
            self._nearest_minima.append(self._distance_group_min)
            self._distance_group_count = 0
            self._distance_group_min = math.inf
        return distance

    def _reachable_graph(self, root: CandidateBelief) -> list[list[CandidateBelief]]:
        layers = super()._reachable_graph(root)
        self._last_layers = layers
        return layers

    def action_values(self, belief: CandidateBelief) -> np.ndarray:
        branch_records: list[dict[str, Any]] = []
        original = self.model.representative_observations

        def observed(predicted: np.ndarray, branch_count: int):
            observations, weights = original(predicted, branch_count)
            branch_records.append(
                {
                    "support_size": int(len(observations)),
                    "weight_sum": float(np.sum(weights)),
                    "weights": np.asarray(weights, dtype=np.float64).tolist(),
                }
            )
            return observations, weights

        self.model.representative_observations = observed
        self._distance_group_size = self.config.belief_points
        self._distance_group_count = 0
        self._distance_group_min = math.inf
        self._nearest_minima = []
        started = time.perf_counter()
        try:
            values = super().action_values(belief)
        finally:
            self.model.representative_observations = original
        elapsed = time.perf_counter() - started
        graph_calls = self.config.horizon * self.config.belief_points
        graph_records = branch_records[:graph_calls]
        backup_records = branch_records[graph_calls:]
        selected = []
        cursor = 0
        for depth in range(self.config.horizon):
            for _ in range(self.config.belief_points):
                record = graph_records[cursor]
                branch = (cursor % self.config.belief_points * 104729 + self.seed + depth) % record["support_size"]
                selected.append(
                    {
                        "depth": depth,
                        "branch_index": int(branch),
                        "support_size": record["support_size"],
                        "selected_mass": float(record["weights"][branch]),
                    }
                )
                cursor += 1
        layer_rows = []
        for depth, layer in enumerate(self._last_layers):
            digests = {
                hashlib.sha256(np.ascontiguousarray(item.probabilities, dtype="<f8").tobytes()).hexdigest()
                for item in layer
            }
            layer_rows.append(
                {"depth": depth, "size": len(layer), "unique_beliefs": len(digests)}
            )
        nearest = np.asarray(self._nearest_minima, dtype=np.float64)
        depth_coverage = []
        for depth in range(self.config.horizon):
            rows = [row for row in selected if row["depth"] == depth]
            available = max(row["support_size"] for row in rows)
            depth_coverage.append(
                {
                    "depth": depth,
                    "unique_selected_branches": len({row["branch_index"] for row in rows}),
                    "maximum_support_size": available,
                    "branch_index_coverage": len({row["branch_index"] for row in rows}) / available,
                }
            )
        if self._distance_group_count != 0:
            raise AssertionError("nearest-distance instrumentation lost group alignment")
        if any(abs(row["weight_sum"] - 1.0) > 1e-12 for row in branch_records):
            raise AssertionError("branch weights do not sum to one")
        self.last_audit = {
            "elapsed_seconds": elapsed,
            "graph_layers": layer_rows,
            "duplicate_point_rate": float(
                1.0 - sum(row["unique_beliefs"] for row in layer_rows) / max(sum(row["size"] for row in layer_rows), 1)
            ),
            "graph_selected_branches": selected,
            "graph_selected_mass_mean": float(np.mean([row["selected_mass"] for row in selected])),
            "graph_selected_mass_min": float(np.min([row["selected_mass"] for row in selected])),
            "graph_branch_index_coverage": depth_coverage,
            "graph_branch_index_coverage_mean": float(
                np.mean([row["branch_index_coverage"] for row in depth_coverage])
            ),
            "predictive_support_size_min": min(row["support_size"] for row in branch_records),
            "predictive_support_size_max": max(row["support_size"] for row in branch_records),
            "backup_branch_calls": len(backup_records),
            "backup_branch_weight_sum_min": min(row["weight_sum"] for row in backup_records),
            "backup_branch_weight_sum_max": max(row["weight_sum"] for row in backup_records),
            "nearest_point_distance_min": float(nearest.min()) if len(nearest) else 0.0,
            "nearest_point_distance_mean": float(nearest.mean()) if len(nearest) else 0.0,
            "nearest_point_distance_max": float(nearest.max()) if len(nearest) else 0.0,
        }
        return values


class SharedObjectQMDP:
    """Finite-horizon QMDP using only one already-deployed CandidatePOMDP object."""

    def __init__(self, pomdp: Any, horizon: int = 5, gamma: float = GAMMA):
        self.pomdp = pomdp
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.transition_calls = 0
        self.reward_calls = 0
        self._memo: dict[tuple[Any, ...], np.ndarray] = {}

    def _state_values(
        self, remaining: int, capacity: float, timestep: int,
        previous_observation: float, current_observation: float,
    ) -> np.ndarray:
        if remaining == 0:
            return np.zeros(self.pomdp.hidden_count, dtype=np.float64)
        key = (
            remaining, float(capacity), int(timestep),
            float(previous_observation), float(current_observation),
        )
        if key in self._memo:
            return self._memo[key]
        q = np.zeros((self.pomdp.hidden_count, self.pomdp.context.num_actions), dtype=np.float64)
        raw_states = self.pomdp.state_abundances() * self.pomdp.model.survey_scale
        for action in range(self.pomdp.context.num_actions):
            matrix = self.pomdp.transition_matrix(capacity, action)
            self.transition_calls += 1
            following_capacity = self.pomdp.model.next_capacity(capacity, action)
            continuation = self._state_values(
                remaining - 1, following_capacity, timestep + 1,
                current_observation, current_observation,
            )
            immediate = np.zeros(self.pomdp.hidden_count, dtype=np.float64)
            for state in range(self.pomdp.hidden_count):
                probabilities = np.zeros(self.pomdp.hidden_count, dtype=np.float64)
                probabilities[state] = 1.0
                belief = CandidateBelief(
                    probabilities, capacity, previous_observation,
                    float(raw_states[state]), timestep,
                )
                immediate[state] = self.pomdp.expected_public_reward(
                    belief, action, matrix[state]
                )
                self.reward_calls += 1
            q[:, action] = immediate + self.gamma * (matrix @ continuation)
        value = np.max(q, axis=1)
        if not np.all(np.isfinite(value)):
            raise AssertionError("nonfinite QMDP state value")
        self._memo[key] = value
        return value

    def action_values(self, belief: CandidateBelief) -> np.ndarray:
        values = np.zeros(self.pomdp.context.num_actions, dtype=np.float64)
        for action in range(self.pomdp.context.num_actions):
            predicted = self.pomdp.predict(belief, action)
            self.transition_calls += 1
            immediate = self.pomdp.expected_public_reward(belief, action, predicted)
            self.reward_calls += 1
            continuation = self._state_values(
                self.horizon - 1,
                self.pomdp.model.next_capacity(belief.capacity, action),
                belief.timestep + 1,
                belief.current_observation,
                belief.current_observation,
            )
            values[action] = immediate + self.gamma * float(np.dot(predicted, continuation))
        if not np.all(np.isfinite(values)):
            raise AssertionError("nonfinite QMDP action values")
        return values


class TinyModel:
    survey_scale = 1.0

    @staticmethod
    def next_capacity(capacity: float, action: int) -> float:
        del action
        return float(capacity)


class TinyProbePOMDP:
    """Two-state CandidatePOMDP interface for exact QMDP/PBVI validation."""

    def __init__(self, information_probe: bool):
        self.information_probe = information_probe
        self.hidden_count = 2
        self.context = SimpleNamespace(num_actions=3 if information_probe else 2)
        self.model = TinyModel()
        self.config = FaithfulPlannerConfig(
            state_bins=5, capacity_bins=2, observation_bins=5,
            transition_samples=1, observation_samples=1,
            belief_points=12, observation_branches=2,
            horizon=2 if information_probe else 3,
        )

    def state_abundances(self) -> np.ndarray:
        return np.asarray([0.0, 1.0])

    def transition_matrix(self, capacity: float, action: int) -> np.ndarray:
        del capacity, action
        return np.eye(2)

    def predict(self, belief: CandidateBelief, action: int) -> np.ndarray:
        return belief.probabilities @ self.transition_matrix(belief.capacity, action)

    def expected_public_reward(
        self, belief: CandidateBelief, action: int, predicted: np.ndarray,
    ) -> float:
        del predicted
        if not self.information_probe:
            return (0.2, 0.1)[action]
        if action == 0:
            return 0.1
        correct_state = action - 1
        return float(belief.probabilities[correct_state])

    def representative_observations(
        self, predicted: np.ndarray, branch_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        del branch_count
        return np.asarray([0.0, 1.0]), predicted.copy()

    def update(
        self, belief: CandidateBelief, action: int, observation: float,
    ) -> tuple[CandidateBelief, float]:
        if self.information_probe and action != 0:
            probabilities = belief.probabilities.copy()
        else:
            probabilities = np.zeros(2)
            probabilities[int(observation)] = 1.0
        return CandidateBelief(
            probabilities, belief.capacity, belief.current_observation,
            observation, belief.timestep + 1,
        ), 0.0

    def model_hash(self) -> str:
        return f"tiny_probe_{int(self.information_probe)}"


def assert_no_native_solver() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    forbidden = [name for name in imports if name.endswith("native_solver")]
    loaded = [name for name in sys.modules if name.endswith("native_solver")]
    if forbidden or loaded:
        raise AssertionError(f"native solver access: imports={forbidden}, loaded={loaded}")
    return {"static_imports": forbidden, "runtime_modules": loaded, "status": "PASS"}


def validate_qmdp() -> dict[str, Any]:
    assert_declared_environment()
    native_guard = assert_no_native_solver()
    full = TinyProbePOMDP(False)
    full_root = CandidateBelief(np.asarray([0.4, 0.6]), 1.0, 0.0, 0.0, 0)
    full_pbvi = PointBasedPlanner(full, full.config, GAMMA, 61101).action_values(full_root)
    full_qmdp = SharedObjectQMDP(full, horizon=3).action_values(full_root)
    full_exact = np.asarray([0.5705, 0.4705])
    full_error = float(max(np.max(abs(full_pbvi - full_exact)), np.max(abs(full_qmdp - full_exact))))
    if full_error > Q_TOLERANCE or int(np.argmax(full_pbvi)) != int(np.argmax(full_qmdp)):
        raise AssertionError("fully observable QMDP validation failed")

    info = TinyProbePOMDP(True)
    info_root = CandidateBelief(np.asarray([0.5, 0.5]), 1.0, 0.0, 0.0, 0)
    info_pbvi = PointBasedPlanner(info, info.config, GAMMA, 61102).action_values(info_root)
    info_qmdp = SharedObjectQMDP(info, horizon=2).action_values(info_root)
    if int(np.argmax(info_pbvi)) != 0 or abs(info_pbvi[0] - 1.05) > Q_TOLERANCE:
        raise AssertionError(f"information PBVI failed: {info_pbvi}")
    if int(np.argmax(info_qmdp)) not in (1, 2) or abs(np.max(info_qmdp) - 1.45) > Q_TOLERANCE:
        raise AssertionError(f"information QMDP failed: {info_qmdp}")
    result = {
        "schema": "e1_qmdp_exact_validation_v1",
        "status": "PASS",
        "created_utc": now(),
        "fully_observable": {
            "exact": full_exact.tolist(), "pbvi": full_pbvi.tolist(),
            "qmdp": full_qmdp.tolist(), "maximum_error": full_error,
        },
        "information_gathering": {
            "exact_pbvi_selected_action": 0,
            "exact_pbvi_value": 1.05,
            "pbvi_values": info_pbvi.tolist(),
            "qmdp_selected_action": int(np.argmax(info_qmdp)),
            "qmdp_value": float(np.max(info_qmdp)),
            "qmdp_values": info_qmdp.tolist(),
        },
        "native_solver_guard": native_guard,
        "implementation_digest": hashlib.sha256(inspect.getsource(SharedObjectQMDP).encode()).hexdigest(),
        "recursion_continuation_sign": "PLUS",
    }
    result["validation_digest"] = canonical_digest(result)
    atomic_json(PHASE3_ROOT / "QMDP_EXACT_VALIDATION.json", result)
    return result


def rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def centred(values: np.ndarray) -> np.ndarray:
    return values - float(np.max(values))


def ranking(values: np.ndarray) -> list[int]:
    return np.argsort(-values, kind="stable").astype(int).tolist()


def top_margin(values: np.ndarray) -> float:
    ordered = np.sort(values)
    return float(ordered[-1] - ordered[-2])


def public_transition(result: Any) -> PublicTransition:
    return PublicTransition(
        observation=result.observation,
        done=result.done,
        truncated=result.truncated,
        terminated=bool(result.done and not result.truncated),
        public_info=result.public_info.copy(),
    )


def pomdp_identity(pomdp: Any) -> dict[str, Any]:
    return {
        "object_id": id(pomdp),
        "model_object_id": id(pomdp.model),
        "model_parameter_hash": pomdp.model.parameter_hash(),
        "transition_kernel_hash": core.full_transition_kernel_hash(pomdp),
        "raw_transition_matrix_hash": raw_transition_matrix_hash(pomdp),
        "abundance_grid_hash": core.array_hash(pomdp.abundance_grid),
        "capacity_grid_hash": core.array_hash(core.capacity_grid(pomdp)),
        "wrapper_digest": core.diagnostic_wrapper_code_digest(),
        "observation_update_digest": hashlib.sha256(
            (inspect.getsource(core.StableObservationPOMDP) + inspect.getsource(core.NoisyStateE1POMDP.update)).encode()
        ).hexdigest(),
        "objective_flag": pomdp.objective_flag,
    }


def raw_transition_matrix_hash(pomdp: Any) -> str:
    digest = hashlib.sha256()
    for capacity in core.capacity_grid(pomdp):
        for action in range(pomdp.context.num_actions):
            digest.update(np.asarray([capacity, action], dtype="<f8").tobytes())
            digest.update(
                np.ascontiguousarray(
                    pomdp.transition_matrix(float(capacity), action), dtype="<f8"
                ).tobytes()
            )
    return digest.hexdigest()


def build_config(**changes: Any) -> FaithfulPlannerConfig:
    config = replace(core.planner_config(), **changes)
    config.validate()
    return config


def make_cell(population: str, sigma: float, config: FaithfulPlannerConfig) -> dict[str, Any]:
    cid = core.cell_id(population, sigma)
    base = core.build_arms()[cid]
    if config == core.planner_config():
        return base
    registered = base["policies"]["A3"].pomdp.model
    fitted = base["policies"]["A4"].pomdp.model
    reg_context = core.method_context(base["actions"], sigma, registered.survey_scale)
    fit_context = core.method_context(base["actions"], sigma, fitted.survey_scale)
    seed = base["planning_seed"]
    base["policies"] = {
        "A1": core.TrueStatePolicy("A1", core.TrueStateE1POMDP(registered, reg_context, config, seed, base["env_cfg"], "A1"), seed),
        "A2": core.TrueStatePolicy("A2", core.TrueStateE1POMDP(fitted, fit_context, config, seed, base["env_cfg"], "A2"), seed),
        "A3": core.NoisyStatePolicy("A3", core.NoisyStateE1POMDP(registered, reg_context, config, seed, base["env_cfg"], "A3", True), seed),
        "A4": core.NoisyStatePolicy("A4", core.NoisyStateE1POMDP(fitted, fit_context, config, seed, base["env_cfg"], "A4", False), seed),
    }
    return base


def replace_with_audited(policy: Any) -> AuditedPointBasedPlanner:
    planner = AuditedPointBasedPlanner(policy.pomdp, policy.pomdp.config, GAMMA, policy.pomdp.seed)
    policy.planner = planner
    return planner


def common_cdf_metrics(left: CandidateBelief, left_pomdp: Any, right: CandidateBelief, right_pomdp: Any) -> dict[str, float]:
    lx = left_pomdp.state_abundances() * left_pomdp.model.survey_scale
    rx = right_pomdp.state_abundances() * right_pomdp.model.survey_scale
    grid = np.unique(np.concatenate((lx, rx)))
    lp = np.asarray([left.probabilities[lx <= x].sum() for x in grid])
    rp = np.asarray([right.probabilities[rx <= x].sum() for x in grid])
    wasserstein = float(np.sum(np.abs(lp[:-1] - rp[:-1]) * np.diff(grid))) if len(grid) > 1 else 0.0
    return {
        "wasserstein_1": wasserstein,
        "common_cdf_max_error": float(np.max(np.abs(lp - rp))),
        "raw_mean_error": abs(float(np.dot(left.probabilities, lx)) - float(np.dot(right.probabilities, rx))),
    }


def run_v6() -> dict[str, Any]:
    assert_declared_environment()
    population, sigma = "Crab-eating fox", 0.1
    cid = core.cell_id(population, sigma)
    base_cell = make_cell(population, sigma, build_config(state_bins=41, observation_bins=41))
    fine_cell = make_cell(population, sigma, build_config(state_bins=61, observation_bins=41))
    base_policy = base_cell["policies"]["A3"]
    fine_policy = fine_cell["policies"]["A3"]
    base_planner, fine_planner = replace_with_audited(base_policy), replace_with_audited(fine_policy)
    base_identity, fine_identity = pomdp_identity(base_policy.pomdp), pomdp_identity(fine_policy.pomdp)
    allowed_config = {"state_bins"}
    config_diff = {
        key for key in asdict(base_policy.pomdp.config)
        if asdict(base_policy.pomdp.config)[key] != asdict(fine_policy.pomdp.config)[key]
    }
    if config_diff != allowed_config:
        raise AssertionError(f"V6 config drift: {config_diff}")
    held_keys = ("model_parameter_hash", "capacity_grid_hash", "wrapper_digest", "observation_update_digest", "objective_flag")
    if any(base_identity[key] != fine_identity[key] for key in held_keys):
        raise AssertionError("V6 held-fixed identity drift")

    env = ContinuousEcologyEnv(base_cell["env_cfg"])
    reset = env.reset(core.VALIDATION_SEED)
    base_belief = base_policy.pomdp.initial_belief(float(reset.observation))
    fine_belief = fine_policy.pomdp.initial_belief(float(reset.observation))
    records = []
    actions = []
    started = time.perf_counter()
    for step in range(50):
        bq = base_planner.action_values(base_belief)
        fq = fine_planner.action_values(fine_belief)
        if not np.all(np.isfinite(bq)) or not np.all(np.isfinite(fq)):
            raise AssertionError("V6 nonfinite action values")
        ba, fa = int(np.argmax(bq)), int(np.argmax(fq))
        metrics = common_cdf_metrics(base_belief, base_policy.pomdp, fine_belief, fine_policy.pomdp)
        records.append({
            "step": step, "baseline_q": bq.tolist(), "comparison_q": fq.tolist(),
            "baseline_centred": centred(bq).tolist(), "comparison_centred": centred(fq).tolist(),
            "baseline_action": ba, "comparison_action": fa,
            "baseline_ranking": ranking(bq), "comparison_ranking": ranking(fq),
            "baseline_margin": top_margin(bq), "comparison_margin": top_margin(fq),
            "belief_metrics": metrics,
            "baseline_zero_mass": float(base_belief.probabilities[0]),
            "comparison_zero_mass": float(fine_belief.probabilities[0]),
            "baseline_probability_sum": float(base_belief.probabilities.sum()),
            "comparison_probability_sum": float(fine_belief.probabilities.sum()),
            "baseline_graph": base_planner.last_audit,
            "comparison_graph": fine_planner.last_audit,
        })
        if ba != fa:
            result = {
                "schema": "e1_v6_discretisation_v1", "status": "FAILED_ACTION_SENSITIVITY",
                "created_utc": now(), "cell": cid, "arm": "A3", "records": records,
                "failure_step": step, "baseline_identity": base_identity,
                "comparison_identity": fine_identity,
            }
            atomic_json(PHASE3_ROOT / "V6_DISCRETISATION.json", result)
            raise AssertionError(f"V6 selected action differs at step {step}")
        result = env.step(ba)
        actions.append(ba)
        base_belief, _ = base_policy.pomdp.runtime_update(base_belief, ba, float(result.observation))
        fine_belief, _ = fine_policy.pomdp.runtime_update(fine_belief, ba, float(result.observation))
        if result.done and step != 49:
            raise AssertionError("V6 validation replay ended early")
    action_hash = core.array_hash(np.asarray(actions, dtype=np.float64))
    controlling_hash = load_json(NEW_PHASE2)["validation_episodes"][cid]["A3"]["actions_hash"]
    if action_hash != controlling_hash:
        raise AssertionError("V6 41-bin action hash did not reproduce controlling path")
    if base_policy.pomdp.old_fallback_calls or fine_policy.pomdp.old_fallback_calls:
        raise AssertionError("V6 old fallback reached")
    n0 = base_cell["env_cfg"].N0
    output = {
        "schema": "e1_v6_discretisation_v1", "status": "PASS", "created_utc": now(),
        "controlling_baseline_digest": CONTROLLING_BUILD_DIGEST,
        "cell": cid, "arm": "A3", "validation_seed": core.VALIDATION_SEED,
        "planning_seed": core.PLANNING_SEEDS[cid], "config_difference": sorted(config_diff),
        "action_hash": action_hash, "action_disagreement_count": 0,
        "action_disagreement_locations": [], "baseline_identity": base_identity,
        "comparison_identity": fine_identity,
        "initial_prior_raw_snapping_error": {
            "41": abs(float(base_policy.pomdp.abundance_grid[np.argmax(base_policy.pomdp.initial_belief(float(reset.observation)).probabilities)]) * base_policy.pomdp.model.survey_scale) - n0,
            "61": abs(float(fine_policy.pomdp.abundance_grid[np.argmax(fine_policy.pomdp.initial_belief(float(reset.observation)).probabilities)]) * fine_policy.pomdp.model.survey_scale) - n0,
        },
        "records": records, "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": rss_mib(), "fallback_calls": 0,
        "scientific_returns_read": False,
    }
    output["probe_digest"] = canonical_digest(output)
    atomic_json(PHASE3_ROOT / "V6_DISCRETISATION.json", output)
    return output


def run_v8() -> dict[str, Any]:
    environment = assert_declared_environment()
    qmdp_validation = load_json(PHASE3_ROOT / "QMDP_EXACT_VALIDATION.json")
    if qmdp_validation["status"] != "PASS":
        raise AssertionError("QMDP exact validation is not passing")
    if load_json(PHASE3_ROOT / "V6_DISCRETISATION.json")["status"] != "PASS":
        raise AssertionError("V6 is not passing")
    native_before = assert_no_native_solver()
    cells = core.build_arms()
    controlling = load_json(NEW_PHASE2)
    output_cells: dict[str, Any] = {}
    started = time.perf_counter()
    for cid, cell in cells.items():
        output_cells[cid] = {}
        for arm in ("A3", "A4"):
            policy = cell["policies"][arm]
            planner = replace_with_audited(policy)
            qmdp = SharedObjectQMDP(policy.pomdp, horizon=5, gamma=GAMMA)
            if not (qmdp.pomdp is planner.model is policy.pomdp):
                raise AssertionError("V8 QMDP/PBVI POMDP object identity failed")
            identity_before = pomdp_identity(policy.pomdp)
            env = ContinuousEcologyEnv(cell["env_cfg"])
            reset = env.reset(core.VALIDATION_SEED)
            belief = policy.pomdp.initial_belief(float(reset.observation))
            actions = []
            records = []
            arm_started = time.perf_counter()
            for step in range(50):
                pbvi_values = planner.action_values(belief)
                qmdp_values = qmdp.action_values(belief)
                pbvi_action = int(np.argmax(pbvi_values))
                qmdp_action = int(np.argmax(qmdp_values))
                records.append(
                    {
                        "step": step,
                        "pbvi_q": pbvi_values.tolist(),
                        "qmdp_q": qmdp_values.tolist(),
                        "pbvi_centred": centred(pbvi_values).tolist(),
                        "qmdp_centred": centred(qmdp_values).tolist(),
                        "pbvi_action": pbvi_action,
                        "qmdp_action": qmdp_action,
                        "pbvi_ranking": ranking(pbvi_values),
                        "qmdp_ranking": ranking(qmdp_values),
                        "pbvi_margin": top_margin(pbvi_values),
                        "qmdp_margin": top_margin(qmdp_values),
                        "pbvi_graph": planner.last_audit,
                    }
                )
                result = env.step(pbvi_action)
                actions.append(pbvi_action)
                belief, _ = policy.pomdp.runtime_update(
                    belief, pbvi_action, float(result.observation)
                )
                if result.done and step != 49:
                    raise AssertionError(f"V8 replay ended early at {cid}/{arm}")
            action_hash = core.array_hash(np.asarray(actions, dtype=np.float64))
            expected_hash = controlling["validation_episodes"][cid][arm]["actions_hash"]
            if action_hash != expected_hash:
                raise AssertionError(f"V8 PBVI baseline action drift at {cid}/{arm}")
            if policy.pomdp.old_fallback_calls:
                raise AssertionError(f"V8 old fallback at {cid}/{arm}")
            identity_after = pomdp_identity(policy.pomdp)
            for key in identity_before:
                if key not in {"object_id", "model_object_id"} and identity_before[key] != identity_after[key]:
                    raise AssertionError(f"V8 identity drift at {cid}/{arm}/{key}")
            disagreement = [
                row["step"] for row in records
                if row["pbvi_action"] != row["qmdp_action"]
            ]
            output_cells[cid][arm] = {
                "status": "PASS_DIAGNOSTIC",
                "identity": identity_before,
                "same_candidate_pomdp_object": True,
                "pbvi_action_hash": action_hash,
                "disagreement_locations": disagreement,
                "disagreement_count": len(disagreement),
                "disagreement_rate": len(disagreement) / 50.0,
                "qmdp_transition_calls": qmdp.transition_calls,
                "qmdp_reward_calls": qmdp.reward_calls,
                "elapsed_seconds": time.perf_counter() - arm_started,
                "peak_rss_mib": rss_mib(),
                "records": records,
            }
    native_after = assert_no_native_solver()
    output = {
        "schema": "e1_v8_qmdp_no_return_v1",
        "status": "PASS_DIAGNOSTIC",
        "created_utc": now(),
        "controlling_baseline_digest": CONTROLLING_BUILD_DIGEST,
        "environment": environment,
        "qmdp_validation_digest": qmdp_validation["validation_digest"],
        "native_solver_guard_before": native_before,
        "native_solver_guard_after": native_after,
        "cells": output_cells,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mib": rss_mib(),
        "scientific_returns_read": False,
    }
    output["probe_digest"] = canonical_digest(output)
    atomic_json(PHASE3_ROOT / "V8_QMDP_COMPARISONS.json", output)
    return output


V7_TASKS = [
    (cid, arm)
    for cid in (
        "fox_ricker_sigma_0.1", "fox_ricker_sigma_0.2",
        "tiger_ricker_sigma_0.1", "tiger_ricker_sigma_0.2",
    )
    for arm in ("A2", "A3", "A4")
]


def v7_root() -> Path:
    return PHASE3_ROOT / "V7" / CONTROLLING_BUILD_DIGEST


def task_root(cid: str, arm: str) -> Path:
    return v7_root() / cid / arm


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def trace_policy_roots(cell: dict[str, Any], arm: str) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    policy = cell["policies"][arm]
    planner = replace_with_audited(policy)
    env = ContinuousEcologyEnv(cell["env_cfg"])
    reset = env.reset(core.VALIDATION_SEED)
    if arm == "A2":
        filt = core.oracle_filter(cell["env_cfg"])
        external = filt.set_true_state(float(reset.evaluator_info["state"]), 0, reset.public_info)
    else:
        external = core.raw_filter(cell["env_cfg"]).reset(
            reset.observation, core.VALIDATION_SEED + 10_000
        )
    policy.reset(core.VALIDATION_SEED + 20_000)
    observation = float(reset.observation)
    roots, capacities, previous, current, timesteps, actions, q_values = [], [], [], [], [], [], []
    audits = []
    for step in range(50):
        action = int(policy.act(external, observation))
        root = policy.internal_belief
        if root is None:
            raise AssertionError("V7 policy did not expose its deployed internal belief")
        roots.append(root.probabilities.copy())
        capacities.append(root.capacity)
        previous.append(root.previous_observation)
        current.append(root.current_observation)
        timesteps.append(root.timestep)
        actions.append(action)
        q_values.append(np.asarray(policy.last_diagnostics["action_scores"], dtype=np.float64))
        audits.append(planner.last_audit)
        result = env.step(action)
        policy.observe(external, action, public_transition(result))
        if arm == "A2":
            external = filt.set_true_state(
                float(result.evaluator_info["state"]), step + 1, result.public_info
            )
        observation = float(result.observation)
        if result.done and step != 49:
            raise AssertionError("V7 query replay ended early")
    return {
        "probabilities": np.asarray(roots),
        "capacity": np.asarray(capacities),
        "previous_observation": np.asarray(previous),
        "current_observation": np.asarray(current),
        "timestep": np.asarray(timesteps, dtype=np.int64),
        "action": np.asarray(actions, dtype=np.int64),
        "q_values": np.asarray(q_values),
    }, audits


def checkpoint_payload(
    root_index: int, values: np.ndarray, audit: dict[str, Any], provenance_digest: str,
) -> dict[str, Any]:
    return {
        "schema": "e1_v7_root_checkpoint_v1",
        "status": "COMPLETE",
        "root_index": root_index,
        "q_values": values.tolist(),
        "centred_advantages": centred(values).tolist(),
        "selected_action": int(np.argmax(values)),
        "ranking": ranking(values),
        "top_two_margin": top_margin(values),
        "planner_audit": audit,
        "provenance_digest": provenance_digest,
        "scientific_returns_read": False,
    }


def prepare_v7() -> dict[str, Any]:
    environment = assert_declared_environment()
    if load_json(PHASE3_ROOT / "V8_QMDP_COMPARISONS.json")["status"] != "PASS_DIAGNOSTIC":
        raise AssertionError("V8 is not complete")
    cells = core.build_arms()
    controlling = load_json(NEW_PHASE2)
    task_manifests = []
    for index, (cid, arm) in enumerate(V7_TASKS):
        cell = cells[cid]
        arrays, audits = trace_policy_roots(cell, arm)
        action_hash = core.array_hash(arrays["action"].astype(np.float64))
        expected_hash = controlling["validation_episodes"][cid][arm]["actions_hash"]
        if action_hash != expected_hash:
            raise AssertionError(f"V7 query baseline drift at {cid}/{arm}")
        root = task_root(cid, arm)
        bank_path = root / "query_bank.npz"
        if bank_path.exists():
            raise AssertionError(f"V7 query bank output collision: {bank_path}")
        atomic_npz(bank_path, **arrays)
        identity = pomdp_identity(cell["policies"][arm].pomdp)
        manifest = {
            "schema": "e1_v7_task_manifest_v1",
            "status": "PREPARED",
            "task_index": index,
            "cell": cid,
            "arm": arm,
            "controlling_baseline_digest": CONTROLLING_BUILD_DIGEST,
            "query_bank_path": str(bank_path.resolve()),
            "query_bank_hash": sha256_file(bank_path),
            "query_root_count": 50,
            "baseline_action_hash": action_hash,
            "planning_seed": core.PLANNING_SEEDS[cid],
            "validation_seed": core.VALIDATION_SEED,
            "environment": environment,
            "code_digest": sha256_file(Path(__file__)),
            "identity": identity,
            "baseline_config": asdict(cell["policies"][arm].pomdp.config),
            "expanded_config": asdict(build_config(horizon=15, belief_points=256)),
            "config_difference_whitelist": ["belief_points", "horizon"],
            "scientific_returns_read": False,
        }
        difference = {
            key for key in manifest["baseline_config"]
            if manifest["baseline_config"][key] != manifest["expanded_config"][key]
        }
        if difference != {"horizon", "belief_points"}:
            raise AssertionError(f"V7 config whitelist failed at {cid}/{arm}")
        manifest["provenance_digest"] = canonical_digest(manifest)
        atomic_json(root / "task_manifest.json", manifest)
        baseline_dir = root / "horizon_5_beliefs_32"
        for root_index in range(50):
            payload = checkpoint_payload(
                root_index, arrays["q_values"][root_index], audits[root_index],
                manifest["provenance_digest"],
            )
            atomic_json(baseline_dir / f"root_{root_index:03d}.json", payload)
        atomic_json(
            baseline_dir / "manifest.json",
            {
                "status": "COMPLETE", "budget": {"horizon": 5, "belief_points": 32},
                "root_count": 50, "provenance_digest": manifest["provenance_digest"],
                "action_hash": action_hash,
            },
        )
        task_manifests.append(manifest)
    master = {
        "schema": "e1_v7_master_manifest_v1", "status": "PREPARED",
        "created_utc": now(), "controlling_baseline_digest": CONTROLLING_BUILD_DIGEST,
        "environment": environment, "tasks": task_manifests,
        "query_bank_hash": canonical_digest([row["query_bank_hash"] for row in task_manifests]),
        "array": "0-11%12", "scientific_returns_read": False,
    }
    master["manifest_digest"] = canonical_digest(master)
    atomic_json(v7_root() / "MASTER_MANIFEST.json", master)
    (v7_root() / "slurm").mkdir(parents=True, exist_ok=True)
    return master


def verify_checkpoint(path: Path, provenance_digest: str) -> bool:
    if not path.exists():
        return False
    payload = load_json(path)
    if payload.get("status") != "COMPLETE" or payload.get("provenance_digest") != provenance_digest:
        raise AssertionError(f"incompatible V7 checkpoint: {path}")
    return True


def run_v7_task(index: int, smoke_only: bool = False) -> dict[str, Any]:
    environment = assert_declared_environment()
    if not 0 <= index < len(V7_TASKS):
        raise ValueError("V7 task index outside 0..11")
    cid, arm = V7_TASKS[index]
    root = task_root(cid, arm)
    manifest = load_json(root / "task_manifest.json")
    if manifest["environment"] != environment or manifest["code_digest"] != sha256_file(Path(__file__)):
        raise AssertionError("V7 task environment/code drift")
    bank_path = Path(manifest["query_bank_path"])
    if sha256_file(bank_path) != manifest["query_bank_hash"]:
        raise AssertionError("V7 query bank hash drift")
    expanded_cell = make_cell(
        "Crab-eating fox" if cid.startswith("fox") else "Amur tiger",
        0.1 if cid.endswith("0.1") else 0.2,
        build_config(horizon=15, belief_points=256),
    )
    policy = expanded_cell["policies"][arm]
    identity = pomdp_identity(policy.pomdp)
    held = ("model_parameter_hash", "raw_transition_matrix_hash", "abundance_grid_hash", "capacity_grid_hash", "wrapper_digest", "observation_update_digest", "objective_flag")
    for key in held:
        if identity[key] != manifest["identity"][key]:
            raise AssertionError(f"V7 held-fixed drift {cid}/{arm}/{key}")
    planner = replace_with_audited(policy)
    with np.load(bank_path, allow_pickle=False) as bank:
        arrays = {key: np.asarray(bank[key]) for key in bank.files}
    output_dir = root / "horizon_15_beliefs_256"
    count = 1 if smoke_only else 50
    started = time.perf_counter()
    for root_index in range(count):
        path = output_dir / f"root_{root_index:03d}.json"
        if verify_checkpoint(path, manifest["provenance_digest"]):
            continue
        belief = CandidateBelief(
            arrays["probabilities"][root_index], float(arrays["capacity"][root_index]),
            float(arrays["previous_observation"][root_index]),
            float(arrays["current_observation"][root_index]), int(arrays["timestep"][root_index]),
        )
        values = planner.action_values(belief)
        if not np.all(np.isfinite(values)):
            raise AssertionError("V7 nonfinite action values")
        payload = checkpoint_payload(
            root_index, values, planner.last_audit, manifest["provenance_digest"]
        )
        payload["elapsed_seconds"] = planner.last_audit["elapsed_seconds"]
        payload["peak_rss_mib"] = rss_mib()
        atomic_json(path, payload)
    if smoke_only:
        smoke = {
            "status": "PASS", "task_index": index, "cell": cid, "arm": arm,
            "constructed_only_one_root": True, "elapsed_seconds": time.perf_counter() - started,
            "scientific_returns_read": False,
        }
        atomic_json(root / "SMOKE.json", smoke)
        return smoke
    expanded_rows = [load_json(output_dir / f"root_{i:03d}.json") for i in range(50)]
    baseline_rows = [load_json(root / "horizon_5_beliefs_32" / f"root_{i:03d}.json") for i in range(50)]
    disagreements = [i for i in range(50) if expanded_rows[i]["selected_action"] != baseline_rows[i]["selected_action"]]
    base_coverage = float(np.mean([row["planner_audit"]["graph_branch_index_coverage_mean"] for row in baseline_rows]))
    expanded_coverage = float(np.mean([row["planner_audit"]["graph_branch_index_coverage_mean"] for row in expanded_rows]))
    base_distance = float(np.mean([row["planner_audit"]["nearest_point_distance_mean"] for row in baseline_rows]))
    expanded_distance = float(np.mean([row["planner_audit"]["nearest_point_distance_mean"] for row in expanded_rows]))
    escalation_reasons = []
    if cid.startswith("fox") and arm in {"A2", "A4"} and disagreements:
        escalation_reasons.append("fox_A2_A4_selected_action_change")
    if not (cid.startswith("fox") and arm in {"A2", "A4"}) and len(disagreements) > 2:
        escalation_reasons.append("selected_action_changes_above_5_percent")
    if expanded_coverage <= base_coverage:
        escalation_reasons.append("graph_support_coverage_not_improved")
    if expanded_distance >= base_distance:
        escalation_reasons.append("nearest_point_distance_not_improved")
    final = {
        "schema": "e1_v7_task_result_v1", "status": "COMPLETE",
        "task_index": index, "cell": cid, "arm": arm,
        "completed_roots": 50, "disagreement_locations": disagreements,
        "disagreement_rate": len(disagreements) / 50.0,
        "baseline_action_hash": manifest["baseline_action_hash"],
        "expanded_action_hash": core.array_hash(np.asarray([row["selected_action"] for row in expanded_rows], dtype=np.float64)),
        "baseline_coverage": base_coverage, "expanded_coverage": expanded_coverage,
        "baseline_nearest_distance": base_distance, "expanded_nearest_distance": expanded_distance,
        "scientific_escalation": bool(escalation_reasons),
        "escalation_reasons": escalation_reasons,
        "elapsed_seconds": time.perf_counter() - started, "peak_rss_mib": rss_mib(),
        "provenance_digest": manifest["provenance_digest"],
        "scientific_returns_read": False,
    }
    final["result_digest"] = canonical_digest(final)
    atomic_json(output_dir / "manifest.json", final)
    return final


def command_result(payload: dict[str, Any]) -> int:
    print(json.dumps({key: payload[key] for key in payload if key in {"status", "comparison_digest", "validation_digest", "probe_digest", "manifest_digest", "task_index", "cell", "arm", "elapsed_seconds"}}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("compare-baselines")
    sub.add_parser("validate-qmdp")
    sub.add_parser("run-v6")
    sub.add_parser("run-v8")
    sub.add_parser("prepare-v7")
    worker = sub.add_parser("run-v7-task")
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.command == "compare-baselines":
        return command_result(compare_baselines())
    if args.command == "validate-qmdp":
        return command_result(validate_qmdp())
    if args.command == "run-v6":
        return command_result(run_v6())
    if args.command == "run-v8":
        return command_result(run_v8())
    if args.command == "prepare-v7":
        return command_result(prepare_v7())
    return command_result(run_v7_task(args.task_index, args.smoke_only))


if __name__ == "__main__":
    raise SystemExit(main())
