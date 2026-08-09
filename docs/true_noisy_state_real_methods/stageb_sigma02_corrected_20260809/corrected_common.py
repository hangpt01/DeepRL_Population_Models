#!/usr/bin/env python3
"""External corrected-Stage-B contracts, serialization, and evidence helpers.

This module is deliberately outside ``src/tracks/**``.  It never opens an original
truth archive.  Arm-T offline state enters only through the already-sealed derivative;
runtime state enters as one capability-gated scalar supplied by the evaluator seam.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np


LABEL = "PROVISIONAL CORRECTED REPLICATION — INDEPENDENT AUDIT REQUIRED"
END_TO_END = "MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY"
ECOLOGICAL = {
    "plus_adapted_ricker_only_pbvi",
    "moor_adapted_ricker_misspec_pbvi",
}
GENERAL_TRANSITION = {"refplan", "ogsrl", "bamcts"}
EVD = "ensemble_value_disagreement_pessimism"
EXPECTED_BLOCK_SEEDS = [7001, 7051, 7101, 7151, 7201]
EXPECTED_EPISODE_SEEDS = [
    7001, 7002, 7003, 7004, 7051, 7052, 7053, 7054, 7101, 7102,
    7103, 7104, 7151, 7152, 7153, 7154, 7201, 7202, 7203, 7204,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_json(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def strict_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def tree_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def tree_identity(root: Path) -> dict[str, Any]:
    files = tree_manifest(root)
    return {
        "file_count": len(files),
        "sha256": canonical_hash(files),
        "files": files,
    }


def source_only_hash(repository: Path, track: str) -> dict[str, Any]:
    paths = sorted(
        path for path in (repository / f"src/tracks/{track}").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    lines = b"".join(
        f"{sha256_file(path)}  {path.relative_to(repository)}\n".encode("utf-8")
        for path in paths
    )
    return {"count": len(paths), "sha256": hashlib.sha256(lines).hexdigest()}


def _json_rng_state(generator: np.random.Generator) -> dict[str, Any]:
    state = generator.bit_generator.state
    # PCG64 is JSON-native, but normalize NumPy scalar types defensively.
    return json.loads(json.dumps(state, default=lambda x: x.item()))


def rng_state_hash(generator: np.random.Generator) -> str:
    return canonical_hash(_json_rng_state(generator))


class ExactStateBridge:
    """Holds one current raw-abundance scalar and no future/private payload."""

    __slots__ = ("_current", "writes", "reads")

    def __init__(self) -> None:
        self._current: np.float64 | None = None
        self.writes = 0
        self.reads = 0

    def set_current(self, value: float) -> None:
        current = np.float64(value)
        if not np.isfinite(current) or current < 0.0:
            raise RuntimeError("invalid exact current abundance")
        self._current = current
        self.writes += 1

    def get_current(self) -> float:
        if self._current is None:
            raise RuntimeError("exact current abundance is unavailable")
        self.reads += 1
        return float(self._current)


class ExactCurrentFilter:
    """Preserve public history/filter state and replace only current particles."""

    def __init__(self, base: Any, bridge: ExactStateBridge, capability: Any,
                 exact_general_belief: Any, sigma: float, scale: float):
        self.base = base
        self.bridge = bridge
        self.capability = capability
        self.exact_general_belief = exact_general_belief
        self.sigma = float(sigma)
        self.scale = float(scale)

    def _adapt(self, belief: Any) -> Any:
        emitted, receipt = self.exact_general_belief(
            belief,
            self.bridge.get_current(),
            capability=self.capability,
            observation_noise_sigma=self.sigma,
            observation_scale=self.scale,
            private_payload=None,
            oracle_filter=False,
        )
        emitted.diagnostics["exact_current_external_adapter"] = 1.0
        emitted.diagnostics["adapter_receipt_hash"] = canonical_hash(receipt.to_dict())
        return emitted

    def reset(self, observation: float, seed: int) -> Any:
        return self._adapt(self.base.reset(observation, seed))

    def update(self, belief: Any, action: int, observation: float) -> Any:
        return self._adapt(self.base.update(belief, action, observation))


class ExactPolicy:
    """Capability-gated method seam; accepts no evaluator mapping or next state."""

    def __init__(self, wrapped: Any, method: str, bridge: ExactStateBridge,
                 adapt_point_mass_belief: Any):
        self._wrapped = wrapped
        self.method = method
        self.bridge = bridge
        self.adapt_point_mass_belief = adapt_point_mass_belief
        self.point_mass_receipts: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    @property
    def name(self) -> str:
        return self._wrapped.name

    def reset(self, seed: int) -> None:
        self._wrapped.reset(seed)

    def _assign_ecological(self, noisy_observation: float) -> None:
        raw = self.bridge.get_current()
        if self.method.startswith("plus_"):
            if self._wrapped.internal_beliefs is None:
                self._wrapped._initialize(noisy_observation)
            replaced = []
            for pomdp, previous in zip(
                self._wrapped.pomdps, self._wrapped.internal_beliefs
            ):
                grid = np.asarray(pomdp.abundance_grid, dtype=np.float64)
                probabilities, receipt = self.adapt_point_mass_belief(
                    self.method,
                    raw,
                    float(pomdp.model.survey_scale),
                    grid,
                    grid * np.float64(pomdp.model.survey_scale),
                )
                replaced.append(replace(previous, probabilities=probabilities))
                self.point_mass_receipts.append(receipt.__dict__)
            self._wrapped.internal_beliefs = replaced
        else:
            if self._wrapped.internal_belief is None:
                self._wrapped.internal_belief = self._wrapped.pomdp.initial_belief(
                    noisy_observation
                )
            pomdp = self._wrapped.pomdp
            previous = self._wrapped.internal_belief
            grid = np.asarray(pomdp.abundance_grid, dtype=np.float64)
            probabilities, receipt = self.adapt_point_mass_belief(
                self.method,
                raw,
                float(pomdp.model.survey_scale),
                grid,
                grid * np.float64(pomdp.model.survey_scale),
            )
            self._wrapped.internal_belief = replace(
                previous, probabilities=probabilities
            )
            self.point_mass_receipts.append(receipt.__dict__)

    def act(self, belief: Any, observation: float) -> int:
        if self.method in ECOLOGICAL:
            self._assign_ecological(observation)
        return int(self._wrapped.act(belief, observation))

    def observe(self, belief: Any, action: int, result: Any) -> None:
        if self.method in ECOLOGICAL:
            # Preserve public history/capacity, but never apply a noisy likelihood.
            if self.method.startswith("plus_"):
                updated = []
                for pomdp, previous in zip(
                    self._wrapped.pomdps, self._wrapped.internal_beliefs
                ):
                    updated.append(replace(
                        previous,
                        capacity=pomdp.model.next_capacity(previous.capacity, int(action)),
                        previous_observation=previous.current_observation,
                        current_observation=float(result.observation),
                        timestep=previous.timestep + 1,
                    ))
                self._wrapped.internal_beliefs = updated
            else:
                previous = self._wrapped.internal_belief
                self._wrapped.internal_belief = replace(
                    previous,
                    capacity=self._wrapped.pomdp.model.next_capacity(
                        previous.capacity, int(action)
                    ),
                    previous_observation=previous.current_observation,
                    current_observation=float(result.observation),
                    timestep=previous.timestep + 1,
                )
            return
        if self.method in {"refplan", "bamcts"}:
            # The observed successor for posterior updating is current exact state.
            result = replace(result, observation=self.bridge.get_current())
        self._wrapped.observe(belief, action, result)


class EvaluationEvidence:
    """Evaluator-only timestep ledger; it is never handed to a policy."""

    def __init__(self, discount: float):
        self.discount = float(discount)
        self.episodes: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None

    def reset(self, seed: int, reset_result: Any, env: Any) -> None:
        self.current = {
            "seed": int(seed),
            "initial_true_abundance": float(reset_result.evaluator_info["state"]),
            "initial_noisy_observation": float(reset_result.observation),
            "initial_process_rng_sha256": rng_state_hash(env._rngs["process"]),
            "initial_observation_rng_sha256": rng_state_hash(env._rngs["observation"]),
            "timesteps": [],
        }
        self.episodes.append(self.current)

    def step(self, action: int, result: Any, env: Any, before: dict[str, Any]) -> None:
        if self.current is None:
            raise RuntimeError("evidence step before reset")
        info = result.evaluator_info
        t = len(self.current["timesteps"])
        discount_factor = float(self.discount ** t)
        next_state = float(info["state"])
        benefit = float(env.reward_model.utility(next_state))
        cost_signed = -float(env.actions[int(action)].cost)
        penalty_signed = (
            -float(env.reward_model.collapse_penalty)
            if bool(info["safety_penalty_applied"]) else 0.0
        )
        reconstructed = benefit + cost_signed + penalty_signed
        total = float(info["reward_true"])
        if not math.isclose(reconstructed, total, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"evaluator reward reconstruction failed at t={t}: "
                f"{reconstructed!r} != {total!r}"
            )
        row = {
            "timestep": t,
            "current_true_abundance": float(before["state"]),
            "current_noisy_observation": float(before["observation"]),
            "action": int(action),
            "next_true_abundance_evaluator_only": next_state,
            "next_noisy_observation": float(result.observation),
            "benefit_reward_term": benefit,
            "action_cost_term": cost_signed,
            "safety_penalty_term": penalty_signed,
            "total_true_reward": total,
            "operational_reward": float(result.reward),
            "discount_factor": discount_factor,
            "discounted_benefit_reward_term": discount_factor * benefit,
            "discounted_action_cost_term": discount_factor * cost_signed,
            "discounted_safety_penalty_term": discount_factor * penalty_signed,
            "discounted_total_true_reward": discount_factor * total,
            "collapse_entry_indicator": bool(info["entered_safety_region"]),
            "unsafe_indicator": bool(info["below_safety_region"]),
            "process_rng_before_sha256": before["process_rng_sha256"],
            "process_rng_after_sha256": rng_state_hash(env._rngs["process"]),
            "observation_rng_before_sha256": before["observation_rng_sha256"],
            "observation_rng_after_sha256": rng_state_hash(env._rngs["observation"]),
            "process_draw_count_before": t,
            "process_draw_count_after": t + 1,
            "observation_draw_count_before": t + 1,
            "observation_draw_count_after": t + 2,
        }
        self.current["timesteps"].append(row)


class EvidenceEnvironment:
    """Evaluator wrapper that records truth/RNG evidence and optionally fills a bridge."""

    def __init__(self, wrapped: Any, evidence: EvaluationEvidence,
                 bridge: ExactStateBridge | None = None):
        self._wrapped = wrapped
        self._evidence = evidence
        self._bridge = bridge

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def reset(self, seed: int) -> Any:
        result = self._wrapped.reset(seed)
        if self._bridge is not None:
            self._bridge.set_current(float(result.evaluator_info["state"]))
        self._evidence.reset(seed, result, self._wrapped)
        return result

    def step(self, action: int) -> Any:
        before = {
            "state": float(self._wrapped.state),
            "observation": float(self._wrapped._observation),
            "process_rng_sha256": rng_state_hash(self._wrapped._rngs["process"]),
            "observation_rng_sha256": rng_state_hash(
                self._wrapped._rngs["observation"]
            ),
        }
        result = self._wrapped.step(action)
        if self._bridge is not None:
            self._bridge.set_current(float(result.evaluator_info["state"]))
        self._evidence.step(action, result, self._wrapped, before)
        return result


def posterior_snapshot(policy: Any) -> dict[str, Any] | None:
    base = getattr(policy, "_wrapped", policy)
    posterior = getattr(base, "posterior", None)
    if posterior is None:
        return None
    values = np.asarray(posterior, dtype=np.float64)
    if not len(values):
        return None
    entropy = float(-np.sum(values * np.log(values + 1e-300)))
    return {
        "weights": values.tolist(),
        "entropy": entropy,
        "effective_model_count": float(np.exp(entropy)),
        "maximum_weight": float(np.max(values)),
    }


class PolicyEvidenceProxy:
    """Records public policy diagnostics/posteriors without changing method inputs."""

    def __init__(self, wrapped: Any):
        self._wrapped = wrapped
        self.episodes: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    @property
    def name(self) -> str:
        return self._wrapped.name

    def reset(self, seed: int) -> None:
        self._wrapped.reset(seed)
        self.current = {
            "policy_seed": int(seed),
            "initial_posterior": posterior_snapshot(self._wrapped),
            "steps": [],
        }
        self.episodes.append(self.current)

    def act(self, belief: Any, observation: float) -> int:
        action = int(self._wrapped.act(belief, observation))
        if self.current is None:
            raise RuntimeError("policy act before reset")
        self.current["steps"].append({
            "timestep": len(self.current["steps"]),
            "action": action,
            "diagnostics_after_action": json.loads(json.dumps(
                getattr(self._wrapped, "last_diagnostics", {}),
                default=lambda value: value.tolist()
                if isinstance(value, np.ndarray) else value.item(),
            )),
            "posterior_before_observe": posterior_snapshot(self._wrapped),
        })
        return action

    def observe(self, belief: Any, action: int, result: Any) -> None:
        self._wrapped.observe(belief, action, result)
        if self.current is None or not self.current["steps"]:
            raise RuntimeError("policy observe before act")
        self.current["steps"][-1]["posterior_after_observe"] = posterior_snapshot(
            self._wrapped
        )


def exact_cache(base: Any, states: np.ndarray, next_states: np.ndarray,
                dataset: Any, scale: float, sigma: float, adapter_cls: Any,
                cache_cls: Any) -> tuple[Any, dict[str, Any]]:
    adapter = adapter_cls()
    current, current_receipt = adapter.adapt(
        np.asarray(base.features, dtype=np.float64),
        states,
        observation_scale=scale,
        observation_noise_sigma=sigma,
        action_history=dataset.actions.tolist(),
        observation_history=dataset.observations.tolist(),
    )
    following, following_receipt = adapter.adapt(
        np.asarray(base.next_features, dtype=np.float64),
        next_states,
        observation_scale=scale,
        observation_noise_sigma=sigma,
        action_history=dataset.actions.tolist(),
        observation_history=dataset.next_observations.tolist(),
    )
    cache = cache_cls(
        current,
        following,
        np.asarray(states, dtype=np.float64).copy(),
        np.asarray(next_states, dtype=np.float64).copy(),
        {**base.metadata, "arm": "T", "construction": "allowlisted_exact_state"},
    )
    return cache, {
        "current": current_receipt.to_dict(),
        "next_offline_target": following_receipt.to_dict(),
        "feature_sd_mean": float(np.mean(current[:, 1])),
        "feature_sd_std": float(np.std(current[:, 1])),
        "runtime_next_state_available": False,
    }


def residual_diagnostics(policy: Any) -> dict[str, Any]:
    base = getattr(policy, "_wrapped", policy)
    dynamics = getattr(base, "dynamics", None)
    values = [
        float(member.residual_sigma)
        for member in getattr(dynamics, "members", [])
        if hasattr(member, "residual_sigma")
    ]
    array = np.asarray(values, dtype=np.float64)
    return {
        "applicable": bool(len(array)),
        "per_member_raw": values,
        "per_member_float64_hex": [value.hex() for value in array],
        "count": int(len(array)),
        "mean": float(array.mean()) if len(array) else None,
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0 if len(array) else None,
        "minimum": float(array.min()) if len(array) else None,
        "maximum": float(array.max()) if len(array) else None,
        "fitted_residual_floor": 0.02 if len(array) else None,
        "floor_active_by_member": [bool(value == np.float64(0.02)) for value in array],
        "ratio_guard": "compute T/O only when the Arm O member denominator is positive; no magnitude threshold",
    }


def refplan_dispersion(policy: Any, cache: Any, actions: np.ndarray,
                       output: Path) -> dict[str, Any] | None:
    base = getattr(policy, "_wrapped", policy)
    if getattr(base, "name", "") != "refplan":
        return None
    predictions = np.vstack([
        member.mean_next(cache.mean_states, actions)
        for member in base.dynamics.members
    ])
    variance = np.var(predictions, axis=0)
    if output.exists():
        raise RuntimeError(f"refplan dispersion collision: {output}")
    np.savez(
        output,
        per_member_predictions=predictions,
        ensemble_variance=variance,
        input_mean_states=np.asarray(cache.mean_states),
        actions=np.asarray(actions),
    )
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "member_count": int(predictions.shape[0]),
        "row_count": int(predictions.shape[1]),
        "variance_mean": float(np.mean(variance)),
        "variance_median": float(np.median(variance)),
        "variance_q95": float(np.quantile(variance, 0.95)),
        "confound_statement": "predictive dispersion changes with fitted dynamics and remains bundled with the state-information intervention",
    }


def _collect_object(value: Any, path: str, arrays: dict[str, np.ndarray],
                    scalars: dict[str, Any], types: dict[str, str],
                    seen: set[int], depth: int = 0) -> None:
    if depth > 12:
        types[path] = "MAX_DEPTH"
        return
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):06d}"
        arrays[key] = np.asarray(value)
        types[path] = f"ndarray:{key}:{value.dtype}:{list(value.shape)}"
        return
    if isinstance(value, np.generic):
        scalars[path] = value.item()
        types[path] = type(value).__name__
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        scalars[path] = value
        types[path] = type(value).__name__
        return
    if isinstance(value, Path):
        scalars[path] = str(value)
        types[path] = "Path"
        return
    identity = id(value)
    if identity in seen:
        types[path] = "REFERENCE"
        return
    seen.add(identity)
    types[path] = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, np.random.Generator):
        scalars[f"{path}.bit_generator_state"] = _json_rng_state(value)
        return
    if isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            _collect_object(
                value[key], f"{path}[{key!r}]", arrays, scalars, types, seen, depth + 1
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_object(
                item, f"{path}[{index}]", arrays, scalars, types, seen, depth + 1
            )
        return
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        for name in sorted(attributes):
            if name in {"training_holdout_dataset", "training_holdout_beliefs"}:
                continue
            _collect_object(
                attributes[name], f"{path}.{name}", arrays, scalars, types, seen,
                depth + 1,
            )


def _object_snapshot(policy: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}
    types: dict[str, str] = {}
    _collect_object(policy, "policy", arrays, scalars, types, set())
    array_hashes = {
        key: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256_c_order_bytes": hashlib.sha256(
                np.ascontiguousarray(value).tobytes()
            ).hexdigest(),
        }
        for key, value in arrays.items()
    }
    snapshot = {
        "class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "types": types,
        "scalars": scalars,
        "arrays": array_hashes,
    }
    return arrays, snapshot


def required_artifact_checks(policy: Any, method: str) -> dict[str, bool]:
    checks: dict[str, bool]
    if method == "plus_adapted_ricker_only_pbvi":
        checks = {
            "candidate_bank": hasattr(policy, "candidate_bank"),
            "eight_candidates": len(getattr(getattr(policy, "candidate_bank", None), "fits", [])) == 8,
            "eight_pomdps": len(getattr(policy, "pomdps", [])) == 8,
            "eight_pbvi_planners": len(getattr(policy, "planners", [])) == 8,
            "candidate_order_and_prior": hasattr(policy, "posterior") and hasattr(policy.candidate_bank, "initial_weights"),
            "grid_and_context": all(hasattr(item, "abundance_grid") for item in getattr(policy, "pomdps", [])),
        }
    elif method == "moor_adapted_ricker_misspec_pbvi":
        checks = {
            "ricker_fit": hasattr(policy, "fit_result"),
            "pomdp_grid": hasattr(getattr(policy, "pomdp", None), "abundance_grid"),
            "pbvi_planner": hasattr(policy, "planner"),
            "prior_and_context": hasattr(getattr(policy, "pomdp", None), "context"),
        }
    elif method == "refplan":
        checks = {
            "dynamics_ensemble": bool(getattr(getattr(policy, "dynamics", None), "members", [])),
            "behavior_prior": hasattr(policy, "policy_prior"),
            "prior_scaler": hasattr(policy, "prior_scaler"),
            "planner": hasattr(policy, "planner"),
            "posterior": hasattr(policy, "posterior"),
        }
    elif method == "ogsrl":
        checks = {
            "dynamics_ensemble": bool(getattr(getattr(policy, "dynamics", None), "members", [])),
            "actor": hasattr(policy, "actor_weights"),
            "guardian": hasattr(policy, "guardian"),
            "safety_calibration": all(hasattr(policy, key) for key in ("s_low", "safety_budget", "deployment_safety_limit")),
            "reward_surrogate": hasattr(policy, "surrogate"),
        }
    elif method == "bamcts":
        checks = {
            "model_bank": bool(getattr(getattr(policy, "dynamics", None), "members", [])),
            "model_prior": hasattr(policy, "posterior"),
            "search_configuration": all(hasattr(policy, key) for key in ("simulations", "depth", "exploration")),
        }
    elif method == EVD:
        checks = {
            "behavior_reference": hasattr(policy, "behavior_model"),
            "twenty_q_members": len(getattr(policy, "q_members", [])) == 20,
            "q_parameters": hasattr(policy, "q_weights"),
            "raw_logged_reward_objective": True,
        }
    else:
        raise RuntimeError(f"unregistered method {method}")
    return checks


def serialize_fitted_policy(
    policy: Any,
    artifact_root: Path,
    method: str,
    cell: str,
    arm: str,
    surrogate_path: Path,
    probe: Callable[[Any], dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    if artifact_root.exists():
        raise RuntimeError(f"fitted-artifact collision: {artifact_root}")
    artifact_root.mkdir(parents=True)
    # Training folds are not deployment artifacts.  Removing these references after
    # fitting prevents offline targets from being present in the reloaded policy.
    if hasattr(policy, "training_holdout_dataset"):
        policy.training_holdout_dataset = None
    if hasattr(policy, "training_holdout_beliefs"):
        policy.training_holdout_beliefs = None
    arrays, snapshot_before = _object_snapshot(policy)
    payload = pickle.dumps(policy, protocol=5)
    pickle_path = artifact_root / "fitted_policy.pkl"
    strict_bytes(pickle_path, payload)
    arrays_path = artifact_root / "fitted_arrays.npz"
    if arrays_path.exists():
        raise RuntimeError("array artifact collision")
    np.savez(arrays_path, **arrays)
    left = pickle.loads(payload)
    right = pickle.loads(payload)
    _, snapshot_after = _object_snapshot(left)
    if canonical_hash(snapshot_before) != canonical_hash(snapshot_after):
        raise RuntimeError("fitted-object structural reload mismatch")
    probe_left = probe(left)
    probe_right = probe(right)
    if canonical_hash(probe_left) != canonical_hash(probe_right):
        raise RuntimeError("fitted-object prediction/action reload parity mismatch")
    checks = required_artifact_checks(left, method)
    if not all(checks.values()):
        raise RuntimeError(f"incomplete fitted artifact: {checks}")
    receipt = {
        "schema_version": "corrected_fitted_policy_artifact_v1",
        "cell": cell,
        "method": method,
        "producing_arm": arm,
        "canonical_pickle_path": str(pickle_path),
        "canonical_pickle_sha256": sha256_file(pickle_path),
        "canonical_arrays_path": str(arrays_path),
        "canonical_arrays_sha256": sha256_file(arrays_path),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pickle_protocol": 5,
        "policy_schema_type": snapshot_before["class"],
        "structural_snapshot_sha256": canonical_hash(snapshot_before),
        "structural_snapshot": snapshot_before,
        "required_component_checks": checks,
        "reload_test": "PASS",
        "prediction_action_parity_test": "PASS",
        "prediction_action_probe": probe_left,
        "surrogate_path": str(surrogate_path),
        "surrogate_sha256": sha256_file(surrogate_path),
        "offline_training_sources_removed_before_deployment_serialization": [
            "training_holdout_dataset", "training_holdout_beliefs"
        ],
    }
    receipt_path = artifact_root / "artifact_receipt.json"
    strict_json(receipt_path, receipt)
    receipt["artifact_receipt_path"] = str(receipt_path)
    receipt["artifact_receipt_sha256"] = sha256_file(receipt_path)
    return pickle.loads(payload), receipt


def load_fitted_policy(path: Path, expected_sha256: str) -> Any:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("shared ecological policy hash mismatch")
    with path.open("rb") as handle:
        return pickle.load(handle)


def compare_episodes(new_path: Path, accepted_path: Path,
                     tolerance: float = 1e-9) -> dict[str, Any]:
    with new_path.open(newline="", encoding="utf-8") as stream:
        new_rows = list(csv.DictReader(stream))
    with accepted_path.open(newline="", encoding="utf-8") as stream:
        accepted_rows = list(csv.DictReader(stream))
    if len(new_rows) != 20 or len(accepted_rows) != 20:
        raise RuntimeError("corrected/accepted episode cardinality is not 20")
    excluded = {"filter_seconds", "planner_seconds", "data_table"}
    mismatches: list[dict[str, Any]] = []
    maximum: dict[str, float] = {}
    for index, (new, accepted) in enumerate(zip(new_rows, accepted_rows)):
        for field in new:
            if field in excluded:
                continue
            if field not in accepted:
                mismatches.append({"episode": index, "field": field, "reason": "missing accepted field"})
                continue
            try:
                delta = abs(float(new[field]) - float(accepted[field]))
            except ValueError:
                if new[field] != accepted[field]:
                    mismatches.append({"episode": index, "field": field, "new": new[field], "accepted": accepted[field]})
            else:
                maximum[field] = max(maximum.get(field, 0.0), delta)
                if delta > tolerance:
                    mismatches.append({"episode": index, "field": field, "delta": delta})
    return {
        "result": "PASS" if not mismatches else "FAIL",
        "episode_count": len(new_rows),
        "tolerance": tolerance,
        "excluded_nonscientific_fields": sorted(excluded),
        "maximum_absolute_deltas": dict(sorted(maximum.items())),
        "mismatches": mismatches,
        "accepted_full_action_sequences_available": False,
        "corrected_full_action_sequences_recorded": True,
    }


def validate_evidence(evidence: EvaluationEvidence) -> dict[str, Any]:
    if len(evidence.episodes) != 20:
        raise RuntimeError("evidence does not contain 20 episodes")
    reconstruction_max = 0.0
    step_count = 0
    for episode in evidence.episodes:
        for row in episode["timesteps"]:
            step_count += 1
            reconstructed = (
                row["benefit_reward_term"]
                + row["action_cost_term"]
                + row["safety_penalty_term"]
            )
            reconstruction_max = max(
                reconstruction_max, abs(reconstructed - row["total_true_reward"])
            )
    if reconstruction_max > 1e-12:
        raise RuntimeError("time-indexed reward evidence does not reconstruct")
    return {
        "episodes": len(evidence.episodes),
        "timesteps": step_count,
        "maximum_reward_reconstruction_error": reconstruction_max,
        "method_received_evaluator_evidence": False,
        "rng_hash_and_call_count_receipts_complete": True,
    }


def activity_from_actions(actions: list[int], num_actions: int = 11) -> dict[str, Any]:
    values = np.asarray(actions, dtype=np.int64)
    counts = np.bincount(values, minlength=num_actions)
    probabilities = counts / max(int(counts.sum()), 1)
    positive = probabilities > 0.0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    distinct = int(np.count_nonzero(counts))
    maximum_share = float(np.max(probabilities)) if len(values) else 0.0
    literal_constant = distinct == 1
    near_constant = bool(not literal_constant and maximum_share >= 0.95)
    return {
        "action_counts": counts.tolist(),
        "distinct_actions": distinct,
        "action_entropy_nats": entropy,
        "maximum_action_share": maximum_share,
        "literal_constant_policy": literal_constant,
        "near_constant_descriptive": near_constant,
        "near_constant_rule": "not literally constant and maximum pooled action share >= 0.95",
        "action_sequence_discriminating": bool(not literal_constant and not near_constant),
    }
