#!/usr/bin/env python3
"""Shared external contracts for the corrected Stage B recovery replication."""

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
ECO = {"plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"}
TRANSITION_METHODS = {"refplan", "ogsrl", "bamcts"}
EVD = "ensemble_value_disagreement_pessimism"
BLOCK_SEEDS = [7001, 7051, 7101, 7151, 7201]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


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


def tree_identity(root: Path) -> dict[str, Any]:
    files = {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    }
    return {"file_count": len(files), "sha256": canonical_sha(files), "files": files}


def source_only_hash(root: Path, track: str) -> dict[str, Any]:
    paths = sorted(
        path for path in (root / "src/tracks" / track).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts and path.suffix not in {".pyc", ".pyo"}
    )
    content = b"".join(
        f"{sha(path)}  {path.relative_to(root)}\n".encode() for path in paths
    )
    return {"count": len(paths), "sha256": hashlib.sha256(content).hexdigest()}


def verify_hash_manifest(root: Path, manifest: Path) -> int:
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"pre-execution hash mismatch: {relative}")
        count += 1
    return count


class CurrentStateBridge:
    """One current raw-abundance scalar; no mapping, future value, or metadata."""

    __slots__ = ("_value", "reads", "writes")

    def __init__(self) -> None:
        self._value: float | None = None
        self.reads = 0
        self.writes = 0

    def set(self, value: float) -> None:
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise RuntimeError("invalid current abundance")
        self._value = value
        self.writes += 1

    def get(self) -> float:
        if self._value is None:
            raise RuntimeError("current abundance unavailable")
        self.reads += 1
        return self._value


class ExactFilter:
    def __init__(self, base: Any, bridge: CurrentStateBridge, capability: Any,
                 adapter: Any, sigma: float, scale: float):
        self.base, self.bridge, self.capability = base, bridge, capability
        self.adapter, self.sigma, self.scale = adapter, float(sigma), float(scale)

    def _exact(self, belief: Any) -> Any:
        output, _receipt = self.adapter(
            belief, self.bridge.get(), capability=self.capability,
            observation_noise_sigma=self.sigma, observation_scale=self.scale,
            private_payload=None, oracle_filter=False,
        )
        output.diagnostics["corrected_exact_current_adapter"] = 1.0
        return output

    def reset(self, observation: float, seed: int) -> Any:
        return self._exact(self.base.reset(observation, seed))

    def update(self, belief: Any, action: int, observation: float) -> Any:
        return self._exact(self.base.update(belief, action, observation))


class ExactPolicy:
    """Context-preserving exact-state policy seam outside frozen method code."""

    def __init__(self, policy: Any, method: str, bridge: CurrentStateBridge,
                 point_mass_adapter: Callable[..., Any]):
        self._policy, self.method, self.bridge = policy, method, bridge
        self.point_mass_adapter = point_mass_adapter
        self.point_mass_receipts: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    @property
    def name(self) -> str:
        return self._policy.name

    def reset(self, seed: int) -> None:
        self._policy.reset(seed)

    def _point_mass(self, noisy_observation: float) -> None:
        raw = self.bridge.get()
        if self.method.startswith("plus_"):
            if self._policy.internal_beliefs is None:
                self._policy._initialize(noisy_observation)
            updated = []
            for pomdp, old in zip(self._policy.pomdps, self._policy.internal_beliefs):
                grid = np.asarray(pomdp.abundance_grid, dtype=np.float64)
                probs, receipt = self.point_mass_adapter(
                    self.method, raw, float(pomdp.model.survey_scale), grid,
                    grid * np.float64(pomdp.model.survey_scale),
                )
                updated.append(replace(old, probabilities=probs))
                self.point_mass_receipts.append(receipt.__dict__)
            self._policy.internal_beliefs = updated
        else:
            if self._policy.internal_belief is None:
                self._policy.internal_belief = self._policy.pomdp.initial_belief(noisy_observation)
            pomdp, old = self._policy.pomdp, self._policy.internal_belief
            grid = np.asarray(pomdp.abundance_grid, dtype=np.float64)
            probs, receipt = self.point_mass_adapter(
                self.method, raw, float(pomdp.model.survey_scale), grid,
                grid * np.float64(pomdp.model.survey_scale),
            )
            self._policy.internal_belief = replace(old, probabilities=probs)
            self.point_mass_receipts.append(receipt.__dict__)

    def act(self, belief: Any, observation: float) -> int:
        if self.method in ECO:
            self._point_mass(observation)
        return int(self._policy.act(belief, observation))

    def observe(self, belief: Any, action: int, result: Any) -> None:
        if self.method in ECO:
            # Public history and capacity advance, noisy likelihood does not.
            if self.method.startswith("plus_"):
                values = []
                for pomdp, old in zip(self._policy.pomdps, self._policy.internal_beliefs):
                    values.append(replace(
                        old, capacity=pomdp.model.next_capacity(old.capacity, action),
                        previous_observation=old.current_observation,
                        current_observation=float(result.observation), timestep=old.timestep + 1,
                    ))
                self._policy.internal_beliefs = values
            else:
                old = self._policy.internal_belief
                self._policy.internal_belief = replace(
                    old, capacity=self._policy.pomdp.model.next_capacity(old.capacity, action),
                    previous_observation=old.current_observation,
                    current_observation=float(result.observation), timestep=old.timestep + 1,
                )
            return
        if self.method in {"refplan", "bamcts"}:
            result = replace(result, observation=self.bridge.get())
        self._policy.observe(belief, action, result)


def rng_hash(generator: np.random.Generator) -> str:
    state = json.loads(json.dumps(generator.bit_generator.state, default=lambda x: x.item()))
    return canonical_sha(state)


class EvidenceLedger:
    def __init__(self, discount: float):
        self.discount = float(discount)
        self.episodes: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None

    def reset(self, seed: int, result: Any, env: Any) -> None:
        self.current = {
            "seed": int(seed), "initial_true_abundance": float(result.evaluator_info["state"]),
            "initial_noisy_observation": float(result.observation), "timesteps": [],
            "initial_process_rng_sha256": rng_hash(env._rngs["process"]),
            "initial_observation_rng_sha256": rng_hash(env._rngs["observation"]),
        }
        self.episodes.append(self.current)

    def step(self, action: int, result: Any, env: Any, before: dict[str, Any]) -> None:
        if self.current is None:
            raise RuntimeError("evidence step before reset")
        t = len(self.current["timesteps"])
        info = result.evaluator_info
        next_state = float(info["state"])
        benefit = float(env.reward_model.utility(next_state))
        cost = -float(env.actions[action].cost)
        penalty = -float(env.reward_model.collapse_penalty) if info["safety_penalty_applied"] else 0.0
        total = float(info["reward_true"])
        if not math.isclose(benefit + cost + penalty, total, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("reward-component reconstruction failure")
        gamma = float(self.discount ** t)
        self.current["timesteps"].append({
            "timestep": t, "current_true_abundance": before["state"],
            "current_noisy_observation": before["observation"], "action": int(action),
            "next_true_abundance_evaluator_only": next_state,
            "next_noisy_observation": float(result.observation),
            "benefit_reward_term": benefit, "action_cost_term": cost,
            "safety_penalty_term": penalty, "total_true_reward": total,
            "discount_factor": gamma, "discounted_benefit_reward_term": gamma * benefit,
            "discounted_action_cost_term": gamma * cost,
            "discounted_safety_penalty_term": gamma * penalty,
            "discounted_total_true_reward": gamma * total,
            "collapse_entry_indicator": bool(info["entered_safety_region"]),
            "unsafe_indicator": bool(info["below_safety_region"]),
            "process_rng_before_sha256": before["process"],
            "process_rng_after_sha256": rng_hash(env._rngs["process"]),
            "observation_rng_before_sha256": before["observation_rng"],
            "observation_rng_after_sha256": rng_hash(env._rngs["observation"]),
            "process_draw_count_before": t, "process_draw_count_after": t + 1,
            "observation_draw_count_before": t + 1,
            "observation_draw_count_after": t + 2,
        })


class EvidenceEnv:
    def __init__(self, env: Any, ledger: EvidenceLedger,
                 bridge: CurrentStateBridge | None = None):
        self._env, self._ledger, self._bridge = env, ledger, bridge

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, seed: int) -> Any:
        result = self._env.reset(seed)
        if self._bridge is not None:
            self._bridge.set(result.evaluator_info["state"])
        self._ledger.reset(seed, result, self._env)
        return result

    def step(self, action: int) -> Any:
        before = {"state": float(self._env.state), "observation": float(self._env._observation),
                  "process": rng_hash(self._env._rngs["process"]),
                  "observation_rng": rng_hash(self._env._rngs["observation"])}
        result = self._env.step(action)
        if self._bridge is not None:
            self._bridge.set(result.evaluator_info["state"])
        self._ledger.step(action, result, self._env, before)
        return result


def posterior(policy: Any) -> dict[str, Any] | None:
    base = getattr(policy, "_policy", policy)
    values = getattr(base, "posterior", None)
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float64)
    entropy = float(-np.sum(values * np.log(values + 1e-300)))
    return {"weights": values.tolist(), "entropy": entropy,
            "effective_model_count": float(np.exp(entropy)),
            "maximum_weight": float(values.max())}


class PolicyLedger:
    def __init__(self, policy: Any):
        self._policy = policy
        self.episodes: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    @property
    def name(self) -> str:
        return self._policy.name

    def reset(self, seed: int) -> None:
        self._policy.reset(seed)
        self.current = {"policy_seed": int(seed), "initial_posterior": posterior(self._policy), "steps": []}
        self.episodes.append(self.current)

    def act(self, belief: Any, observation: float) -> int:
        action = int(self._policy.act(belief, observation))
        if self.current is None:
            raise RuntimeError("policy act before reset")
        diagnostics = json.loads(json.dumps(
            getattr(self._policy, "last_diagnostics", {}),
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(),
        ))
        self.current["steps"].append({
            "timestep": len(self.current["steps"]), "action": action,
            "diagnostics_after_action": diagnostics,
            "posterior_before_observe": posterior(self._policy),
        })
        return action

    def observe(self, belief: Any, action: int, result: Any) -> None:
        self._policy.observe(belief, action, result)
        self.current["steps"][-1]["posterior_after_observe"] = posterior(self._policy)


def make_exact_cache(base: Any, states: np.ndarray, next_states: np.ndarray,
                     dataset: Any, scale: float, sigma: float, adapter_cls: Any,
                     cache_cls: Any) -> tuple[Any, dict[str, Any]]:
    adapter = adapter_cls()
    current, rc = adapter.adapt(
        base.features, states, observation_scale=scale, observation_noise_sigma=sigma,
        action_history=dataset.actions.tolist(), observation_history=dataset.observations.tolist(),
    )
    following, rn = adapter.adapt(
        base.next_features, next_states, observation_scale=scale, observation_noise_sigma=sigma,
        action_history=dataset.actions.tolist(), observation_history=dataset.next_observations.tolist(),
    )
    cache = cache_cls(current, following, states.copy(), next_states.copy(),
                      {**base.metadata, "arm": "T", "allowlisted_exact_state": True})
    return cache, {"current": rc.to_dict(), "offline_next_target": rn.to_dict(),
                   "feature_sd_mean": float(current[:, 1].mean()),
                   "feature_sd_std": float(current[:, 1].std()),
                   "runtime_next_state_available": False}


def residual_receipt(policy: Any) -> dict[str, Any]:
    base = getattr(policy, "_policy", policy)
    values = np.asarray([
        float(member.residual_sigma)
        for member in getattr(getattr(base, "dynamics", None), "members", [])
        if hasattr(member, "residual_sigma")
    ], dtype=np.float64)
    return {"applicable": bool(len(values)), "per_member_raw": values.tolist(),
            "per_member_float64_hex": [value.hex() for value in values],
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else None),
            "minimum": float(values.min()) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None,
            "fitted_floor": 0.02 if len(values) else None,
            "floor_active": [bool(value == 0.02) for value in values],
            "ratio_guard": "Arm O denominator > 0; no numerical magnitude threshold"}


def _walk(value: Any, prefix: str, arrays: dict[str, np.ndarray],
          scalars: dict[str, Any], seen: set[int], depth: int = 0) -> None:
    if depth > 12:
        scalars[prefix] = "MAX_DEPTH"
        return
    if isinstance(value, np.ndarray):
        arrays[prefix] = value
        return
    if isinstance(value, np.generic):
        scalars[prefix] = value.item()
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        scalars[prefix] = value
        return
    if isinstance(value, Path):
        scalars[prefix] = str(value)
        return
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, np.random.Generator):
        scalars[prefix + ".rng_state_sha256"] = rng_hash(value)
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _walk(value[key], f"{prefix}[{key!r}]", arrays, scalars, seen, depth + 1)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, f"{prefix}[{index}]", arrays, scalars, seen, depth + 1)
    elif hasattr(value, "__dict__"):
        for key in sorted(value.__dict__):
            if key not in {"training_holdout_dataset", "training_holdout_beliefs"}:
                _walk(value.__dict__[key], f"{prefix}.{key}", arrays, scalars, seen, depth + 1)


def fitted_checks(policy: Any, method: str) -> dict[str, bool]:
    if method.startswith("plus_"):
        return {"candidate_bank": hasattr(policy, "candidate_bank"),
                "eight_candidates": len(policy.candidate_bank.fits) == 8,
                "eight_pomdps": len(policy.pomdps) == 8,
                "eight_planners": len(policy.planners) == 8,
                "prior": hasattr(policy, "posterior")}
    if method.startswith("moor_"):
        return {"fit": hasattr(policy, "fit_result"), "pomdp": hasattr(policy, "pomdp"),
                "grid": hasattr(policy.pomdp, "abundance_grid"), "planner": hasattr(policy, "planner")}
    if method == "refplan":
        return {"dynamics": hasattr(policy, "dynamics"), "behavior_prior": hasattr(policy, "policy_prior"),
                "prior_scaler": hasattr(policy, "prior_scaler"), "planner": hasattr(policy, "planner")}
    if method == "ogsrl":
        return {key: hasattr(policy, key) for key in
                ("dynamics", "guardian", "actor_weights", "s_low", "safety_budget", "surrogate")}
    if method == "bamcts":
        return {key: hasattr(policy, key) for key in
                ("dynamics", "posterior", "simulations", "depth", "exploration")}
    if method == EVD:
        return {"behavior": hasattr(policy, "behavior_model"), "q_members": len(policy.q_members) == 20,
                "q_weights": hasattr(policy, "q_weights"), "raw_rewards": True}
    raise RuntimeError(f"unknown method {method}")


def save_fitted(policy: Any, target: Path, method: str, cell: str, arm: str,
                surrogate: Path, probe: Callable[[Any], Any]) -> tuple[Any, dict[str, Any]]:
    if target.exists():
        raise RuntimeError(f"fitted target collision: {target}")
    target.mkdir(parents=True)
    policy.training_holdout_dataset = None
    policy.training_holdout_beliefs = None
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}
    _walk(policy, "policy", arrays, scalars, set())
    array_ids = {key: {"dtype": str(value.dtype), "shape": list(value.shape),
                       "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                 for key, value in arrays.items()}
    structural = {"class": f"{type(policy).__module__}.{type(policy).__qualname__}",
                  "arrays": array_ids, "scalars": scalars}
    payload = pickle.dumps(policy, protocol=5)
    policy_path = target / "fitted_policy.pkl"
    strict_bytes(policy_path, payload)
    array_path = target / "fitted_parameters.npz"
    np.savez(array_path, **{f"v{index:06d}": value for index, value in enumerate(arrays.values())})
    left, right = pickle.loads(payload), pickle.loads(payload)
    if canonical_sha(probe(left)) != canonical_sha(probe(right)):
        raise RuntimeError("prediction/action parity failed after reload")
    checks = fitted_checks(left, method)
    if not all(checks.values()):
        raise RuntimeError(f"incomplete fitted artifact: {checks}")
    receipt = {"schema_version": "corrected_fitted_artifact_v1", "cell": cell,
               "method": method, "producing_arm": arm, "pickle_path": str(policy_path),
               "pickle_sha256": sha(policy_path), "parameter_path": str(array_path),
               "parameter_sha256": sha(array_path), "structural_sha256": canonical_sha(structural),
               "structural_inventory": structural, "required_checks": checks,
               "reload_test": "PASS", "prediction_action_parity": "PASS",
               "probe": probe(left), "surrogate_path": str(surrogate),
               "surrogate_sha256": sha(surrogate), "python": platform.python_version(),
               "numpy": np.__version__, "pickle_protocol": 5,
               "offline_fold_references_removed_before_serialization": True}
    receipt_path = target / "artifact_receipt.json"
    strict_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    receipt["receipt_sha256"] = sha(receipt_path)
    return pickle.loads(payload), receipt


def load_fitted(path: Path, expected: str) -> Any:
    if sha(path) != expected:
        raise RuntimeError("fitted policy byte identity mismatch")
    with path.open("rb") as handle:
        return pickle.load(handle)


def compare_accepted(current: Path, accepted: Path, tolerance: float = 1e-9) -> dict[str, Any]:
    with current.open(newline="", encoding="utf-8") as handle:
        new = list(csv.DictReader(handle))
    with accepted.open(newline="", encoding="utf-8") as handle:
        old = list(csv.DictReader(handle))
    if len(new) != 20 or len(old) != 20:
        raise RuntimeError("episode cardinality mismatch")
    excluded = {"filter_seconds", "planner_seconds", "data_table"}
    differences = []
    maximum: dict[str, float] = {}
    for episode, (a, b) in enumerate(zip(new, old)):
        for key in a:
            if key in excluded:
                continue
            if key not in b:
                differences.append({"episode": episode, "field": key, "reason": "missing"})
                continue
            try:
                delta = abs(float(a[key]) - float(b[key]))
            except ValueError:
                if a[key] != b[key]:
                    differences.append({"episode": episode, "field": key})
            else:
                maximum[key] = max(maximum.get(key, 0.0), delta)
                if delta > tolerance:
                    differences.append({"episode": episode, "field": key, "delta": delta})
    return {"result": "PASS" if not differences else "FAIL", "tolerance": tolerance,
            "episode_count": 20, "excluded": sorted(excluded),
            "maximum_absolute_deltas": maximum, "differences": differences,
            "historical_action_sequences_available": False,
            "corrected_action_sequences_recorded": True}


def activity(actions: list[int]) -> dict[str, Any]:
    counts = np.bincount(np.asarray(actions, dtype=int), minlength=11)
    probs = counts / max(counts.sum(), 1)
    entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    distinct, max_share = int(np.count_nonzero(counts)), float(probs.max())
    constant = distinct == 1
    near = bool(not constant and max_share >= 0.95)
    return {"action_counts": counts.tolist(), "distinct_actions": distinct,
            "entropy_nats": entropy, "maximum_action_share": max_share,
            "literal_constant": constant, "near_constant": near,
            "near_constant_rule": "not literal constant and maximum pooled action share >= 0.95",
            "action_sequence_discriminating": bool(not constant and not near)}
