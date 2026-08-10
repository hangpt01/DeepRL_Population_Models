"""External, lossless serializers for fitted objects in the frozen track packages.

The frozen ``BasePolicy.save_fit_artifacts`` seam is intentionally a no-op for several
registered methods.  This module therefore serializes the complete fitted instance graph
without editing frozen source.  It accepts only explicitly registered frozen root classes,
records every instance attribute, reloads into fresh objects, and executes a real frozen
prediction/action method for parity.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import (
    ContractError,
    canonical_json_bytes,
    require_exact_keys,
    require_nonempty_string,
    require_sha256,
    sha256_bytes,
    strict_json_loads,
)
from .registration import EXPECTED_SOURCE_HASHES


SCHEMA_VERSION = "corrected_stageb_frozen_object_graph_v2"
BASE_POLICY_ATTRIBUTES = frozenset(
    {
        "hidden",
        "method_context",
        "model_cfg",
        "planner_cfg",
        "num_actions",
        "seed",
        "rng",
        "last_diagnostics",
        "fit_diagnostics",
        "training_history",
        "training_holdout_dataset",
        "training_holdout_beliefs",
    }
)
ROOT_SPECS: dict[str, tuple[str, str, frozenset[str]]] = {
    "general_dynamics_ensemble": (
        "general",
        "real_ecology_benchmark.dynamics.ContinuousDynamicsEnsemble",
        frozenset({"members", "num_actions", "K_ref", "rng"}),
    ),
    "refplan_fitted_policy": (
        "general",
        "real_ecology_benchmark.methods.refplan.RefPlanPolicy",
        BASE_POLICY_ATTRIBUTES
        | frozenset(
            {
                "dynamics",
                "posterior",
                "planner",
                "policy_prior",
                "prior_weights",
                "prior_scaler",
                "max_planning_members",
            }
        ),
    ),
    "ogsrl_fitted_policy": (
        "general",
        "real_ecology_benchmark.methods.ogsrl.OGSRLPolicy",
        BASE_POLICY_ATTRIBUTES
        | frozenset(
            {
                "dynamics",
                "guardian",
                "surrogate",
                "s_low",
                "actor_weights",
                "lambda_safety",
                "lambda_ood",
                "safety_budget",
                "ood_budget",
                "train_iterations",
                "deployment_safety_limit",
                "deployment_ood_limit",
            }
        ),
    ),
    "bamcts_fitted_policy": (
        "general",
        "real_ecology_benchmark.methods.bamcts.BAMCTSPolicy",
        BASE_POLICY_ATTRIBUTES
        | frozenset({"dynamics", "posterior", "simulations", "depth", "exploration"}),
    ),
    "evd_fitted_policy": (
        "general",
        "real_ecology_benchmark.methods.ensemble_value_disagreement.EnsembleValueDisagreementPolicy",
        BASE_POLICY_ATTRIBUTES
        | frozenset(
            {
                "behavior_model",
                "q_members",
                "feature_dim",
                "q_weights",
                "ensemble_size",
                "cql_alpha",
                "disagreement_penalty",
                "fit_iterations",
            }
        ),
    ),
    "plus_fitted_policy": (
        "ecological",
        "real_ecology_benchmark.methods.plus_faithful.PLUSRickerOnlyFaithfulPBVIPolicy",
        BASE_POLICY_ATTRIBUTES
        | frozenset(
            {
                "candidate_bank",
                "fit_cache_statuses",
                "pomdps",
                "planners",
                "posterior",
                "faithful_config",
                "internal_beliefs",
            }
        ),
    ),
    "moor_fitted_policy": (
        "ecological",
        "real_ecology_benchmark.methods.moor_faithful.MOORFaithfulRickerPBVIPolicy",
        BASE_POLICY_ATTRIBUTES
        | frozenset(
            {
                "fit_result",
                "fit_cache_status",
                "pomdp",
                "planner",
                "faithful_config",
                "internal_belief",
            }
        ),
    ),
}
ALLOWED_OPERATIONS = {
    "general_dynamics_ensemble": frozenset({"predict"}),
    "refplan_fitted_policy": frozenset({"act"}),
    "ogsrl_fitted_policy": frozenset({"act"}),
    "bamcts_fitted_policy": frozenset({"act"}),
    "evd_fitted_policy": frozenset({"act"}),
    "plus_fitted_policy": frozenset({"act"}),
    "moor_fitted_policy": frozenset({"act"}),
}
TRACK_ROOTS = {
    "general": Path("src/tracks/general"),
    "ecological": Path("src/tracks/ecological"),
}
POST_CALL_VOLATILE_FIELDS = {
    "real_ecology_benchmark.faithful_pomdp.CandidatePOMDP": frozenset({"kernel_build_seconds"}),
    "real_ecology_benchmark.planners.pbvi.PointBasedPlanner": frozenset({"elapsed_seconds"}),
}


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _class_from_name(name: str, *, track: str, repository_root: Path) -> type[Any]:
    if not name.startswith("real_ecology_benchmark."):
        raise ContractError("serialized object class is outside the frozen package allowlist")
    module_name, _, qualname = name.rpartition(".")
    module = importlib.import_module(module_name)
    value: Any = module
    for part in qualname.split("."):
        if part == "<locals>" or not part:
            raise ContractError("serialized object has an unimportable class")
        value = getattr(value, part)
    if not isinstance(value, type):
        raise ContractError("serialized class reference did not resolve to a type")
    source = inspect.getsourcefile(value)
    if source is None:
        raise ContractError("frozen object class has no inspectable source file")
    source_path = Path(source).resolve()
    allowed_root = (repository_root / TRACK_ROOTS[track]).resolve()
    if not source_path.is_relative_to(allowed_root):
        raise ContractError("serialized class source is outside the registered frozen track")
    return value


def _object_attributes(value: Any) -> dict[str, Any]:
    attributes = dict(vars(value)) if hasattr(value, "__dict__") else {}
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in {"__dict__", "__weakref__"} and hasattr(value, name):
                attributes[name] = getattr(value, name)
    if not attributes:
        raise ContractError(f"frozen object {_qualified_name(value)} has no serializable state")
    return attributes


def _encode(
    value: Any,
    *,
    track: str,
    repository_root: Path,
    active: set[int],
    object_memo: dict[int, str] | None = None,
) -> Any:
    if object_memo is None:
        object_memo = {}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ContractError("non-finite fitted scalar is forbidden")
        return {"$float64_hex": np.float64(value).hex()}
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value):
            raise ContractError("non-finite fitted NumPy scalar is forbidden")
        return {
            "$numpy_scalar": {
                "dtype": value.dtype.str,
                "bytes_b64": base64.b64encode(value.tobytes()).decode("ascii"),
            }
        }
    if isinstance(value, np.ndarray):
        identity = id(value)
        if identity in object_memo:
            return {"$object_ref": object_memo[identity]}
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ContractError("non-finite fitted array is forbidden")
        object_id = f"object-{len(object_memo)}"
        object_memo[identity] = object_id
        contiguous = np.ascontiguousarray(value)
        return {
            "$ndarray": {
                "id": object_id,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "bytes_b64": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
            }
        }
    if isinstance(value, bytes):
        return {"$bytes_b64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return {"$path": value.as_posix()}
    identity = id(value)
    if identity in object_memo:
        return {"$object_ref": object_memo[identity]}
    if identity in active:
        raise ContractError("cyclic fitted-object graphs are not serializable")
    active.add(identity)
    try:
        if isinstance(value, tuple):
            return {
                "$tuple": [
                    _encode(
                        item,
                        track=track,
                        repository_root=repository_root,
                        active=active,
                        object_memo=object_memo,
                    )
                    for item in value
                ]
            }
        if isinstance(value, list):
            return {
                "$list": [
                    _encode(
                        item,
                        track=track,
                        repository_root=repository_root,
                        active=active,
                        object_memo=object_memo,
                    )
                    for item in value
                ]
            }
        if isinstance(value, (set, frozenset)):
            items = [
                _encode(
                    item,
                    track=track,
                    repository_root=repository_root,
                    active=active,
                    object_memo=object_memo,
                )
                for item in value
            ]
            items.sort(key=canonical_json_bytes)
            return {"$frozenset" if isinstance(value, frozenset) else "$set": items}
        if isinstance(value, Mapping):
            items = [
                [
                    _encode(
                        key,
                        track=track,
                        repository_root=repository_root,
                        active=active,
                        object_memo=object_memo,
                    ),
                    _encode(
                        item,
                        track=track,
                        repository_root=repository_root,
                        active=active,
                        object_memo=object_memo,
                    ),
                ]
                for key, item in value.items()
            ]
            items.sort(key=lambda pair: canonical_json_bytes(pair[0]))
            return {"$mapping": items}
        if isinstance(value, np.random.Generator):
            return {
                "$numpy_generator": {
                    "bit_generator": type(value.bit_generator).__name__,
                    "state": _encode(
                        value.bit_generator.state,
                        track=track,
                        repository_root=repository_root,
                        active=active,
                        object_memo=object_memo,
                    ),
                }
            }
        cls = _class_from_name(_qualified_name(value), track=track, repository_root=repository_root)
        if cls is not type(value):
            raise ContractError("frozen class resolution changed during serialization")
        object_id = f"object-{len(object_memo)}"
        object_memo[identity] = object_id
        attributes = _object_attributes(value)
        return {
            "$object": {
                "id": object_id,
                "class": _qualified_name(value),
                "attributes": {
                    name: _encode(
                        item,
                        track=track,
                        repository_root=repository_root,
                        active=active,
                        object_memo=object_memo,
                    )
                    for name, item in sorted(attributes.items())
                },
            }
        }
    finally:
        active.remove(identity)


def _decode(
    value: Any,
    *,
    track: str,
    repository_root: Path,
    object_memo: dict[str, Any] | None = None,
) -> Any:
    if object_memo is None:
        object_memo = {}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if not isinstance(value, Mapping) or len(value) != 1:
        raise ContractError("malformed fitted-object graph node")
    tag, payload = next(iter(value.items()))
    if tag == "$float64_hex":
        number = np.float64(float.fromhex(payload))
        if not np.isfinite(number) or number.hex() != payload:
            raise ContractError("invalid serialized float64")
        return float(number)
    if tag == "$numpy_scalar":
        require_exact_keys(payload, {"dtype", "bytes_b64"}, "NumPy scalar")
        dtype = np.dtype(payload["dtype"])
        raw = base64.b64decode(payload["bytes_b64"], validate=True)
        if len(raw) != dtype.itemsize:
            raise ContractError("NumPy scalar byte length mismatch")
        return np.frombuffer(raw, dtype=dtype, count=1)[0]
    if tag == "$ndarray":
        require_exact_keys(payload, {"id", "dtype", "shape", "bytes_b64"}, "NumPy array")
        object_id = payload["id"]
        if not isinstance(object_id, str) or not object_id or object_id in object_memo:
            raise ContractError("NumPy array object identity is invalid")
        dtype = np.dtype(payload["dtype"])
        shape = payload["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape
        ):
            raise ContractError("NumPy array shape is invalid")
        raw = base64.b64decode(payload["bytes_b64"], validate=True)
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected:
            raise ContractError("NumPy array byte length mismatch")
        array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        object_memo[object_id] = array
        return array
    if tag == "$bytes_b64":
        return base64.b64decode(payload, validate=True)
    if tag == "$path":
        if not isinstance(payload, str):
            raise ContractError("serialized path must be text")
        return Path(payload)
    if tag in {"$tuple", "$list", "$set", "$frozenset"}:
        if not isinstance(payload, list):
            raise ContractError("serialized container must be a list")
        items = [
            _decode(
                item,
                track=track,
                repository_root=repository_root,
                object_memo=object_memo,
            )
            for item in payload
        ]
        if tag == "$tuple":
            return tuple(items)
        if tag == "$list":
            return items
        return frozenset(items) if tag == "$frozenset" else set(items)
    if tag == "$mapping":
        if not isinstance(payload, list):
            raise ContractError("serialized mapping must be an item list")
        result: dict[Any, Any] = {}
        for pair in payload:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ContractError("serialized mapping item is malformed")
            key = _decode(
                pair[0],
                track=track,
                repository_root=repository_root,
                object_memo=object_memo,
            )
            if key in result:
                raise ContractError("serialized mapping contains duplicate keys")
            result[key] = _decode(
                pair[1],
                track=track,
                repository_root=repository_root,
                object_memo=object_memo,
            )
        return result
    if tag == "$numpy_generator":
        require_exact_keys(payload, {"bit_generator", "state"}, "NumPy generator")
        bit_generator_type = getattr(np.random, payload["bit_generator"], None)
        if bit_generator_type not in {
            np.random.PCG64,
            np.random.MT19937,
            np.random.Philox,
            np.random.SFC64,
        }:
            raise ContractError("unregistered NumPy bit generator")
        bit_generator = bit_generator_type()
        bit_generator.state = _decode(
            payload["state"],
            track=track,
            repository_root=repository_root,
            object_memo=object_memo,
        )
        return np.random.Generator(bit_generator)
    if tag == "$object_ref":
        if not isinstance(payload, str) or payload not in object_memo:
            raise ContractError("serialized frozen-object reference is unresolved")
        return object_memo[payload]
    if tag == "$object":
        require_exact_keys(payload, {"id", "class", "attributes"}, "frozen object")
        cls = _class_from_name(payload["class"], track=track, repository_root=repository_root)
        attributes = payload["attributes"]
        object_id = payload["id"]
        if (
            not isinstance(object_id, str)
            or not object_id
            or object_id in object_memo
            or not isinstance(attributes, Mapping)
            or not attributes
        ):
            raise ContractError("serialized frozen object has no attributes")
        instance = object.__new__(cls)
        object_memo[object_id] = instance
        for name, item in attributes.items():
            if not isinstance(name, str) or not name:
                raise ContractError("serialized object attribute name is invalid")
            object.__setattr__(
                instance,
                name,
                _decode(
                    item,
                    track=track,
                    repository_root=repository_root,
                    object_memo=object_memo,
                ),
            )
        return instance
    raise ContractError(f"unknown fitted-object graph tag: {tag}")


def _validate_root(component: str, value: Any) -> tuple[str, frozenset[str]]:
    if component not in ROOT_SPECS:
        raise ContractError("unregistered frozen fitted-object component")
    track, expected_class, required_attributes = ROOT_SPECS[component]
    if _qualified_name(value) != expected_class:
        raise ContractError("fitted-object root class does not match its registered component")
    attributes = _object_attributes(value)
    missing = sorted(required_attributes - set(attributes))
    if missing:
        raise ContractError(f"fitted-object root is incomplete; missing={missing}")
    if component == "general_dynamics_ensemble":
        members = attributes["members"]
        if not isinstance(members, list) or len(members) != 5:
            raise ContractError("registered general dynamics requires exactly five fitted members")
        for member in members:
            required_member = {"coefficients", "residual_sigma", "num_actions", "K_ref", "env_cfg"}
            missing_member = required_member - set(_object_attributes(member))
            if missing_member:
                raise ContractError("fitted dynamics member state is incomplete")
    if component.endswith("_fitted_policy"):
        if attributes.get("hidden") is not True or "env_cfg" in attributes:
            raise ContractError(
                "Stage B fitted policies require the sanitized hidden MethodContext"
            )
    if component == "ogsrl_fitted_policy":
        if "surrogate" not in attributes or "s_low" not in attributes:
            raise ContractError("hidden fitted OGSRL lacks its surrogate or safety calibration")
        if "critics" in attributes:
            critics = attributes["critics"]
            if not isinstance(critics, list) or len(critics) != 3:
                raise ContractError("present OGSRL critics must contain reward/safety/OOD members")
    if component == "evd_fitted_policy":
        if not isinstance(attributes["q_members"], list) or len(attributes["q_members"]) != 20:
            raise ContractError("fitted EVD must contain all twenty Q members")
    return track, required_attributes


def serialize_frozen_fitted_object(
    component: str,
    fitted_object: Any,
    *,
    repository_root: Path,
) -> bytes:
    """Serialize every attribute of one registered real fitted frozen object."""

    root = Path(repository_root).resolve()
    track, required_attributes = _validate_root(component, fitted_object)
    graph = _encode(fitted_object, track=track, repository_root=root, active=set())
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "component": component,
        "track": track,
        "root_class": _qualified_name(fitted_object),
        "source_only_sha256": EXPECTED_SOURCE_HASHES[f"{track}_source_only_sha256"],
        "required_root_attributes": sorted(required_attributes),
        "complete_root_attribute_names": sorted(_object_attributes(fitted_object)),
        "object_graph": graph,
    }
    return canonical_json_bytes(envelope)


def load_frozen_fitted_object(
    payload: bytes,
    *,
    repository_root: Path,
) -> Any:
    """Load a canonical frozen-object graph into a fresh instance."""

    envelope = strict_json_loads(payload)
    if not isinstance(envelope, Mapping) or canonical_json_bytes(envelope) != payload:
        raise ContractError("fitted-object payload must be canonical JSON")
    require_exact_keys(
        envelope,
        {
            "schema_version",
            "component",
            "track",
            "root_class",
            "source_only_sha256",
            "required_root_attributes",
            "complete_root_attribute_names",
            "object_graph",
        },
        "fitted-object envelope",
    )
    if envelope["schema_version"] != SCHEMA_VERSION or envelope["component"] not in ROOT_SPECS:
        raise ContractError("fitted-object envelope schema/component mismatch")
    component = envelope["component"]
    track, expected_class, required = ROOT_SPECS[component]
    if (
        envelope["track"] != track
        or envelope["root_class"] != expected_class
        or envelope["source_only_sha256"] != EXPECTED_SOURCE_HASHES[f"{track}_source_only_sha256"]
        or envelope["required_root_attributes"] != sorted(required)
    ):
        raise ContractError("fitted-object envelope controlling binding mismatch")
    root = Path(repository_root).resolve()
    value = _decode(envelope["object_graph"], track=track, repository_root=root)
    _validate_root(component, value)
    if sorted(_object_attributes(value)) != envelope["complete_root_attribute_names"]:
        raise ContractError("fitted-object reload omitted or fabricated root attributes")
    if serialize_frozen_fitted_object(component, value, repository_root=root) != payload:
        raise ContractError("fitted-object fresh reload byte parity failed")
    return value


def _parity_hash(value: Any, *, track: str, repository_root: Path) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            _encode(value, track=track, repository_root=repository_root, active=set())
        )
    )


def _normalize_post_call_graph(value: Any) -> Any:
    """Zero only registered wall-clock counters before post-action comparison.

    The counters remain losslessly serialized in the fitted artifact. They are excluded only
    from post-call equivalence because sequential parity calls cannot have bit-identical wall
    time; fitted parameters, caches, decisions, invocation counts, diagnostics, and RNG states
    remain exact.
    """

    if isinstance(value, list):
        return [_normalize_post_call_graph(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    result = {key: _normalize_post_call_graph(item) for key, item in value.items()}
    object_node = result.get("$object")
    if isinstance(object_node, Mapping):
        class_name = object_node.get("class")
        attributes = object_node.get("attributes")
        volatile = POST_CALL_VOLATILE_FIELDS.get(class_name, frozenset())
        if isinstance(attributes, dict):
            for field in volatile:
                if field in attributes:
                    attributes[field] = {"$float64_hex": np.float64(0.0).hex()}
    return result


def _post_call_state_hash(value: Any, *, track: str, repository_root: Path) -> str:
    encoded = _encode(value, track=track, repository_root=repository_root, active=set())
    return sha256_bytes(canonical_json_bytes(_normalize_post_call_graph(encoded)))


def _first_graph_difference(left: Any, right: Any, path: str = "root") -> str:
    if type(left) is not type(right):
        return f"{path}:type"
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return f"{path}:keys"
        for key in sorted(left):
            difference = _first_graph_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}:length"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_graph_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    return "" if left == right else path


def validate_frozen_fitted_object_roundtrip(
    component: str,
    fitted_object: Any,
    *,
    operation: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
    repository_root: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Reload a real frozen object and require real prediction/action plus state parity."""

    if operation not in ALLOWED_OPERATIONS.get(component, frozenset()):
        raise ContractError("unregistered fitted-object parity operation")
    root = Path(repository_root).resolve()
    payload = serialize_frozen_fitted_object(component, fitted_object, repository_root=root)
    reloaded = load_frozen_fitted_object(payload, repository_root=root)
    track = ROOT_SPECS[component][0]
    encoded_args = _encode(tuple(args), track=track, repository_root=root, active=set())
    encoded_kwargs = _encode(dict(kwargs or {}), track=track, repository_root=root, active=set())
    original_args = _decode(encoded_args, track=track, repository_root=root)
    reloaded_args = _decode(encoded_args, track=track, repository_root=root)
    original_kwargs = _decode(encoded_kwargs, track=track, repository_root=root)
    reloaded_kwargs = _decode(encoded_kwargs, track=track, repository_root=root)
    original_output = getattr(fitted_object, operation)(*original_args, **original_kwargs)
    reloaded_output = getattr(reloaded, operation)(*reloaded_args, **reloaded_kwargs)
    original_output_sha = _parity_hash(original_output, track=track, repository_root=root)
    reloaded_output_sha = _parity_hash(reloaded_output, track=track, repository_root=root)
    if original_output_sha != reloaded_output_sha:
        raise ContractError("real fitted-object prediction/action parity failed")
    original_post_graph = _normalize_post_call_graph(
        _encode(fitted_object, track=track, repository_root=root, active=set())
    )
    reloaded_post_graph = _normalize_post_call_graph(
        _encode(reloaded, track=track, repository_root=root, active=set())
    )
    original_post = sha256_bytes(canonical_json_bytes(original_post_graph))
    reloaded_post = sha256_bytes(canonical_json_bytes(reloaded_post_graph))
    if original_post != reloaded_post:
        difference = _first_graph_difference(original_post_graph, reloaded_post_graph)
        raise ContractError(f"real fitted-object post-call state/RNG parity failed at {difference}")
    receipt = {
        "schema_version": "corrected_stageb_frozen_object_parity_v1",
        "component": component,
        "track": track,
        "root_class": _qualified_name(fitted_object),
        "serialized_sha256": sha256_bytes(payload),
        "operation": operation,
        "operation_input": {"args": encoded_args, "kwargs": encoded_kwargs},
        "operation_input_sha256": sha256_bytes(
            canonical_json_bytes({"args": encoded_args, "kwargs": encoded_kwargs})
        ),
        "output_sha256": original_output_sha,
        "post_call_state_sha256": original_post,
        "fresh_object_reload": True,
        "complete_instance_graph": True,
        "training_history_alone_qualifies": False,
        "result": "PASS",
    }
    return payload, receipt


def validate_frozen_object_parity_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_component: str,
    expected_serialized_sha256: str,
) -> None:
    require_exact_keys(
        receipt,
        {
            "schema_version",
            "component",
            "track",
            "root_class",
            "serialized_sha256",
            "operation",
            "operation_input",
            "operation_input_sha256",
            "output_sha256",
            "post_call_state_sha256",
            "fresh_object_reload",
            "complete_instance_graph",
            "training_history_alone_qualifies",
            "result",
        },
        "frozen fitted-object parity receipt",
    )
    if expected_component not in ROOT_SPECS:
        raise ContractError("unregistered expected fitted-object component")
    track, root_class, _required = ROOT_SPECS[expected_component]
    expected = {
        "schema_version": "corrected_stageb_frozen_object_parity_v1",
        "component": expected_component,
        "track": track,
        "root_class": root_class,
        "serialized_sha256": expected_serialized_sha256,
        "fresh_object_reload": True,
        "complete_instance_graph": True,
        "training_history_alone_qualifies": False,
        "result": "PASS",
    }
    for field, value in expected.items():
        if receipt[field] != value:
            raise ContractError(f"frozen fitted-object parity binding mismatch: {field}")
    if receipt["operation"] not in ALLOWED_OPERATIONS[expected_component]:
        raise ContractError("frozen fitted-object receipt operation mismatch")
    for field in (
        "serialized_sha256",
        "operation_input_sha256",
        "output_sha256",
        "post_call_state_sha256",
    ):
        require_sha256(receipt[field], f"frozen fitted-object receipt {field}")
    operation_input = receipt["operation_input"]
    if not isinstance(operation_input, Mapping):
        raise ContractError("frozen fitted-object operation input must be an object")
    require_exact_keys(
        operation_input,
        {"args", "kwargs"},
        "frozen fitted-object operation input",
    )
    if sha256_bytes(canonical_json_bytes(operation_input)) != receipt["operation_input_sha256"]:
        raise ContractError("frozen fitted-object operation input hash mismatch")


@contextmanager
def _registered_track_imports(track: str, repository_root: Path) -> Any:
    """Temporarily select one frozen package when both tracks share its import name."""

    if track not in TRACK_ROOTS:
        raise ContractError("unregistered frozen track")
    prefix = "real_ecology_benchmark"
    saved_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == prefix or name.startswith(f"{prefix}.")
    }
    saved_path = list(sys.path)
    for name in saved_modules:
        del sys.modules[name]
    track_root = str((repository_root / TRACK_ROOTS[track]).resolve())
    sys.path[:] = [
        item
        for item in sys.path
        if item not in {str((repository_root / value).resolve()) for value in TRACK_ROOTS.values()}
    ]
    sys.path.insert(0, track_root)
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def revalidate_frozen_object_parity(
    payload: bytes,
    receipt: Mapping[str, Any],
    *,
    expected_component: str,
    repository_root: Path,
) -> None:
    """Independently reload and replay a published real-object parity receipt."""

    root = Path(repository_root).resolve()
    expected_hash = sha256_bytes(payload)
    validate_frozen_object_parity_receipt(
        receipt,
        expected_component=expected_component,
        expected_serialized_sha256=expected_hash,
    )
    track = ROOT_SPECS[expected_component][0]
    operation_input = receipt["operation_input"]
    with _registered_track_imports(track, root):
        fitted = load_frozen_fitted_object(payload, repository_root=root)
        args = _decode(
            operation_input["args"],
            track=track,
            repository_root=root,
        )
        kwargs = _decode(
            operation_input["kwargs"],
            track=track,
            repository_root=root,
        )
        if not isinstance(args, tuple) or not isinstance(kwargs, Mapping):
            raise ContractError("frozen fitted-object operation input types are invalid")
        replay_payload, replay_receipt = validate_frozen_fitted_object_roundtrip(
            expected_component,
            fitted,
            operation=receipt["operation"],
            args=args,
            kwargs=kwargs,
            repository_root=root,
        )
    if replay_payload != payload or replay_receipt != dict(receipt):
        raise ContractError("published frozen fitted-object parity cannot be reproduced")


def scientific_component_for_method(method: str) -> str:
    mapping = {
        "plus_adapted_ricker_only_pbvi": "plus_fitted_policy",
        "moor_adapted_ricker_misspec_pbvi": "moor_fitted_policy",
        "refplan": "refplan_fitted_policy",
        "ogsrl": "ogsrl_fitted_policy",
        "bamcts": "bamcts_fitted_policy",
        "ensemble_value_disagreement_pessimism": "evd_fitted_policy",
    }
    return require_nonempty_string(mapping.get(method), "registered fitted-object method")
