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
REPLAY_DIAGNOSTIC_SCHEMA_VERSION = "corrected_stageb_frozen_replay_diagnostic_v1"
_DIAGNOSTIC_INLINE_VALUE_LIMIT = 512


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


def _graph_path(parent: str, child: str | int) -> str:
    token = str(child).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{token}"


def _object_reference_definitions(value: Any) -> dict[str, dict[str, str]]:
    definitions: dict[str, dict[str, str]] = {}

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for tag, kind in (("$object", "object"), ("$ndarray", "ndarray")):
                payload = node.get(tag)
                if isinstance(payload, Mapping) and isinstance(payload.get("id"), str):
                    object_id = payload["id"]
                    if object_id in definitions:
                        raise ContractError("diagnostic graph contains a duplicate object id")
                    definition = {"definition_path": path, "kind": kind}
                    if kind == "object" and isinstance(payload.get("class"), str):
                        definition["class"] = payload["class"]
                    definitions[object_id] = definition
            for key in sorted(node):
                visit(node[key], _graph_path(path, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, _graph_path(path, index))

    visit(value, "root")
    return definitions


def _canonicalize_object_references(value: Any) -> tuple[Any, dict[str, Any]]:
    """Replace traversal ids with structural paths for diagnosis, never parity."""

    definitions = _object_reference_definitions(value)
    references: dict[str, dict[str, str]] = {}

    def visit(node: Any, path: str) -> Any:
        if isinstance(node, list):
            return [visit(item, _graph_path(path, index)) for index, item in enumerate(node)]
        if not isinstance(node, Mapping):
            return node
        if set(node) == {"$object_ref"}:
            target = node["$object_ref"]
            if not isinstance(target, str) or target not in definitions:
                raise ContractError("diagnostic graph contains an unresolved object reference")
            target_path = definitions[target]["definition_path"]
            references[path] = {
                "raw_target_id": target,
                "target_definition_path": target_path,
            }
            return {"$object_ref": {"target_definition_path": target_path}}
        result = {key: visit(node[key], _graph_path(path, key)) for key in sorted(node)}
        for tag in ("$object", "$ndarray"):
            payload = node.get(tag)
            result_payload = result.get(tag)
            if isinstance(payload, Mapping) and isinstance(result_payload, dict):
                object_id = payload.get("id")
                if isinstance(object_id, str) and object_id in definitions:
                    result_payload["id"] = {
                        "definition_path": definitions[object_id]["definition_path"]
                    }
        return result

    canonical = visit(value, "root")
    raw_topology = {
        "definitions": {key: definitions[key] for key in sorted(definitions)},
        "references": {key: references[key] for key in sorted(references)},
    }
    canonical_topology = {
        "definitions": sorted(
            (
                {
                    "definition_path": item["definition_path"],
                    "kind": item["kind"],
                    **({"class": item["class"]} if "class" in item else {}),
                }
                for item in definitions.values()
            ),
            key=lambda item: item["definition_path"],
        ),
        "references": [
            {
                "reference_path": path,
                "target_definition_path": references[path]["target_definition_path"],
            }
            for path in sorted(references)
        ],
    }
    return canonical, {
        "raw": raw_topology,
        "raw_sha256": sha256_bytes(canonical_json_bytes(raw_topology)),
        "canonical": canonical_topology,
        "canonical_sha256": sha256_bytes(canonical_json_bytes(canonical_topology)),
    }


def _canonical_path_digest_map(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}

    def visit(node: Any, path: str) -> None:
        result[path] = sha256_bytes(canonical_json_bytes(node))
        if isinstance(node, Mapping):
            for key in sorted(node):
                visit(node[key], _graph_path(path, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, _graph_path(path, index))

    visit(value, "root")
    return result


def _rng_state_digests(value: Any) -> dict[str, str]:
    result: dict[str, str] = {}

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            generator = node.get("$numpy_generator")
            if isinstance(generator, Mapping) and "state" in generator:
                result[path] = sha256_bytes(canonical_json_bytes(generator["state"]))
            for key in sorted(node):
                visit(node[key], _graph_path(path, key))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, _graph_path(path, index))

    visit(value, "root")
    return result


def _bounded_diagnostic_value(value: Any) -> dict[str, Any]:
    encoded = canonical_json_bytes(value)
    result: dict[str, Any] = {
        "sha256": sha256_bytes(encoded),
        "canonical_byte_count": len(encoded),
    }
    if len(encoded) <= _DIAGNOSTIC_INLINE_VALUE_LIMIT:
        result["value"] = value
    else:
        result["value_omitted_from_receipt"] = True
    return result


def _value_at_difference_path(value: Any, path: str) -> Any:
    current = value
    suffix = path.removeprefix("root")
    while suffix:
        if suffix.startswith("."):
            suffix = suffix[1:]
            end = len(suffix)
            for separator in (".", "["):
                index = suffix.find(separator)
                if index >= 0:
                    end = min(end, index)
            current = current[suffix[:end]]
            suffix = suffix[end:]
        elif suffix.startswith("["):
            close = suffix.index("]")
            current = current[int(suffix[1:close])]
            suffix = suffix[close + 1 :]
        else:
            raise ContractError("diagnostic difference path is malformed")
    return current


def _graph_difference_evidence(left: Any, right: Any) -> dict[str, Any] | None:
    difference = _first_graph_difference(left, right)
    if not difference:
        return None
    if difference.endswith((":type", ":keys", ":length")):
        path, reason = difference.rsplit(":", 1)
    else:
        path, reason = difference, "value"
    return {
        "path": path,
        "reason": reason,
        "left": _bounded_diagnostic_value(_value_at_difference_path(left, path)),
        "right": _bounded_diagnostic_value(_value_at_difference_path(right, path)),
    }


def _graph_diagnostic(encoded_graph: Any) -> dict[str, Any]:
    normalized = _normalize_post_call_graph(encoded_graph)
    reference_normalized, topology = _canonicalize_object_references(normalized)
    return {
        "normalized_graph_sha256": sha256_bytes(canonical_json_bytes(normalized)),
        "reference_normalized_graph_sha256": sha256_bytes(
            canonical_json_bytes(reference_normalized)
        ),
        "canonical_per_path_sha256": _canonical_path_digest_map(reference_normalized),
        "object_reference_topology": topology,
        "rng_state_sha256_by_path": _rng_state_digests(reference_normalized),
        "_normalized_graph": normalized,
        "_reference_normalized_graph": reference_normalized,
    }


def _public_graph_diagnostic(
    value: Mapping[str, Any], *, include_normalized_graph: bool = False
) -> dict[str, Any]:
    result = {key: item for key, item in value.items() if not key.startswith("_")}
    if include_normalized_graph:
        result["normalized_graph"] = value["_normalized_graph"]
    return result


def diagnose_frozen_fitted_object_roundtrip(
    component: str,
    fitted_object: Any,
    *,
    operation: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
    repository_root: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Publish bounded producer-time graphs without relaxing the strict parity contract."""

    if operation not in ALLOWED_OPERATIONS.get(component, frozenset()):
        raise ContractError("unregistered fitted-object parity operation")
    root = Path(repository_root).resolve()
    payload = serialize_frozen_fitted_object(component, fitted_object, repository_root=root)
    reloaded = load_frozen_fitted_object(payload, repository_root=root)
    track = ROOT_SPECS[component][0]
    encoded_args = _encode(tuple(args), track=track, repository_root=root, active=set())
    encoded_kwargs = _encode(dict(kwargs or {}), track=track, repository_root=root, active=set())
    operation_input = {"args": encoded_args, "kwargs": encoded_kwargs}
    original_pre = _graph_diagnostic(
        _encode(fitted_object, track=track, repository_root=root, active=set())
    )
    reloaded_pre = _graph_diagnostic(
        _encode(reloaded, track=track, repository_root=root, active=set())
    )
    original_args = _decode(encoded_args, track=track, repository_root=root)
    reloaded_args = _decode(encoded_args, track=track, repository_root=root)
    original_kwargs = _decode(encoded_kwargs, track=track, repository_root=root)
    reloaded_kwargs = _decode(encoded_kwargs, track=track, repository_root=root)
    original_output = getattr(fitted_object, operation)(*original_args, **original_kwargs)
    reloaded_output = getattr(reloaded, operation)(*reloaded_args, **reloaded_kwargs)
    original_output_encoded = _encode(
        original_output, track=track, repository_root=root, active=set()
    )
    reloaded_output_encoded = _encode(
        reloaded_output, track=track, repository_root=root, active=set()
    )
    original_output_sha = sha256_bytes(canonical_json_bytes(original_output_encoded))
    reloaded_output_sha = sha256_bytes(canonical_json_bytes(reloaded_output_encoded))
    original_post = _graph_diagnostic(
        _encode(fitted_object, track=track, repository_root=root, active=set())
    )
    reloaded_post = _graph_diagnostic(
        _encode(reloaded, track=track, repository_root=root, active=set())
    )
    exact_equal = (
        original_post["normalized_graph_sha256"] == reloaded_post["normalized_graph_sha256"]
    )
    reference_equal = (
        original_post["reference_normalized_graph_sha256"]
        == reloaded_post["reference_normalized_graph_sha256"]
    )
    diagnostic = {
        "schema_version": REPLAY_DIAGNOSTIC_SCHEMA_VERSION,
        "component": component,
        "track": track,
        "root_class": _qualified_name(fitted_object),
        "serialized_sha256": sha256_bytes(payload),
        "normalization_contract": {
            "post_call_volatile_fields": {
                key: sorted(value) for key, value in sorted(POST_CALL_VOLATILE_FIELDS.items())
            },
            "new_volatile_fields_added": False,
            "object_reference_labels_canonicalized_for_diagnosis_only": True,
            "strict_parity_uses_reference_labels": True,
        },
        "operation": {
            "name": operation,
            "arguments": encoded_args,
            "keyword_arguments": encoded_kwargs,
            "arguments_sha256": sha256_bytes(canonical_json_bytes(encoded_args)),
            "keyword_arguments_sha256": sha256_bytes(canonical_json_bytes(encoded_kwargs)),
            "combined_input_sha256": sha256_bytes(canonical_json_bytes(operation_input)),
        },
        "pre_call": {
            "original_live_object": _public_graph_diagnostic(original_pre),
            "fresh_reloaded_object": _public_graph_diagnostic(reloaded_pre),
        },
        "post_call": {
            "original_live_object": _public_graph_diagnostic(
                original_post, include_normalized_graph=True
            ),
            "fresh_reloaded_object": _public_graph_diagnostic(
                reloaded_post, include_normalized_graph=True
            ),
        },
        "output": {
            "original_live_object_sha256": original_output_sha,
            "fresh_reloaded_object_sha256": reloaded_output_sha,
            "selected_action_or_output": _bounded_diagnostic_value(original_output_encoded),
        },
        "comparison": {
            "output_exactly_equal": original_output_sha == reloaded_output_sha,
            "strict_post_call_state_equal": exact_equal,
            "reference_normalized_post_call_state_equal": reference_equal,
            "object_reference_renumbering_only": not exact_equal and reference_equal,
            "first_exact_difference": _graph_difference_evidence(
                original_post["_normalized_graph"], reloaded_post["_normalized_graph"]
            ),
            "first_substantive_difference": _graph_difference_evidence(
                original_post["_reference_normalized_graph"],
                reloaded_post["_reference_normalized_graph"],
            ),
        },
        "producer_time_original_vs_reloaded_equality_passed": (
            original_output_sha == reloaded_output_sha and exact_equal
        ),
        "information_boundary": {
            "evaluator_constructed": False,
            "truth_accessed": False,
            "runtime_next_states_accessed": False,
            "returns_calculated": False,
        },
    }
    return payload, diagnostic


def validate_frozen_fitted_object_roundtrip_with_diagnostic(
    component: str,
    fitted_object: Any,
    *,
    operation: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None = None,
    repository_root: Path,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    payload, diagnostic = diagnose_frozen_fitted_object_roundtrip(
        component,
        fitted_object,
        operation=operation,
        args=args,
        kwargs=kwargs,
        repository_root=repository_root,
    )
    comparison = diagnostic["comparison"]
    if not comparison["output_exactly_equal"]:
        raise ContractError("real fitted-object prediction/action parity failed")
    if not comparison["strict_post_call_state_equal"]:
        difference = comparison["first_exact_difference"]
        path = difference["path"] if isinstance(difference, Mapping) else "root"
        raise ContractError(f"real fitted-object post-call state/RNG parity failed at {path}")
    operation_evidence = diagnostic["operation"]
    receipt = {
        "schema_version": "corrected_stageb_frozen_object_parity_v1",
        "component": component,
        "track": diagnostic["track"],
        "root_class": diagnostic["root_class"],
        "serialized_sha256": diagnostic["serialized_sha256"],
        "operation": operation,
        "operation_input": {
            "args": operation_evidence["arguments"],
            "kwargs": operation_evidence["keyword_arguments"],
        },
        "operation_input_sha256": operation_evidence["combined_input_sha256"],
        "output_sha256": diagnostic["output"]["original_live_object_sha256"],
        "post_call_state_sha256": diagnostic["post_call"]["original_live_object"][
            "normalized_graph_sha256"
        ],
        "fresh_object_reload": True,
        "complete_instance_graph": True,
        "training_history_alone_qualifies": False,
        "result": "PASS",
    }
    return payload, receipt, diagnostic


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

    payload, receipt, _diagnostic = validate_frozen_fitted_object_roundtrip_with_diagnostic(
        component,
        fitted_object,
        operation=operation,
        args=args,
        kwargs=kwargs,
        repository_root=repository_root,
    )
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


def validate_frozen_replay_diagnostic(
    diagnostic: Mapping[str, Any],
    *,
    expected_component: str,
    expected_serialized_sha256: str,
) -> None:
    """Validate the bounded diagnostic envelope and all internally checkable hashes."""

    require_exact_keys(
        diagnostic,
        {
            "schema_version",
            "component",
            "track",
            "root_class",
            "serialized_sha256",
            "normalization_contract",
            "operation",
            "pre_call",
            "post_call",
            "output",
            "comparison",
            "producer_time_original_vs_reloaded_equality_passed",
            "information_boundary",
        },
        "frozen replay diagnostic",
    )
    if expected_component not in ROOT_SPECS:
        raise ContractError("unregistered expected fitted-object component")
    track, root_class, _required = ROOT_SPECS[expected_component]
    expected = {
        "schema_version": REPLAY_DIAGNOSTIC_SCHEMA_VERSION,
        "component": expected_component,
        "track": track,
        "root_class": root_class,
        "serialized_sha256": expected_serialized_sha256,
    }
    for field, value in expected.items():
        if diagnostic[field] != value:
            raise ContractError(f"frozen replay diagnostic binding mismatch: {field}")
    operation = diagnostic["operation"]
    require_exact_keys(
        operation,
        {
            "name",
            "arguments",
            "keyword_arguments",
            "arguments_sha256",
            "keyword_arguments_sha256",
            "combined_input_sha256",
        },
        "frozen replay diagnostic operation",
    )
    if operation["name"] not in ALLOWED_OPERATIONS[expected_component]:
        raise ContractError("frozen replay diagnostic operation mismatch")
    calculated_operation_hashes = {
        "arguments_sha256": sha256_bytes(canonical_json_bytes(operation["arguments"])),
        "keyword_arguments_sha256": sha256_bytes(
            canonical_json_bytes(operation["keyword_arguments"])
        ),
        "combined_input_sha256": sha256_bytes(
            canonical_json_bytes(
                {"args": operation["arguments"], "kwargs": operation["keyword_arguments"]}
            )
        ),
    }
    for field, value in calculated_operation_hashes.items():
        if operation[field] != value:
            raise ContractError(f"frozen replay diagnostic operation hash mismatch: {field}")
    normalization = diagnostic["normalization_contract"]
    require_exact_keys(
        normalization,
        {
            "post_call_volatile_fields",
            "new_volatile_fields_added",
            "object_reference_labels_canonicalized_for_diagnosis_only",
            "strict_parity_uses_reference_labels",
        },
        "frozen replay diagnostic normalization contract",
    )
    expected_volatile = {
        key: sorted(value) for key, value in sorted(POST_CALL_VOLATILE_FIELDS.items())
    }
    if normalization != {
        "post_call_volatile_fields": expected_volatile,
        "new_volatile_fields_added": False,
        "object_reference_labels_canonicalized_for_diagnosis_only": True,
        "strict_parity_uses_reference_labels": True,
    }:
        raise ContractError("frozen replay diagnostic normalization contract changed")
    graph_required = {
        "normalized_graph_sha256",
        "reference_normalized_graph_sha256",
        "canonical_per_path_sha256",
        "object_reference_topology",
        "rng_state_sha256_by_path",
    }
    for phase in ("pre_call", "post_call"):
        phase_value = diagnostic[phase]
        require_exact_keys(
            phase_value,
            {"original_live_object", "fresh_reloaded_object"},
            f"frozen replay diagnostic {phase}",
        )
        for role in ("original_live_object", "fresh_reloaded_object"):
            graph = phase_value[role]
            expected_graph_keys = graph_required | (
                {"normalized_graph"} if phase == "post_call" else set()
            )
            require_exact_keys(
                graph, expected_graph_keys, f"frozen replay diagnostic {phase} graph"
            )
            for field in ("normalized_graph_sha256", "reference_normalized_graph_sha256"):
                require_sha256(graph[field], f"frozen replay diagnostic {phase} {field}")
            path_map = graph["canonical_per_path_sha256"]
            rng_map = graph["rng_state_sha256_by_path"]
            if not isinstance(path_map, Mapping) or "root" not in path_map:
                raise ContractError("frozen replay diagnostic path map is incomplete")
            if path_map["root"] != graph["reference_normalized_graph_sha256"]:
                raise ContractError("frozen replay diagnostic root path hash mismatch")
            if not isinstance(rng_map, Mapping):
                raise ContractError("frozen replay diagnostic RNG map is malformed")
            for path, digest in (*path_map.items(), *rng_map.items()):
                require_nonempty_string(path, "frozen replay diagnostic graph path")
                require_sha256(digest, "frozen replay diagnostic graph digest")
            topology = graph["object_reference_topology"]
            require_exact_keys(
                topology,
                {"raw", "raw_sha256", "canonical", "canonical_sha256"},
                "frozen replay diagnostic topology",
            )
            if (
                sha256_bytes(canonical_json_bytes(topology["raw"])) != topology["raw_sha256"]
                or sha256_bytes(canonical_json_bytes(topology["canonical"]))
                != topology["canonical_sha256"]
            ):
                raise ContractError("frozen replay diagnostic topology hash mismatch")
            if phase == "post_call":
                normalized_graph = graph["normalized_graph"]
                reference_normalized, calculated_topology = _canonicalize_object_references(
                    normalized_graph
                )
                calculated = {
                    "normalized_graph_sha256": sha256_bytes(canonical_json_bytes(normalized_graph)),
                    "reference_normalized_graph_sha256": sha256_bytes(
                        canonical_json_bytes(reference_normalized)
                    ),
                    "canonical_per_path_sha256": _canonical_path_digest_map(reference_normalized),
                    "object_reference_topology": calculated_topology,
                    "rng_state_sha256_by_path": _rng_state_digests(reference_normalized),
                }
                for field, expected_value in calculated.items():
                    if graph[field] != expected_value:
                        raise ContractError(
                            f"frozen replay diagnostic post-call graph mismatch: {field}"
                        )
    boundary = diagnostic["information_boundary"]
    expected_boundary = {
        "evaluator_constructed": False,
        "truth_accessed": False,
        "runtime_next_states_accessed": False,
        "returns_calculated": False,
    }
    if boundary != expected_boundary:
        raise ContractError("frozen replay diagnostic information boundary failed")
    output = diagnostic["output"]
    require_exact_keys(
        output,
        {
            "original_live_object_sha256",
            "fresh_reloaded_object_sha256",
            "selected_action_or_output",
        },
        "frozen replay diagnostic output",
    )
    for field in ("original_live_object_sha256", "fresh_reloaded_object_sha256"):
        require_sha256(output[field], f"frozen replay diagnostic output {field}")
    selected = output["selected_action_or_output"]
    allowed_selected_keys = {
        "sha256",
        "canonical_byte_count",
        "value",
        "value_omitted_from_receipt",
    }
    if not isinstance(selected, Mapping) or not set(selected).issubset(allowed_selected_keys):
        raise ContractError("frozen replay diagnostic selected output is malformed")
    if set(selected) not in (
        {"sha256", "canonical_byte_count", "value"},
        {"sha256", "canonical_byte_count", "value_omitted_from_receipt"},
    ):
        raise ContractError("frozen replay diagnostic selected output is incomplete")
    require_sha256(selected["sha256"], "frozen replay diagnostic selected output")
    byte_count = selected["canonical_byte_count"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
        raise ContractError("frozen replay diagnostic selected output byte count is invalid")
    if "value" in selected:
        encoded_value = canonical_json_bytes(selected["value"])
        if len(encoded_value) != byte_count or sha256_bytes(encoded_value) != selected["sha256"]:
            raise ContractError("frozen replay diagnostic selected output hash mismatch")
    elif selected["value_omitted_from_receipt"] is not True:
        raise ContractError("frozen replay diagnostic output omission marker is invalid")
    comparison = diagnostic["comparison"]
    require_exact_keys(
        comparison,
        {
            "output_exactly_equal",
            "strict_post_call_state_equal",
            "reference_normalized_post_call_state_equal",
            "object_reference_renumbering_only",
            "first_exact_difference",
            "first_substantive_difference",
        },
        "frozen replay diagnostic comparison",
    )
    expected_pass = (
        comparison["output_exactly_equal"] and comparison["strict_post_call_state_equal"]
    )
    if diagnostic["producer_time_original_vs_reloaded_equality_passed"] is not expected_pass:
        raise ContractError("frozen replay diagnostic producer equality flag mismatch")
    if comparison["object_reference_renumbering_only"] is not (
        not comparison["strict_post_call_state_equal"]
        and comparison["reference_normalized_post_call_state_equal"]
    ):
        raise ContractError("frozen replay diagnostic reference-renumbering flag mismatch")


def compare_frozen_replay_diagnostics(
    producer: Mapping[str, Any], independent: Mapping[str, Any]
) -> dict[str, Any]:
    """Locate the first canonical cross-process replay difference without classifying it."""

    if producer["serialized_sha256"] != independent["serialized_sha256"]:
        raise ContractError("producer and independent diagnostics bind different objects")
    if producer["operation"] != independent["operation"]:
        raise ContractError("producer and independent diagnostics bind different operations")
    first: dict[str, Any] | None = None
    role_results: dict[str, Any] = {}
    for role in ("original_live_object", "fresh_reloaded_object"):
        producer_graph = producer["post_call"][role]
        independent_graph = independent["post_call"][role]
        producer_map = producer_graph["canonical_per_path_sha256"]
        independent_map = independent_graph["canonical_per_path_sha256"]
        all_paths = sorted(set(producer_map) | set(independent_map))
        differing = [
            path for path in all_paths if producer_map.get(path) != independent_map.get(path)
        ]
        leaf_differences = [
            path
            for path in differing
            if not any(other.startswith(f"{path}/") for other in differing)
        ]
        role_results[role] = {
            "reference_normalized_graph_equal": not differing,
            "producer_reference_normalized_graph_sha256": producer_graph[
                "reference_normalized_graph_sha256"
            ],
            "independent_reference_normalized_graph_sha256": independent_graph[
                "reference_normalized_graph_sha256"
            ],
            "differing_path_count": len(differing),
        }
        if first is None and leaf_differences:
            path = leaf_differences[0]
            first = {
                "role": role,
                "path": path,
                "producer_sha256": producer_map.get(path),
                "independent_sha256": independent_map.get(path),
                "producer_path_present": path in producer_map,
                "independent_path_present": path in independent_map,
            }
    producer_rng = producer["post_call"]["original_live_object"]["rng_state_sha256_by_path"]
    independent_rng = independent["post_call"]["original_live_object"]["rng_state_sha256_by_path"]
    rng_paths = sorted(set(producer_rng) | set(independent_rng))
    rng_differences = [
        {
            "path": path,
            "producer_sha256": producer_rng.get(path),
            "independent_sha256": independent_rng.get(path),
        }
        for path in rng_paths
        if producer_rng.get(path) != independent_rng.get(path)
    ]
    topology_equal = all(
        producer["post_call"][role]["object_reference_topology"]["canonical_sha256"]
        == independent["post_call"][role]["object_reference_topology"]["canonical_sha256"]
        for role in ("original_live_object", "fresh_reloaded_object")
    )
    return {
        "schema_version": "corrected_stageb_cross_process_replay_comparison_v1",
        "serialized_sha256": producer["serialized_sha256"],
        "operation_input_sha256": producer["operation"]["combined_input_sha256"],
        "role_results": role_results,
        "first_substantive_difference": first,
        "rng_differences": rng_differences,
        "canonical_object_reference_topology_equal": topology_equal,
        "exact_cross_process_parity": first is None,
        "classification_deferred_until_source_inspection": True,
    }


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


def independently_diagnose_frozen_object_parity(
    payload: bytes,
    receipt: Mapping[str, Any],
    producer_diagnostic: Mapping[str, Any],
    *,
    expected_component: str,
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay once in the caller's process and compare it with producer graph evidence."""

    root = Path(repository_root).resolve()
    payload_sha256 = sha256_bytes(payload)
    validate_frozen_object_parity_receipt(
        receipt,
        expected_component=expected_component,
        expected_serialized_sha256=payload_sha256,
    )
    validate_frozen_replay_diagnostic(
        producer_diagnostic,
        expected_component=expected_component,
        expected_serialized_sha256=payload_sha256,
    )
    operation_input = receipt["operation_input"]
    if (
        producer_diagnostic["operation"]["name"] != receipt["operation"]
        or producer_diagnostic["operation"]["arguments"] != operation_input["args"]
        or producer_diagnostic["operation"]["keyword_arguments"] != operation_input["kwargs"]
    ):
        raise ContractError("producer diagnostic and parity receipt operation mismatch")
    track = ROOT_SPECS[expected_component][0]
    with _registered_track_imports(track, root):
        fitted = load_frozen_fitted_object(payload, repository_root=root)
        args = _decode(operation_input["args"], track=track, repository_root=root)
        kwargs = _decode(operation_input["kwargs"], track=track, repository_root=root)
        if not isinstance(args, tuple) or not isinstance(kwargs, Mapping):
            raise ContractError("frozen fitted-object operation input types are invalid")
        replay_payload, independent = diagnose_frozen_fitted_object_roundtrip(
            expected_component,
            fitted,
            operation=receipt["operation"],
            args=args,
            kwargs=kwargs,
            repository_root=root,
        )
    if replay_payload != payload:
        raise ContractError("independent diagnostic changed the serialized fitted object")
    validate_frozen_replay_diagnostic(
        independent,
        expected_component=expected_component,
        expected_serialized_sha256=payload_sha256,
    )
    comparison = compare_frozen_replay_diagnostics(producer_diagnostic, independent)
    return independent, comparison


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
