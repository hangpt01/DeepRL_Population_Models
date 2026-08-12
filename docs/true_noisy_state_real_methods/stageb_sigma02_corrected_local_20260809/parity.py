"""Fail-closed accepted-parity eligibility, comparison, and diagnostic contracts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical_plan import RNG_RECEIPT_SCHEMA_VERSION, validate_rng_contract_document
from .common import (
    ContractError,
    canonical_json_bytes,
    require_exact_keys,
    require_git_sha,
    require_sha256,
    sha256_bytes,
    sha256_file,
)


PARITY_BASELINE_BINDING_SCHEMA_VERSION = "corrected_stageb_parity_baseline_binding_v1"
PARITY_COMPARISON_SCHEMA_VERSION = "corrected_stageb_parity_comparison_v1"
PARITY_FAILURE_SCHEMA_VERSION = "corrected_stageb_arm_o_parity_failure_v1"
PARITY_PASS_SCHEMA_VERSION = "corrected_stageb_task_arm_o_parity_v2"
PARITY_FAILURE_NAMESPACE = "arm-o-failure-diagnostics"
PARITY_FAILURE_FILENAME = "ACCEPTED_PARITY_FAILURE.json"
PARITY_TOLERANCE = 1e-9
PARITY_IDENTITY_FIELDS = ("episode", "seed", "block_seed")
LEGACY_FAST_TRACK_CANARY_PATH_COMPONENT = "i2b_fasttrack_integration_canary_20260808"
INTERPRETER_FIELDS = (
    "role",
    "track",
    "absolute_interpreter_path",
    "resolved_executable_path",
    "python_version",
    "full_python_version",
    "numpy_version",
)
REQUIRED_EXECUTION_DIMENSIONS = (
    "repository_commit",
    "source_manifest_sha256",
    "stageb_driver_sha256",
    "producer_registration_sha256",
    "producer_driver_inputs_sha256",
    "rng_contract",
    "interpreter",
    "fitted_object_sha256",
    "component_hashes",
    "artifact_plan_sha256",
    "evaluator",
    "comparison_contract",
)
_TRUTH_FIELD_TOKENS = (
    "true_return",
    "true_state",
    "truth",
    "safety_threshold",
    "mvp_threshold",
)
_FORBIDDEN_DIAGNOSTIC_KEYS = frozenset(
    {
        "next_states",
        "runtime_next_states",
        "truth_trajectory",
        "true_return",
        "current_true_abundance",
        "min_true_state",
        "final_true_state",
        "safety_threshold",
        "mvp_threshold",
        "reward_true",
    }
)


class ParityEligibilityError(ContractError):
    """Carry a safe eligibility assessment to the failure-only publisher."""

    def __init__(self, message: str, assessment: Mapping[str, Any]):
        super().__init__(message)
        self.assessment = dict(assessment)


def is_legacy_fast_track_canary_path(path: Path) -> bool:
    return LEGACY_FAST_TRACK_CANARY_PATH_COMPONENT in Path(path).parts


def truth_bearing_field(field: str) -> bool:
    lowered = field.lower()
    return any(token in lowered for token in _TRUTH_FIELD_TOKENS)


def canonical_relative_delta(observed: float, reference: float) -> float:
    """Return |O-R|/max(|O|,|R|), with the all-zero case defined as zero."""

    observed_value = float(observed)
    reference_value = float(reference)
    if not math.isfinite(observed_value) or not math.isfinite(reference_value):
        raise ContractError("relative parity delta requires finite operands")
    absolute = abs(observed_value - reference_value)
    scale = max(abs(observed_value), abs(reference_value))
    return 0.0 if scale == 0.0 else absolute / scale


def _string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ContractError(f"{label} must be a{' non-empty' if nonempty else ''} list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ContractError(f"{label} entries must be nonempty strings")
    if len(set(value)) != len(value):
        raise ContractError(f"{label} entries must be unique")
    return list(value)


def build_parity_comparison_contract(
    *,
    reference_columns: Sequence[str],
    exact_fields: Sequence[str],
    numeric_fields: Sequence[str],
    excluded_fields: Sequence[str],
    absolute_tolerance: float = PARITY_TOLERANCE,
) -> dict[str, Any]:
    value = {
        "schema_version": PARITY_COMPARISON_SCHEMA_VERSION,
        "absolute_tolerance": absolute_tolerance,
        "identity_fields": list(PARITY_IDENTITY_FIELDS),
        "reference_columns": list(reference_columns),
        "exact_fields": list(exact_fields),
        "numeric_fields": list(numeric_fields),
        "excluded_fields": list(excluded_fields),
    }
    return validate_parity_comparison_contract(value)


def validate_parity_comparison_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("parity comparison contract must be an object")
    require_exact_keys(
        value,
        {
            "schema_version",
            "absolute_tolerance",
            "identity_fields",
            "reference_columns",
            "exact_fields",
            "numeric_fields",
            "excluded_fields",
        },
        "parity comparison contract",
    )
    if value["schema_version"] != PARITY_COMPARISON_SCHEMA_VERSION:
        raise ContractError("parity comparison contract schema mismatch")
    tolerance = value["absolute_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ContractError("parity absolute tolerance must be numeric")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance != PARITY_TOLERANCE:
        raise ContractError("parity absolute tolerance differs from the registered value")
    identity = _string_list(value["identity_fields"], "parity identity fields", nonempty=True)
    if tuple(identity) != PARITY_IDENTITY_FIELDS:
        raise ContractError("parity identity field contract mismatch")
    columns = _string_list(value["reference_columns"], "parity reference columns", nonempty=True)
    exact = _string_list(value["exact_fields"], "parity exact fields")
    numeric = _string_list(value["numeric_fields"], "parity numeric fields")
    excluded = _string_list(value["excluded_fields"], "parity excluded fields")
    sets = [set(exact), set(numeric), set(excluded)]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ContractError("parity comparison field classes overlap")
    if set(columns) != set().union(*sets):
        raise ContractError("parity comparison fields do not cover the reference columns")
    if not set(identity).issubset(exact):
        raise ContractError("parity identity fields must use exact comparison")
    return {
        "schema_version": PARITY_COMPARISON_SCHEMA_VERSION,
        "absolute_tolerance": tolerance,
        "identity_fields": identity,
        "reference_columns": columns,
        "exact_fields": exact,
        "numeric_fields": numeric,
        "excluded_fields": excluded,
    }


def _validate_interpreter(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ContractError("parity baseline interpreter binding must be an object")
    require_exact_keys(value, set(INTERPRETER_FIELDS), "parity baseline interpreter")
    parsed: dict[str, str] = {}
    for field in INTERPRETER_FIELDS:
        item = value[field]
        if not isinstance(item, str) or not item:
            raise ContractError(f"parity baseline interpreter {field} must be nonempty text")
        parsed[field] = item
    return parsed


def _validate_evaluator(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("parity baseline evaluator binding must be an object")
    require_exact_keys(
        value,
        {
            "family",
            "reward_mode",
            "config_sha256",
            "evaluation_identity_sha256",
            "episode_ids",
            "horizon",
            "discount",
            "num_actions",
        },
        "parity baseline evaluator",
    )
    if value["family"] != "allee" or value["reward_mode"] != "safe":
        raise ContractError("parity baseline evaluator/reward binding mismatch")
    config = require_sha256(value["config_sha256"], "parity baseline config")
    identity = require_sha256(
        value["evaluation_identity_sha256"], "parity baseline evaluation identity"
    )
    episodes = value["episode_ids"]
    if not isinstance(episodes, list) or len(episodes) != 20:
        raise ContractError("parity baseline must bind twenty evaluation identities")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in episodes):
        raise ContractError("parity baseline episode identities must be integers")
    if value["horizon"] != 50 or value["discount"] != 0.95 or value["num_actions"] != 11:
        raise ContractError("parity baseline horizon/discount/action binding mismatch")
    return {
        "family": "allee",
        "reward_mode": "safe",
        "config_sha256": config,
        "evaluation_identity_sha256": identity,
        "episode_ids": list(episodes),
        "horizon": 50,
        "discount": 0.95,
        "num_actions": 11,
    }


def validate_parity_baseline_binding(value: Any) -> dict[str, Any]:
    """Validate either an explicitly legacy binding or a complete eligible binding."""

    if not isinstance(value, Mapping):
        raise ContractError("parity baseline binding must be an object")
    classification = value.get("classification")
    if classification == "LEGACY_INELIGIBLE":
        require_exact_keys(
            value,
            {"schema_version", "classification", "producer_label", "reason", "missing_bindings"},
            "legacy parity baseline binding",
        )
        if value["schema_version"] != PARITY_BASELINE_BINDING_SCHEMA_VERSION:
            raise ContractError("parity baseline binding schema mismatch")
        producer = value["producer_label"]
        reason = value["reason"]
        if (
            not isinstance(producer, str)
            or not producer
            or not isinstance(reason, str)
            or not reason
        ):
            raise ContractError("legacy parity baseline labels must be nonempty text")
        missing = _string_list(value["missing_bindings"], "legacy missing bindings", nonempty=True)
        return {
            "schema_version": PARITY_BASELINE_BINDING_SCHEMA_VERSION,
            "classification": "LEGACY_INELIGIBLE",
            "producer_label": producer,
            "reason": reason,
            "missing_bindings": missing,
        }
    require_exact_keys(
        value,
        {
            "schema_version",
            "classification",
            "producer_registration_sha256",
            "producer_driver_inputs_sha256",
            "repository_commit",
            "source_manifest_sha256",
            "stageb_driver_sha256",
            "rng_contract",
            "interpreter",
            "fitted_object_sha256",
            "component_hashes",
            "artifact_plan_sha256",
            "evaluator",
            "comparison_contract",
        },
        "eligible parity baseline binding",
    )
    if value["schema_version"] != PARITY_BASELINE_BINDING_SCHEMA_VERSION:
        raise ContractError("parity baseline binding schema mismatch")
    if classification != "ELIGIBLE":
        raise ContractError("parity baseline classification is unsupported")
    components = value["component_hashes"]
    if not isinstance(components, Mapping) or not components:
        raise ContractError("parity baseline component hashes must be a non-empty object")
    parsed_components: dict[str, str] = {}
    for name, digest in components.items():
        if not isinstance(name, str) or not name:
            raise ContractError("parity baseline component names must be nonempty text")
        parsed_components[name] = require_sha256(digest, f"parity baseline component {name}")
    return {
        "schema_version": PARITY_BASELINE_BINDING_SCHEMA_VERSION,
        "classification": "ELIGIBLE",
        "producer_registration_sha256": require_sha256(
            value["producer_registration_sha256"], "parity baseline producer registration"
        ),
        "producer_driver_inputs_sha256": require_sha256(
            value["producer_driver_inputs_sha256"], "parity baseline producer DRIVER_INPUTS"
        ),
        "repository_commit": require_git_sha(
            value["repository_commit"], "parity baseline repository commit"
        ),
        "source_manifest_sha256": require_sha256(
            value["source_manifest_sha256"], "parity baseline source manifest"
        ),
        "stageb_driver_sha256": require_sha256(
            value["stageb_driver_sha256"], "parity baseline Stage B driver"
        ),
        "rng_contract": validate_rng_contract_document(value["rng_contract"]),
        "interpreter": _validate_interpreter(value["interpreter"]),
        "fitted_object_sha256": require_sha256(
            value["fitted_object_sha256"], "parity baseline fitted object"
        ),
        "component_hashes": dict(sorted(parsed_components.items())),
        "artifact_plan_sha256": require_sha256(
            value["artifact_plan_sha256"], "parity baseline artifact plan"
        ),
        "evaluator": _validate_evaluator(value["evaluator"]),
        "comparison_contract": validate_parity_comparison_contract(value["comparison_contract"]),
    }


def _assessment(
    *, classification: str, missing: Sequence[str], mismatched: Sequence[str]
) -> dict[str, Any]:
    eligible = classification == "ELIGIBLE" and not missing and not mismatched
    return {
        "schema_version": "corrected_stageb_parity_baseline_eligibility_v1",
        "classification": classification,
        "eligible": eligible,
        "missing_bindings": sorted(set(missing)),
        "mismatched_bindings": sorted(set(mismatched)),
        "result": "PASS" if eligible else "FAIL",
    }


def validate_parity_baseline_eligibility(
    value: Any,
    *,
    expected: Mapping[str, Any],
    accepted_csv: Path,
) -> dict[str, Any]:
    """Require complete, source-equivalent provenance before comparison may gate success."""

    if is_legacy_fast_track_canary_path(accepted_csv):
        assessment = _assessment(
            classification="LEGACY_INELIGIBLE",
            missing=REQUIRED_EXECUTION_DIMENSIONS,
            mismatched=("legacy_canary_misclassified_as_current",),
        )
        raise ParityEligibilityError(
            "2026-08-08 fast-track canary is ineligible for scientific acceptance",
            assessment,
        )

    if not isinstance(value, Mapping):
        assessment = _assessment(
            classification="UNBOUND_LEGACY",
            missing=REQUIRED_EXECUTION_DIMENSIONS,
            mismatched=(),
        )
        raise ParityEligibilityError(
            "accepted parity baseline is unbound legacy evidence", assessment
        )
    try:
        parsed = validate_parity_baseline_binding(value)
    except ContractError as exc:
        missing = [field for field in REQUIRED_EXECUTION_DIMENSIONS if field not in value]
        assessment = _assessment(
            classification=str(value.get("classification", "MALFORMED")),
            missing=missing,
            mismatched=("baseline_binding_structure",),
        )
        raise ParityEligibilityError(str(exc), assessment) from exc
    if parsed["classification"] == "LEGACY_INELIGIBLE":
        assessment = _assessment(
            classification="LEGACY_INELIGIBLE",
            missing=parsed["missing_bindings"],
            mismatched=(),
        )
        raise ParityEligibilityError(
            "legacy canary parity baseline is ineligible for scientific acceptance", assessment
        )
    mismatched: list[str] = []
    for field in (
        "repository_commit",
        "source_manifest_sha256",
        "stageb_driver_sha256",
        "rng_contract",
        "interpreter",
        "fitted_object_sha256",
        "component_hashes",
        "artifact_plan_sha256",
        "evaluator",
    ):
        if parsed[field] != expected[field]:
            mismatched.append(field)
    comparison = parsed["comparison_contract"]
    try:
        with Path(accepted_csv).open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = next(reader)
    except (OSError, StopIteration, UnicodeError) as exc:
        assessment = _assessment(
            classification="ELIGIBLE",
            missing=(),
            mismatched=(*mismatched, "reference_csv_header"),
        )
        raise ParityEligibilityError(
            "parity reference CSV header is unavailable", assessment
        ) from exc
    if header != comparison["reference_columns"]:
        mismatched.append("comparison_contract.reference_columns")
    assessment = _assessment(classification="ELIGIBLE", missing=(), mismatched=mismatched)
    if assessment["result"] != "PASS":
        raise ParityEligibilityError("parity baseline execution binding mismatch", assessment)
    return {"binding": parsed, "assessment": assessment}


def _scalar_hash(value: Any, *, numeric: bool) -> str:
    if numeric:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            canonical = {"numeric_text": str(value)}
        else:
            canonical = (
                {"finite_float_hex": parsed.hex()}
                if math.isfinite(parsed)
                else {"nonfinite_numeric_text": str(value)}
            )
    else:
        canonical = {"exact_text": str(value)}
    return sha256_bytes(canonical_json_bytes(canonical))


def _mismatch(
    *,
    episode: int,
    field: str,
    comparison_type: str,
    observed: Any,
    reference: Any,
    absolute_delta: float | None = None,
    relative_delta: float | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    numeric = comparison_type.startswith("numeric")
    return {
        "episode": episode,
        "field_name": field,
        "comparison_type": comparison_type,
        "truth_bearing": truth_bearing_field(field),
        "observed_canonical_sha256": _scalar_hash(observed, numeric=numeric),
        "reference_canonical_sha256": _scalar_hash(reference, numeric=numeric),
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "reason": reason,
    }


def compare_accepted_parity(
    rows: Sequence[Mapping[str, Any]],
    accepted_path: Path,
    comparison_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the registered fields without ever returning raw mismatch values."""

    contract = validate_parity_comparison_contract(comparison_contract)
    with Path(accepted_path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        accepted = list(reader)
        header = list(reader.fieldnames or ())
    mismatches: list[dict[str, Any]] = []
    exact_mismatches: list[dict[str, Any]] = []
    numeric_deltas: dict[str, float] = {}
    relative_deltas: dict[str, float] = {}
    failed_numeric: dict[str, float] = {}
    if header != contract["reference_columns"]:
        mismatches.append(
            _mismatch(
                episode=-1,
                field="reference_columns",
                comparison_type="exact",
                observed=header,
                reference=contract["reference_columns"],
                reason="reference header differs from eligible baseline binding",
            )
        )
    if len(rows) != 20 or len(accepted) != 20:
        mismatches.append(
            _mismatch(
                episode=-1,
                field="episode_count",
                comparison_type="exact",
                observed=len(rows),
                reference=len(accepted),
                reason="parity requires twenty ordered observed and reference rows",
            )
        )
    for index, (observed, reference) in enumerate(zip(rows, accepted)):
        for field in contract["exact_fields"]:
            if field not in observed:
                detail = _mismatch(
                    episode=index,
                    field=field,
                    comparison_type="exact",
                    observed={"missing": True},
                    reference=reference.get(field, {"missing": True}),
                    reason="missing observed field",
                )
                mismatches.append(detail)
                exact_mismatches.append({"episode": index, "field": field, "reason": "missing"})
            elif str(observed[field]) != str(reference[field]):
                detail = _mismatch(
                    episode=index,
                    field=field,
                    comparison_type="exact",
                    observed=observed[field],
                    reference=reference[field],
                    reason="exact mismatch",
                )
                mismatches.append(detail)
                exact_mismatches.append({"episode": index, "field": field})
        for field in contract["numeric_fields"]:
            if field not in observed:
                detail = _mismatch(
                    episode=index,
                    field=field,
                    comparison_type="numeric_missing",
                    observed={"missing": True},
                    reference=reference.get(field, {"missing": True}),
                    reason="missing observed field",
                )
                mismatches.append(detail)
                exact_mismatches.append({"episode": index, "field": field, "reason": "missing"})
                continue
            try:
                observed_number = float(observed[field])
                reference_number = float(reference[field])
            except (TypeError, ValueError):
                mismatches.append(
                    _mismatch(
                        episode=index,
                        field=field,
                        comparison_type="numeric_invalid",
                        observed=observed[field],
                        reference=reference[field],
                        reason="registered numeric field is not numeric",
                    )
                )
                exact_mismatches.append({"episode": index, "field": field, "reason": "not numeric"})
                continue
            if not math.isfinite(observed_number) or not math.isfinite(reference_number):
                mismatches.append(
                    _mismatch(
                        episode=index,
                        field=field,
                        comparison_type="numeric_non_finite",
                        observed=observed[field],
                        reference=reference[field],
                        reason="non-finite numeric parity input",
                    )
                )
                exact_mismatches.append({"episode": index, "field": field, "reason": "non-finite"})
                continue
            absolute = abs(observed_number - reference_number)
            relative = canonical_relative_delta(observed_number, reference_number)
            numeric_deltas[field] = max(numeric_deltas.get(field, 0.0), absolute)
            relative_deltas[field] = max(relative_deltas.get(field, 0.0), relative)
            if absolute > contract["absolute_tolerance"]:
                failed_numeric[field] = max(failed_numeric.get(field, 0.0), absolute)
                mismatches.append(
                    _mismatch(
                        episode=index,
                        field=field,
                        comparison_type="numeric",
                        observed=observed[field],
                        reference=reference[field],
                        absolute_delta=absolute,
                        relative_delta=relative,
                        reason="absolute tolerance exceeded",
                    )
                )
    result = "PASS" if not mismatches else "FAIL"
    return {
        "schema_version": PARITY_PASS_SCHEMA_VERSION,
        "episode_count": 20,
        "absolute_tolerance": contract["absolute_tolerance"],
        "identity_fields": list(contract["identity_fields"]),
        "exact_fields": list(contract["exact_fields"]),
        "numeric_fields": list(contract["numeric_fields"]),
        "excluded_nonscientific_fields": list(contract["excluded_fields"]),
        "maximum_absolute_deltas": dict(sorted(numeric_deltas.items())),
        "maximum_relative_deltas": dict(sorted(relative_deltas.items())),
        "exact_mismatches": exact_mismatches,
        "failed_numeric_fields": dict(sorted(failed_numeric.items())),
        "mismatches": mismatches,
        "historical_full_action_sequences_available": False,
        "new_full_action_sequences_recorded": True,
        "result": result,
    }


def canonical_reference_rows_sha256(path: Path) -> str:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return sha256_bytes(canonical_json_bytes(rows))


def _validate_parity_eligibility_assessment(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ContractError("parity eligibility assessment must be an object")
    require_exact_keys(
        value,
        {
            "schema_version",
            "classification",
            "eligible",
            "missing_bindings",
            "mismatched_bindings",
            "result",
        },
        "parity eligibility assessment",
    )
    if value["schema_version"] != "corrected_stageb_parity_baseline_eligibility_v1":
        raise ContractError("parity eligibility assessment schema mismatch")
    missing = _string_list(value["missing_bindings"], "parity missing bindings")
    mismatched = _string_list(value["mismatched_bindings"], "parity mismatched bindings")
    eligible = value["eligible"]
    result = value["result"]
    if not isinstance(value["classification"], str) or not value["classification"]:
        raise ContractError("parity eligibility classification must be nonempty text")
    if not isinstance(eligible, bool) or result not in {"PASS", "FAIL"}:
        raise ContractError("parity eligibility decision is malformed")
    if (result == "PASS") != eligible or (eligible and (missing or mismatched)):
        raise ContractError("parity eligibility decision is internally inconsistent")


def _validate_parity_current_provenance(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ContractError("current parity provenance must be an object")
    require_exact_keys(
        value,
        {
            "registration_sha256",
            "driver_inputs_sha256",
            "repository_commit",
            "source_manifest_sha256",
            "stageb_driver_sha256",
            "rng_contract",
            "interpreter",
            "fitted_object_sha256",
            "component_hashes",
            "artifact_plan_sha256",
            "evaluator",
        },
        "current parity provenance",
    )
    for field in (
        "registration_sha256",
        "driver_inputs_sha256",
        "source_manifest_sha256",
        "stageb_driver_sha256",
        "fitted_object_sha256",
        "artifact_plan_sha256",
    ):
        require_sha256(value[field], f"current parity {field}")
    require_git_sha(value["repository_commit"], "current parity repository commit")
    validate_rng_contract_document(value["rng_contract"])
    _validate_interpreter(value["interpreter"])
    components = value["component_hashes"]
    if not isinstance(components, Mapping) or not components:
        raise ContractError("current parity component hashes must be non-empty")
    for name, digest in components.items():
        if not isinstance(name, str) or not name:
            raise ContractError("current parity component name must be nonempty text")
        require_sha256(digest, f"current parity component {name}")
    _validate_evaluator(value["evaluator"])


def _validate_parity_reference_provenance(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ContractError("reference parity provenance must be an object")
    require_exact_keys(
        value,
        {
            "episodes_csv_absolute_path",
            "episodes_csv_registered_sha256",
            "episodes_csv_observed_sha256",
            "binding",
        },
        "reference parity provenance",
    )
    path = value["episodes_csv_absolute_path"]
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ContractError("reference parity path must be nonempty absolute text")
    registered = require_sha256(
        value["episodes_csv_registered_sha256"], "registered reference parity file"
    )
    observed = require_sha256(
        value["episodes_csv_observed_sha256"], "observed reference parity file"
    )
    if registered != observed:
        raise ContractError("reference parity file changed after registration")
    validate_parity_baseline_binding(value["binding"])


def _validate_parity_comparison_summary(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ContractError("parity comparison summary must be an object")
    for field in ("stage", "comparison_type", "absolute_tolerance", "result"):
        if field not in value:
            raise ContractError(f"parity comparison summary is missing {field}")
    if value["stage"] not in {
        "BASELINE_ELIGIBILITY",
        "BASELINE_TASK_BINDING",
        "VALUE_COMPARISON",
    }:
        raise ContractError("parity comparison stage is unsupported")
    if not isinstance(value["comparison_type"], str) or not value["comparison_type"]:
        raise ContractError("parity comparison type must be nonempty text")
    tolerance = value["absolute_tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ContractError("parity comparison tolerance must be numeric")
    if not math.isfinite(float(tolerance)) or float(tolerance) != PARITY_TOLERANCE:
        raise ContractError("parity comparison tolerance differs from registration")
    if value["result"] != "FAIL":
        raise ContractError("failure diagnostic comparison must remain FAIL")


def _validate_parity_rng_evidence(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ContractError("parity RNG evidence must be an object")
    require_exact_keys(
        value,
        {
            "receipt_schema_version",
            "rng_contract",
            "receipt_count",
            "receipts_canonical_sha256",
            "status",
        },
        "parity RNG evidence",
    )
    if value["receipt_schema_version"] != RNG_RECEIPT_SCHEMA_VERSION:
        raise ContractError("parity RNG receipt schema mismatch")
    validate_rng_contract_document(value["rng_contract"])
    count = value["receipt_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ContractError("parity RNG receipt count must be a nonnegative integer")
    require_sha256(value["receipts_canonical_sha256"], "parity RNG receipt collection")
    status = value["status"]
    if status == "VALIDATED_POST_ROLLOUT" and count != 1000:
        raise ContractError("post-rollout parity evidence must bind 1000 RNG receipts")
    if status == "NOT_CONSTRUCTED" and count != 0:
        raise ContractError("pre-rollout parity evidence cannot contain RNG receipts")
    if status not in {"VALIDATED_POST_ROLLOUT", "NOT_CONSTRUCTED"}:
        raise ContractError("parity RNG evidence status is unsupported")


def validate_parity_failure_diagnostic(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("parity failure diagnostic must be an object")
    require_exact_keys(
        value,
        {
            "schema_version",
            "result",
            "accepted",
            "qualifies_as_task_result",
            "automatic_retry_permitted",
            "task_index",
            "arm",
            "cell",
            "method",
            "current_provenance",
            "reference_provenance",
            "eligibility",
            "comparison",
            "mismatches",
            "observed_rows_sha256",
            "reference_rows_sha256",
            "rng_evidence",
            "information_boundary",
        },
        "parity failure diagnostic",
    )
    if value["schema_version"] != PARITY_FAILURE_SCHEMA_VERSION:
        raise ContractError("parity failure diagnostic schema mismatch")
    if (
        value["result"] != "FAIL"
        or value["accepted"] is not False
        or value["qualifies_as_task_result"] is not False
        or value["automatic_retry_permitted"] is not False
        or value["arm"] != "O"
    ):
        raise ContractError("parity failure diagnostic could be confused with success")
    for digest_field in ("observed_rows_sha256", "reference_rows_sha256"):
        require_sha256(value[digest_field], f"parity failure {digest_field}")
    task_index = value["task_index"]
    if isinstance(task_index, bool) or not isinstance(task_index, int) or not 0 <= task_index < 12:
        raise ContractError("parity failure task index is outside the registered range")
    for field in ("cell", "method"):
        if not isinstance(value[field], str) or not value[field]:
            raise ContractError(f"parity failure {field} must be nonempty text")
    _validate_parity_current_provenance(value["current_provenance"])
    _validate_parity_reference_provenance(value["reference_provenance"])
    _validate_parity_eligibility_assessment(value["eligibility"])
    _validate_parity_comparison_summary(value["comparison"])
    _validate_parity_rng_evidence(value["rng_evidence"])
    boundary = value["information_boundary"]
    if boundary != {
        "raw_truth_values_published": False,
        "runtime_next_states_published": False,
        "truth_trajectory_published": False,
        "mismatch_values_hashed": True,
    }:
        raise ContractError("parity failure information boundary mismatch")

    def reject_forbidden_keys(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).lower() in _FORBIDDEN_DIAGNOSTIC_KEYS:
                    raise ContractError(f"forbidden parity failure payload key: {key}")
                reject_forbidden_keys(nested)
        elif isinstance(item, list):
            for nested in item:
                reject_forbidden_keys(nested)

    reject_forbidden_keys(value)
    for detail in value["mismatches"]:
        if not isinstance(detail, Mapping):
            raise ContractError("parity mismatch detail must be an object")
        require_exact_keys(
            detail,
            {
                "episode",
                "field_name",
                "comparison_type",
                "truth_bearing",
                "observed_canonical_sha256",
                "reference_canonical_sha256",
                "absolute_delta",
                "relative_delta",
                "reason",
            },
            "parity mismatch detail",
        )
        require_sha256(detail["observed_canonical_sha256"], "observed mismatch value")
        require_sha256(detail["reference_canonical_sha256"], "reference mismatch value")
        for delta in (detail["absolute_delta"], detail["relative_delta"]):
            if delta is not None and (
                isinstance(delta, bool)
                or not isinstance(delta, (int, float))
                or not math.isfinite(delta)
            ):
                raise ContractError("parity mismatch deltas must be finite or null")
    return dict(value)


def reference_file_sha256(path: Path) -> str:
    return sha256_file(Path(path))
