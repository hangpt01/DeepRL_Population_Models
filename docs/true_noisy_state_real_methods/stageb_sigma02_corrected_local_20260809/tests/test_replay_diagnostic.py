from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from .. import real_artifacts
from ..common import ContractError, canonical_json_bytes, sha256_bytes, strict_json_loads
from ..fit_probe import publish_fit_probe
from ..publication import load_success_receipt
from ..real_artifacts import (
    _canonicalize_object_references,
    _decode,
    _graph_diagnostic,
    _graph_difference_evidence,
    _registered_track_imports,
    compare_frozen_replay_diagnostics,
    diagnose_frozen_fitted_object_roundtrip,
    independently_diagnose_frozen_object_parity,
    load_frozen_fitted_object,
    revalidate_frozen_object_parity,
    validate_frozen_fitted_object_roundtrip_with_diagnostic,
    validate_frozen_object_parity_receipt,
    validate_frozen_replay_diagnostic,
)
from .conftest import HASHES
from .frozen_fixtures import frozen_object_evidence


REPOSITORY = Path(__file__).resolve().parents[4]
CELL = "amur_tiger__allee__sigma_0p2"


def _producer_diagnostic(method: str, component: str, track: str):
    payload, receipt_payload = frozen_object_evidence(method, CELL, HASHES[0])
    receipt = strict_json_loads(receipt_payload)
    with _registered_track_imports(track, REPOSITORY):
        fitted = load_frozen_fitted_object(payload, repository_root=REPOSITORY)
        args = _decode(receipt["operation_input"]["args"], track=track, repository_root=REPOSITORY)
        kwargs = _decode(
            receipt["operation_input"]["kwargs"], track=track, repository_root=REPOSITORY
        )
        replay_payload, diagnostic = diagnose_frozen_fitted_object_roundtrip(
            component,
            fitted,
            operation=receipt["operation"],
            args=args,
            kwargs=kwargs,
            repository_root=REPOSITORY,
        )
    assert replay_payload == payload
    validate_frozen_replay_diagnostic(
        diagnostic,
        expected_component=component,
        expected_serialized_sha256=sha256_bytes(payload),
    )
    return payload, receipt, diagnostic


def test_representative_general_policy_is_independently_revalidated():
    payload, receipt, producer = _producer_diagnostic("refplan", "refplan_fitted_policy", "general")
    independent, comparison = independently_diagnose_frozen_object_parity(
        payload,
        receipt,
        producer,
        expected_component="refplan_fitted_policy",
        repository_root=REPOSITORY,
    )
    assert independent["producer_time_original_vs_reloaded_equality_passed"] is True
    assert comparison["exact_cross_process_parity"] is True
    assert comparison["first_substantive_difference"] is None


def test_deliberate_post_call_difference_reports_exact_path_values_and_digests():
    left = {"state": {"cache": {"counter": 4}, "unchanged": [1, 2]}}
    right = deepcopy(left)
    right["state"]["cache"]["counter"] = 5
    evidence = _graph_difference_evidence(left, right)
    assert evidence == {
        "path": "root.state.cache.counter",
        "reason": "value",
        "left": {
            "sha256": sha256_bytes(canonical_json_bytes(4)),
            "canonical_byte_count": 1,
            "value": 4,
        },
        "right": {
            "sha256": sha256_bytes(canonical_json_bytes(5)),
            "canonical_byte_count": 1,
            "value": 5,
        },
    }


def test_rng_state_difference_is_reported_separately():
    _payload, _receipt, producer = _producer_diagnostic(
        "refplan", "refplan_fitted_policy", "general"
    )
    independent = deepcopy(producer)
    rng_map = independent["post_call"]["original_live_object"]["rng_state_sha256_by_path"]
    assert rng_map
    rng_path = sorted(rng_map)[0]
    rng_map[rng_path] = HASHES[7]
    comparison = compare_frozen_replay_diagnostics(producer, independent)
    assert comparison["rng_differences"] == [
        {
            "path": rng_path,
            "producer_sha256": producer["post_call"]["original_live_object"][
                "rng_state_sha256_by_path"
            ][rng_path],
            "independent_sha256": HASHES[7],
        }
    ]


def test_object_reference_renumbering_is_not_a_substantive_difference():
    left = {
        "$object": {
            "id": "object-0",
            "class": "example.Root",
            "attributes": {"alias": {"$object_ref": "object-0"}},
        }
    }
    right = deepcopy(left)
    right["$object"]["id"] = "object-91"
    right["$object"]["attributes"]["alias"]["$object_ref"] = "object-91"
    left_canonical, left_topology = _canonicalize_object_references(left)
    right_canonical, right_topology = _canonicalize_object_references(right)
    assert canonical_json_bytes(left) != canonical_json_bytes(right)
    assert left_canonical == right_canonical
    assert left_topology["raw_sha256"] != right_topology["raw_sha256"]
    assert left_topology["canonical_sha256"] == right_topology["canonical_sha256"]


def test_exact_post_call_parity_remains_mandatory(monkeypatch):
    payload, _receipt, diagnostic = _producer_diagnostic(
        "refplan", "refplan_fitted_policy", "general"
    )
    failed = deepcopy(diagnostic)
    failed["comparison"]["strict_post_call_state_equal"] = False
    failed["comparison"]["reference_normalized_post_call_state_equal"] = False
    failed["comparison"]["first_exact_difference"] = {
        "path": "root.$object.attributes.last_diagnostics",
        "reason": "value",
        "left": {"sha256": HASHES[0], "canonical_byte_count": 1, "value": 1},
        "right": {"sha256": HASHES[1], "canonical_byte_count": 1, "value": 2},
    }
    failed["comparison"]["first_substantive_difference"] = failed["comparison"][
        "first_exact_difference"
    ]
    failed["producer_time_original_vs_reloaded_equality_passed"] = False

    def fake_diagnostic(*_args, **_kwargs):
        return payload, failed

    monkeypatch.setattr(real_artifacts, "diagnose_frozen_fitted_object_roundtrip", fake_diagnostic)
    with pytest.raises(ContractError, match="post-call state/RNG parity failed"):
        validate_frozen_fitted_object_roundtrip_with_diagnostic(
            "refplan_fitted_policy",
            object(),
            operation="act",
            args=(),
            repository_root=REPOSITORY,
        )


@pytest.mark.parametrize(
    ("method", "component"),
    [
        ("plus_adapted_ricker_only_pbvi", "plus_fitted_policy"),
        ("moor_adapted_ricker_misspec_pbvi", "moor_fitted_policy"),
    ],
)
def test_ecological_strict_parity_receipts_remain_unchanged(method, component):
    payload, receipt_payload = frozen_object_evidence(method, CELL, HASHES[0])
    receipt = strict_json_loads(receipt_payload)
    validate_frozen_object_parity_receipt(
        receipt,
        expected_component=component,
        expected_serialized_sha256=sha256_bytes(payload),
    )
    assert receipt["schema_version"] == "corrected_stageb_frozen_object_parity_v1"
    assert receipt["result"] == "PASS"
    revalidate_frozen_object_parity(
        payload,
        receipt,
        expected_component=component,
        repository_root=REPOSITORY,
    )


def test_diagnostic_is_bounded_and_contains_no_scientific_evaluator_evidence():
    _payload, _receipt, diagnostic = _producer_diagnostic(
        "refplan", "refplan_fitted_policy", "general"
    )
    assert diagnostic["information_boundary"] == {
        "evaluator_constructed": False,
        "truth_accessed": False,
        "runtime_next_states_accessed": False,
        "returns_calculated": False,
    }
    encoded = canonical_json_bytes(diagnostic)
    assert b"truth.npz" not in encoded
    assert b'"next_states"' not in encoded
    assert b'"return"' not in encoded
    assert len(encoded) < 20_000_000


def test_graph_diagnostic_exposes_rng_and_reference_topology():
    graph = {
        "$object": {
            "id": "object-0",
            "class": "example.Root",
            "attributes": {
                "rng": {
                    "$numpy_generator": {
                        "bit_generator": "PCG64",
                        "state": {"counter": 1},
                    }
                },
                "self": {"$object_ref": "object-0"},
            },
        }
    }
    evidence = _graph_diagnostic(graph)
    assert evidence["rng_state_sha256_by_path"]
    assert evidence["object_reference_topology"]["canonical"]["references"] == [
        {
            "reference_path": "root/$object/attributes/self",
            "target_definition_path": "root",
        }
    ]


def test_fit_probe_atomically_checksum_covers_separate_diagnostic(tmp_path):
    payload, receipt_payload = frozen_object_evidence("refplan", CELL, HASHES[0])
    receipt = strict_json_loads(receipt_payload)
    with _registered_track_imports("general", REPOSITORY):
        fitted = load_frozen_fitted_object(payload, repository_root=REPOSITORY)
        args = _decode(
            receipt["operation_input"]["args"],
            track="general",
            repository_root=REPOSITORY,
        )
        kwargs = _decode(
            receipt["operation_input"]["kwargs"],
            track="general",
            repository_root=REPOSITORY,
        )
        target = publish_fit_probe(
            task_index=2,
            policy=fitted,
            public_view_sha256=HASHES[0],
            controlling_commit="a" * 40,
            repository_root=REPOSITORY,
            output_root=tmp_path,
            operation_args=args,
            operation_kwargs=kwargs,
        )
    success = load_success_receipt(target / "PUBLICATION_SUCCESS.json")
    assert "FROZEN_REPLAY_DIAGNOSTIC.json" in success["files"]
    diagnostic_payload = (target / "FROZEN_REPLAY_DIAGNOSTIC.json").read_bytes()
    assert sha256_bytes(diagnostic_payload) == success["files"]["FROZEN_REPLAY_DIAGNOSTIC.json"]
    assert strict_json_loads(diagnostic_payload)["schema_version"] == (
        "corrected_stageb_frozen_replay_diagnostic_v1"
    )
