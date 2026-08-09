"""Synthetic-only tests for the bounded I2 Increment A contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np

from adapter_interfaces import (
    ALIAS_MAP,
    ARTIFACT_DIGEST_FIELDS,
    CONSTANT_THRESHOLD,
    END_TO_END_LABEL,
    FEATURE_NAMES,
    ContractViolation,
    ContextSnapshot,
    ExactStateFeatureAdapter,
    LearnedArtifactSnapshot,
    RecordingRNG,
    adapt_point_mass_belief,
    build_preprocessing_receipt,
    build_refplan_disclosure,
    classify_interpretation,
    capture_context_snapshot,
    fit_end_to_end_preprocessor,
    passes_constant_validation_guard,
    validate_constructed_features,
    validate_context_preservation,
    validate_preprocessor_artifact,
    validate_serialized_preprocessor_parity,
)
from information_boundary import (
    CAPABILITY_FUTURE_OFFLINE_NEXT_STATES,
    CAPABILITY_FUTURE_OFFLINE_STATES,
    CAPABILITY_METADATA_BINDING,
    FORBIDDEN_FIELDS,
    REGISTERED_KEY_ORDER,
    AccessController,
    BoundaryViolation,
    MetadataOnlyNpzInspector,
    SyntheticProvenance,
    validate_synthetic_archive,
)
from synthetic_fixtures import (
    SYNTHETIC_BINDING,
    SYNTHETIC_PROVENANCE,
    make_base_public_features,
    make_exact_abundance,
    make_synthetic_truth_archive,
    reduction_based_sd_residue,
    synthetic_row_tokens,
    synthetic_units,
    write_synthetic_npz,
)


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]


def external_test_root() -> Path:
    raw = os.environ.get("I2A_TEST_TMPDIR")
    if not raw:
        raise RuntimeError("I2A_TEST_TMPDIR must name a fresh external synthetic directory")
    root = Path(raw).resolve()
    if not root.is_dir() or root == REPOSITORY_ROOT or REPOSITORY_ROOT in root.parents:
        raise RuntimeError("I2A_TEST_TMPDIR must exist outside the repository")
    return root


def make_adapted(rows: int = 16):
    base = make_base_public_features(rows)
    actions = tuple(range(rows))
    observations = tuple(float(value) for value in base[:, 7])
    adapted, receipt = ExactStateFeatureAdapter().adapt(
        base,
        make_exact_abundance(rows),
        observation_scale=20.0,
        observation_noise_sigma=0.2,
        action_history=actions,
        observation_history=observations,
    )
    return base, adapted, receipt, actions, observations


def artifact_snapshot(token: str = "same") -> LearnedArtifactSnapshot:
    def digest(field_name: str) -> str:
        return hashlib.sha256(f"synthetic:{token}:{field_name}".encode("utf-8")).hexdigest()

    return LearnedArtifactSnapshot(
        preprocessor_sha256=digest("preprocessor_sha256"),
        feature_fit_sha256=digest("feature_fit_sha256"),
        dynamics_fit_sha256=digest("dynamics_fit_sha256"),
        residual_sigma_hex=(np.float64(0.2).hex(),),
        surrogate_sha256=digest("surrogate_sha256"),
        reward_model_sha256=digest("reward_model_sha256"),
        safety_calibration_sha256=digest("safety_calibration_sha256"),
        ensemble_sha256=digest("ensemble_sha256"),
        policy_sha256=digest("policy_sha256"),
    )


def eligibility_kwargs(method: str) -> dict[str, bool]:
    if method == "refplan":
        return {"refplan_frozen_fit_secondary": True}
    return {"direct_point_mass_replacement": True}


class ExactFeatureTests(unittest.TestCase):
    def test_01_assigned_sd_is_bit_exact_positive_zero(self):
        _, adapted, receipt, _, _ = make_adapted()
        self.assertTrue(np.all(adapted[:, 1].view(np.uint64) == 0))
        self.assertEqual(receipt.maximum_absolute_assigned_sd, 0.0)

    def test_02_reduction_can_leave_nonzero_residue(self):
        residue = reduction_based_sd_residue()
        self.assertGreater(residue, 0.0)
        self.assertLessEqual(residue, np.float64(8.882e-16))
        self.assertFalse(bool(residue == 0.0))

    def test_03_quantiles_are_bit_exact_mean_aliases(self):
        _, adapted, _, _, _ = make_adapted()
        for target, source in ALIAS_MAP.items():
            self.assertTrue(
                np.array_equal(adapted[:, target].view(np.uint64), adapted[:, source].view(np.uint64))
            )

    def test_04_state_block_rank_is_one(self):
        _, adapted, receipt, _, _ = make_adapted()
        self.assertEqual(np.linalg.matrix_rank(adapted[:, :5]), 1)
        self.assertEqual(receipt.state_block_rank, 1)

    def test_05_feature_order_and_dtype_preserved(self):
        base, adapted, receipt, _, _ = make_adapted()
        self.assertEqual(base.shape, adapted.shape)
        self.assertEqual(adapted.dtype, np.dtype("float64"))
        self.assertEqual(receipt.feature_names, FEATURE_NAMES)
        reordered = list(FEATURE_NAMES)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaises(ContractViolation):
            ExactStateFeatureAdapter().adapt(
                base,
                make_exact_abundance(),
                feature_names=reordered,
                observation_scale=20.0,
                observation_noise_sigma=0.2,
                action_history=range(16),
                observation_history=base[:, 7],
            )

    def test_06_existing_constant_columns_remain_present(self):
        base, adapted, _, _, _ = make_adapted()
        self.assertEqual(adapted.shape[1], 10)
        self.assertTrue(np.array_equal(base[:, 9], adapted[:, 9]))
        self.assertTrue(np.all(adapted[:, 5] == 0.0))

    def test_07_history_and_context_are_unchanged(self):
        base, adapted, receipt, actions, observations = make_adapted()
        before = capture_context_snapshot(
            base,
            feature_names=FEATURE_NAMES,
            action_history=actions,
            observation_history=observations,
        )
        after = capture_context_snapshot(
            adapted,
            feature_names=FEATURE_NAMES,
            action_history=tuple(actions),
            observation_history=tuple(observations),
        )
        validate_context_preservation(before, after)
        self.assertEqual(receipt.context_parity, "PASS")
        self.assertEqual(receipt.preserved_context_sha256_before, before.sha256)
        self.assertEqual(receipt.preserved_context_sha256_after, after.sha256)

    def test_08_wholesale_history_zeroing_is_rejected(self):
        base, adapted, _, actions, observations = make_adapted()
        bad = adapted.copy()
        bad[:, 6:10] = 0.0
        before = capture_context_snapshot(
            base,
            feature_names=FEATURE_NAMES,
            action_history=actions,
            observation_history=observations,
        )
        after = capture_context_snapshot(
            bad,
            feature_names=FEATURE_NAMES,
            action_history=actions,
            observation_history=observations,
        )
        with self.assertRaises(ContractViolation):
            validate_context_preservation(before, after)

    def test_08b_independent_context_mutations_all_fail(self):
        base, adapted, _, actions, observations = make_adapted()
        before = capture_context_snapshot(
            base,
            feature_names=FEATURE_NAMES,
            action_history=actions,
            observation_history=observations,
        )
        after = capture_context_snapshot(
            adapted,
            feature_names=FEATURE_NAMES,
            action_history=actions,
            observation_history=observations,
        )
        mutations: dict[str, ContextSnapshot] = {
            "observation_history": replace(
                after,
                observation_history_float64_hex=after.observation_history_float64_hex[:-1]
                + (np.float64(999.0).hex(),),
            ),
            "action_history": replace(
                after, action_history=after.action_history[:-1] + (999,)
            ),
            "ordering": replace(
                after, feature_names=(FEATURE_NAMES[1], FEATURE_NAMES[0]) + FEATURE_NAMES[2:]
            ),
            "dtype": replace(after, feature_dtype="float32"),
            "shape": replace(after, feature_shape=(after.feature_shape[0], 9)),
        }
        for context_index in (1, 2, 3):
            rows = [list(row) for row in after.protected_context_float64_hex]
            rows[0][context_index] = np.float64(999.0).hex()
            mutations[f"feature_{context_index + 6}"] = replace(
                after, protected_context_float64_hex=tuple(tuple(row) for row in rows)
            )
        for name, mutation in mutations.items():
            with self.subTest(name=name), self.assertRaises(ContractViolation):
                validate_context_preservation(before, mutation)

    def test_09_zero_sigma_or_scale_route_is_rejected(self):
        base = make_base_public_features()
        common = dict(
            base_features=base,
            exact_abundance=make_exact_abundance(),
            action_history=range(16),
            observation_history=base[:, 7],
        )
        for scale, sigma in ((0.0, 0.2), (1e-15, 0.2), (20.0, 0.0), (20.0, 1e-15)):
            with self.subTest(scale=scale, sigma=sigma), self.assertRaises(ContractViolation):
                ExactStateFeatureAdapter().adapt(
                    **common,
                    observation_scale=scale,
                    observation_noise_sigma=sigma,
                )

    def test_10_jitter_and_epsilon_sd_are_rejected(self):
        _, adapted, _, _, _ = make_adapted()
        for residue in (1e-16, 1e-9, 1e-6):
            bad = adapted.copy()
            bad[:, 1] = np.float64(residue)
            with self.subTest(residue=residue), self.assertRaises(ContractViolation):
                validate_constructed_features(bad)


class PreprocessingTests(unittest.TestCase):
    def test_11_mask_offsets_scale_and_alias_map(self):
        _, adapted, _, _, _ = make_adapted()
        artifact = fit_end_to_end_preprocessor(adapted)
        self.assertTrue(artifact.constant_mask[1])
        self.assertTrue(artifact.constant_mask[5])
        self.assertTrue(artifact.constant_mask[9])
        self.assertEqual(artifact.scales[1], 1.0)
        self.assertEqual(artifact.alias_map, tuple(sorted(ALIAS_MAP.items())))

    def test_12_guard_accepts_assignment_and_rejects_noncompliance(self):
        _, adapted, _, _, _ = make_adapted()
        validate_constructed_features(adapted)
        self.assertLessEqual(np.std(adapted[:, 1]), CONSTANT_THRESHOLD)
        self.assertTrue(passes_constant_validation_guard(1e-8))
        self.assertFalse(passes_constant_validation_guard(np.nextafter(1e-8, np.inf)))
        bad = adapted.copy()
        bad[:, 1] = np.linspace(0.0, 2e-8, len(bad), dtype=np.float64)
        with self.assertRaises(ContractViolation):
            validate_constructed_features(bad)

    def test_13_offline_runtime_transform_parity(self):
        _, adapted, _, _, _ = make_adapted()
        artifact = fit_end_to_end_preprocessor(adapted)
        serialized = artifact.to_bytes()
        reloaded, checks = validate_serialized_preprocessor_parity(
            artifact, serialized, adapted, expected_state_block_rank=1
        )
        first = artifact.transform(adapted)
        second = reloaded.transform(adapted.copy())
        self.assertIsNot(artifact, reloaded)
        self.assertTrue(np.array_equal(first.view(np.uint64), second.view(np.uint64)))
        self.assertTrue(np.all(first[:, 1].view(np.uint64) == 0))
        self.assertTrue(all(checks.values()))
        self.assertEqual(artifact.sha256, reloaded.sha256)

    def test_13b_incomplete_or_changed_serialization_fails_parity(self):
        _, adapted, _, _, _ = make_adapted()
        artifact = fit_end_to_end_preprocessor(adapted)
        original = json.loads(artifact.to_bytes().decode("utf-8"))
        variants = []
        incomplete = dict(original); incomplete.pop("alias_map"); variants.append(incomplete)
        changed = copy.deepcopy(original)
        changed["offsets_float64_hex"][0] = np.float64(999.0).hex()
        variants.append(changed)
        malformed = copy.deepcopy(original)
        malformed["constant_mask"] = malformed["constant_mask"][:-1]
        variants.append(malformed)
        for payload in variants:
            serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            with self.assertRaises(ContractViolation):
                validate_serialized_preprocessor_parity(
                    artifact, serialized, adapted, expected_state_block_rank=1
                )

    def test_14_runtime_constant_tolerance_fails_closed(self):
        _, adapted, _, _, _ = make_adapted()
        artifact = fit_end_to_end_preprocessor(adapted)
        bad = adapted.copy()
        bad[:, 1] = np.float64(1.0001e-6)
        with self.assertRaises(ContractViolation):
            artifact.transform(bad)

    def test_15_artifact_mask_alias_scale_and_hash_mismatches_fail(self):
        _, adapted, _, _, _ = make_adapted()
        artifact = fit_end_to_end_preprocessor(adapted)
        mutations = [
            replace(artifact, constant_mask=(False,) * 10),
            replace(artifact, alias_map=((2, 1), (3, 0), (4, 0))),
            replace(artifact, scales=tuple(1e-8 if i == 1 else v for i, v in enumerate(artifact.scales))),
            replace(artifact, feature_names=tuple(reversed(FEATURE_NAMES))),
        ]
        for bad in mutations:
            with self.subTest(bad=bad), self.assertRaises(ContractViolation):
                validate_preprocessor_artifact(bad)
        with self.assertRaises(ContractViolation):
            validate_preprocessor_artifact(artifact, expected_sha256="0" * 64)

    def test_16_unexpected_unmasked_runtime_degeneracy_fails(self):
        _, adapted, _, _, _ = make_adapted()
        artifact = fit_end_to_end_preprocessor(adapted)
        bad = adapted.copy()
        bad[:, 6] = bad[0, 6]
        with self.assertRaises(ContractViolation):
            artifact.transform(bad)


class InformationBoundaryTests(unittest.TestCase):
    def valid_archive(self):
        return make_synthetic_truth_archive()

    def validate(self, archive=None, units=None, tokens=None, binding=SYNTHETIC_BINDING, provenance=None):
        return validate_synthetic_archive(
            archive or self.valid_archive(),
            provenance=provenance or SYNTHETIC_PROVENANCE,
            declared_units=units or synthetic_units(),
            row_alignment_tokens=tokens or synthetic_row_tokens(),
            expected_public_dataset_sha256=binding,
        )

    def test_17_valid_synthetic_schema_passes(self):
        result = self.validate()
        self.assertEqual(result["schema_result"], "PASS")
        self.assertEqual(result["real_scientific_arrays_materialized"], 0)

    def test_18_schema_variants_fail_closed(self):
        variants = {}
        missing = self.valid_archive(); missing.pop("theta"); variants["missing"] = (missing, None, None, SYNTHETIC_BINDING)
        extra = self.valid_archive(); extra["extra"] = np.zeros(4000); variants["extra"] = (extra, None, None, SYNTHETIC_BINDING)
        reordered = self.valid_archive(); reordered.move_to_end("C"); variants["reordered"] = (reordered, None, None, SYNTHETIC_BINDING)
        wrong_dtype = self.valid_archive(); wrong_dtype["states"] = wrong_dtype["states"].astype(np.float32); variants["dtype"] = (wrong_dtype, None, None, SYNTHETIC_BINDING)
        wrong_shape = self.valid_archive(); wrong_shape["states"] = wrong_shape["states"][:-1]; variants["shape"] = (wrong_shape, None, None, SYNTHETIC_BINDING)
        nonfinite = self.valid_archive(); nonfinite["states"][0] = np.nan; variants["nonfinite"] = (nonfinite, None, None, SYNTHETIC_BINDING)
        wrong_units = synthetic_units(); wrong_units["states"] = "latent"; variants["units"] = (self.valid_archive(), wrong_units, None, SYNTHETIC_BINDING)
        misaligned = synthetic_row_tokens(); misaligned["states"] = "different"; variants["alignment"] = (self.valid_archive(), None, misaligned, SYNTHETIC_BINDING)
        variants["binding"] = (self.valid_archive(), None, None, "wrong-binding")
        for name, (archive, units, tokens, binding) in variants.items():
            with self.subTest(name=name), self.assertRaises(BoundaryViolation):
                self.validate(archive=archive, units=units, tokens=tokens, binding=binding)

    def test_19_real_derived_fixture_provenance_is_rejected(self):
        with self.assertRaises(BoundaryViolation):
            self.validate(
                provenance=SyntheticProvenance(
                    generator="bad", derived_from_real_arrays=True, source_paths=("truth.npz",)
                )
            )

    def test_20_runtime_next_states_is_rejected(self):
        controller = AccessController()
        with self.assertRaises(BoundaryViolation):
            controller.request(
                "next_states", "runtime", CAPABILITY_FUTURE_OFFLINE_NEXT_STATES
            )
        self.assertFalse(controller.receipt.requests[-1]["allowed"])

    def test_21_every_forbidden_field_is_rejected_by_name(self):
        controller = AccessController()
        for field_name in sorted(FORBIDDEN_FIELDS):
            with self.subTest(field=field_name), self.assertRaises(BoundaryViolation):
                controller.request(field_name, "any", CAPABILITY_METADATA_BINDING)

    def test_22_future_capabilities_are_explicit_but_i2a_denies_arrays(self):
        capabilities = AccessController.future_capabilities()
        self.assertIn(CAPABILITY_FUTURE_OFFLINE_STATES, capabilities)
        self.assertIn(CAPABILITY_FUTURE_OFFLINE_NEXT_STATES, capabilities)
        self.assertTrue(capabilities[CAPABILITY_FUTURE_OFFLINE_NEXT_STATES]["offline_fitting_only"])
        controller = AccessController()
        for field_name, capability in (
            ("states", CAPABILITY_FUTURE_OFFLINE_STATES),
            ("next_states", CAPABILITY_FUTURE_OFFLINE_NEXT_STATES),
        ):
            with self.assertRaises(BoundaryViolation):
                controller.request(field_name, "offline_fitting", capability)

    def test_23_metadata_only_inspector_loads_no_scientific_payload(self):
        with tempfile.TemporaryDirectory(dir=external_test_root()) as temp_dir:
            path = Path(temp_dir) / "synthetic_truth.npz"
            write_synthetic_npz(path)
            result = MetadataOnlyNpzInspector().inspect(
                path,
                expected_public_dataset_sha256=SYNTHETIC_BINDING,
                declared_units=synthetic_units(),
            )
        receipt = result["access_receipt"]
        self.assertEqual(receipt["payload_loaded_fields"], ["metadata_json"])
        self.assertEqual(receipt["scientific_payload_loaded_fields"], [])
        self.assertEqual(receipt["scientific_payload_read_bytes"], 0)
        self.assertEqual(receipt["real_scientific_arrays_materialized"], 0)
        self.assertEqual(receipt["boundary_result"], "PASS")


class MethodAndInterpretationTests(unittest.TestCase):
    def test_24_plus_moor_point_mass_and_survey_scale(self):
        grid = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        raw_grid = np.asarray([0.0, 20.0, 40.0, 60.0], dtype=np.float64)
        for method in ("plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"):
            belief, receipt = adapt_point_mass_belief(method, 45.0, 20.0, grid, raw_grid)
            self.assertEqual(receipt.latent_abundance, 2.25)
            self.assertEqual(receipt.selected_bin_index, 2)
            self.assertEqual(receipt.mapped_raw_abundance, 40.0)
            self.assertEqual(np.sum(belief), 1.0)
            self.assertEqual(np.count_nonzero(belief), 1)
        with self.assertRaises(ContractViolation):
            adapt_point_mass_belief(
                "plus_adapted_ricker_only_pbvi",
                45.0,
                20.0,
                grid,
                np.asarray([0.0, 20.0, 41.0, 60.0], dtype=np.float64),
            )

    def test_25_refplan_frozen_fit_requires_byte_hash_identity(self):
        baseline = artifact_snapshot()
        same = copy.deepcopy(baseline)
        disabled = classify_interpretation("refplan", baseline, same)
        self.assertFalse(disabled["frozen_fit_eligible"])
        eligible = classify_interpretation(
            "refplan", baseline, same, refplan_frozen_fit_secondary=True
        )
        self.assertTrue(eligible["frozen_fit_eligible"])
        changed = replace(
            same, preprocessor_sha256=hashlib.sha256(b"different").hexdigest()
        )
        rejected = classify_interpretation(
            "refplan", baseline, changed, refplan_frozen_fit_secondary=True
        )
        self.assertFalse(rejected["frozen_fit_eligible"])
        self.assertEqual(rejected["label"], END_TO_END_LABEL)

    def test_26_any_learned_artifact_difference_forces_bundle_label(self):
        baseline = artifact_snapshot()
        methods = (
            "refplan",
            "plus_adapted_ricker_only_pbvi",
            "moor_adapted_ricker_misspec_pbvi",
        )
        for method in methods:
            for field_name in baseline.__dataclass_fields__:
                replacement = (
                    (np.float64(0.3).hex(),)
                    if field_name == "residual_sigma_hex"
                    else hashlib.sha256(f"changed:{field_name}".encode("utf-8")).hexdigest()
                )
                changed = replace(baseline, **{field_name: replacement})
                with self.subTest(method=method, field=field_name):
                    result = classify_interpretation(
                        method, baseline, changed, **eligibility_kwargs(method)
                    )
                    self.assertEqual(result["identity_validation"], "PASS")
                    self.assertEqual(result["changed_artifacts"], [field_name])
                    self.assertEqual(result["label"], END_TO_END_LABEL)
                    self.assertFalse(result["causal_state_representation_label_permitted"])

    def test_26b_every_required_identity_fails_closed_when_malformed_or_omitted(self):
        baseline = artifact_snapshot()
        methods = (
            "refplan",
            "plus_adapted_ricker_only_pbvi",
            "moor_adapted_ricker_misspec_pbvi",
        )
        sha_invalid = (
            "",
            None,
            "unknown",
            "TBD",
            "NA",
            "N/A",
            "null",
            "   ",
            "a" * 63,
            "a" * 65,
            "g" * 64,
            "A" * 64,
            "sha256:" + "a" * 64,
        )
        residual_invalid = (
            "",
            None,
            (),
            ("unknown",),
            ("TBD",),
            ("NA",),
            ("N/A",),
            ("null",),
            (" ",),
            ("nan",),
            ("inf",),
            ("0x1.999999999999ap-3", ""),
            [np.float64(0.2).hex()],
        )
        for method in methods:
            for field_name in baseline.__dataclass_fields__:
                invalid_values = (
                    residual_invalid if field_name == "residual_sigma_hex" else sha_invalid
                )
                for bad_value in invalid_values:
                    both_bad_o = dict(baseline.__dict__)
                    both_bad_t = dict(baseline.__dict__)
                    both_bad_o[field_name] = bad_value
                    both_bad_t[field_name] = bad_value
                    with self.subTest(
                        method=method,
                        field=field_name,
                        bad_arm="both",
                        bad_value=repr(bad_value),
                    ):
                        result = classify_interpretation(
                            method,
                            both_bad_o,
                            both_bad_t,
                            **eligibility_kwargs(method),
                        )
                        self.assertFalse(result["frozen_fit_eligible"])
                        self.assertEqual(result["identity_validation"], "FAIL")
                        self.assertIn(field_name, result["eligibility_reason"])
                        self.assertEqual(result["label"], END_TO_END_LABEL)
                    for bad_arm in ("Arm O", "Arm T"):
                        arm_o = dict(baseline.__dict__)
                        arm_t = dict(baseline.__dict__)
                        target = arm_o if bad_arm == "Arm O" else arm_t
                        target[field_name] = bad_value
                        with self.subTest(
                            method=method,
                            field=field_name,
                            bad_arm=bad_arm,
                            bad_value=repr(bad_value),
                        ):
                            result = classify_interpretation(
                                method, arm_o, arm_t, **eligibility_kwargs(method)
                            )
                            self.assertFalse(result["frozen_fit_eligible"])
                            self.assertEqual(result["identity_validation"], "FAIL")
                            self.assertIn(field_name, result["eligibility_reason"])
                            self.assertEqual(result["label"], END_TO_END_LABEL)
                for missing_arm in ("Arm O", "Arm T"):
                    arm_o = dict(baseline.__dict__)
                    arm_t = dict(baseline.__dict__)
                    target = arm_o if missing_arm == "Arm O" else arm_t
                    target.pop(field_name)
                    with self.subTest(
                        method=method, field=field_name, missing_arm=missing_arm
                    ):
                        result = classify_interpretation(
                            method, arm_o, arm_t, **eligibility_kwargs(method)
                        )
                        self.assertFalse(result["frozen_fit_eligible"])
                        self.assertEqual(result["identity_validation"], "FAIL")
                        self.assertIn(field_name, result["eligibility_reason"])
                        self.assertEqual(result["label"], END_TO_END_LABEL)

    def test_26c_equal_valid_identities_still_require_method_condition(self):
        baseline = artifact_snapshot()
        for method in (
            "refplan",
            "plus_adapted_ricker_only_pbvi",
            "moor_adapted_ricker_misspec_pbvi",
        ):
            with self.subTest(method=method):
                rejected = classify_interpretation(method, baseline, copy.deepcopy(baseline))
                self.assertEqual(rejected["identity_validation"], "PASS")
                self.assertFalse(rejected["frozen_fit_eligible"])
                accepted = classify_interpretation(
                    method,
                    baseline,
                    copy.deepcopy(baseline),
                    **eligibility_kwargs(method),
                )
                self.assertTrue(accepted["frozen_fit_eligible"])
                self.assertEqual(accepted["label"], "FROZEN-FIT/STATE-INPUT-ONLY")

    def test_27_plus_moor_frozen_fit_requires_direct_replacement(self):
        baseline = artifact_snapshot()
        for method in ("plus_adapted_ricker_only_pbvi", "moor_adapted_ricker_misspec_pbvi"):
            self.assertFalse(
                classify_interpretation(method, baseline, baseline)["frozen_fit_eligible"]
            )
            self.assertTrue(
                classify_interpretation(
                    method, baseline, baseline, direct_point_mass_replacement=True
                )["frozen_fit_eligible"]
            )

    def test_28_refplan_disclosure_is_mandatory_and_complete(self):
        complete = {
            "fit_time_sd_mean": 0.1,
            "fit_time_sd_std": 0.03,
            "planner_root_sd": 0.0,
            "arm_o_standardized_offset": -2.7018,
            "arm_t_standardized_offset": 0.0,
            "species": "amur_tiger",
            "arm_t_removes_preexisting_mismatch": True,
        }
        receipt = build_refplan_disclosure(complete)
        self.assertTrue(receipt["disclosure_complete"])
        self.assertEqual(receipt["strict_json_roundtrip"], "PASS")
        self.assertEqual(receipt["tiger_reference_offset"], -2.7018)
        self.assertEqual(receipt["fox_reference_offset"], -2.7348)
        json.dumps(receipt, allow_nan=False, sort_keys=True)
        for field_name in complete:
            incomplete = dict(complete)
            incomplete.pop(field_name)
            with self.subTest(missing=field_name), self.assertRaises(ContractViolation):
                build_refplan_disclosure(incomplete)

    def test_28b_refplan_disclosure_rejects_nonfinite_and_inconsistent_values(self):
        complete = {
            "fit_time_sd_mean": 0.1,
            "fit_time_sd_std": 0.03,
            "planner_root_sd": 0.0,
            "arm_o_standardized_offset": -2.7018,
            "arm_t_standardized_offset": 0.0,
            "species": "amur_tiger",
            "arm_t_removes_preexisting_mismatch": True,
        }
        numeric_fields = (
            "fit_time_sd_mean",
            "fit_time_sd_std",
            "planner_root_sd",
            "arm_o_standardized_offset",
            "arm_t_standardized_offset",
        )
        for field_name in numeric_fields:
            for bad_value in (np.nan, np.inf, -np.inf, "0.0", None):
                bad = dict(complete)
                bad[field_name] = bad_value
                with self.subTest(field=field_name, value=repr(bad_value)), self.assertRaises(
                    ContractViolation
                ):
                    build_refplan_disclosure(bad)
        invalid_variants = {
            "unknown_species": {**complete, "species": "unknown"},
            "arm_o_species_mismatch": {
                **complete,
                "arm_o_standardized_offset": -2.7348,
            },
            "negative_fit_mean": {**complete, "fit_time_sd_mean": -0.1},
            "negative_fit_std": {**complete, "fit_time_sd_std": -0.1},
            "negative_zero_root": {**complete, "planner_root_sd": -0.0},
            "nonzero_root": {**complete, "planner_root_sd": 1e-12},
            "non_boolean_flag": {
                **complete,
                "arm_t_removes_preexisting_mismatch": 1,
            },
            "removed_but_nonzero": {**complete, "arm_t_standardized_offset": -2.7018},
            "retained_but_different": {
                **complete,
                "arm_t_removes_preexisting_mismatch": False,
                "arm_t_standardized_offset": 0.0,
            },
            "extra_field": {**complete, "unregistered": 1},
        }
        for name, bad in invalid_variants.items():
            with self.subTest(name=name), self.assertRaises(ContractViolation):
                build_refplan_disclosure(bad)

    def test_28c_refplan_fox_reference_and_retained_mismatch_are_exact(self):
        fox = build_refplan_disclosure(
            {
                "fit_time_sd_mean": 0.1,
                "fit_time_sd_std": 0.03,
                "planner_root_sd": 0.0,
                "arm_o_standardized_offset": -2.7348,
                "arm_t_standardized_offset": -2.7348,
                "species": "crab_eating_fox",
                "arm_t_removes_preexisting_mismatch": False,
            }
        )
        self.assertEqual(fox["registered_species_reference_offset"], -2.7348)
        self.assertEqual(fox["arm_t_standardized_offset"], -2.7348)

    def test_29_rng_exact_value_and_call_state_parity(self):
        zero_rng = RecordingRNG(20260808)
        noisy_rng = RecordingRNG(20260808)
        before = json.dumps(zero_rng.state, sort_keys=True)
        locs = [
            np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            np.asarray([4.0, 5.0], dtype=np.float64),
        ]
        for loc in locs:
            zero_draw = zero_rng.normal(loc, 0.0)
            noisy_rng.normal(loc, 0.2)
            self.assertTrue(np.array_equal(zero_draw.view(np.uint64), loc.view(np.uint64)))
        after = json.dumps(zero_rng.state, sort_keys=True)
        self.assertNotEqual(before, after)
        self.assertEqual(zero_rng.calls, noisy_rng.calls)
        self.assertEqual(zero_rng.state, noisy_rng.state)
        zero_next = zero_rng.normal(np.asarray([6.0], dtype=np.float64), 0.2)
        noisy_next = noisy_rng.normal(np.asarray([6.0], dtype=np.float64), 0.2)
        self.assertTrue(np.array_equal(zero_next.view(np.uint64), noisy_next.view(np.uint64)))

    def test_30_all_six_method_contracts_are_complete_and_explicit(self):
        contracts = json.loads((HERE / "METHOD_CONTRACTS.json").read_text(encoding="utf-8"))
        expected = {
            "plus_adapted_ricker_only_pbvi",
            "moor_adapted_ricker_misspec_pbvi",
            "refplan",
            "ogsrl",
            "bamcts",
            "ensemble_value_disagreement_pessimism",
        }
        self.assertEqual(set(contracts["contracts"]), expected)
        required = {
            "state_interface",
            "preserved_context",
            "direct_point_mass_belief",
            "survey_scale_conversion",
            "preprocessing",
            "possible_later_changes",
            "frozen_fit_eligibility",
            "end_to_end_only_conditions",
            "forbidden_shortcuts",
        }
        for method, contract in contracts["contracts"].items():
            with self.subTest(method=method):
                self.assertTrue(required.issubset(contract))
                self.assertTrue(contract["forbidden_shortcuts"])

    def test_31_receipt_schema_contains_registered_requirements(self):
        schema = json.loads((HERE / "PREPROCESSING_RECEIPT_SCHEMA.json").read_text(encoding="utf-8"))
        required = set(schema["required"])
        for field in (
            "feature_names",
            "dtype",
            "constant_mask",
            "alias_rank_collapse_map",
            "maximum_absolute_assigned_sd",
            "refplan_standardized_sd_offsets",
            "rng_call_count_parity",
            "rng_state_advancement_parity",
            "serialization_parity",
            "parity_result",
        ):
            self.assertIn(field, required)

    def test_32_full_synthetic_preprocessing_receipt_is_emitted(self):
        _, adapted, adaptation_receipt, _, _ = make_adapted()
        artifact = fit_end_to_end_preprocessor(adapted)
        disclosure = build_refplan_disclosure(
            {
                "fit_time_sd_mean": 0.1,
                "fit_time_sd_std": 0.03,
                "planner_root_sd": 0.0,
                "arm_o_standardized_offset": -2.7018,
                "arm_t_standardized_offset": 0.0,
                "species": "amur_tiger",
                "arm_t_removes_preexisting_mismatch": True,
            }
        )
        receipt = build_preprocessing_receipt(
            method="refplan",
            species="synthetic_amur_tiger",
            cell="synthetic_amur_tiger__allee__sigma_0p2",
            features=adapted,
            artifact=artifact,
            adaptation_receipt=adaptation_receipt,
            arm_o_artifact_sha256="a" * 64,
            refplan_disclosure=disclosure,
            rng_call_count_parity=True,
            rng_state_advancement_parity=True,
        )
        schema = json.loads((HERE / "PREPROCESSING_RECEIPT_SCHEMA.json").read_text(encoding="utf-8"))
        self.assertEqual(set(receipt), set(schema["required"]))
        self.assertEqual(receipt["maximum_absolute_assigned_sd"], 0.0)
        self.assertEqual(receipt["alias_rank_collapse_map"]["arm_t_state_block_rank"], 1)
        self.assertEqual(receipt["parity_result"], "PASS")


if __name__ == "__main__":
    unittest.main()
