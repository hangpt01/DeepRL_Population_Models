from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fasttrack_wrappers import (
    END_TO_END_LABEL, METHODS, Arm, ExternalMethodWrapper, IntegrationViolation,
    assign_ecological_internal_beliefs, direct_point_mass_latent_belief,
    interpretation_label, issue_exact_state_capability, validate_artifact_axis,
)


@dataclass
class FakeBelief:
    states: np.ndarray
    log_weights: np.ndarray
    regimes: np.ndarray
    contexts: np.ndarray
    observation: float
    diagnostics: dict[str, float]


def belief() -> FakeBelief:
    return FakeBelief(
        states=np.array([10., 11., 12.], dtype=np.float64),
        log_weights=np.log(np.array([.2, .3, .5], dtype=np.float64)),
        regimes=np.array([0, 0, 0], dtype=np.int8),
        contexts=np.array([[8., 9., 1.], [8., 9., 1.], [8., 9., 1.]], dtype=np.float64),
        observation=9.0,
        diagnostics={"public_observation_filter": 1.0},
    )


class FastTrackTests(unittest.TestCase):
    def setUp(self):
        self.cap = issue_exact_state_capability(purpose="i2b_synthetic_or_future_authorized_runtime")

    def test_01_all_six_arm_o_wrappers_preserve_identity_and_context(self):
        for method in METHODS:
            base = belief()
            emitted, receipt = ExternalMethodWrapper(method).route_general(base)
            self.assertIs(emitted, base)
            self.assertTrue(receipt["input_identity_preserved"])

    def test_02_arm_t_disabled_without_enable_flag(self):
        with self.assertRaises(IntegrationViolation):
            ExternalMethodWrapper("refplan", arm=Arm.T)

    def test_03_arm_t_disabled_without_capability(self):
        wrapper = ExternalMethodWrapper("refplan", arm=Arm.T, exact_state_enabled=True)
        with self.assertRaises(IntegrationViolation):
            wrapper.route_general(belief(), exact_current_abundance=13.)

    def test_04_sigma_and_scale_collapse_rejected(self):
        wrapper = ExternalMethodWrapper("refplan", arm=Arm.T, exact_state_enabled=True)
        for sigma, scale in ((0., 1.), (1e-15, 1.), (.2, 0.), (.2, 1e-15)):
            with self.subTest(sigma=sigma, scale=scale), self.assertRaises(IntegrationViolation):
                wrapper.route_general(belief(), exact_current_abundance=13., capability=self.cap,
                                      observation_noise_sigma=sigma, observation_scale=scale)

    def test_05_runtime_next_state_and_forbidden_fields_rejected(self):
        wrapper = ExternalMethodWrapper("bamcts", arm=Arm.T, exact_state_enabled=True)
        for field in ("next_states", "family", "reward_true", "theta", "evaluator_info"):
            with self.subTest(field=field), self.assertRaises(IntegrationViolation):
                wrapper.route_general(belief(), exact_current_abundance=13., capability=self.cap,
                                      private_payload={field: 1})

    def test_06_context_history_dtype_and_observation_preserved(self):
        base = belief(); before = base.contexts.copy()
        emitted, receipt = ExternalMethodWrapper(
            "refplan", arm=Arm.T, exact_state_enabled=True
        ).route_general(base, exact_current_abundance=13., capability=self.cap)
        self.assertTrue(np.array_equal(emitted.contexts, before))
        self.assertEqual(emitted.contexts.dtype, np.dtype("float64"))
        self.assertEqual(emitted.observation, base.observation)
        self.assertEqual(emitted.diagnostics, base.diagnostics)
        self.assertEqual(receipt["runtime_next_states"], "IMPOSSIBLE_NO_ARGUMENT")

    def test_07_exact_state_particles_and_weights(self):
        emitted, _ = ExternalMethodWrapper(
            "ogsrl", arm=Arm.T, exact_state_enabled=True
        ).route_general(belief(), exact_current_abundance=13., capability=self.cap)
        self.assertTrue(np.array_equal(emitted.states, np.full(3, 13., dtype=np.float64)))
        self.assertAlmostEqual(float(np.exp(emitted.log_weights).sum()), 1.)

    def test_08_plus_moor_conversion_uses_survey_scale(self):
        grid = np.array([0., 1., 2., 3.], dtype=np.float64)
        b, receipt = direct_point_mass_latent_belief(20., survey_scale=10., latent_grid=grid)
        self.assertEqual(int(np.argmax(b)), 2)
        self.assertEqual(receipt["selected_raw_float64_hex"], np.float64(20.).hex())

    def test_09_raw_state_is_not_snapped_directly(self):
        grid = np.array([0., 1., 2., 3.], dtype=np.float64)
        b, _ = direct_point_mass_latent_belief(20., survey_scale=10., latent_grid=grid)
        self.assertNotEqual(int(np.argmax(b)), int(np.argmin(np.abs(grid - 20.))))

    def test_10_ecological_seam_assigns_all_candidates(self):
        class Model: survey_scale = 10.
        class POMDP:
            model = Model(); grid = np.array([0., 1., 2., 3.], dtype=np.float64)
        class Candidate:
            pomdp = POMDP(); internal_belief = None
        class Policy: candidates = [Candidate(), Candidate()]
        p = Policy()
        receipts = assign_ecological_internal_beliefs(
            p, 20., capability=self.cap, observation_noise_sigma=.2, observation_scale=1.)
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(np.argmax(c.internal_belief) == 2 for c in p.candidates))

    def test_11_artifact_applicability_cannot_fabricate_hashes(self):
        validate_artifact_axis({"status": "DEFINITIONALLY_NOT_APPLICABLE", "sha256": None})
        validate_artifact_axis({"status": "UNRESOLVED", "sha256": None})
        with self.assertRaises(IntegrationViolation):
            validate_artifact_axis({"status": "DEFINITIONALLY_NOT_APPLICABLE", "sha256": "0" * 64})
        with self.assertRaises(IntegrationViolation):
            validate_artifact_axis({"status": "APPLICABLE", "sha256": "not-a-hash"})

    def test_12_changed_artifacts_emit_end_to_end_label(self):
        self.assertEqual(interpretation_label(fitted_artifacts_equal=False, residual_sigma_equal=True), END_TO_END_LABEL)
        self.assertEqual(interpretation_label(fitted_artifacts_equal=True, residual_sigma_equal=False), END_TO_END_LABEL)

    def test_13_rng_state_is_unchanged(self):
        rng = np.random.default_rng(116)
        before = json.dumps(rng.bit_generator.state, sort_keys=True)
        ExternalMethodWrapper("ensemble_value_disagreement_pessimism").route_general(belief(), rng=rng)
        after = json.dumps(rng.bit_generator.state, sort_keys=True)
        self.assertEqual(before, after)

    def test_14_serialization_reload_parity(self):
        root = Path(os.environ["I2B_TEST_TMPDIR"])
        value = {"methods": list(METHODS), "label": END_TO_END_LABEL}
        path = root / "receipt.json"
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), value)

    def test_15_arm_o_rejects_derived_exact_input(self):
        with self.assertRaises(IntegrationViolation):
            ExternalMethodWrapper("refplan").route_general(
                belief(), exact_current_abundance=13., capability=self.cap
            )


if __name__ == "__main__":
    unittest.main()
