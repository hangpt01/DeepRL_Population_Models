from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ..artifacts import (
    MATCHED_SURROGATE_LABEL,
    CanonicalArtifact,
    MatchedSurrogateBinding,
    SharedEcologicalPolicy,
    build_ecological_arm_interfaces,
    evd_surrogate_applicability,
    seal_matched_arm_o_surrogate,
    validate_complete_artifact_bundle,
    validate_ecological_arm_pair,
    validate_registered_ecological_policy_set,
    validate_registered_surrogate_set,
    verify_matched_surrogate_arms,
)
from ..common import ContractError, sha256_bytes
from ..real_artifacts import (
    load_frozen_fitted_object,
    validate_frozen_fitted_object_roundtrip,
)
from .conftest import HASHES, artifact_components


@pytest.mark.parametrize(
    "method",
    [
        "plus_adapted_ricker_only_pbvi",
        "moor_adapted_ricker_misspec_pbvi",
        "refplan",
        "ogsrl",
        "bamcts",
        "ensemble_value_disagreement_pessimism",
    ],
)
def test_complete_artifact_serialization_and_reload_parity(method):
    components = artifact_components(method)
    receipt = validate_complete_artifact_bundle(
        method,
        components,
        prediction_fixture=np.arange(6, dtype=np.float64).reshape(2, 3),
        expected_component_hashes={key: sha256_bytes(value) for key, value in components.items()},
    )
    assert receipt.reload_parity
    assert len(receipt.bundle_sha256) == 64


def test_missing_fitted_artifact_identity_rejected():
    components = artifact_components("ogsrl")
    del components["ogsrl_guardian"]
    with pytest.raises(ContractError, match="incomplete"):
        validate_complete_artifact_bundle(
            "ogsrl",
            components,
            prediction_fixture=np.ones((2, 3), dtype=np.float64),
            expected_component_hashes={
                key: sha256_bytes(value) for key, value in components.items()
            },
        )


def test_training_history_never_qualifies_as_fitted_identity():
    artifact = CanonicalArtifact(
        "training_history.csv", "refplan", "synthetic", {"loss": [1.0, 0.5]}
    )
    with pytest.raises(ContractError, match="Training-history|training-history"):
        artifact.to_bytes()


def test_serialization_reload_mismatch_rejected():
    malformed = artifact_components("refplan")["reward_surrogate"] + b" "
    with pytest.raises(ContractError, match="canonical"):
        CanonicalArtifact.from_bytes(malformed)


def test_matched_surrogate_is_one_arm_o_artifact_for_both_arms(tmp_path):
    binding = seal_matched_arm_o_surrogate(
        tmp_path / "surrogate.json",
        cell="tiger",
        source_public_view_sha256=HASHES[0],
        state={
            "feature_order": ["state", "context", "time"],
            "weights": np.ones(3, dtype=np.float64),
            "bias": 0.0,
        },
        fitting_arm="O",
    )
    receipt = verify_matched_surrogate_arms(binding)
    assert receipt["label"] == MATCHED_SURROGATE_LABEL
    assert receipt["arm_o_sha256"] == receipt["arm_t_sha256"]


def test_arm_t_surrogate_refit_is_impossible(tmp_path):
    with pytest.raises(ContractError, match="Arm T"):
        seal_matched_arm_o_surrogate(
            tmp_path / "surrogate.json",
            cell="fox",
            source_public_view_sha256=HASHES[0],
            state={
                "feature_order": ["state"],
                "weights": np.ones(1, dtype=np.float64),
                "bias": 0.0,
            },
            fitting_arm="T",
        )
    components = artifact_components("refplan")
    surrogate = CanonicalArtifact.from_bytes(components["reward_surrogate"])
    components["reward_surrogate"] = CanonicalArtifact(
        surrogate.component,
        surrogate.method,
        "Arm T exact-state view",
        surrogate.state,
    ).to_bytes()
    with pytest.raises(ContractError, match="Arm O public"):
        validate_complete_artifact_bundle(
            "refplan",
            components,
            prediction_fixture=np.ones((2, 3), dtype=np.float64),
            expected_component_hashes={
                key: sha256_bytes(value) for key, value in components.items()
            },
        )


def test_surrogate_byte_mismatch_rejected(tmp_path):
    binding = seal_matched_arm_o_surrogate(
        tmp_path / "surrogate.json",
        cell="fox",
        source_public_view_sha256=HASHES[0],
        state={
            "feature_order": ["state", "context", "time"],
            "weights": np.ones(3, dtype=np.float64),
            "bias": 0.0,
        },
        fitting_arm="O",
    )
    binding.path.write_bytes(binding.path.read_bytes() + b"corrupt")
    with pytest.raises(ContractError, match="changed"):
        verify_matched_surrogate_arms(binding)


def test_ecological_shared_policy_bytes_and_context():
    components = artifact_components("plus_adapted_ricker_only_pbvi")
    policy_o = SharedEcologicalPolicy(
        "plus_adapted_ricker_only_pbvi",
        "amur_tiger__allee__sigma_0p2",
        components,
        10.0,
    )
    policy_t = SharedEcologicalPolicy(
        "plus_adapted_ricker_only_pbvi",
        "amur_tiger__allee__sigma_0p2",
        components,
        10.0,
    )
    interface_o, interface_t = build_ecological_arm_interfaces(
        raw_abundance=13.0,
        noisy_belief=np.array([0.2, 0.5, 0.3], dtype=np.float64),
        survey_scale=10.0,
        latent_grid=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        action_history=[1, 2],
        observation_history=[10.0, 11.0, 12.0],
        timestep=2,
    )
    receipt = validate_ecological_arm_pair(
        policy_o,
        policy_t,
        observation_sigma=0.2,
        arm_o_interface=interface_o,
        arm_t_interface=interface_t,
    )
    assert receipt["identical_policy_bytes"]


def test_ecological_conversion_receipt_must_identify_actual_point_mass():
    components = artifact_components("plus_adapted_ricker_only_pbvi")
    policy = SharedEcologicalPolicy(
        "plus_adapted_ricker_only_pbvi",
        "amur_tiger__allee__sigma_0p2",
        components,
        10.0,
    )
    interface_o, interface_t = build_ecological_arm_interfaces(
        raw_abundance=13.0,
        noisy_belief=np.array([0.2, 0.5, 0.3], dtype=np.float64),
        survey_scale=10.0,
        latent_grid=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        action_history=[],
        observation_history=[12.0],
        timestep=0,
    )
    contradictory = dict(interface_t.conversion_receipt or {})
    contradictory["selected_index"] = 0 if contradictory["selected_index"] != 0 else 2
    with pytest.raises(ContractError, match="index"):
        validate_ecological_arm_pair(
            policy,
            policy,
            observation_sigma=0.2,
            arm_o_interface=interface_o,
            arm_t_interface=replace(interface_t, conversion_receipt=contradictory),
        )


def test_ecological_policy_mismatch_sigma_zero_and_allee_leak_rejected():
    components = artifact_components("moor_adapted_ricker_misspec_pbvi")
    policy_o = SharedEcologicalPolicy(
        "moor_adapted_ricker_misspec_pbvi",
        "amur_tiger__allee__sigma_0p2",
        components,
        10.0,
    )
    different_components = dict(components)
    surrogate = CanonicalArtifact.from_bytes(different_components["reward_surrogate"])
    surrogate_state = dict(surrogate.state)
    surrogate_state["weights"] = np.asarray(surrogate_state["weights"]) + 1.0
    different_components["reward_surrogate"] = CanonicalArtifact(
        surrogate.component,
        surrogate.method,
        surrogate.fit_source,
        surrogate_state,
    ).to_bytes()
    mismatched = SharedEcologicalPolicy(
        "moor_adapted_ricker_misspec_pbvi",
        "amur_tiger__allee__sigma_0p2",
        different_components,
        10.0,
    )
    interface_o, interface_t = build_ecological_arm_interfaces(
        raw_abundance=13.0,
        noisy_belief=np.array([0.2, 0.5, 0.3], dtype=np.float64),
        survey_scale=10.0,
        latent_grid=np.array([0.5, 1.0, 1.5], dtype=np.float64),
        action_history=[],
        observation_history=[12.0],
        timestep=0,
    )
    with pytest.raises(ContractError, match="identical complete"):
        validate_ecological_arm_pair(
            policy_o,
            mismatched,
            observation_sigma=0.2,
            arm_o_interface=interface_o,
            arm_t_interface=interface_t,
        )
    with pytest.raises(ContractError, match="sigma"):
        validate_ecological_arm_pair(
            policy_o,
            policy_o,
            observation_sigma=0.0,
            arm_o_interface=interface_o,
            arm_t_interface=interface_t,
        )


def test_ecological_policy_rejects_arbitrary_bytes_instead_of_complete_artifacts():
    with pytest.raises(ContractError, match="complete artifact bundle"):
        SharedEcologicalPolicy(
            "plus_adapted_ricker_only_pbvi",
            "amur_tiger__allee__sigma_0p2",
            {
                "ricker_fit_cache": b"ricker",
                "reward_surrogate": b"surrogate",
                "pbvi_policy": b"pbvi",
            },
            10.0,
        ).validate()


def test_ecological_policy_set_rejects_incomplete_or_duplicate_cells():
    components = artifact_components("plus_adapted_ricker_only_pbvi")
    policy = SharedEcologicalPolicy(
        "plus_adapted_ricker_only_pbvi",
        "amur_tiger__allee__sigma_0p2",
        components,
        10.0,
    )
    with pytest.raises(ContractError, match="incomplete"):
        validate_registered_ecological_policy_set([(policy, policy)])
    with pytest.raises(ContractError, match="duplicate"):
        validate_registered_ecological_policy_set([(policy, policy)] * 4)


def test_placeholder_state_and_unchanged_expected_hash_rejected():
    with pytest.raises(ContractError, match="key mismatch"):
        CanonicalArtifact(
            "dynamics_ensemble",
            "refplan",
            "synthetic",
            {"placeholder_but_nonempty": 1},
        ).to_bytes()
    components = artifact_components("refplan")
    expected = {key: sha256_bytes(value) for key, value in components.items()}
    artifact = CanonicalArtifact.from_bytes(components["dynamics_ensemble"])
    altered = dict(components)
    state = dict(artifact.state)
    state["weights"] = np.asarray(state["weights"]) + 1.0
    altered["dynamics_ensemble"] = CanonicalArtifact(
        artifact.component, artifact.method, artifact.fit_source, state
    ).to_bytes()
    with pytest.raises(ContractError, match="hash mismatch"):
        validate_complete_artifact_bundle(
            "refplan",
            altered,
            prediction_fixture=np.ones((2, 3), dtype=np.float64),
            expected_component_hashes=expected,
        )


def test_surrogate_cross_cell_and_source_rebinding_rejected(tmp_path):
    binding = seal_matched_arm_o_surrogate(
        tmp_path / "surrogate.json",
        cell="amur_tiger__allee__sigma_0p2",
        source_public_view_sha256=HASHES[0],
        state={
            "feature_order": ["state", "context", "time"],
            "weights": np.ones(3, dtype=np.float64),
            "bias": 0.0,
        },
        fitting_arm="O",
    )
    with pytest.raises(ContractError, match="cross-cell"):
        MatchedSurrogateBinding(
            "crab_eating_fox__allee__sigma_0p2",
            binding.path,
            binding.sha256,
            binding.source_public_view_sha256,
        ).load_for_arm("O", "refplan")
    with pytest.raises(ContractError, match="source-public-view"):
        MatchedSurrogateBinding(
            binding.cell,
            binding.path,
            binding.sha256,
            HASHES[1],
        ).load_for_arm("T", "ogsrl")


def test_registered_surrogate_set_requires_one_binding_per_cell(tmp_path):
    bindings = []
    for index, cell in enumerate(
        ("amur_tiger__allee__sigma_0p2", "crab_eating_fox__allee__sigma_0p2")
    ):
        bindings.append(
            seal_matched_arm_o_surrogate(
                tmp_path / f"surrogate-{index}.json",
                cell=cell,
                source_public_view_sha256=HASHES[index],
                state={
                    "feature_order": ["state", "context", "time"],
                    "weights": np.ones(3, dtype=np.float64),
                    "bias": 0.0,
                },
                fitting_arm="O",
            )
        )
    assert set(validate_registered_surrogate_set(bindings)) == {
        "amur_tiger__allee__sigma_0p2",
        "crab_eating_fox__allee__sigma_0p2",
    }
    with pytest.raises(ContractError, match="one binding|duplicated"):
        validate_registered_surrogate_set([bindings[0], bindings[0]])


def test_evd_surrogate_is_definitionally_not_applicable_without_fake_hash():
    receipt = evd_surrogate_applicability("ensemble_value_disagreement_pessimism")
    assert receipt["status"] == "DEFINITIONALLY_NOT_APPLICABLE"
    assert receipt["sha256"] is None


def test_real_frozen_dynamics_object_roundtrip_runs_real_prediction_parity():
    repository = Path(__file__).resolve().parents[4]
    general_track = str(repository / "src/tracks/general")
    if general_track not in sys.path:
        sys.path.insert(0, general_track)
    from real_ecology_benchmark.dynamics import (  # noqa: PLC0415
        ContinuousDynamicsEnsemble,
        LinearDynamicsMember,
    )

    members = [
        LinearDynamicsMember(
            coefficients=np.linspace(0.01, 0.36, 36, dtype=np.float64),
            residual_sigma=0.02,
            num_actions=11,
            K_ref=250.0,
            env_cfg=None,
        )
        for _ in range(5)
    ]
    fitted = ContinuousDynamicsEnsemble(members, seed=7)
    payload, receipt = validate_frozen_fitted_object_roundtrip(
        "general_dynamics_ensemble",
        fitted,
        operation="predict",
        args=(
            np.array([10.0, 20.0], dtype=np.float64),
            np.array([0, 10], dtype=np.int64),
        ),
        repository_root=repository,
    )
    reloaded = load_frozen_fitted_object(payload, repository_root=repository)
    assert type(reloaded) is ContinuousDynamicsEnsemble
    assert receipt["fresh_object_reload"] is True
    assert receipt["complete_instance_graph"] is True
    assert all(
        np.array_equal(a.coefficients.view(np.uint64), b.coefficients.view(np.uint64))
        for a, b in zip(fitted.members, reloaded.members)
    )


def test_real_frozen_object_rejects_incomplete_generic_substitute():
    repository = Path(__file__).resolve().parents[4]
    with pytest.raises(ContractError, match="root class"):
        validate_frozen_fitted_object_roundtrip(
            "ogsrl_fitted_policy",
            {"action_matrix": [[1.0]], "bias": [0.0]},
            operation="act",
            args=(),
            repository_root=repository,
        )


def test_real_frozen_evd_policy_roundtrip_runs_real_action_parity():
    repository = Path(__file__).resolve().parents[4]
    general_track = str(repository / "src/tracks/general")
    if general_track not in sys.path:
        sys.path.insert(0, general_track)
    from real_ecology_benchmark.behavior_model import (  # noqa: PLC0415
        CalibratedBehaviorModel,
    )
    from real_ecology_benchmark.config import (  # noqa: PLC0415
        MethodContext,
        ModelConfig,
        PlannerConfig,
    )
    from real_ecology_benchmark.methods.ensemble_value_disagreement import (  # noqa: PLC0415
        BootstrapQMember,
        EnsembleValueDisagreementPolicy,
    )
    from real_ecology_benchmark.realdata import PUBLIC_ACTION_CHANNELS  # noqa: PLC0415
    from real_ecology_benchmark.types import BeliefState  # noqa: PLC0415

    context = MethodContext(
        num_actions=11,
        action_costs=(0.0,) * 11,
        action_channels=(next(iter(sorted(PUBLIC_ACTION_CHANNELS))),) * 11,
        observation_noise_sigma=0.2,
        horizon=50,
        observation_scale=100.0,
        pop_id="pop_synthetic",
    )
    fitted = EnsembleValueDisagreementPolicy(context, ModelConfig(), PlannerConfig(), seed=9)
    feature_dim = 10
    fitted.behavior_model = CalibratedBehaviorModel(
        weights=np.zeros((11, feature_dim + 1), dtype=np.float64),
        mean=np.zeros(feature_dim + 1, dtype=np.float64),
        scale=np.ones(feature_dim + 1, dtype=np.float64),
    )
    fitted.q_members = [
        BootstrapQMember(
            q_weights=np.full((11, feature_dim + 1), index / 100.0, dtype=np.float64),
            bootstrap_episode_ids=np.arange(4, dtype=np.int32),
            seed=100 + index,
            bellman_mse=0.01,
        )
        for index in range(20)
    ]
    fitted.feature_dim = feature_dim
    fitted.q_weights = np.mean(np.stack([member.q_weights for member in fitted.q_members]), axis=0)
    belief = BeliefState(
        states=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        contexts=np.ones((4, 3), dtype=np.float64),
        regimes=np.zeros(4, dtype=np.int8),
        log_weights=np.log(np.full(4, 0.25, dtype=np.float64)),
        observation=25.0,
    )
    _payload, receipt = validate_frozen_fitted_object_roundtrip(
        "evd_fitted_policy",
        fitted,
        operation="act",
        args=(belief, 25.0),
        repository_root=repository,
    )
    assert receipt["operation"] == "act"
    assert receipt["result"] == "PASS"


def test_real_frozen_refplan_ogsrl_and_bamcts_run_real_action_parity():
    repository = Path(__file__).resolve().parents[4]
    general_track = str(repository / "src/tracks/general")
    if general_track not in sys.path:
        sys.path.insert(0, general_track)
    from real_ecology_benchmark.behavior_model import (  # noqa: PLC0415
        CalibratedBehaviorModel,
    )
    from real_ecology_benchmark.config import (  # noqa: PLC0415
        MethodContext,
        ModelConfig,
        PlannerConfig,
    )
    from real_ecology_benchmark.methods.bamcts import BAMCTSPolicy  # noqa: PLC0415
    from real_ecology_benchmark.methods.ogsrl import (  # noqa: PLC0415
        OGSRLPolicy,
        PublicKNNGuardian,
    )
    from real_ecology_benchmark.methods.refplan import RefPlanPolicy  # noqa: PLC0415
    from real_ecology_benchmark.public_models import (  # noqa: PLC0415
        PublicDynamicsEnsemble,
        PublicDynamicsMember,
        PublicParticlePlanner,
    )
    from real_ecology_benchmark.public_surrogate import (  # noqa: PLC0415
        PublicFeatureSpec,
        PublicRewardRiskSurrogate,
    )
    from real_ecology_benchmark.realdata import PUBLIC_ACTION_CHANNELS  # noqa: PLC0415
    from real_ecology_benchmark.types import BeliefState  # noqa: PLC0415

    action_count = 11
    pop_id = "pop_synthetic"
    feature_spec = PublicFeatureSpec(
        observation_scale=100.0,
        continuous_mean=np.zeros(7, dtype=np.float64),
        continuous_std=np.ones(7, dtype=np.float64),
        pop_vocabulary=(pop_id,),
        num_actions=action_count,
        public_horizon=50,
    )
    surrogate = PublicRewardRiskSurrogate(
        feature_spec=feature_spec,
        reward_coefficients=np.zeros(20, dtype=np.float64),
        risk_coefficients=None,
        constant_risk=0.1,
        action_costs=np.zeros(action_count, dtype=np.float64),
        diagnostics={"synthetic_fixture": 1.0},
        public_data_hash="a" * 64,
        split_seed=17,
    )
    context = MethodContext(
        num_actions=action_count,
        action_costs=(0.0,) * action_count,
        action_channels=(next(iter(sorted(PUBLIC_ACTION_CHANNELS))),) * action_count,
        observation_noise_sigma=0.2,
        horizon=50,
        observation_scale=100.0,
        pop_id=pop_id,
        surrogate=surrogate,
    )
    model_cfg = ModelConfig(ensemble_size=5)
    planner_cfg = PlannerConfig(
        horizon=1,
        sequences=11,
        particles=2,
        bamcts_depth=1,
        bamcts_simulations=2,
        ogsrl_cost_horizon=1,
        ogsrl_deployment_rollouts=2,
    )
    members = [
        PublicDynamicsMember(
            coefficients=np.concatenate(
                (
                    np.array([0.18 + index * 0.001, 0.4], dtype=np.float64),
                    np.zeros(34, dtype=np.float64),
                )
            ),
            residual_sigma=0.02,
            num_actions=action_count,
            observation_scale=100.0,
        )
        for index in range(5)
    ]
    dynamics = PublicDynamicsEnsemble(members, seed=23)
    belief = BeliefState(
        states=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        contexts=np.column_stack(
            (
                np.full(4, 15.0, dtype=np.float64),
                np.full(4, 20.0, dtype=np.float64),
                np.arange(4, dtype=np.float64),
            )
        ),
        regimes=np.zeros(4, dtype=np.int8),
        log_weights=np.log(np.full(4, 0.25, dtype=np.float64)),
        observation=25.0,
    )
    feature_dim = belief.public_features(context.observation_scale).shape[0]
    prior = CalibratedBehaviorModel(
        weights=np.full((action_count, feature_dim + 1), 0.001, dtype=np.float64),
        mean=np.zeros(feature_dim + 1, dtype=np.float64),
        scale=np.ones(feature_dim + 1, dtype=np.float64),
    )

    refplan = RefPlanPolicy(context, model_cfg, planner_cfg, seed=31)
    refplan.dynamics = dynamics
    refplan.posterior = np.full(5, 0.2, dtype=np.float64)
    refplan.policy_prior = prior
    refplan.prior_weights = prior.weights
    refplan.prior_scaler = (prior.mean, prior.scale)
    refplan.planner = PublicParticlePlanner(context, planner_cfg, seed=31)

    bamcts = BAMCTSPolicy(context, model_cfg, planner_cfg, simulations=2, depth=1, seed=37)
    bamcts.dynamics = PublicDynamicsEnsemble(members, seed=23)
    bamcts.posterior = np.full(5, 0.2, dtype=np.float64)

    guardian_feature_dim = 2 + action_count
    ogsrl = OGSRLPolicy(context, model_cfg, planner_cfg, seed=41)
    ogsrl.dynamics = PublicDynamicsEnsemble(members, seed=23)
    ogsrl.guardian = PublicKNNGuardian(
        anchors=np.zeros((2, guardian_feature_dim), dtype=np.float64),
        mean=np.zeros(guardian_feature_dim, dtype=np.float64),
        scale=np.ones(guardian_feature_dim, dtype=np.float64),
        threshold=1e6,
        num_actions=action_count,
        observation_scale=100.0,
    )
    ogsrl.surrogate = surrogate
    ogsrl.s_low = 50.0
    ogsrl.actor_weights = np.zeros((action_count, 3), dtype=np.float64)
    ogsrl.lambda_safety = 1.0
    ogsrl.lambda_ood = 1.0

    for component, policy in (
        ("refplan_fitted_policy", refplan),
        ("ogsrl_fitted_policy", ogsrl),
        ("bamcts_fitted_policy", bamcts),
    ):
        with np.errstate(all="ignore"):
            payload, receipt = validate_frozen_fitted_object_roundtrip(
                component,
                policy,
                operation="act",
                args=(belief, 25.0),
                repository_root=repository,
            )
        reloaded = load_frozen_fitted_object(payload, repository_root=repository)
        assert type(reloaded) is type(policy)
        if component == "refplan_fitted_policy":
            assert reloaded.prior_weights is reloaded.policy_prior.weights
            assert reloaded.planner.context is reloaded.method_context
        assert receipt["result"] == "PASS"


def test_real_frozen_plus_and_moor_run_real_pbvi_action_parity():
    repository = Path(__file__).resolve().parents[4]
    general_track = str(repository / "src/tracks/general")
    ecological_track = str(repository / "src/tracks/ecological")
    original_path = list(sys.path)
    for name in tuple(sys.modules):
        if name == "real_ecology_benchmark" or name.startswith("real_ecology_benchmark."):
            del sys.modules[name]
    sys.path[:] = [item for item in sys.path if item not in {general_track, ecological_track}]
    sys.path.insert(0, ecological_track)
    try:
        from real_ecology_benchmark.config import (  # noqa: PLC0415
            FaithfulConfig,
            FaithfulFitConfig,
            FaithfulModelConfig,
            FaithfulPlannerConfig,
            MethodContext,
            ModelConfig,
            PlannerConfig,
        )
        from real_ecology_benchmark.faithful_ecology import (  # noqa: PLC0415
            MechanisticModel,
        )
        from real_ecology_benchmark.faithful_fit import (  # noqa: PLC0415
            RICKER_ONLY_CANDIDATE_CONSTRUCTION,
            CandidateBank,
            FitResult,
        )
        from real_ecology_benchmark.faithful_pomdp import (  # noqa: PLC0415
            CandidatePOMDP,
        )
        from real_ecology_benchmark.methods.moor_faithful import (  # noqa: PLC0415
            MOORFaithfulRickerPBVIPolicy,
        )
        from real_ecology_benchmark.methods.plus_faithful import (  # noqa: PLC0415
            PLUSRickerOnlyFaithfulPBVIPolicy,
        )
        from real_ecology_benchmark.planners.pbvi import (  # noqa: PLC0415
            PointBasedPlanner,
        )
        from real_ecology_benchmark.types import BeliefState  # noqa: PLC0415

        action_count = 11
        channels = ("none",) * action_count
        context = MethodContext(
            num_actions=action_count,
            action_costs=(0.0,) * action_count,
            action_channels=channels,
            observation_noise_sigma=0.2,
            horizon=50,
            observation_scale=100.0,
            pop_id="pop_synthetic",
        )
        faithful = FaithfulConfig(
            model=FaithfulModelConfig(forms=("ricker",), candidates_per_form=8, prior="uniform"),
            fit=FaithfulFitConfig(starts=1, iterations=1, mc_paths=1, regime_mc_paths=8),
            planner=FaithfulPlannerConfig(
                state_bins=5,
                capacity_bins=2,
                observation_bins=5,
                transition_samples=1,
                observation_samples=1,
                belief_points=1,
                observation_branches=2,
                horizon=1,
            ),
        )

        def fitted_result(index: int) -> FitResult:
            growth = np.zeros(action_count, dtype=np.float64)
            growth[0] = 0.20 + index * 0.001
            model = MechanisticModel(
                form="ricker",
                growth=growth,
                mortality=np.zeros(action_count, dtype=np.float64),
                capacity_increment=np.zeros(action_count, dtype=np.float64),
                stocking=np.zeros(action_count, dtype=np.float64),
                reset_log_mean=np.log(0.65),
                reset_log_scale=0.08,
                initial_capacity=1.0,
                capacity_ceiling=1.8,
                process_scale=0.03,
                observation_scale=0.05,
                survey_scale=100.0,
                depensation_thresholds=np.array([0.2, 0.35], dtype=np.float64),
                theta_exponent=2.0,
                regime_multipliers=np.array([0.7, 1.2], dtype=np.float64),
                regime_matrix=np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
                action_channels=channels,
                candidate_id=f"synthetic_ricker_{index:02d}",
            )
            return FitResult(
                model=model,
                objective=float(index),
                holdout_normalized_survey_sse=1.0,
                selected_gradient_norm=0.1,
                curvature_condition_estimate=2.0,
                start_gradient_norms=(0.1,),
                start_objective_traces=((1.0,),),
                finite_start_parameter_std_mean=0.0,
                selected_start=0,
                start_objectives=(1.0,),
                action_rows=(1,) * action_count,
                action_episodes=(1,) * action_count,
                sparse_actions=(),
                fit_episode_ids=(0,),
                holdout_episode_ids=(1,),
                public_data_hash="a" * 64,
                random_bank_hash="b" * 64,
                optimizer="torch_lbfgs",
                iterations=1,
                fit_cache_key=f"synthetic-cache-{index:02d}",
            )

        fits = tuple(fitted_result(index) for index in range(8))
        bank = CandidateBank(
            fits,
            np.full(8, 1.0 / 8.0, dtype=np.float64),
            "uniform",
            RICKER_ONLY_CANDIDATE_CONSTRUCTION,
        )
        pomdps = [
            CandidatePOMDP(item.model, context, faithful.planner, 100 + index)
            for index, item in enumerate(fits)
        ]
        planners = [
            PointBasedPlanner(pomdp, faithful.planner, 0.95, 200 + index)
            for index, pomdp in enumerate(pomdps)
        ]
        plus = PLUSRickerOnlyFaithfulPBVIPolicy(
            context,
            ModelConfig(),
            PlannerConfig(horizon=1),
            seed=43,
            faithful_cfg=faithful,
        )
        plus.candidate_bank = bank
        plus.fit_cache_statuses = ["synthetic"] * 8
        plus.pomdps = pomdps
        plus.planners = planners
        plus.posterior = bank.initial_weights.copy()

        moor = MOORFaithfulRickerPBVIPolicy(
            context,
            ModelConfig(),
            PlannerConfig(horizon=1),
            seed=47,
            faithful_cfg=faithful,
        )
        moor.fit_result = fits[0]
        moor.fit_cache_status = "synthetic"
        moor.pomdp = CandidatePOMDP(fits[0].model, context, faithful.planner, 300)
        moor.planner = PointBasedPlanner(moor.pomdp, faithful.planner, 0.95, 301)

        belief = BeliefState(
            states=np.array([10.0, 20.0], dtype=np.float64),
            contexts=np.ones((2, 3), dtype=np.float64),
            regimes=np.zeros(2, dtype=np.int8),
            log_weights=np.log(np.full(2, 0.5, dtype=np.float64)),
            observation=25.0,
            diagnostics={"public_observation_filter": 1.0},
        )
        for component, policy in (
            ("plus_fitted_policy", plus),
            ("moor_fitted_policy", moor),
        ):
            payload, receipt = validate_frozen_fitted_object_roundtrip(
                component,
                policy,
                operation="act",
                args=(belief, 25.0),
                repository_root=repository,
            )
            reloaded = load_frozen_fitted_object(payload, repository_root=repository)
            assert type(reloaded) is type(policy)
            if component == "plus_fitted_policy":
                assert reloaded.planners[0].model is reloaded.pomdps[0]
                assert reloaded.candidate_bank.fits[0].model is reloaded.pomdps[0].model
            else:
                assert reloaded.planner.model is reloaded.pomdp
                assert reloaded.fit_result.model is reloaded.pomdp.model
            assert receipt["result"] == "PASS"
    finally:
        for name in tuple(sys.modules):
            if name == "real_ecology_benchmark" or name.startswith("real_ecology_benchmark."):
                del sys.modules[name]
        sys.path[:] = original_path
