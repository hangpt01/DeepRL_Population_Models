from dataclasses import replace
import json

import numpy as np
import pytest

from real_ecology_benchmark.config import (
    FaithfulFitConfig,
    FaithfulModelConfig,
    MethodContext,
    FaithfulPlannerConfig,
)
from real_ecology_benchmark.faithful_artifacts import load_model, save_model
from real_ecology_benchmark.dataset import TrajectoryDataset, dataset_sha256
from real_ecology_benchmark.faithful_ecology import (
    advance_regimes,
    fixed_regime_matrix,
)
from real_ecology_benchmark.faithful_fit import (
    ACTION_PARAMETER_COUNT,
    _decode,
    _initial_raw,
    _ordered_indices,
    _random_bank,
    _regime_schedule,
    _symmetric_positive,
    _trajectory_objective,
    _to_model,
    _torch_noiseless_transition,
    action_parameter_count,
    conditional_survey_mean_factor,
    fit_transition_hash,
    fit_cache_key,
    fit_mechanistic_model,
    load_or_fit_mechanistic_model,
    split_history_episodes,
)
from real_ecology_benchmark.methods import METHODS
from real_ecology_benchmark.faithful_pomdp import CandidatePOMDP

from .faithful_fixtures import public_context, public_dataset


REAL_CHANNELS = (
    "none",
    "rate",
    "rate",
    "rate",
    "rate",
    "capacity",
    "capacity",
    "rate+capacity",
    "rate+capacity",
    "rate+capacity",
    "state",
)


def real_context() -> MethodContext:
    return MethodContext(
        num_actions=11,
        action_costs=(0.0,) * 11,
        action_channels=REAL_CHANNELS,
        observation_noise_sigma=0.4,
        horizon=25,
        observation_scale=100.0,
        pop_id="pop_0123456789abcdef",
        surrogate=None,
    )


def test_real_channel_map_has_exact_14_parameters_and_structural_zeros():
    torch = pytest.importorskip("torch")
    context = real_context()
    assert action_parameter_count(context.action_channels) == ACTION_PARAMETER_COUNT == 14
    raw = torch.as_tensor(
        _initial_raw(context, "ricker", FaithfulModelConfig(), seed=8),
        dtype=torch.float64,
    )
    values = _decode(raw, context, "ricker", FaithfulModelConfig(), 0.9, torch)
    capacity = values["capacity_increment"].detach().numpy()
    stocking = values["stocking"].detach().numpy()
    for action, channel in enumerate(REAL_CHANNELS):
        assert (
            capacity[action] == 0.0
            if channel not in {"capacity", "rate+capacity"}
            else capacity[action] > 0.0
        )
        assert stocking[action] == 0.0 if channel != "state" else stocking[action] > 0.0
    growth = values["growth"].detach().numpy()
    mortality = values["mortality"].detach().numpy()
    assert not np.any((growth > 0.0) & (mortality > 0.0))


def test_fixed_pi_has_no_optimizer_coordinates():
    context = real_context()
    expected_dimensions = {"ricker": 19, "allee": 20, "theta": 20, "regime": 23}
    for form, expected in expected_dimensions.items():
        raw = _initial_raw(context, form, FaithfulModelConfig(), seed=8)
        assert len(raw) == expected
    assert not fixed_regime_matrix(0.9).flags.writeable


def test_signed_rate_zero_uses_symmetric_clarke_subgradients():
    torch = pytest.importorskip("torch")
    rate = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    growth = _symmetric_positive(rate, torch)
    growth.backward()
    assert rate.grad.item() == pytest.approx(0.5)
    rate.grad.zero_()
    mortality = _symmetric_positive(-rate, torch)
    mortality.backward()
    assert rate.grad.item() == pytest.approx(-0.5)
    rates = torch.as_tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    growth = _symmetric_positive(rates, torch)
    mortality = _symmetric_positive(-rates, torch)
    assert torch.all(growth * mortality == 0.0)
    assert torch.equal(growth, torch.as_tensor([0.0, 0.0, 1.0]))
    assert torch.equal(mortality, torch.as_tensor([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("sigma", [0.0, 0.4])
def test_conditional_survey_mean_recovers_latent_scale_without_noise_draw(sigma):
    latent = np.asarray([0.2, 0.7, 1.4])
    factor = conditional_survey_mean_factor(sigma)
    conditional_observation = latent * factor
    recovered = conditional_observation / factor
    assert np.allclose(recovered, latent, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("seed", [3, 17, 41])
def test_sigma_point_four_scalar_scale_recovery_and_removed_bias(seed):
    sigma = 0.4
    latent = 0.73
    observations = latent * np.exp(np.random.default_rng(seed).normal(0.0, sigma, size=20_000))
    fitted = float(np.mean(observations) / conditional_survey_mean_factor(sigma))
    assert abs(fitted / latent - 1.0) <= 0.03
    assert np.exp(-(sigma**2)) == pytest.approx(0.8521437889662113)
    assert 1.0 - np.exp(-(sigma**2)) == pytest.approx(0.14785621103378865)


def _controlled_conditional_mean_dataset() -> TrajectoryDataset:
    from .faithful_fixtures import ricker_model

    model = ricker_model()
    sigma = 0.4
    factor = conditional_survey_mean_factor(sigma)
    observations, following, actions, episode_ids, timesteps, dones = ([] for _ in range(6))
    for episode in range(12):
        latent = float(np.exp(model.reset_log_mean))
        capacity = model.initial_capacity
        for timestep in range(8):
            action = (episode + timestep) % model.num_actions
            next_latent = float(model.noiseless_next(latent, capacity, action))
            observations.append(latent * model.survey_scale * factor)
            following.append(next_latent * model.survey_scale * factor)
            actions.append(action)
            episode_ids.append(episode)
            timesteps.append(timestep)
            dones.append(timestep == 7)
            capacity = model.next_capacity(capacity, action)
            latent = next_latent
    actions_array = np.asarray(actions, dtype=np.int64)
    done_array = np.asarray(dones)
    costs = np.asarray([0.0, 0.1, 0.2])
    dataset = TrajectoryDataset(
        observations=np.asarray(observations),
        actions=actions_array,
        rewards=np.zeros(len(actions_array)),
        next_observations=np.asarray(following),
        dones=done_array,
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        timestep=np.asarray(timesteps, dtype=np.int32),
        metadata={
            "expose_rk": "hidden",
            "num_actions": 3,
            "observation_noise_sigma": sigma,
            "reward_mode": "safe",
            "horizon": 8,
            "regime_label": "hidden-demographics_structure-unknown",
        },
        costs=costs[actions_array],
        pop_ids=np.full(len(actions_array), "pop_0123456789abcdef"),
        terminated=np.zeros(len(actions_array), dtype=bool),
        truncated=done_array,
        action_costs=costs,
    )
    dataset.validate()
    return dataset


def test_ordered_sigma_point_four_controlled_ricker_scale_recovery():
    pytest.importorskip("torch")
    from .faithful_fixtures import ricker_model

    dataset = _controlled_conditional_mean_dataset()
    context = replace(public_context(), observation_noise_sigma=0.4, horizon=8)
    fit_cfg = FaithfulFitConfig(starts=2, iterations=30, mc_paths=4)
    truth = float(np.exp(ricker_model().reset_log_mean))
    errors, gradients = [], []
    for seed in (11, 29, 47):
        result = fit_mechanistic_model(
            dataset, context, "ricker", FaithfulModelConfig(), fit_cfg, seed=seed
        )
        fitted = float(np.exp(result.model.reset_log_mean + 0.5 * result.model.reset_log_scale**2))
        errors.append(abs(fitted / truth - 1.0))
        gradients.append(result.selected_gradient_norm)
    assert float(np.median(errors)) <= 0.05
    assert np.all(np.isfinite(gradients))


def test_ricker_objective_has_no_sampled_survey_noise_channel():
    torch = pytest.importorskip("torch")
    dataset = public_dataset(episodes=3, length=4)
    context = replace(public_context(), observation_noise_sigma=0.4)
    model_cfg = FaithfulModelConfig()
    fit_cfg = FaithfulFitConfig(starts=1, iterations=2, mc_paths=2)
    ordered = _ordered_indices(dataset, np.unique(dataset.episode_id))
    initial, process, uniforms, _ = _random_bank([len(indices) for indices in ordered], 2, 9)
    raw = torch.as_tensor(_initial_raw(context, "ricker", model_cfg, 7), dtype=torch.float64)
    first = _trajectory_objective(
        raw,
        dataset,
        ordered,
        context,
        "ricker",
        model_cfg,
        fit_cfg,
        (initial, process, uniforms),
        0.9,
        torch,
    )
    second = _trajectory_objective(
        raw,
        dataset,
        ordered,
        context,
        "ricker",
        model_cfg,
        fit_cfg,
        (initial, process, 1.0 - uniforms),
        0.9,
        torch,
    )
    assert float(first) == pytest.approx(float(second), rel=0.0, abs=0.0)


def test_zero_noise_fit_is_bit_deterministic_under_same_process_bank():
    pytest.importorskip("torch")
    dataset = public_dataset(episodes=4, length=5)
    context = replace(public_context(), observation_noise_sigma=0.0)
    fit_cfg = FaithfulFitConfig(starts=1, iterations=2, mc_paths=1)
    first = fit_mechanistic_model(
        dataset, context, "ricker", FaithfulModelConfig(), fit_cfg, seed=37
    )
    second = fit_mechanistic_model(
        dataset, context, "ricker", FaithfulModelConfig(), fit_cfg, seed=37
    )
    assert conditional_survey_mean_factor(0.0) == 1.0
    assert first.objective == second.objective
    assert first.model.parameter_hash() == second.model.parameter_hash()


def test_canonical_regime_law_uses_current_state_then_switches():
    matrix = fixed_regime_matrix(0.8)
    current = np.asarray([0, 0, 1, 1])
    uniforms = np.asarray([0.79, 0.81, 0.19, 0.21])
    assert np.array_equal(advance_regimes(current, uniforms, matrix), [0, 1, 0, 1])


@pytest.mark.parametrize("form", ["ricker", "allee", "theta", "regime"])
def test_fitter_and_deployed_one_step_transitions_are_identical(form):
    torch = pytest.importorskip("torch")
    context = public_context()
    model_cfg = FaithfulModelConfig()
    raw_numpy = _initial_raw(context, form, model_cfg, seed=31)
    raw = torch.as_tensor(raw_numpy, dtype=torch.float64)
    parameters = _decode(raw, context, form, model_cfg, 0.9, torch)
    model = _to_model(raw_numpy, context, form, model_cfg, "identity", 0.9)
    latent = torch.as_tensor([0.0, 0.4, 0.8], dtype=torch.float64)
    regimes = torch.as_tensor([0, 1, 0], dtype=torch.int64)
    fitted, fitted_capacity = _torch_noiseless_transition(
        parameters, form, latent, parameters["initial_capacity"], 1, regimes, torch
    )
    deployed = model.noiseless_next(latent.numpy(), model.initial_capacity, 1, regimes.numpy())
    assert np.allclose(fitted.detach().numpy(), deployed, rtol=1e-12, atol=1e-12)
    assert float(fitted_capacity) == pytest.approx(
        model.next_capacity(model.initial_capacity, 1), rel=0.0, abs=1e-12
    )


def test_fixed_pi_is_immutable_and_artifact_round_trip_is_byte_identical(tmp_path):
    matrix = fixed_regime_matrix(0.9)
    assert not matrix.flags.writeable
    model = _to_model(
        _initial_raw(public_context(), "regime", FaithfulModelConfig(), 4),
        public_context(),
        "regime",
        FaithfulModelConfig(),
        "pi_roundtrip",
        0.9,
    )
    save_model(tmp_path / "regime.npz", model)
    loaded = load_model(tmp_path / "regime.npz")
    assert loaded.regime_law_hash == model.regime_law_hash
    assert loaded.regime_matrix.tobytes() == model.regime_matrix.tobytes()
    assert not loaded.regime_matrix.flags.writeable


@pytest.mark.parametrize("persistence", [0.8, 0.9, 0.97])
def test_pomdp_regime_transition_frequency_matches_fixed_pi(persistence):
    model = _to_model(
        _initial_raw(public_context(), "regime", FaithfulModelConfig(), 12),
        public_context(),
        "regime",
        FaithfulModelConfig(),
        "frequency",
        persistence,
    )
    samples = 4000
    config = FaithfulPlannerConfig(
        state_bins=5,
        capacity_bins=2,
        observation_bins=5,
        transition_samples=samples,
        observation_samples=8,
        belief_points=2,
        observation_branches=2,
        horizon=1,
    )
    pomdp = CandidatePOMDP(model, public_context(), config, seed=22)
    matrix = pomdp.transition_matrix(model.initial_capacity, action=0)
    bins = len(pomdp.abundance_grid)
    source = 2
    observed_stay = float(matrix[source, :bins].sum())
    tolerance = 5.0 * np.sqrt(persistence * (1.0 - persistence) / samples) + 1.0 / samples
    assert abs(observed_stay - persistence) <= tolerance


def test_fit_cache_is_reward_independent_and_transition_sensitive(tmp_path):
    pytest.importorskip("torch")
    dataset = public_dataset(episodes=4, length=5)
    fit_cfg = FaithfulFitConfig(starts=1, iterations=2, mc_paths=1)
    model_cfg = FaithfulModelConfig()
    first, status = load_or_fit_mechanistic_model(
        dataset,
        public_context(),
        "ricker",
        model_cfg,
        fit_cfg,
        71,
        cache_dir=tmp_path,
    )
    assert status == "miss_fitted"
    reward_changed = replace(
        dataset,
        rewards=np.arange(len(dataset), dtype=np.float64),
        metadata={**dataset.metadata, "reward_mode": "yield"},
    )
    second, status = load_or_fit_mechanistic_model(
        reward_changed,
        replace(public_context(), reward_mode="yield"),
        "ricker",
        model_cfg,
        fit_cfg,
        71,
        cache_dir=tmp_path,
    )
    assert status == "hit"
    assert first.transition_data_hash == second.transition_data_hash
    assert dataset_sha256(dataset) != dataset_sha256(reward_changed)
    assert first.model.parameter_hash() == second.model.parameter_hash()
    changed_observations = replace(dataset, observations=dataset.observations.copy() + 1.0)
    assert fit_transition_hash(changed_observations) != fit_transition_hash(dataset)


def test_sensitivity_cache_keys_reuse_only_registered_shared_fits():
    pytest.importorskip("torch")
    dataset = public_dataset(episodes=4, length=5)
    context = public_context()
    history, holdout = split_history_episodes(dataset, 0.8, 71)
    model_small = FaithfulModelConfig(candidates_per_form=3)
    model_large = FaithfulModelConfig(candidates_per_form=8)
    fit_paths8 = FaithfulFitConfig(starts=1, iterations=2, mc_paths=16, regime_mc_paths=8)
    fit_paths32 = replace(fit_paths8, regime_mc_paths=32)

    def key(form, model, fit):
        return fit_cache_key(
            dataset,
            context,
            form,
            model,
            fit,
            71,
            "candidate_00_00",
            history,
            holdout,
            0.9,
        )

    assert key("ricker", model_small, fit_paths8) == key("ricker", model_large, fit_paths32)
    assert key("regime", model_small, fit_paths8) != key("regime", model_large, fit_paths32)


def test_partial_or_wrong_schema_fit_cache_fails_loudly(tmp_path):
    pytest.importorskip("torch")
    dataset = public_dataset(episodes=4, length=5)
    args = (
        dataset,
        public_context(),
        "ricker",
        FaithfulModelConfig(),
        FaithfulFitConfig(starts=1, iterations=2, mc_paths=1),
        81,
    )
    result, _ = load_or_fit_mechanistic_model(*args, cache_dir=tmp_path)
    metadata = tmp_path / f"{result.fit_cache_key}.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["cache_schema"] = "void_provisional_cache"
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_or_fit_mechanistic_model(*args, cache_dir=tmp_path)


def test_only_adopted_external_method_ids_are_registered():
    assert "plus_adapted_mechanistic_pbvi" in METHODS
    assert "moor_adapted_ricker_misspec_pbvi" in METHODS
    assert "plus_faithful_pbvi" not in METHODS
    assert "moor_faithful_ricker_misspec_pbvi" not in METHODS


def test_registered_regime_candidate_allocations_are_exact_nested_prefixes():
    expected = {
        3: {0.8: 1, 0.9: 1, 0.97: 1},
        4: {0.8: 1, 0.9: 2, 0.97: 1},
        8: {0.8: 2, 0.9: 4, 0.97: 2},
    }
    for count, allocation in expected.items():
        schedule = _regime_schedule(count)
        assert len(schedule) == count
        assert {
            persistence: sum(item[0] == persistence for item in schedule)
            for persistence in allocation
        } == allocation
    assert _regime_schedule(8)[:4] == _regime_schedule(4)
    assert _regime_schedule(4)[:3] == _regime_schedule(3)
