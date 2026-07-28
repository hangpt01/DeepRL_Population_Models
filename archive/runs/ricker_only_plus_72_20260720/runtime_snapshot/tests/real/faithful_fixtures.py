"""Small public-only fixtures for paper-faithful ecology tests."""

from __future__ import annotations

import numpy as np

from real_ecology_benchmark.config import MethodContext
from real_ecology_benchmark.dataset import TrajectoryDataset
from real_ecology_benchmark.faithful_ecology import MechanisticModel


def public_context(num_actions: int = 3) -> MethodContext:
    return MethodContext(
        num_actions=num_actions,
        action_costs=tuple(float(x) for x in np.linspace(0.0, 0.2, num_actions)),
        action_channels=tuple(
            ("none", "rate+capacity", "state")[index]
            if index < 3
            else "rate"
            for index in range(num_actions)
        ),
        observation_noise_sigma=0.05,
        horizon=8,
        observation_scale=100.0,
        pop_id="pop_0123456789abcdef",
        reward_mode="safe",
        surrogate=None,
    )


def ricker_model(num_actions: int = 3) -> MechanisticModel:
    return MechanisticModel(
        form="ricker",
        growth=np.asarray([0.28, 0.12, 0.28])[:num_actions],
        mortality=np.zeros(3)[:num_actions],
        capacity_increment=np.asarray([0.0, 0.03, 0.0])[:num_actions],
        stocking=np.asarray([0.0, 0.0, 0.05])[:num_actions],
        reset_log_mean=np.log(0.65),
        reset_log_scale=0.08,
        initial_capacity=1.0,
        capacity_ceiling=1.8,
        process_scale=0.03,
        observation_scale=0.05,
        survey_scale=100.0,
        depensation_thresholds=np.asarray([0.2, 0.35]),
        theta_exponent=2.0,
        regime_multipliers=np.asarray([0.7, 1.2]),
        regime_matrix=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        action_channels=tuple(("none", "rate+capacity", "state")[:num_actions]),
        candidate_id="fixture",
    )


def public_dataset(episodes: int = 6, length: int = 8) -> TrajectoryDataset:
    model = ricker_model()
    observations = []
    next_observations = []
    actions = []
    episode_id = []
    timesteps = []
    dones = []
    rng = np.random.default_rng(941)
    for episode in range(episodes):
        latent = float(np.exp(model.reset_log_mean + rng.normal(0.0, model.reset_log_scale)))
        capacity = model.initial_capacity
        for timestep in range(length):
            action = (episode + timestep) % model.num_actions
            observation = latent * model.survey_scale * np.exp(rng.normal(0.0, 0.02))
            latent = float(model.noiseless_next(latent, capacity, action))
            capacity = model.next_capacity(capacity, action)
            following = latent * model.survey_scale * np.exp(rng.normal(0.0, 0.02))
            observations.append(observation)
            next_observations.append(following)
            actions.append(action)
            episode_id.append(episode)
            timesteps.append(timestep)
            dones.append(timestep == length - 1)
    count = len(actions)
    actions_array = np.asarray(actions, dtype=np.int64)
    action_costs = np.asarray([0.0, 0.1, 0.2])
    dataset = TrajectoryDataset(
        observations=np.asarray(observations),
        actions=actions_array,
        rewards=np.zeros(count),
        next_observations=np.asarray(next_observations),
        dones=np.asarray(dones),
        episode_id=np.asarray(episode_id, dtype=np.int32),
        timestep=np.asarray(timesteps, dtype=np.int32),
        metadata={
            "expose_rk": "hidden",
            "num_actions": 3,
            "observation_noise_sigma": 0.05,
            "reward_mode": "safe",
            "horizon": length,
            "regime_label": "hidden-demographics_structure-unknown",
        },
        costs=action_costs[actions_array],
        pop_ids=np.full(count, "pop_0123456789abcdef"),
        terminated=np.zeros(count, dtype=bool),
        truncated=np.asarray(dones),
        action_costs=action_costs,
    )
    dataset.validate()
    return dataset
