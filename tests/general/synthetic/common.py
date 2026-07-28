from __future__ import annotations

from dataclasses import replace
from functools import partial

from real_ecology_benchmark.beliefs import ParticleFilter, ReferenceProposal, cache_dataset_beliefs
from real_ecology_benchmark.collector import collect_dataset
from real_ecology_benchmark.config import (
    BenchmarkConfig,
    DatasetConfig,
    EvaluationConfig,
    FilterConfig,
    ModelConfig,
    PlannerConfig,
    synthetic_environment,
)
from real_ecology_benchmark.envs import make_env


def tiny_config(kind="allee", sigma=0.1, actions=5):
    return BenchmarkConfig(
        seed=116,
        environment=synthetic_environment(
            kind=kind,
            num_actions=actions,
            observation_noise_sigma=sigma,
            horizon=12,
        ),
        dataset=DatasetConfig(transitions=180, episode_length=10),
        filter=FilterConfig(particles=48, proposal="reference", proposal_sigma=0.12),
        model=ModelConfig(ensemble_size=3, ridge=1e-2),
        planner=PlannerConfig(horizon=3, sequences=12, particles=6, pessimism=0.1),
        evaluation=EvaluationConfig(
            seeds=[7001], episodes_per_seed=1, horizon=6,
            output_dir="/tmp/ecology_benchmark_test"
        ),
    )


def tiny_data(cfg=None):
    cfg = cfg or tiny_config()
    dataset, private = collect_dataset(
        make_env(cfg.environment), cfg.dataset.transitions, cfg.dataset.episode_length,
        cfg.seed, cfg.environment.privileged_behavior,
    )
    proposal = ReferenceProposal(cfg.environment, cfg.filter.proposal_sigma)
    factory = partial(ParticleFilter, cfg.environment, cfg.filter, proposal)
    cache = cache_dataset_beliefs(
        dataset, factory, 1234, cfg.environment.K_ref, cfg.environment.safety_threshold
    )
    return cfg, dataset, private, factory, cache
