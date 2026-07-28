#!/usr/bin/env python3
"""Small deterministic fit/kernel/PBVI probe for launch thread equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from real_ecology_benchmark.config import (
    FaithfulFitConfig,
    FaithfulModelConfig,
    FaithfulPlannerConfig,
)
from real_ecology_benchmark.faithful_fit import _model_vector, fit_mechanistic_model
from real_ecology_benchmark.faithful_pomdp import CandidatePOMDP
from real_ecology_benchmark.planners.pbvi import PointBasedPlanner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/real"))
from faithful_fixtures import public_context, public_dataset  # noqa: E402


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    value = str(args.threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    dataset = public_dataset(episodes=4, length=5)
    context = public_context()
    fit = fit_mechanistic_model(
        dataset,
        context,
        "ricker",
        FaithfulModelConfig(),
        FaithfulFitConfig(starts=2, iterations=5, mc_paths=2),
        seed=37,
    )
    planner_cfg = FaithfulPlannerConfig(
        state_bins=9,
        capacity_bins=3,
        observation_bins=9,
        transition_samples=8,
        observation_samples=8,
        belief_points=4,
        observation_branches=3,
        horizon=2,
    )
    pomdp = CandidatePOMDP(fit.model, context, planner_cfg, seed=11)
    kernels = np.asarray(
        [
            pomdp.transition_matrix(fit.model.initial_capacity, action)
            for action in range(context.num_actions)
        ]
    )
    belief = pomdp.initial_belief(60.0)
    action_values = PointBasedPlanner(pomdp, planner_cfg, 0.95, seed=17).action_values(belief)
    parameters = _model_vector(fit.model)
    print(
        json.dumps(
            {
                "threads": args.threads,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "objective": fit.objective,
                "parameters": parameters.tolist(),
                "parameter_hash": fit.model.parameter_hash(),
                "kernel_hash": array_hash(kernels),
                "kernels": kernels.tolist(),
                "pbvi_action_values": action_values.tolist(),
                "pbvi_action_values_hash": array_hash(action_values),
                "pbvi_action": int(np.argmax(action_values)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
