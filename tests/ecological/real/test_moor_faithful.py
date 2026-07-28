from dataclasses import replace

import numpy as np
import pytest

from real_ecology_benchmark.config import FaithfulFitConfig, FaithfulModelConfig
from real_ecology_benchmark.dataset import (
    PUBLIC_CONTROL_FIELDS,
    PUBLIC_FIELDS,
    PUBLIC_SANITIZED_FIELDS,
)
from real_ecology_benchmark.faithful_fit import fit_mechanistic_model, ordered_trajectory_sse

from .faithful_fixtures import public_context, public_dataset, ricker_model


def test_ordered_objective_uses_within_episode_order():
    dataset = public_dataset()
    baseline = ordered_trajectory_sse(ricker_model(), dataset)
    reversed_dataset = dataset.subset(np.ones(len(dataset), dtype=bool))
    for indices in reversed_dataset.episode_indices():
        reversed_dataset.actions[indices] = reversed_dataset.actions[indices][::-1]
        reversed_dataset.next_observations[indices] = reversed_dataset.next_observations[indices][
            ::-1
        ]
    assert not np.isclose(ordered_trajectory_sse(ricker_model(), reversed_dataset), baseline)


def test_ordered_objective_is_invariant_to_whole_episode_permutation():
    dataset = public_dataset()
    baseline = ordered_trajectory_sse(ricker_model(), dataset)
    blocks = dataset.episode_indices()[::-1]
    order = np.concatenate(blocks)
    row_fields = PUBLIC_FIELDS + PUBLIC_SANITIZED_FIELDS + PUBLIC_CONTROL_FIELDS
    replacements = {
        name: getattr(dataset, name)[order].copy()
        for name in row_fields
        if name != "episode_id" and getattr(dataset, name) is not None
    }
    replacements["episode_id"] = np.concatenate(
        [
            np.full(len(indices), new_id, dtype=dataset.episode_id.dtype)
            for new_id, indices in enumerate(blocks)
        ]
    )
    replacements["metadata"] = {**dataset.metadata, "whole_episode_permutation": True}
    permuted = replace(dataset, **replacements)
    permuted.validate()

    assert np.isclose(
        ordered_trajectory_sse(ricker_model(), permuted),
        baseline,
        rtol=0.0,
        atol=1e-12,
    )


def test_autodiff_fit_selects_minimum_finite_start():
    pytest.importorskip("torch")
    result = fit_mechanistic_model(
        public_dataset(),
        public_context(),
        "ricker",
        FaithfulModelConfig(),
        FaithfulFitConfig(starts=2, iterations=4, mc_paths=2),
        seed=29,
    )
    assert np.isfinite(result.objective)
    assert result.objective == min(result.start_objectives)
    assert result.model.form == "ricker"
