from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from real_ecology_benchmark.config import FaithfulConfig, FaithfulFitConfig, FaithfulModelConfig
from real_ecology_benchmark.faithful_fit import fit_mechanistic_model
from real_ecology_benchmark.privacy import assert_hidden_method_artifact

from .faithful_fixtures import public_context, public_dataset


def test_faithful_config_has_no_private_demographic_fields():
    assert_hidden_method_artifact(FaithfulConfig(), root="faithful_config")


def test_faithful_modules_do_not_import_table_or_evaluator_modules():
    root = Path(__file__).resolve().parents[2] / "src" / "real_ecology_benchmark"
    for name in ("faithful_ecology.py", "faithful_fit.py", "faithful_pomdp.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "realdata" not in text
        assert "native_solver" not in text
        assert "EnvironmentConfig" not in text
        assert "load_private" not in text


def test_fit_is_table_independent_but_public_data_dependent():
    pytest.importorskip("torch")
    dataset = public_dataset(episodes=4, length=5)
    fit_config = FaithfulFitConfig(starts=1, iterations=3, mc_paths=1)
    model_config = FaithfulModelConfig()
    with patch("real_ecology_benchmark.realdata.pops_for", side_effect=AssertionError), \
         patch("real_ecology_benchmark.realdata.load_pops", side_effect=AssertionError), \
         patch("real_ecology_benchmark.realdata.load_actions", side_effect=AssertionError), \
         patch("real_ecology_benchmark.realdata.load_effects", side_effect=AssertionError), \
         patch("real_ecology_benchmark.actions.resolve_actions", side_effect=AssertionError), \
         patch("real_ecology_benchmark.native_solver.NativeSolver.build", side_effect=AssertionError):
        baseline = fit_mechanistic_model(
            dataset, public_context(), "ricker", model_config, fit_config, seed=71
        )
    assert_hidden_method_artifact(baseline, root="faithful_fit")
    changed = dataset.subset(np.ones(len(dataset), dtype=bool))
    changed.next_observations *= 1.15
    perturbed = fit_mechanistic_model(
        changed, public_context(), "ricker", model_config, fit_config, seed=71
    )
    assert baseline.model.parameter_hash() != perturbed.model.parameter_hash()
