#!/usr/bin/env python3
"""Non-gating hidden-r/K identifiability and public-signal diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark import realdata
from real_ecology_benchmark.config import MethodContext, load_config, real_environment_like
from real_ecology_benchmark.dataset import load_private
from real_ecology_benchmark.native_fit import PUBLIC_FORM_CANDIDATES, build_fitted_solver
from real_ecology_benchmark.pipeline import ensure_dataset
from real_ecology_benchmark.public_surrogate import (
    evaluator_only_surrogate_diagnostics,
    fit_public_surrogate,
)


DEFAULT_CELLS = (
    ("Amur tiger", "ricker", 0.0, "safe"),
    ("Iberian lynx", "allee", 0.2, "safe"),
)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def coefficients_sha256(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_default.yaml"))
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs" / "hidden_rk" / "phase0"),
    )
    args = parser.parse_args()
    root = Path(args.output_root)
    records = []
    for population, form, sigma, reward_mode in DEFAULT_CELLS:
        cfg = load_config(args.config)
        cfg.environment = real_environment_like(cfg.environment, population, form)
        cfg.environment = replace(
            cfg.environment,
            expose_rk="hidden",
            observation_noise_sigma=sigma,
            reward_mode=reward_mode,
        )
        cfg.dataset.transitions = 4000
        cfg.dataset.episode_length = 25
        cell = Path(f"reward_{reward_mode}") / slug(population) / form / f"sigma_{sigma:g}"
        cfg.dataset.output = str(root / "datasets" / cell / "public.npz")
        cfg.dataset.private_output = str(root / "private" / cell / "truth.npz")
        dataset = ensure_dataset(cfg, regenerate=False)
        surrogate = fit_public_surrogate(dataset, seed=cfg.seed + 20_000)
        context = MethodContext(
            num_actions=cfg.environment.num_actions,
            action_costs=tuple(float(value) for value in dataset.action_costs),
            action_channels=realdata.public_action_channels(
                cfg.environment.data_dir or realdata.DATA_DIR
            ),
            observation_noise_sigma=sigma,
            horizon=cfg.environment.horizon,
            observation_scale=surrogate.observation_scale,
            pop_id=str(dataset.pop_ids[0]),
            reward_mode=reward_mode,
            surrogate=surrogate,
        )
        fitted = [
            build_fitted_solver(dataset, context, candidate, cfg.model, cfg.planner)[0]
            for candidate in PUBLIC_FORM_CANDIDATES
        ]
        private = load_private(cfg.dataset.private_output)
        metadata = dataset.metadata
        record = {
            "population": population,
            "private_family": form,
            "sigma_obs": sigma,
            "reward_mode": reward_mode,
            "expose_rk": "hidden",
            "target_rows": metadata["target_rows"],
            "actual_rows": metadata["actual_rows"],
            "overshoot_rows": metadata["overshoot_rows"],
            "episode_count": metadata["episode_count"],
            "surrogate": surrogate.diagnostics,
            "evaluator_only_surrogate": evaluator_only_surrogate_diagnostics(
                surrogate, dataset, private
            ),
            "candidate_fit_loss": {
                candidate: float(model.fit_loss)
                for candidate, model in zip(PUBLIC_FORM_CANDIDATES, fitted)
            },
            "candidate_coefficients_sha256": coefficients_sha256(
                [model.coefficients for model in fitted]
            ),
            "interpretation": (
                "non-gating characterization only; do not retune, increase the "
                "4000-row target, add a public threshold, or drop safe-mode rows"
            ),
        }
        target = root / "reports" / cell.with_suffix(".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        records.append(record)
    summary = {"phase0": "non-gating", "cells": records}
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
