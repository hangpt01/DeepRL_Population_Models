#!/usr/bin/env python3
"""H12 observation-scale dose response with demographic fit-cache reuse."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIAGNOSTICS_DIR))
from repo_paths import (  # noqa: E402
    ECOLOGICAL_SCRIPTS,
    ECOLOGICAL_SRC,
    FOLLOWUPS_OUTPUT,
    REPO_ROOT,
    require_scientific_tables,
)

BASE_DIR = REPO_ROOT / "scripts" / "diagnostics" / "replay"
OUT = Path(os.environ.get("DEEPRL_H12_OUTPUT", FOLLOWUPS_OUTPUT / "H12")) / "arms"
sys.path[:0] = [str(BASE_DIR), str(ECOLOGICAL_SCRIPTS), str(ECOLOGICAL_SRC)]

import run_diagnostic_replay as base  # noqa: E402
from real_ecology_benchmark.config import load_config  # noqa: E402
from real_ecology_benchmark.evaluator import ContinuousEvaluator  # noqa: E402
from real_ecology_benchmark.faithful_fit import (  # noqa: E402
    build_candidate_bank, load_or_fit_mechanistic_model,
)
from real_ecology_benchmark.faithful_pomdp import CandidatePOMDP  # noqa: E402
from real_ecology_benchmark.methods import METHODS  # noqa: E402
from real_ecology_benchmark.methods.moor_faithful import MOORFaithfulRickerPBVIPolicy  # noqa: E402
from real_ecology_benchmark.pipeline import (  # noqa: E402
    ensure_dataset, _load_or_fit_public_surrogate, _hidden_method_context,
    make_filter_factory,
)
from real_ecology_benchmark.planners.pbvi import PointBasedPlanner  # noqa: E402
from real_ecology_benchmark.public_surrogate import (  # noqa: E402
    PublicFeatureSpec, PublicRewardRiskSurrogate, _episode_split,
)

CELLS = {
    "A1": ("Crab-eating fox", "ricker", "0.2"),
    "A2": ("Crab-eating fox", "allee", "0.2"),
    "A3": ("Crab-eating fox", "regime", "0.2"),
    "A4": ("Crab-eating fox", "theta", "0.2"),
    "B2": ("Amur tiger", "ricker", "0.1"),
    "B3": ("Amur tiger", "allee", "0.2"),
}
MULTIPLIERS = (0.5, 0.7, 0.85, 1.2, 1.5, 2.0, 3.0)


def scaled_surrogate(dataset, original, target_scale):
    """Refit only the public reward surrogate using the requested basis scale."""
    fit_mask, holdout = _episode_split(dataset, original.split_seed)
    fitted = PublicFeatureSpec.fit(dataset, fit_mask)
    provisional = replace(
        fitted, observation_scale=float(target_scale),
        continuous_mean=np.zeros(7), continuous_std=np.ones(7),
    )
    continuous = provisional.continuous(dataset)
    spec = replace(
        provisional,
        continuous_mean=continuous[fit_mask].mean(axis=0),
        continuous_std=np.maximum(continuous[fit_mask].std(axis=0), 1e-8),
    )
    design = spec.design(dataset)
    X = design[fit_mask]
    penalty = 1e-3 * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        X.T @ X + penalty, X.T @ dataset.rewards[fit_mask])
    pred = design @ coefficients
    diagnostics = {
        "dose_response_public_surrogate_refit": True,
        "fit_rows": int(fit_mask.sum()), "holdout_rows": int(holdout.sum()),
        "reward_holdout_R2": float(1 - np.sum((dataset.rewards[holdout] - pred[holdout]) ** 2) /
                                   max(np.sum((dataset.rewards[holdout] -
                                              dataset.rewards[holdout].mean()) ** 2), 1e-300)),
        "observation_scale": float(target_scale),
    }
    return PublicRewardRiskSurrogate(
        spec, coefficients, None,
        float((np.sum(dataset.terminated[fit_mask]) + 1) / (fit_mask.sum() + 2)),
        np.asarray(dataset.action_costs), diagnostics,
        original.public_data_hash, original.split_seed,
    )


def build_replanned(cell, side, arm):
    pop, family, sigma = CELLS[cell]
    row = base.manifest_row(side, pop, family, sigma)
    cfg = load_config(str(base.P10_PACKAGE / "configs" / base.CONFIG[side]))
    cfg, _, _ = base.rrmr.apply_row_config(cfg, row, base.P10_DATA / side)
    cfg.validate()
    dataset = ensure_dataset(cfg, regenerate=False)
    original_surrogate, _, status = _load_or_fit_public_surrogate(cfg, dataset)
    if status != "loaded":
        raise AssertionError("public surrogate cache unexpectedly missed")
    original_context = _hidden_method_context(cfg, dataset, original_surrogate)
    original_scale = float(original_context.observation_scale)
    multiplier = (float(cfg.environment.K_ref) / original_scale
                  if arm == "kref" else float(arm))
    target_scale = original_scale * multiplier
    surrogate = scaled_surrogate(dataset, original_surrogate, target_scale)
    context = replace(original_context, observation_scale=target_scale,
                      surrogate=surrogate)
    context.validate()
    filter_factory, _ = make_filter_factory(
        cfg, dataset, "faithful_internal", context)
    policy_seed = cfg.seed + 40_000

    started = time.perf_counter()
    if side == "plus":
        bank, statuses = build_candidate_bank(
            dataset, original_context, cfg.faithful.model, cfg.faithful.fit,
            policy_seed + 11_000, cfg.faithful.fit_cache_dir)
        # build_candidate_bank returns an immutable tuple in the frozen runtime.
        # Coerce only the container type; every entry must still be exactly "hit".
        if tuple(statuses) != ("hit",) * len(statuses):
            raise AssertionError(f"demographic cache miss: {statuses}")
        models = [replace(fit.model, survey_scale=target_scale) for fit in bank.fits]
        # Route through the exact accepted method registration used by the
        # parity-validated m=1 replay (the ricker-only PLUS subclass).
        policy = METHODS[base.METHOD_ID["plus"]](
            context, cfg.model, cfg.planner, seed=policy_seed,
            faithful_cfg=cfg.faithful)
        policy.candidate_bank = bank
        policy.fit_cache_statuses = statuses
        policy.pomdps = [
            CandidatePOMDP(model, context, cfg.faithful.planner,
                           policy_seed + 12_000 + i)
            for i, model in enumerate(models)
        ]
        policy.planners = [
            PointBasedPlanner(pomdp, cfg.faithful.planner, cfg.planner.discount,
                              policy_seed + 13_000 + i)
            for i, pomdp in enumerate(policy.pomdps)
        ]
        policy.posterior = bank.initial_weights.copy()
        fit_count = len(statuses)
    else:
        fit, status_fit = load_or_fit_mechanistic_model(
            dataset, original_context, "ricker", cfg.faithful.model,
            cfg.faithful.fit, policy_seed + 7_000,
            candidate_id="moor_ricker_00",
            cache_dir=cfg.faithful.fit_cache_dir)
        if status_fit != "hit":
            raise AssertionError(f"demographic cache miss: {status_fit}")
        model = replace(fit.model, survey_scale=target_scale)
        policy = MOORFaithfulRickerPBVIPolicy(
            context, cfg.model, cfg.planner, seed=policy_seed,
            faithful_cfg=cfg.faithful)
        policy.fit_result, policy.fit_cache_status = fit, status_fit
        policy.pomdp = CandidatePOMDP(
            model, context, cfg.faithful.planner, policy_seed + 8_000)
        policy.planner = PointBasedPlanner(
            policy.pomdp, cfg.faithful.planner, cfg.planner.discount,
            policy_seed + 9_000)
        fit_count = 1
    plan_seconds = time.perf_counter() - started
    return cfg, filter_factory, policy, {
        "population": pop, "environment": family, "sigma_obs": sigma,
        "original_observation_scale": original_scale,
        "multiplier": multiplier, "target_observation_scale": target_scale,
        "surrogate_holdout_R2": surrogate.diagnostics["reward_holdout_R2"],
        "cached_demographic_models": fit_count, "plan_seconds": plan_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=tuple(CELLS))
    parser.add_argument("side", choices=("plus", "moor"))
    parser.add_argument("arm", choices=tuple(str(x) for x in MULTIPLIERS) + ("kref",))
    args = parser.parse_args()
    require_scientific_tables()
    if base.sha(base.ACCEPTED_CSV) != base.ACCEPTED_CSV_SHA:
        raise AssertionError("accepted controlling CSV hash changed")
    cfg, factory, policy, metadata = build_replanned(
        args.cell, args.side, args.arm)
    started = time.perf_counter()
    rows = ContinuousEvaluator(cfg, factory, "faithful_internal").run(policy)
    eval_seconds = time.perf_counter() - started
    summary = base.summarize7(rows)
    accepted = base.accepted_fields(
        metadata["population"], metadata["environment"],
        metadata["sigma_obs"], base.METHOD_ID[args.side])
    result = {
        "schema": "H12_dose_response_arm_v1", "cell": args.cell,
        "method": args.side, "arm": args.arm, **metadata,
        "summary": summary,
        "accepted_m1_return_mean": accepted["return_mean"],
        "return_degradation_from_m1": summary["return_mean"] - accepted["return_mean"],
        "eval_seconds": eval_seconds, "recomputed_fits": 0,
        "public_surrogate_refit": True,
        "demographic_fit_cache_only": True,
        "channels": "belief_abundance_mapping_and_surrogate_basis",
        "belief_only_variant_run": False,
        "belief_only_variant_reason": (
            "Channels are separately rewired in code; the requested coupled arm "
            "does not accidentally conflate an unseparable implementation."),
        "accepted_artifacts_modified": False, "reranked": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    label = f"{args.cell}__{args.side}__m_{args.arm.replace('.', 'p')}.json"
    (OUT / label).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
