#!/usr/bin/env python3
"""Return-blind adequacy + mechanism + runtime probe for the corrected general-RL baselines.

Computes ONLY model-internal diagnostics -- action/episode coverage, held-out
dynamics error, behavior likelihood, guardian support/risk prevalence, ensemble
diversity, BA-MCTS in-tree belief movement/tree statistics, and fitted-Q
bootstrap disagreement -- plus per-method fit/act timing and
peak memory.  It never reads ``operational_return`` / ``true_return`` / survival
return / rankings.  Writes a JSON to an isolated output directory.

Usage:
    python scripts/general_adequacy_probe.py --population "Amur tiger" \
        --transitions 4000 --episode 25 --act-steps 25 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark import realdata
from real_ecology_benchmark.beliefs import PublicObservationFilter, cache_public_beliefs
from real_ecology_benchmark.collector import collect_dataset
from real_ecology_benchmark.config import (
    FilterConfig, MethodContext, ModelConfig, PlannerConfig, real_environment,
)
from real_ecology_benchmark.behavior_model import behavior_nll, fit_reference_behavior
from real_ecology_benchmark.envs import make_env
from real_ecology_benchmark.methods.bamcts import BAMCTSPolicy
from real_ecology_benchmark.methods.ensemble_value_disagreement import (
    EnsembleValueDisagreementPolicy,
)
from real_ecology_benchmark.methods.ogsrl import OGSRLPolicy
from real_ecology_benchmark.methods.refplan import RefPlanPolicy
from real_ecology_benchmark.public_models import PublicDynamicsEnsemble
from real_ecology_benchmark.public_surrogate import fit_public_surrogate
from real_ecology_benchmark.training_monitor import _subset_cache


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _episode_split(dataset, holdout_fraction, seed):
    episodes = np.unique(dataset.episode_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    cut = max(1, min(len(episodes) - 1, int(round((1 - holdout_fraction) * len(episodes)))))
    train_ids, hold_ids = episodes[:cut], episodes[cut:]
    tr = np.isin(dataset.episode_id, train_ids)
    ho = np.isin(dataset.episode_id, hold_ids)
    return tr, ho


def probe(population: str, family: str, sigma: float, transitions: int, episode: int,
          act_steps: int, seed: int) -> dict:
    diag: dict = {"population_public": "opaque", "family_private_hidden": True}
    env = real_environment(population, family, expose_rk="hidden", observation_noise_sigma=sigma)
    t0 = time.perf_counter()
    dataset, private = collect_dataset(make_env(env), transitions, episode, seed=seed)
    diag["collect_seconds"] = time.perf_counter() - t0
    surrogate = fit_public_surrogate(dataset, seed=seed + 20_000)
    context = MethodContext(
        num_actions=env.num_actions,
        action_costs=tuple(float(v) for v in dataset.action_costs),
        action_channels=realdata.public_action_channels(),
        observation_noise_sigma=env.observation_noise_sigma,
        horizon=env.horizon,
        observation_scale=surrogate.observation_scale,
        pop_id=str(dataset.pop_ids[0]),
        reward_mode=env.reward_mode,
        surrogate=surrogate,
    )
    filter_cfg = FilterConfig(particles=64, proposal="learned")
    factory = lambda: PublicObservationFilter(context, filter_cfg)
    cache = cache_public_beliefs(dataset, factory, seed=seed + 30_000,
                                 observation_scale=context.observation_scale)

    # ---- shared / data adequacy (return-blind) ----
    K = env.num_actions
    tr, ho = _episode_split(dataset, 0.2, seed + 1)
    train_dataset = dataset.subset(tr)
    holdout_dataset = dataset.subset(ho)
    train_cache = _subset_cache(cache, tr)
    holdout_cache = _subset_cache(cache, ho)
    per_action = np.bincount(dataset.actions.astype(int), minlength=K)
    diag["data"] = {
        "transitions": int(len(dataset)),
        "episodes": int(len(np.unique(dataset.episode_id))),
        "train_episodes": int(len(np.unique(dataset.episode_id[tr]))),
        "holdout_episodes": int(len(np.unique(dataset.episode_id[ho]))),
        "transitions_per_action": per_action.tolist(),
        "min_action_support": int(per_action.min()),
        "actions_below_50": int((per_action < 50).sum()),
        "observation_scale": float(context.observation_scale),
    }
    # held-out dynamics RMSE (public ensemble)
    ens = PublicDynamicsEnsemble.fit(dataset, cache, context, ModelConfig(ensemble_size=5), seed)
    mean_pred, var_pred, _ = ens.predict(cache.mean_states[ho], dataset.actions[ho])
    rmse = float(np.sqrt(np.mean((mean_pred - cache.next_mean_states[ho]) ** 2)))
    diag["dynamics"] = {
        "holdout_rmse": rmse,
        "mean_ensemble_disagreement": float(np.mean(var_pred)),
    }
    ref_w = fit_reference_behavior(cache.features[tr], dataset.actions[tr], K)
    diag["behavior"] = {
        "reference_nll_holdout": float(
            behavior_nll(ref_w, cache.features[ho], dataset.actions[ho])
        )
    }
    diag["risk_channel"] = {
        "terminated_prevalence": float(surrogate.diagnostics.get("terminated_prevalence", 0.0)),
        "risk_fallback": surrogate.diagnostics.get("risk_fallback", "none"),
    }

    planner = PlannerConfig(horizon=5, sequences=96, particles=32, pessimism=0.5,
                            bamcts_depth=8, bamcts_simulations=256, ogsrl_cost_horizon=25)
    model = ModelConfig(ensemble_size=5)

    def time_method(policy):
        policy.training_holdout_dataset = holdout_dataset
        policy.training_holdout_beliefs = holdout_cache
        t = time.perf_counter()
        cpu0 = time.process_time()
        fit_diag = policy.fit(train_dataset, train_cache)
        fit_s = time.perf_counter() - t
        fit_cpu_s = time.process_time() - cpu0
        belief = factory().reset(float(dataset.observations[0]), seed + 5)
        act_t = []
        for step in range(act_steps):
            s = time.perf_counter()
            policy.act(belief, float(belief.observation))
            act_t.append(time.perf_counter() - s)
            belief = factory().update(belief, 0, float(dataset.next_observations[step % len(dataset)]))
        return fit_diag, fit_s, fit_cpu_s, np.asarray(act_t)

    methods = {}

    # RefPlan: posterior entropy movement across a short observe sequence
    rp = RefPlanPolicy(context, model, planner, seed=seed + 40_000)
    fitd, fits, fitcpu, actt = time_method(rp)
    ent0 = float(-np.sum(rp.posterior * np.log(rp.posterior + 1e-12)))
    methods["refplan"] = {
        "fit_seconds": fits, "fit_cpu_seconds": fitcpu,
        "act_ms_mean": float(actt.mean() * 1e3),
        "act_ms_p95": float(np.percentile(actt, 95) * 1e3),
        "posterior_entropy": ent0, "members": float(len(rp.dynamics.members)),
    }

    # OGSRL: ConOpt training + degeneracy report
    og = OGSRLPolicy(context, model, planner, seed=seed + 40_000)
    fitd, fits, fitcpu, actt = time_method(og)
    methods["ogsrl"] = {
        "fit_seconds": fits, "fit_cpu_seconds": fitcpu,
        "act_ms_mean": float(actt.mean() * 1e3),
        "act_ms_p95": float(np.percentile(actt, 95) * 1e3),
        "hidden_actor_trained": fitd.get("hidden_actor_trained"),
        "lambda_ood": fitd.get("lambda_ood"), "lambda_safety": fitd.get("lambda_safety"),
        "safety_channel_degenerate": fitd.get("safety_channel_degenerate"),
        "low_abundance_scale": fitd.get("low_abundance_scale"),
        "low_abundance_cost_prevalence": fitd.get("low_abundance_cost_prevalence"),
        "low_abundance_cost_mean": fitd.get("low_abundance_cost_mean"),
        "safety_budget": fitd.get("safety_budget"),
        "cost_horizon": fitd.get("cost_horizon"),
        "behavior_normalized_cost": fitd.get("behavior_normalized_cost"),
        "behavior_cost_episode_count": fitd.get("behavior_cost_episode_count"),
        "behavior_cost_standard_deviation": fitd.get("behavior_cost_standard_deviation"),
        "behavior_cost_standard_error": fitd.get("behavior_cost_standard_error"),
        "behavior_cost_ci95_low": fitd.get("behavior_cost_ci95_low"),
        "behavior_cost_ci95_high": fitd.get("behavior_cost_ci95_high"),
        "safety_dual_moved": fitd.get("safety_dual_moved"),
        "guardian_threshold": fitd.get("guardian_threshold"),
    }

    # BA-MCTS: tree statistics + in-tree belief movement
    bm = BAMCTSPolicy(context, model, planner, simulations=256, depth=8, seed=seed + 40_000)
    fitd, fits, fitcpu, actt = time_method(bm)
    # belief-movement diagnostic: KL(root -> updated) after one member-consistent step
    n = len(bm.dynamics.members)
    member = bm.dynamics.members[0]
    cur = float(np.median(dataset.observations[dataset.observations > 0]))
    foll = float(member.mean_next(np.asarray([cur]), np.asarray([0]))[0])
    lb = np.log(np.full(n, 1.0 / n)) + bm._public_member_loglik(cur, 0, foll)
    moved = bm._belief_from_logweights(lb)
    kl = float(np.sum(moved * (np.log(moved + 1e-12) - np.log(1.0 / n))))
    methods["bamcts"] = {
        "fit_seconds": fits, "fit_cpu_seconds": fitcpu,
        "act_ms_mean": float(actt.mean() * 1e3),
        "act_ms_p95": float(np.percentile(actt, 95) * 1e3),
        "tree_nodes_last": bm.last_diagnostics.get("tree_nodes"),
        "root_belief_kl_after_one_step": kl,
    }

    # Bootstrap Q-ensemble value-disagreement baseline.
    evd = EnsembleValueDisagreementPolicy(
        context, model, planner, ensemble_size=20, seed=seed + 40_000
    )
    fitd, fits, fitcpu, actt = time_method(evd)
    methods["ensemble_value_disagreement_pessimism"] = {
        "reader_label": evd.reader_label,
        "fit_seconds": fits, "fit_cpu_seconds": fitcpu,
        "act_ms_mean": float(actt.mean() * 1e3),
        "act_ms_p95": float(np.percentile(actt, 95) * 1e3),
        "ensemble_members": fitd.get("q_ensemble_members"),
        "bootstrap_unique_fraction_mean": fitd.get(
            "q_ensemble_bootstrap_unique_fraction_mean"
        ),
        "behavior_reference_nll_holdout": fitd.get("behavior_reference_nll_holdout"),
        "q_disagreement_observed": fitd.get("q_ensemble_disagreement_observed"),
        "q_disagreement_unobserved": fitd.get("q_ensemble_disagreement_unobserved"),
    }

    diag["methods"] = methods
    diag["peak_rss_mb"] = peak_rss_mb()
    diag["resources"] = {
        "threads": {
            name: os.environ.get(name, "unset")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "allocated_cores_per_task": 1,
        "task_hours": "report separately from measured CPU/core/wall hours",
    }
    return diag


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--population", default="Amur tiger")
    p.add_argument("--family", default="ricker")
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--transitions", type=int, default=4000)
    p.add_argument("--episode", type=int, default=25)
    p.add_argument("--act-steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=116)
    p.add_argument(
        "--out", default=str(ROOT / "real_ecology_runs" / "general_adequacy_phase2d")
    )
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = probe(args.population, args.family, args.sigma, args.transitions,
                   args.episode, args.act_steps, args.seed)
    result["wall_seconds_total"] = time.perf_counter() - t0
    tag = f"{args.population.replace(' ', '_')}_{args.family}_sig{args.sigma}_n{args.transitions}"
    path = out / f"adequacy_{tag}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
