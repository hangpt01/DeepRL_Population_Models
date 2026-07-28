"""End-to-end construction of data, filter, beliefs, methods, and evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from .backend import get_active_backend, resolve_backend_for_workload, set_active_backend
from .beliefs import (
    BeliefCache,
    DiscreteGridFilter,
    LearnedLinearProposal,
    MechanisticProposal,
    OracleStateFilter,
    ParticleFilter,
    RawObservationFilter,
    ReferenceProposal,
    PublicBeliefCache,
    PublicObservationFilter,
    cache_dataset_beliefs,
    cache_public_beliefs,
)
from .collector import collect_dataset
from .config import (
    BenchmarkConfig,
    MethodContext,
    hides_rk,
    is_setpoint_cumulative,
    normalize_control_mode,
)
from .dataset import (
    dataset_sha256,
    load_private,
    load_public,
    save_private,
    save_public,
)
from .evaluator import ContinuousEvaluator
from .methods import FAITHFUL_METHODS, METHODS, NATIVE_METHODS, BasePolicy
from .public_surrogate import (
    PublicRewardRiskSurrogate,
    SURROGATE_VERSION,
    evaluator_only_surrogate_diagnostics,
    fit_public_surrogate,
)
from .telemetry import elapsed_since, peak_rss_mb, perf_seconds
from .training_monitor import (
    attach_training_history,
    record_final_fit_metrics,
    save_training_artifacts,
    split_train_holdout,
)


def _reward_mode_output_root(cfg: BenchmarkConfig) -> Path:
    """Evaluation output root, namespaced by reward_mode for real cells.

    A separate agent is trained per reward_mode (E6'); putting the mode in the
    default output path prevents a ``safe`` run from overwriting a ``yield`` run
    when both share ``evaluation.output_dir``.  (The other cell dimensions are
    expected to be disambiguated by the caller's per-cell ``output_dir``, as in
    the manifest runners.)
    """

    root = Path(cfg.evaluation.output_dir)
    if is_setpoint_cumulative(cfg.environment):
        # Namespace by effective backend too, so CPU and GPU rows never clobber
        # each other (fairness rule) and can be aggregated as separate tables.
        backend = get_active_backend().name
        return (
            root
            / f"data_{cfg.environment.data_mode}"
            / f"backend_{backend}"
            / f"regime_{cfg.environment.expose_rk}"
            / f"reward_{cfg.environment.reward_mode}"
        )
    return root


def _validate_dataset_cell(cfg: BenchmarkConfig, dataset) -> None:
    if hides_rk(cfg.environment):
        expected = {
            "expose_rk": "hidden",
            "num_actions": cfg.environment.num_actions,
            "observation_noise_sigma": cfg.environment.observation_noise_sigma,
            "reward_mode": cfg.environment.reward_mode,
            "horizon": cfg.environment.horizon,
        }
        mismatch = {
            key: (dataset.metadata.get(key), value)
            for key, value in expected.items()
            if dataset.metadata.get(key) != value
        }
        if mismatch:
            raise ValueError(
                "cached hidden dataset does not match the requested cell; use a "
                f"regime-specific path or --regenerate. Mismatches: {mismatch}"
            )
        dataset.validate()
        return
    recorded = dataset.metadata.get("environment", {})
    expected = cfg.environment.__dict__
    keys = (
        "kind", "population", "num_actions", "control_mode", "data_mode",
        "observation_noise_sigma", "process_noise_sigma", "safety_threshold",
        "K_base", "r_base_low", "r_base_high",
        # Reward-affecting fields: the logged dataset.rewards depend on these, so a
        # cache from a different reward setting must not be silently reused.
        "reward_mode", "safety_penalty_mode", "collapse_penalty", "alpha",
        # Reporting/feature fields: stale diagnostics and belief caches are
        # misleading if these floors change.
        "mvp_threshold",
    )
    def recorded_value(key):
        if key == "control_mode":
            return normalize_control_mode(recorded.get(key, "tier2_one_step"))
        if key == "data_mode":
            if key in recorded:
                return recorded[key]
            return "real" if is_setpoint_cumulative(recorded_value("control_mode")) else "synthetic"
        return recorded.get(key)

    def expected_value(key):
        if key == "control_mode":
            return normalize_control_mode(expected.get(key, "tier2_one_step"))
        return expected.get(key)

    mismatch = {key: (recorded_value(key), expected_value(key)) for key in keys
                if recorded_value(key) != expected_value(key)}
    if mismatch:
        raise ValueError(
            "cached dataset does not match the requested cell; use a cell-specific "
            f"path or --regenerate. Mismatches: {mismatch}"
        )


def ensure_dataset(cfg: BenchmarkConfig, regenerate: bool = False):
    public_path = Path(cfg.dataset.output)
    private_path = Path(cfg.dataset.private_output)
    if not regenerate and public_path.exists() and private_path.exists():
        dataset = load_public(public_path)
        _validate_dataset_cell(cfg, dataset)
        return dataset
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = public_path.with_suffix(public_path.suffix + ".lock")
    acquired = False
    deadline = time.monotonic() + 3600.0
    while not acquired:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            acquired = True
        except FileExistsError:
            if public_path.exists() and private_path.exists() and not regenerate:
                dataset = load_public(public_path)
                _validate_dataset_cell(cfg, dataset)
                return dataset
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for dataset lock {lock_path}")
            time.sleep(1.0)
    try:
        if regenerate or not (public_path.exists() and private_path.exists()):
            from .envs import make_env
            dataset, private = collect_dataset(
                make_env(cfg.environment), cfg.dataset.transitions,
                cfg.dataset.episode_length, cfg.seed, cfg.environment.privileged_behavior,
            )
            temporary_public = public_path.with_name(
                f".{public_path.stem}.{os.getpid()}.tmp.npz"
            )
            temporary_private = private_path.with_name(
                f".{private_path.stem}.{os.getpid()}.tmp.npz"
            )
            save_public(temporary_public, dataset)
            save_private(temporary_private, private)
            os.replace(temporary_public, public_path)
            os.replace(temporary_private, private_path)
    finally:
        lock_path.unlink(missing_ok=True)
    dataset = load_public(public_path)
    _validate_dataset_cell(cfg, dataset)
    return dataset


def _load_or_fit_public_surrogate(
    cfg: BenchmarkConfig,
    dataset,
) -> tuple[PublicRewardRiskSurrogate, Path, str]:
    """Return the one shared public surrogate for a hidden-regime dataset."""

    public_hash = dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset)
    source = Path(cfg.dataset.output)
    split_seed = cfg.seed + 20_000
    path = source.with_name(
        f"{source.stem}.regime_hidden.public_surrogate.v{SURROGATE_VERSION}.seed{split_seed}.npz"
    )
    if path.exists():
        surrogate = PublicRewardRiskSurrogate.load(path)
        if surrogate.public_data_hash == public_hash:
            return surrogate, path, "loaded"
    surrogate = fit_public_surrogate(dataset, seed=split_seed)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    surrogate.save(temporary)
    os.replace(temporary, path)
    return surrogate, path, "fitted"


def _hidden_method_context(
    cfg: BenchmarkConfig,
    dataset,
    surrogate: PublicRewardRiskSurrogate,
) -> MethodContext:
    if not hides_rk(cfg.environment):
        raise ValueError("hidden MethodContext requested for expose_rk='full'")
    if dataset.action_costs is None or dataset.pop_ids is None:
        raise ValueError("hidden dataset is missing mandatory public costs/pop_id")
    population_tokens = sorted(str(value) for value in set(dataset.pop_ids.tolist()))
    if len(population_tokens) != 1:
        raise ValueError("one benchmark cell must contain exactly one opaque pop_id")
    context = MethodContext(
        num_actions=int(cfg.environment.num_actions),
        action_costs=tuple(float(value) for value in dataset.action_costs),
        observation_noise_sigma=float(dataset.metadata["observation_noise_sigma"]),
        horizon=int(dataset.metadata["horizon"]),
        observation_scale=float(surrogate.observation_scale),
        pop_id=population_tokens[0],
        reward_mode=str(dataset.metadata["reward_mode"]),
        surrogate=surrogate,
    )
    context.validate()
    return context


def make_filter_factory(
    cfg: BenchmarkConfig,
    dataset,
    mode: str | None = None,
    method_context: MethodContext | None = None,
):
    selected = mode or cfg.filter.proposal
    if hides_rk(cfg.environment):
        if method_context is None:
            raise ValueError("hidden filter construction requires MethodContext")
        forbidden = {"reference", "ricker", "true_family", "oracle"}
        if selected in forbidden:
            raise ValueError(
                f"filter={selected!r} is not method-visible when expose_rk='hidden'"
            )
        if selected not in {"raw", "learned", "native_discrete", "faithful_internal"}:
            raise ValueError(f"unknown hidden filter mode {selected!r}")
        return (
            lambda: PublicObservationFilter(method_context, cfg.filter),
            None,
        )
    if selected == "reference":
        proposal = ReferenceProposal(cfg.environment, cfg.filter.proposal_sigma)
        factory = lambda: ParticleFilter(cfg.environment, cfg.filter, proposal)
    elif selected == "raw":
        proposal = ReferenceProposal(cfg.environment, cfg.filter.proposal_sigma)
        factory = lambda: RawObservationFilter(cfg.environment, cfg.filter, proposal)
    elif selected == "learned":
        proposal_path = Path(cfg.dataset.output).with_name(
            Path(cfg.dataset.output).stem + ".learned_filter.npz"
        )
        proposal = None
        if proposal_path.exists():
            loaded = LearnedLinearProposal.load(proposal_path)
            if loaded.is_compatible_with(cfg.environment):
                proposal = loaded
        if proposal is None:
            proposal = LearnedLinearProposal.fit(
                dataset, cfg.environment, ridge=cfg.model.ridge
            )
            temporary = proposal_path.with_name(
                f".{proposal_path.stem}.{os.getpid()}.tmp.npz"
            )
            proposal.save(temporary)
            try:
                os.replace(temporary, proposal_path)
            except FileNotFoundError:
                pass
        factory = lambda: ParticleFilter(cfg.environment, cfg.filter, proposal)
    elif selected == "ricker":
        proposal = MechanisticProposal(cfg.environment, "ricker")
        factory = lambda: ParticleFilter(cfg.environment, cfg.filter, proposal)
    elif selected == "true_family":
        proposal = MechanisticProposal(cfg.environment, cfg.environment.kind)
        factory = lambda: ParticleFilter(cfg.environment, cfg.filter, proposal)
    elif selected == "oracle":
        proposal = ReferenceProposal(cfg.environment, cfg.filter.proposal_sigma)
        factory = lambda: OracleStateFilter(cfg.environment, cfg.filter, proposal)
    elif selected == "native_discrete":
        from .native_solver import NativeSolver

        proposal = NativeSolver.build(
            cfg.environment,
            state_bins=cfg.model.native_state_bins,
            discount=cfg.planner.discount,
            iterations=cfg.model.native_vi_iterations,
            tolerance=cfg.model.native_vi_tolerance,
        )
        factory = lambda: DiscreteGridFilter(cfg.environment, proposal)
    else:
        raise ValueError(f"unknown filter mode {selected!r}")
    return factory, proposal


def build_method(
    method: str,
    cfg: BenchmarkConfig,
    dataset,
    filter_factory,
    cache: BeliefCache | None = None,
    timings: dict[str, float] | None = None,
    holdout_dataset=None,
    holdout_cache: BeliefCache | None = None,
    split_info: dict[str, object] | None = None,
    method_context: MethodContext | None = None,
) -> tuple[BasePolicy, object]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method}; choose from {sorted(METHODS)}")
    if cache is None and method not in FAITHFUL_METHODS:
        start = perf_seconds()
        if hides_rk(cfg.environment):
            if method_context is None:
                raise ValueError("hidden policy construction requires MethodContext")
            cache = cache_public_beliefs(
                dataset,
                filter_factory,
                cfg.seed + 30_000,
                method_context.observation_scale,
            )
        else:
            cache = cache_dataset_beliefs(
                dataset,
                filter_factory,
                cfg.seed + 30_000,
                cfg.environment.K_ref,
                cfg.environment.safety_threshold,
            )
        if timings is not None:
            timings["belief_cache_seconds"] = (
                timings.get("belief_cache_seconds", 0.0) + elapsed_since(start)
            )
    start = perf_seconds()
    policy_config = method_context if hides_rk(cfg.environment) else cfg.environment
    if policy_config is None:
        raise ValueError("hidden policy construction requires MethodContext")
    policy_kwargs = {"seed": cfg.seed + 40_000}
    if method in FAITHFUL_METHODS:
        policy_kwargs["faithful_cfg"] = cfg.faithful
    policy = METHODS[method](policy_config, cfg.model, cfg.planner, **policy_kwargs)
    attach_training_history(
        policy,
        method,
        cfg,
        split_info or {},
        holdout_dataset,
        holdout_cache,
    )
    if timings is not None:
        timings["policy_init_seconds"] = elapsed_since(start)
    start = perf_seconds()
    policy.fit_diagnostics = policy.fit(dataset, cache)
    if getattr(policy, "training_history", None) is not None:
        policy.log_training(-1, "train", policy.fit_diagnostics, phase="fit_diagnostics")
    record_final_fit_metrics(policy, dataset, cache, holdout_dataset, holdout_cache)
    if timings is not None:
        timings["fit_seconds"] = elapsed_since(start)
    return policy, cache


def run_method(
    method: str,
    cfg: BenchmarkConfig,
    filter_mode: str | None = None,
    regenerate: bool = False,
):
    row_start = perf_seconds()
    selected_filter = filter_mode or cfg.filter.proposal
    if method in NATIVE_METHODS and selected_filter != "native_discrete":
        raise ValueError(
            f"{method} requires filter='native_discrete', got {selected_filter!r}"
        )
    if method in FAITHFUL_METHODS:
        if selected_filter != "faithful_internal":
            raise ValueError(
                f"{method} requires filter='faithful_internal', got {selected_filter!r}"
            )
        if not hides_rk(cfg.environment):
            raise ValueError(f"{method} is registered only for expose_rk='hidden'")
    # Resolve + activate the compute backend once for this run (numpy CPU default;
    # cupy GPU when requested and available).  Fails loud in strict mode.
    backend = set_active_backend(
        resolve_backend_for_workload(cfg.compute, f"method:{method}", cfg.environment)
    )
    timings: dict[str, float] = {}
    start = perf_seconds()
    dataset = ensure_dataset(cfg, regenerate)
    timings["dataset_seconds"] = elapsed_since(start)
    surrogate = None
    surrogate_path = None
    surrogate_cache_status = "not_applicable"
    method_context = None
    if hides_rk(cfg.environment):
        start = perf_seconds()
        surrogate, surrogate_path, surrogate_cache_status = _load_or_fit_public_surrogate(
            cfg, dataset
        )
        method_context = _hidden_method_context(cfg, dataset, surrogate)
        timings["surrogate_seconds"] = elapsed_since(start)
    start = perf_seconds()
    factory, _proposal = make_filter_factory(
        cfg, dataset, selected_filter, method_context
    )
    timings["filter_factory_seconds"] = elapsed_since(start)
    # The native belief depends on the discretization, so the cache key carries the
    # grid size: a coarse-vs-fine resolution check shares one dataset path and would
    # otherwise silently reuse the other resolution's cached beliefs.
    cache_key = f"regime_{cfg.environment.expose_rk}.{selected_filter}"
    if selected_filter == "native_discrete" and not hides_rk(cfg.environment):
        cache_key = f"{selected_filter}_b{cfg.model.native_state_bins}"
    cache_path = Path(cfg.dataset.output).with_name(
        Path(cfg.dataset.output).stem + f".{cache_key}.beliefs.npz"
    )
    if selected_filter == "oracle":
        # Oracle beliefs require truth and are evaluator-only; never cache them for training.
        raise ValueError("oracle filter is an evaluator-only ablation, not a training input")
    start = perf_seconds()
    cache = None
    if method not in FAITHFUL_METHODS:
        if cache_path.exists():
            cache = (
                PublicBeliefCache.load(cache_path)
                if hides_rk(cfg.environment)
                else BeliefCache.load(cache_path)
            )
        else:
            if hides_rk(cfg.environment):
                cache = cache_public_beliefs(
                    dataset,
                    factory,
                    cfg.seed + 30_000,
                    method_context.observation_scale,
                )
            else:
                cache = cache_dataset_beliefs(
                    dataset, factory, cfg.seed + 30_000,
                    cfg.environment.K_ref, cfg.environment.safety_threshold,
                )
            temporary = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp.npz")
            cache.save(temporary)
            try:
                os.replace(temporary, cache_path)
            except FileNotFoundError:
                pass
    timings["belief_cache_seconds"] = elapsed_since(start)
    start = perf_seconds()
    if method in FAITHFUL_METHODS:
        episodes = np.unique(dataset.episode_id)
        shuffled = episodes.copy()
        np.random.default_rng(cfg.seed + cfg.training.split_seed_offset).shuffle(shuffled)
        holdout_count = (
            min(len(shuffled) - 1, max(1, int(round(cfg.training.holdout_fraction * len(shuffled)))))
            if cfg.training.enabled and cfg.training.holdout_fraction > 0.0 and len(shuffled) > 1
            else 0
        )
        holdout_ids = shuffled[:holdout_count]
        train_mask = ~np.isin(dataset.episode_id, holdout_ids)
        train_dataset = dataset.subset(train_mask) if holdout_count else dataset
        holdout_dataset = dataset.subset(~train_mask) if holdout_count else None
        train_cache = holdout_cache = None
        split_info = {
            "training_enabled": bool(cfg.training.enabled),
            "holdout_fraction": float(cfg.training.holdout_fraction),
            "train_transitions": len(train_dataset),
            "holdout_transitions": 0 if holdout_dataset is None else len(holdout_dataset),
            "train_episodes": train_dataset.num_episodes,
            "holdout_episodes": 0 if holdout_dataset is None else holdout_dataset.num_episodes,
            "holdout_original_episode_ids": [int(x) for x in sorted(holdout_ids)],
            "faithful_internal_beliefs": True,
        }
    else:
        train_dataset, train_cache, holdout_dataset, holdout_cache, split_info = split_train_holdout(
            dataset, cache, cfg
        )
    timings["training_split_seconds"] = elapsed_since(start)
    policy, train_cache = build_method(
        method,
        cfg,
        train_dataset,
        factory,
        train_cache,
        timings,
        holdout_dataset,
        holdout_cache,
        split_info,
        method_context,
    )
    evaluation_factory = factory
    if hides_rk(cfg.environment) and method == "moor_native":
        evaluation_factory = policy.hidden_filter_factory()
    evaluator = ContinuousEvaluator(cfg, evaluation_factory, selected_filter)
    start = perf_seconds()
    rows = evaluator.run(policy)
    timings["evaluation_seconds"] = elapsed_since(start)
    output = (
        _reward_mode_output_root(cfg) / method / (filter_mode or cfg.filter.proposal)
    )
    start = perf_seconds()
    extra_summary = {
        "fit_diagnostics": policy.fit_diagnostics,
        "output_dir": str(output),
        "dataset_sha256": dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset),
        "training_split": split_info,
        "expose_rk": cfg.environment.expose_rk,
        "regime_label": dataset.metadata.get("regime_label"),
        "target_rows": dataset.metadata.get("target_rows"),
        "actual_rows": dataset.metadata.get("actual_rows", len(dataset)),
        "overshoot_rows": dataset.metadata.get("overshoot_rows"),
        "episode_count": dataset.metadata.get("episode_count", dataset.num_episodes),
        **backend.to_dict(),
    }
    if surrogate is not None:
        extra_summary.update(
            {
                "public_surrogate_path": str(surrogate_path),
                "public_surrogate_cache_status": surrogate_cache_status,
                "public_surrogate_diagnostics": surrogate.diagnostics,
                "safe_mode_interpretation": (
                    "information-limited under the private safety objective when "
                    "public termination and reward-tail signals are weak"
                ),
            }
        )
        try:
            private = load_private(cfg.dataset.private_output)
            extra_summary["evaluator_only_surrogate_diagnostics"] = (
                evaluator_only_surrogate_diagnostics(surrogate, dataset, private)
            )
        except (FileNotFoundError, ValueError, KeyError):
            extra_summary["evaluator_only_surrogate_diagnostics"] = {}
    summary = evaluator.save(
        rows, output, extra_summary
    )
    if train_cache is not None:
        train_cache.save(output / "offline_beliefs.npz")
    summary.update(policy.save_fit_artifacts(output))
    summary.update(save_training_artifacts(
        getattr(policy, "training_history", None), output, cfg.training
    ))
    timings["summary_save_seconds"] = elapsed_since(start)
    timings["row_seconds"] = elapsed_since(row_start)
    timings["peak_rss_mb"] = peak_rss_mb()
    summary.update(timings)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def run_oracle_state_ablation(
    method: str,
    cfg: BenchmarkConfig,
    training_filter: str = "learned",
):
    """Train normally, then evaluate with private true state as the belief ceiling."""
    if hides_rk(cfg.environment):
        raise ValueError(
            "oracle-state beliefs are evaluator-private and unavailable to hidden methods"
        )
    backend = set_active_backend(
        resolve_backend_for_workload(
            cfg.compute, f"oracle_ablation:{method}", cfg.environment
        )
    )
    dataset = ensure_dataset(cfg, False)
    train_factory, _ = make_filter_factory(cfg, dataset, training_filter)
    cache = cache_dataset_beliefs(
        dataset, train_factory, cfg.seed + 30_000,
        cfg.environment.K_ref, cfg.environment.safety_threshold,
    )
    train_dataset, train_cache, holdout_dataset, holdout_cache, split_info = split_train_holdout(
        dataset, cache, cfg
    )
    policy, _ = build_method(
        method,
        cfg,
        train_dataset,
        train_factory,
        train_cache,
        holdout_dataset=holdout_dataset,
        holdout_cache=holdout_cache,
        split_info=split_info,
    )
    oracle_factory, _ = make_filter_factory(cfg, dataset, "oracle")
    evaluator = ContinuousEvaluator(cfg, oracle_factory, "oracle_state")
    rows = evaluator.run(policy)
    output = _reward_mode_output_root(cfg) / method / "oracle_state"
    summary = evaluator.save(
        rows,
        output,
        {
            "fit_diagnostics": policy.fit_diagnostics,
            "dataset_sha256": dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset),
            "training_split": split_info,
            **backend.to_dict(),
        },
    )
    summary.update(save_training_artifacts(
        getattr(policy, "training_history", None), output, cfg.training
    ))
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
