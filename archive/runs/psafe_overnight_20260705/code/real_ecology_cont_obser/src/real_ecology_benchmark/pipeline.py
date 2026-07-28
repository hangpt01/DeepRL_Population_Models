"""End-to-end construction of data, filter, beliefs, methods, and evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from .backend import get_active_backend, resolve_backend_for_workload, set_active_backend
from .beliefs import (
    BeliefCache,
    LearnedLinearProposal,
    MechanisticProposal,
    OracleStateFilter,
    ParticleFilter,
    RawObservationFilter,
    ReferenceProposal,
    cache_dataset_beliefs,
)
from .collector import collect_dataset
from .config import BenchmarkConfig
from .dataset import load_public, save_private, save_public
from .evaluator import ContinuousEvaluator
from .methods import METHODS, BasePolicy
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
    the Tier-2/3 manifest runners.)
    """

    root = Path(cfg.evaluation.output_dir)
    if cfg.environment.control_mode == "real_setpoint":
        # Namespace by effective backend too, so CPU and GPU rows never clobber
        # each other (fairness rule) and can be aggregated as separate tables.
        backend = get_active_backend().name
        return root / f"backend_{backend}" / f"reward_{cfg.environment.reward_mode}"
    return root


def _validate_dataset_cell(cfg: BenchmarkConfig, dataset) -> None:
    recorded = dataset.metadata.get("environment", {})
    expected = cfg.environment.__dict__
    keys = (
        "kind", "population", "num_actions", "control_mode", "observation_noise_sigma",
        "process_noise_sigma", "safety_threshold", "K_base", "r_base_low", "r_base_high",
        # Reward-affecting fields: the logged dataset.rewards depend on these, so a
        # cache from a different reward setting must not be silently reused.
        "reward_mode", "safety_penalty_mode", "collapse_penalty", "alpha",
        # Reporting/feature fields: stale diagnostics and belief caches are
        # misleading if these floors change.
        "mvp_threshold",
    )
    def recorded_value(key):
        if key == "control_mode":
            return recorded.get(key, "tier2_one_step")
        return recorded.get(key)

    mismatch = {key: (recorded_value(key), expected.get(key)) for key in keys
                if recorded_value(key) != expected.get(key)}
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


def make_filter_factory(cfg: BenchmarkConfig, dataset, mode: str | None = None):
    selected = mode or cfg.filter.proposal
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
) -> tuple[BasePolicy, object]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method}; choose from {sorted(METHODS)}")
    if cache is None:
        start = perf_seconds()
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
    policy = METHODS[method](
        cfg.environment, cfg.model, cfg.planner, seed=cfg.seed + 40_000
    )
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
    # Resolve + activate the compute backend once for this run (numpy CPU default;
    # cupy GPU when requested and available).  Fails loud in strict mode.
    backend = set_active_backend(
        resolve_backend_for_workload(cfg.compute, f"method:{method}", cfg.environment)
    )
    timings: dict[str, float] = {}
    start = perf_seconds()
    dataset = ensure_dataset(cfg, regenerate)
    timings["dataset_seconds"] = elapsed_since(start)
    selected_filter = filter_mode or cfg.filter.proposal
    start = perf_seconds()
    factory, _proposal = make_filter_factory(cfg, dataset, selected_filter)
    timings["filter_factory_seconds"] = elapsed_since(start)
    cache_path = Path(cfg.dataset.output).with_name(
        Path(cfg.dataset.output).stem + f".{selected_filter}.beliefs.npz"
    )
    if selected_filter == "oracle":
        # Oracle beliefs require truth and are evaluator-only; never cache them for training.
        raise ValueError("oracle filter is an evaluator-only ablation, not a training input")
    start = perf_seconds()
    if cache_path.exists():
        cache = BeliefCache.load(cache_path)
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
    )
    evaluator = ContinuousEvaluator(cfg, factory, selected_filter)
    start = perf_seconds()
    rows = evaluator.run(policy)
    timings["evaluation_seconds"] = elapsed_since(start)
    output = (
        _reward_mode_output_root(cfg) / method / (filter_mode or cfg.filter.proposal)
    )
    start = perf_seconds()
    summary = evaluator.save(
        rows, output, {
            "fit_diagnostics": policy.fit_diagnostics,
            "output_dir": str(output),
            "training_split": split_info,
            **backend.to_dict(),
        }
    )
    train_cache.save(output / "offline_beliefs.npz")
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
