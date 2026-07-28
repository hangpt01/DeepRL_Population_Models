"""Command-line interface for the ecology benchmark package."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile

from . import dummydata, realdata
from .collector import calibration_summary, collect_dataset
from .config import (
    dummy_environment,
    dummy_environment_like,
    environment_with_kind_defaults,
    load_config,
    real_environment,
    real_environment_like,
    synthetic_environment,
    uses_dummy_data,
    uses_real_data,
    uses_synthetic_data,
)
from .dataset import load_private, load_public, save_private, save_public
from .envs import make_env
from .gate import run_decision_gate, save_gate
from .manifest import aggregate_summaries, make_manifest, make_motivation_native_manifest
from .methods import METHODS
from .pipeline import _validate_dataset_cell, run_method, run_oracle_state_ablation


FILTER_CHOICES = ["learned", "reference", "raw", "ricker", "true_family", "native_discrete"]


def _config(args):
    cfg = load_config(args.config)
    if getattr(args, "data_mode", None):
        family = getattr(args, "environment", None) or cfg.environment.kind
        if args.data_mode == "dummy" and not uses_dummy_data(cfg.environment):
            cfg.environment = dummy_environment(
                getattr(args, "population", None) or dummydata.DEFAULT_POPULATION,
                family,
                horizon=cfg.environment.horizon,
                observation_noise_sigma=cfg.environment.observation_noise_sigma,
                process_noise_sigma=cfg.environment.process_noise_sigma,
                reward_mode=cfg.environment.reward_mode,
                safety_penalty_mode=cfg.environment.safety_penalty_mode,
            )
        elif args.data_mode == "real" and not uses_real_data(cfg.environment):
            cfg.environment = real_environment(
                getattr(args, "population", None) or "Amur tiger",
                family,
                data_dir=cfg.environment.data_dir or realdata.DATA_DIR,
                horizon=cfg.environment.horizon,
                observation_noise_sigma=cfg.environment.observation_noise_sigma,
                process_noise_sigma=cfg.environment.process_noise_sigma,
                reward_mode=cfg.environment.reward_mode,
                safety_penalty_mode=cfg.environment.safety_penalty_mode,
            )
        elif args.data_mode == "synthetic" and not uses_synthetic_data(cfg.environment):
            cfg.environment = synthetic_environment(
                family,
                horizon=cfg.environment.horizon,
                observation_noise_sigma=cfg.environment.observation_noise_sigma,
                process_noise_sigma=cfg.environment.process_noise_sigma,
            )
    if getattr(args, "population", None):
        # Rebuild the per-population cell (K/N0/caps/s_safe from data) while
        # PRESERVING the config's env overrides (observation noise, horizon, ...).
        family = getattr(args, "environment", None) or cfg.environment.kind
        if uses_dummy_data(cfg.environment):
            cfg.environment = dummy_environment_like(cfg.environment, args.population, family)
        elif uses_real_data(cfg.environment):
            cfg.environment = real_environment_like(cfg.environment, args.population, family)
        else:
            raise SystemExit("--population is only valid for real or dummy data modes")
    elif getattr(args, "environment", None):
        cfg.environment = environment_with_kind_defaults(cfg.environment, args.environment)
    if getattr(args, "sigma", None) is not None:
        cfg.environment = replace(cfg.environment, observation_noise_sigma=args.sigma)
        cfg.validate()
    if getattr(args, "reward_mode", None):
        cfg.environment = replace(cfg.environment, reward_mode=args.reward_mode)
        cfg.validate()
    backend = getattr(args, "backend", None)
    device = getattr(args, "device", None)
    strict = getattr(args, "backend_strict", None)
    if backend is not None or device is not None or strict is not None:
        cfg.compute = replace(
            cfg.compute,
            **({"backend": backend} if backend is not None else {}),
            **({"device": device} if device is not None else {}),
            **({"strict": strict == "true"} if strict is not None else {}),
        )
        cfg.validate()
    if getattr(args, "transitions", None):
        cfg.dataset.transitions = args.transitions
    return cfg


def cmd_generate(args):
    cfg = _config(args)
    dataset, private = collect_dataset(
        make_env(cfg.environment), cfg.dataset.transitions, cfg.dataset.episode_length,
        cfg.seed, cfg.environment.privileged_behavior,
    )
    save_public(cfg.dataset.output, dataset)
    save_private(cfg.dataset.private_output, private)
    print(json.dumps(calibration_summary(dataset, private, cfg.environment.safety_threshold), indent=2))


def cmd_calibrate(args):
    cfg = _config(args)
    dataset = private = None
    if Path(cfg.dataset.output).exists() and Path(cfg.dataset.private_output).exists():
        candidate = load_public(cfg.dataset.output)
        try:
            # Guard against reusing a cached dataset from a different cell
            # (population/family/sigma) at the shared dataset path.
            _validate_dataset_cell(cfg, candidate)
            dataset = candidate
            private = load_private(cfg.dataset.private_output)
        except ValueError:
            dataset = private = None
    if dataset is None:
        dataset, private = collect_dataset(
            make_env(cfg.environment), cfg.dataset.transitions, cfg.dataset.episode_length,
            cfg.seed, cfg.environment.privileged_behavior,
        )
    summary = calibration_summary(dataset, private, cfg.environment.safety_threshold)
    print(json.dumps(summary, indent=2))
    if args.require_band and not summary["collapse_band_pass"]:
        raise SystemExit("calibration failed: healthy-start incident collapse is outside [0.15,0.24]")


def cmd_run(args):
    cfg = _config(args)
    summary = run_method(args.method, cfg, args.filter, args.regenerate)
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_oracle_eval(args):
    cfg = _config(args)
    summary = run_oracle_state_ablation(args.method, cfg, args.training_filter)
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_gate(args):
    cfg = _config(args)
    result = run_decision_gate(cfg, args.episodes)
    save_gate(result, args.output)
    compact = {k: v for k, v in result.items() if k != "records"}
    print(json.dumps(compact, indent=2, sort_keys=True))


def cmd_manifest(args):
    if args.motivation_native:
        rows = make_motivation_native_manifest(args.output)
    else:
        rows = make_manifest(args.output, data_mode=args.data_mode)
    print(json.dumps({"rows": rows}, indent=2))


def cmd_aggregate(args):
    baselines = None
    if args.baseline:
        baselines = []
        for item in args.baseline:
            method, sep, filter_name = item.partition(":")
            if not sep:
                raise SystemExit("--baseline entries must be METHOD:FILTER")
            baselines.append((method, filter_name))
    result = aggregate_summaries(
        args.root,
        args.output,
        challenger_filter=args.challenger_filter,
        baselines=baselines,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "summaries"}, indent=2))


def cmd_smoke(args):
    cfg = _config(args)
    with tempfile.TemporaryDirectory(prefix="ecology_smoke_") as tmp:
        cfg.dataset.transitions = 160
        cfg.dataset.episode_length = 12
        cfg.dataset.output = str(Path(tmp) / "public.npz")
        cfg.dataset.private_output = str(Path(tmp) / "private.npz")
        cfg.filter.particles = 64
        cfg.model.ensemble_size = 3
        cfg.planner.sequences = 16
        cfg.planner.particles = 8
        cfg.planner.horizon = 3
        cfg.evaluation.seeds = [9001]
        cfg.evaluation.episodes_per_seed = 1
        cfg.evaluation.horizon = 8
        cfg.evaluation.output_dir = str(Path(tmp) / "eval")
        summary = run_method("mopo", cfg, "reference", regenerate=True)
        print(json.dumps({"status": "ok", "summary": summary}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(prog="real-ecology")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/real_default.yaml")
    common.add_argument("--data-mode", choices=["real", "dummy", "synthetic"],
                        help="data source: real CSV tables, in-code dummy profile, or synthetic controls")
    common.add_argument("--environment", choices=["ricker", "allee", "theta", "regime"],
                        help="map family for the population cell")
    common.add_argument("--population",
                        help="real population to instantiate (overrides the config cell)")
    common.add_argument("--reward-mode", choices=["yield", "safe"],
                        help="reward setting: yield (P=0) or safe (P>0, collapse-aware)")
    common.add_argument("--sigma", type=float)
    common.add_argument("--backend", choices=["numpy", "cupy"],
                        help="compute backend: numpy (CPU, default) or cupy (GPU)")
    common.add_argument("--device", type=int, help="GPU device id for cupy backend")
    common.add_argument("--backend-strict", choices=["true", "false"],
                        help="strict=true fails if the backend is unavailable; false allows fallback")

    p = sub.add_parser("generate", parents=[common])
    p.add_argument("--transitions", type=int)
    p.set_defaults(func=cmd_generate)
    p = sub.add_parser("calibrate", parents=[common])
    p.add_argument("--transitions", type=int)
    p.add_argument("--require-band", action="store_true")
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("run", parents=[common])
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--filter", choices=FILTER_CHOICES)
    p.add_argument("--regenerate", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("oracle-eval", parents=[common])
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--training-filter", choices=["learned", "reference", "raw"], default="learned")
    p.set_defaults(func=cmd_oracle_eval)
    p = sub.add_parser("gate", parents=[common])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--output", default="outputs/gate.json")
    p.set_defaults(func=cmd_gate)
    p = sub.add_parser("smoke", parents=[common])
    p.set_defaults(func=cmd_smoke)
    p = sub.add_parser("manifest")
    p.add_argument("--output", default="outputs/manifest.csv")
    p.add_argument("--data-mode", choices=["synthetic", "real"], default="synthetic")
    p.add_argument("--motivation-native", action="store_true")
    p.set_defaults(func=cmd_manifest)
    p = sub.add_parser("aggregate")
    p.add_argument("--root", default="outputs/evaluation")
    p.add_argument("--output", default="outputs/aggregate.json")
    p.add_argument("--challenger-filter")
    p.add_argument(
        "--baseline",
        action="append",
        help="paired baseline as METHOD:FILTER; repeat for beats-both comparisons",
    )
    p.set_defaults(func=cmd_aggregate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
