#!/usr/bin/env python3
"""Serialize identical public-data fits, then invoke the registered fit runner."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from real_ecology_benchmark.config import load_config  # noqa: E402
from real_ecology_benchmark.dataset import dataset_sha256  # noqa: E402
from real_ecology_benchmark.faithful_fit import (  # noqa: E402
    fit_cache_key,
    fit_transition_hash,
    load_or_fit_mechanistic_model,
    split_history_episodes,
)
from real_ecology_benchmark.pipeline import (  # noqa: E402
    _hidden_method_context,
    _load_or_fit_public_surrogate,
    ensure_dataset,
)
from run_real_manifest_row import apply_row_config, read_manifest_row  # noqa: E402


def full_lock_identity(
    *, runtime_digest: str, config_sha256: str, public_data_hash: str,
    transition_data_hash: str, candidate_index: int, candidate_seed: int,
    cache_key: str,
) -> str:
    payload = {
        "runtime_digest": runtime_digest,
        "config_sha256": config_sha256,
        "public_data_hash": public_data_hash,
        "transition_data_hash": transition_data_hash,
        "candidate_index": candidate_index,
        "candidate_seed": candidate_seed,
        "fit_cache_key": cache_key,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def candidate_lock_specs(cfg, row, dataset, config_path: Path):
    expected_config = str(row.get("config_sha256", ""))
    actual_config = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if not expected_config or actual_config != expected_config:
        raise SystemExit("configuration hash does not match the frozen manifest")
    runtime_digest = str(row.get("runtime_digest_reference", ""))
    if len(runtime_digest) != 64:
        raise SystemExit("runtime digest is missing from the frozen manifest")
    surrogate, _path, _status = _load_or_fit_public_surrogate(cfg, dataset)
    context = _hidden_method_context(cfg, dataset, surrogate)
    seed_root = cfg.seed + 51_000
    if int(row.get("candidate_seed_root", -1)) != seed_root:
        raise SystemExit("candidate seed root does not match the frozen manifest")
    history, holdout = split_history_episodes(
        dataset, cfg.faithful.fit.history_fraction, seed_root
    )
    transition_hash = fit_transition_hash(dataset)
    public_hash = str(dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset))
    specs = []
    for candidate_index in range(8):
        sampled = history
        candidate_seed = seed_root + candidate_index * 10_000
        if candidate_index:
            rng = np.random.default_rng(candidate_seed + 404)
            sampled = tuple(
                int(value) for value in rng.choice(history, size=len(history), replace=True)
            )
        key = fit_cache_key(
            dataset, context, "ricker", cfg.faithful.model, cfg.faithful.fit,
            candidate_seed, f"candidate_00_{candidate_index:02d}", sampled, holdout, 0.90,
        )
        identity = full_lock_identity(
            runtime_digest=runtime_digest,
            config_sha256=expected_config,
            public_data_hash=public_hash,
            transition_data_hash=transition_hash,
            candidate_index=candidate_index,
            candidate_seed=candidate_seed,
            cache_key=key,
        )
        specs.append({
            "identity": identity,
            "cache_key": key,
            "candidate_index": candidate_index,
            "candidate_seed": candidate_seed,
            "fit_episode_ids": sampled,
            "holdout_episode_ids": holdout,
            "context": context,
        })
    if len({spec["identity"] for spec in specs}) != 8:
        raise SystemExit("candidate lock identities are not unique")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("index", type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    row = read_manifest_row(args.manifest, args.index)
    if row.get("run_stage") != "dynamics_fit":
        raise SystemExit("locked fit wrapper accepts only dynamics_fit rows")
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    cfg, _cell, _ = apply_row_config(cfg, row, args.output_root)
    dataset = ensure_dataset(cfg, regenerate=False)
    specs = candidate_lock_specs(cfg, row, dataset, config_path)
    lock_root = Path(args.output_root) / "fit_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        lock_path = lock_root / f"{spec['identity']}.lock"
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            # Recheck under the one-candidate full-identity lock. The scientific
            # loader performs its own cache validation and atomic publication.
            fit, _status = load_or_fit_mechanistic_model(
                dataset, spec["context"], "ricker", cfg.faithful.model,
                cfg.faithful.fit, spec["candidate_seed"],
                candidate_id=f"candidate_00_{spec['candidate_index']:02d}",
                fit_episode_ids=spec["fit_episode_ids"],
                holdout_episode_ids=spec["holdout_episode_ids"],
                regime_persistence=0.90,
                cache_dir=cfg.faithful.fit_cache_dir,
            )
            if fit.fit_cache_key != spec["cache_key"]:
                raise SystemExit("scientific cache key changed while holding its lock")
        # The context manager releases this candidate immediately; no later
        # candidate lock is acquired while an earlier lock remains held.
    runner = ROOT / "scripts/run_adapted_fit_row.py"
    sys.argv = [
        str(runner),
        args.manifest,
        str(args.index),
        "--config",
        args.config,
        "--output-root",
        args.output_root,
    ]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
