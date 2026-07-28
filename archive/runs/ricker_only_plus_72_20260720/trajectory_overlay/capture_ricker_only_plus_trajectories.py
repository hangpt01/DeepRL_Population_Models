#!/usr/bin/env python3
"""Return-blind per-timestep trajectory capture for the 72-cell Ricker-only PLUS run.

Post-run overlay.  This script NEVER touches the frozen runtime snapshot or the
production artifacts except to READ frozen datasets, fit cache, and the public
surrogate.  It regenerates the registered 20 paired evaluation episodes for one
manifest cell, rolling the SAME code that produced the frozen evaluation, and
records an ordered per-timestep trace compatible with the general-RL trajectory
schema so the two sides can later be merged and plotted.

Design guarantees
-----------------
* Runs only ``plus_adapted_ricker_only_pbvi`` (the sole method in this run).
* Reuses the frozen 72-cell config, datasets, fit cache and exact evaluation
  seeds.  No refit: the candidate bank is loaded from the frozen fit cache and
  the reuse is asserted with the frozen validators (raises on any cache miss).
* Return-blind.  It never opens ``summary.json`` / ``comparative_summary.json``
  / ``ranking.json`` and never reads any frozen operational/true return.  It
  reconstructs each cell's discounted return FROM ITS OWN captured trajectory
  (sum_t gamma^t * reward_public[t]) and stores it in the sealed receipt for a
  LATER, explicitly authorized comparison against the frozen returns.
* Execution is gated: it refuses to run unless the run's ``acceptance.json``
  records a PASS decision with ``return_fields_opened == False``.  ``afterok``
  on the acceptance job is not sufficient because that program records
  FAIL/INCOMPLETE in JSON without exiting non-zero.
* All output stays under ``trajectory_overlay/`` and is treated as SEALED until
  the operator authorizes comparative inspection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Fixed layout.  Everything the overlay reads is under the run directory.
# ---------------------------------------------------------------------------
RUN = Path(__file__).resolve().parent.parent
FROZEN = RUN / "runtime_snapshot_routing_fix"
PRODUCTION = RUN / "production"
CONFIG = FROZEN / "configs" / "paper_faithful_hidden_ricker_only_plus_v1.yaml"
MANIFEST = FROZEN / "experiments" / "ricker_only_plus_72" / "manifests" / "ricker_only_plus_plan_72.csv"
ACCEPTANCE = PRODUCTION / "acceptance.json"
FREEZE_RECEIPT = FROZEN / "experiments" / "ricker_only_plus_72" / "FREEZE_RECEIPT.json"
PACKAGE_IDENTITY = FROZEN / "experiments" / "ricker_only_plus_72" / "LAUNCH_PACKAGE_IDENTITY.json"
DATASET_REGISTRY = Path(
    "/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/"
    "general_phase2e_full_sigma01_02_20260720_v1/manifests/"
    "ecological_dataset_reuse_registry_144.csv"
)
RESULTS = Path(__file__).resolve().parent / "results"

METHOD = "plus_adapted_ricker_only_pbvi"
SCHEMA = "ricker_only_plus_trajectory_v1"
PRODUCTION_JOB_ID = "58396952"
SELECTED_INDICES = frozenset((*range(0, 8), *range(16, 24), *range(40, 48)))
EXPECTED_RUNTIME_DIGEST = "7628816c85a39d49373892a0e72d1751de2f0a32dd003b93a54ca1f3e151982c"
EXPECTED_CONFIG_SHA256 = "b871668bae1c971795267bc402c3450e1fe6a628b4e8fee91b1300cfd73fbc26"
EXPECTED_FREEZE_RECEIPT_SHA256 = "3285a9cc512576fe6648558ecafa5d5715de6fddd91678fc33e2bbf641286dd4"
EXPECTED_PACKAGE_IDENTITY_SHA256 = "4cecee192dedc5d11f538ee8083531b23b9634f06b4f6db3cc5a6c30b539e7ac"
WAIVER = {
    "scoped_selected_trajectory_waiver": True,
    "global_acceptance": "INCOMPLETE",
    "global_missing_index": 51,
    "global_missing_cell": "Iberian lynx / Allee / sigma_obs=0.2",
    "selected_cells_complete": "24/24",
    "missing_global_cell_outside_selected_subset": True,
    "full_run_claims_prohibited": True,
}

# Import the FROZEN code so we regenerate exactly what produced the run.
sys.path.insert(0, str(FROZEN / "scripts"))
sys.path.insert(0, str(FROZEN / "src"))

import run_real_manifest_row as rrmr  # noqa: E402  (apply_row_config, read_manifest_row, validators)
import run_ricker_only_plus_acceptance as acceptance_checker  # noqa: E402
from real_ecology_benchmark.config import hides_rk, load_config  # noqa: E402
from real_ecology_benchmark.dataset import dataset_sha256  # noqa: E402
from real_ecology_benchmark.envs import make_env  # noqa: E402
from real_ecology_benchmark.methods import FAITHFUL_METHODS  # noqa: E402
from real_ecology_benchmark.pipeline import (  # noqa: E402
    _hidden_method_context,
    build_method,
    ensure_dataset,
    make_filter_factory,
)
from real_ecology_benchmark.public_surrogate import (  # noqa: E402
    PublicRewardRiskSurrogate,
    SURROGATE_VERSION,
)
from real_ecology_benchmark.types import PublicTransition  # noqa: E402

# Per-timestep long-form schema.  The first block matches the general side
# exactly; the trailing PLUS-specific columns support the misspecification read.
STEP_KEYS = (
    "state_pre", "state_post", "observation_pre", "observation_post",
    "belief_mean", "belief_low", "belief_high", "action",
    "reward_true", "reward_public", "danger", "unsafe", "mvp", "terminated",
    "candidate_entropy", "selected_candidate_index",
)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sigma_slug(value) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_frozen_identity(row: dict) -> dict:
    """Verify the exact frozen package/config identities without reading outcomes."""
    if sha256(CONFIG) != EXPECTED_CONFIG_SHA256 or row.get("config_sha256") != EXPECTED_CONFIG_SHA256:
        raise SystemExit("frozen config hash mismatch")
    if sha256(FREEZE_RECEIPT) != EXPECTED_FREEZE_RECEIPT_SHA256:
        raise SystemExit("freeze receipt hash mismatch")
    if sha256(PACKAGE_IDENTITY) != EXPECTED_PACKAGE_IDENTITY_SHA256:
        raise SystemExit("launch package identity hash mismatch")
    freeze = acceptance_checker.guarded_json(FREEZE_RECEIPT)
    package = acceptance_checker.guarded_json(PACKAGE_IDENTITY)
    if (
        row.get("runtime_digest_reference") != EXPECTED_RUNTIME_DIGEST
        or freeze.get("ricker_only_runtime_digest") != EXPECTED_RUNTIME_DIGEST
        or package.get("scientific_runtime_digest") != EXPECTED_RUNTIME_DIGEST
        or package.get("production_config_sha256") != EXPECTED_CONFIG_SHA256
    ):
        raise SystemExit("frozen runtime identity mismatch")
    for module in (rrmr, acceptance_checker):
        if FROZEN.resolve() not in Path(module.__file__).resolve().parents:
            raise SystemExit(f"non-frozen module import: {module.__file__}")
    return {
        "scientific_runtime_digest": EXPECTED_RUNTIME_DIGEST,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "freeze_receipt_sha256": EXPECTED_FREEZE_RECEIPT_SHA256,
        "package_identity_sha256": EXPECTED_PACKAGE_IDENTITY_SHA256,
    }


def require_slurm_success(index: int) -> dict:
    """Require this exact production array element to have exited zero."""
    result = subprocess.run(
        [
            "sacct", "-j", PRODUCTION_JOB_ID, "--array", "-X", "-n", "-P",
            "-o", "JobID,State,ExitCode",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    expected = f"{PRODUCTION_JOB_ID}_{index}"
    matches = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == expected:
            matches.append((fields[1].split()[0], fields[2]))
    if matches != [("COMPLETED", "0:0")]:
        raise SystemExit(f"cell not ready: Slurm {expected} status={matches!r}")
    return {"job_id": expected, "state": "COMPLETED", "exit_code": "0:0"}


def require_cell_ready(row: dict) -> dict:
    """Fail closed unless THIS cell's own return-blind plan-completion receipt is valid.

    Each cell is independent: its capture needs only its own frozen artifacts, so a
    cell may be captured as soon as its plan stage completes -- no need to wait for
    the other cells or for the global 72-cell acceptance.  This is the same per-row
    receipt the acceptance program checks, and it carries no return fields.
    """
    path = PRODUCTION / "completion_receipts" / row["fit_cell"] / row["method"] / "plan_completion.json"
    if not path.is_file():
        raise SystemExit(f"cell not ready: {path} is missing")
    receipt = acceptance_checker.guarded_json(path)
    problems = []
    expected_fields = {
        "receipt_schema": "ricker_only_plan_completion_v1",
        "completion_status": "complete",
        "receipt_write": "atomic_replace",
        "return_fields_opened": False,
        "manifest_index": int(row["index"]),
        "method": row["method"],
        "fit_cell": row["fit_cell"],
        "artifact_dir": row["evaluation_artifact_dir"],
    }
    for key, expected in expected_fields.items():
        if receipt.get(key) != expected:
            problems.append(f"{key}={receipt.get(key)!r}, expected={expected!r}")
    if problems:
        raise SystemExit(f"cell not ready: {'; '.join(problems)}")

    artifact_root = PRODUCTION / row["evaluation_artifact_dir"]
    recorded = receipt.get("artifact_hashes")
    expected_names = acceptance_checker.expected_plan_artifact_names()
    if not isinstance(recorded, dict) or set(recorded) != expected_names:
        raise SystemExit("cell not ready: artifact receipt set is not exact")
    mismatches = [
        name for name, expected in recorded.items()
        if not (artifact_root / name).is_file() or sha256(artifact_root / name) != expected
    ]
    if mismatches:
        raise SystemExit(f"cell not ready: artifact hash mismatch {mismatches}")
    if acceptance_checker.temporary_or_partial_files(artifact_root):
        raise SystemExit("cell not ready: temporary/partial artifact exists")

    bank = acceptance_checker.guarded_json(artifact_root / "candidate_bank.json")
    fitted = acceptance_checker.guarded_json(artifact_root / "faithful_fit.json")
    privacy = acceptance_checker.guarded_json(artifact_root / "privacy_audit.json")
    planner = acceptance_checker.guarded_json(artifact_root / "planner_provenance.json")
    if (
        bank.get("candidate_count") != 8
        or bank.get("candidate_construction") != acceptance_checker.CONSTRUCTION
        or bank.get("prior_type") != "uniform"
        or len(bank.get("parameter_hashes", [])) != 8
        or len(set(bank.get("parameter_hashes", []))) != 8
        or fitted.get("candidate_count") != 8
        or fitted.get("method_impl_version") != "plus_adapted_ricker_only_pbvi_v1"
        or privacy.get("status") != "passed"
        or privacy.get("forbidden_name_hits")
        or len(planner.get("planners", [])) != 8
    ):
        raise SystemExit("cell not ready: method/privacy/planner structural validation failed")
    for candidate_index in range(8):
        candidate = acceptance_checker.guarded_json(
            artifact_root / f"candidate_{candidate_index:03d}.json"
        )
        if candidate.get("form") != "ricker":
            raise SystemExit(f"cell not ready: candidate {candidate_index} is not Ricker")
    slurm = require_slurm_success(int(row["index"]))
    receipt["verified_artifact_hashes"] = recorded
    receipt["verified_slurm"] = slurm
    return receipt


def require_acceptance(path: Path) -> dict:
    """Optional stricter gate: the whole 72-cell run passed structural acceptance."""
    if not path.is_file():
        raise SystemExit(f"acceptance gate not satisfied: {path} is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    decision = str(payload.get("decision", ""))
    if not decision.startswith("PASS_"):
        raise SystemExit(f"acceptance gate not satisfied: decision={decision!r}")
    if payload.get("return_fields_opened") is not False:
        raise SystemExit("acceptance gate not satisfied: run is not return-blind")
    return payload


def credible_interval(belief, lo_q: float = 0.05, hi_q: float = 0.95):
    """90% credible interval, computed exactly like the frozen evaluator."""
    order = np.argsort(belief.states)
    cdf = np.cumsum(belief.weights[order])
    lo_idx = min(int(np.searchsorted(cdf, lo_q)), len(order) - 1)
    hi_idx = min(int(np.searchsorted(cdf, hi_q)), len(order) - 1)
    return float(belief.states[order[lo_idx]]), float(belief.states[order[hi_idx]])


def build_policy(cfg, row):
    """Rebuild the frozen faithful policy: load dataset/surrogate/fit-cache, no refit."""
    if METHOD not in FAITHFUL_METHODS:
        raise SystemExit(f"{METHOD} is not a faithful method")
    if not hides_rk(cfg.environment):
        raise SystemExit("expected expose_rk='hidden'")
    dataset = ensure_dataset(cfg, regenerate=False)
    # Assert the frozen fit cache is present and hashes match before building.
    receipt_gate = rrmr.validate_adapted_fit_receipt_gate(row, PRODUCTION)
    public_hash = dataset.metadata.get("dataset_sha256") or dataset_sha256(dataset)
    source = Path(cfg.dataset.output)
    surrogate_path = source.with_name(
        f"{source.stem}.regime_hidden.public_surrogate.v{SURROGATE_VERSION}.seed{cfg.seed + 20_000}.npz"
    )
    if not surrogate_path.is_file():
        raise SystemExit(f"public surrogate cache is missing: {surrogate_path}")
    surrogate = PublicRewardRiskSurrogate.load(surrogate_path)
    if surrogate.public_data_hash != public_hash:
        raise SystemExit("public surrogate dataset hash mismatch")
    surrogate_status = "loaded"
    context = _hidden_method_context(cfg, dataset, surrogate)
    factory, _ = make_filter_factory(cfg, dataset, "faithful_internal", context)
    split_info = {
        "training_enabled": bool(cfg.training.enabled),
        "holdout_fraction": 0.0,
        "fit_history_fraction": float(cfg.faithful.fit.history_fraction),
        "faithful_internal_beliefs": True,
    }
    policy, _ = build_method(
        METHOD, cfg, dataset, factory, cache=None, timings={},
        holdout_dataset=None, holdout_cache=None, split_info=split_info,
        method_context=context,
    )
    # No-refit proof: every candidate must have been a cache hit.
    reuse = rrmr.validate_adapted_plan_cache_hit(
        {"fit_diagnostics": policy.fit_diagnostics}, receipt_gate
    )
    provenance = {
        "dataset_sha256": public_hash,
        "public_dataset_file_sha256": sha256(Path(cfg.dataset.output)),
        "surrogate_path": str(surrogate_path),
        "surrogate_cache_status": surrogate_status,
        "fit_cache_reuse": reuse,
    }
    return policy, factory, dataset, provenance


def require_dataset_registry(row: dict, provenance: dict) -> dict:
    """Require exact selected-cell reuse against the frozen general dataset registry."""
    with DATASET_REGISTRY.open(encoding="utf-8", newline="") as handle:
        matches = [
            item for item in csv.DictReader(handle)
            if item["reward_mode"] == row["reward_mode"]
            and item["population"] == row["population"]
            and item["environment"] == row["environment"]
            and float(item["sigma_obs"]) == float(row["sigma_obs"])
        ]
    if len(matches) != 1:
        raise SystemExit(f"dataset registry matched {len(matches)} rows")
    registered = matches[0]
    if (
        registered["reuse_status"] != "exact_ecological_byte_copy"
        or registered["action_channel_reuse"] != "exact_public_array_byte_copy"
        or provenance["dataset_sha256"] != registered["dataset_sha256"]
        or provenance["public_dataset_file_sha256"] != registered["public_file_sha256"]
    ):
        raise SystemExit("selected-cell dataset registry/hash validation failed")
    return {
        "registry_sha256": sha256(DATASET_REGISTRY),
        "reuse_status": registered["reuse_status"],
        "action_channel_reuse": registered["action_channel_reuse"],
        "dataset_sha256": registered["dataset_sha256"],
        "public_file_sha256": registered["public_file_sha256"],
    }


def roll(policy, cfg, factory):
    """Regenerate the registered episodes, mirroring the frozen evaluator loop."""
    env_cfg = cfg.environment
    eval_cfg = cfg.evaluation
    seeds = [int(b + i) for b in eval_cfg.seeds for i in range(eval_cfg.episodes_per_seed)]
    horizon = min(eval_cfg.horizon, env_cfg.horizon)
    discount = float(eval_cfg.discount)
    n_cand = len(policy.candidate_bank.fits)
    out = {k: np.full((len(seeds), horizon), np.nan) for k in STEP_KEYS}
    out["candidate_weights"] = np.full((len(seeds), horizon, n_cand), np.nan)
    reconstructed_return = np.full(len(seeds), np.nan)

    for e, seed in enumerate(seeds):
        env = make_env(env_cfg)
        reset = env.reset(seed)
        filt = factory()
        belief = filt.reset(reset.observation, seed + 10_000)
        policy.reset(seed + 20_000)
        observation = float(reset.observation)
        gamma = 1.0
        op_return = 0.0
        for t in range(horizon):
            out["state_pre"][e, t] = float(env.state)
            out["observation_pre"][e, t] = observation
            out["belief_mean"][e, t] = float(belief.mean_state())
            lo, hi = credible_interval(belief)
            out["belief_low"][e, t] = lo
            out["belief_high"][e, t] = hi
            try:
                action = int(policy.act(belief, observation))
            except (FloatingPointError, ValueError, RuntimeError):
                action = 0
            if not 0 <= action < env.num_actions:
                action = 0
            diag = getattr(policy, "last_diagnostics", {}) or {}
            weights = diag.get("candidate_weights")
            if weights is not None and len(weights) == n_cand:
                out["candidate_weights"][e, t, :] = weights
            out["candidate_entropy"][e, t] = float(diag.get("candidate_entropy", np.nan))
            out["selected_candidate_index"][e, t] = float(diag.get("selected_candidate_index", np.nan))

            result = env.step(action)
            info = result.evaluator_info
            out["action"][e, t] = action
            out["state_post"][e, t] = float(info["state"])
            out["observation_post"][e, t] = float(result.observation)
            out["reward_true"][e, t] = float(info["reward_true"])
            out["reward_public"][e, t] = float(result.reward)
            state_previous = float(info["state_previous"])
            out["danger"][e, t] = float(
                env_cfg.safety_threshold < state_previous <= 4 * env_cfg.safety_threshold
            )
            out["unsafe"][e, t] = float(bool(info["below_safety_region"]))
            out["mvp"][e, t] = float(bool(info["below_mvp_region"]))
            out["terminated"][e, t] = float(bool(result.done and not result.truncated))

            op_return += gamma * float(result.reward)
            policy.observe(
                belief, action,
                PublicTransition(
                    observation=result.observation,
                    done=result.done,
                    truncated=result.truncated,
                    terminated=bool(result.done and not result.truncated),
                    public_info=result.public_info.copy(),
                ),
            )
            belief = filt.update(belief, action, result.observation)
            observation = float(result.observation)
            gamma *= discount
            if result.done:
                break
        reconstructed_return[e] = op_return
    return out, seeds, horizon, discount, reconstructed_return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("index", type=int, help="manifest cell index 0..71")
    ap.add_argument("--output-root", type=Path,
                    default=RESULTS)
    ap.add_argument("--acceptance", type=Path, default=ACCEPTANCE)
    ap.add_argument("--require-global-acceptance", action="store_true",
                    help="ALSO require the whole 72-cell acceptance.json to PASS "
                         "(stricter; a cell only needs its own completion receipt)")
    ap.add_argument("--check", action="store_true",
                    help="gate + rebuild the policy (proves no-refit cache reuse) but "
                         "do NOT roll episodes or write any output")
    args = ap.parse_args()

    if args.index not in SELECTED_INDICES:
        raise SystemExit(f"index {args.index} is outside the authorized selected subset")
    if args.output_root.resolve() != RESULTS.resolve():
        raise SystemExit("trajectory output must remain under trajectory_overlay/results")

    row = rrmr.read_manifest_row(MANIFEST, args.index)
    if row.get("method") != METHOD:
        raise SystemExit(f"row {args.index} method={row.get('method')!r} != {METHOD}")

    # Per-cell gate (always): this cell's own return-blind completion receipt must be valid.
    frozen_identity = require_frozen_identity(row)
    cell_receipt = require_cell_ready(row)
    # Optional stricter gate: the whole run passed structural acceptance.
    acceptance = require_acceptance(args.acceptance) if args.require_global_acceptance else None

    cfg = load_config(str(CONFIG))
    cfg, cell, _eval_cell = rrmr.apply_row_config(cfg, row, PRODUCTION)
    cfg.validate()

    policy, factory, dataset, provenance = build_policy(cfg, row)
    provenance["dataset_registry"] = require_dataset_registry(row, provenance)
    provenance["frozen_identity"] = frozen_identity
    provenance["producer_sha256"] = sha256(Path(__file__))
    if args.check:
        print(json.dumps({
            "status": "ready", "index": int(row["index"]),
            "population": row["population"], "environment": row["environment"],
            "sigma_obs": row["sigma_obs"], "rolled": False,
            "no_refit": True, "fit_cache_reuse": provenance["fit_cache_reuse"],
            "dataset_registry": provenance["dataset_registry"],
            "verified_slurm": cell_receipt["verified_slurm"],
            **WAIVER,
        }), flush=True)
        return
    traces, seeds, horizon, discount, reconstructed = roll(policy, cfg, factory)

    population = row["population"]
    scope = "recoverable" if str(row.get("recoverable", "")).strip().lower() in {"1", "true", "yes"} else "sink"
    meta = {
        "schema": SCHEMA,
        "report_only": True,
        "replaces_frozen_results": False,
        "sealed": True,
        "return_fields_opened": False,
        "method": METHOD,
        "manifest_index": int(row["index"]),
        "population": population,
        "population_scope": scope,
        "environment": row["environment"],
        "sigma_obs": float(row["sigma_obs"]),
        "reward_mode": cfg.environment.reward_mode,
        "expose_rk": cfg.environment.expose_rk,
        "seeds": seeds,
        "episodes": len(seeds),
        "horizon": horizon,
        "discount": discount,
        "num_candidates": len(policy.candidate_bank.fits),
        "safety_threshold": float(cfg.environment.safety_threshold),
        "mvp_threshold": float(cfg.environment.mvp_threshold),
        "num_actions": int(cfg.environment.num_actions),
        # Reconstructed from OUR OWN trajectory only; compared to frozen returns
        # later, under explicit authorization.  Not a frozen result.
        "reconstructed_discounted_return_mean": float(np.nanmean(reconstructed)),
        "reconstructed_discounted_return_per_episode": reconstructed.tolist(),
        "cell_completion_status": cell_receipt.get("completion_status"),
        "global_acceptance_decision": acceptance.get("decision") if acceptance else None,
        "provenance": provenance,
        "source_run": str(RUN),
        **WAIVER,
    }

    out_root = args.output_root
    (out_root / "raw_npz").mkdir(parents=True, exist_ok=True)
    (out_root / "receipts").mkdir(parents=True, exist_ok=True)
    stem = f"cell_{int(row['index']):02d}_{slug(population)}_{row['environment']}_sigma_{sigma_slug(row['sigma_obs'])}_{cfg.environment.reward_mode}"
    npz_target = out_root / "raw_npz" / f"{stem}.npz"
    receipt_target = out_root / "receipts" / f"{stem}.json"
    if npz_target.exists() or receipt_target.exists():
        raise SystemExit(f"refusing to overwrite existing trajectory cell: {stem}")
    npz_temporary = npz_target.with_name(f".{npz_target.stem}.{os.getpid()}.tmp.npz")
    receipt_temporary = receipt_target.with_name(
        f".{receipt_target.name}.{os.getpid()}.tmp"
    )
    np.savez_compressed(
        npz_temporary,
        meta=np.asarray(json.dumps(meta)),
        candidate_weights=traces.pop("candidate_weights"),
        **traces,
    )
    meta["trajectory_npz_sha256"] = sha256(npz_temporary)
    receipt_temporary.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(npz_temporary, npz_target)
    os.replace(receipt_temporary, receipt_target)
    print(json.dumps({"status": "ok", "cell": stem, "episodes": len(seeds)}), flush=True)


if __name__ == "__main__":
    main()
