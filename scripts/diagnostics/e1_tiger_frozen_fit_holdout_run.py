#!/usr/bin/env python3
"""Prospectively registered, zero-refit tiger frozen-fit holdout scoring.

This runner has exactly two natural task identities.  It loads a frozen fit
through the archived read-only cache loader, reconstructs the registered
128/32 episode split, and scores the untouched holdout episodes.  It never
calls a fitting or optimizer route.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


REPO = Path("/fs04/scratch2/ce25/DeepRL_Population_Models")
OUTPUT = REPO / "outputs" / "e1_tiger_frozen_fit_holdout_predictive_20260805_v1"
ARCHIVE = Path(
    "/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs"
)
SOURCE_RUN = ARCHIVE / "adapted_32cell_diagnostic_pending"
SOURCE_CODE = SOURCE_RUN / "code"
TARGET_RUN = ARCHIVE / "three_species_ecological_p10_correction_20260723_v1" / "moor"
PYTHON = Path("/fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10")
COMMIT = "3291eefbe64b5ab19019120bb0ba34a0a9586a53"
REGISTRATION = OUTPUT / "STAGE1_REGISTRATION.json"
REGISTRATION_SIDECAR = OUTPUT / "STAGE1_REGISTRATION.json.sha256"
FIT_SEED = 47116
HISTORY_FRACTION = 0.8
FIT_IDS_HASH = "8a4ba3f56227fe6bf5abc4350f7879c5725078a572bfbe5260e60a18541ac6a3"
HOLDOUT_IDS_HASH = "7db31af32a30facebdf3c490188c4cefee20b67ad9c94926e252ddbc4203aa35"
FIT_CONFIG_DIGEST = "2686c2e09edbeadfa145d26a3c5df12135cf1dc8df4c7a12508370778a2a6e0a"
T_CRITICAL_DF31 = 2.0395134463964077
PRIMARY_POINTS_PER_SD = 32
CHECK_POINTS_PER_SD = 16
TAIL_SD = 12.0
MEAN_ATOL_RAW = 1.0e-5
MEAN_RTOL = 1.0e-8
NLPD_ATOL = 1.0e-6
PIT_ATOL = 1.0e-7
EXPECTED_CPU = "AMD EPYC 9534"
SOURCE_HASHES = {
    SOURCE_CODE / "src/real_ecology_benchmark/faithful_fit.py":
        "f425e0912cb9e5c0eb49224b15cde8dd23a9083d1bb43acb2a0b3902bd249fa4",
    SOURCE_CODE / "src/real_ecology_benchmark/faithful_ecology.py":
        "9ec66daa5976fdbae9f24b4587e25d141197bf92b6fe077644cc6b6b13b3c22c",
    SOURCE_CODE / "src/real_ecology_benchmark/dataset.py":
        "fc77b04af162810a32d4e8d3a328cb716eb2155da4e2293dda8cd77e5e5d6215",
    SOURCE_CODE / "scripts/run_adapted_fit_row.py":
        "6c0cf9088a54e41b04a3f53fef768a9edc5c122136f990faf4f11e43998b59c7",
    SOURCE_CODE / "src/real_ecology_benchmark/observation.py":
        "b88cf2dbcbf0a2ea3d411aca345a004c30522f1705744479e46583d68e5fd74b",
}


def cell(
    sigma: float,
    token: str,
    key: str,
    array_hash: str,
    metadata_hash: str,
    receipt_hash: str,
    parameter_hash: str,
    public_dataset_hash: str,
    transition_hash: str,
    public_file_hash: str,
    private_file_hash: str,
) -> dict[str, Any]:
    fit_root = TARGET_RUN / "fit_cache"
    data_root = TARGET_RUN / "datasets/regime_hidden/reward_safe/amur_tiger/ricker" / token
    private_root = TARGET_RUN / "private/regime_hidden/reward_safe/amur_tiger/ricker" / token
    receipt = (
        TARGET_RUN / "fit_receipts/regime_hidden/reward_safe/amur_tiger/ricker"
        / token / "moor_adapted_ricker_misspec_pbvi/fit_receipt.json"
    )
    return {
        "sigma": sigma,
        "token": token,
        "cache_key": key,
        "array_path": fit_root / f"{key}.npz",
        "array_sha256": array_hash,
        "metadata_path": fit_root / f"{key}.json",
        "metadata_sha256": metadata_hash,
        "receipt_path": receipt,
        "receipt_sha256": receipt_hash,
        "parameter_hash": parameter_hash,
        "public_path": data_root / "public.npz",
        "public_file_sha256": public_file_hash,
        "public_dataset_hash": public_dataset_hash,
        "transition_data_hash": transition_hash,
        "private_path": private_root / "truth.npz",
        "private_file_sha256": private_file_hash,
    }


CELLS = {
    0: cell(
        0.1,
        "sigma_0p1",
        "c8eb8b067aaeaa3c26ebdca2f0175cc59dc2d940589db11e2d056f83bd317f47",
        "66fb78fc7e8045b7a82ae297a62dc4201f4539b6b44488a637cee5e3ba4241a1",
        "961a3db9e1b31d9296ee95e62d64e11ad7b1be4aec42e74cd057ad68e8580702",
        "987ae0ac7db4cedfc896d0f47d58749849769da6fe85ad42113a89e77867e912",
        "bf43a542e5e540d619a774a5c57652645fcbe40680ef2da7fa1ce3f7a47b057d",
        "2cc7611c7bbcb6f72cbda49db73cfc24bde41b297564581dc90a5f60cf3c0d85",
        "c700802a980326ccdac69ac54bd710c736ef31d0bbbb1af05ed7d3702a8a748b",
        "b8dce12ab815ab5e095841aa8971f97a2c1abc6606f1352e6db525fcfdf18103",
        "fc681a899686b063613a99519579af21794ba40d0b20517f6557bac8704021da",
    ),
    1: cell(
        0.2,
        "sigma_0p2",
        "bab1b4157d333d18b5cdae3d4738c00efc82b70ebd74e2984c19efa628c17e44",
        "bc4c35bf30c6233b6ffadb4250236df069b9afb3288203af943b7f304c5aed54",
        "66812eb838209b134d224924da8ba4f0561d6103eb78a94cda25856d532bc99d",
        "afbd0f2c2c11335bc405e81a894c57911006093b5d647b3b02b29c80d528f9d0",
        "b8b3a347347c4c8b29d784d0b1261d02a0b82f005ec6d74c34c2d91e0341f500",
        "04f38e874ce0f95cdf69acfdd639b511c1c828972dd7e043193c751609768080",
        "3523f09c1ea49e7bd5080b469301d252d5cf51f97cc1fa7486937f3c08c89d9f",
        "e599bd403f6fbeec384f86b48fa05891bfedeb4d3f3117c19b6be855f0cf969c",
        "2a97f3a9ff55c04926a64603f8a8f1fc577dedcda436544803a9d8918ed46cba",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NOT_AVAILABLE_NAN"
        if value == math.inf:
            return "POSITIVE_INFINITY"
        if value == -math.inf:
            return "NEGATIVE_INFINITY"
    return value


def write_json_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary collision {temporary}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def write_csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if not rows:
        raise ValueError(f"no rows for {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(f"SHA-256 mismatch for {path}: {actual} != {expected}")
    return actual


def registration_hash() -> str:
    expected_line = REGISTRATION_SIDECAR.read_text(encoding="utf-8").strip()
    fields = expected_line.split()
    if len(fields) != 2 or fields[1] != REGISTRATION.name:
        raise AssertionError("invalid registration SHA-256 sidecar")
    actual = sha256_file(REGISTRATION)
    if actual != fields[0]:
        raise AssertionError("sealed registration hash mismatch")
    return actual


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.casefold().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "NOT AVAILABLE"


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def environment_record(require_slurm: bool) -> dict[str, Any]:
    if sys.version_info[:3] != (3, 10, 14) or np.__version__ != "2.2.6":
        raise AssertionError(
            f"wrong runtime Python={platform.python_version()} NumPy={np.__version__}"
        )
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise AssertionError(f"wrong Python executable {sys.executable}")
    if git_output("rev-parse", "HEAD") != COMMIT:
        raise AssertionError("source commit changed")
    model = cpu_model()
    if require_slurm and EXPECTED_CPU not in model:
        raise AssertionError(f"wrong CPU model {model!r}")
    required_env = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if require_slurm:
        for key, expected in required_env.items():
            if os.environ.get(key) != expected:
                raise AssertionError(f"{key}={os.environ.get(key)!r}, expected {expected!r}")
        if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
            raise AssertionError("SLURM_CPUS_PER_TASK must equal 1")
    return {
        "created_utc": utc_now(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "executable": str(Path(sys.executable).resolve()),
        "commit": COMMIT,
        "branch": git_output("branch", "--show-current"),
        "cpu_model": model,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "NOT AVAILABLE outside Slurm"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "NOT AVAILABLE outside Slurm"),
            "array_task_id": os.environ.get(
                "SLURM_ARRAY_TASK_ID", "NOT AVAILABLE outside Slurm"
            ),
            "cpus_per_task": os.environ.get(
                "SLURM_CPUS_PER_TASK", "NOT AVAILABLE outside Slurm"
            ),
            "partition": os.environ.get("SLURM_JOB_PARTITION", "comp"),
            "qos": "normal",
            "constraint": "EPYC9534",
        },
        "thread_environment": {
            key: os.environ.get(key, "NOT AVAILABLE") for key in required_env
        },
        "declared_environment_digest":
            "55e7d0acfb4bc65ad2ad8859b6016f3e06de6b26d55290a768577d28407080fb",
    }


def verify_static_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    reg_hash = registration_hash()
    source_actual = {str(path): assert_hash(path, expected) for path, expected in SOURCE_HASHES.items()}
    artifact_actual = {
        "fit_array": assert_hash(cfg["array_path"], cfg["array_sha256"]),
        "fit_metadata": assert_hash(cfg["metadata_path"], cfg["metadata_sha256"]),
        "fit_receipt": assert_hash(cfg["receipt_path"], cfg["receipt_sha256"]),
        "public_file": assert_hash(cfg["public_path"], cfg["public_file_sha256"]),
        "private_file": assert_hash(cfg["private_path"], cfg["private_file_sha256"]),
    }
    metadata = read_json(cfg["metadata_path"])
    receipt = read_json(cfg["receipt_path"])
    if metadata.get("cache_key") != cfg["cache_key"]:
        raise AssertionError("frozen metadata cache key mismatch")
    if metadata.get("array_hash") != cfg["array_sha256"]:
        raise AssertionError("frozen metadata array hash mismatch")
    if metadata.get("model", {}).get("parameter_hash") != cfg["parameter_hash"]:
        raise AssertionError("frozen metadata parameter hash mismatch")
    fit = metadata.get("fit", {})
    if fit.get("public_data_hash") != cfg["public_dataset_hash"]:
        raise AssertionError("frozen metadata public data hash mismatch")
    if fit.get("transition_data_hash") != cfg["transition_data_hash"]:
        raise AssertionError("frozen metadata transition hash mismatch")
    if int(fit.get("iterations", -1)) != 100 or fit.get("optimizer") != "torch_lbfgs":
        raise AssertionError("frozen fit configuration mismatch")
    if receipt.get("fit_cache_keys") != [cfg["cache_key"]]:
        raise AssertionError("fit receipt cache key mismatch")
    if receipt.get("parameter_hashes") != [cfg["parameter_hash"]]:
        raise AssertionError("fit receipt parameter hash mismatch")
    if receipt.get("transition_data_hash") != cfg["transition_data_hash"]:
        raise AssertionError("fit receipt transition hash mismatch")
    return {
        "registration_sha256": reg_hash,
        "source_sha256": source_actual,
        "artifact_sha256": artifact_actual,
    }


def normal_logpdf(value: float, means: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        raise AssertionError("registered positive noise scale required")
    z = (float(value) - means) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def mixture_logpdf(
    value: float, means: np.ndarray, sigma: float, weights: np.ndarray
) -> float:
    return logsumexp(np.log(weights) + normal_logpdf(value, means, sigma))


def standard_normal_cdf(values: np.ndarray) -> np.ndarray:
    scale = math.sqrt(2.0)
    return np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / scale)) for value in values),
        dtype=np.float64,
        count=len(values),
    )


def mixture_cdf(
    value: float, means: np.ndarray, sigma: float, weights: np.ndarray
) -> float:
    return float(np.dot(weights, standard_normal_cdf((float(value) - means) / sigma)))


def normal_grid(mean: float, sigma: float, points_per_sd: int) -> tuple[np.ndarray, np.ndarray]:
    return mixture_grid(
        np.asarray([mean]), np.asarray([1.0]), sigma, points_per_sd
    )[:2]


def mixture_grid(
    means: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    points_per_sd: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    means = np.asarray(means, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(means) != len(weights) or len(means) == 0:
        raise AssertionError("invalid posterior mixture")
    if sigma <= 0.0 or points_per_sd < 8:
        raise AssertionError("invalid registered quadrature")
    dx_target = sigma / float(points_per_sd)
    lower = float(np.min(means) - TAIL_SD * sigma)
    upper = float(np.max(means) + TAIL_SD * sigma)
    count = int(math.ceil((upper - lower) / dx_target)) + 1
    if count > 200_000:
        raise AssertionError(f"adaptive log-state quadrature exceeded 200000 nodes: {count}")
    grid = np.linspace(lower, upper, count, dtype=np.float64)
    density = np.zeros(count, dtype=np.float64)
    constant = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    for start in range(0, len(means), 256):
        stop = min(start + 256, len(means))
        z = (grid[:, None] - means[None, start:stop]) / sigma
        density += constant * (
            np.exp(-0.5 * z * z) @ weights[start:stop]
        )
    increments = np.diff(grid)
    if not np.allclose(increments, increments[0], rtol=1e-12, atol=1e-15):
        raise AssertionError("quadrature grid is not uniform")
    dx = float(increments[0])
    mass = density * dx
    mass[0] *= 0.5
    mass[-1] *= 0.5
    raw_mass = float(np.sum(mass))
    tolerance = 1.0e-10 if points_per_sd >= PRIMARY_POINTS_PER_SD else 1.0e-8
    if not math.isfinite(raw_mass) or abs(raw_mass - 1.0) > tolerance:
        raise AssertionError(
            f"posterior quadrature mass {raw_mass} outside tolerance {tolerance}"
        )
    mass /= raw_mass
    return grid, mass, {
        "node_count": count,
        "raw_mass": raw_mass,
        "dx": dx,
        "analytic_normal_tail_bound": 2.0 * 1.776482112077679e-33,
    }


def initial_posterior(
    model: Any, observation: float, points_per_sd: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if observation <= 0.0:
        raise AssertionError(
            "nonpositive initial survey has zero probability under the frozen positive reset model"
        )
    value = math.log(observation / model.survey_scale)
    prior_var = model.reset_log_scale**2
    obs_var = model.observation_scale**2
    posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    posterior_mean = posterior_var * (
        model.reset_log_mean / prior_var + value / obs_var
    )
    grid, mass, diagnostic = mixture_grid(
        np.asarray([posterior_mean]),
        np.asarray([1.0]),
        math.sqrt(posterior_var),
        points_per_sd,
    )
    return grid, mass, {
        **diagnostic,
        "posterior_log_mean": posterior_mean,
        "posterior_log_sd": math.sqrt(posterior_var),
    }


def update_posterior(
    predictive_log_means: np.ndarray,
    prior_weights: np.ndarray,
    process_sd: float,
    observation_sd: float,
    survey_scale: float,
    observation: float,
    points_per_sd: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if observation <= 0.0:
        raise AssertionError(
            "nonpositive conditioning survey has zero probability under the frozen fitted model"
        )
    value = math.log(observation / survey_scale)
    total_sd = math.sqrt(process_sd**2 + observation_sd**2)
    log_component = np.log(prior_weights) + normal_logpdf(
        value, predictive_log_means, total_sd
    )
    evidence = logsumexp(log_component)
    component_weights = np.exp(log_component - evidence)
    posterior_var = (process_sd**2 * observation_sd**2) / (
        process_sd**2 + observation_sd**2
    )
    component_means = (
        observation_sd**2 * predictive_log_means + process_sd**2 * value
    ) / (process_sd**2 + observation_sd**2)
    grid, mass, diagnostic = mixture_grid(
        component_means,
        component_weights,
        math.sqrt(posterior_var),
        points_per_sd,
    )
    return grid, mass, {**diagnostic, "log_evidence": evidence}


def predict_transition(
    model: Any,
    log_grid: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    action: int,
    observation_target: float,
    latent_target: float,
) -> dict[str, float]:
    internal = np.exp(log_grid)
    noiseless = np.asarray(
        model.noiseless_next(internal, capacity, int(action), 0), dtype=np.float64
    )
    if np.any(noiseless <= 0.0) or not np.all(np.isfinite(noiseless)):
        raise AssertionError("frozen positive-state Ricker transition left log-state support")
    means = np.log(noiseless)
    process_sd = float(model.process_scale)
    observation_sd = float(model.observation_scale)
    total_sd = math.sqrt(process_sd**2 + observation_sd**2)
    expected_internal = float(
        np.dot(weights, np.exp(means + 0.5 * process_sd**2))
    )
    latent_mean = float(model.survey_scale * expected_internal)
    observation_mean = float(
        latent_mean * math.exp(0.5 * observation_sd**2)
    )

    def score(target: float, sd: float) -> tuple[float, float]:
        if target < 0.0 or not math.isfinite(target):
            raise AssertionError(f"invalid held-out target {target}")
        if target == 0.0:
            return -math.inf, 0.0
        log_value = math.log(target / model.survey_scale)
        log_density = mixture_logpdf(log_value, means, sd, weights) - math.log(target)
        pit = mixture_cdf(log_value, means, sd, weights)
        return log_density, pit

    latent_log_density, latent_pit = score(float(latent_target), process_sd)
    observation_log_density, observation_pit = score(float(observation_target), total_sd)
    return {
        "latent_mean": latent_mean,
        "observation_mean": observation_mean,
        "latent_log_density": latent_log_density,
        "observation_log_density": observation_log_density,
        "latent_pit": latent_pit,
        "observation_pit": observation_pit,
        "predictive_log_means": means,
        "process_sd": process_sd,
        "observation_sd": observation_sd,
    }


def compare_numerical(primary: float, check: float, kind: str) -> float:
    if math.isinf(primary) and primary == check:
        return 0.0
    difference = abs(primary - check)
    if kind == "mean":
        tolerance = MEAN_ATOL_RAW + MEAN_RTOL * abs(primary)
    elif kind == "nlpd":
        tolerance = NLPD_ATOL
    elif kind == "pit":
        tolerance = PIT_ATOL
    else:
        raise AssertionError(f"unknown convergence metric {kind}")
    if not math.isfinite(difference) or difference > tolerance:
        raise AssertionError(
            f"quadrature convergence failed for {kind}: difference={difference}, "
            f"tolerance={tolerance}"
        )
    return difference


def episode_indices(dataset: Any, episode_ids: Iterable[int]) -> list[np.ndarray]:
    output = []
    for episode in episode_ids:
        idx = np.flatnonzero(dataset.episode_id == int(episode))
        idx = idx[np.argsort(dataset.timestep[idx], kind="stable")]
        if len(idx) == 0:
            raise AssertionError(f"holdout episode {episode} has no rows")
        output.append(idx)
    return output


def evaluate_once(
    model: Any,
    dataset: Any,
    private: Any,
    holdout_ids: tuple[int, ...],
    points_per_sd: int,
    retain_rows: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = {
        key: []
        for key in (
            "latent_mean", "observation_mean", "latent_nlpd", "observation_nlpd",
            "latent_pit", "observation_pit",
        )
    }
    node_counts: list[int] = []
    raw_mass_errors: list[float] = []
    for indices in episode_indices(dataset, holdout_ids):
        first = int(indices[0])
        log_grid, weights, diagnostic = initial_posterior(
            model, float(dataset.observations[first]), points_per_sd
        )
        node_counts.append(int(diagnostic["node_count"]))
        raw_mass_errors.append(abs(float(diagnostic["raw_mass"]) - 1.0))
        capacity = float(model.initial_capacity)
        previous_next_observation: float | None = None
        for position, row_index_value in enumerate(indices):
            row_index = int(row_index_value)
            current_observation = float(dataset.observations[row_index])
            if previous_next_observation is not None and current_observation != previous_next_observation:
                raise AssertionError("public episode observation chain is not exact")
            target_observation = float(dataset.next_observations[row_index])
            target_latent = float(private.next_states[row_index])
            action = int(dataset.actions[row_index])
            prediction = predict_transition(
                model,
                log_grid,
                weights,
                capacity,
                action,
                target_observation,
                target_latent,
            )
            latent_nlpd = -float(prediction["latent_log_density"])
            observation_nlpd = -float(prediction["observation_log_density"])
            values["latent_mean"].append(float(prediction["latent_mean"]))
            values["observation_mean"].append(float(prediction["observation_mean"]))
            values["latent_nlpd"].append(latent_nlpd)
            values["observation_nlpd"].append(observation_nlpd)
            values["latent_pit"].append(float(prediction["latent_pit"]))
            values["observation_pit"].append(float(prediction["observation_pit"]))
            next_capacity = float(model.next_capacity(capacity, action))
            if retain_rows:
                rows.append(
                    {
                        "episode_id": int(dataset.episode_id[row_index]),
                        "timestep": int(dataset.timestep[row_index]),
                        "source_row_index": row_index,
                        "action": action,
                        "capacity_before_internal": capacity,
                        "capacity_after_internal": next_capacity,
                        "current_observed_survey_raw": current_observation,
                        "target_next_observed_survey_raw": target_observation,
                        "predicted_next_observed_survey_mean_raw": prediction[
                            "observation_mean"
                        ],
                        "observed_survey_error_raw": prediction["observation_mean"]
                        - target_observation,
                        "observed_survey_abs_error_raw": abs(
                            prediction["observation_mean"] - target_observation
                        ),
                        "observed_survey_squared_error_raw": (
                            prediction["observation_mean"] - target_observation
                        ) ** 2,
                        "observed_survey_log_predictive_density": prediction[
                            "observation_log_density"
                        ],
                        "observed_survey_nlpd": observation_nlpd,
                        "observed_survey_pit": prediction["observation_pit"],
                        "target_next_true_latent_abundance_raw": target_latent,
                        "predicted_next_latent_abundance_mean_raw": prediction["latent_mean"],
                        "latent_abundance_error_raw": prediction["latent_mean"]
                        - target_latent,
                        "latent_abundance_abs_error_raw": abs(
                            prediction["latent_mean"] - target_latent
                        ),
                        "latent_abundance_squared_error_raw": (
                            prediction["latent_mean"] - target_latent
                        ) ** 2,
                        "latent_abundance_log_predictive_density": prediction[
                            "latent_log_density"
                        ],
                        "latent_abundance_nlpd": latent_nlpd,
                        "latent_abundance_pit": prediction["latent_pit"],
                        "excluded": False,
                        "exclusion_reason": "",
                    }
                )
            capacity = next_capacity
            previous_next_observation = target_observation
            if position + 1 < len(indices):
                log_grid, weights, diagnostic = update_posterior(
                    np.asarray(prediction["predictive_log_means"]),
                    weights,
                    float(prediction["process_sd"]),
                    float(prediction["observation_sd"]),
                    float(model.survey_scale),
                    target_observation,
                    points_per_sd,
                )
                node_counts.append(int(diagnostic["node_count"]))
                raw_mass_errors.append(abs(float(diagnostic["raw_mass"]) - 1.0))
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in values.items()}
    return rows, arrays, {
        "points_per_sd": points_per_sd,
        "tail_sd": TAIL_SD,
        "minimum_nodes": min(node_counts),
        "maximum_nodes": max(node_counts),
        "maximum_quadrature_mass_error": max(raw_mass_errors),
    }


def t_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) != 32:
        raise AssertionError(f"Student-t summary requires 32 episodes, got {len(values)}")
    if np.any(~np.isfinite(values)):
        return {
            "n_episodes": 32,
            "mean": math.inf if np.any(np.isposinf(values)) else math.nan,
            "ci_95_lower": "NOT AVAILABLE: nonfinite episode metric",
            "ci_95_upper": "NOT AVAILABLE: nonfinite episode metric",
            "method": "ordinary two-sided Student-t, df=31",
            "t_critical": T_CRITICAL_DF31,
        }
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    half = T_CRITICAL_DF31 * sd / math.sqrt(len(values))
    return {
        "n_episodes": 32,
        "mean": mean,
        "sample_sd": sd,
        "ci_95_lower": mean - half,
        "ci_95_upper": mean + half,
        "method": "ordinary two-sided Student-t, df=31",
        "t_critical": T_CRITICAL_DF31,
    }


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_episode: list[dict[str, Any]] = []
    targets = {
        "next_observed_survey": (
            "observed_survey_abs_error_raw",
            "observed_survey_squared_error_raw",
            "observed_survey_nlpd",
            "observed_survey_pit",
        ),
        "next_true_latent_abundance": (
            "latent_abundance_abs_error_raw",
            "latent_abundance_squared_error_raw",
            "latent_abundance_nlpd",
            "latent_abundance_pit",
        ),
    }
    summary: dict[str, Any] = {}
    episode_ids = sorted({int(row["episode_id"]) for row in rows})
    if len(episode_ids) != 32:
        raise AssertionError("exactly 32 holdout episodes required")
    for target, fields in targets.items():
        abs_field, square_field, nlpd_field, pit_field = fields
        episode_mae, episode_rmse, episode_nlpd = [], [], []
        for episode in episode_ids:
            subset = [row for row in rows if int(row["episode_id"]) == episode]
            abs_values = np.asarray([row[abs_field] for row in subset], dtype=np.float64)
            square_values = np.asarray([row[square_field] for row in subset], dtype=np.float64)
            nlpd_values = np.asarray([row[nlpd_field] for row in subset], dtype=np.float64)
            mae = float(np.mean(abs_values))
            rmse = float(math.sqrt(float(np.mean(square_values))))
            mean_nlpd = float(np.mean(nlpd_values))
            episode_mae.append(mae)
            episode_rmse.append(rmse)
            episode_nlpd.append(mean_nlpd)
            per_episode.append(
                {
                    "episode_id": episode,
                    "target": target,
                    "transition_count": len(subset),
                    "mae_raw": mae,
                    "rmse_raw": rmse,
                    "mean_per_transition_nlpd": mean_nlpd,
                    "total_nlpd": float(np.sum(nlpd_values)),
                    "mean_pit": float(np.mean([row[pit_field] for row in subset])),
                    "excluded_transition_count": 0,
                }
            )
        all_abs = np.asarray([row[abs_field] for row in rows], dtype=np.float64)
        all_square = np.asarray([row[square_field] for row in rows], dtype=np.float64)
        all_nlpd = np.asarray([row[nlpd_field] for row in rows], dtype=np.float64)
        all_pit = np.asarray([row[pit_field] for row in rows], dtype=np.float64)
        summary[target] = {
            "target_label": (
                "observation/survey-scale error; noisy next survey, never true abundance"
                if target == "next_observed_survey"
                else "latent-state error; evaluator-only untouched true next abundance"
            ),
            "units": "raw Amur tiger abundance/survey-count scale",
            "transition_pooled_descriptive_only": {
                "transition_count": len(rows),
                "mae": float(np.mean(all_abs)),
                "rmse": float(math.sqrt(float(np.mean(all_square)))),
                "mean_per_transition_nlpd": float(np.mean(all_nlpd)),
                "total_nlpd": float(np.sum(all_nlpd)),
                "mean_pit": float(np.mean(all_pit)),
            },
            "episode_equal_with_ordinary_student_t_95_ci": {
                "episode_mae": t_summary(np.asarray(episode_mae)),
                "episode_rmse": t_summary(np.asarray(episode_rmse)),
                "episode_mean_per_transition_nlpd": t_summary(
                    np.asarray(episode_nlpd)
                ),
            },
            "excluded_transitions": 0,
            "exclusion_reasons": [],
        }
    return per_episode, summary


def self_test() -> None:
    means = np.asarray([-0.2, 0.3])
    weights = np.asarray([0.4, 0.6])
    sigma = 0.17
    grid, mass, diagnostic = mixture_grid(means, weights, sigma, 32)
    if abs(float(np.sum(mass)) - 1.0) > 1e-14:
        raise AssertionError("mixture grid self-test mass failed")
    numerical_mean = float(np.dot(mass, np.exp(grid)))
    exact_mean = float(np.dot(weights, np.exp(means + 0.5 * sigma**2)))
    if abs(numerical_mean - exact_mean) > 1e-8:
        raise AssertionError("mixture grid self-test mean failed")
    if not (0.0 < mixture_cdf(0.1, means, sigma, weights) < 1.0):
        raise AssertionError("mixture CDF self-test failed")
    print(json.dumps({"status": "PASS", "diagnostic": diagnostic}, sort_keys=True))


def run_task(task_id: int) -> None:
    if task_id not in CELLS:
        raise ValueError("task id must be 0 or 1")
    cfg = CELLS[task_id]
    token = str(cfg["token"])
    task_dir = OUTPUT / "tasks" / token
    if task_dir.exists():
        raise FileExistsError(f"completed or partial task namespace exists: {task_dir}")
    task_dir.mkdir(parents=True, exist_ok=False)
    start_wall = utc_now()
    start = time.perf_counter()
    failure_path = task_dir / "TASK_FAILURE.json"
    try:
        environment = environment_record(require_slurm=True)
        static = verify_static_inputs(cfg)
        if str(SOURCE_CODE / "src") not in sys.path:
            sys.path.insert(0, str(SOURCE_CODE / "src"))
        from real_ecology_benchmark import faithful_fit  # noqa: PLC0415
        from real_ecology_benchmark.dataset import (  # noqa: PLC0415
            dataset_sha256,
            load_private,
            load_public,
        )

        forbidden_calls: list[str] = []

        def forbidden_fit(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            forbidden_calls.append("attempted")
            raise AssertionError("fitting/optimizer route is prohibited in Stage 1")

        faithful_fit.fit_mechanistic_model = forbidden_fit
        faithful_fit.load_or_fit_mechanistic_model = forbidden_fit
        faithful_fit.build_candidate_bank = forbidden_fit
        faithful_fit.require_torch = forbidden_fit
        fit = faithful_fit._load_fit_cache(  # noqa: SLF001
            cfg["array_path"].parent,
            cfg["cache_key"],
            cfg["public_dataset_hash"],
        )
        if fit.model.parameter_hash() != cfg["parameter_hash"]:
            raise AssertionError("loaded model parameter hash mismatch")
        if fit.fit_cache_key not in ("", cfg["cache_key"]):
            raise AssertionError("loaded fit cache identity mismatch")
        if "torch" in sys.modules:
            raise AssertionError("PyTorch/optimizer runtime was unexpectedly imported")

        dataset = load_public(cfg["public_path"])
        private = load_private(cfg["private_path"])
        private.validate(dataset)
        if dataset_sha256(dataset) != cfg["public_dataset_hash"]:
            raise AssertionError("recomputed authoritative public dataset hash mismatch")
        if faithful_fit.fit_transition_hash(dataset) != cfg["transition_data_hash"]:
            raise AssertionError("recomputed fit-transition hash mismatch")
        if private.metadata.get("public_dataset_sha256") != cfg["public_dataset_hash"]:
            raise AssertionError("private/public sidecar identity mismatch")
        if float(dataset.metadata.get("observation_noise_sigma")) != cfg["sigma"]:
            raise AssertionError("public data noise-cell mismatch")
        if abs(float(fit.model.observation_scale) - cfg["sigma"]) > 0.0:
            raise AssertionError("frozen model observation scale mismatch")

        fit_ids, holdout_ids = faithful_fit.split_history_episodes(
            dataset, HISTORY_FRACTION, FIT_SEED
        )
        fit_ids = tuple(int(value) for value in fit_ids)
        holdout_ids = tuple(int(value) for value in holdout_ids)
        if len(fit_ids) != 128 or len(holdout_ids) != 32:
            raise AssertionError("registered 128/32 split size mismatch")
        if set(fit_ids) & set(holdout_ids):
            raise AssertionError("train/holdout episode overlap")
        all_ids = set(int(value) for value in np.unique(dataset.episode_id))
        if set(fit_ids) | set(holdout_ids) != all_ids:
            raise AssertionError("train/holdout split is not complete")
        if canonical_digest(list(fit_ids)) != FIT_IDS_HASH:
            raise AssertionError("fit episode ID hash mismatch")
        if canonical_digest(list(holdout_ids)) != HOLDOUT_IDS_HASH:
            raise AssertionError("holdout episode ID hash mismatch")
        if fit_ids != tuple(int(value) for value in fit.fit_episode_ids):
            raise AssertionError("reconstructed fit IDs differ from frozen artifact")
        if holdout_ids != tuple(int(value) for value in fit.holdout_episode_ids):
            raise AssertionError("reconstructed holdout IDs differ from frozen artifact")

        rows, primary, primary_diagnostic = evaluate_once(
            fit.model,
            dataset,
            private,
            holdout_ids,
            PRIMARY_POINTS_PER_SD,
            retain_rows=True,
        )
        _, check, check_diagnostic = evaluate_once(
            fit.model,
            dataset,
            private,
            holdout_ids,
            CHECK_POINTS_PER_SD,
            retain_rows=False,
        )
        if len(rows) != 800:
            raise AssertionError(f"expected 800 holdout transitions, got {len(rows)}")
        convergence: dict[str, float] = {}
        for key in ("latent_mean", "observation_mean"):
            differences = [
                compare_numerical(float(a), float(b), "mean")
                for a, b in zip(primary[key], check[key])
            ]
            convergence[f"maximum_{key}_absolute_difference"] = max(differences)
        for key in ("latent_nlpd", "observation_nlpd"):
            differences = [
                compare_numerical(float(a), float(b), "nlpd")
                for a, b in zip(primary[key], check[key])
            ]
            convergence[f"maximum_{key}_absolute_difference"] = max(differences)
        for key in ("latent_pit", "observation_pit"):
            differences = [
                compare_numerical(float(a), float(b), "pit")
                for a, b in zip(primary[key], check[key])
            ]
            convergence[f"maximum_{key}_absolute_difference"] = max(differences)
        for row in rows:
            row["cell"] = token
            row["sigma_obs"] = cfg["sigma"]
            row["independent_unit"] = "holdout_episode"
            row["conditional_on_collection_logs"] = 1
            row["conditional_on_frozen_fits"] = 1
        rows = [
            {key: row[key] for key in ("cell", "sigma_obs", *[k for k in row if k not in {"cell", "sigma_obs"}])}
            for row in rows
        ]
        per_episode, summary = summarize(rows)
        for row in per_episode:
            row["cell"] = token
            row["sigma_obs"] = cfg["sigma"]
        per_episode = [
            {key: row[key] for key in ("cell", "sigma_obs", *[k for k in row if k not in {"cell", "sigma_obs"}])}
            for row in per_episode
        ]
        elapsed = time.perf_counter() - start
        zero_refit = {
            "status": "PASS",
            "fits_executed": 0,
            "optimizer_calls": 0,
            "forbidden_fit_guard_calls": len(forbidden_calls),
            "cache_loader": "archived faithful_fit._load_fit_cache (read-only; no miss route)",
            "cache_key": cfg["cache_key"],
            "cache_array_sha256_before": static["artifact_sha256"]["fit_array"],
            "cache_array_sha256_after": assert_hash(cfg["array_path"], cfg["array_sha256"]),
            "cache_metadata_sha256_before": static["artifact_sha256"]["fit_metadata"],
            "cache_metadata_sha256_after": assert_hash(
                cfg["metadata_path"], cfg["metadata_sha256"]
            ),
            "fit_receipt_sha256_before": static["artifact_sha256"]["fit_receipt"],
            "fit_receipt_sha256_after": assert_hash(
                cfg["receipt_path"], cfg["receipt_sha256"]
            ),
            "torch_imported": "torch" in sys.modules,
        }
        if forbidden_calls or zero_refit["torch_imported"]:
            raise AssertionError("zero-refit guard failed")
        cell_summary = {
            "schema": "e1_tiger_frozen_fit_holdout_cell_summary_v1",
            "status": "PASS",
            "created_utc": utc_now(),
            "cell": token,
            "sigma_obs": cfg["sigma"],
            "independent_statistical_unit": "holdout_episode",
            "fit_episode_count": 128,
            "holdout_episode_count": 32,
            "holdout_transition_count": 800,
            "fit_episode_ids_hash": FIT_IDS_HASH,
            "holdout_episode_ids_hash": HOLDOUT_IDS_HASH,
            "fit_seed": FIT_SEED,
            "fit_config_digest": FIT_CONFIG_DIGEST,
            "conditionality": (
                "Conditional on one collection log and one frozen fit for this noise cell."
            ),
            "targets": summary,
            "numerical_integration": {
                "primary": primary_diagnostic,
                "independent_resolution_check": check_diagnostic,
                "convergence": convergence,
                "thresholds": {
                    "mean_atol_raw": MEAN_ATOL_RAW,
                    "mean_rtol": MEAN_RTOL,
                    "nlpd_atol": NLPD_ATOL,
                    "pit_atol": PIT_ATOL,
                },
            },
            "exclusions": {
                "count": 0,
                "reasons": [],
                "policy": "no silent exclusion of zero, boundary, nonfinite, or low-density targets",
            },
            "elapsed_seconds": elapsed,
        }
        source_record = {
            "cell": token,
            "registration_sha256": static["registration_sha256"],
            "source_sha256": static["source_sha256"],
            "artifact_sha256": static["artifact_sha256"],
            "dataset_route": {
                "public": str(cfg["public_path"]),
                "private_evaluator_sidecar": str(cfg["private_path"]),
                "loader": "archived dataset.load_public/load_private",
            },
            "heldout_targets": {
                "survey": "TrajectoryDataset.next_observations (noisy observed next survey)",
                "latent": "PrivateTrajectoryData.next_states (evaluator-only true next abundance)",
            },
            "fit_and_holdout_ids": {
                "fit_count": len(fit_ids),
                "holdout_count": len(holdout_ids),
                "fit_ids_hash": canonical_digest(list(fit_ids)),
                "holdout_ids_hash": canonical_digest(list(holdout_ids)),
                "overlap_count": len(set(fit_ids) & set(holdout_ids)),
                "complete_partition": set(fit_ids) | set(holdout_ids) == all_ids,
            },
            "private_target_used_as_filter_input": False,
        }
        write_csv_new(task_dir / "PER_TRANSITION.csv", rows)
        write_csv_new(task_dir / "PER_EPISODE.csv", per_episode)
        write_json_new(task_dir / "CELL_SUMMARY.json", cell_summary)
        write_json_new(task_dir / "ZERO_REFIT_EVIDENCE.json", zero_refit)
        write_json_new(task_dir / "SOURCE_AND_SPLIT_EVIDENCE.json", source_record)
        write_json_new(task_dir / "ENVIRONMENT.json", environment)
        receipt = {
            "schema": "e1_tiger_frozen_fit_holdout_task_receipt_v1",
            "status": "PASS",
            "created_utc": utc_now(),
            "started_utc": start_wall,
            "task_id": task_id,
            "cell": token,
            "sigma_obs": cfg["sigma"],
            "registration_sha256": static["registration_sha256"],
            "cache_key": cfg["cache_key"],
            "model_parameter_hash": cfg["parameter_hash"],
            "public_dataset_hash": cfg["public_dataset_hash"],
            "transition_data_hash": cfg["transition_data_hash"],
            "fits_executed": 0,
            "optimizer_calls": 0,
            "holdout_episode_count": 32,
            "holdout_transition_count": 800,
            "excluded_transition_count": 0,
            "elapsed_seconds": elapsed,
        }
        write_json_new(task_dir / "TASK_RECEIPT.json", receipt)
        manifest = {}
        for path in sorted(task_dir.iterdir()):
            if path.is_file() and path.name != "SHA256_MANIFEST.json":
                manifest[path.name] = sha256_file(path)
        write_json_new(
            task_dir / "SHA256_MANIFEST.json",
            {"schema": "sha256_manifest_v1", "files": manifest},
        )
        print(json.dumps(receipt, sort_keys=True))
    except Exception as error:
        if not failure_path.exists():
            try:
                write_json_new(
                    failure_path,
                    {
                        "schema": "e1_tiger_frozen_fit_holdout_task_failure_v1",
                        "status": "FAIL_CLOSED",
                        "created_utc": utc_now(),
                        "task_id": task_id,
                        "cell": token,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "fits_executed": 0,
                    },
                )
            except Exception:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--static-preflight", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.static_preflight is not None:
        cfg = CELLS[args.static_preflight]
        print(
            json.dumps(
                {
                    "environment": environment_record(require_slurm=False),
                    "static": verify_static_inputs(cfg),
                    "cell": cfg["token"],
                    "status": "PASS",
                    "holdout_outcomes_opened": False,
                    "metrics_calculated": False,
                },
                sort_keys=True,
            )
        )
        return
    if args.task_id is None:
        parser.error("--task-id is required outside self-test/preflight")
    run_task(args.task_id)


if __name__ == "__main__":
    main()
