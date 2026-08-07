#!/usr/bin/env python3
"""Bounded post-failure numerical feasibility benchmark; not scientific scoring."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from typing import Any

import mpmath as mp
import numpy as np


REPO = Path("/fs04/scratch2/ce25/DeepRL_Population_Models")
OUT = REPO / ".verification/e1_stage1_tail_safe_feasibility_20260806_v1"
ARCHIVE = Path("/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs")
SOURCE_RUN = ARCHIVE / "adapted_32cell_diagnostic_pending"
SOURCE_CODE = SOURCE_RUN / "code"
TARGET_RUN = ARCHIVE / "three_species_ecological_p10_correction_20260723_v1/moor"
PYTHON = Path("/fs04/scratch2/ce25/hphung/conda/envs/poprl/bin/python3.10")
COMMIT = "3291eefbe64b5ab19019120bb0ba34a0a9586a53"
REGISTRATION = OUT / "BENCHMARK_REGISTRATION.json"
REGISTRATION_SHA = OUT / "BENCHMARK_REGISTRATION.json.sha256"
PROTOTYPE_SHA = OUT / "PROTOTYPE_SOURCE_V2.sha256"
EXECUTION_SEAL = OUT / "EXECUTION_SEAL.json"
EXECUTION_SEAL_SHA = OUT / "EXECUTION_SEAL.json.sha256"
FIT_SEED = 47116
HISTORY_FRACTION = 0.8
MEAN_ATOL = 1.0e-6
MEAN_RTOL = 1.0e-9
LOG_ATOL = 1.0e-6
PIT_ATOL = 1.0e-8
DIGIT_ATOL = 1.0e-8
PROD_GH = 16
REF_GH = 24
PROD_CAP = 256
REF_CAP = 192
UNREDUCED_CAP = 100_000
HALF_WIDTH_CAP = 64.0
EXPANSION_CAP = 12
TAIL_RELATIVE_CAP = 1.0e-10
ENDPOINT_LOG_DROP = 45.0


CELLS = {
    "sigma_0p1": {
        "sigma": 0.1,
        "cache_key": "c8eb8b067aaeaa3c26ebdca2f0175cc59dc2d940589db11e2d056f83bd317f47",
        "public_hash": "2cc7611c7bbcb6f72cbda49db73cfc24bde41b297564581dc90a5f60cf3c0d85",
    },
    "sigma_0p2": {
        "sigma": 0.2,
        "cache_key": "bab1b4157d333d18b5cdae3d4738c00efc82b70ebd74e2984c19efa628c17e44",
        "public_hash": "04f38e874ce0f95cdf69acfdd639b511c1c828972dd7e043193c751609768080",
    },
}


CASES = [
    ("sigma_0p1", 129, 0, ["hash_selected_step0"]),
    ("sigma_0p1", 18, 0, ["known_failed_step0"]),
    ("sigma_0p1", 18, 4, ["known_failed_mean_row"]),
    ("sigma_0p1", 113, 6, ["previous_prediction_worst_1"]),
    ("sigma_0p1", 113, 7, ["previous_prediction_worst_2"]),
    ("sigma_0p1", 113, 8, ["previous_prediction_worst_3"]),
    ("sigma_0p1", 113, 9, ["previous_prediction_worst_4"]),
    ("sigma_0p1", 113, 10, ["previous_prediction_worst_5"]),
    ("sigma_0p1", 101, 5, ["disclosed_lower_survey_ratio"]),
    ("sigma_0p1", 132, 8, ["disclosed_upper_survey_ratio"]),
    ("sigma_0p1", 18, 24, ["central_survey_ratio_candidate"]),
    ("sigma_0p2", 48, 0, ["hash_selected_step0"]),
    ("sigma_0p2", 2, 0, ["recovered_exact_failed_row"]),
    ("sigma_0p2", 101, 5, ["disclosed_lower_survey_ratio"]),
    ("sigma_0p2", 132, 8, ["disclosed_upper_survey_ratio"]),
    ("sigma_0p2", 28, 18, ["central_survey_ratio_candidate"]),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def logsumexp(values: np.ndarray) -> float:
    m = float(np.max(values))
    return m + math.log(float(np.sum(np.exp(values - m))))


def normal_logpdf(z: float, means: np.ndarray, sds: np.ndarray) -> np.ndarray:
    x = (float(z) - means) / sds
    return -0.5 * x * x - np.log(sds) - 0.5 * math.log(2.0 * math.pi)


def normal_logcdf(x: float) -> float:
    if x < -8.0:
        # Mills-series seed is stable in the registered tail range.
        xx = -x
        return -0.5 * xx * xx - math.log(xx) - 0.5 * math.log(2 * math.pi) + math.log1p(-1 / (xx * xx) + 3 / (xx**4))
    p = 0.5 * math.erfc(-x / math.sqrt(2.0))
    return math.log(p)


def normal_logsf(x: float) -> float:
    return normal_logcdf(-x)


@dataclass(frozen=True)
class Mix:
    logw: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    atom_logw: float = -math.inf

    def normalized(self) -> "Mix":
        parts = list(self.logw)
        if math.isfinite(self.atom_logw):
            parts.append(self.atom_logw)
        total = logsumexp(np.asarray(parts, dtype=np.float64))
        return Mix(self.logw - total, self.means, self.sds, self.atom_logw - total)

    @property
    def count(self) -> int:
        return len(self.logw)


def mix_hash(mix: Mix) -> str:
    h = hashlib.sha256()
    for array in (mix.logw, mix.means, mix.sds):
        h.update(np.asarray(array, dtype="<f8").tobytes())
    h.update(np.asarray([mix.atom_logw], dtype="<f8").tobytes())
    return h.hexdigest()


def initial_mix(model: Any, observation: float) -> Mix:
    z = math.log(observation / model.survey_scale)
    vp = model.reset_log_scale**2
    vo = model.observation_scale**2
    v = 1.0 / (1.0 / vp + 1.0 / vo)
    mean = v * (model.reset_log_mean / vp + z / vo)
    return Mix(np.asarray([0.0]), np.asarray([mean]), np.asarray([math.sqrt(v)]), -math.inf)


def gh_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0) * x, w / math.sqrt(math.pi)


def transition_mix(model: Any, mix: Mix, capacity: float, action: int, gh_n: int) -> Mix:
    nodes, weights = gh_rule(gh_n)
    source = mix.means[:, None] + mix.sds[:, None] * nodes[None, :]
    internal = np.exp(source)
    loc = np.asarray(model.noiseless_next(internal, capacity, action, 0), dtype=np.float64)
    if np.any(loc <= 0.0) or np.any(~np.isfinite(loc)):
        raise AssertionError("positive continuous component left positive Ricker support")
    means = np.log(loc).ravel()
    logw = (mix.logw[:, None] + np.log(weights)[None, :]).ravel()
    sds = np.full(len(means), float(model.process_scale), dtype=np.float64)
    atom_logw = mix.atom_logw
    if math.isfinite(mix.atom_logw) and float(model.stocking[action]) > 0.0:
        zero_loc = float(model.noiseless_next(0.0, capacity, action, 0))
        if zero_loc <= 0.0:
            raise AssertionError("stocking failed to move zero atom to positive support")
        logw = np.append(logw, mix.atom_logw)
        means = np.append(means, math.log(zero_loc))
        sds = np.append(sds, float(model.process_scale))
        atom_logw = -math.inf
    return Mix(logw, means, sds, atom_logw).normalized()


def positive_update(pred: Mix, observation: float, survey_scale: float, obs_sd: float) -> Mix:
    if observation <= 0.0:
        raise AssertionError("benchmark positive-survey update received nonpositive survey")
    z = math.log(observation / survey_scale)
    total = np.sqrt(pred.sds**2 + obs_sd**2)
    evidence = pred.logw + normal_logpdf(z, pred.means, total)
    post_var = 1.0 / (1.0 / pred.sds**2 + 1.0 / obs_sd**2)
    post_mean = post_var * (pred.means / pred.sds**2 + z / obs_sd**2)
    # Positive observations have exactly zero likelihood under the zero atom.
    return Mix(evidence, post_mean, np.sqrt(post_var), -math.inf).normalized()


def weighted_moment(logw: np.ndarray, means: np.ndarray, sds: np.ndarray) -> tuple[float, float, float]:
    lw = logsumexp(logw)
    w = np.exp(logw - lw)
    mean = float(np.dot(w, means))
    var = float(np.dot(w, sds**2 + (means - mean) ** 2))
    return lw, mean, math.sqrt(max(var, 1.0e-30))


def envelope_reduce(mix: Mix, cap: int) -> tuple[Mix, dict[str, Any]]:
    if mix.count <= cap:
        return mix, {"before": mix.count, "after": mix.count, "protected": mix.count, "cap_reached": False}
    left = float(np.min(mix.means - 16.0 * mix.sds))
    right = float(np.max(mix.means + 16.0 * mix.sds))
    anchors = np.linspace(left, right, 2049)
    protected: set[int] = {int(np.argmin(mix.means)), int(np.argmax(mix.means))}
    for start in range(0, len(anchors), 64):
        block = anchors[start:start + 64]
        scores = mix.logw[:, None] + normal_logpdf(0.0, mix.means[:, None] - block[None, :], mix.sds[:, None])
        protected.update(int(v) for v in np.argmax(scores, axis=0))
    if len(protected) >= cap:
        return mix, {"before": mix.count, "after": mix.count, "protected": len(protected), "cap_reached": True}
    rest = np.asarray(sorted(set(range(mix.count)) - protected), dtype=np.int64)
    protected_array = np.asarray(sorted(protected), dtype=np.int64)
    slots = cap - len(protected_array)
    order = rest[np.argsort(mix.means[rest], kind="stable")]
    rw = np.exp(mix.logw[order] - logsumexp(mix.logw[order]))
    cumulative = np.cumsum(rw)
    groups = np.minimum((cumulative * slots).astype(int), slots - 1)
    out_lw = list(mix.logw[protected_array])
    out_m = list(mix.means[protected_array])
    out_s = list(mix.sds[protected_array])
    for group in range(slots):
        idx = order[groups == group]
        if len(idx):
            lw, mean, sd = weighted_moment(mix.logw[idx], mix.means[idx], mix.sds[idx])
            out_lw.append(lw); out_m.append(mean); out_s.append(sd)
    result = Mix(np.asarray(out_lw), np.asarray(out_m), np.asarray(out_s), mix.atom_logw).normalized()
    return result, {"before": mix.count, "after": result.count, "protected": len(protected), "cap_reached": False, "anchor_left": left, "anchor_right": right, "anchors": len(anchors)}


def reference_reduce(mix: Mix, cap: int) -> tuple[Mix, dict[str, Any]]:
    if mix.count <= cap:
        return mix, {"before": mix.count, "after": mix.count, "cap_reached": False}
    order = np.argsort(mix.means, kind="stable")
    w = np.exp(mix.logw[order] - logsumexp(mix.logw[order]))
    groups = np.minimum((np.cumsum(w) * cap).astype(int), cap - 1)
    lw: list[float] = []; means: list[float] = []; sds: list[float] = []
    for group in range(cap):
        idx = order[groups == group]
        if len(idx):
            a, b, c = weighted_moment(mix.logw[idx], mix.means[idx], mix.sds[idx])
            lw.append(a); means.append(b); sds.append(c)
    return Mix(np.asarray(lw), np.asarray(means), np.asarray(sds), mix.atom_logw).normalized(), {"before": mix.count, "after": len(lw), "cap_reached": False}


def score_analytic(mix: Mix, target: float, survey_scale: float, extra_sd: float) -> dict[str, Any]:
    if target <= 0.0:
        raise AssertionError("registered benchmark target is not positive")
    z = math.log(target / survey_scale)
    sds = np.sqrt(mix.sds**2 + extra_sd**2)
    comp = mix.logw + normal_logpdf(z, mix.means, sds)
    log_density = logsumexp(comp) - math.log(target)
    lc = np.asarray([lw + normal_logcdf((z - m) / s) for lw, m, s in zip(mix.logw, mix.means, sds)])
    ls = np.asarray([lw + normal_logsf((z - m) / s) for lw, m, s in zip(mix.logw, mix.means, sds)])
    log_cdf = logsumexp(lc)
    log_sf = logsumexp(ls)
    dominant = int(np.argmax(comp))
    return {"log_density": log_density, "log_cdf": log_cdf, "log_survival": log_sf, "pit": math.exp(log_cdf) if log_cdf > -745 else 0.0, "dominant_component": dominant, "dominant_component_mean": float(mix.means[dominant]), "dominant_component_log_weight": float(mix.logw[dominant])}


def predictive_mean(mix: Mix, survey_scale: float, extra_sd: float) -> float:
    values = mix.logw + mix.means + 0.5 * (mix.sds**2 + extra_sd**2)
    return survey_scale * math.exp(logsumexp(values))


def mp_logsumexp(values: list[mp.mpf]) -> mp.mpf:
    maximum = max(values)
    return maximum + mp.log(mp.fsum(mp.exp(v - maximum) for v in values))


class ReferenceEvaluator:
    def __init__(self, model: Any, posterior: Mix, capacity: float, action: int, dps: int):
        self.model = model; self.posterior = posterior; self.capacity = capacity; self.action = action
        self.dps = dps; self.evaluations = 0; self.panels = 0; self.expansions = 0
        self.max_half_width = 0.0; self.max_tail_ratio = 0.0; self.all_interior = True; self.cap_reached = False
        self.lw = [mp.mpf(repr(float(x))) for x in posterior.logw]
        self.mm = [mp.mpf(repr(float(x))) for x in posterior.means]
        self.ss = [mp.mpf(repr(float(x))) for x in posterior.sds]

    def log_posterior(self, u: mp.mpf) -> mp.mpf:
        half = mp.mpf("0.5"); log2pi = mp.log(2 * mp.pi)
        return mp_logsumexp([lw - mp.log(sd) - half * log2pi - half * ((u - mean) / sd) ** 2 for lw, mean, sd in zip(self.lw, self.mm, self.ss)])

    def transition_location(self, u: mp.mpf) -> mp.mpf:
        x = mp.exp(u); a = self.action
        stock = mp.mpf(repr(float(self.model.stocking[a])))
        managed = x + stock
        next_cap = min(max(mp.mpf(repr(float(self.capacity))) + mp.mpf(repr(float(self.model.capacity_increment[a]))), mp.mpf(repr(float(self.model.initial_capacity)))), mp.mpf(repr(float(self.model.capacity_ceiling))))
        exponent = mp.mpf(repr(float(self.model.growth[a]))) * (1 - managed / next_cap) - mp.mpf(repr(float(self.model.mortality[a])))
        exponent = min(max(exponent, mp.mpf(-40)), mp.mpf(40))
        return mp.log(managed) + exponent

    def _integrate(self, log_integrand, label: str) -> tuple[mp.mpf, dict[str, Any]]:
        center0 = float(self.posterior.means[int(np.argmax(self.posterior.logw))])
        spread = max(0.05, 8.0 * float(np.max(self.posterior.sds)))
        base_center = mp.mpf(repr(center0))
        half_width = mp.mpf(repr(spread))
        final_value = None; final_diag = None
        for expansion in range(EXPANSION_CAP + 1):
            left = base_center - half_width; right = base_center + half_width
            # Re-locate the global sampled maximum after every expansion. This
            # prevents an initially truncated tail mode from remaining unseen.
            scan = [left + (right - left) * i / 512 for i in range(513)]
            scan_values = [log_integrand(x) for x in scan]
            imax = int(max(range(len(scan_values)), key=lambda i: scan_values[i]))
            lo = scan[max(0, imax - 1)]; hi = scan[min(len(scan) - 1, imax + 1)]
            phi = (mp.sqrt(5) - 1) / 2
            for _ in range(80):
                c = hi - phi * (hi - lo); d = lo + phi * (hi - lo)
                if log_integrand(c) < log_integrand(d): lo = c
                else: hi = d
            center = (lo + hi) / 2; peak = log_integrand(center)
            gl = log_integrand(left); gr = log_integrand(right)
            segments = 16
            points = [left + (right - left) * i / segments for i in range(segments + 1)]
            def shifted(u):
                self.evaluations += 1
                return mp.exp(log_integrand(u) - peak)
            pieces = [mp.quad(shifted, [points[i], points[i + 1]]) for i in range(segments)]
            self.panels += segments
            integral_shifted = mp.fsum(pieces)
            eps = max(mp.mpf("1e-8"), half_width * mp.mpf("1e-6"))
            dl = (log_integrand(left + eps) - gl) / eps
            dr = (gr - log_integrand(right - eps)) / eps
            left_bound = mp.exp(gl - peak) / max(dl, mp.mpf("1e-40")) if dl > 0 else mp.inf
            right_bound = mp.exp(gr - peak) / max(-dr, mp.mpf("1e-40")) if dr < 0 else mp.inf
            tail_ratio = (left_bound + right_bound) / max(integral_shifted, mp.mpf("1e-100000"))
            endpoint_drop = min(peak - gl, peak - gr)
            interior = left < center < right
            final_value = mp.exp(peak) * integral_shifted
            final_diag = {"label": label, "maximizer": str(center), "left": str(left), "right": str(right), "half_width": float(half_width), "endpoint_log_drop": float(endpoint_drop), "tail_bound_ratio": float(tail_ratio), "interior": bool(interior), "expansions": expansion}
            if interior and endpoint_drop >= ENDPOINT_LOG_DROP and tail_ratio <= TAIL_RELATIVE_CAP:
                self.expansions += expansion; self.max_half_width = max(self.max_half_width, float(half_width)); self.max_tail_ratio = max(self.max_tail_ratio, float(tail_ratio)); self.all_interior &= bool(interior)
                return final_value, final_diag
            half_width *= mp.mpf("1.75")
            if float(half_width) > HALF_WIDTH_CAP:
                break
        self.cap_reached = True
        assert final_diag is not None and final_value is not None
        return final_value, final_diag

    def score(self, target: float, extra_sd: float, label: str) -> dict[str, Any]:
        z = mp.log(mp.mpf(repr(target)) / mp.mpf(repr(float(self.model.survey_scale))))
        sd = mp.sqrt(mp.mpf(repr(float(self.model.process_scale))) ** 2 + mp.mpf(repr(extra_sd)) ** 2)
        lognorm = mp.log(sd) + mp.log(2 * mp.pi) / 2
        def log_density_integrand(u):
            loc = self.transition_location(u)
            return self.log_posterior(u) - lognorm - (z - loc) ** 2 / (2 * sd**2)
        density_logscale, ddiag = self._integrate(log_density_integrand, f"{label}_density")
        def log_cdf_integrand(u):
            loc = self.transition_location(u)
            p = mp.erfc(-(z - loc) / (sd * mp.sqrt(2))) / 2
            return self.log_posterior(u) + mp.log(p)
        cdf, cdiag = self._integrate(log_cdf_integrand, f"{label}_cdf")
        def log_sf_integrand(u):
            loc = self.transition_location(u)
            p = mp.erfc((z - loc) / (sd * mp.sqrt(2))) / 2
            return self.log_posterior(u) + mp.log(p)
        sf, sdiag = self._integrate(log_sf_integrand, f"{label}_survival")
        return {"log_density": float(mp.log(density_logscale) - mp.log(mp.mpf(repr(target)))), "log_cdf": float(mp.log(cdf)), "log_survival": float(mp.log(sf)), "pit": float(cdf), "integration": [ddiag, cdiag, sdiag]}

    def mean(self, extra_sd: float) -> tuple[float, dict[str, Any]]:
        factor = mp.mpf(repr(float(self.model.process_scale))) ** 2 + mp.mpf(repr(extra_sd)) ** 2
        def log_integrand(u): return self.log_posterior(u) + self.transition_location(u) + factor / 2
        value, diag = self._integrate(log_integrand, "predictive_mean")
        return float(mp.mpf(repr(float(self.model.survey_scale))) * value), diag


def load_cell(token: str):
    cfg = CELLS[token]
    sys.path.insert(0, str(SOURCE_CODE / "src"))
    from real_ecology_benchmark import faithful_fit
    from real_ecology_benchmark.dataset import load_private, load_public
    fit = faithful_fit._load_fit_cache(TARGET_RUN / "fit_cache", cfg["cache_key"], cfg["public_hash"])
    public = load_public(TARGET_RUN / f"datasets/regime_hidden/reward_safe/amur_tiger/ricker/{token}/public.npz")
    private = load_private(TARGET_RUN / f"private/regime_hidden/reward_safe/amur_tiger/ricker/{token}/truth.npz")
    private.validate(public)
    return fit, public, private


def verify_seals() -> dict[str, str]:
    side = REGISTRATION_SHA.read_text().strip().split()
    if len(side) != 2 or side[1] != REGISTRATION.name or sha256_file(REGISTRATION) != side[0]:
        raise AssertionError("benchmark registration seal mismatch")
    proto = PROTOTYPE_SHA.read_text().strip().split()
    source = Path(__file__).resolve()
    if len(proto) != 2 or proto[1] != source.name or sha256_file(source) != proto[0]:
        raise AssertionError("prototype source seal mismatch")
    execution_side = EXECUTION_SEAL_SHA.read_text().strip().split()
    if len(execution_side) != 2 or execution_side[1] != EXECUTION_SEAL.name or sha256_file(EXECUTION_SEAL) != execution_side[0]:
        raise AssertionError("execution seal mismatch")
    return {"registration_sha256": side[0], "prototype_sha256": proto[0], "execution_seal_sha256": execution_side[0]}


def run_case(task_id: int) -> None:
    seals = verify_seals()
    token, episode, timestep, labels = CASES[task_id]
    start_wall = time.perf_counter(); start_cpu = time.process_time()
    fit, public, private = load_cell(token); model = fit.model
    ids = tuple(int(v) for v in fit.holdout_episode_ids)
    if episode not in ids:
        raise AssertionError("registered case is not in frozen holdout split")
    indices = np.flatnonzero(public.episode_id == episode)
    indices = indices[np.argsort(public.timestep[indices], kind="stable")]
    target_positions = np.flatnonzero(public.timestep[indices] == timestep)
    if len(target_positions) != 1:
        raise AssertionError("registered timestep identity is not unique")
    target_position = int(target_positions[0]); target_index = int(indices[target_position])
    prod = initial_mix(model, float(public.observations[int(indices[0])]))
    ref60 = initial_mix(model, float(public.observations[int(indices[0])]))
    ref80 = initial_mix(model, float(public.observations[int(indices[0])]))
    capacity_prod = float(model.initial_capacity); capacity_ref = float(model.initial_capacity)
    history: list[dict[str, Any]] = []
    caps_approached = False
    for position in range(target_position + 1):
        i = int(indices[position]); action = int(public.actions[i])
        before = mix_hash(prod)
        pred_a = transition_mix(model, prod, capacity_prod, action, PROD_GH)
        if pred_a.count > UNREDUCED_CAP:
            raise AssertionError("unreduced benchmark safety cap reached")
        pred_b, red = envelope_reduce(pred_a, PROD_CAP)
        caps_approached |= bool(red["cap_reached"])
        ref_pred60 = transition_mix(model, ref60, capacity_ref, action, REF_GH)
        ref_pred80 = transition_mix(model, ref80, capacity_ref, action, REF_GH)
        ref_pred60, rr60 = reference_reduce(ref_pred60, REF_CAP)
        ref_pred80, rr80 = reference_reduce(ref_pred80, REF_CAP)
        caps_approached |= bool(rr60["cap_reached"] or rr80["cap_reached"])
        next_capacity_prod = float(model.next_capacity(capacity_prod, action))
        next_capacity_ref = float(model.next_capacity(capacity_ref, action))
        if position < target_position:
            observation = float(public.next_observations[i])
            next_prod = positive_update(pred_b, observation, float(model.survey_scale), float(model.observation_scale))
            next_prod, post_red = envelope_reduce(next_prod, PROD_CAP)
            next_ref60 = positive_update(ref_pred60, observation, float(model.survey_scale), float(model.observation_scale))
            next_ref60, post_ref60 = reference_reduce(next_ref60, REF_CAP)
            next_ref80 = positive_update(ref_pred80, observation, float(model.survey_scale), float(model.observation_scale))
            next_ref80, post_ref80 = reference_reduce(next_ref80, REF_CAP)
            history.append({"position": position, "source_row_index": i, "posterior_hash_before": before, "predictive_components_unreduced": pred_a.count, "predictive_components_reduced": pred_b.count, "posterior_components_after": next_prod.count, "reference_components_60": next_ref60.count, "reference_components_80": next_ref80.count, "reference_digit_mean_difference": abs(predictive_mean(next_ref60, float(model.survey_scale), 0.0) - predictive_mean(next_ref80, float(model.survey_scale), 0.0)), "production_reduction": red, "reference_reduction_60": rr60, "reference_reduction_80": rr80})
            prod, ref60, ref80 = next_prod, next_ref60, next_ref80
            capacity_prod, capacity_ref = next_capacity_prod, next_capacity_ref
    frozen_hashes = {"filter": mix_hash(prod), "predictive_unreduced": mix_hash(pred_a), "predictive_reduced": mix_hash(pred_b)}
    targets = {"latent": float(private.next_states[target_index]), "survey": float(public.next_observations[target_index])}
    analytic: dict[str, Any] = {"unreduced": {}, "reduced": {}}
    for variant, mixture in (("unreduced", pred_a), ("reduced", pred_b)):
        analytic[variant]["latent_mean"] = predictive_mean(mixture, float(model.survey_scale), 0.0)
        analytic[variant]["survey_mean"] = predictive_mean(mixture, float(model.survey_scale), float(model.observation_scale))
        analytic[variant]["latent"] = score_analytic(mixture, targets["latent"], float(model.survey_scale), 0.0)
        analytic[variant]["survey"] = score_analytic(mixture, targets["survey"], float(model.survey_scale), float(model.observation_scale))
    score_order_hashes: dict[str, Any] = {}
    for order in (("latent", "survey"), ("survey", "latent")):
        _ = [score_analytic(pred_b, targets[k], float(model.survey_scale), 0.0 if k == "latent" else float(model.observation_scale)) for k in order]
        score_order_hashes["_then_".join(order)] = {"filter": mix_hash(prod), "predictive": mix_hash(pred_b)}
    synthetic = score_analytic(pred_b, targets["latent"] * 1.137, float(model.survey_scale), 0.0)
    del synthetic
    after_synthetic_hash = mix_hash(prod)
    if target_position + 1 < len(indices):
        actual_next = float(public.next_observations[target_index])
        follow1 = positive_update(pred_b, actual_next, float(model.survey_scale), float(model.observation_scale))
        _ = score_analytic(pred_b, targets["survey"], float(model.survey_scale), float(model.observation_scale))
        follow2 = positive_update(pred_b, actual_next, float(model.survey_scale), float(model.observation_scale))
        subsequent_equal = mix_hash(follow1) == mix_hash(follow2)
    else:
        subsequent_equal = True
    references: dict[str, Any] = {}
    for dps, refmix in ((60, ref60), (80, ref80)):
        with mp.workdps(dps):
            evaluator = ReferenceEvaluator(model, refmix, capacity_ref, int(public.actions[target_index]), dps)
            latent_mean, mean_diag = evaluator.mean(0.0)
            survey_mean = latent_mean * math.exp(0.5 * float(model.observation_scale) ** 2)
            latent = evaluator.score(targets["latent"], 0.0, "latent")
            survey = evaluator.score(targets["survey"], float(model.observation_scale), "survey")
            references[str(dps)] = {"latent_mean": latent_mean, "survey_mean": survey_mean, "latent": latent, "survey": survey, "resource": {"evaluations": evaluator.evaluations, "adaptive_panels": evaluator.panels, "domain_expansions": evaluator.expansions, "maximum_half_width": evaluator.max_half_width, "maximum_tail_bound_ratio": evaluator.max_tail_ratio, "all_maximizers_interior": evaluator.all_interior, "cap_reached": evaluator.cap_reached}, "mean_integration": mean_diag}
            caps_approached |= evaluator.cap_reached
    ref = references["80"]
    errors: dict[str, Any] = {}
    checks: list[bool] = []
    for variant in ("unreduced", "reduced"):
        vm = analytic[variant]
        errors[variant] = {
            "latent_mean_raw": abs(vm["latent_mean"] - ref["latent_mean"]),
            "survey_mean_raw": abs(vm["survey_mean"] - ref["survey_mean"]),
            "latent_log_density_nat": abs(vm["latent"]["log_density"] - ref["latent"]["log_density"]),
            "latent_log_cdf_nat": abs(vm["latent"]["log_cdf"] - ref["latent"]["log_cdf"]),
            "latent_log_survival_nat": abs(vm["latent"]["log_survival"] - ref["latent"]["log_survival"]),
            "survey_log_density_nat": abs(vm["survey"]["log_density"] - ref["survey"]["log_density"]),
            "survey_log_cdf_nat": abs(vm["survey"]["log_cdf"] - ref["survey"]["log_cdf"]),
            "survey_log_survival_nat": abs(vm["survey"]["log_survival"] - ref["survey"]["log_survival"]),
            "survey_pit_absolute": abs(vm["survey"]["pit"] - ref["survey"]["pit"]),
        }
        e = errors[variant]
        checks.extend([e["latent_mean_raw"] <= MEAN_ATOL + MEAN_RTOL * abs(ref["latent_mean"]), e["survey_mean_raw"] <= MEAN_ATOL + MEAN_RTOL * abs(ref["survey_mean"]), e["latent_log_density_nat"] <= LOG_ATOL, e["latent_log_cdf_nat"] <= LOG_ATOL, e["latent_log_survival_nat"] <= LOG_ATOL, e["survey_log_density_nat"] <= LOG_ATOL, e["survey_log_cdf_nat"] <= LOG_ATOL, e["survey_log_survival_nat"] <= LOG_ATOL, e["survey_pit_absolute"] <= PIT_ATOL])
    digit_errors = {"latent_mean_raw": abs(references["60"]["latent_mean"] - ref["latent_mean"]), "survey_mean_raw": abs(references["60"]["survey_mean"] - ref["survey_mean"]), "latent_log_density_nat": abs(references["60"]["latent"]["log_density"] - ref["latent"]["log_density"]), "latent_log_cdf_nat": abs(references["60"]["latent"]["log_cdf"] - ref["latent"]["log_cdf"]), "latent_log_survival_nat": abs(references["60"]["latent"]["log_survival"] - ref["latent"]["log_survival"]), "survey_log_density_nat": abs(references["60"]["survey"]["log_density"] - ref["survey"]["log_density"]), "survey_log_cdf_nat": abs(references["60"]["survey"]["log_cdf"] - ref["survey"]["log_cdf"]), "survey_log_survival_nat": abs(references["60"]["survey"]["log_survival"] - ref["survey"]["log_survival"])}
    checks.extend(v <= DIGIT_ATOL for v in digit_errors.values())
    target_independence = {"hashes_frozen_before_targets": frozen_hashes, "score_order_hashes": score_order_hashes, "after_synthetic_target_filter_hash": after_synthetic_hash, "score_order_invariant": len({v["filter"] + v["predictive"] for v in score_order_hashes.values()}) == 1, "synthetic_target_invariant": after_synthetic_hash == frozen_hashes["filter"], "subsequent_filter_invariant": subsequent_equal}
    checks.extend([target_independence["score_order_invariant"], target_independence["synthetic_target_invariant"], target_independence["subsequent_filter_invariant"], not caps_approached])
    if "central_survey_ratio_candidate" in labels:
        checks.append(1.0e-12 < ref["survey"]["pit"] < 1.0 - 1.0e-12)
    elapsed_wall = time.perf_counter() - start_wall; elapsed_cpu = time.process_time() - start_cpu
    output = {"schema": "e1_stage1_tail_safe_feasibility_case_v1", "status": "PASS" if all(checks) else "FAIL", "created_utc": utc_now(), "task_id": task_id, "cell": token, "episode_id": episode, "timestep": timestep, "source_row_index": target_index, "labels": labels, "action": int(public.actions[target_index]), "current_survey_raw": float(public.observations[target_index]), "targets": targets, "seals": seals, "support": {"zero_atom_carried_exactly": True, "posterior_zero_atom_log_weight": prod.atom_logw, "continuous_support_positive": True}, "production": analytic, "reference": references, "errors_vs_80_digit_reference": errors, "reference_60_vs_80_errors": digit_errors, "history_update_error_record": history, "target_independence": target_independence, "mixture_reduction": {"final": red, "pointwise_tail_acceptance": errors["reduced"], "unreduced_components": pred_a.count, "reduced_components": pred_b.count}, "criteria": {"mean": "1e-6 raw + 1e-9*abs(reference)", "log_quantities_nat": LOG_ATOL, "central_pit_absolute": PIT_ATOL, "digit_agreement": DIGIT_ATOL, "all_pass": all(checks), "cap_reached_or_approached": caps_approached}, "resources": {"cpu_seconds": elapsed_cpu, "wall_seconds": elapsed_wall, "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "production_gh_nodes": PROD_GH, "reference_gh_nodes": REF_GH, "unreduced_safety_cap": UNREDUCED_CAP, "production_component_cap": PROD_CAP, "reference_component_cap": REF_CAP}}
    task_dir = OUT / "tasks" / f"task_{task_id:02d}"
    task_dir.mkdir(parents=True, exist_ok=False)
    write_json_new(task_dir / "CASE_RESULT.json", output)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.task_id < len(CASES): raise SystemExit("invalid task ID")
    if sys.version_info[:3] != (3, 10, 14) or np.__version__ != "2.2.6" or Path(sys.executable).resolve() != PYTHON.resolve(): raise AssertionError("runtime mismatch")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1": raise AssertionError("one Slurm CPU required")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE"):
        if os.environ.get(key) != "1": raise AssertionError(f"{key} must equal one")
    run_case(args.task_id)


if __name__ == "__main__":
    main()
