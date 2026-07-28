#!/usr/bin/env python3
"""Search behavior-mixture profiles that bring the failed families into the
[0.15, 0.24] healthy-start collapse band (targeting ~0.20).

Two search directions are needed because the families fail for opposite reasons:

* ``rescue``  (allee_5a, regime_5a): cut the abundance-BLIND probing
  (\\emph{random}, \\emph{threshold_probe} warmup) that crashes fragile
  low-but-healthy starts, feeding \\emph{rescue_dwell}.
* ``regulate`` (theta_5a, theta_10a): a harvest_probe-DOMINANT mixture that
  regulates abundance to a safe mid-band; stocking is counterproductive for theta
  (the high-r/theta map overshoots and crashes above K), so rescue tilts do NOT
  help there.

Both directions interpolate from the default profile by a scalar ``tilt`` in
[0,1].  Each candidate is scored by the mean healthy-start collapse rate over
several seeds; finalists are verified at the production budget (75k) and seed
across multiple seeds, with sigma-independence and full action coverage checked.
Only collector/behavior-mixture settings are tuned; biology, equations, action
tables, safety_threshold, reward/collapse semantics, initial_log_sigma,
low_start_probability and privileged behavior are untouched.

The committed profiles in collector.py can be RECONSTRUCTED by this search (they
were originally found by an equivalent manual sweep, before this script existed;
this is reconstruction, not the original provenance):
  allee_5a, regime_5a : direction=rescue,   tilt=0.30
  theta_5a, theta_10a : direction=regulate, tilt=1.00
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from real_ecology_benchmark.collector import (  # noqa: E402
    CollectorProfile, DEFAULT_PROFILE, calibration_summary, collect_dataset,
)
from real_ecology_benchmark.config import environment_with_kind_defaults, load_config  # noqa: E402
from real_ecology_benchmark.envs import make_env  # noqa: E402

EPISODE_LENGTH = 25
SEARCH_N = 30000
VERIFY_N = 75000
TARGET = 0.20
BAND = (0.15, 0.24)
# Calibration-only seeds, disjoint from the held-out evaluation blocks
# (7001,7051,7101,7151,7201, each spanning 50 episodes -> 7001..7250).  116 is the
# production dataset seed (the binding band check); 217/318 add robustness.
SEARCH_SEEDS = [116, 217]
VERIFY_SEEDS = [116, 217, 318]

# (component anchor at tilt=1, harvest_probe_prob endpoints, rescue_harvest endpoints, donothing endpoints)
DIRECTIONS = {
    # cut random + threshold_probe, feed rescue_dwell; keep harvest_probe for coverage.
    "rescue":   dict(anchor=(0.04, 0.25, 0.67, 0.04), hpp=(0.85, 0.60), rhp=(0.45, 0.20), rdn=(0.60, 0.75)),
    # harvest_probe-dominant regulation to a mid-band; only donothing drops.
    "regulate": dict(anchor=(0.08, 0.70, 0.18, 0.04), hpp=(0.85, 0.85), rhp=(0.45, 0.45), rdn=(0.60, 0.40)),
}


def profile_for(name, direction, tilt):
    spec = DIRECTIONS[direction]
    base = np.asarray(DEFAULT_PROFILE.component_probabilities)
    mix = (1 - tilt) * base + tilt * np.asarray(spec["anchor"])
    mix = mix / mix.sum()
    lerp = lambda ab: round(ab[0] * (1 - tilt) + ab[1] * tilt, 4)
    return CollectorProfile(
        name=name,
        component_probabilities=tuple(round(float(p), 6) for p in mix),
        harvest_probe_prob=lerp(spec["hpp"]),
        rescue_harvest_prob=lerp(spec["rhp"]),
        rescue_donothing_prob=lerp(spec["rdn"]),
    )


def rate(kind, na, profile, seed, n, sigma=0.0):
    base = load_config(str(ROOT / "configs/synthetic_full.yaml")).environment
    cfg = environment_with_kind_defaults(base, kind)
    cfg = replace(cfg, num_actions=na, observation_noise_sigma=sigma)
    ds, truth = collect_dataset(make_env(cfg), n, EPISODE_LENGTH, seed, True, profile)
    return calibration_summary(ds, truth, cfg.safety_threshold)


def search_family(kind, na):
    name = f"{kind}_{na}a"
    print(f"\n=== {name} ===", flush=True)
    best = None
    for direction in DIRECTIONS:
        for tilt in np.round(np.linspace(0.0, 1.0, 11), 2):
            prof = profile_for(name, direction, float(tilt))
            r = float(np.mean([rate(kind, na, prof, s, SEARCH_N)[
                "incident_collapse_rate_healthy_starts"] for s in SEARCH_SEEDS]))
            in_band = BAND[0] <= r <= BAND[1]
            cand = (not in_band, abs(r - TARGET), direction, float(tilt), prof, r)
            if best is None or cand[:2] < best[:2]:
                best = cand
            print(f"  {direction:8} tilt={tilt:.2f} rate={r:.4f}{' *' if in_band else ''}", flush=True)
    _, _, direction, tilt, prof, r = best
    print(f"  -> chosen: direction={direction} tilt={tilt:.2f} (search rate={r:.4f})", flush=True)
    return direction, tilt, prof


# Exact (direction, tilt) that RECONSTRUCTS each committed collector.py profile.
COMMITTED = {
    ("allee", 5): ("rescue", 0.30),
    ("regime", 5): ("rescue", 0.30),
    ("theta", 5): ("regulate", 1.00),
    ("theta", 10): ("regulate", 1.00),
}


def check_committed():
    """Confirm the committed profiles are exactly reproducible from this script."""
    from real_ecology_benchmark.collector import PROFILES
    ok = True
    for (kind, na), (direction, tilt) in COMMITTED.items():
        reproduced = profile_for(f"{kind}_{na}a", direction, tilt)
        committed = PROFILES[(kind, na)]
        same = (reproduced.component_probabilities == committed.component_probabilities
                and reproduced.harvest_probe_prob == committed.harvest_probe_prob
                and reproduced.rescue_harvest_prob == committed.rescue_harvest_prob
                and reproduced.rescue_donothing_prob == committed.rescue_donothing_prob)
        ok &= same
        print(f"  {kind}_{na}a: {direction} tilt={tilt} -> reproduces committed: {same}", flush=True)
    print(f"ALL COMMITTED PROFILES REPRODUCIBLE: {ok}", flush=True)
    if not ok:
        raise SystemExit("a committed profile is not reproducible from (direction, tilt)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="allee:5,regime:5,theta:5,theta:10")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--check-committed", action="store_true",
                    help="only verify the committed profiles are reproducible, then exit")
    args = ap.parse_args()
    if args.check_committed:
        check_committed()
        return
    families = [(k, int(a)) for k, a in (f.split(":") for f in args.families.split(","))]
    chosen = {}
    for kind, na in families:
        d, t, prof = search_family(kind, na)
        chosen[(kind, na)] = prof

    print("\n===== CHOSEN PROFILES (compare against collector.py PROFILES) =====", flush=True)
    for (kind, na), prof in chosen.items():
        print(f'    ("{kind}", {na}): {prof!r},', flush=True)

    if args.verify:
        print("\n===== VERIFY @75k, multi-seed + sigma-independence + coverage =====", flush=True)
        ok = True
        for (kind, na), prof in chosen.items():
            rates = {s: rate(kind, na, prof, s, VERIFY_N)["incident_collapse_rate_healthy_starts"]
                     for s in VERIFY_SEEDS}
            s_indep = rate(kind, na, prof, 116, VERIFY_N, sigma=0.4)[
                "incident_collapse_rate_healthy_starts"]
            summ = rate(kind, na, prof, 116, VERIFY_N)
            freqs = summ["action_frequency"]
            band_all = all(BAND[0] <= v <= BAND[1] for v in rates.values())
            indep = np.isclose(rates[116], s_indep)
            cover = len(freqs) == na
            ok &= band_all and indep and cover
            print(f"  {kind}_{na}a: rates(seeds)={ {k: round(v,4) for k,v in rates.items()} } "
                  f"sigma_indep={indep} band_all={band_all} coverage={len(freqs)}/{na}", flush=True)
        print(f"\nALL PASS: {ok}", flush=True)
        if not ok:
            raise SystemExit("a chosen profile failed verification")


if __name__ == "__main__":
    main()
