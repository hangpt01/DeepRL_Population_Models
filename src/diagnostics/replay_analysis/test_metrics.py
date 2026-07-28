"""Unit tests: every assertion is an analytically known value."""
import numpy as np

import metrics as M
from constants import (COLLAPSE_PENALTY, DISCOUNT_SUM, SPECIES,
                       dominated_actions, max_reachable_abundance)
from synthetic import make_evd_log, make_plus_log

TOL = 1e-9
_fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        _fails.append(name)


print("constants")
check("vulture max reachable abundance == 42.94",
      abs(max_reachable_abundance("egyptian_vulture") - 42.94) < 0.01,
      f"{max_reachable_abundance('egyptian_vulture'):.4f}")
check("vulture below its own safety threshold",
      max_reachable_abundance("egyptian_vulture") < SPECIES["egyptian_vulture"]["s_safe"])
check("vulture dominated actions == {5,6,7,8,9}",
      set(dominated_actions("egyptian_vulture")) == {5, 6, 7, 8, 9},
      str(sorted(dominated_actions("egyptian_vulture"))))
check("tiger dominated actions == {5,6}",
      set(dominated_actions("amur_tiger")) == {5, 6})
check("fox has no dominated actions",
      dominated_actions("crab_eating_fox") == {})
check("discount sum == 18.4611",
      abs(DISCOUNT_SUM - 18.46110) < 1e-4, f"{DISCOUNT_SUM:.5f}")
check("always-unsafe penalty offset == -184.611",
      abs(-COLLAPSE_PENALTY * DISCOUNT_SUM + 184.611) < 1e-2)

print("\nM5 raw vs centred (action-independent disagreement)")
log = make_plus_log(action_independent_disagreement=True)
m5 = M.m5_raw_vs_centred(log)
check("centred variance == 0", abs(m5["centred_mean"]) < 1e-18, f"{m5['centred_mean']:.3e}")
check("raw variance > 0", m5["raw_mean"] > 1.0, f"{m5['raw_mean']:.3f}")
check("action-independent fraction == 1", abs(m5["action_independent_fraction"] - 1.0) < 1e-12)

log2 = make_plus_log(action_independent_disagreement=False, seed=7)
m5b = M.m5_raw_vs_centred(log2)
check("decision-relevant disagreement gives centred > 0", m5b["centred_mean"] > 0.1,
      f"{m5b['centred_mean']:.3f}")

print("\nM2 / M3 (candidates agree on the deployed action)")
m2, m3 = M.m2_switch_fractions(log), M.m3_candidate_agreement(log)
check("switch_vs_MAP == 0", m2["switch_vs_MAP"] == 0.0)
check("switch_vs_uniform == 0", m2["switch_vs_uniform"] == 0.0)
check("pairwise agreement == 1", abs(m3["mean_pairwise_agreement"] - 1.0) < TOL)
check("fraction unanimous == 1", abs(m3["fraction_unanimous"] - 1.0) < TOL)

print("\nM1 posterior movement")
m1_static = M.m1_posterior_movement(make_plus_log(posterior_moves=False))
m1_moving = M.m1_posterior_movement(make_plus_log(posterior_moves=True))
check("uniform posterior -> no drift", not m1_static["posterior_moved"])
check("uniform entropy == ln 8", abs(m1_static["entropy_mean"] - np.log(8)) < 1e-12)
check("moving posterior detected", m1_moving["posterior_moved"])
check("moving posterior loses entropy",
      m1_moving["entropy_end_mean"] < m1_static["entropy_mean"] - 0.1)

print("\nM6 reward decomposition")
m6 = M.m6_reward_decomposition(log)
check("identity utility-cost-penalty holds", m6["identity_residual"] < 1e-12,
      f"{m6['identity_residual']:.2e}")
check("penalty dominates a sink return", m6["penalty_share_of_magnitude"] > 0.9,
      f"{m6['penalty_share_of_magnitude']:.3f}")
m6p = M.m6_reward_decomposition(log, accepted_return_mean=m6["return_mean"])
check("parity gate passes on itself", m6p["parity_pass"])
m6f = M.m6_reward_decomposition(log, accepted_return_mean=m6["return_mean"] + 1e-6)
check("parity gate catches a 1e-6 drift", not m6f["parity_pass"])

print("\nM7 family degeneracy")
check("r_pos == 0 everywhere -> frac 1.0",
      M.m7_family_degeneracy(log)["frac_r_pos_zero"] == 1.0)
check("r_pos > 0 -> frac 0.0",
      M.m7_family_degeneracy(make_plus_log(r_pos=0.2))["frac_r_pos_zero"] == 0.0)

print("\nM13 surrogate error")
m13 = M.m13_surrogate_error(log)
check("under-penalised danger detected as positive bias", m13["me_unsafe"] > 1.9,
      f"{m13['me_unsafe']:.3f}")
check("EVD is excluded from surrogate analysis",
      not M.m13_surrogate_error(make_evd_log())["available"])

print("\nM11 EVD")
m11 = M.m11_evd(make_evd_log())
check("dominant action -> penalty changes nothing",
      m11["switch_fraction_penalty_vs_mean"] == 0.0)
check("collapse attributed to conservative Q",
      m11["collapse_attributable_to"] == "conservative_Q")

print("\nM14 / M15")
m14 = M.m14_observation_model_status(log)
check("PLUS flagged as sigma-aware", m14["uses_sigma_obs"])
check("EVD flagged as NOT sigma-aware",
      not M.m14_observation_model_status(make_evd_log())["uses_sigma_obs"])
m15 = M.m15_state_discrimination(log)
check("myopic oracle identified as a2", m15["modal_oracle_action"] == 2)
check("deployed (a5) != oracle (a2)", m15["deployed_equals_myopic_oracle"] == 0.0)

print("\nmissing-field handling")
stripped = make_plus_log()
stripped.vectors.pop("w_t")
check("absent w_t degrades gracefully",
      M.m1_posterior_movement(stripped)["available"] is False)
allm = M.compute_all(make_plus_log(), accepted_return_mean=None)
check("compute_all returns every metric", len(allm) == len(M.ALL_METRICS))
check("compute_all never raises", all(isinstance(v, dict) for v in allm.values()))

print("\n" + ("ALL TESTS PASSED" if not _fails else f"FAILURES: {_fails}"))
raise SystemExit(1 if _fails else 0)
