"""Metrics M1-M15 from DIAGNOSTIC_REPLAY_RERUN_SPEC.md section 1.6.

Every function takes a ReplayLog and returns a plain dict.  Functions return
{"available": False, "reason": ...} rather than raising when a field the
replay did not log is required, so a partial run still yields a full report.
"""
from __future__ import annotations

import numpy as np

from constants import (COLLAPSE_PENALTY, DISCOUNT_SUM, GAMMA, N_ACTIONS,
                       N_PLUS_CANDIDATES, OGSRL_S_LOW, SIGMA_AWARE_METHODS,
                       SPECIES, SURROGATE_METHODS, WEAK_REGIME_INDEX)

_NA = lambda why: {"available": False, "reason": why}


def _col(log, name):
    return log.frame[name].to_numpy(dtype=float)


# ---------------------------------------------------------------- M1
def m1_posterior_movement(log):
    """PLUS candidate-posterior movement over the episode."""
    if "w_t" not in log.vectors:
        return _NA("w_t not logged (spec 11.A item 1)")
    w = log.vectors["w_t"]
    J = w.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(w > 0, w * np.log(w), 0.0), axis=1)
    uniform = 1.0 / J
    tv = 0.5 * np.abs(w - uniform).sum(axis=1)
    first_concentrate, drift = [], []
    for _, mask in log.episode_slices():
        we, tve = w[mask], tv[mask]
        mx = we.max(axis=1)
        hit = np.flatnonzero(mx > 0.5)
        first_concentrate.append(int(hit[0]) if hit.size else -1)
        drift.append(float(np.abs(we[-1] - we[0]).sum()))
    return dict(
        available=True,
        entropy_mean=float(H.mean()), entropy_max=float(np.log(J)),
        entropy_start=float(H[0]), entropy_end_mean=float(
            np.mean([H[m][-1] for _, m in log.episode_slices()])),
        effective_candidates_mean=float(np.exp(H).mean()),
        tv_from_uniform_mean=float(tv.mean()),
        l1_drift_mean=float(np.mean(drift)),
        first_step_max_w_gt_half=float(np.mean(
            [f for f in first_concentrate if f >= 0])) if any(
                f >= 0 for f in first_concentrate) else None,
        episodes_that_concentrated=int(sum(f >= 0 for f in first_concentrate)),
        n_episodes=len(drift),
        posterior_moved=bool(np.mean(drift) > 1e-9),
    )


# ---------------------------------------------------------------- M2
def m2_switch_fractions(log):
    """Does posterior averaging / updating ever change the action?"""
    need = ("argmax_weighted", "argmax_map_only", "argmax_uniform")
    if not all(n in log.frame.columns for n in need):
        return _NA("argmax counterfactuals not logged")
    aw = _col(log, "argmax_weighted")
    return dict(
        available=True,
        switch_vs_MAP=float(np.mean(aw != _col(log, "argmax_map_only"))),
        switch_vs_uniform=float(np.mean(aw != _col(log, "argmax_uniform"))),
        n_decisions=int(len(aw)),
    )


# ---------------------------------------------------------------- M3
def m3_candidate_agreement(log):
    """Pairwise agreement of the candidate greedy policies along the path."""
    if "argmax_cand" not in log.vectors:
        return _NA("argmax_cand not logged")
    a = log.vectors["argmax_cand"]
    n, J = a.shape
    agree = np.zeros(n)
    pairs = 0
    for i in range(J):
        for j in range(i + 1, J):
            agree += (a[:, i] == a[:, j])
            pairs += 1
    agree /= pairs
    unanimous = np.array([len(np.unique(row)) == 1 for row in a])
    modal_share = np.array(
        [np.bincount(row.astype(int)).max() / J for row in a])
    return dict(
        available=True, n_candidates=int(J),
        mean_pairwise_agreement=float(agree.mean()),
        fraction_unanimous=float(unanimous.mean()),
        mean_modal_share=float(modal_share.mean()),
        min_modal_share=float(modal_share.min()),
    )


# ---------------------------------------------------------------- M4
def m4_margins(log, eps=(1e-3, 1e-2)):
    col = ("margin_top1_top2" if "margin_top1_top2" in log.frame.columns
           else None)
    if col is None:
        return _NA("margin_top1_top2 not logged")
    m = _col(log, col)
    out = dict(available=True, min=float(m.min()),
               p5=float(np.percentile(m, 5)), median=float(np.median(m)),
               mean=float(m.mean()), max=float(m.max()))
    for e in eps:
        out[f"frac_below_{e:g}"] = float(np.mean(m < e))
    return out


# ---------------------------------------------------------------- M5
def m5_raw_vs_centred(log):
    """Cross-candidate value disagreement, before and after removing the
    action-independent component.  Large raw with small centred means the
    candidates disagree about the world but not about what to do."""
    if "q_cand" not in log.vectors:
        return _NA("q_cand not logged")
    q = log.vectors["q_cand"]                       # (n, J, A)
    raw = np.nanmean(np.nanvar(q, axis=1), axis=1)  # var over candidates
    qc = q - np.nanmean(q, axis=2, keepdims=True)   # centre each candidate
    centred = np.nanmean(np.nanvar(qc, axis=1), axis=1)
    rm, cm = float(np.nanmean(raw)), float(np.nanmean(centred))
    return dict(available=True, raw_mean=rm, centred_mean=cm,
                ratio_centred_over_raw=(cm / rm if rm > 0 else float("nan")),
                action_independent_fraction=(1.0 - cm / rm) if rm > 0 else
                float("nan"))


# ---------------------------------------------------------------- M6
def m6_reward_decomposition(log, accepted_return_mean=None, tol=1e-9):
    need = ("utility_t", "cost_t", "penalty_flag_t", "reward_t", "t")
    if not all(n in log.frame.columns for n in need):
        return _NA("reward component columns not logged")
    t = _col(log, "t")
    disc = GAMMA ** t
    util = _col(log, "utility_t") * disc
    cost = _col(log, "cost_t") * disc
    pen = COLLAPSE_PENALTY * _col(log, "penalty_flag_t") * disc
    rew = _col(log, "reward_t") * disc
    per_ep = []
    for _, mask in log.episode_slices():
        per_ep.append((util[mask].sum(), cost[mask].sum(),
                       pen[mask].sum(), rew[mask].sum()))
    per_ep = np.array(per_ep)
    ret = float(per_ep[:, 3].mean())
    out = dict(
        available=True,
        utility_disc_mean=float(per_ep[:, 0].mean()),
        cost_disc_mean=float(per_ep[:, 1].mean()),
        penalty_disc_mean=float(per_ep[:, 2].mean()),
        return_mean=ret, return_sd=float(per_ep[:, 3].std(ddof=1))
        if len(per_ep) > 1 else 0.0,
        identity_residual=float(np.max(np.abs(
            per_ep[:, 3] - (per_ep[:, 0] - per_ep[:, 1] - per_ep[:, 2])))),
        penalty_share_of_magnitude=float(
            per_ep[:, 2].mean() / max(abs(ret), 1e-12)),
    )
    if accepted_return_mean is not None:
        out["accepted_return_mean"] = float(accepted_return_mean)
        out["parity_delta"] = abs(ret - accepted_return_mean)
        out["parity_pass"] = bool(out["parity_delta"] <= tol)
    return out


# ---------------------------------------------------------------- M7
def m7_family_degeneracy(log):
    """Fraction of steps with r_pos == 0: the family-collapse condition."""
    if "r_pos_t" not in log.frame.columns:
        return _NA("r_pos_t not logged")
    r = _col(log, "r_pos_t")
    return dict(available=True, frac_r_pos_zero=float(np.mean(r == 0.0)),
                r_pos_max=float(r.max()),
                families_collapse_everywhere=bool(np.all(r == 0.0)))


# ---------------------------------------------------------------- M8
def m8_depensation(log):
    if not {"m_t", "allee_C_episode"} <= set(log.frame.columns):
        return _NA("m_t / allee_C_episode not logged (Allee cells only)")
    m, C = _col(log, "m_t"), _col(log, "allee_C_episode")
    valid = ~np.isnan(C)
    if not valid.any():
        return _NA("no Allee threshold in this cell")
    return dict(available=True,
                frac_below_threshold=float(np.mean(m[valid] < C[valid])),
                min_ratio_m_over_C=float(np.nanmin(m[valid] / C[valid])))


# ---------------------------------------------------------------- M9
def m9_regime(log, weak_index=WEAK_REGIME_INDEX):
    if "regime_z_t" not in log.frame.columns:
        return _NA("regime_z_t not logged (regime cells only)")
    z = _col(log, "regime_z_t")
    sw = (_col(log, "regime_switched_t") if "regime_switched_t"
          in log.frame.columns else np.abs(np.diff(z, prepend=z[0])) > 0)
    per_ep = [float(np.nansum(sw[m])) for _, m in log.episode_slices()]
    out = dict(available=True, switches_per_episode_mean=float(np.mean(per_ep)),
               switches_per_episode_sd=float(np.std(per_ep, ddof=1))
               if len(per_ep) > 1 else 0.0,
               frac_steps_weak_regime=float(np.mean(z == weak_index)))
    if "action_t" in log.frame.columns:
        a = _col(log, "action_t")
        changed = np.abs(np.diff(a, prepend=a[0])) > 0
        idx = np.flatnonzero(sw > 0)
        near = [changed[min(i + 1, len(changed) - 1)] for i in idx]
        out["frac_switches_followed_by_action_change"] = (
            float(np.mean(near)) if near else float("nan"))
    return out


# ---------------------------------------------------------------- M10
def m10_noise_sensitivity(log_low, log_high):
    """Per-step action agreement between matched seeds at two noise levels."""
    for lg in (log_low, log_high):
        if "action_t" not in lg.frame.columns:
            return _NA("action_t not logged")
    a = log_low.frame[["seed", "t", "action_t"]]
    b = log_high.frame[["seed", "t", "action_t"]]
    merged = a.merge(b, on=["seed", "t"], suffixes=("_lo", "_hi"))
    if merged.empty:
        return _NA("no matched (seed, t) pairs")
    agree = (merged["action_t_lo"] == merged["action_t_hi"]).to_numpy()
    return dict(available=True, matched_steps=int(len(merged)),
                action_agreement=float(agree.mean()))


# ---------------------------------------------------------------- M11
def m11_evd(log, lam=0.1, recovery_actions=(3, 4, 7, 8, 9, 10)):
    if not {"qbar", "qvar"} <= set(log.vectors):
        return _NA("qbar / qvar not logged")
    qbar, qvar = log.vectors["qbar"], log.vectors["qvar"]
    pess = np.nanargmax(qbar - lam * qvar, axis=1)
    plain = np.nanargmax(qbar, axis=1)
    rank = np.argsort(np.argsort(-qbar, axis=1), axis=1)  # 0 = best
    return dict(
        available=True,
        switch_fraction_penalty_vs_mean=float(np.mean(pess != plain)),
        mean_rank_of_recovery_actions=float(
            np.nanmean(rank[:, list(recovery_actions)])),
        best_action_is_recovery_frac=float(
            np.mean(np.isin(plain, recovery_actions))),
        collapse_attributable_to="conservative_Q"
        if float(np.mean(pess != plain)) < 0.01 else "pessimism_penalty",
    )


# ---------------------------------------------------------------- M12
def m12_ogsrl(log, species=None, sigma_index=0, tol=1e-6):
    cols = set(log.frame.columns)
    if not {"ogsrl_c25", "ogsrl_budget"} <= cols:
        return _NA("ogsrl_c25 / ogsrl_budget not logged")
    c25, budget = _col(log, "ogsrl_c25"), _col(log, "ogsrl_budget")
    slack = budget - c25
    out = dict(available=True, frac_constraint_binds=float(
        np.mean(slack <= tol)), mean_slack=float(slack.mean()),
        min_slack=float(slack.min()))
    if species in OGSRL_S_LOW:
        out["s_low_public"] = OGSRL_S_LOW[species][sigma_index]
        out["s_safe_true"] = SPECIES[species]["s_safe"]
        out["proxy_ratio"] = out["s_low_public"] / out["s_safe_true"]
    if "ogsrl_fallback" in cols:
        out["fallback_fraction"] = float(np.mean(_col(log, "ogsrl_fallback")))
    return out


# ---------------------------------------------------------------- M13
def m13_surrogate_error(log):
    """Planner-believed reward vs evaluator-scored reward, along the path."""
    if log.method in ("evd",):
        return _NA("EVD trains on raw dataset.rewards; no surrogate (audit 2 P0-2)")
    if not {"surrogate_reward_t", "reward_t"} <= set(log.frame.columns):
        return _NA("surrogate_reward_t not logged")
    err = _col(log, "surrogate_reward_t") - _col(log, "reward_t")
    out = dict(available=True, me_all=float(err.mean()),
               rmse_all=float(np.sqrt((err ** 2).mean())))
    if "penalty_flag_t" in log.frame.columns:
        unsafe = _col(log, "penalty_flag_t") > 0
        for name, mask in (("unsafe", unsafe), ("safe", ~unsafe)):
            out[f"n_{name}"] = int(mask.sum())
            out[f"me_{name}"] = (float(err[mask].mean()) if mask.any()
                                 else float("nan"))
            out[f"rmse_{name}"] = (float(np.sqrt((err[mask] ** 2).mean()))
                                   if mask.any() else float("nan"))
        out["danger_stratum_ever_visited"] = bool(unsafe.any())
    return out


# ---------------------------------------------------------------- M14
def m14_observation_model_status(log):
    """Reporting guard: three methods never model observation noise."""
    uses_sigma = log.method in SIGMA_AWARE_METHODS
    has_belief = any(c.startswith("belief_") for c in log.frame.columns) or \
        any(k.startswith("belief_") for k in log.vectors)
    return dict(available=True, method=log.method, uses_sigma_obs=uses_sigma,
                belief_fields_present=has_belief,
                uses_public_reward_surrogate=log.method in SURROGATE_METHODS,
                comparable_with_filtering_methods=uses_sigma,
                note=("" if uses_sigma else
                      "sigma_obs NOT USED by this method (audit 2 P0-6); do not "
                      "present its diagnostics alongside PLUS/MOOR/RefPlan "
                      "filtering results"))


# ---------------------------------------------------------------- M15
def m15_state_discrimination(log):
    """From the deterministic 11-action shadow rollout."""
    if "true_reward_all_actions" not in log.vectors:
        return _NA("true_reward_all_actions not logged")
    R = log.vectors["true_reward_all_actions"]
    srt = np.sort(R, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    oracle = np.nanargmax(R, axis=1)
    out = dict(available=True, true_margin_mean=float(margin.mean()),
               true_margin_median=float(np.median(margin)),
               true_margin_min=float(margin.min()))
    if "action_t" in log.frame.columns:
        a = _col(log, "action_t").astype(int)
        out["deployed_equals_myopic_oracle"] = float(np.mean(a == oracle))
        out["modal_oracle_action"] = int(np.bincount(oracle).argmax())
        out["modal_deployed_action"] = int(np.bincount(a).argmax())
    return out


ALL_METRICS = {
    "M1_posterior_movement": m1_posterior_movement,
    "M2_switch_fractions": m2_switch_fractions,
    "M3_candidate_agreement": m3_candidate_agreement,
    "M4_margins": m4_margins,
    "M5_raw_vs_centred": m5_raw_vs_centred,
    "M6_reward_decomposition": m6_reward_decomposition,
    "M7_family_degeneracy": m7_family_degeneracy,
    "M8_depensation": m8_depensation,
    "M9_regime": m9_regime,
    "M11_evd": m11_evd,
    "M13_surrogate_error": m13_surrogate_error,
    "M14_observation_model_status": m14_observation_model_status,
    "M15_state_discrimination": m15_state_discrimination,
}


def compute_all(log, accepted_return_mean=None, species=None):
    out = {}
    for name, fn in ALL_METRICS.items():
        try:
            if name == "M6_reward_decomposition":
                out[name] = fn(log, accepted_return_mean=accepted_return_mean)
            elif name == "M11_evd" and log.method != "evd":
                out[name] = _NA("EVD only")
            elif name in ("M1_posterior_movement", "M2_switch_fractions",
                          "M3_candidate_agreement", "M5_raw_vs_centred") \
                    and log.method != "plus":
                out[name] = _NA("PLUS only")
            else:
                out[name] = fn(log)
        except Exception as exc:                      # never lose the report
            out[name] = {"available": False, "reason": f"error: {exc!r}"}
    if log.method == "ogsrl":
        out["M12_ogsrl"] = m12_ogsrl(log, species=species)
    return out
