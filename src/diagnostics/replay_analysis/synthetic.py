"""Synthetic replay logs with analytically known metric values.

Used to unit-test metrics.py before real logs exist.  Each generator documents
the exact value each metric must return, so a test failure means the metric is
wrong rather than the data being surprising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from constants import COLLAPSE_PENALTY, COST_STEP, GAMMA, SPECIES
from loader import ReplayLog


def make_plus_log(n_episodes=3, n_steps=50, n_cand=8, n_act=11,
                  species="egyptian_vulture", constant_action=5,
                  r_pos=0.0, posterior_moves=False,
                  action_independent_disagreement=True, seed=0):
    """A PLUS log with controlled structure.

    Guarantees when action_independent_disagreement=True:
      M5 centred_mean == 0 exactly (candidates differ by a per-candidate,
         action-independent offset), so ratio_centred_over_raw == 0.
      M3 mean_pairwise_agreement == 1.0 and fraction_unanimous == 1.0.
      M2 both switch fractions == 0.
      M7 frac_r_pos_zero == 1.0 when r_pos == 0.
    """
    rng = np.random.default_rng(seed)
    sp = SPECIES[species]
    n = n_episodes * n_steps
    seeds = np.repeat(np.arange(7001, 7001 + n_episodes), n_steps)
    t = np.tile(np.arange(n_steps), n_episodes)

    x = np.empty(n)
    for e in range(n_episodes):
        xv = sp["N0"]
        for s in range(n_steps):
            x[e * n_steps + s] = xv
            xv = max(xv * 0.95, 0.1)
    x_next = np.concatenate([x[1:], [x[-1] * 0.95]])

    cost = np.full(n, COST_STEP[constant_action])
    utility = x_next / (x_next + sp["K_ref"])
    penalty_flag = (x_next <= sp["s_safe"]).astype(float)
    reward = utility - cost - COLLAPSE_PENALTY * penalty_flag

    base = rng.normal(0, 1.0, size=(n, n_act))
    if action_independent_disagreement:
        offs = rng.normal(0, 5.0, size=(n, n_cand, 1))
        q_cand = base[:, None, :] + offs           # centred variance == 0
    else:
        q_cand = base[:, None, :] + rng.normal(0, 5.0, size=(n, n_cand, n_act))
    q_cand[:, :, constant_action] += 50.0          # every candidate agrees

    if posterior_moves:
        w = np.zeros((n, n_cand))
        for e in range(n_episodes):
            for s in range(n_steps):
                sharp = np.exp(np.linspace(0, 4.0 * s / n_steps, n_cand))
                w[e * n_steps + s] = sharp / sharp.sum()
    else:
        w = np.full((n, n_cand), 1.0 / n_cand)

    q_weighted = np.einsum("nj,nja->na", w, q_cand)
    argmax_w = q_weighted.argmax(axis=1)
    srt = np.sort(q_weighted, axis=1)
    margin = srt[:, -1] - srt[:, -2]

    true_R = rng.normal(0, 0.01, size=(n, n_act))
    true_R[:, 2] += 0.05                           # myopic oracle prefers a2

    frame = pd.DataFrame(dict(
        seed=seeds, t=t, x_true_t=x, x_true_next=x_next,
        obs_t=x * rng.lognormal(0, 0.1, n),
        action_t=np.full(n, constant_action, dtype=float),
        cost_t=cost, utility_t=utility, penalty_flag_t=penalty_flag,
        reward_t=reward,
        r_setpoint_t=np.full(n, sp["r_ricker"][constant_action]),
        r_pos_t=np.full(n, r_pos),
        r_mort_t=np.full(n, -min(sp["r_ricker"][constant_action], 0.0)),
        k_t=np.full(n, sp["K_ref"]), m_t=x,
        regime_z_t=np.zeros(n), regime_switched_t=np.zeros(n),
        allee_C_episode=np.full(n, np.nan),
        theta_exponent_episode=np.full(n, np.nan),
        surrogate_reward_t=reward + 2.0 * penalty_flag,   # under-penalises danger
        argmax_weighted=argmax_w.astype(float),
        argmax_map_only=q_cand[:, 0, :].argmax(axis=1).astype(float),
        argmax_uniform=q_cand.mean(axis=1).argmax(axis=1).astype(float),
        margin_top1_top2=margin,
    ))
    vectors = dict(w_t=w, q_cand=q_cand, q_weighted=q_weighted,
                   argmax_cand=q_cand.argmax(axis=2).astype(float),
                   true_reward_all_actions=true_R)
    return ReplayLog(f"{species}__ricker__0.1", "plus", frame, vectors,
                     [], "synthetic")


def make_evd_log(n=200, n_act=11, lam=0.1, seed=1):
    """EVD log where the pessimism penalty changes no action (switch == 0)."""
    rng = np.random.default_rng(seed)
    qbar = rng.normal(0, 1.0, size=(n, n_act))
    qbar[:, 0] += 10.0                       # a0 dominates by a wide margin
    qvar = rng.uniform(0, 0.5, size=(n, n_act))
    frame = pd.DataFrame(dict(seed=np.full(n, 7001), t=np.arange(n),
                              action_t=np.zeros(n)))
    return ReplayLog("amur_tiger__ricker__0.1", "evd", frame,
                     dict(qbar=qbar, qvar=qvar), [], "synthetic")
