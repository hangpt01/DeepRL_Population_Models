"""Simulation of management with MC-UAMS policy."""

from __future__ import annotations

import numpy as np

from read_policyx import interp_policy2, update_belief


def sum_2_by_2(vector: np.ndarray) -> np.ndarray:
    n = len(vector) // 2
    out = np.zeros(n, dtype=float)
    for i in range(n):
        out[i] = vector[2 * i] + vector[2 * i + 1]
    return out


def sim_mdp_momdp_policy(
    state_prior: np.ndarray,
    tmax: int,
    tr_mdp: np.ndarray,
    rew_mdp: np.ndarray,
    tr_momdp: np.ndarray,
    obs_momdp: np.ndarray,
    alpha_momdp,
    disc: float = 0.95,
    n_it: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    s = alpha_momdp["vectors"].shape[0]
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_it):
        v = 0.0
        rand = rng.random()
        if rand <= state_prior[0]:
            real_state = [1]
            belief = np.tile(np.array([1.0 / s, 0.0]), s)[None, :]
        else:
            real_state = [2]
            belief = np.tile(np.array([0.0, 1.0 / s]), s)[None, :]

        belief_mod = sum_2_by_2(belief[0, :])
        output = interp_policy2(
            belief_mod,
            obs=real_state[0],
            alpha=alpha_momdp["vectors"],
            alpha_action=alpha_momdp["action"],
            alpha_obs=alpha_momdp["obs"],
        )
        actions = [int(output[1])]
        mod_probs = [belief_mod]
        rewards = [0.0, rew_mdp[real_state[0] - 1, actions[0] - 1]]

        for i in range(1, tmax + 1):
            o1 = real_state[i - 1]
            a1 = actions[i - 1]
            rand = rng.random()
            if rand <= tr_mdp[o1 - 1, 0, a1 - 1]:
                o2 = 1
            else:
                o2 = 2
            real_state.append(o2)

            s_p = update_belief(belief[i - 1, :], tr_momdp, obs_momdp, o2, a1)
            belief = np.vstack([belief, s_p])
            belief_modi = sum_2_by_2(s_p)
            mod_probs.append(belief_modi)

            output = interp_policy2(
                belief_modi,
                obs=o2,
                alpha=alpha_momdp["vectors"],
                alpha_action=alpha_momdp["action"],
                alpha_obs=alpha_momdp["obs"],
            )
            next_action = int(output[1])
            actions.append(next_action)
            rewards.append(rewards[i] + (disc**i) * rew_mdp[o2 - 1, next_action - 1])

        rows.append(np.concatenate([mod_probs[-1], np.asarray(rewards, dtype=float)]))

    return np.vstack(rows)
