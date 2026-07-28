"""Simulation of management using PUBD policy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
DIRICHLET_DIR = CURRENT_DIR.parent / "Dirichlet solver"
if str(DIRICHLET_DIR) not in sys.path:
    sys.path.insert(0, str(DIRICHLET_DIR))

from dirichlet_solver import get_index, init_alphas


def update_state(hyperstate: np.ndarray, a: int, o: int, s: int, a_count: int) -> np.ndarray:
    hyperstate_new = hyperstate.copy()
    physical_state = int(hyperstate_new[s**2 * a_count])
    idx = s**2 * (a - 1) + (physical_state - 1) * s + (o - 1)
    hyperstate_new[idx] += 1
    hyperstate_new[s**2 * a_count] = float(o)
    return hyperstate_new


def sim_mdp_parameter_uncertainty(
    state_prior: np.ndarray,
    tr_mdp: np.ndarray,
    rew_mdp: np.ndarray,
    states,
    state_index_map,
    V_MDP,
    disc: float = 0.95,
    n_it: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    s = rew_mdp.shape[0]
    a_count = rew_mdp.shape[1]
    tmax = V_MDP["policy"].shape[1]
    rng = np.random.default_rng(seed)
    all_values = []

    for _ in range(n_it):
        rand = rng.random()
        if rand <= state_prior[0]:
            physical_state = [1]
        else:
            physical_state = [2]

        alphas = init_alphas(s, a_count)
        state = np.concatenate([alphas, [float(physical_state[0])]])
        index = int(get_index(state, states, key_to_idx=state_index_map)[0])
        actions = [int(V_MDP["policy"][index - 1, 0])]
        values = [0.0, rew_mdp[physical_state[0] - 1, actions[0] - 1]]

        for i in range(1, tmax):
            o1 = physical_state[i - 1]
            a1 = actions[i - 1]
            rand = rng.random()
            if rand <= tr_mdp[o1 - 1, 0, a1 - 1]:
                o2 = 1
            else:
                o2 = 2
            physical_state.append(o2)

            state = update_state(state, a1, o2, s, a_count)
            index = int(get_index(state, states, key_to_idx=state_index_map)[0])
            a2 = int(V_MDP["policy"][index - 1, i])
            actions.append(a2)
            values.append(values[i] + (disc**i) * rew_mdp[o2 - 1, a2 - 1])

        all_values.append(values)

    return np.asarray(all_values, dtype=float)
