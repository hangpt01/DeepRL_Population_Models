"""PUBD/Dirichlet solver translated from R."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from config_loader import load_example_from_env
from mdp_tools import finite_horizon


def init_alphas(s: int, a: int) -> np.ndarray:
    return np.ones(s**2 * a, dtype=float)


def next_states_list(hyperstate: np.ndarray, action: int, s: int, a: int) -> Tuple[np.ndarray, np.ndarray]:
    next_states = []
    probabilities = []
    alphas = hyperstate[: s**2 * a]
    physical_state = int(hyperstate[s**2 * a])

    start = (action - 1) * s**2 + (physical_state - 1) * s
    block = alphas[start : start + s]
    denom = float(np.sum(block))
    for s_next in range(1, s + 1):
        proba = block[s_next - 1] / denom
        probabilities.append(proba)

        next_alphas = alphas.copy()
        idx = (action - 1) * s**2 + (physical_state - 1) * s + (s_next - 1)
        next_alphas[idx] += 1
        next_states.append(np.concatenate([next_alphas, [float(s_next)]]))

    return np.vstack(next_states), np.asarray(probabilities, dtype=float)


def _state_key(v: np.ndarray) -> Tuple[int, ...]:
    # States are integer-valued in this model.
    return tuple(int(round(x)) for x in v.tolist())


def build_state_index_map(all_states: List[np.ndarray]) -> Dict[Tuple[int, ...], int]:
    """Build a reusable mapping from hyperstate key to 1-based index."""
    index_matrix = np.vstack(all_states)
    return {_state_key(row): i + 1 for i, row in enumerate(index_matrix)}


def get_index(
    s: np.ndarray, all_states: List[np.ndarray], key_to_idx: Dict[Tuple[int, ...], int] | None = None
) -> np.ndarray:
    if key_to_idx is None:
        key_to_idx = build_state_index_map(all_states)

    if s.ndim == 1:
        return np.array([key_to_idx[_state_key(s)]], dtype=int)
    return np.asarray([key_to_idx[_state_key(row)] for row in s], dtype=int)


def indexes_probabilities(alphas: np.ndarray, physical_state: int, s: int, a: int, tmax: int):
    s0 = np.concatenate([alphas, [float(physical_state)]])[None, :]
    states: List[np.ndarray] = [s0]

    data_action = [pd.DataFrame({"index": [], "next_index": [], "probability": []}) for _ in range(a)]
    for time in range(2, tmax + 1):
        all_next_states = np.empty((0, s**2 * a + 1), dtype=float)
        mat_old = states[time - 2]
        n_old = mat_old.shape[0]

        for row_id in range(n_old):
            for action in range(1, a + 1):
                nxt, probs = next_states_list(mat_old[row_id, :], action, s, a)
                all_next_states = np.vstack([all_next_states, nxt])
                all_next_states = np.unique(all_next_states, axis=0)
                if len(states) < time:
                    states.append(all_next_states)
                else:
                    states[time - 1] = all_next_states

                index = get_index(mat_old[row_id, :], states)[0]
                next_index = get_index(nxt, states)
                new_data = pd.DataFrame(
                    {
                        "index": np.repeat(index, len(next_index)),
                        "next_index": next_index,
                        "probability": probs,
                    }
                )
                data_action[action - 1] = pd.concat([data_action[action - 1], new_data], ignore_index=True)

    return {"data_action": data_action, "states": states}


def transition_matrix(data_action: List[pd.DataFrame], states: List[np.ndarray], a: int) -> List[np.ndarray]:
    transition_probabilities = []
    max_index = np.vstack(states).shape[0]
    for action in range(a):
        tr = np.zeros((max_index, max_index), dtype=float)
        df = data_action[action]
        for _, row in df.iterrows():
            i = int(row["index"]) - 1
            j = int(row["next_index"]) - 1
            tr[i, j] += float(row["probability"])

        last_states = states[-1]
        ilast = get_index(last_states, states) - 1
        tr[ilast, ilast] = 1.0
        transition_probabilities.append(tr)
    return transition_probabilities


def reward_matrix(states: List[np.ndarray], a: int, reward: np.ndarray) -> np.ndarray:
    index_matrix = np.vstack(states)
    physical_states = index_matrix[:, -1].astype(int)
    n_states = len(physical_states)
    rew = np.zeros((n_states, a), dtype=float)
    for action in range(1, a + 1):
        rew[:, action - 1] = reward[physical_states - 1, action - 1]
    return rew


def dirichlet_solver_from_config(cfg: Dict) -> Dict:
    reward = np.asarray(cfg["reward"], dtype=float)
    gamma = float(cfg["gamma"])
    tmax = int(cfg["Tmax"])
    b_full = np.asarray(cfg["b_full"], dtype=float)
    a = reward.shape[1]
    s = reward.shape[0]

    physical_state = int(np.argmax(b_full) + 1)
    alphas = init_alphas(s, a)
    data = indexes_probabilities(alphas, physical_state, s, a, tmax)
    data_action = data["data_action"]
    states = data["states"]

    tr_mdp = transition_matrix(data_action, states, a)
    rew_mdp = reward_matrix(states, a, reward)
    res = finite_horizon(tr_mdp, rew_mdp, gamma, tmax)
    state_index_map = build_state_index_map(states)
    return {"V_MDP": {"V": res.V, "policy": res.policy}, "states": states, "state_index_map": state_index_map}


def dirichlet_solver(example_module: str = "examples2states2actions") -> Dict:
    cfg = load_example_from_env(__file__, default_module=example_module)
    return dirichlet_solver_from_config(cfg)
