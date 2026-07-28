"""Build transition, reward, and observation matrices for hmMDP simulations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from config_loader import load_example_from_env, repo_root_from_here


def _read_mean_parameters(file_example_module: str) -> tuple[dict, pd.DataFrame]:
    cfg = load_example_from_env(__file__, default_module=file_example_module)
    root = repo_root_from_here(__file__)
    data_mean_parameters = pd.read_csv(root / cfg["file_mean_params"])
    return cfg, data_mean_parameters


def transition_hmMDP(file_example_module: str) -> np.ndarray:
    cfg, data_mean_parameters = _read_mean_parameters(file_example_module)
    reward = np.asarray(cfg["reward"], dtype=float)
    num_s, num_a = reward.shape
    mod = data_mean_parameters["opt"].to_numpy(dtype=int)
    num_mod = len(mod)

    tr_by_action = [np.zeros((num_mod * num_s, num_mod * num_s), dtype=float) for _ in range(num_a)]
    p_cols = [c for c in data_mean_parameters.columns if c.startswith("p")]

    for i in range(num_mod):
        mod_id = mod[i]
        params = data_mean_parameters.loc[data_mean_parameters["opt"] == mod_id, p_cols].iloc[0].to_numpy()
        for act_id in range(1, num_a + 1):
            tr = tr_by_action[act_id - 1]
            idx = (act_id - 1) * 2
            mat = np.array(
                [
                    [params[idx], 1 - params[idx]],
                    [params[idx + 1], 1 - params[idx + 1]],
                ],
                dtype=float,
            )
            block = slice(i * num_s, (i + 1) * num_s)
            tr[block, block] = mat

    return np.stack(tr_by_action, axis=2)


def reward_hmMDP(file_example_module: str) -> np.ndarray:
    cfg, data_mean_parameters = _read_mean_parameters(file_example_module)
    reward = np.asarray(cfg["reward"], dtype=float)
    num_s, num_a = reward.shape
    num_mod = len(data_mean_parameters["opt"])

    rew_momdp = np.zeros((num_mod * num_s, num_a), dtype=float)
    for act_id in range(num_a):
        rew_momdp[:, act_id] = np.tile(reward[:, act_id], num_mod)
    return rew_momdp


def obs_hmMDP(file_example_module: str) -> np.ndarray:
    cfg, data_mean_parameters = _read_mean_parameters(file_example_module)
    reward = np.asarray(cfg["reward"], dtype=float)
    num_s, num_a = reward.shape
    num_mod = len(data_mean_parameters["opt"])

    obs_base = np.vstack([np.eye(num_s) for _ in range(num_mod)])
    obs_momdp = np.stack([obs_base for _ in range(num_a)], axis=2)
    return obs_momdp
