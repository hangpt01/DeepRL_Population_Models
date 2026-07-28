"""Build the hmMDP POMDPX file using MC-UAMS."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config_loader import load_example_from_env, repo_root_from_here
from generate_pomdpx import write_hmMDPx
from mean_parameters import mean_parameters


def main() -> None:
    cfg = load_example_from_env(__file__, default_module="gouldian")
    root = repo_root_from_here(__file__)

    reward = np.asarray(cfg["reward"], dtype=float)
    gamma = float(cfg["gamma"])
    b_full = np.asarray(cfg["b_full"], dtype=float)

    seed_env = os.environ.get("UAMS_SEED", "").strip()
    seed = int(seed_env) if seed_env else None
    mean_df = mean_parameters(reward, gamma, n_trials=int(1e4), seed=seed)
    mean_path = root / cfg["file_mean_params"]
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    mean_df.to_csv(mean_path, index=False)

    data_mean_parameters = pd.read_csv(mean_path)
    num_s, num_a = reward.shape
    mod = data_mean_parameters["opt"].to_numpy(dtype=int)
    num_mod = len(mod)
    b_par = np.repeat(1.0 / num_mod, num_mod)

    transition = []
    p_cols = [c for c in data_mean_parameters.columns if c.startswith("p")]
    for mod_id in mod:
        params = data_mean_parameters.loc[data_mean_parameters["opt"] == mod_id, p_cols].iloc[0].to_numpy()
        model = np.zeros((num_s, num_s, num_a), dtype=float)
        for act_id in range(1, num_a + 1):
            i = (act_id - 1) * 2
            mat = np.array(
                [
                    [params[i], 1 - params[i]],
                    [params[i + 1], 1 - params[i + 1]],
                ],
                dtype=float,
            )
            model[:, :, act_id - 1] = mat
        transition.append(model)

    write_hmMDPx(
        transition,
        reward,
        b_full,
        b_par,
        gamma,
        str(root / cfg["file_pomdpx"]),
    )


if __name__ == "__main__":
    main()
