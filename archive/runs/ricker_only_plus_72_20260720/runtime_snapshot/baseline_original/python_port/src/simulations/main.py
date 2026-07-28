"""Run simulations for MC-UAMS, PUBD, and optimal benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parent / "utils"
DIRICHLET_DIR = CURRENT_DIR.parent / "Dirichlet solver"
for p in (UTILS_DIR, DIRICHLET_DIR, CURRENT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_matrices_hmMDP import obs_hmMDP, transition_hmMDP
from config_loader import load_example_from_env, repo_root_from_here
from dirichlet_solver import dirichlet_solver_from_config
from generate_random_mdp import generate_random_mdp
from mdp_tools import finite_horizon
from read_policyx import read_policyx2
from sim_mdp_momdp_policy import sim_mdp_momdp_policy
from sim_mdp_parameter_uncertainty import sim_mdp_parameter_uncertainty


def _build_tr_mdp(row: np.ndarray, s: int, a: int) -> np.ndarray:
    mats = []
    for act in range(a):
        vals = row[act * s * s : (act + 1) * s * s]
        mats.append(np.asarray(vals, dtype=float).reshape(s, s))
    return np.stack(mats, axis=2)


def _build_hmmdp_from_meanparams(mean_df: pd.DataFrame, reward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_s, num_a = reward.shape
    mod = mean_df["opt"].to_numpy(dtype=int)
    num_mod = len(mod)
    tr_by_action = [np.zeros((num_mod * num_s, num_mod * num_s), dtype=float) for _ in range(num_a)]
    p_cols = [c for c in mean_df.columns if c.startswith("p")]
    for i in range(num_mod):
        mod_id = mod[i]
        params = mean_df.loc[mean_df["opt"] == mod_id, p_cols].iloc[0].to_numpy()
        for act_id in range(1, num_a + 1):
            tr = tr_by_action[act_id - 1]
            idx = (act_id - 1) * 2
            mat = np.array(
                [[params[idx], 1 - params[idx]], [params[idx + 1], 1 - params[idx + 1]]],
                dtype=float,
            )
            block = slice(i * num_s, (i + 1) * num_s)
            tr[block, block] = mat
    tr_momdp = np.stack(tr_by_action, axis=2)
    obs_base = np.vstack([np.eye(num_s) for _ in range(num_mod)])
    obs_momdp = np.stack([obs_base for _ in range(num_a)], axis=2)
    return tr_momdp, obs_momdp


def main() -> None:
    cfg = load_example_from_env(__file__, default_module="examples2states2actions")
    root = repo_root_from_here(__file__)

    reward = np.asarray(cfg["reward"], dtype=float)
    b_full = np.asarray(cfg["b_full"], dtype=float)
    gamma = float(cfg["gamma"])
    tmax_hmmdp = int(cfg["Tmax_hmMDP"])
    n_mdp = int(cfg["N_MDP"])
    s, a = reward.shape

    file_mdp = generate_random_mdp(reward, n_mdp, str(root / cfg["file_random_mdp"]))
    random_mdp = pd.read_csv(file_mdp).to_numpy()

    alpha_momdp = read_policyx2(str(root / cfg["file_outpolicy"]))
    tr_momdp = transition_hmMDP(cfg["module_name"])
    obs_momdp = obs_hmMDP(cfg["module_name"])

    l = dirichlet_solver_from_config(cfg)
    V_MDP = l["V_MDP"]
    states = l["states"]
    state_index_map = l["state_index_map"]

    tab_hmMDP_res, tab_params_res, v_opt_res = [], [], []
    is_gouldian = cfg["module_name"] == "gouldian"
    if is_gouldian:
        alpha_momdp_4exp = read_policyx2(str(root / cfg["file_outpolicy_4Exp"]))
        mean_4exp = pd.read_csv(root / cfg["file_mean_params_4Exp"])
        tr_momdp_4exp, obs_momdp_4exp = _build_hmmdp_from_meanparams(mean_4exp, reward)
        tab_hmMDP_4exp_res = []

    for mdp_id in range(random_mdp.shape[0]):
        tr_mdp = _build_tr_mdp(random_mdp[mdp_id, :], s, a)
        rew_mdp = reward.copy()

        tab_hmMDP = sim_mdp_momdp_policy(
            b_full, tmax_hmmdp, tr_mdp, rew_mdp, tr_momdp, obs_momdp, alpha_momdp, gamma
        )
        n_mod = alpha_momdp["vectors"].shape[0]
        trajectory_hmMDP = tab_hmMDP[:, n_mod : tmax_hmmdp + n_mod]

        tab_params = sim_mdp_parameter_uncertainty(b_full, tr_mdp, rew_mdp, states, state_index_map, V_MDP, gamma)
        if is_gouldian:
            tab_hmMDP_4exp = sim_mdp_momdp_policy(
                b_full,
                tmax_hmmdp,
                tr_mdp,
                rew_mdp,
                tr_momdp_4exp,
                obs_momdp_4exp,
                alpha_momdp_4exp,
                gamma,
            )
            n_mod_4exp = alpha_momdp_4exp["vectors"].shape[0]
            trajectory_hmMDP_4exp = tab_hmMDP_4exp[:, n_mod_4exp : tmax_hmmdp + n_mod_4exp]
        mdp_sol = finite_horizon(tr_mdp, rew_mdp, gamma, tmax_hmmdp)
        v_opt = mdp_sol.V[int(np.argmax(b_full)), ::-1]

        tab_hmMDP_res.append(trajectory_hmMDP)
        tab_params_res.append(tab_params)
        v_opt_res.append(v_opt)
        if is_gouldian:
            tab_hmMDP_4exp_res.append(trajectory_hmMDP_4exp)

    out_hmmdp = np.vstack(tab_hmMDP_res)
    out_params = np.vstack(tab_params_res)
    out_opt = np.vstack(v_opt_res)

    Path(root / cfg["file_simulations_hmMDP"]).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_hmmdp).to_csv(root / cfg["file_simulations_hmMDP"], index=False)
    pd.DataFrame(out_params).to_csv(root / cfg["file_simulations_params"], index=False)
    pd.DataFrame(out_opt).to_csv(root / cfg["file_simulations_opt"], index=False)
    if is_gouldian:
        pd.DataFrame(np.vstack(tab_hmMDP_4exp_res)).to_csv(root / cfg["file_simulations_hmMDP_4Exp"], index=False)


if __name__ == "__main__":
    main()
