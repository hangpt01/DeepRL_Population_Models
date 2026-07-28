"""Analyse simulation outputs and compute performance tables."""

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


def _ci95(x: np.ndarray) -> float:
    return 1.96 * np.std(x, ddof=1) / max(np.sqrt(len(x)), 1.0)


def main() -> None:
    cfg = load_example_from_env(__file__, default_module="potoroo")
    root = repo_root_from_here(__file__)
    n_mdp = int(cfg["N_MDP"])
    tmax = int(cfg["Tmax"])
    tmax_hmmdp = int(cfg["Tmax_hmMDP"])

    tab_hmMDP_res = pd.read_csv(root / cfg["file_simulations_hmMDP"])
    tab_params_res = pd.read_csv(root / cfg["file_simulations_params"])
    tab_opt_res = pd.read_csv(root / cfg["file_simulations_opt"])
    mdp_ids = np.repeat(np.arange(1, n_mdp + 1), 100)
    tab_hmMDP_res["mdp_id"] = mdp_ids
    tab_params_res["mdp_id"] = mdp_ids
    is_gouldian = cfg["module_name"] == "gouldian"
    tab_hmMDP_4exp_res_mean = None
    if is_gouldian and "file_simulations_hmMDP_4Exp" in cfg:
        tab_hmMDP_4exp_res = pd.read_csv(root / cfg["file_simulations_hmMDP_4Exp"])
        tab_hmMDP_4exp_res["mdp_id"] = mdp_ids
        tab_hmMDP_4exp_res_mean = tab_hmMDP_4exp_res.groupby("mdp_id").mean(numeric_only=True).reset_index()

    tab_hmMDP_res_mean = tab_hmMDP_res.groupby("mdp_id").mean(numeric_only=True).reset_index()
    tab_params_res_mean = tab_params_res.groupby("mdp_id").mean(numeric_only=True).reset_index()

    col_tmax = f"V{tmax}"
    sim_tmax = pd.DataFrame(
        {
            "V_opt": tab_opt_res[col_tmax],
            "V_hmMDP": tab_hmMDP_res_mean[col_tmax],
            "V_params": tab_params_res_mean[col_tmax],
        }
    )

    col_inf = f"V{tmax_hmmdp}"
    sim_inf = pd.DataFrame({"V_opt": tab_opt_res[col_inf], "V_hmMDP": tab_hmMDP_res_mean[col_inf]})

    row_hmMDP = {
        "perf_mean_Tmax": np.mean((sim_tmax["V_opt"] - sim_tmax["V_hmMDP"]) / sim_tmax["V_opt"]),
        "perf_sd_Tmax": _ci95(((sim_tmax["V_opt"] - sim_tmax["V_hmMDP"]) / sim_tmax["V_opt"]).to_numpy()),
        "perf_mean_inf": np.mean((sim_inf["V_opt"] - sim_inf["V_hmMDP"]) / sim_inf["V_opt"]),
        "perf_sd_inf": _ci95(((sim_inf["V_opt"] - sim_inf["V_hmMDP"]) / sim_inf["V_opt"]).to_numpy()),
    }
    row_params = {
        "perf_mean_Tmax": np.mean((sim_tmax["V_opt"] - sim_tmax["V_params"]) / sim_tmax["V_opt"]),
        "perf_sd_Tmax": _ci95(((sim_tmax["V_opt"] - sim_tmax["V_params"]) / sim_tmax["V_opt"]).to_numpy()),
        "perf_mean_inf": np.nan,
        "perf_sd_inf": np.nan,
    }

    rows = [row_hmMDP]
    idx = ["MC-UAMS"]
    if tab_hmMDP_4exp_res_mean is not None and col_tmax in tab_hmMDP_4exp_res_mean.columns:
        row_4exp = {
            "perf_mean_Tmax": np.mean((sim_tmax["V_opt"] - tab_hmMDP_4exp_res_mean[col_tmax]) / sim_tmax["V_opt"]),
            "perf_sd_Tmax": _ci95(((sim_tmax["V_opt"] - tab_hmMDP_4exp_res_mean[col_tmax]) / sim_tmax["V_opt"]).to_numpy()),
            "perf_mean_inf": np.mean((sim_inf["V_opt"] - tab_hmMDP_4exp_res_mean[col_inf]) / sim_inf["V_opt"]),
            "perf_sd_inf": _ci95(((sim_inf["V_opt"] - tab_hmMDP_4exp_res_mean[col_inf]) / sim_inf["V_opt"]).to_numpy()),
        }
        rows.append(row_4exp)
        idx.append("4 Exp")
    rows.append(row_params)
    idx.append("PUBD")
    performance = pd.DataFrame(rows, index=idx)
    out = root / cfg["performance_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    performance.to_csv(out)


if __name__ == "__main__":
    main()
