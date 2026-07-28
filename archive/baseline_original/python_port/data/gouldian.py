import numpy as np

CONFIG = {
    "reward": np.array(
        [
            [0.0, -5.0, -5.0, -5.0],
            [20.0, 15.0, 15.0, 15.0],
        ],
        dtype=float,
    ),
    "gamma": 0.9,
    "b_full": np.array([1.0, 0.0], dtype=float),
    "Tmax": 5,
    "Tmax_hmMDP": 50,
    "N_MDP": 100,
    "file_mean_params": "res/meanparams/meanparamsgouldian.csv",
    "file_random_mdp": "res/randomMDP/gouldian_randomMDP.csv",
    "file_pomdpx": "res/POMDPX/gouldian.pomdpx",
    "file_outpolicy": "data/POLICYX/gouldian.policyx",
    "file_simulations_opt": "res/simopt/simoptgouldian.csv",
    "file_simulations_hmMDP": "res/simhmMDP/simhmMDPgouldian.csv",
    "file_simulations_params": "res/simparams/simparamsgouldian.csv",
    "performance_file": "res/performance/performancegouldian.csv",
    "file_mean_params_4Exp": "res/meanparams/meanparamsgouldian4Exp.csv",
    "file_pomdpx_4Exp": "data/gouldian4Exp.pomdpx",
    "file_outpolicy_4Exp": "data/POLICYX/gouldian4Exp.policyx",
    "file_simulations_hmMDP_4Exp": "res/simhmMDP/simhmMDP4Expgouldian.csv",
    "file_simulations_opt_expert_model": "res/expert_model/simoptgouldian_expert_model.csv",
    "file_simulations_hmMDP_expert_model": "res/expert_model/simhmMDPgouldian_expert_model.csv",
    "file_simulations_hmMDP_4Exp_expert_model": "res/expert_model/simhmMDP4Expgouldian_expert_model.csv",
    "file_simulations_params_expert_model": "res/expert_model/simparamsgouldian_expert_model.csv",
    "performance_file_expert_model": "res/performance/performancegouldian_expert_model.csv",
}
