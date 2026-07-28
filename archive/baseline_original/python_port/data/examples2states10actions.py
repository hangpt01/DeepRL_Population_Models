import numpy as np

N = 10
CONFIG = {
    "reward": np.array(
        [
            np.linspace(0, 3, N),
            np.linspace(4, 9, N),
        ],
        dtype=float,
    ),
    "gamma": 0.9,
    "b_full": np.array([1.0, 0.0], dtype=float),
    "Tmax": 3,
    "Tmax_hmMDP": 50,
    "N_MDP": 100,
    "file_mean_params": "res/meanparams/meanparams2states10actions.csv",
    "file_random_mdp": "res/randomMDP/2states10actions_randomMDP.csv",
    "file_pomdpx": "res/POMDPX/2states10actions.pomdpx",
    "file_outpolicy": "data/POLICYX/2states10actions.policyx",
    "file_simulations_opt": "res/simopt/simopt2states10actions.csv",
    "file_simulations_hmMDP": "res/simhmMDP/simhmMDP2states10actions.csv",
    "file_simulations_params": "res/simparams/simparams2states10actions.csv",
    "performance_file": "res/performance/performance2states10actions.csv",
}
