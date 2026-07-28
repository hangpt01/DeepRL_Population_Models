import numpy as np

N = 4
CONFIG = {
    "reward": np.array(
        [
            np.linspace(1, 3, N),
            np.linspace(3.5, 6, N),
        ],
        dtype=float,
    ),
    "gamma": 0.9,
    "b_full": np.array([1.0, 0.0], dtype=float),
    "Tmax": 5,
    "Tmax_hmMDP": 50,
    "N_MDP": 100,
    "file_mean_params": "res/meanparams/meanparams2states4actions.csv",
    "file_random_mdp": "res/randomMDP/2states4actions_randomMDP.csv",
    "file_pomdpx": "res/POMDPX/2states4actions.pomdpx",
    "file_outpolicy": "data/POLICYX/2states4actions.policyx",
    "file_simulations_opt": "res/simopt/simopt2states4actions.csv",
    "file_simulations_hmMDP": "res/simhmMDP/simhmMDP2states4actions.csv",
    "file_simulations_params": "res/simparams/simparams2states4actions.csv",
    "performance_file": "res/performance/performance2states4actions.csv",
}
