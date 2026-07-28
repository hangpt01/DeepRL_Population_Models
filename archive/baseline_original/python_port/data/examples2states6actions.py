import numpy as np

N = 6
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
    "Tmax": 4,
    "Tmax_hmMDP": 50,
    "N_MDP": 100,
    "file_mean_params": "res/meanparams/meanparams2states6actions.csv",
    "file_random_mdp": "res/randomMDP/2states6actions_randomMDP.csv",
    "file_pomdpx": "res/POMDPX/2states6actions.pomdpx",
    "file_outpolicy": "data/POLICYX/2states6actions.policyx",
    "file_simulations_opt": "res/simopt/simopt2states6actions.csv",
    "file_simulations_hmMDP": "res/simhmMDP/simhmMDP2states6actions.csv",
    "file_simulations_params": "res/simparams/simparams2states6actions.csv",
    "performance_file": "res/performance/performance2states6actions.csv",
}
