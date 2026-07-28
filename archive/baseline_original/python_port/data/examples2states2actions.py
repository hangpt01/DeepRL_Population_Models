import numpy as np

CONFIG = {
    "reward": np.array([[1.0, 3.0], [5.0, 7.0]], dtype=float),
    "gamma": 0.9,
    "b_full": np.array([1.0, 0.0], dtype=float),
    "Tmax": 8,
    "Tmax_hmMDP": 50,
    "N_MDP": 100,
    "file_mean_params": "res/meanparams/meanparams2states2actions.csv",
    "file_random_mdp": "res/randomMDP/2states2actions_randomMDP.csv",
    "file_pomdpx": "res/POMDPX/2states2actions.pomdpx",
    "file_outpolicy": "data/POLICYX/2states2actions.policyx",
    "file_simulations_opt": "res/simopt/simopt2states2actions.csv",
    "file_simulations_hmMDP": "res/simhmMDP/simhmMDP2states2actions.csv",
    "file_simulations_params": "res/simparams/simparams2states2actions.csv",
    "performance_file": "res/performance/performance2states2actions.csv",
}
