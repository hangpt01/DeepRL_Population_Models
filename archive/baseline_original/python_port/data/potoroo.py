import numpy as np

CONFIG = {
    "reward": np.array(
        [
            [0.00, -1.00, -2.00, -2.36, -3.36, -4.36],
            [20.00, 19.00, 18.00, 17.64, 16.64, 15.64],
        ],
        dtype=float,
    ),
    "gamma": 0.9,
    "b_full": np.array([1.0, 0.0], dtype=float),
    "Tmax": 4,
    "Tmax_hmMDP": 50,
    "N_MDP": 100,
    "file_mean_params": "res/meanparams/meanparamspotoroo.csv",
    "file_random_mdp": "res/randomMDP/potoroo_randomMDP.csv",
    "file_pomdpx": "res/POMDPX/potoroo.pomdpx",
    "file_outpolicy": "data/POLICYX/potoroo.policyx",
    "file_simulations_opt": "res/simopt/simoptpotoroo.csv",
    "file_simulations_hmMDP": "res/simhmMDP/simhmMDPpotoroo.csv",
    "file_simulations_params": "res/simparams/simparamspotoroo.csv",
    "performance_file": "res/performance/performancepotoroo.csv",
}
