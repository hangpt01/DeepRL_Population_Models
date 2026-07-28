"""Monte Carlo estimation of universal model parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd


def value_function(init: str, a: float, b: float, r_low: float, r_high: float, disc: float) -> float:
    if init == "low":
        return (r_low * (1 - disc + disc * b) + r_high * (disc - disc * a)) / (
            (1 - disc) * (1 - a * disc + b * disc)
        )
    return (r_high * (1 - a * disc) + r_low * b * disc) / ((1 - disc) * (1 - a * disc + b * disc))


def value_all_policies(init: str, parameters: np.ndarray, rew: np.ndarray, disc: float) -> np.ndarray:
    values = []
    n = parameters.shape[1]
    for i in range(n):
        for j in range(n):
            values.append(value_function(init, parameters[0, i], parameters[1, j], rew[0, i], rew[1, j], disc))
    return np.asarray(values, dtype=float)


def mean_parameters(rewards: np.ndarray, disc: float, n_trials: int = int(1e4), seed: int | None = None) -> pd.DataFrame:
    n_param = rewards.size
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(n_trials):
        params = rng.random(n_param).reshape(2, -1)
        values_low = value_all_policies("low", params, rewards, disc)
        values_high = value_all_policies("high", params, rewards, disc)
        pol_opt_low = int(np.argmax(values_low) + 1)
        pol_opt_high = int(np.argmax(values_high) + 1)

        flat_params = params.reshape(-1, order="F")
        row_low = {f"p{i+1}": flat_params[i] for i in range(n_param)}
        row_low["opt"] = pol_opt_low
        row_low["init"] = "low"

        row_high = {f"p{i+1}": flat_params[i] for i in range(n_param)}
        row_high["opt"] = pol_opt_high
        row_high["init"] = "high"

        rows.extend([row_low, row_high])

    df = pd.DataFrame(rows)
    p_cols = [f"p{i+1}" for i in range(n_param)]
    return df.groupby("opt", as_index=False)[p_cols].mean()
