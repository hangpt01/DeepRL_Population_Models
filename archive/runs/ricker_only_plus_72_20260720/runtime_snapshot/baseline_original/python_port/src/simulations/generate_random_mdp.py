"""Generate random true MDPs used for simulations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_random_mdp(reward: np.ndarray, n: int, file_path: str, seed: int | None = None) -> str:
    s = 2
    a = reward.shape[1]
    rng = np.random.default_rng(seed)
    random_mdp = []
    for _ in range(n):
        random_values = []
        for _ in range(2 * a):
            r = float(rng.random())
            random_values.extend([r, 1 - r])
        random_mdp.append(random_values)

    out = Path(file_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(random_mdp).to_csv(out, index=False)
    return str(out)
