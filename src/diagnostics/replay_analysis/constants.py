"""Registered constants for the three-species P=10 benchmark.

Every value here is sourced from the read-only code audits, not assumed.
  s_safe, K_ref          : audit 1, B2   (config.py:332,337)
  cost_step              : audit 2, P0-1 (identical across species)
  r_setpoint tables      : audit 2, P0-1 / audit 1, C1
  gamma, horizon         : accepted evaluator
  process_noise_sigma=0  : audit 1, C3 (config.py:82)
"""
from __future__ import annotations

GAMMA = 0.95
HORIZON = 50
COLLAPSE_PENALTY = 10.0
N_ACTIONS = 11
N_PLUS_CANDIDATES = 8
PROCESS_NOISE_SIGMA = 0.0

#: discount weight sum, sum_{t=0}^{49} gamma^t
DISCOUNT_SUM = sum(GAMMA ** t for t in range(HORIZON))  # 18.4616...

SPECIES = {
    "amur_tiger": dict(
        K_ref=250.0, s_safe=25.0, safety_fraction=0.10, N0=200.0,
        r_ricker=[-0.0458, -0.1857, -0.4753, 0.0275, 0.0707,
                  -0.0458, -0.0458, 0.0275, 0.0275, 0.0707, -0.0458],
        r_lgm=[-0.0448, -0.1695, -0.3783, 0.0279, 0.0733,
               -0.0448, -0.0448, 0.0279, 0.0279, 0.0733, -0.0448],
        dK=[0, 0, 0, 0, 0, 25.0, 75.0, 25.0, 75.0, 75.0, 0],
        dN=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20.0],
    ),
    "crab_eating_fox": dict(
        K_ref=41.0, s_safe=10.25, safety_fraction=0.25, N0=41.0,
        r_ricker=[0.3230, 0.2098, -0.0131, 0.3779, 0.4418,
                  0.3230, 0.3230, 0.3779, 0.3779, 0.4418, 0.3230],
        r_lgm=[0.3813, 0.2334, -0.0130, 0.4592, 0.5555,
               0.3813, 0.3813, 0.4592, 0.4592, 0.5555, 0.3813],
        dK=[0, 0, 0, 0, 0, 4.1, 12.3, 4.1, 12.3, 12.3, 0],
        dN=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.1],
    ),
    "egyptian_vulture": dict(
        K_ref=325.0, s_safe=81.25, safety_fraction=0.25, N0=41.0,
        r_ricker=[-0.0912, -0.2179, -0.2558, -0.0198, -0.0098,
                  -0.0912, -0.0912, -0.0198, -0.0198, -0.0098, -0.0912],
        # KNOWN INERT DISCREPANCY (F1): entries a1/a2/a3/a7/a8 were copied
        # from the Ricker column instead of the LGM table. No current caller
        # selects r_column="r_lgm", and the derived dominance/bound results are
        # unchanged. Preserve these values pending a separate reviewed fix.
        r_lgm=[-0.0872, -0.2179, -0.2558, -0.0198, -0.0098,
               -0.0872, -0.0872, -0.0198, -0.0198, -0.0098, -0.0872],
        dK=[0, 0, 0, 0, 0, 32.5, 97.5, 32.5, 97.5, 97.5, 0],
        dN=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.1],
    ),
}

#: identical across all species (audit 2, P0-1)
COST_STEP = [0.0, -0.05, -0.10, 0.25, 0.50, 0.1875, 0.50, 0.4375, 0.75, 1.0, 0.3125]

ACTION_NAMES = [
    "a0 do nothing", "a1 sustainable harvest", "a2 aggressive harvest",
    "a3 predator/disease control", "a4 breeding support",
    "a5 moderate restoration", "a6 intensive restoration",
    "a7 integrated (light)", "a8 adaptive trial",
    "a9 flagship programme", "a10 translocation",
]

#: OGSRL public proxy threshold, audit 2 P0-3 (ricker sigma 0.1 / 0.2)
OGSRL_S_LOW = {
    "amur_tiger": (35.91, 35.15),
    "crab_eating_fox": (31.90, 29.82),
    "egyptian_vulture": (22.93, 22.41),
}

#: methods that consume the shared learned public reward surrogate (audit 2, P0-2)
SURROGATE_METHODS = {"plus", "moor", "refplan", "ogsrl", "bamcts"}
#: the only method trained on raw dataset.rewards
RAW_REWARD_METHODS = {"evd"}
#: methods that consume observation_noise_sigma (audit 2, P0-6)
SIGMA_AWARE_METHODS = {"plus", "moor", "refplan"}

WEAK_REGIME_INDEX = 1          # regime whose growth multiplier is 0.65
REGIME_PERSISTENCE = 0.90
ALLEE_C_FRACTION = (0.18, 0.30)
THETA_RANGE = (3.0, 6.0)


def dominated_actions(species: str, r_column: str = "r_ricker") -> dict[int, int]:
    """Actions strictly dominated because their dK is dynamically inert.

    When r_setpoint <= 0 we have r_pos = 0, the family map collapses to
    x' = (x + dN) * exp(-r_mort), and carrying capacity never enters.  Two
    actions sharing (r_setpoint, dN) are then dynamically identical, so the
    cheaper one strictly dominates.

    Returns {dominated_action: dominating_action}.
    """
    spec = SPECIES[species]
    r, dN, cost = spec[r_column], spec["dN"], COST_STEP
    out: dict[int, int] = {}
    for a in range(N_ACTIONS):
        if r[a] > 0:
            continue  # dK is live; no dominance argument
        for b in range(N_ACTIONS):
            if a == b or r[b] != r[a] or dN[b] != dN[a]:
                continue
            if cost[b] < cost[a]:
                if a not in out or cost[b] < cost[out[a]]:
                    out[a] = b
    return out


def max_reachable_abundance(species: str, r_column: str = "r_ricker") -> float:
    """Supremum of abundance over all policies, when r_pos == 0 for every action.

    With no growth term the map is x' = (x + dN_a) * exp(-h_a).  Only actions
    with dN > 0 can increase x, and every action has h > 0, so the constant
    best-dN action dominates and its fixed point bounds every trajectory.
    Returns nan when some action has r > 0 (the argument does not apply).
    """
    spec = SPECIES[species]
    r, dN = spec[r_column], spec["dN"]
    if any(v > 0 for v in r):
        return float("nan")
    import math
    best = 0.0
    for a in range(N_ACTIONS):
        if dN[a] <= 0:
            continue
        decay = math.exp(r[a])           # r <= 0, so this is exp(-h)
        if decay >= 1.0:
            return float("inf")
        best = max(best, dN[a] * decay / (1.0 - decay))
    return best
