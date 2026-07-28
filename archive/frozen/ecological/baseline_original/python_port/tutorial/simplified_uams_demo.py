"""
Simplified UAMS tutorial on a toy 2-state/2-action problem.

Goal:
  - Show data representation used in this project
  - Show a simplified "proposal" pipeline:
      1) sample transition parameters
      2) group them by optimal policy
      3) build representative (universal) models
      4) manage an unknown true system with belief updates
  - Compare against simple baselines

No SARSOP/POMDPX required here.
This is intentionally compact and educational, not production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


LOW = 0
HIGH = 1


@dataclass
class ToyProblem:
    reward: np.ndarray  # shape [2, A]
    gamma: float
    horizon: int
    true_params: np.ndarray  # [p1L, p1H, p2L, p2H] for A=2
    b_full: np.ndarray  # initial physical-state distribution


def params_to_transition(params: np.ndarray, n_actions: int) -> np.ndarray:
    """
    Convert flattened parameters [S*A] into transition tensor [S, S, A].
    For action a:
      - p_aL = P(low -> low)
      - p_aH = P(high -> low)
      Transition matrix is:
        [[p_aL, 1-p_aL],
         [p_aH, 1-p_aH]]
    """
    tr = np.zeros((2, 2, n_actions), dtype=float)
    for a in range(n_actions):
        p_l = params[2 * a]
        p_h = params[2 * a + 1]
        tr[:, :, a] = np.array([[p_l, 1 - p_l], [p_h, 1 - p_h]], dtype=float)
    return tr


def policy_id_for_2x2_stationary(tr: np.ndarray, rew: np.ndarray, gamma: float) -> int:
    """
    Policy ID in {0,1,2,3} for stationary policy mapping:
      id 0: low->a0, high->a0
      id 1: low->a0, high->a1
      id 2: low->a1, high->a0
      id 3: low->a1, high->a1
    We choose the policy with largest average value over initial states.
    """
    candidates = [(0, 0), (0, 1), (1, 0), (1, 1)]
    values = []
    for a_low, a_high in candidates:
        t_pi = np.array(    # shape [2, 2], state-action transition probabilities for each state
            [
                tr[LOW, :, a_low],
                tr[HIGH, :, a_high],
            ],
            dtype=float,
        )
        # breakpoint()
        r_pi = np.array([rew[LOW, a_low], rew[HIGH, a_high]], dtype=float)  # immediate rewards for each state
        # Solve (I - gamma*T) V = R
        v = np.linalg.solve(np.eye(2) - gamma * t_pi, r_pi)
        values.append(float(np.mean(v)))
    return int(np.argmax(values))


def build_simplified_universal_models(
    reward: np.ndarray,
    gamma: float,
    n_actions: int,
    n_trials: int = 2000,
    seed: int = 0,
) -> Dict[int, np.ndarray]:
    """
    Simplified MC-UAMS idea:
      - Draw random parameter vectors
      - Find optimal policy region of each draw
      - Average params within each region -> representative model
    Returns {policy_id -> mean_params}.
    """
    rng = np.random.default_rng(seed)
    buckets: Dict[int, List[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}     # solve for 4 policies
    for _ in range(n_trials):
        p = rng.random(2 * n_actions)
        tr = params_to_transition(p, n_actions)
        pid = policy_id_for_2x2_stationary(tr, reward, gamma)
        buckets[pid].append(p)  # count number of trials that policy is optimal

    reps: Dict[int, np.ndarray] = {}     # {policy_id -> mean_params}
    for pid, arr in buckets.items():
        if len(arr) > 0:
            reps[pid] = np.mean(np.asarray(arr), axis=0)
    return reps

def expected_transition_from_belief(belief: np.ndarray, models: List[np.ndarray]) -> np.ndarray:
    """Belief-weighted transition model. Outputs expected transition matrix from multiple models and belief."""
    out = np.zeros_like(models[0], dtype=float)
    for w, m in zip(belief, models):
        out += w * m
    return out


def finite_horizon_dp(tr: np.ndarray, rew: np.ndarray, gamma: float, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """DP for known MDP; returns value and policy for each time/state.
    Solve MDP for H remaining steps via backward induction, for time t = horizon - 1 to 0.
    q(s, a) = r(s, a) + gamma * E[V(s', a')]     # action-value function
    v(s, t) = max_a q(s, a)                      # value function
    policy(s, t) = argmax_a q(s, a)
    """
    n_states, _, n_actions = tr.shape
    v = np.zeros((n_states, horizon + 1), dtype=float)
    policy = np.zeros((n_states, horizon), dtype=int)
    for t in range(horizon - 1, -1, -1):
        q = np.zeros((n_states, n_actions), dtype=float)
        for a in range(n_actions):
            q[:, a] = rew[:, a] + gamma * (tr[:, :, a] @ v[:, t + 1])       # matrix multiplication of transition matrix [S,S] and value function [S]
        policy[:, t] = np.argmax(q, axis=1)
        v[:, t] = q[np.arange(n_states), policy[:, t]]
    return v, policy


def sample_next_state(rng: np.random.Generator, tr: np.ndarray, s: int, a: int) -> int:
    p_low = tr[s, LOW, a]      # reads from true transition matrix
    return LOW if rng.random() <= p_low else HIGH


def update_model_belief(
    belief: np.ndarray,
    models: List[np.ndarray],
    s_prev: int,
    a: int,
    s_next: int,
) -> np.ndarray:
    """Bayes update over hidden models given observed transition.
    Belief not over states, but over models.
    """
    likelihood = np.array([m[s_prev, s_next, a] for m in models], dtype=float)
    post = belief * likelihood
    z = np.sum(post)
    if z <= 0:
        return np.repeat(1.0 / len(models), len(models))
    return post / z


def run_episode_proposal(problem: ToyProblem, model_params: List[np.ndarray], rng: np.random.Generator) -> float:
    """Simplified adaptive policy: DP on belief-weighted model + Bayes belief updates."""
    models = [params_to_transition(p, problem.reward.shape[1]) for p in model_params]
    true_tr = params_to_transition(problem.true_params, problem.reward.shape[1])
    belief = np.repeat(1.0 / len(models), len(models))      # no prior preference, use uniform belief for states
    s = LOW if rng.random() <= problem.b_full[LOW] else HIGH

    total = 0.0
    for t in range(problem.horizon):
        tr_exp = expected_transition_from_belief(belief, models)
        _, pol = finite_horizon_dp(tr_exp, problem.reward, problem.gamma, problem.horizon - t)     # choose the best value for each time step using the horizon backward -> this time step
        a = int(pol[s, 0])
        total += (problem.gamma**t) * problem.reward[s, a]
        s_next = sample_next_state(rng, true_tr, s, a)
        belief = update_model_belief(belief, models, s, a, s_next)
        s = s_next
    return float(total)


def run_episode_fixed(problem: ToyProblem, action: int, rng: np.random.Generator) -> float:
    """Baseline: always do one fixed action."""
    true_tr = params_to_transition(problem.true_params, problem.reward.shape[1])
    s = LOW if rng.random() <= problem.b_full[LOW] else HIGH
    total = 0.0
    for t in range(problem.horizon):
        total += (problem.gamma**t) * problem.reward[s, action]
        s = sample_next_state(rng, true_tr, s, action)
    return float(total)


def run_episode_random(problem: ToyProblem, rng: np.random.Generator) -> float:
    """Baseline: pick random action each step."""
    true_tr = params_to_transition(problem.true_params, problem.reward.shape[1])
    s = LOW if rng.random() <= problem.b_full[LOW] else HIGH
    total = 0.0
    n_actions = problem.reward.shape[1]
    for t in range(problem.horizon):
        a = int(rng.integers(0, n_actions))
        total += (problem.gamma**t) * problem.reward[s, a]
        s = sample_next_state(rng, true_tr, s, a)
    return float(total)


def demo() -> None:
    # Tiny toy data: 2 states x 2 actions
    problem = ToyProblem(
        reward=np.array(
            [
                [1.0, 2.0],  # rewards if current state is LOW
                [5.0, 4.0],  # rewards if current state is HIGH
            ],
            dtype=float,
        ),
        gamma=0.9,
        horizon=20,
        # Hidden true transition parameters [a0_L, a0_H, a1_L, a1_H]
        true_params=np.array([0.85, 0.35, 0.65, 0.20], dtype=float),
        b_full=np.array([1.0, 0.0], dtype=float),
    )

    print("=== Simplified UAMS Demo (2-state, 2-action) ===")
    print("\nData representation:")
    print(f"- reward matrix shape: {problem.reward.shape}")
    print(problem.reward)
    print(f"- true params [p(a0,L), p(a0,H), p(a1,L), p(a1,H)]: {problem.true_params.tolist()}")
    print(f"- gamma={problem.gamma}, horizon={problem.horizon}, initial_state_dist={problem.b_full.tolist()}")

    reps = build_simplified_universal_models(       # returns policy_id -> mean_params
        reward=problem.reward,
        gamma=problem.gamma,
        n_actions=problem.reward.shape[1],
        n_trials=3000,
        seed=7,
    )
    model_params = [reps[k] for k in sorted(reps.keys())]
    print(f"\nFlow step: built {len(model_params)} representative hidden models from Monte Carlo policy regions.")
    for i, p in enumerate(model_params, start=1):
        print(f"  model{i}: {np.round(p, 3).tolist()}")

    rng = np.random.default_rng(123)
    n_episodes = 300
    proposal_vals = np.array([run_episode_proposal(problem, model_params, rng) for _ in range(n_episodes)])
    fixed_a0_vals = np.array([run_episode_fixed(problem, 0, rng) for _ in range(n_episodes)])
    fixed_a1_vals = np.array([run_episode_fixed(problem, 1, rng) for _ in range(n_episodes)])
    random_vals = np.array([run_episode_random(problem, rng) for _ in range(n_episodes)])

    # Upper-bound style reference: planner knows true model exactly.
    true_tr = params_to_transition(problem.true_params, 2)
    v_true, _ = finite_horizon_dp(true_tr, problem.reward, problem.gamma, problem.horizon)
    known_model_value = float(v_true[LOW, 0])

    print("\nResults (mean discounted return over episodes):")
    print(f"- Proposal (simplified adaptive): {proposal_vals.mean():.3f}")
    print(f"- Baseline fixed action a0      : {fixed_a0_vals.mean():.3f}")
    print(f"- Baseline fixed action a1      : {fixed_a1_vals.mean():.3f}")
    print(f"- Baseline random action        : {random_vals.mean():.3f}")
    print(f"- Reference (knows true model)  : {known_model_value:.3f}")

    print("\nInterpretation:")
    print("- Proposal learns model belief from observed transitions and adapts actions.")
    print("- Fixed/random baselines do not learn hidden dynamics, so performance is usually lower.")
    print("- The known-model reference is an upper target when uncertainty is removed.")


if __name__ == "__main__":
    demo()
