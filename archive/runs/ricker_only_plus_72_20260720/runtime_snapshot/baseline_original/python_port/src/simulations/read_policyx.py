"""Helpers to read SARSOP .policyx files and evaluate alpha-vectors."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_DIR = CURRENT_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from mdp_tools import bellman_operator


def read_policyx2(file_path: str) -> Dict[str, np.ndarray]:
    root = ET.parse(file_path).getroot()
    vectors = root.findall(".//Vector")
    alpha = []
    alpha_action = []
    alpha_obs = []
    for v in vectors:
        text = (v.text or "").strip()
        alpha.append(np.fromstring(text, sep=" ", dtype=float))
        alpha_action.append(int(v.attrib["action"]) + 1)
        alpha_obs.append(int(v.attrib["obsValue"]) + 1)

    alpha_mat = np.column_stack(alpha) if alpha else np.empty((0, 0), dtype=float)
    return {"vectors": alpha_mat, "action": np.asarray(alpha_action), "obs": np.asarray(alpha_obs)}


def interp_policy2(initial: np.ndarray, obs: int, alpha: np.ndarray, alpha_action: np.ndarray, alpha_obs: np.ndarray):
    ids = np.where(alpha_obs == obs)[0]
    alpha2 = alpha[:, ids]
    alpha_action2 = alpha_action[ids]
    scores = initial @ alpha2
    if np.all(scores == 0):
        return [0.0, 1]
    best = int(np.argmax(scores))
    return [float(scores[best]), int(alpha_action2[best])]


def update_belief(state_prior: np.ndarray, transition: np.ndarray, observation: np.ndarray, z0: int, a0: int) -> np.ndarray:
    l = len(state_prior)
    belief = np.zeros(l, dtype=float)
    for i in range(l):
        belief[i] = float(state_prior @ transition[:, i, a0 - 1] * observation[i, z0 - 1, a0 - 1])
    s = np.sum(belief)
    return belief / s if s > 0 else belief


def mdp_compute_value(P: np.ndarray, PR: np.ndarray, discount: float):
    iter_count = 0
    v = np.zeros(2, dtype=float)
    epsilon = 1e-4
    thresh = epsilon * (1 - discount) / discount
    max_iter = 50000
    while True:
        iter_count += 1
        v_prev = v.copy()
        v, policy = bellman_operator(P, PR, discount, v)
        variation = np.max(v - v_prev)
        if variation < thresh or iter_count == max_iter:
            return {"V": v, "policy": policy}
