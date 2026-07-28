"""Calibrated public behavior-policy diagnostics and proposal model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def augment_features(features: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features)), features])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class CalibratedBehaviorModel:
    """Standardized multinomial-logistic behavior model fitted on train only."""

    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        design = (augment_features(features) - self.mean) / self.scale
        return _softmax(design @ self.weights.T)


def fit_reference_behavior(
    features: np.ndarray,
    actions: np.ndarray,
    num_actions: int,
    ridge: float = 1e-2,
    iterations: int = 400,
    learning_rate: float = 0.3,
) -> CalibratedBehaviorModel:
    """Fit the registered calibrated public behavior model."""

    design = augment_features(np.asarray(features, dtype=np.float64))
    mean = design.mean(axis=0)
    scale = np.maximum(design.std(axis=0), 1e-8)
    mean[0] = 0.0
    scale[0] = 1.0
    standardized = (design - mean) / scale
    weights = np.zeros((num_actions, standardized.shape[1]))
    onehot = np.eye(num_actions)[np.asarray(actions, dtype=int)]
    for _ in range(iterations):
        probs = _softmax(standardized @ weights.T)
        gradient = (probs - onehot).T @ standardized / len(standardized)
        gradient += ridge * weights
        weights -= learning_rate * gradient
    return CalibratedBehaviorModel(weights, mean, scale)


def behavior_nll(
    model: CalibratedBehaviorModel, features: np.ndarray, actions: np.ndarray
) -> float:
    probs = np.clip(model.probabilities(features), 1e-12, 1.0)
    return float(-np.mean(np.log(probs[np.arange(len(features)), actions.astype(int)])))
