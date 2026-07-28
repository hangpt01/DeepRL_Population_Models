"""Privacy / leakage / determinism gates for the corrected hidden general-RL methods.

Goes beyond forbidden-field-name scanning (already covered by
``assert_hidden_method_artifact``) to test complete fitted public artifacts and
the deterministic ``fit -> act -> observe -> act`` lifecycle:

* no fitted array or scalar equals the private carrying capacity ``K_ref`` /
  ``K_base`` or the private ``safety_threshold`` (methods must normalize by the
  data-derived public ``observation_scale`` instead);
* the population token is opaque (no species substring);
* fitting is deterministic given the same seed (complete numeric artifacts);
* a structural-label relabel of the public context does not change the fit
  (family / regime identity cannot leak through the method-visible label).
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from real_ecology_benchmark.config import ModelConfig, PlannerConfig
from real_ecology_benchmark.methods.bamcts import BAMCTSPolicy
from real_ecology_benchmark.methods.ensemble_value_disagreement import (
    EnsembleValueDisagreementPolicy,
)
from real_ecology_benchmark.methods.ogsrl import OGSRLPolicy
from real_ecology_benchmark.methods.refplan import RefPlanPolicy
from real_ecology_benchmark.privacy import assert_hidden_method_artifact
from real_ecology_benchmark.pipeline import _hidden_method_context
from real_ecology_benchmark.types import PublicTransition

from tests.general.real.test_general_paper_mechanisms import _hidden_fixture


MODEL_CFG = ModelConfig(ensemble_size=5)
PLAN_CFG = PlannerConfig(horizon=3, sequences=24, particles=8, ogsrl_cost_horizon=4)


# Public benchmark configuration the method is *constructed* with -- these carry
# public hyperparameters (ensemble size, ridge, VI iteration counts, horizon,
# ...) that are not fitted state and can coincide with round private numbers
# (e.g. native_vi_iterations=250 == a population K_base of 250).  Leakage is
# about FITTED/derived state, so the scan skips these passthrough inputs.
_SCAN_EXCLUDE = {"model_cfg", "planner_cfg", "env_cfg", "rng", "training_history"}

# Complete fitted-artifact hashing excludes only public inputs, runtime RNG/
# telemetry, and externally attached train/holdout views. Every fitted numeric
# member, behavior model, dynamics model, posterior, actor, guardian, and budget
# is traversed without depth or array-length limits.
_ARTIFACT_EXCLUDE = {
    "method_context", "model_cfg", "planner_cfg", "rng", "training_history",
    "training_holdout_dataset", "training_holdout_beliefs", "last_diagnostics",
    "fit_diagnostics",
}


def _complete_public_hash(obj) -> bytes:
    digest = hashlib.sha256()
    active: set[int] = set()

    def update(value):
        if value is None or isinstance(value, (bool, int, float, str, bytes, np.generic)):
            digest.update(type(value).__name__.encode())
            digest.update(repr(value).encode())
            return
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b"array")
            digest.update(str(array.dtype).encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
            return
        identity = id(value)
        if identity in active:
            digest.update(b"cycle")
            digest.update(type(value).__qualname__.encode())
            return
        active.add(identity)
        try:
            if isinstance(value, dict):
                digest.update(b"dict")
                for key in sorted(value, key=repr):
                    update(key); update(value[key])
            elif isinstance(value, (list, tuple)):
                digest.update(type(value).__name__.encode())
                for item in value:
                    update(item)
            elif hasattr(value, "__dict__"):
                digest.update(type(value).__qualname__.encode())
                for key in sorted(vars(value)):
                    if key not in _ARTIFACT_EXCLUDE:
                        update(key); update(vars(value)[key])
            else:
                digest.update(type(value).__qualname__.encode())
                digest.update(repr(value).encode())
        finally:
            active.remove(identity)

    update(obj)
    return digest.digest()


def _iter_scalars(obj, seen=None, depth=0):
    """Yield float scalars reachable from fitted policy state (bounded recursion)."""
    if seen is None:
        seen = set()
    if depth > 6 or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, (int, float, np.floating, np.integer)):
        yield float(obj)
        return
    if isinstance(obj, np.ndarray):
        for v in obj.reshape(-1)[:5000]:
            yield float(v)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            yield from _iter_scalars(v, seen, depth + 1)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _SCAN_EXCLUDE:
                continue
            yield from _iter_scalars(v, seen, depth + 1)
        return
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if k in _SCAN_EXCLUDE:
                continue
            yield from _iter_scalars(v, seen, depth + 1)


def _build(cls, context, cache, dataset, **kw):
    policy = cls(context, MODEL_CFG, PLAN_CFG, seed=7, **kw)
    policy.fit(dataset, cache)
    return policy


class GeneralPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.env_cfg, cls.dataset, cls.private, cls.context,
         cls.filter_cfg, factory, cls.cache) = _hidden_fixture()
        cls.factory = staticmethod(factory)
        cls.K_ref = float(cls.env_cfg.K_ref)
        cls.safety_threshold = float(cls.env_cfg.safety_threshold)
        cls.builders = {
            "refplan": lambda ctx: _build(RefPlanPolicy, ctx, cls.cache, cls.dataset),
            "bamcts": lambda ctx: _build(
                BAMCTSPolicy, ctx, cls.cache, cls.dataset, simulations=24, depth=3
            ),
            "ogsrl": lambda ctx: _build(OGSRLPolicy, ctx, cls.cache, cls.dataset),
            "ensemble_value_disagreement_pessimism": lambda ctx: _build(
                EnsembleValueDisagreementPolicy, ctx, cls.cache, cls.dataset, ensemble_size=6
            ),
        }

    def test_observation_scale_is_not_the_private_scale(self):
        self.assertNotAlmostEqual(self.context.observation_scale, self.K_ref, places=3)
        self.assertNotAlmostEqual(
            self.context.observation_scale, self.safety_threshold, places=3
        )

    def test_no_fitted_value_equals_private_scale_or_threshold(self):
        for name, build in self.builders.items():
            policy = build(self.context)
            values = np.fromiter(_iter_scalars(policy), dtype=float)
            # Exact equality with these specific private floats would be a leak;
            # coincidental equality is effectively impossible at float precision.
            leaks_k = np.isclose(values, self.K_ref, rtol=0.0, atol=1e-6).sum()
            leaks_s = np.isclose(values, self.safety_threshold, rtol=0.0, atol=1e-6).sum()
            self.assertEqual(leaks_k, 0, f"{name}: fitted state contains K_ref exactly")
            self.assertEqual(
                leaks_s, 0, f"{name}: fitted state contains safety_threshold exactly"
            )
            assert_hidden_method_artifact(policy, root=name)

    def test_pop_id_is_opaque(self):
        self.assertTrue(self.context.pop_id.startswith("pop_"))
        self.assertNotIn("tiger", self.context.pop_id.lower())
        self.assertNotIn("amur", self.context.pop_id.lower())

    def test_fit_is_deterministic_under_same_seed(self):
        for name, build in self.builders.items():
            a = build(self.context)
            b = build(self.context)
            va = np.fromiter(_iter_scalars(a), dtype=float)
            vb = np.fromiter(_iter_scalars(b), dtype=float)
            self.assertEqual(len(va), len(vb), f"{name}: fitted shape differs across seeds")
            self.assertTrue(
                np.allclose(va, vb, equal_nan=True),
                f"{name}: identical seed produced different fitted arrays",
            )

    def test_structural_label_relabel_does_not_change_fit(self):
        """The method-visible regime label carries no private structure."""
        relabelled = replace(self.context, regime_label="hidden-demographics_ALT-label")
        self.assertNotEqual(relabelled.regime_label, self.context.regime_label)
        for name, build in self.builders.items():
            base = np.fromiter(_iter_scalars(build(self.context)), dtype=float)
            alt = np.fromiter(_iter_scalars(build(relabelled)), dtype=float)
            self.assertTrue(
                np.allclose(base, alt, equal_nan=True),
                f"{name}: fit depends on the structural label (possible leak)",
            )

    def test_private_value_intervention_invariance(self):
        """One-field private interventions cannot change context, fit, or act."""

        interventions = {
            "K_base": self.env_cfg.K_base * 1.37,
            "r_min": self.env_cfg.r_min - 0.19,
            "r_max": self.env_cfg.r_max + 0.17,
            "kind": "allee" if self.env_cfg.kind != "allee" else "theta",
            "safety_threshold": self.env_cfg.safety_threshold * 1.41,
            "C_low": self.env_cfg.C_low * 0.73,
            "C_high": self.env_cfg.C_high * 1.29,
            "regime_threshold_low": self.env_cfg.regime_threshold_low * 0.61,
            "regime_threshold_high": self.env_cfg.regime_threshold_high * 1.33,
            "population": "PRIVATE_INTERVENTION_SENTINEL",
        }
        base_cfg = SimpleNamespace(environment=self.env_cfg)
        base_context = _hidden_method_context(
            base_cfg, self.dataset, self.context.surrogate
        )
        base_context_bytes = pickle.dumps(base_context, protocol=5)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("lower-level table/config provider accessed during lifecycle")

        lifecycle_patches = (
            patch("real_ecology_benchmark.realdata.pops_for", side_effect=forbidden),
            patch("real_ecology_benchmark.realdata.effects_for", side_effect=forbidden),
            patch("real_ecology_benchmark.realdata.actions_for", side_effect=forbidden),
            patch("real_ecology_benchmark.realdata.public_action_channels", side_effect=forbidden),
            patch("real_ecology_benchmark.actions.resolve_actions", side_effect=forbidden),
            patch("real_ecology_benchmark.native_solver.NativeSolver.build", side_effect=forbidden),
            patch("real_ecology_benchmark.realdata.DATA_DIR", "PRIVATE_DATA_DIR_SENTINEL"),
        )

        def lifecycle(build, context):
            for active in lifecycle_patches:
                active.start()
            try:
                policy = build(context)
                fitted = _complete_public_hash(policy)
                policy.reset(123)
                belief = self.factory().reset(float(self.dataset.observations[0]), 55)
                first = int(policy.act(belief, belief.observation))
                result = PublicTransition(
                    observation=float(self.dataset.next_observations[0]),
                    done=False, truncated=False, terminated=False,
                )
                policy.observe(belief, first, result)
                updated = self.factory().update(belief, first, result.observation)
                second = int(policy.act(updated, updated.observation))
                posterior = _complete_public_hash(getattr(policy, "posterior", None))
                diagnostics = _complete_public_hash(policy.last_diagnostics)
                return fitted, posterior, diagnostics, first, second
            finally:
                for active in reversed(lifecycle_patches):
                    active.stop()

        base = {name: lifecycle(build, base_context) for name, build in self.builders.items()}

        for field, value in interventions.items():
            intervened_env = replace(self.env_cfg, **{field: value})
            intervened_context = _hidden_method_context(
                SimpleNamespace(environment=intervened_env),
                self.dataset,
                self.context.surrogate,
            )
            self.assertEqual(
                pickle.dumps(intervened_context, protocol=5),
                base_context_bytes,
                f"private intervention {field} changed MethodContext bytes",
            )
            for name, build in self.builders.items():
                self.assertEqual(
                    lifecycle(build, intervened_context), base[name],
                    f"{name}: private intervention {field} changed fit/act/observe/act lifecycle",
                )

        # The context boundary itself must also ignore a private data-directory
        # sentinel while preserving the same public action-channel schema.
        changed_dir = replace(self.env_cfg, data_dir="PRIVATE_DATA_DIR_SENTINEL")
        with patch(
            "real_ecology_benchmark.realdata.public_action_channels",
            return_value=self.context.action_channels,
        ):
            changed_dir_context = _hidden_method_context(
                SimpleNamespace(environment=changed_dir), self.dataset, self.context.surrogate
            )
        self.assertEqual(pickle.dumps(changed_dir_context, protocol=5), base_context_bytes)


if __name__ == "__main__":
    unittest.main()
