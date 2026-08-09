# I1 addendum — degenerate (zero-standard-deviation) feature rule

**Date:** 2026-08-08 (Australia/Melbourne)  
**Status:** ADDENDUM TO THE FROZEN I1 REGISTRATION; SUPPLEMENTS, DOES NOT OVERWRITE  
**Registration:** `method_level_state_information_pilot_i1_20260808`  
**Technical authority:** `CLAUDE_ZERO_SD_RULE_MEMO.md`, SHA-256
`2cbd27fbe39bdd5bd415f74eca1e80f84a74cfb85778263831088f0ae18792c4`  
**Execution status:** NOT AUTHORIZED. This addendum creates no data, code, test,
configuration, adapter, model, evaluation, result, or job.

## A1. Purpose and controlling status

This additive registration closes only the constant/near-zero feature-transformation
blocker identified when I2 Increment A stopped. It supplements the original six I1
artifacts without changing them. Revision 3.1, the Claude I1 audit, all other I1
information-boundary, provenance, evaluation, interpretation and stop rules remain
controlling. `SEALED_PREDICTIONS.json` remains byte-identical and unchanged; existing
prediction P7 already covers comparisons with changed fitted artifacts.

This addendum does not authorize I2, truth extraction, real derived data, adapters,
tests, regression, rebaseline, Stage B, Stage C, scientific evaluation, or Slurm work.
Any future implementation that conflicts with this addendum must stop rather than
repair, reinterpret, approximate, or silently replace the registered rule.

## A2. Feature identity and measured behavior

The affected input is `BeliefState.public_features(observation_scale)` column 1,
`sd`, in the fixed ten-column float64 order:

1. `mean`
2. `sd`
3. `q10`
4. `q50`
5. `q90`
6. `extinct`
7. `ctx_prev_obs`
8. `ctx_obs`
9. `ctx_t`
10. `ess_frac`

Indices are zero-based in implementation and receipts. Under a point-mass Arm T belief,
column 1 may be literal `0.0` or may contain floating-point reduction residue as large as
approximately `4.441e-16`. Exact-equality detection such as `sd == 0.0` is therefore
forbidden as the primary rule.

Arm O already contains exactly constant columns 5 (`extinct`) and 9 (`ess_frac`). In
Arm T, columns 2–4 become bit-identical to column 0. The state-derived block, columns
0–4, consequently collapses from measured rank five in Arm O to rank one in Arm T.
The registered feature order must be preserved. Degenerate or collinear columns must
not be removed, merged, jittered, reordered, or silently reconstructed.

## A3. Affected methods

RefPlan-inspired, OGSRL-inspired, BA-MCTS-inspired and EVD pessimism consume the public
representation directly or indirectly and are affected. Adapted PLUS and adapted MOOR
do not consume `public_features`; this rule is vacuous for their registered direct
point-mass latent-belief implementation and does not change their frozen-fit eligibility.

## A4. Primary Arm T construction rule

The future external Arm T adapter must use `numpy.float64` throughout and must construct
the degenerate entries by direct assignment:

```text
features[:, 1]   = 0.0
features[:, 2:5] = features[:, 0][:, None]
```

Column 1 must be assigned the literal float64 value `0.0`; columns 2–4 must be exact
copies of column 0. The adapter must not compute weighted moments and then detect or
repair their residue. It must emit no artificial noise or epsilon variance.

The registered `std(column) <= 1e-8` rule is only a fail-closed validation guard on the
constructed view, not the construction mechanism. If the assigned zero, alias,
finiteness, dtype, order, or rank invariants fail, processing stops; values must not be
silently repaired.

This rule belongs only in a separately authorized external Arm T module. It must not
change `behavior_model.py`, `public_surrogate.py`, `src/tracks/**`, or any Arm O
preprocessing code. Existing floors remain unchanged: `1e-8` for the behavior model and
public surrogate, `1e-6` for the OGSRL kNN guardian, `0.02` for `residual_sigma`, and
`0.03` for filter sigma.

## A5. End-to-end Arm T preprocessing rule

For a primary end-to-end Arm T fit:

- fit preprocessing on the Arm T training view;
- construct an explicit constant-feature mask using the inclusive validation criterion
  `std(column) <= 1e-8`;
- for each masked column, store `offset` as the float64 fitted-column mean and store
  `scale` as literal float64 `1.0`;
- do not use the existing `1e-8` scale floor as the scale for the assigned SD feature;
- emit literal float64 `0.0` for a masked feature rather than calculating
  `(value - offset) / scale`;
- use one shared transformation implementation and the identical serialized offset,
  scale, mask, dtype and feature order at offline fit and runtime;
- serialize floats with exact float64 round-trip behavior, never fixed-decimal text;
- explicitly record the alias group `[0, 2, 3, 4]`, state-block rank, and design-matrix
  condition number; coefficients split across collinear columns must not be interpreted;
- retain every registered column in its registered position.

The runtime guard for a masked constant is
`abs(value - stored_offset) <= 1e-6` in log1p feature space. The bound is absolute, with
no relative scaling term. Every fit-time and runtime column must pass
`numpy.isfinite(...).all()`.

Offline/runtime parity is fail-closed. Feature names and order, float64 dtype, constant
mask, aliases, offsets, scales, transformation code, serialized artifact and artifact
SHA-256 must agree. Runtime loads the fit artifact and never recomputes its statistics.
A missing or reordered column, nonfinite value, masked value outside tolerance, artifact
hash mismatch, or unexpectedly degenerate unmasked column is a hard stop.

A changed preprocessor, feature fit, dynamics fit, surrogate, residual scale, policy,
or any other learned artifact makes the arm contrast an end-to-end bundle comparison and
requires the exact label:

`MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`

It must not be described as frozen-fit, state-input-only, or a causal state-representation
contrast.

## A6. Frozen-fit secondary branch

For RefPlan's optional frozen-fit secondary diagnostic only, reuse the Arm O preprocessor
byte-for-byte. Its artifact hash, feature order, dtype, offsets, scales, masks, learned
fits, policy parameters, and every other frozen artifact must match Arm O. Feed the Arm T
feature vector through that unchanged transformation. Any preprocessor refit or altered
preprocessing artifact disqualifies the comparison from the frozen-fit label.

This branch is distinct from the primary end-to-end estimand. It is vacuous for adapted
PLUS and adapted MOOR and unavailable to OGSRL, BA-MCTS and EVD under the frozen I1
eligibility rules.

## A7. RefPlan mismatch disclosure

The accepted RefPlan prior was fitted with a variable SD feature, while its planner-root
SD was hard-assigned to zero. Claude measured the resulting standardized planner-root
coordinate at approximately `-2.7018` for tiger and `-2.7348` for fox. Every future
RefPlan receipt and report must disclose and record this pre-existing train/runtime
mismatch for both arms.

Arm T may remove the mismatch. The RefPlan true-versus-noisy contrast may therefore
partly reflect correction of an existing feature-distribution mismatch and cannot
automatically be attributed solely to improved abundance information. This disclosure
does not alter the sealed predictions; P7 already governs changed fitted artifacts.
For the frozen-fit branch, the retained approximately -2.70-standard-unit value is an
out-of-distribution extrapolation and must also be receipted.

## A8. RNG invariant

The memo verified that `rng.normal(loc, 0.0)` returns `loc` exactly while still advancing
the RNG state. Future I2 synthetic tests must verify both the exact returned abundance
and identical intended RNG advancement and call count across paired interfaces. No RNG
implementation is authorized by this addendum.

## A9. Mandatory future receipt fields

One receipt is required per method × species × cell × arm before returns are opened.
At minimum it must record:

- schema version, method, species, cell, arm and regime;
- feature schema, names, fixed column order, and feature-order hash;
- dtype and construction mode;
- per-column raw mean, standard deviation, minimum and maximum;
- explicit constant mask, stored offsets and stored scales;
- transformed constant values and zero-variance events;
- alias/rank-collapse map, collinear groups, state-block or design rank, and condition
  number or a registered equivalent;
- `constant_threshold = 1e-8`, inclusive `<=` comparison, and
  `runtime_constant_tolerance = 1e-6`;
- finite-check result;
- Arm O and Arm T preprocessing artifact hashes, fit-view hash, and runtime-transform
  hash;
- fit/runtime parity result and enumerated failures;
- maximum absolute assigned-SD value;
- exact-equality results for columns 2–4 against column 0;
- RefPlan standardized SD offsets for both arms and the mismatch/OOD disclosure;
- RNG-call-count and RNG-state-advancement parity;
- fitted-artifact and every available `residual_sigma` identity required by I1;
- per-method frozen-fit eligibility or end-to-end-only declaration and mandatory label.

The detailed receipt schema in `CLAUDE_ZERO_SD_RULE_MEMO.md` §8 is controlling. Every
listed field is mandatory. `parity_result` must be `PASS` before scientific execution;
`FAIL` is a hard stop.

## A10. Mandatory future I2 synthetic tests

During the currently contemplated I2 Increment A, all fixtures must be synthetic. Tests
must cover:

1. the exact ten-column float64 schema and fixed order;
2. direct literal-zero assignment to column 1, including cases where reduction-based
   construction would leave nonzero residue;
3. bit-exact aliases from columns 2–4 to column 0;
4. `std <= 1e-8` as a validation guard only, including the inclusive boundary;
5. constant mask, float64 offset, literal scale `1.0`, and literal transformed `0.0`;
6. identical offline-fit and runtime transforms and serialized-artifact hash identity;
7. rank-collapse and condition-number diagnostics without dropping or reordering columns;
8. finite-value and runtime `1e-6` tolerance checks;
9. negative fail-closed cases for missing/reordered columns, wrong dtype, nonfinite input,
   mismatched masks/offsets/scales/aliases/hashes, out-of-tolerance masked values, and
   unexpected degeneracy of an unmasked column;
10. Arm O isolation and byte-identical RefPlan frozen-fit preprocessing reuse;
11. RefPlan standardized SD offsets and required mismatch disclosures for both arms;
12. exact `rng.normal(loc, 0.0)` output plus paired RNG call-count and state-advancement
    parity;
13. complete synthetic receipt emission, including maximum assigned-SD magnitude,
    exact alias checks, eligibility declaration, and `parity_result = PASS`.

These requirements authorize no implementation or test execution now.

## A11. Permitted and forbidden actions

This documentation-only addendum permits only creation and validation of its four named
documentation artifacts and read-only integrity checks.

It forbids editing any existing I1, audit, memo, plan, accepted-output, source, test,
configuration, result, or frozen-track artifact; opening or extracting real truth arrays
or private truth fields; constructing derived Arm T data; implementing adapters or
preprocessors; fitting models; running tests, regressions, rebaselines or evaluations;
creating scientific results or execution namespaces; submitting Slurm jobs; and beginning
I2, Stage B or Stage C.

## A12. Authorization status

- Original I1 registration: frozen and unchanged.
- This zero-SD addendum: documentation-only freeze authorized by the user.
- Sealed predictions: unchanged.
- I2 and I2 Increment A implementation: not authorized by this addendum.
- Truth access/extraction and real derived Arm T data: not authorized.
- Code, adapters, tests and configurations: not authorized.
- Regression, rebaseline, evaluation, Stage B and Stage C: not authorized.
- Slurm jobs: none submitted.
- Accepted outputs and frozen tracks: untouched.

