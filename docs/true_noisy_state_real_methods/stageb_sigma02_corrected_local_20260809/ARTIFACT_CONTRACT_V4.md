# Artifact Contract V4

Artifact Contract V4 is a tracked, evaluator-free correction to the fitted-artifact
projection contract. It supersedes V3 for future fit probes and pre-execution evidence but
does not alter or validate any V1--V3 probe receipt. No scientific fit, evaluator, truth
archive, runtime `next_states`, action rollout, or return is authorized by this contract.

## Ecological fitted cache

The canonical ecological cache uses schema
`corrected_stageb_ricker_fit_cache_v2`. For the ordered `C` fitted candidates and `A`
registered actions, it projects the real `MechanisticModel` directly:

- integer `candidate_ids` and unique real `candidate_labels`;
- registered `form` and the complete public `action_channels` matrix;
- float64 `growth`, `mortality`, `capacity_increment`, and `stocking` matrices;
- candidate-wise float64 `process_scale`, `observation_scale`, `survey_scale`,
  `initial_capacity`, `capacity_ceiling`, `reset_log_mean`, `reset_log_scale`, and
  `theta_exponent`;
- float64 depensation thresholds, regime multipliers, and regime matrices;
- constants `adapted_mechanistic_v2` and `discrete_current_then_switch_v1`; and
- each real `MechanisticModel.parameter_hash()`.

All numeric values are direct projections. The contract never creates a scalar `r`, never
computes `growth - mortality`, and never substitutes a scalar capacity or ecological
`residual_sigma`. Growth and mortality are nonnegative and mutually exclusive per
candidate/action. Capacity increment and stocking must agree with the registered public
action channel. Regime rows must sum to one within `1e-10`.

The ecological `residual_process_scales` component now stores `process_scale`. It is finite,
nonnegative, unfloored, candidate-ordered, and bit-identical to the cache, including
subnormal and near-zero positive values. It binds the complete V2 cache by SHA-256.

## Method-scoped scale diagnostics

The transition-diagnostic receipt is
`corrected_stageb_paired_transition_diagnostics_v2`. The registration analysis rule is
`corrected_stageb_analysis_rules_v2`.

- RefPlan, OGSRL, and BA-MCTS retain `residual_sigma >= 0.02`, exact floor-active
  semantics, member identities, and dynamics-ensemble bindings.
- PLUS and MOOR report exact candidate `process_scale`, `below_general_floor`, a null
  applicable ecological floor, and the explicit semantics
  `general_learned_dynamics_only`. Their values are never clamped.

The previous-results-known disclosure remains required. This schema change does not make a
state-representation causal claim and does not change the end-to-end estimand.

## Evidence and fit probe

Only `corrected_stageb_artifact_bundle_evidence_v4` is valid for new V4 evidence. The V3 and
V4 validators reject one another. V4 retains component-specific deterministic fixtures,
their manifest, canonical component hashes, fresh reload/replay, matched Arm-O surrogate
requirements, ecological O/T shared-policy identity, and the closed information boundary.

`fit_probe.py` is the single tracked V4 probe implementation. Its twelve task identities use
the exact two registered cells, all six complete method names, and the registered interpreter
roles. It projects an already fitted caller-supplied source object; it cannot fit a model or
construct an evaluator. The source projection diagnostic is fsynced before component
validation. Successful publication is atomic and binds the receipt to the tracked script
SHA-256 and controlling commit. Runtime future state, truth, and returns are absent.

The OGSRL `(11,3)` actor is projected as a `(2,11)` weight matrix from columns 1--2 and an
11-vector intercept from column 0. The crab-eating-fox display name and the complete EVD
method identifier are fixed. Real-object roundtrip validation uses the registered operation,
arguments, keyword arguments, and repository root.

## Evidence boundary

All V3 probe receipts are explicitly excluded from final V4 evidence. A future real probe
must run from committed V4 bytes and create new receipts for all twelve method/cell paths.
This development pass creates no registration, key, fit receipt, driver output, or scientific
result.
