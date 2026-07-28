# Codex implementation handoff: corrected adapted PLUS/MOOR

Date: 2026-07-17

Controlling contract: `CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md`.

## Scope and non-actions

Implemented the corrected source/config/test/manifest tooling only. No code snapshot was frozen, no
canary or scientific rollout was run, no Slurm job was submitted, and the preserved void run
`real_ecology_runs/paper_faithful_one_cell_4000_20260717_v1/` was not modified.

## Adopted external method IDs

- `plus_adapted_mechanistic_pbvi`
- `moor_adapted_ricker_misspec_pbvi`

The provisional `plus_faithful_pbvi` and `moor_faithful_ricker_misspec_pbvi` IDs are absent from the
runnable registry. Internal filenames retain `faithful` only to avoid unnecessary module churn.

## Implemented scientific corrections

- Strict channel-only public loader for the eleven categorical action channels.
- Fourteen free action effects: eight signed rates, five capacity effects, and one stocking effect.
- Exact signed split with registered bounds and symmetric Clarke half-gradients at zero.
- Form-specific optimizer coordinates; inactive Allee/theta/regime parameters are not optimized.
- Conditional lognormal survey mean `x * exp(sigma_o^2 / 2)` with no sampled survey-noise bank.
- Fixed symmetric regime candidates at persistence `0.80`, `0.90`, and `0.97`; Pi is immutable and
  absent from optimizer coordinates.
- One current-regime abundance update followed by a discrete Pi transition in fitting and deployment.
- Primary objective has no legacy shrinkage, group, or complementarity penalties.
- Separate `hierarchical_weak_v1` diagnostic config; it is never an automatic fallback.

## Cache and provenance

- `fit_transition_hash_v1` covers only observations, actions, next observations, episode IDs,
  timesteps, terminated flags, and truncated flags, including dtype and shape.
- Reward arrays, reward mode, action costs, private fields, and the full dataset hash are excluded from
  fit identity. The full hash remains provenance only.
- Per-key file locks, atomic writes, array hashes, parameter hashes, and fail-loud partial/wrong-schema
  handling are implemented.
- Shared candidate keys are stable across 12/16/32 banks. Regime path sensitivities change only regime
  fit keys; non-regime fits remain reusable.
- Repeated-root hashing is supported by `scripts/hash_paper_faithful_snapshot.py` while preserving the
  prior single-root digest convention.

## Staged execution tooling

- `diagnostic_fit`: 64 return-blind dynamics-fit rows.
- `diagnostic_plan`: 128 reward-specific planning/evaluation rows.
- `sensitivity`: 320 isolated PLUS rows across five configurations.
- `canary`: eight rows, covering two cells, two methods, and both reward modes.
- Fit-only rows use `scripts/run_adapted_fit_row.py` and
  `scripts/slurm/run_adapted_fit_row.sh`; they write cache/model receipts and do not construct an
  evaluator or open return fields.
- Manifests record channel schema, observation protocol, candidate allocation, Pi grid, regime paths,
  regularization variant, cache policy, run stage, blinding rule, and CPU ceilings. Headline
  authorization is explicitly false.

## Acceptance evidence

The tests cover conditional-mean scalar recovery, ordered controlled-Ricker recovery at sigma 0.4,
zero-noise determinism, no sampled survey channel, one-step fitter/deployment identity for all four
forms, immutable Pi artifact round trips, Pi Monte Carlo/kernel agreement, exact 44-to-14 reduction,
Clarke gradients, table independence, reward-independent cache reuse, corruption failure, nested
sensitivity keys, candidate allocations, method renaming, and adapted PLUS integration.

Verification commands:

```bash
PYTHONPATH=src .venv-paper-faithful/bin/python -m pytest -q

.venv-paper-faithful/bin/ruff check --ignore E402,E731 \
  <changed source, script, and test files>
```

The two Ruff exclusions are established repository-wide issues in path-bootstrapping scripts and
legacy lambda factories, not exemptions for corrected fitter logic.

## Still deliberately blocked

- No corrected canary has run, so corrected runtime and memory are unmeasured.
- No subset, sensitivity suite, or headline experiment is authorized.
- The 48 CPU-hour cumulative canary ceiling and 600 CPU-hour sensitivity ceiling are recorded, but
  execution still requires separate approval and measured projections.
