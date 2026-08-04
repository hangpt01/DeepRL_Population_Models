# Verification and diagnostic layer

## Maintainer gates

The `Makefile` defines:

| target | what it establishes |
|---|---|
| `verify-self-tests` | 36 synthetic assertions for M1–M15 metric functions |
| `verify-constants` | registered vulture reachability bound and dominated-action identities |
| `verify-integrity` | frozen/scientific/copied hash ledgers and accepted CSV hash |
| `verify-negative-gate` | a corrupt parity case exits nonzero |
| `verify-cell` | one ecological accepted replay, seven-field parity, cache reuse |
| `verify-cell-general` | one accepted general Tier-B replay |
| `verify-standalone-s2` | repository-routed S2 planning-only execution |
| `test-ecological/general` | track-isolated unit suites |

Run integrity, both test suites, the negative gate, and at least one parity cell
before trusting a number.

## M1–M15

`src/diagnostics/replay_analysis/metrics.py` returns structured
`{"available":False,"reason":...}` when required logged fields are absent.

| metric | hypothesis |
|---|---|
| M1 posterior movement | does PLUS’s candidate belief leave its prior? |
| M2 switch fractions | does posterior weighting change actions versus MAP/uniform? |
| M3 candidate agreement | do candidates prescribe different actions? |
| M4 margins | are selected actions robust or near ties? |
| M5 raw vs centred | how much candidate disagreement is action-independent? |
| M6 reward decomposition | does utility-cost-penalty reconstruct accepted return? |
| M7 family degeneracy | how often is `r_pos=0`, erasing family terms? |
| M8 depensation | are Allee thresholds actually visited? |
| M9 regime | are weak regimes/switches visited and action-relevant? |
| M10 noise sensitivity | do matched sigma cells change behavior? |
| M11 EVD | does disagreement penalty alter recovery action rank/choice? |
| M12 OGSRL | does constraint/guardian bind, override, or fallback? |
| M13 surrogate error | error on visited safe/unsafe strata |
| M14 observation-model status | which method uses sigma/beliefs/surrogate? |
| M15 state discrimination | deployed versus myopic true-state oracle and action margins |

`constants.py` is a registered audit-derived table, not the runtime scientific
input. It documents an inert Egyptian-vulture `r_lgm` copy discrepancy because
current callers do not select that analysis column. `schema_report.csv` records
which replay fields make each metric computable.

## Replay and follow-ups

`scripts/diagnostics/replay/run_diagnostic_replay.py` instruments accepted
ecological policies, checks dataset hash/cache reuse, reconstructs seven fields,
and enforces parity. `run_tier_b_replay.py` does the general arm.
`constant_action_sweep.py` and `a0_baseline.py` supply controls.

Follow-ups:

- S2: cross-family oracle transfer regret, identifiability and VPI;
- H12: dose-response to surrogate degradation/scaling;
- H14: stratified surrogate error with matched RNG state;
- S6: new-fit sensitivity/probe (receipt records recomputed fits);
- reward screen: constant-action reward/feasibility identities.

Independent review files are claims, but current code confirms major fixes:
`scripts/verify_integrity.py` now enforces coverage; parity enforcement is
nonzero via `enforce_acceptance()`; `run_s2.py` now enumerates all 11 common
first actions with observation-consistent continuation and computes regret as
own-family minus transferred policy. This fixes the older five-candidate VPI
baseline and sign inversion described in `S2_ORACLE_REVIEW.md`. Remaining review
concerns include external data dependence and statistical scope.

The real test suites cover privacy, equation identities, PBVI, faithful
artifacts, native baselines, canary receipts, and real ecology. Synthetic/dummy
tests heavily cover shapes, schemas, calibration and gate mechanics. Passing
tests proves encoded identities and regressions, not ecological external
validity.

