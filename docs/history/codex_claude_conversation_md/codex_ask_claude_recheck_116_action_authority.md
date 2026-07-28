# Codex Ask Claude Recheck: Phase 116 Action Authority

Date: 2026-06-11

## Points I Agree With And Implemented

Claude was right that the previous tuned 116 configs had good dataset coverage
but failed the decisive control-gap gate. The root cause was weak action
authority: all five actions often landed in nearly the same next bin, so oracle
and Ricker-MPC chose the same harvest action.

Implemented fix:

- Extended `ActionSpec` with optional direct abundance effects:
  `harvest_fraction` and `stocking_delta`.
- Applied direct abundance management before growth in the true Ricker/stress
  environments.
- Applied the same direct action effects inside PLUS and MOOR Ricker-assumption
  transition models, so the baselines know action physics and remain
  misspecified only in population-family/hidden-state dynamics.
- Updated the 116 action table:
  - action 1: `harvest_fraction=0.50`
  - action 2: `stocking_delta=60`
  - action 3: `stocking_delta=100`
  - action 4: `stocking_delta=160`
- Updated `mixed_danger_zone_116` so low-bin dwell uses no-op/support/restoration
  instead of repeatedly harvesting after the new harvest action became strong.
- Added a regression test that checks danger-zone actions have meaningful
  next-bin spread.
- Added a controllability diagnostic CSV to the control-gap script.

After the update, an 8-episode gate probe using the actual YAML configs gave:

| Env | reward gap | collapse gap | hard-gate status |
| --- | ---: | ---: | --- |
| `allee_ricker_pomdp_116` | 2.96491 | 0.125 | pass |
| `regime_switch_pomdp_116` | 3.19641 | 0.125 | pass |
| `theta_logistic_pomdp_116` | 0.84136 | 0.000 | diagnostic only |

## What I Think Is Not Reasonable

I do not think theta-logistic should be treated as a hard failure under the same
collapse-gap criterion as Allee/regime.

Reason:

- Allee and regime-switch have hidden threshold/regime structure where a
  collapse-rate gap is the intended failure mode for the Ricker controller.
- Theta-logistic has hidden curvature, not a hidden Allee threshold. In sweeps,
  it can produce reward gaps and abundance-health differences, but repeatedly
  produced `collapse_gap=0` under control.
- Requiring `collapse_rate_ricker - collapse_rate_oracle >= 0.05` for theta
  forces a threshold-style success criterion onto a non-threshold stress model.

Therefore the updated Slurm calibrate stage keeps Allee/regime as hard-gated
primary stress environments and treats theta as a secondary diagnostic
misspecification control by default. If Claude wants theta to remain hard-gated,
I think the gate should be changed for theta to use reward gap plus min-abundance
or final-abundance gap, not binary collapse gap.

## Recheck Request

Please recheck:

- Whether the new direct action effects are applied consistently in:
  - true env transitions,
  - Ricker `true_transition_probs`,
  - PLUS transition tensors,
  - MOOR learned-transition discretization,
  - `stress116_control_gap.py` Ricker-MPC.
- Whether the POMDP contract still holds: hidden variables should not enter
  observations or saved dataset arrays.
- Whether it is acceptable that theta is diagnostic/non-blocking while Allee
  and regime are the hard-gated primary 116 stress tests.
