# Response to the Audit of the Corrected PLUS/MOOR Plan

Date: 2026-07-17

## Verdict

All four audit findings are reasonable. None is rejected on substance.

## Changes folded into the plan

1. The five-arm, 32-cell PLUS sensitivity suite now has an explicit `F/K/P/E` projection and a
   separate hard requested ceiling of 600 CPU-hours, including the fixed 1.5 contingency.
2. Candidate banks are nested: the 12- and 16-candidate arms are immutable prefixes of the one
   32-candidate fit, and regime-path arms refit only four regime candidates. This reduces the fit
   sizing model from `5.875*F16` to `2.625*F16`, while preserving five separate reward-specific
   planning arms.
3. The void runtime implies more than 250 CPU-hours of suite fitting, or more than 375 CPU-hours with
   contingency, before planning. The corrected canary must replace this estimate before submission.
4. Signed effective rates are pinned to `r_min=-1.5`, `r_max=2.0`, with exact transform
   `r=0.25+1.75*tanh(q)` and the already registered symmetric Clarke treatment at zero.
5. Primary projections now report both conservative reward-specific kernel construction and a
   kernel-reuse case. Reuse is permitted only after the canary separately times kernel creation and
   proves safe/yield transition/observation kernel hashes identical and reward-free.
6. The plan states that no inferential parameter claims will be made, so profile/bootstrap intervals
   remain optional and omitted from required execution.
7. The exact multi-directory digest shell recipe is published in the plan.

## Point not executed now

The suggestion to extend `hash_paper_faithful_snapshot.py` to accept multiple roots is technically
reasonable, but modifying that script now would violate the controlling plan-only gate. The plan
schedules repeated-root verifier support as an implementation task after approval. Publishing the
exact current recipe closes reproducibility of the existing digest without changing source.

No scientific decision was reopened. No code, configuration, dependency, snapshot, void-run artifact,
or job was changed by this response.
