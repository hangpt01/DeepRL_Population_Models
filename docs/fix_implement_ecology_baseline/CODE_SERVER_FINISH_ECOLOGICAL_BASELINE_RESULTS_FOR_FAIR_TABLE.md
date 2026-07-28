# Code-Server Guide: Finish PLUS/MOOR Results for a Fair General-RL Comparison

## Purpose

This guide is for the AI agent working in the code-server repository. The objective is no longer a
fresh paper-faithfulness audit. The implementations have already been documented and tested. The
objective is to finish the missing ecological-baseline result rows, validate exact comparability,
and produce a durable fair comparison package against the completed general-RL experiment.

Do not infer current scheduler or artifact state solely from this document. Verify it first.

## Copy-paste prompt for the code-server agent

```text
Continue the paper-aligned ecological-baseline result work with the specific goal of producing a
fair comparison table against the completed Phase-2E general-RL package.

This is not a new implementation audit. First read, in order:
1. MATCHED_GENERAL_ECOLOGICAL_RESULTS.md
2. PLUS_MOOR_RICKER_STRESS_TEST_SUPERVISOR_EXPLANATION.tex
3. GENERAL_RL_PAPER_IDEAS_AND_ECOLOGY_ADAPTATIONS.tex
4. LAUNCH_PLAN.md
5. ACCEPTANCE_V2_FROZEN_RUNTIME_OBSERVABLE_20260719.md
6. PHASE2_DATA_BUDGET_AND_HYPERPARAM_ADEQUACY_PLAN.md

Treat MATCHED_GENERAL_ECOLOGICAL_RESULTS.md as the latest outcome/status record. The two TeX files
explain methods but do not contain the latest ecological outcomes and may contain stale operational
status. Do not rerun completed rows merely because the TeX files omit their results.

First inspect and report only. Verify current squeue/sacct, repository commit, runtime/configuration
hashes, manifests, artifact roots, acceptance records, and exact dataset identities. Do not submit
jobs or modify frozen artifacts until I approve a completion scope.

Reconstruct and confirm this expected state:
- General RL: 576/576 rows complete and structurally accepted, covering 9 populations × 4 families
  × sigma {0.1,0.2} × rewards {safe,yield} × 4 methods.
- Ricker-only PLUS: safe mode only, 71/72 cells complete. Index 51 (Iberian lynx / Allee /
  sigma=0.2) timed out. Two Crab-eating-fox/Theta cells have nonmatching dataset hashes and are
  excluded from strict paired comparison.
- Corrected MOOR: 32/32 fits and 32/32 plan/evaluation rows complete in the old diagnostic. Only 16
  rows overlap the general domain: Amur tiger and Egyptian vulture × 4 families × sigma {0.1,0.2}
  × safe. The per-row structural checks passed, but the obsolete combined PLUS/MOOR experiment gate
  remains INCOMPLETE because it was coupled to the failed cross-family PLUS arm and had Slurm-ID and
  wrapper-import problems.
- Ecological yield-mode results are absent.

Report any difference from this state before planning new work.

Prepare four explicit completion options:

A. Three-species safe-only pilot.
B. Three-species safe+yield pilot.
C. Full nine-species safe-only table.
D. Full nine-species safe+yield table.

For each option report exact new fit, plan, evaluation, acceptance, CPU-hour, wall-time, memory, and
artifact counts. Distinguish reusable reward-independent fits from reward-dependent planners,
actors/Q-functions, and evaluations.

Do not choose the three species silently. If the already discussed panel is used, it is Amur tiger,
Crab-eating fox, and Egyptian vulture. Explain that this panel represents recoverable decline,
strong recoverable growth, and a severe demographic sink. Verify the corrected complete-episode
Crab-eating-fox datasets before using them.

The minimum completion arithmetic expected from the current record is:
- Full safe table: rerun/repair 3 PLUS rows (one timeout plus two exact-data alignment rows) and run
  56 missing MOOR safe rows for the seven populations absent from the MOOR diagnostic.
- Full safe+yield table: complete the safe work above, then add 72 PLUS yield plan/evaluation rows and
  72 MOOR yield plan/evaluation rows, reusing reward-independent fits where identities match.
- Three-species safe table for Amur/Crab fox/Egyptian: rerun the two Crab-fox/Theta PLUS rows on the
  authoritative general dataset and run 8 missing Crab-fox MOOR safe rows. The existing Amur and
  Egyptian MOOR rows may be reused after method-specific acceptance.
- Three-species safe+yield table: add 24 PLUS yield and 24 MOOR yield rows, reusing exact fits. This
  is 58 new ecological evaluations in total if the two PLUS alignment rows and eight missing MOOR
  safe rows are included.

Recompute these counts from the actual repository and manifests; do not trust arithmetic without
checking row identities.

Use the authoritative general dataset registry as the pairing target. A formal paired comparison
requires exact equality of population, family, sigma, reward, public dataset hash, evaluation seeds,
horizon, discount, and environment randomness. Never call design-matched but hash-different cells
exactly paired.

Do not use plus_native, moor_native, hidden_rk_comparison_20260716, the cancelled cross-family PLUS
plans, or old mixed-trajectory artifacts as substitutes.

Correct the acceptance workflow for future MOOR rows without rewriting history:
- create a MOOR-specific acceptance decision rather than coupling it to cross-family PLUS;
- support the cluster's actual raw array-element job identifiers;
- set the required script import path in the wrapper;
- run acceptance with afterany so failed/missing rows produce INCOMPLETE;
- retain the old combined experiment-level INCOMPLETE decision unchanged; and
- record return_fields_opened=false before any new outcome access.

Before expanding MOOR, add launch/runtime-compatible diagnostic logging that does not change policy
behavior:
- per-timestep action IDs and observations for selected diagnostic cells;
- PBVI action values, argmax margins, and legal-action checks;
- action entropy and constant-policy flag;
- fit/kernel/filter/planner hashes;
- explicit completion receipts; and
- separate any policy exception, invalid-action, guardian-infeasibility, and intentional-fallback
  counters where the shared evaluator supports them.

If logging changes runtime source, create a new version/digest and prove behavioral parity on a
small completed overlap before mixing it with old rows. Prefer a launch/reporting overlay when
possible. Do not change the MOOR or PLUS scientific algorithms merely to avoid constant policies.
First complete and report the frozen baseline. Any planner/hyperparameter improvement is a separate
named sensitivity.

The fair result table must keep methods separate:
- RefPlan-inspired;
- OGSRL-inspired actor + guardian + fallback composite;
- BA-MCTS-inspired;
- value-disagreement pessimism, Delphic-motivated only;
- Ricker-only adapted PLUS; and
- corrected Ricker MOOR.

Do not report a cellwise Best general envelope as one method. Keep safe and yield in separate panels.
Keep recoverable and sink populations separate and also report individual species. For each method
report signed operational return, collapse entry, unsafe fraction, MVP breach/fraction, persistence,
minimum/mean/final abundance, economic cost, action entropy, and fallback/feasibility counts where
applicable. Pair method gaps by cell and evaluation seed, and use the method-cell—not individual
episodes—as the default unit for across-cell uncertainty.

Preserve these interpretation warnings:
- OGSRL results are for the deployed actor/guardian/fallback composite, not an actor-only policy.
- Corrected MOOR is nearly constant in the existing diagnostic (31/32 deterministic cells).
- Corrected MOOR and Ricker-only PLUS are exactly tied on their current 16-cell overlap because both
  choose the same constant actions; this is policy degeneracy, not algorithmic equivalence.
- Sink collapse_entry=0 does not imply safety when the population begins below the threshold.
- The 4,000-transition budget remains data-adequacy-unverified.

For every completion option, state what it would make scientifically claimable and what would remain
unsupported. Recommend the smallest option that produces a meaningful fair table, but do not submit
it. End with exact manifests to create, rows to reuse, rows to rerun, scheduler commands/resources,
acceptance steps, and a GO/NO-GO recommendation. Wait for approval before changing code or launching.
```

## Authoritative current result record

The local file that already contains the available ecological results is:

- `MATCHED_GENERAL_ECOLOGICAL_RESULTS.md`

It should be read before the supervisor-facing TeX files. The TeX documents explain the algorithms
and adaptations but currently do not carry the completed result tables.

## What is already available

### General RL

- 576/576 method rows complete and accepted.
- Nine populations, four families, two noise levels, two reward modes, four methods.
- Both safe and yield results.
- Full episode-level result metrics.
- General-only per-timestep trajectories for Amur tiger, Puerto Rican parrot, and Egyptian vulture.

### Ricker-only PLUS

- Safe mode only.
- 71/72 cells completed.
- One timed-out cell: Iberian lynx / Allee / `sigma_obs=0.2`.
- Two Crab-eating-fox/Theta cells are not exact-data matches to the general registry.
- Strict matched comparison currently uses 69 cells.
- Results are usable with incompleteness and exact-data warnings.

### Corrected MOOR

- 32/32 fit and 32/32 plan/evaluation tasks completed in the two-population diagnostic.
- Sixteen safe cells match the general domain exactly.
- Scoped outcomes have already been opened under explicit authorization.
- Existing behavior is nearly constant and does not discriminate MOOR from PLUS on the overlap.
- No seven-population extension and no yield arm currently exist.

## What actually needs to run

### Minimum meaningful full-scope safe comparison

To compare all six methods over the same 72 safe cells:

1. rerun the missing Iberian-lynx PLUS row;
2. rerun the two Crab-eating-fox/Theta PLUS rows on the authoritative general datasets;
3. run corrected MOOR for the remaining seven populations across four families and two noise levels
   (`7 × 4 × 2 = 56` rows); and
4. apply separate method-specific return-blind acceptance before opening new outcomes.

This produces a balanced nine-population safe-mode table but still does not support ecological yield
comparisons.

### Full two-reward comparison

After completing safe mode:

- add 72 Ricker-only PLUS yield plan/evaluation rows; and
- add 72 corrected-MOOR yield plan/evaluation rows.

Reuse exact reward-independent fits. Do not assume that reward-dependent policies or planners may be
reused.

### Efficient three-species checkpoint

For Amur tiger, Crab-eating fox, and Egyptian vulture:

- safe only: two PLUS exact-data alignment reruns plus eight new Crab-fox MOOR rows;
- safe+yield: the safe work plus 24 PLUS yield and 24 MOOR yield rows.

This checkpoint is much cheaper and provides a balanced hard-recoverable/easy-recoverable/sink table
before spending compute on all nine species.

## Table structure required for a meaningful comparison

The final table should not be one pooled mean. At minimum provide:

1. recoverable safe panel;
2. sink safe panel;
3. recoverable yield panel;
4. sink yield panel;
5. family-specific paired return gaps;
6. noise-specific paired return gaps;
7. individual-species results; and
8. action-degeneracy/fallback diagnostics beside performance.

Recommended columns:

| Method | Cells | Return | Collapse entry | Unsafe | MVP breach | Persistence | Min N | Final N | Cost | Action entropy | Fallback/constant-policy note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

Use signed returns. Report uncertainty from paired cell-level differences. Never let pooled sink
returns obscure recoverable results.

## Completion principle

Finish the existing frozen methods before introducing richer models, larger data budgets, or tuned
planner settings. Once the fair frozen table exists, run the three-species adequacy and sensitivity
plan in `THREE_SPECIES_PERFORMANCE_AND_ADEQUACY_PLAN.md` as a separately versioned next stage.

