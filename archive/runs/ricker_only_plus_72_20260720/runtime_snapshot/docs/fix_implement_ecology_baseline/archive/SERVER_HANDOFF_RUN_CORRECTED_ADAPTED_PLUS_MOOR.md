# Server handoff: run the corrected adapted PLUS/MOOR experiment

**Status:** implementation complete; corrected canary and larger experiments have not run.

**Next-chat objective:** independently audit the corrected implementation, finish the small remaining
canary-freeze tooling, then run only the registered return-blind corrected canary under its staged CPU
ceilings. Do not submit the diagnostic subset, sensitivity suite, or headline sweep without later
explicit approval.

## Read first, in order

1. `docs/fix_implement_ecology_baseline/CORRECTED_PLUS_MOOR_IMPLEMENTATION_PLAN.md`
2. `docs/fix_implement_ecology_baseline/CODEX_IMPLEMENTATION_HANDOFF_CORRECTED_PLUS_MOOR.md`
3. `docs/fix_implement_ecology_baseline/CLAUDE_AUDIT_CORRECTED_PLUS_MOOR_IMPLEMENTATION.md`
4. `docs/fix_implement_ecology_baseline/CLAUDE_AUDIT_CORRECTED_PLUS_MOOR_PLAN.md`
5. `docs/fix_implement_ecology_baseline/CLAUDE_DIRECTIVE_FIX_BEFORE_RESUBMIT_PLUS_MOOR.md`
6. `docs/fix_implement_ecology_baseline/DECISIONS_FOR_CORRECTED_PLUS_MOOR_PLAN.md`
7. This handoff.

Treat the corrected implementation plan and recorded PI decisions as authoritative. Audit comments are
evidence and review context, not permission to reopen settled choices silently.

## Current verified state

- Adopted method IDs:
  - `plus_adapted_mechanistic_pbvi`
  - `moor_adapted_ricker_misspec_pbvi`
- Superseded IDs are not runnable:
  - `plus_faithful_pbvi`
  - `moor_faithful_ricker_misspec_pbvi`
- Full test suite: **154 passed**.
- Focused Ruff check passed; repository-wide legacy `E402` and `E731` findings were excluded.
- Current repeated-root digest over `src/`, `configs/`, and `scripts/`:
  `8b4c7c44f23b1622e70b580b77cfec9f22e49abb3596690e54010d73df9e56db`.
- No Slurm jobs were running when this handoff was written.
- No corrected snapshot or canary exists yet.
- The preserved void run was not modified.

Reverify rather than trusting these values:

```bash
PYTHONPATH=src .venv-paper-faithful/bin/python -m pytest -q
python scripts/hash_paper_faithful_snapshot.py src configs scripts \
  --expected 8b4c7c44f23b1622e70b580b77cfec9f22e49abb3596690e54010d73df9e56db
squeue -u hphung
```

If source/config/script files legitimately change while finishing canary tooling, record both the old
and new digests. Never pretend the old digest still identifies the modified tree.

### Post-audit digest and minor-finding adjudication

Claude's implementation audit reproduced digest `6456fdf...` before the canonical docs checker was
updated to recognize the two adopted method IDs. The only later `src/configs/scripts` change was the
narrow `scripts/check_docs.py` registry check; `make docs-check` now passes. The current digest is
therefore `8b4c7c...`, as recorded above. Keep both values in provenance rather than rewriting the
audit's as-observed result.

Claude's M1 claim is not supported by the authoritative plan: Section 3 already specifies exactly
`r_a = 0.25 + 1.75*tanh(q_a)`, the transform implemented in code. No plan correction is needed.

Claude's M2 observation is accurate as metadata but is not an implementation defect: the 32-cell
diagnostic subset intentionally has no approved CPU ceiling yet. The canary must provide measured
projections, after which the PI must supply `B_full`/the subset ceiling. This remains a blocking future
decision; do not invent a budget. Claude's M3 reminder is correct: corrected runtime is unmeasured.

## Scientific contract that must remain fixed

- These are adapted mechanistic baselines, not exact paper reproductions.
- Hidden methods receive categorical public action channels only, not private action-table values.
- Action model: eight signed rates, five capacity effects, one stocking effect.
- Signed-rate bounds are `[-1.5, 2.0]`; exact max split uses the symmetric Clarke convention at zero.
- Survey fitting uses `x * exp(sigma_o^2/2)` and no sampled survey-noise draw.
- Regime persistence candidates are fixed at `p in {0.80, 0.90, 0.97}`.
- Abundance advances under current regime `z_t`, then Pi samples `z_{t+1}`.
- Pi is immutable and never optimized.
- Primary regularization is `none_structural_v1`.
- `hierarchical_weak_v1` is a separate diagnostic only and cannot replace a failed primary result.
- PLUS primary bank is 16 candidates with allocation `4/4/4/(1,2,1)`.
- Primary regime process-path count is 16.
- Fit identity excludes reward, action cost, evaluator fields, private sidecars, and full dataset hash.
- Fit cache reuse across safe/yield is required; reward-specific planners are rebuilt separately.

## Important preserved artifacts

Do not edit, delete, reuse, or relabel these frozen trees:

- `real_ecology_runs/paper_faithful_one_cell_4000_20260717_v1/`
  - scientifically void after the MOOR survey-target and PLUS regime-law defects;
  - retained for runtime/provenance inspection only.
- `real_ecology_runs/paper_faithful_smoke_20260717_v4/`
  - provisional interface smoke only;
  - one candidate per form and not evidence for corrected 16-candidate PLUS.
- `real_ecology_runs/hidden_rk_comparison_20260716/`
  - separate completed hidden-r/K benchmark result;
  - its native methods are ecology-inspired approximations, not the corrected methods here.

Use a completely new run root for corrected work.

## Implementation surfaces to audit

- `src/real_ecology_benchmark/faithful_ecology.py`
- `src/real_ecology_benchmark/faithful_fit.py`
- `src/real_ecology_benchmark/faithful_pomdp.py`
- `src/real_ecology_benchmark/faithful_artifacts.py`
- `src/real_ecology_benchmark/methods/plus_faithful.py`
- `src/real_ecology_benchmark/methods/moor_faithful.py`
- `src/real_ecology_benchmark/methods/__init__.py`
- `src/real_ecology_benchmark/config.py`
- `src/real_ecology_benchmark/realdata.py`
- `src/real_ecology_benchmark/privacy.py`
- `src/real_ecology_benchmark/pipeline.py`
- `scripts/make_paper_faithful_manifest.py`
- `scripts/run_adapted_fit_row.py`
- `scripts/run_real_manifest_row.py`
- `scripts/hash_paper_faithful_snapshot.py`
- `tests/real/test_corrected_adapted_mechanistic.py`

## Remaining pre-execution tooling task

Do not submit the current `canary` manifest directly. The generator currently emits eight combined
plan/evaluation rows, while the corrected contract calls for a return-blind fit stage followed by
reward-specific planning/evaluation.

Before freezing a run, add and test two modes:

1. `canary_fit`
   - two dynamics cells:
     - Amur tiger / Ricker / `sigma_o=0.1`;
     - Egyptian vulture / regime / `sigma_o=0.4`;
   - two methods per cell;
   - four fit-only rows total;
   - use `scripts/run_adapted_fit_row.py`;
   - no planner, evaluator, episode return, or action-quality field may be opened.
2. `canary_plan`
   - the same two cells and two methods;
   - safe and yield reward modes;
   - eight rows total;
   - each row must reference the immutable fit-cache key/model hashes produced by `canary_fit`;
   - every fit must be a cache hit; a miss is a blocking failure.

Extend `scripts/prepare_paper_faithful_smoke.py` or add a clearly named corrected preparer that:

- freezes `src/`, `configs/`, `scripts/`, `real_ecology_data/`, `pyproject.toml`, and the dependency lock;
- generates both canary manifests;
- records their SHA-256 hashes and the repeated-root source digest;
- records current method IDs and implementation versions;
- records `headline_sweep_authorized: false`;
- records first-stage ceiling 24 CPU-hours and cumulative ceiling 48 CPU-hours;
- refuses to overwrite an existing run root;
- never reads returns while preparing or accepting the canary.

Add tests for exact row counts, cell keys, reward exclusion from fit rows, reward inclusion in plan rows,
method IDs, cache-key references, and overwrite refusal. Return the tooling diff and tests for review
before submitting the canary if anything scientifically material changes.

## Corrected canary sequence

After the preparer and manifests pass audit, use a new root such as:

```text
real_ecology_runs/adapted_plus_moor_corrected_canary_20260717_v1/
```

Run in this order:

1. Freeze the corrected snapshot and manifests.
2. Run only the Amur-tiger fit rows.
3. Verify cache artifacts, finite objectives, model hashes, privacy audit, runtime, memory, and row count.
4. Run Amur safe/yield planning rows; require all dynamics fits to be cache hits.
5. Compute actual CPU usage. Stop if the first-cell total exceeds 24 CPU-hours.
6. Only if step 5 passes, run Egyptian-vulture fit rows and then its safe/yield planning rows.
7. Stop if cumulative corrected-canary usage exceeds 48 CPU-hours.
8. Run return-blind acceptance and runtime projection.

The acceptance review may open only:

- fit/cache status and immutable keys;
- convergence and holdout diagnostics;
- parameter/model/Pi/law hashes;
- candidate allocation and diversity diagnostics;
- planner invocation counts and hashes;
- fit, kernel, PBVI, evaluation timing;
- peak RSS, task state, exit code, and artifact completeness.

Do not inspect or summarize returns, chosen actions, unsafe fractions, method wins, or policy quality
until the diagnostic scope and any subsequent run are separately authorized.

## Submission discipline

- Use the frozen snapshot, never the mutable live tree.
- First run `sbatch --test-only`.
- Submit fit and plan stages with explicit dependencies; plan rows must not start before fit acceptance.
- Preserve stdout, stderr, Slurm job IDs, `sacct` output, manifests, registration, dependency lock, and
  all cache/artifact hashes.
- Do not delete failed or canceled runs. Mark scientifically unusable runs with `VOID.md`.
- Cancellation preserves artifacts; it never licenses cleanup.
- Do not submit all diagnostic or sensitivity rows to “save time.”

## CPU projection gate after the canary

Measure separately for each method:

- `F_m`: fit plus immutable model serialization;
- `K_m`: transition/observation kernel construction;
- `P_mr`: reward-specific PBVI;
- `E_mr`: evaluation.

Report conservative and kernel-reuse projections exactly as preregistered in Section 10 of the plan,
with the fixed 1.5 contingency. Do not propose the 32-cell diagnostic subset until its measured
projection is available. Do not propose the five-arm sensitivity suite unless
`1.5 * C_suite <= 600 CPU-hours`. No headline sweep has a current authorization or budget.

## Required next-chat deliverable

At the end of the next chat, write an audit/run handoff in this folder containing:

- files changed for canary tooling;
- test and digest evidence;
- frozen run root and hashes;
- exact Slurm job IDs and dependency graph, if submitted;
- per-stage state, exit code, elapsed time, CPU time, and peak RSS;
- fit/cache hit evidence across reward modes;
- artifact completeness and privacy evidence;
- ceiling accounting;
- a clear statement that returns were or were not opened;
- blockers and the exact next authorized action.

Stop after the corrected canary and its return-blind runtime audit. Do not continue automatically into
the diagnostic subset, sensitivity suite, or headline experiment.
