# Phase 2 — 32-cell 4,000-transition diagnostic: final execution schedule

**Date:** 2026-07-18
**Status:** PLAN ONLY — **do not submit until explicitly approved.** No returns inspected.
**Run:** the already-frozen 32-cell / 4,000-transition ecological diagnostic, time-boxed 6–8 h.

Measured per-stage wall times (from the validated canary, `Elapsed`):

| stage | central | optimistic | upper (straggler) |
|---|---:|---:|---:|
| PLUS fit / cell | 3.9 h | 3.67 h | 5.0 h |
| MOOR fit / cell | 0.34 h | 0.33 h | 0.5 h |
| PLUS plan+eval / cell (safe) | 1.6 h | 1.42 h | 2.5 h |
| MOOR plan+eval / cell (safe) | 0.17 h | 0.16 h | 0.3 h |

Fit is **single-threaded** (canary `TotalCPU` 4:03:47 ≈ `Elapsed` 4:05:25 → 2nd allocated core idle).

## 1. Critical-path timing

All **64 fit jobs start together** (64 × 2 CPU = 128 CPU ≤ 256). Each **plan job starts the instant its
own fit succeeds** (element-wise `aftercorr`; fit[i]/plan[i] are verified aligned on (cell, method)).

- 32 MOOR fits finish at ~0.34 h → their plans finish by ~0.5 h.
- 32 PLUS fits finish at ~3.9 h (central) / ~5.0 h (upper straggler: regime + sink cells).
- The **critical path is the slowest PLUS (fit → plan)** chain:

| | slowest PLUS fit | + its plan | + overhead* | **wall-clock completion** |
|---|---:|---:|---:|---:|
| optimistic | 3.67 | 1.42 | 0.15 | **≈ 5.2 h** |
| central | 3.9 | 1.6 | 0.25 | **≈ 5.7 h** |
| upper | 5.0 | 2.5 | 0.3 | **≈ 7.7 h** |

\* overhead = scheduler queue latency (assumed prompt under quota) + per-task structural check (~1 min
between fit and plan) + final acceptance (~2 min).

**Fits inside the 6–8 h box in all but a severe-straggler case** (upper ≈ 7.7 h leaves little margin; a
busy queue could push it toward 8 h — see the deadline policy in §4).

**Concurrency profile (never exceeds 256 CPU):** peak is **128 CPU** at t=0 (64 fits). MOOR plans (≤64
CPU) overlap only the first ~0.5 h while 32 PLUS fits run (64 CPU) → ≤128. PLUS plans (64 CPU) run after
PLUS fits finish. 256 is never binding.

## 2. Pipelined Slurm design

Two aligned 64-task arrays + one acceptance job. `$RUN` = the frozen run root.

```bash
# 1) Fit array — 64 tasks (32 PLUS + 32 MOOR), reward-independent, write model to transition-hash cache.
FIT=$(sbatch --parsable --job-name=adapt32-fit \
      --array=0-63%100 --cpus-per-task=2 --mem=4G --time=08:00:00 --no-requeue \
      --export=ALL,ROOT=$RUN/code,MANIFEST=$RUN/manifests/diagnostic_fit_64.csv,STAGE=fit \
      $RUN/code/scripts/slurm/run_adapted_fit_row.sh)

# 2) Plan+eval array — 64 tasks; task i starts the moment fit task i SUCCEEDS (element-wise).
PLAN=$(sbatch --parsable --job-name=adapt32-plan \
      --array=0-63%100 --dependency=aftercorr:$FIT --cpus-per-task=2 --mem=4G --time=03:00:00 --no-requeue \
      --export=ALL,ROOT=$RUN/code,MANIFEST=$RUN/manifests/diagnostic_plan_safe_64.csv,STAGE=plan \
      $RUN/code/scripts/slurm/run_adapted_fit_row.sh)

# 3) Acceptance — runs after all plans finish (even if some failed), return-blind.
sbatch --job-name=adapt32-accept --dependency=afterany:$PLAN --cpus-per-task=1 --mem=8G --time=00:20:00 \
      --export=ALL,RUN=$RUN $RUN/code/scripts/run_paper_faithful_acceptance.py-wrapper
```

- **`aftercorr:$FIT`** is the exact "start each plan as soon as its fit passes" mechanism: plan task *i*
  depends on fit task *i* with `afterok` semantics. **Alignment is verified** (fit[i]/plan[i] share
  (cell, method)). It does **not** wait for all fits.
- **Structural gate between fit and plan:** the fit row exits **non-zero** if its own structural check
  fails (non-finite objective, privacy/consistency failure); `aftercorr`'s `afterok` semantics then
  **skip that cell's plan**, preserving the failed fit's diagnostics.
- **No rerun of successful fits:** `--no-requeue` (no auto-retry) + the transition-hash fit cache; the
  plan job **loads** the cached model and never refits.
- **≤256 CPU:** `%100` array concurrency is a guard; the natural peak is 128 CPU (§1).

## 3. Corrected CPU / core-hour accounting

The earlier "300 CPU-hour" ceiling was ambiguous. This cluster **bills allocated core-hours**
(`CPUTime = AllocCPUS × Elapsed`). Reported separately (central estimate):

| Measure | Definition | Central | Upper |
|---|---|---:|---:|
| **Aggregate elapsed job-hours** | Σ `Elapsed` over all 128 jobs | 32·3.9+32·0.34+32·1.6+32·0.17 = **192 h** | ~264 h |
| **Actual TotalCPU hours** | Σ `TotalCPU` (fit single-threaded ⇒ ≈ Elapsed) | **≈ 190 CPU-h** | ~262 |
| **Allocated / billed core-hours** | Σ `CPUTime` = 2 × Σ Elapsed | **≈ 385 core-h** | ~528 |
| **Max possible core-hours** | Σ (`--time` cap × AllocCPUS) = 64·8·2 + 64·3·2 | **1,408 core-h** | (cap) |

**Evaluation of the proposed 450 allocated-core-hour ceiling (warning 360):**
- Central billed ≈ **385** → under 450 ✓; the **360 warning fires** (385 > 360) — appropriate early alert.
- To reach 450, aggregate `Elapsed` must hit **225 h** (+17 % over central); the **straggler upper
  (~528)** would **breach 450**. So **450 is marginal — sufficient centrally, at risk on the tail.**
- The old 300 ceiling, read as allocated core-hours, is **insufficient** (central 385 > 300) — correctly replaced.

**Recommendation (not silently approved):** for a deadline run keeping the exact validated config
(**2 CPU/job**, no re-validation), set the ceiling to **500 allocated core-hours, warning at 400** — this
covers the straggler tail. If you prefer the proposed **450 / 360**, it holds centrally but must be
enforced by the §4 deadline-stop (which will leave late cells incomplete). *Efficiency note (future
runs, not this deadline run):* because the fit is single-threaded, dropping to **1 CPU/job**
(OMP_NUM_THREADS=1) **halves billing** to ~190 core-h at the same wall-clock, making 450 very comfortable
— but it needs a one-cell fit-hash re-validation vs the canary, so it is deferred out of the deadline path.

## 4. Deadline failure policy

- **No automatic retries** (`--no-requeue`) — a retry could breach the deadline or ceiling.
- **Preserve failed cells and their diagnostics** — never delete; a failed fit skips its plan but its
  `faithful_fit.json`/privacy/provenance artifacts remain for inspection.
- **At 6 elapsed hours:** report completion state (fits done, plans done, cells complete, current billed
  core-hours) **without reading any performance return**.
- **At the approved hard deadline** (recommend **8 h**): `scancel` all **pending** array tasks and place
  a hold so **no new job starts**; **already-running near-complete jobs run to their `--time`** under
  this policy (they are close to done). Do not launch replacements.
- **Ceiling enforcement:** monitor billed core-hours via `sacct`; at the **warning** threshold alert; at
  the **ceiling**, hold new submissions (same mechanism as the deadline hold).
- **Do not alter hyperparameters** (candidates, paths, starts, grids, tolerances) to make cells finish.

## 5. Scientific label

Registered name: **"Registered 32-cell hidden-demographics ecological-baseline diagnostic at the
inherited matched 4,000-transition budget."** Explicitly:

- **4,000 is retained for comparability** with the existing general-RL experiment (matched budget), not
  because it is shown adequate;
- **formal 4k-vs-8k adequacy remains pending** (the nested-budget study is deferred for the deadline);
- **poor performance cannot yet be attributed** solely to algorithm quality **or** to data scarcity;
- **safe-mode results are information-limited** under the private safety objective;
- **this is not the full 288-cell experiment.**

## 6. Return policy

The pipeline **may compute and save** returns (the plan/eval stage writes `summary.json`). **Do not
inspect, summarize, compare, rank, or tune against any return** (`operational_return`, `true_return`,
survival return, method comparison) **until structural + scientific acceptance (§5 of the preregistration
package, table criteria) is complete.** Acceptance itself is return-blind (`return_fields_opened=false`).

## Run manifest and hashes (frozen; unchanged)

| Artifact | Rows | sha256 |
|---|---:|---|
| `diagnostic_fit_64.csv` | 64 | `a8f39d83eb70a32be2c42fabecc56ca4af4d3d40edb97f87242934085fd7ee86` |
| `diagnostic_plan_safe_64.csv` | 64 | `f0322718a78d485827c39661e2f1ded4cbde1d8bccd4cd053f4d1fe93dec087d` |
| runtime code digest | — | `f70ec7122263ec00da0a4950994dfe483ed29c71bc15e477007ca9ae0a6d2333` |
| durable snapshot commit | — | `fd50c38c` (branch `snapshot/paper-faithful-plus-moor-20260718`) |

## GO / NO-GO

**GO — conditional on your explicit approval of two items:**
1. **Resource ceiling:** **500 allocated core-hours (warning 400)** recommended [or the proposed 450/360,
   accepting the straggler-tail truncation risk] — central billed ≈ 385, upper ≈ 528.
2. **Hard deadline = 8 h**, with the §4 stop policy (6 h status report; no new jobs after 8 h).

Everything else is ready and frozen: implementation verified, snapshot durable, manifests hashed,
critical path ~5.7 h central (within box), Slurm pipeline defined, returns quarantined until acceptance,
scientific label registered. On approval the three `sbatch` calls in §2 are submitted; **I will not
submit until you say so.**

**Nothing submitted; no returns inspected.**
