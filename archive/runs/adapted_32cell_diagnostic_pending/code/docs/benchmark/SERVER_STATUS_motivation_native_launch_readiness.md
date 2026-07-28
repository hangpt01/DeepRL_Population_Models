# Launch Readiness — Motivation Experiment (native ecological baselines)

**Date:** 2026-07-11
**Run package:** `real_ecology_runs/motivation_native_20260711/`
**Base commit:** `e74526e`
**Verdict:** **Launch package is READY. One gate is still open — do not submit the sweep yet.**

This is the single-page status for the motivation experiment. The detailed reasoning lives in:
`SERVER_HANDOFF_motivation_experiment.md` (the task) · `SERVER_CONTEXT_motivation_native_baselines_gate.md` (gate, code paths, corrections) · `SERVER_AUDIT_motivation_native_baselines_gate.md` (audit) · `SERVER_PLAN_motivation_native_run.md` (the run plan).

---

## 1. Gate status

| # | Gate | State |
|---|---|---|
| 1 | `make test` | **105 tests OK** |
| 2 | `make docs-check` | **PASS** |
| 3 | Frozen run package + provenance | **done** (`code/`, `PROVENANCE.txt`) |
| 4 | Manifests: 1440 main / 576 natives-only | **verified** |
| 5 | Cell-block alignment (BS=5 main, BS=2 native) | **verified — 0 bad blocks** |
| 6 | Row runner driven end-to-end through the frozen package | **verified** (row 164) |
| 7 | Slurm canary accepted (`sbatch --test-only`) | **verified** |
| 8 | **§7 acceptance battery — all 8 checks PASS** | **OPEN — running at time of writing** |

**Gate 8 is the only blocker.** The battery runs 288 rows and takes ~2 h (the adapted-PLUS rows in phase 2 dominate at ~104 s each). It had 0 failures at every check so far, but a row count written into this document goes stale within minutes, so **do not trust a number here — query the live state**:

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models
# progress and failures
echo "rows: $(grep -c '^  ok' outputs/acceptance_battery.log)/288  failures: $(grep -c '^  FAIL' outputs/acceptance_battery.log)"
# THE gate: the report must exist and every check must be true
python -c "import json;r=json.load(open('outputs/acceptance_20260711/acceptance_report.json'));\
[print(('PASS' if c['pass'] else 'FAIL'), k) for k,c in r['checks'].items()];\
print('LAUNCHABLE' if all(c['pass'] for c in r['checks'].values()) else 'BLOCKED')"
```

The battery exits non-zero if any check fails, so a completed run that wrote `acceptance_report.json` with an exit code of 0 is itself the green light.

> **Do not run `submit_blocks.sh` until `acceptance_report.json` exists and every one of the 8 checks reports PASS.**

---

## 2. What the acceptance battery checks

Slice: 4 populations × 4 families × σ ∈ {0, 0.4} × both reward modes, at the audited eval protocol (5 seeds × 4 episodes × horizon 50). The populations are chosen to exercise every axis:

| Population | Why it is in the slice |
|---|---|
| Amur tiger | recoverable; the Ricker cell where the native assumption is *correct* |
| Iberian lynx | recoverable; the four mechanistic forms disagree most (245/385 states) |
| Egyptian vulture | demographic **sink** — *no* action has `r_setpoint > 0` |
| Spotted turtle | the one population where all four forms agree everywhere |

| Check | Method |
|---|---|
| A1 gate resolved | build-required; native solver added, evidence in the context doc |
| A2 native not crippled | native vs adapted `filter=ricker` on every Ricker cell, both PLUS and MOOR |
| A3 same offline dataset | `dataset_sha256` compared across all methods within each cell |
| A4 both reward modes | safe P=5 / yield P=0; asserts `effective_collapse_penalty` and disjoint output dirs |
| A5 metrics schema | native summary metric keys ≡ general's; per-σ/family/recoverable fields in `episodes.csv` |
| A6 resolution | natives at 31 vs 91 bins; asserts the **sign** of (general − native) does not flip |
| A7 reward leakage | **runtime proof**: poison `dataset.rewards` (×−137 + 991), assert fitted actions are bit-identical |
| A8 zero failures | any raised row fails; **any `fallback_count > 0` fails** |

A7 and A8 are the two that earn their keep. A7 catches leakage through the *offline* channel, which the structural `PublicTransition` guard does not cover. A8 matters because the evaluator swallows `ValueError`/`RuntimeError` from `policy.act` into "do nothing" ([evaluator.py:100](../../src/real_ecology_benchmark/evaluator.py#L100)) — a broken solver can otherwise masquerade as a healthy conservative policy.

---

## 3. Launch sequence (after gate 8 goes green)

```bash
RUN=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/motivation_native_20260711

# 1. CANARY -- array task 32, NOT 0.  Task 0 is Egyptian vulture (the sink), the
#    worst possible canary.  Task 32 = rows 160-164 = Amur tiger/ricker/sigma=0/safe.
sbatch --job-name=motiv-canary --array=32 \
  --export=ALL,ROOT=$RUN/code,CONFIG=$RUN/code/configs/motivation_native.yaml,OUTPUT_ROOT=$RUN/_canaries/canary,BS=5 \
  $RUN/run_block.sh $RUN/manifests/manifest.csv

# 2. VERIFY the canary: 5 summary.json files, every fallback_count == 0, a single
#    shared dataset_sha256, non-degenerate action histograms.
#    (The earlier row-164 dry-run only proved the NATIVE path; the canary is the
#     first thing that drives refplan/bamcts/ogsrl through the frozen package.)

# 3. RELEASE the sweep
$RUN/submit_blocks.sh
```

`submit_blocks.sh` submits: main grid (288 tasks, `BS=5`, `%240`), two native resolution replicates (288 tasks each, `BS=2`), and the analysis job on an `afterany` dependency. **865 tasks total — under the `MaxSubmit=1000` cap.**

---

## 4. Non-negotiable guardrails

Each of these is a bug already hit at least once:

- **Never request a GPU.** `backend.py` exposes a CuPy kernel only for `mechanistic_transition`, `method:plus`, and `oracle_ablation:plus`. All five methods in this run are NumPy-only, so a GPU allocation buys **zero speedup** and parks the sweep behind GPU fair-share — the stall that cost us the stress-116 run.
- **`MaxSubmit=1000`** → rows must be packed; 1440 one-row jobs would be rejected.
- **`BS` must be a multiple of 5** (main) / **2** (natives), so a cell's rows stay in one task and two tasks never race on the same `ensure_dataset` lock.
- **Aggregate with the cross-filter arguments** (`challenger_filter="learned"`, `baselines=((plus_native, native_discrete), (moor_native, native_discrete))`). The default same-filter path returns an **empty** headline table, silently. `run_analysis.sh` now aborts loudly if the comparison comes out empty.
- **`collapse_penalty: 5.0` must stay pinned** in `motivation_native.yaml` — the code default is **10.0**. (`yield` mode forces P=0 automatically.)
- **Do not touch the eval protocol** (5 seeds × 4 episodes × horizon 50); it is locked by handoff §5 for comparability with the prior audited run.
- **Jobs run from `code/` (the frozen snapshot), never the live tree.**

---

## 5. Cost (measured, per row, one CPU core)

| Row | Time | Share of a cell |
|---|---|---|
| dataset gen (per **cell**) | 0.6 s | ~0% |
| `plus_native` | 4 s | 0.4% |
| `moor_native` | 8 s | 0.9% |
| `refplan` | 53 s | 5.8% |
| `bamcts` | 254 s | 27.7% |
| `ogsrl` | 596 s | **65.1%** |

Per cell ≈ **916 s**. Main grid = 288 cells ≈ **73 CPU-hours** → at 240 concurrent cores, **~35 min of compute** in two waves. Plus ~1–2 CPU-hours for the resolution replicate.

**The 12–16 h budget is not the constraint; queue wait is.** The three general methods are 98.6% of the cost and OGSRL alone is 65%. The native baselines this whole build was about are ~1% of it.

---

## 6. What to expect in the results (so nothing looks like a bug when it isn't)

- **PLUS-native and MOOR-native are genuinely distinct methods** now. PLUS's posterior over the four mechanistic forms beats MOOR's single fitted Ricker where the forms disagree (Iberian lynx/allee: 9.93 vs 8.60; regime: 10.46 vs 8.82).
- **Spotted turtle:** all four forms agree everywhere, so PLUS and MOOR will legitimately coincide. Not a bug.
- **Egyptian vulture and Bottlenose dolphin** are demographic sinks — the vulture has *no* action with positive growth, so management can only slow the decline. Sinks are reported **separately** from the seven recoverables.
- **The money plot** is return and collapse **by σ and by dynamics family**: the naive Ricker-form solver should degrade visibly on allee/theta/regime cells. Report per-σ and per-family — **never pooled**, and never pooled across reward mode (the objectives differ; use the reward-agnostic battery to compare modes).

---

## 7. Open design note (not a blocker)

`plus_native`'s candidate bank spans the **four mechanistic forms**, not Ricker `K`. This was a deliberate correction: `K` is unidentifiable in this setting (6 of the 11 actions have `r_setpoint ≤ 0`, which zeroes `r_pos` and cancels `K` out of the Ricker exponent), and a K-bank made PLUS-native produce byte-identical episodes to MOOR-native. The full reasoning — including the fact that the original corollary in the context doc was **wrong** — is recorded in `SERVER_CONTEXT_motivation_native_baselines_gate.md` §3.
