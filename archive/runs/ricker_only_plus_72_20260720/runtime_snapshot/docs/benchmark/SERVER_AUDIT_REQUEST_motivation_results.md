# Audit Request — Motivation Experiment Results

**For:** codex
**Run:** `real_ecology_runs/motivation_native_20260711/`
**Frozen commit:** `e74526e` · frozen 2026-07-11T22:49 · finished 2026-07-12T02:09
**Claim under audit:** the motivation experiment completed cleanly and **contradicts the paper's premise** — native ecological baselines beat general offline MBRL in 99.5% of cells.

Full writeup: `SERVER_RESULTS_motivation_native_run.md`. This file is the short version plus the exact commands to attack each claim.

---

## 0. Slurm outcome (all COMPLETED)

| job | array | state |
|---|---|---|
| 58255038 motiv-main | 288 tasks × 5 rows | 288 COMPLETED |
| 58255039 motiv-res31 | 288 × 2 | 288 COMPLETED |
| 58255040 motiv-res91 | 288 × 2 | 288 COMPLETED |
| 58255041 motiv-analysis | 1 | COMPLETED |

---

## 1. Integrity (please verify independently)

```
main summaries            1440 / 1440
resolution summaries      1152 / 1152
cells                     288    | cells with >1 dataset hash: 0
non-OGSRL rows w/ fallback   0
models                    bamcts, moor_native, ogsrl, plus_native, refplan
reward modes              safe, yield        sigmas  0.0 0.1 0.2 0.4       populations  9
```
Pre-launch acceptance battery: **8/8 PASS** (`outputs/acceptance_20260711/acceptance_report.json`, 288 rows).

---

## 2. THE HEADLINE — the number to attack

> **General beats BOTH natives in 4 / 864 paired cells = 0.5%.**

Beats-both rate (recoverable populations only):

| method | mean rate | range over σ / reward mode |
|---|---|---|
| `refplan` | **0.000** | 0.00 – 0.00 |
| `bamcts` | **0.000** | 0.00 – 0.00 |
| `ogsrl` | 0.018 | 0.00 – 0.04 |

Return by σ (recoverable, safe, best-of-side) — **native ahead everywhere**:

| σ | general | native | gap |
|---|---|---|---|
| 0.0 | 8.076 | 8.774 | −0.698 |
| 0.1 | 7.647 | 8.767 | −1.119 |
| 0.2 | 7.616 | 8.734 | −1.118 |
| 0.4 | 7.462 | 8.672 | −1.210 |

Return by dynamics family — **gap is WIDEST on allee/theta/regime**, the families where the naive Ricker solver was supposed to fail:

| family | general | native | gap |
|---|---|---|---|
| ricker | 7.749 | 8.420 | −0.670 |
| allee | 7.762 | 8.895 | **−1.133** |
| theta | 7.828 | 8.935 | **−1.107** |
| regime | 7.462 | 8.698 | **−1.235** |

---

## 3. Claims I want you to try to break

### C1 — "The run is clean; the hypothesis failed, not the pipeline."
Check: row counts, all-COMPLETED, one dataset hash per cell, zero non-OGSRL fallbacks, 8/8 acceptance.
```bash
cd real_ecology_runs/motivation_native_20260711
find outputs/main -name summary.json | wc -l          # expect 1440
sacct -j 58255038 -X --format=State -n | sort | uniq -c
python -c "import json;r=json.load(open('../../outputs/acceptance_20260711/acceptance_report.json'));print(all(c['pass'] for c in r['checks'].values()))"
```

### C2 — "OGSRL's 171 `fallback_count > 0` rows are its designed guardian, NOT crashes."
This is the one where I first cried wolf, so please re-check it hard. `fallback_count` conflates
genuine exceptions (evaluator.py:102,106) with OGSRL's `hard_fallback` flag (evaluator.py:136 ←
ogsrl.py:433). I drove the worst cell (Egyptian vulture / theta) end-to-end: **0 exceptions,
50/50 guardian activations, episode completes.** The guardian saturates because the vulture starts
below its safety floor (N0=41 < s_safe=81.2) with **no action having positive growth**, so no action
is feasible.
**Attack:** is there any row where the fallback is a real exception? Is my Egyptian-vulture repro representative?

### C3 — "Resolution is stable: 0/288 flips."
The analysis job first reported **FAIL, 2/288**. I claim that was a bug in *my check*: `np.sign`
returns 0 for an exact tie and I counted the tie as a third sign. The two cells were
`+1, +1, 0` — an exact tie at b61 (general 9.088964 vs native 9.088964), never a reversal. At σ=0
the env is deterministic, so two policies that coincide return bit-identical values.
**Attack:** is "reversal only" the right definition? Should near-ties (|diff| ~1e-3) be flagged too?

### C4 — "Planning depth is NOT the explanation." (the diagnosis most likely to be wrong)
Natives solve with tabular VI (250 iters, γ=0.95); generals use a **5-step** particle MPC. Obvious
confound. Truncating native VI to a matched 5-step lookahead on Amur tiger/ricker/σ=0.2/safe:

| native VI depth | 1 | 2 | 3 | **5** | 10 | 250 |
|---|---|---|---|---|---|---|
| return | −0.389 | 4.945 | 5.428 | **5.438** | 5.540 | 5.662 |

Same cell, generals: `ogsrl` 4.211, `refplan` 3.486, `bamcts` 2.811. So depth buys only 0.22 (~4%),
and the native still beats all three at matched lookahead.
**Attack:** is VI-iteration count a fair proxy for MPC lookahead depth? (A VI *iteration* is a full
Bellman backup over all states; an MPC *step* is a rollout — these are not obviously commensurate.
**This is the weakest link in my reasoning and the thing I most want checked.**) Is one cell enough?

### C5 — "The natives win because they're near-oracle, not naive."
`r_eff = clip(delta_r[a])` comes straight from the public action table and `K_base` from
`species.csv`, so a mechanistic solver has an almost-exact dynamics model; only the functional form
is unknown and that is second-order. The generals must learn dynamics from 4000 transitions.
**Attack:** the adapted `moor`/`plus` read the *same* tables and still score ~3.2 — so if information
were the whole story, they should also win. What does that imply about C4/C5? (I think it means the
*solver* matters too, which partially re-opens C4. Please push on this.)

### C6 — "The generals may simply be under-tuned." (untested)
`bamcts` 2.811 and `refplan` 3.486 on a cell where a tabular solver gets 5.662 is a big gap. I have
**not** run a planner-capacity sweep. If a bigger horizon / more sequences / less pessimism closes a
~1.1 gap, the headline is about hyperparameters, not ecology.
**Attack:** is this the right next experiment, and is the current planner config (`horizon=5,
sequences=96, particles=32, pessimism=0.5`) defensible as "a fair shot" for the general methods?

---

## 4. Known defects (already found, not yet fixed)

1. **`fallback_count` conflates crashes with OGSRL's guardian** (evaluator.py:136). It cannot be used as a crash detector. Should be split into two counters.
2. **My acceptance battery's general-method slot was `refplan` only** — it never ran `bamcts`/`ogsrl`, which is why C2's guardian behaviour never showed up pre-launch. A8's "zero fallbacks" was therefore scoped narrower than it appeared.
3. `np.sign` tie bug (C3) — fixed in `run_analysis.sh` and `scripts/run_motivation_acceptance.py`.

---

## 5. What I am explicitly NOT doing

I am not tuning the experiment until it yields the desired ordering. If the result stands, it stands.
The open PI decision (report the negative result / redesign the env / first settle C6) is in
`SERVER_RESULTS_motivation_native_run.md` §4.
