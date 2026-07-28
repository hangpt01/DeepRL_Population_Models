# Results — Motivation Experiment (native ecological baselines vs general offline MBRL)

**Date:** 2026-07-12
**Run:** `real_ecology_runs/motivation_native_20260711/`
**Status:** run COMPLETE and CLEAN. **The result contradicts the paper's motivating hypothesis.**

---

## 1. The headline, stated plainly

The motivation experiment was designed to show that **general offline MBRL beats naive ecological baselines** in a continuous, noisy, structurally-uncertain conservation setting. 

**It shows the opposite, comprehensively.**

> **The native ecological baselines beat the general methods in 860 of 864 paired cells (99.5%).**

| Beats-both rate (general challenger vs best native, recoverable populations) | | |
|---|---|---|
| `refplan` | **0.00** at every σ, both reward modes | (28 cells each) |
| `bamcts` | **0.00** at every σ, both reward modes | |
| `ogsrl` | **0.00–0.04** | |

This is not a marginal loss. It is a near-total inversion of the expected ordering, stable across every noise level, every dynamics family, and both reward modes.

### Return by σ (recoverable populations, safe mode, best-of-side)

| σ | general | native | collapse (gen) | collapse (nat) |
|---|---|---|---|---|
| 0.0 | 8.076 | **8.774** | 0.000 | 0.000 |
| 0.1 | 7.647 | **8.767** | 0.000 | 0.000 |
| 0.2 | 7.616 | **8.734** | 0.000 | 0.000 |
| 0.4 | 7.462 | **8.672** | 0.000 | 0.000 |

### The money plot, inverted (recoverable, safe, pooled over σ)

| family | general | native | gap (gen − nat) |
|---|---|---|---|
| ricker | 7.749 | 8.420 | **−0.670** |
| allee | 7.762 | 8.895 | **−1.133** |
| theta | 7.828 | 8.935 | **−1.107** |
| regime | 7.462 | 8.698 | **−1.235** |

**The gap is *widest* on allee / theta / regime** — precisely the families where a naive Ricker-form solver was supposed to visibly degrade. The intended structural handicap is not merely absent; the native baselines do *relatively better* there.

---

## 2. The run itself is clean — this is not an artifact

Everything that could have produced a spurious result was checked and holds:

- **1440/1440** main-grid rows and **1152/1152** resolution rows completed. All four Slurm arrays `COMPLETED`. Zero failed rows.
- **All 8 acceptance checks passed** on a 288-row battery before launch, including the not-crippled check (natives beat their adapted counterparts on all 32 Ricker cells by +1.0 to +2.68) and a runtime reward-leakage proof.
- **288/288 cells have a single shared `dataset_sha256`** — the general and native methods provably consumed the identical offline dataset.
- **Resolution: 0/288 cells flip the headline** between coarse (31), main (61), and fine (91) native grids. The result is not a discretization artifact.

### Two false alarms I raised and then disproved (recorded so they are not re-raised)

1. **"171 rows have `fallback_count > 0`."** These are **not** crashes. `fallback_count` conflates genuine exceptions with OGSRL's *designed* deployment guardian ([ogsrl.py:433](../../src/real_ecology_benchmark/methods/ogsrl.py#L433)), which flags `hard_fallback` and takes the least-violating action when no action satisfies its OOD and safety constraints. Verified on the worst cell (Egyptian vulture / theta): **0 exceptions, 50/50 guardian activations**, episode completes normally. The guardian saturates on demographic sinks because the vulture starts *below* its safety floor (N0=41 < s_safe=81.2) with **no action having positive growth** — every action is unsafe, so nothing is feasible. That is a correct and meaningful behaviour, not a bug. **Known defect (cosmetic):** the evaluator should split these into two counters; as written, `fallback_count` cannot be used as a crash detector.
2. **"Resolution check FAILED, 2/288 cells flip."** A bug in *my check*, since fixed. The two cells showed `+1, +1, 0` — an exact **tie** at the main grid, not a reversal. `np.sign` returns `0` for a tie and my code counted that as a third distinct sign. At σ=0 the environment is deterministic, so a general method and a native that find the same policy return bit-identical values. Corrected test (reversal = wins at one grid, loses at another): **0/288 flips.**

---

## 3. Why the natives win — diagnosis

### It is NOT planning depth (tested, largely rejected)

The natives solve to convergence with tabular value iteration (250 iterations, γ=0.95); the general methods plan with a **5-step** particle MPC. An obvious confound. Truncating the native's value iteration to the same 5-step lookahead, on Amur tiger / ricker / σ=0.2 / safe:

| native VI depth | return |
|---|---|
| 1 | −0.389 |
| 2 | 4.945 |
| 3 | 5.428 |
| **5 (matched to MPC horizon)** | **5.438** |
| 10 | 5.540 |
| 250 (as run) | 5.662 |

Depth accounts for only **0.22 of the native's edge (~4%)**. At *matched* lookahead the native still scores 5.438 against `ogsrl` 4.211, `refplan` 3.486, `bamcts` 2.811 on the same cell. **Planning depth is not the explanation.**

### It IS an informational asymmetry: the "naive" baselines are near-oracle

In this environment the transition is essentially **determined by two public tables**:

- `r_eff = clip(delta_r[a], r_min, r_max)` — the per-action growth set-point, read straight from `actions.csv` / `action_effects_long.csv`;
- `K_base` — read straight from `species.csv`.

A mechanistic solver that reads those tables therefore has an **almost exact dynamics model**. The only thing it does not know is the *functional form* (Ricker vs Allee vs theta vs regime) — and the evidence above shows that misspecification is **second-order** for action ranking.

Meanwhile the general offline MBRL methods must **learn the dynamics from 4000 transitions** with an approximate model.

So the comparison as constructed is not "general MBRL vs naive ecology". It is closer to:

> **near-exact hand-specified model + exact solver  vs  learned approximate model + short-horizon planner.**

The natives should win that, and they do.

---

## 4. What this means for the paper (PI decision — not mine to make)

The premise the paper is built on — *three escalating layers of uncertainty (state → parameter → structural), with structural/model-form uncertainty breaking naive ecological solvers* — **is not instantiated by this environment as built.** Specifically:

- **Parameter uncertainty is absent.** `r` is a public function of the action and `K_base` is in the species table. There is nothing to be uncertain about. (This is the same finding that forced PLUS-native's candidate bank away from `K` and onto the four mechanistic forms — see the context doc §3.)
- **Structural uncertainty is present but weak.** 60% of (population, action) pairs engage the density-dependent term, so the forms genuinely differ — yet the misspecification costs the native solver almost nothing in action ranking.
- **Collapse pressure is near-absent among recoverables** (`collapse_entry` = 0.000 for both sides at every σ). The conservation/safety angle — the paper's core motivation — **never activates** on the populations that matter. It only bites on the two demographic sinks (collapse_entry 0.229, unsafe_fraction 0.657), which are reported separately by design.

I want to be explicit about the thing I am *not* doing: I am not proposing to tune this experiment until it produces the desired ordering. The honest readings are:

**(a) Report the negative result and reframe.** In this real-ecology conservation setting, a well-specified discretized ecological solver is *very strong*, and general offline MBRL does not beat it. That is a legitimate — and, for the field, useful — finding.

**(b) The environment does not test what the paper claims it tests.** If the paper needs structural uncertainty to bite, the environment must stop handing the solver the exact per-action growth rates. That is an environment-design change, and it invalidates comparability with the prior audited runs.

**(c) Check whether the general methods are simply under-powered.** `bamcts` at 2.811 and `refplan` at 3.486 on a cell where a tabular solver gets 5.66 is a large gap. These are the audited implementations from prior runs, but "audited" is not "well-tuned for this setting."

Options (a) and (b) are mutually exclusive framings. (c) should be settled first, because if the general methods are simply weak, the headline is about tuning, not about ecology.

---

## 5. Artifacts

- `analysis/aggregate_main.json` — 1440 summaries, 864 paired cells, `beats_both_cells`, `beats_both_rate`
- `analysis/aggregate_res_b31.json`, `analysis/aggregate_res_b91.json` — resolution replicates
- `analysis/resolution_check.json` — 0/288 flips (PASS)
- `outputs/acceptance_20260711/acceptance_report.json` — 8/8 acceptance checks PASS
- `PROVENANCE.txt` — frozen commit + working-tree state
