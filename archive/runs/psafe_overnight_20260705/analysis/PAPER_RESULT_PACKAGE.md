# Real-ecology benchmark — paper result package

Compiled from `RESULTS_SUMMARY.md`, `DECISION_psafe.md`, `P5_control_review.md`,
and `learned_vs_raw_p5.md`. All numbers are copied from the audited analysis
artifacts (no re-computation here). Setting: continuous-state POMDP, 9 real
populations, per-step **occupancy** safety penalty, `reward_mode=safe` unless
noted. Metric `true_return_mean` (== operational return in real-setpoint mode);
higher is better.

---

## Table 1 — P_safe penalty decision (safe mode)

Filter-consistent decision subset (general learners `learned`; PLUS/MOOR
`ricker`; P5 `raw` excluded), averaged over family × σ_obs. 1008 comparable rows
per penalty.

**Amur tiger (recoverable)**

| P | true_return | collapse_entry | unsafe_frac | final_state | min_state | econ_cost |
|---|---:|---:|---:|---:|---:|---:|
| 2  | 2.84  | 0.095 | 0.021 | 83.7  | 56.8 | 6.842 |
| 5  | **2.17**  | 0.057 | 0.018 | 97.1  | 64.4 | 7.911 |
| 10 | 0.85  | 0.053 | 0.019 | 97.2  | 65.6 | 8.462 |
| 20 | -2.51 | 0.065 | 0.024 | 105.4 | 67.4 | 9.079 |

**Puerto Rican parrot (recoverable)**

| P | true_return | collapse_entry | unsafe_frac | final_state | min_state | econ_cost |
|---|---:|---:|---:|---:|---:|---:|
| 2  | 2.57  | 0.176 | 0.070 | 158.5 | 65.0 | 3.443 |
| 5  | **0.62**  | 0.163 | 0.065 | 162.5 | 66.2 | 3.987 |
| 10 | -2.29 | 0.161 | 0.063 | 181.3 | 66.7 | 4.897 |
| 20 | -8.25 | 0.159 | 0.062 | 182.8 | 66.8 | 5.393 |

**Egyptian vulture (sink)**

| P | true_return | collapse_entry | unsafe_frac | final_state | min_state | econ_cost |
|---|---:|---:|---:|---:|---:|---:|
| 2  | -37.63  | 0.000 | 1.000 | 2.8 | 2.1 | 3.928 |
| 5  | -93.35  | 0.000 | 1.000 | 3.0 | 2.1 | 4.993 |
| 10 | -186.04 | 0.000 | 1.000 | 2.8 | 2.1 | 6.133 |
| 20 | -370.91 | 0.000 | 1.000 | 3.2 | 2.6 | 7.163 |

Decision rule: largest P keeping both recoverable decision-pops' safe return ≥ 0
while reducing collapse entries vs P=2 → **collapse_penalty = 5**. P=20
over-penalises (returns negative); P=10 already pushes the parrot negative. The
vulture is penalty-dominated at every P (see Table 4).

---

## Table 2 — General learners vs ecological baselines (P5, safe)

Best general learner (of ogsrl/mopo/bamcts/refplan/delphic, `learned`) vs best of
PLUS-ricker / MOOR-ricker, on matched population × family × noise cells.

| Scope | Cells | Best general | Best baseline | Δ | Wins vs best baseline |
|---|---:|---:|---:|---:|---:|
| All 9 populations | 144 | -3.93 | -4.96 | +1.04 | 102/144 (70.8%) |
| Recoverable only  | 112 | 7.74  | 6.28  | +1.46 | **96/112 (85.7%)** |
| Sinks only        | 32  | -44.76| -44.30| -0.46 | 6/32 (18.8%) |

Per general method (P5, safe; "beats both" = beats PLUS-ricker and MOOR-ricker):

| Method | mean return | Δ vs PLUS | Δ vs MOOR | beats both | recov. beats both | mean unsafe | mean collapse |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ogsrl** | -5.06 | +0.16 | +0.26 | 81/144 (56.2%) | **79/112 (70.5%)** | 0.113 | 0.014 |
| bamcts    | -5.69 | -0.47 | -0.37 | 68/144 (47.2%) | 64/112 (57.1%) | 0.126 | 0.047 |
| refplan   | -5.31 | -0.08 | +0.02 | 38/144 (26.4%) | 37/112 (33.0%) | 0.121 | 0.034 |
| mopo      | -5.45 | -0.22 | -0.13 | 34/144 (23.6%) | 34/112 (30.4%) | 0.121 | 0.036 |
| delphic   | -9.77 | -4.54 | -4.44 | 33/144 (22.9%) | 33/112 (29.5%) | 0.185 | 0.188 |

OGSRL is the strongest single general method.

---

## Table 3 — Learned vs raw filter ablation (P5, safe)

Same cells, learned particle filter vs raw observations (no filtering).

| Method | learned | raw | Δ (learned−raw) | unsafe learned/raw |
|---|---:|---:|---:|---:|
| bamcts  | -5.69 | -5.37 | -0.32 | 0.126 / 0.126 |
| delphic | -9.76 | -9.10 | -0.67 | 0.185 / 0.199 |
| moor    | -4.50 | -4.69 | +0.18 | 0.112 / 0.115 |
| mopo    | -5.45 | -5.42 | -0.03 | 0.121 / 0.122 |
| ogsrl   | -5.06 | -4.49 | -0.58 | 0.113 / 0.112 |
| plus    | -5.23 | -5.73 | +0.50 | 0.129 / 0.136 |
| refplan | -5.31 | -5.29 | -0.02 | 0.121 / 0.121 |

By noise level (mean over methods/pops/families): σ=0.0 Δ+0.03, σ=0.1 Δ−0.42,
σ=0.2 Δ−0.21, σ=0.4 Δ+0.06 — within run noise at every level. PLUS is
filter-inert (learned ≡ ricker); MOOR is filter-sensitive.

---

## Table 4 — Sink populations reported separately (P5, safe)

Same filter-consistent policy subset as Table 1 (general learners `learned`,
PLUS/MOOR `ricker`), averaged over family × σ_obs.

| Sink | unsafe_frac | min_state | final_state | true_return | regime |
|---|---:|---:|---:|---:|---|
| Egyptian vulture | 1.000 | 2.1 | 3.0 | -93.3 | always below s_safe; penalty-dominated |
| Bottlenose dolphin | 0.097 | 10.4 | 13.2 | -2.8 | mostly above s_safe, below absolute MVP floor |

The two sinks are not equivalent and must not be pooled with recoverables or with
each other. Only translocation (a10) adds individuals to a sink.

---

## Result text (draft)

**Penalty calibration.** Under the per-step occupancy safety penalty, we select
the collapse-penalty weight on the safe-mode return of the recoverable decision
populations. Increasing the penalty raises final abundance but reduces
operational return; collapse entries are lower than P=2 for the recoverable
decision populations, though not strictly monotonic for every population at every
P. The largest tested penalty that keeps both recoverable decision populations'
mean safe return non-negative is `collapse_penalty = 5` (Amur tiger 2.17, Puerto
Rican parrot 0.62). At P=10 the parrot is already negative while the tiger remains
positive; by P=20 both are over-penalised (−2.51 and −8.25). We therefore lock
`P_safe = 5`. This is an **average-case** choice: it is made on the mean across
family and noise conditions, and the return distribution retains a negative tail
(the parrot's mean at P5 is only +0.62, and individual cells remain negative even
at P5). A worst-case criterion would select a smaller penalty (P≈3–4); we report
P5 as the average-case operating point and provide the full per-cell distribution.

**Method comparison.** At the locked penalty, general offline-RL learners
outperform the two ecological baselines (PLUS-Ricker, MOOR-Ricker) on the
recoverable populations, winning 96 of 112 matched population×family×noise cells
(85.7%) with a +1.46 mean-return margin; OGSRL is the strongest single method
(beats both baselines on 70.5% of recoverable cells at the lowest collapse rate,
0.014). This advantage is specific to recoverable populations: on the two
demographic sinks the general learners do not help (6/32 cells), because no
vital-rate action can lift those populations and only translocation adds
individuals. Sinks are consequently reported separately (Table 4) and never pooled
with recoverables — and the two sinks are themselves distinct: the Egyptian
vulture sits permanently below its safety floor (unsafe fraction 1.0,
penalty-dominated at every penalty), whereas the bottlenose dolphin sits mostly
above its relative floor (unsafe fraction 0.097) though below the absolute
minimum-viable-population floor.

**Filter ablation.** We find that learned particle filtering yields no material
control-return advantage over raw observations at the locked penalty: per-method
differences are within run-to-run noise and not consistently in the learned
filter's favour, even at the highest observation noise (σ=0.4, Δ≈+0.06). We report
this as an ablation rather than a headline result; it concerns control return
only and does not test latent state-estimation accuracy, for which raw
observations provide no estimate. (PLUS is filter-inert in our setup; MOOR is
filter-sensitive.)

---

## Artifact provenance

- **Run:** `real_ecology_runs/psafe_overnight_20260705/`, frozen code+data
  snapshot; all jobs CPU.
- **`analysis/all_metrics.csv`: 7488 rows.** Per penalty: safe `learned` 1008 +
  `ricker` 288 (all four penalties), plus P5 `raw` 1008; P10 also carries the
  `yield` mode (learned 1008 + ricker 288). Grid = 9 populations × 4 families
  (ricker/allee/theta/regime) × 4 σ_obs (0.0/0.1/0.2/0.4) × 7 methods × filter
  fronts.
- **Backend:** `numpy` for all 7488 rows (no GPU; no CPU/GPU pooling).
- **Data table:** `…/psafe_overnight_20260705/code/real_ecology_data`
  (single authoritative vendored table, frozen).
- **Configs:** `real_experiment_p{2,5,10,20}.yaml`, identical except
  `collapse_penalty ∈ {2,5,10,20}`; `safety_penalty_mode: occupancy` throughout.
  Budget: 4000 dataset transitions, episode length 25, 256 filter particles,
  ensemble 5, planner horizon 5 × 96 sequences × 32 particles; evaluation 5 seeds
  × 4 episodes × horizon 50.
- **Completion:** 6,480 grid rows + 1,008 P5 raw-ablation rows, 0 failures.
- **Tests:** real-ecology suite 37/37 passing (incl. the `beats_both_rate`
  aggregate regression test). `cli aggregate` fixed in the live package; the
  frozen snapshot retains the pre-fix code by design — aggregate via the live
  package against the frozen data tree.
- **Not run (by decision):** calibration gates, PLUS-on-GPU, sink action-preference
  diagnostic, pooled-agent architecture.
