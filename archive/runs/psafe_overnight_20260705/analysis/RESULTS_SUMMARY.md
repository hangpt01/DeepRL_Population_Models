# Real-ecology P_safe experiment — result summary (writeup-facing)

Run: `psafe_overnight_20260705` (CPU/numpy, frozen snapshot). Full penalty grid
`collapse_penalty ∈ {2,5,10,20}` + a P5 learned-vs-raw filter ablation.
6,480 grid rows + 1,008 raw-ablation rows, 0 failures. Audited 2026-07-05.

## 1. P_safe decision: collapse_penalty = 5 (average-case)

Chosen on the filter-consistent safe-mode decision subset (general learners
`learned`; PLUS/MOOR `ricker`; P5 `raw` excluded), 1008 comparable rows/penalty.

Recoverable decision-pop safe `true_return` by penalty:

| P | Amur tiger | Puerto Rican parrot |
|---|---:|---:|
| 2 | 2.84 | 2.57 |
| 5 | **2.17** | **0.62** |
| 10 | 0.85 | -2.29 |
| 20 | -2.51 | -8.25 |

Rule: largest P keeping recoverable decision-pops' safe return ≥ 0 while reducing
collapse entries vs P=2 → **P = 5**. P=20 clearly over-penalises (returns go
negative); P=10 already pushes the parrot negative.

**Caveat (average-case):** the pick is on the mean. The distribution has a
negative tail — the parrot's mean at P5 is only +0.62 (near break-even), and some
individual cells stay negative even at P5. Report P5 as an average-case lock and
show the cell distribution; a worst-case criterion would push lower (P3–4).

## 2. General learners vs ecological baselines (P5, safe)

Best general learner vs best of PLUS-ricker / MOOR-ricker on matched
population×family×noise cells (`true_return_mean`, higher better):

| Scope | Cells | Best general | Best baseline | Δ | Wins |
|---|---:|---:|---:|---:|---:|
| All 9 populations | 144 | -3.93 | -4.96 | +1.04 | 102/144 (71%) |
| Recoverable only | 112 | 7.74 | 6.28 | +1.46 | **96/112 (86%)** |
| Sinks only | 32 | -44.76 | -44.30 | -0.46 | 6/32 (19%) |

Single strongest general method: **OGSRL** (beats both baselines on 79/112
recoverable cells; mean unsafe 0.113, mean collapse 0.014). General learners help
on recoverable populations; they do not help on the two sinks.

## 3. Learned vs raw filter (P5 ablation): null control result

Learned particle filtering gives **no material control-return advantage** over raw
observations, even at the noisiest setting (σ=0.4 Δ≈+0.06); per-method deltas are
within run noise. Report as an ablation, not a headline gain. This does not test
state-estimation accuracy (raw has no latent estimate). Note: **PLUS is
filter-inert** (learned≡ricker); **MOOR is filter-sensitive**. Detail in
`learned_vs_raw_p5.md`.

## 4. Sink caveat: report vulture and dolphin separately

The two demographic sinks are **not equivalent** and must not be pooled with
recoverables (or each other):

| Sink @ P5 safe | unsafe_frac | min_state | true_return |
|---|---:|---:|---:|
| Egyptian vulture | 1.00 (always below s_safe) | 2.1 | -93 (penalty-dominated) |
| Bottlenose dolphin | 0.075 (mostly above s_safe) | 10.3 | -1.5 |

The vulture is stuck below the floor and penalty-dominated at every P; only
translocation (a10) adds individuals. The dolphin sits below the absolute MVP
floor but mostly above its relative `s_safe`, so the occupancy penalty rarely
fires — a qualitatively different regime.

## 5. Remaining open items (none block writeup)

- **Calibration gates** at P5 — optional; only if a data-quality/behavior-policy
  coverage claim is made.
- **Sink action logging** (whole-episode / a10 frequency) — optional; needed only
  to make a strong sink action-preference claim. Current logs have danger-zone
  action fractions + action_entropy, not global per-action histograms.
- **Pooled-agent architecture** — deferred; this is the per-population-cell
  benchmark (spec mandates one pooled agent, a separate migration).
- **PLUS-on-GPU** — deferred; unnecessary at this budget (PLUS = 90s/row CPU).
- The `cli aggregate` bug is fixed in the live package (regression-tested); the
  frozen snapshot retains the old code by design — aggregate via the live package.

Artifacts: `all_metrics.csv` (7488 rows, filter/backend/data_table columns),
`rollup.csv` (by penalty×mode×method×filter), `DECISION_psafe.md`,
`P5_control_review.md`, `learned_vs_raw_p5.md`, `convergence_p*.png`.
