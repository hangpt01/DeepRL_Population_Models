# Handoff to Codex — P_safe probe before the real-ecology reward-mode sweep

Date: 2026-07-05
From: Claude review of your doc-alignment + safety-penalty fixes
Scope: modify only inside `discrete_action_cont_obser/real_ecology_cont_obser/`

## TL;DR

Your safety-penalty fix is verified correct and consistent. Two decisions are
made: (1) run the **per-population-cell** benchmark now, pooled-agent migration
later; (2) **probe-then-decide** `P_safe` before any full sweep. This file is the
plan for that probe. **Do not launch the full/headline sweep, and do not start
the pooled migration, until the probe below is done and `collapse_penalty` is
locked.**

## What is already verified (no action needed)

Reviewed against `29_6_Real_Ecology_Setting_Implementation_Plan.tex` (E6/E6') and
`29_6_REAL_ECOLOGY_IMPLEMENTATION_ALIGNMENT_GUIDE.md`:

- `safety_penalty_indicator` defaults to `occupancy` and is threaded identically
  through env, MPC (`planning.py`), BA-MCTS, fitted-Q (`value.py`), OGSRL
  rollouts, and `envs.py`. `crossing` preserved for ablation.
- Egyptian vulture (`N0=41 < s_safe=0.25*325=81.25`) is now penalized every step
  it stays below the floor — the dead-penalty hole is closed.
- MVP diagnostics (`mvp_threshold`, `mvp_fraction`, `mvp_breach`,
  `mvp_occupancy`) emitted and in cache validation.
- Named safety-tier constants (`config.py:28`); Iberian lynx added to the test.
- `BeliefState.features()` hard-requires real scales (no `500`/`50` defaults).
- 35 tests pass; smoke `ok`.

## The one calibration gap the fix created

Switching `crossing -> occupancy` changed the safe penalty from **once per
episode** to **once per step below `s_safe`**. But `collapse_penalty` is still
`10.0` (`config.py:79`) and `test_safe_penalty_exceeds_default_healthy_episode`
still checks only the *single-event* threshold. Over a 25-50 step horizon a
below-safe trajectory can accrue ~10x-50x the healthy benefit, so for depleted /
sink cells the safe-mode return risks being pure penalty with no usable gradient.
Spec E6 says re-fit `P_safe` after resolving crossing-vs-occupancy — that re-fit
has not happened. That is the entire purpose of this probe.

`collapse_penalty` is a persisted env key (`config.py:396`), so it can be set per
cell under `environment:` in the YAML.

## Probe design

**Purpose:** pick `collapse_penalty` for the per-step occupancy penalty, and
confirm runtime. Nothing here is a reportable result.

**Cells — 3 populations spanning the regimes, both reward modes:**

- `Amur tiger` (ricker) — recoverable, starts healthy (normal case)
- `Egyptian vulture` (theta or ricker) — true sink (`r_max<0`), starts below
  `s_safe` (the case the occupancy fix targets; only `a10` translocation adds
  individuals)
- `Puerto Rican parrot` (ricker) — mid-scale, `c_safe=0.20` tier boundary

**Sweep:** `collapse_penalty in {2, 5, 10, 20}` x `reward_mode in {yield, safe}`,
`observation_noise_sigma=0.1`, held-out eval seeds. `yield` cells are penalty-
independent (control it), so you only really vary `collapse_penalty` on the
`safe` cells; keep one `yield` cell per population as the no-penalty reference.

**Methods:** enough to read the return scale, not the full seven. Use `ogsrl`
(safety-constrained, most sensitive to the penalty), `mopo`, and `moor`.

**Budget:** short — base it on `configs/real_probe.yaml` (2000 transitions,
ep_len 25, eval horizon 50, 2 seeds). Do **not** use full-experiment budgets.

## Decision rule for locking P_safe

Pick the **smallest** `collapse_penalty` such that, in `safe` mode:

1. **Tiger** — the safe agent still takes beneficial (non-`a0`) actions; it is
   not frozen into pure risk-avoidance. (Penalty shapes, does not swamp.)
2. **Vulture** — reducing penalty is only achievable via `a10` translocation;
   the agent should prefer `a10` over `a0`. (Penalty is meaningful for a sink.)
3. **Parrot** — safe vs yield agents are behaviorally distinguishable (different
   below-safe occupancy / action mix), not identical.

Read these off `mvp_fraction`, below-`s_safe` step fraction, action histograms,
and safe-vs-yield return separation in the evaluation summaries. Lock the chosen
value, update `collapse_penalty` default (or set it explicitly in the experiment
config) and update `test_safe_penalty_exceeds_default_healthy_episode` to reflect
the per-step occupancy accounting rather than a single event.

## Pre-flight check before running (blocker)

Confirm the active data table is the intended latest. The authoritative table in
this checkout is `discrete_action_cont_obser/real_ecology_data/`; the redundant
package-local `revised_cost_action_table/` copy has been removed. Before spending
compute, verify the loader reports that path in generated metadata. If the table
changes, regenerate datasets/caches (they are keyed on `data_table`,
`safety_penalty_mode`, and `mvp_threshold`, so stale caches will be rejected —
good).

## Commands

Smoke first, then the probe cells. From
`discrete_action_cont_obser/real_ecology_cont_obser/`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml
```

Build a probe config per cell (copy `configs/real_probe.yaml`, set
`environment.population`, `environment.reward_mode`, `environment.kind`, and
`environment.collapse_penalty`), then run the generate -> run -> aggregate flow
(`cli generate|run|aggregate`, mirroring the existing scripts). If you script the
sweep, add the probe rows to `scripts/make_real_experiment_manifests.py` behind a
`--probe` flag rather than editing `PILOT_POPULATIONS`/`PROBE_ROWS` in place.

## Do NOT do yet

- Do not launch the full reward-mode sweep until `collapse_penalty` is locked.
- Do not start the pooled-agent migration (dataset schema / filter context /
  method features / runners). It is deferred by explicit decision; per-cell runs
  first.
- Do not pool sink (vulture, dolphin) and recoverable populations in any summary.
- Keep backend declared and recorded; do not pool CPU/GPU rows silently.

## Report back

After the probe: the chosen `collapse_penalty`, the per-cell safe-vs-yield return
separation and action mixes that justified it, runtime per cell, and confirmation
of which data table is active.
