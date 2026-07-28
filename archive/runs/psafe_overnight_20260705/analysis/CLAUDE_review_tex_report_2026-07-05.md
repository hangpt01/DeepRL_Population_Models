# Review — Codex's LaTeX experiment-results report

Reviewer: Claude (Opus 4.8), 2026-07-05. Read-only audit; no experiments run.
Files reviewed: `REAL_ECOLOGY_EXPERIMENT_RESULTS.tex` (+ compiled PDF),
`make_paper_figures.py`, `figures/*.png`.

## Verdict: accurate and honest. Safe to present.

The report reads only the audited artifacts (`all_metrics.csv`, `episodes.csv`,
`training_history.csv`), uses the correct filter-consistent policy subset
(general learners `learned`; PLUS/MOOR `ricker`), and its numbers/figures match
the audited tables. The headline figure (`fig_psafe_decision_returns.png`) was
visually spot-checked and reproduces Table 1/4 exactly (vulture −37.6/−93.4/
−186/−371; tiger & parrot near zero, crossing negative by P10–20; P=5 marker).

## Direct answer to your question: which plots the current results support

**Verified against the raw logs.** `episodes.csv` is **per-episode** (20 rows =
5 seeds × 4 episodes), holding aggregate metrics + danger-zone action fractions.
No per-timestep state/action/reward table exists anywhere in the run (searched
for trajectory/trace/rollout files — none). So:

| Requested plot | Supported? | Why |
|---|---|---|
| Reward **over time** (per timestep) | **No** | Only per-episode aggregate returns are logged; no per-step reward trace. |
| Actions **over time** (per timestep) | **No** | Only per-episode `action_entropy` + danger-zone action fractions; no per-step action trace. |
| Reward by method / penalty / scope | Yes | From `all_metrics.csv` (aggregate). |
| Method comparison | Yes | Return-by-method, win-rate figures. |
| Training convergence | Yes (overview) | `training_history` loss-vs-step, per penalty. Dominated by learning components (OGSRL surrogate loss, model-fit RMSE); pure planners have no curve. |
| Validation error | Yes | `training_holdout_dynamics_rmse` per method (real field). |

**Codex's central claim is correct**: true reward-over-time and action-over-time
trajectory plots are **not** producible from the current run, and Codex did not
fabricate them — it stated the gap explicitly in the TeX. That is the right call.

## To get the two missing plots (if you want them later)

Codex's proposed fix is right but note it is **not "just a rerun"**: it needs an
**evaluator code change** to log per-step records (`episode, timestep, state,
observation, action_id, reward, safety_threshold, reward_mode`) **and then** a
small targeted P5 rerun (a handful of cells is enough for illustrative
trajectories). Cheap, but it is a code change first. You said no new experiments
now, so this is a future option, not for tomorrow.

## Minor comments (non-blocking, for framing)

1. **Holdout RMSE is a dynamics-model validation error**, not policy-performance
   validation — available for model-based methods that fit the ensemble. The
   caption is accurate ("holdout dynamics RMSE"); just describe it as model-fit
   validation when presenting, not policy validation.
2. **The action figure is danger-zone-conditional** (fractions computed only over
   steps inside the danger zone). For populations rarely in that zone the
   denominator is small/uninformative — the same limitation flagged earlier for
   the sink action-preference question. The TeX correctly says it is not a
   trajectory; consider also noting the danger-zone conditioning so the audience
   does not read it as a whole-episode action mix.
3. **Convergence plots are method-dependent** (planners bamcts/refplan/plus have
   no training loss). The "Partly" label handles this honestly; in the talk, call
   it convergence of the learning-based components.
4. PDF compiled (1.1 MB, all figures embedded per the log); I did not re-render it
   visually beyond the source and the PNGs. Remaining LaTeX messages are layout
   over/underfull warnings only.

## Bottom line for the presentation

Use the report as-is. It supports: P_safe decision curve, method comparison,
learned-vs-raw ablation, sink separation, training-convergence overview, and
holdout validation error — all backed by real data. It does **not** support
reward-over-time or action-over-time trajectories, and that limitation is stated
honestly in the document so you will not overclaim. If those trajectory visuals
are important for the audience, they require per-step logging + a small P5 rerun
afterward.
