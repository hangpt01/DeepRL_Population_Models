# Codex No-Compute Analysis Cleanup

Date: 2026-07-05

Purpose: document the analysis-only cleanup after the P5 raw-filter ablation finished, so Claude can audit the exact artifacts and assumptions.

## What I Did

No experiments were launched. No Slurm jobs were submitted or cancelled.

I updated the run-local analysis script:

- `discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705/analysis/analyze_psafe.py`

The change makes the P_safe decision table filter-consistent across penalties:

- general learners use `filter=learned`
- ecological baselines `PLUS` and `MOOR` use `filter=ricker`
- P5 `raw` rows are excluded from penalty selection
- P5 `raw` rows remain a separate learned-vs-raw ablation

I then reran the analysis script locally against the finished artifacts:

```bash
python discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705/analysis/analyze_psafe.py \
  discrete_action_cont_obser/real_ecology_runs/psafe_overnight_20260705
```

This is a no-compute analysis refresh over existing `summary.json` files.

## Artifacts Written Or Refreshed

Refreshed:

- `analysis/all_metrics.csv`
- `analysis/rollup.csv`
- `analysis/DECISION_psafe.md`
- `analysis/convergence_p2.png`
- `analysis/convergence_p5.png`
- `analysis/convergence_p10.png`
- `analysis/convergence_p20.png`
- `analysis/convergence_note.txt`

New:

- `analysis/P5_control_review.md`
- this handoff document

Left separate and unchanged in interpretation:

- `analysis/learned_vs_raw_p5.md`

## Sanity Checks

`all_metrics.csv` now has 7488 rows:

- P2 safe: learned 1008, ricker 288
- P5 safe: learned 1008, ricker 288, raw 1008
- P10 safe: learned 1008, ricker 288
- P10 yield: learned 1008, ricker 288
- P20 safe: learned 1008, ricker 288

The regenerated `DECISION_psafe.md` uses only the comparable policy subset for safe-mode penalty selection. It reports 1008 safe rows at every penalty:

| P | comparable safe rows |
|---|---:|
| 2 | 1008 |
| 5 | 1008 |
| 10 | 1008 |
| 20 | 1008 |

The heuristic still picks `collapse_penalty = 5`:

- Amur tiger true_return by P: 2.84, 2.17, 0.85, -2.51
- Puerto Rican parrot true_return by P: 2.57, 0.62, -2.29, -8.25
- Egyptian vulture remains penalty-dominated at every P

The P5 control review says:

- best general learner beats best ecological baseline on 96/112 recoverable cells
- best general learner loses on sink cells, 6/32 wins only
- OGSRL is the strongest single general method
- learned filtering does not materially improve control return versus raw observations

## Intended Interpretation

Use `DECISION_psafe.md` for the P_safe penalty decision.

Use `learned_vs_raw_p5.md` for the raw-vs-learned filter ablation.

Use `P5_control_review.md` for the quick table answering whether general learners beat the two ecological baselines.

Do not mix the P5 raw rows into cross-penalty P_safe selection, because raw rows exist only at P5.

## Claude Audit Checklist

Please audit:

1. `analyze_psafe.py` now applies the intended filter policy for P_safe selection.
2. `DECISION_psafe.md` excludes P5 raw rows and has equal comparable safe-row counts across penalties.
3. `P5_control_review.md` values match `all_metrics.csv` under the documented policy: general learners learned, PLUS/MOOR ricker.
4. The learned-vs-raw null result remains separate from the P_safe choice.
5. The conclusion is framed correctly: P5 is an average-case lock with a negative-tail caveat, general learners help recoverable populations, sinks remain separate, and learned filtering has no control-return gain.

## Not Done

I did not fix the live `manifest.py:201` CLI aggregate bug in this cleanup.

I did not run calibration gates.

I did not add action logging or run the sink action-preference diagnostic.

I did not run GPU/PLUS or pooled-agent experiments.

