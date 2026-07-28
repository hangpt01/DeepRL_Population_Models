# Phase 0 Hidden-r/K Results

Date: 2026-07-16

Status: completed, non-gating. No setting or transition budget was changed from
these results.

Command:

```bash
PYTHONPATH=src python scripts/run_hidden_rk_phase0.py --config configs/hidden_rk.yaml
```

The complete machine-readable result is written to
`outputs/hidden_rk/phase0/summary.json`. That output tree is generated and is not
the source of truth for code or configuration.

## Registered Conditions

- 4,000 target transitions per cell, preserving complete episodes.
- Episode length 25.
- Shared public surrogate version 2, fixed split seed 20,116.
- No tuning, threshold addition, row-budget increase, or method selection from
  these diagnostics.

## Results

| Cell | Actual rows | Episodes | Terminations | Risk fallback | Reward holdout RMSE | Reward holdout MAE | Low-tail RMSE |
|---|---:|---:|---:|---|---:|---:|---:|
| Amur tiger / private Ricker / sigma 0.0 / safe | 4,000 | 160 | 0 | `constant_single_class` | 2.0449 | 1.3566 | 5.1177 |
| Iberian lynx / private Allee / sigma 0.2 / safe | 4,000 | 160 | 0 | `constant_single_class` | 3.1982 | 2.6440 | 3.6655 |

Both cells had zero overshoot. The fitted constant one-step termination risk was
`1 / 3202 = 0.0003123` on the 3,200-row fitting fold, as prescribed by the
Beta(1,1) single-class fallback.

Evaluator-only reward diagnostics, which did not affect fitting:

| Cell | No-private-penalty RMSE | Private-penalty-applied RMSE |
|---|---:|---:|
| Amur tiger / Ricker | 1.6441 | 5.0518 |
| Iberian lynx / Allee | 2.5456 | 4.4733 |

## Interpretation

The public extinction label is inert in both representative cells, matching the
pre-registered expectation. Safe-mode results therefore remain
information-limited under the private safety objective; this is not evidence of
an intrinsic OGSRL safety failure. The reward surrogate has measurable but
imperfect holdout signal, particularly in the public low-reward tail and the
evaluator-only private-penalty stratum. These findings are report-only and do not
authorize post-hoc retuning.
