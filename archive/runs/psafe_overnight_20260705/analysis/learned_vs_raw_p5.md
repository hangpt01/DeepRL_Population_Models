# Learned-vs-raw filter ablation @ collapse_penalty=5 (safe mode)

Ran 2026-07-05 as a follow-up to the P_safe overnight grid. 1008 raw-filter rows
(all 9 pops × 4 families × 4 σ_obs × 7 methods, safe mode) against the frozen
snapshot, folded into `outputs/p5/` alongside the existing learned/ricker rows.
63/63 blocks, 0 failures.

## Result: learned filtering gives no material control-return advantage over raw

Safe-mode `true_return` (== operational return in real_setpoint mode), averaged
over the 9 pops × 4 families × 4 σ:

| method  | learned | raw   | unsafe_frac learned/raw |
|---------|---------|-------|-------------------------|
| bamcts  | -5.69   | -5.37 | 0.126 / 0.126 |
| delphic | -9.76   | -9.10 | 0.185 / 0.199 |
| moor    | -4.50   | -4.69 | 0.112 / 0.115 |
| mopo    | -5.45   | -5.42 | 0.121 / 0.122 |
| ogsrl   | -5.06   | -4.49 | 0.113 / 0.112 |
| plus    | -5.23   | -5.73 | 0.129 / 0.136 |
| refplan | -5.31   | -5.29 | 0.121 / 0.121 |

Differences are within run-to-run noise and not consistently in the learned
filter's favour.

## Noise robustness (where filtering should help most)

Mean safe `true_return` by σ_obs, learned vs raw (avg over methods/pops/families):

| σ_obs | learned | raw   | Δ(learned−raw) |
|-------|---------|-------|----------------|
| 0.0   | -5.19   | -5.22 | +0.03 |
| 0.1   | -5.62   | -5.20 | -0.42 |
| 0.2   | -5.64   | -5.43 | -0.21 |
| 0.4   | -6.99   | -7.05 | +0.06 |

Even at the noisiest setting (σ=0.4) the learned filter does not beat raw
observations on control return.

## Interpretation / caveats

- For **control performance** (return, unsafe fraction), learned filtering is not
  buying anything over raw observations in this real-ecology setting at P5.
- This does **not** say the filter is useless for **state estimation** — raw has
  no latent-state estimate, so a state-accuracy comparison (filter RMSE/coverage)
  is learned-only meaningful and is a different question from control return.
- **PLUS is filter-inert** (learned≡ricker identical, and raw ≈ same);
  **MOOR is filter-sensitive** (540/720 learned-vs-ricker cells differ).
- Honest framing for the paper: "learned filtering did not materially improve
  control outcomes vs raw observations at the locked penalty; we report it as an
  ablation rather than a headline gain." If a filter-value claim is still wanted,
  make it about **state-estimation accuracy**, not control return.

Data: `analysis/all_metrics.csv` (filter column), `analysis/rollup.csv`
(per penalty×mode×method×filter).
