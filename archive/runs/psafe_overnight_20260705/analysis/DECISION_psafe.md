# P_safe decision aid

Safe-mode metrics for the three decision populations, averaged over method/family/sigma on a filter-consistent subset.


Scope: general learners use `filter=learned`; ecological baselines `PLUS` and `MOOR` use `filter=ricker`. P5 `raw` rows are excluded from this penalty-selection table and kept as a separate ablation in `learned_vs_raw_p5.md`.


Comparable safe-row counts by penalty:

| P | rows |
|---|---:|
| 2 | 1008 |
| 5 | 1008 |
| 10 | 1008 |
| 20 | 1008 |

## Amur tiger

| P | unsafe_frac | mvp_frac | collapse_entry | true_return | final_state | min_state | econ_cost |
|---|---|---|---|---|---|---|---|
| 2 | 0.021 | 0.165 | 0.095 | 2.84 | 83.7 | 56.8 | 6.842 |
| 5 | 0.018 | 0.128 | 0.057 | 2.17 | 97.1 | 64.4 | 7.911 |
| 10 | 0.019 | 0.124 | 0.053 | 0.85 | 97.2 | 65.6 | 8.462 |
| 20 | 0.024 | 0.126 | 0.065 | -2.51 | 105.4 | 67.4 | 9.079 |

## Egyptian vulture

| P | unsafe_frac | mvp_frac | collapse_entry | true_return | final_state | min_state | econ_cost |
|---|---|---|---|---|---|---|---|
| 2 | 1.000 | 1.000 | 0.000 | -37.63 | 2.8 | 2.1 | 3.928 |
| 5 | 1.000 | 1.000 | 0.000 | -93.35 | 3.0 | 2.1 | 4.993 |
| 10 | 1.000 | 1.000 | 0.000 | -186.04 | 2.8 | 2.1 | 6.133 |
| 20 | 1.000 | 1.000 | 0.000 | -370.91 | 3.2 | 2.6 | 7.163 |

## Puerto Rican parrot

| P | unsafe_frac | mvp_frac | collapse_entry | true_return | final_state | min_state | econ_cost |
|---|---|---|---|---|---|---|---|
| 2 | 0.070 | 0.070 | 0.176 | 2.57 | 158.5 | 65.0 | 3.443 |
| 5 | 0.065 | 0.065 | 0.163 | 0.62 | 162.5 | 66.2 | 3.987 |
| 10 | 0.063 | 0.063 | 0.161 | -2.29 | 181.3 | 66.7 | 4.897 |
| 20 | 0.062 | 0.062 | 0.159 | -8.25 | 182.8 | 66.8 | 5.393 |

## Heuristic recommendation

- Amur tiger safe true_return by P: 2:2.84, 5:2.17, 10:0.85, 20:-2.51
- Puerto Rican parrot safe true_return by P: 2:2.57, 5:0.62, 10:-2.29, 20:-8.25
- Egyptian vulture is a demographic sink: unsafe_frac=1.0 and safe return scales ~linearly worse with P at every level, so it is penalty-dominated regardless of P (only translocation a10 helps).

**Heuristic pick: collapse_penalty = 5** (largest P keeping recoverable decision-pops' safe return >= 0 while still reducing collapse entries vs P=2). P=20 clearly over-penalises (recoverable returns go negative).

This is an AID, not an auto-committed choice. The penalty axis is fully materialised: the full grid exists at every P, so any choice you make is already backed by complete results.
