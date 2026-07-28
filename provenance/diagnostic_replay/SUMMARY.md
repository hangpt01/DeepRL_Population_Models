# P=10 diagnostic replay — final summary

This is an instrumented **REPLAY** of accepted P=10 policies with side-effect-free logging. It
supersedes nothing and re-ranks nothing. The accepted
`MATCHED_P10_144_METHOD_CELLS.csv`
(`7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c`) remains untouched and
controlling. Return-blindness is not claimed: accepted returns are deliberately used as the parity
invariant. No method, reward, or hyperparameter was changed.

## Completion and parity

All requested Tier A and Tier B cells passed the seven-field gate exactly:

| tier | job | cells / receipts | result | maximum absolute difference |
|---|---:|---:|---:|---:|
| A fleet | 58569801 | 10 | 10 PASS | 0.0 |
| A6 hardware checkpoints | 58548480, 58548481 | 2 | 2 PASS | 0.0 |
| B fleet | 58577745 | 12 | 12 PASS | 0.0 |
| **all receipts** | | **24** | **24 PASS** | **0.0** |

The seven checked fields were return mean, return SD, unsafe fraction, persistence, collapse entry,
minimum population, and economic cost. Every replay used 20 accepted seeds, 50 steps / 51 states
per episode, legal actions, and `recomputed_fits=0`. Tier A reward reconstruction was within
`1e-9`; deterministic repeatability and logger RNG/state non-interference checks passed.

Both arrays ran with one thread under `--constraint=xenon-8452Y` on m3h101 (Intel Xeon Platinum
8452Y). This constraint is parity-critical because m3h also contains AMD nodes. Same-hardware MOOR
and the login-node A6 checkpoint both reproduced exactly. Tier B started at 21:19:57 and its final
task completed at 22:09:55 Australia/Melbourne on 2026-07-27.

## Tier A diagnostics

- **A2 PLUS:** `switch_vs_MAP=0.709`. Raw candidate disagreement was
  `0.8956745867659512`, centred disagreement `0.10508362618136342`, and their ratio
  `0.1173234428374184`. Accepted PLUS return `10.966745450258198` was below MOOR
  `11.61093636068979`: posterior averaging changes actions frequently and is harmful in this cell.
- **A1 margin mass:** PLUS had 5.3% of decisions below a `1e-3` top-two margin, minimum
  `1.7615200921916596e-5`; MOOR had 4.0%, minimum `2.9402732702932255e-4`.
- **A5 return SD:** PLUS and MOOR were both `9.112518314015307e-16`, reported as `0.000`,
  despite per-episode theta redraw. This is exact accepted parity, not a replay discrepancy.
- **A6:** the prior sink-species findings remain: no positive intrinsic growth, posterior movement
  without action changes, candidate disagreement mainly in Q level rather than ranking, and large
  surrogate reward overpricing. These are diagnostics only and do not alter the accepted result.

## Tier B provenance and diagnostics

Tier B did **not** use the ricker-only Tier A snapshot. It used the separately frozen general-method
source at
`general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/code/src`,
with its own config, quarantined regime-hidden/reward-safe datasets, accepted public surrogate, and
learned public belief cache. Its source hashes differ from the ricker-only snapshot. The accepted
**non-faithful** general-method build path was preserved: episode-disjoint 80/20 data, deterministic
public-ensemble fitting during method construction, then the unmodified continuous evaluator.
Those public observation-space ensemble builds are not demographic-slot refits;
`recomputed_fits` remained zero.

- **M11 / EVD:** with `lambda_V=0`, action switches from mean-Q were B1 `0.001`, B2 `0.004`,
  and B3 `0.002`. In B2, recovery actions ranked by mean Q as
  `a5, a10, a6, a4, a3, a7, a9, a8`.
- **M12 / OGSRL:** predicted-safety bind fraction was zero in B1–B3; mean slack was
  `0.0961079411`, `0.0815994257`, and `0.0697238814`. Guardian override fractions were
  `0.000`, `0.995`, and `0.929`; hard-fallback fractions were `0.000`, `0.031`, and `0.184`.
  True safety thresholds in these logs are evaluator-only diagnostics, never policy inputs.
- **M14:** filtering-belief columns are present only for RefPlan. They are explicitly absent for
  OGSRL, BA-MCTS, and EVD, so those methods are not presented as if they shared the
  PLUS/MOOR/RefPlan filtering mechanism.
- **RefPlan:** the `lambda_ref=0` switch fractions were B1 `0.064`, B2 `0.055`, and B3 `0.074`.
- **BA-MCTS:** every root used 256 visits; maximum observed depth was 7. Mean tree-node counts were
  B1 `97.963`, B2 `93.363`, and B3 `108.544`.

## Constant-action reference sweep

The 264 Stage S1 rows are evaluator-only reference policies, never a seventh method and never part
of a ranking change. They show constant-policy headroom in 14/24 cells and expose the Egyptian
vulture sink-species control pathology, but they do not supersede or re-rank the six accepted
methods.

All requested replay work is complete. The machine-readable receipt contains exact source,
artifact, job, parity, and diagnostic details.
