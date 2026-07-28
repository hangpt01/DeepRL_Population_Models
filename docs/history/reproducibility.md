# Reproducibility and compute

All random components receive explicit NumPy generators. Environment parameters, regime switches, process noise, observation noise, initial state, collector actions, filters, models, and planners use separate or reproducibly derived seeds.

Public dataset metadata records the complete environment configuration, action-table hash, seed, behavior information set, episode length, and actual transition count. The pipeline refuses to reuse a cached dataset for a mismatched environment/action/noise cell.

Shared datasets, learned filter proposals, and offline belief caches are written atomically. Manifest workers use a lock around dataset creation so concurrent method rows cannot corrupt shared artifacts.

Evaluation records filter and planner wall time separately and uses paired environment seeds across methods. The full manifest has 384 rows: 336 shared learned/raw rows and 48 Ricker-filter PLUS/MOOR baseline-fidelity rows. The reference implementation is CPU/NumPy; production users may replace the proposal, dynamics ensemble, and planner backends with accelerated implementations while retaining the public interfaces and tests.

Manifest workers write a private-channel calibration artifact and stop before training when healthy-start incident collapse is outside `[0.15, 0.24]`. `--allow-uncalibrated` exists only for explicitly labeled diagnostics. Required Allee/regime gate artifacts remain a separate precondition.

Recommended sequence:

1. run `make test` and `make smoke`;
2. generate and inspect one 75k-transition cell;
3. run calibration and the three-controller gate;
4. run one-cell all-method belief-filter pilot;
5. launch the belief matrix;
6. launch raw and method-proposal ablations;
7. aggregate only after PLUS and MOOR are complete for every compared cell.

If an expensive method exceeds its predeclared wall budget, report it as a failed headline row. A reduced BA-MCTS subset is a separately labeled scalability experiment, not a replacement with fewer evaluation episodes.
