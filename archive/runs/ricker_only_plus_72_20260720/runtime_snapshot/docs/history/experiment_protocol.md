# Experiment protocol

## Headline matrix

- dynamics: Allee-Ricker, theta-logistic, regime-switch;
- actions: 5 and 10;
- observation noise: `0`, `0.1`, `0.2`, `0.4`;
- methods: MOPO, RefPlan, BA-MCTS, PLUS, MOOR, Delphic-CQL, OGSRL;
- filter: learned shared PF primary, raw observation ablation, plus Ricker-filter baseline-fidelity rows for PLUS and MOOR;
- data: approximately 75,000 complete-trajectory transitions per cell;
- evaluation: five held-out seed blocks, 50 episodes each, maximum horizon 50, discount `0.95`.

Base Ricker is a recovery/control cell and is not included in the six-cell headline rate.

## Safety semantics

The common benchmark threshold is physical abundance 50 and is called `safety_threshold`, never `C`. The first transition from above to at/below the threshold incurs the one-time penalty. State is not snapped to zero and may recover. Exact zero is an absorbing terminal state. Initially unsafe episodes are separate from incident collapse and remain in unsafe occupancy.

## Gate

The hard gate compares observation-matched belief oracle and belief Ricker MPC using the same particle-MPC engine. A true-state/parameter clairvoyant oracle is diagnostic only. Allee and regime are hard-gated at approved primary noise levels. Theta is diagnostic; failed theta cells are negative controls.

## Reporting

Report operational return, true return, incident collapse, unsafe occupancy, filter RMSE/log-RMSE, action entropy, fallback count, and filter/planner wall time. Join filter error to return deltas. The beats-both rate compares each general method with `max(PLUS, MOOR)` using paired seeds and bootstrap intervals in downstream analysis.

The generated 384-row manifest consists of 336 shared-filter/raw rows plus 48 Ricker-filter PLUS/MOOR sensitivity rows. Aggregation never compares methods across filter labels; the Ricker rows diagnose baseline fidelity and do not silently replace the shared-filter primary comparison.

Do not compare operational returns across noise levels without the true-return channel: `LogNormal(0,sigma^2)` is median-one but mean-biased upward.
