# Evaluation protocol

Primary implementation:
`src/tracks/general/real_ecology_benchmark/evaluator.py`,
`src/tracks/general/real_ecology_benchmark/pipeline.py`, and
`src/tracks/general/real_ecology_benchmark/envs.py`.

## Trace

`pipeline.run_method()` constructs `ContinuousEvaluator` and calls
`ContinuousEvaluator.run(policy)`. For every configured block seed and four
local episodes, the evaluator:

1. computes `seed=block_seed+local_episode`;
2. creates a fresh environment and resets it; real reset is deterministic
   `s0=N0`, while observation and initial regime use seeded RNG streams;
3. creates a fresh filter, resetting it with `seed+10_000`;
4. resets policy RNG with `seed+20_000`;
5. asks `policy.act(belief,observation)`, replacing specified exceptions or
   invalid actions with action 0;
6. steps the environment, gives the policy only `PublicTransition`, then updates
   the filter;
7. accumulates discounted public and private returns and private-truth metrics;
8. stops at termination or `min(evaluation.horizon,environment.horizon)`;
9. appends one episode row.

`summarize()` computes mean and sample SD (`ddof=1`), and `save()` writes
`episodes.csv` and `summary.json`.

There is no generic model checkpoint loader. The “checkpoint” boundary is the
in-memory fitted policy returned by `pipeline.build_method()`; adapted replay
may reconstruct that object from hash-keyed faithful fit caches before passing
it to `ContinuousEvaluator.run()`. Metrics flow from the fitted object through
fresh `env.step()` calls into episode rows, then `summarize()` into
`summary.json`.

Accepted config uses five block seeds `[7001,7051,7101,7151,7201]`, four
episodes each, horizon 50, discount .95. The realized seeds are 7001–7004,
7051–7054, etc. The policy may be deterministic given its reset seed, but
planners use seeded Monte Carlo/MCTS and the environment has survey noise and,
for regime cells, switching. There is no exploration switch; any stochastic
action search is part of the deployed algorithm.

Evaluation uses the same `EnvironmentConfig` class/values and simulator code as
collection, but a new env instance and fresh seeds. Scientifically this is
in-distribution simulator evaluation, not field validation or domain transfer.
Collection episode seeds are random draws from RNG 116 and are not recorded
publicly, so seed noncollision cannot be proved from arithmetic alone. Belief
cache seed 30116 and policy construction seed 40116 differ from evaluation
seeds; per-episode filter/policy offsets are 10k/20k.

## Fallback ambiguity

`evaluator.py` catches `FloatingPointError`, `ValueError`, and `RuntimeError`
from `act()` and uses action 0; an out-of-range action does the same. It increments
`fallback_count`, also adding `last_diagnostics["hard_fallback"]`. This can
silently turn a broken policy into do-nothing. Episode rows distinguish it from
a genuine a0 selection only through `fallback_count`; the accepted 144-cell CSV
does not contain this field, so external accepted episode/summary artifacts are
needed to audit it.

Private truth supplies collapse entries, unsafe/MVP occupancy, persistence,
economic cost (action table), true-state extrema, filter calibration error, and
true return. Only `PublicTransition` and the observation-derived filter belief
reach the policy, so `evaluator.py` does not feed those metrics back.

`run_oracle_state_ablation()` trains a normal full-information policy, then
uses `OracleStateFilter` only during evaluation. It is a state-observation
ceiling, not a hidden result. `gate.run_decision_gate()` compares true-family
belief filtering, Ricker-misspecified filtering, and `ExactEpisodeProposal`
clairvoyance. The latter receives fixed true episode parameters and point-state
beliefs, so it is a ceiling.

## Offline fit versus online evaluation

| | offline fit | online evaluation |
|---|---|---|
| data | one 4,000-row simulator dataset | 20 fresh simulator episodes |
| policy information | public dataset, sanitized context/cache | observation history, belief, public transition |
| private truth | prohibited | evaluator metrics only |
| reward | EVD logged reward; others often fitted surrogate | exact environment true-next-state reward |
| randomness | collection seed, split/cache/model seeds | listed episode seeds plus 10k/20k offsets |
| updates | one fit/model/solver construction | no fit or parameter update; `observe()` may update online belief/posterior |
| writes | caches, fit artifacts, training history | episode rows and aggregate summary |
