# Scientific validity risks

## Code-level risk register

| risk | severity | evidence and why | verification / suggested fix / status |
|---|---|---|---|
| Surrogate objective mismatch | critical | Five accepted methods call `PublicRewardRiskSurrogate.predict()` while EVD fits logged rewards; all are scored by exact `envs.step()` reward (`public_models.py`, BA-MCTS/OGSRL, `faithful_pomdp.py`, EVD `_fit_member`) | extract every accepted summary’s holdout/tail and private-stratum errors; rerun H12/H14; report exact-reward oracle sensitivity. Open; copied CSV lacks full diagnostics |
| Truth in public reward | high | `collector.py` logs reward computed from true next state; metadata admits truth-derived signal | quantify conditional inversion given action/cost and learned scale; ablate reward from dynamics/belief features. Open |
| Privileged behavior | high | `collect_dataset()` passes `env.state` to `MixedDangerZonePolicy.act()`; `behavior_model.py` estimates actions from public belief features | compare with an observation-realizable collector and coverage/NLL. Open |
| Tuned collapse band | high | profile/start mixture is calibrated toward [.15,.24] in `collector.calibration_summary()` | regenerate several untuned profiles/seeds and repeat fits. Open; robust populations are exempt |
| Egyptian-vulture infeasibility | critical for pooled claims | N0=41, `s_safe=81.25`; `verify_constants.py` asserts max bound 42.937296 | `python scripts/verify_constants.py`; stratify/exclude infeasible cells. Confirmed. They are included in the 144 rows. Returns still differ by action/cost, so “every method scores identically” is refuted |
| Family degeneracy at nonpositive rates | critical | `envs.transition_value()` uses `r_pos=max(r_eff,0)` in every family; M7 reports high zero fractions in sampled replays | run M7 over all accepted cells and stratify results by informative steps. Confirmed mechanism; prevalence incomplete |
| Deterministic population dynamics | high | accepted configs inherit `process_noise_sigma=0`; only regime/survey are stochastic | explicit process-noise sensitivity and separate epistemic/aleatoric claims. Confirmed |
| Silent action-0 fallback | high | `evaluator.run()` catches three exception types and invalid actions | aggregate `fallback_count_mean/max` from all external summaries; fail accepted rows on any fallback. Unclear: accepted CSV omits field |
| Cache reuse across cells | critical | hidden dataset validation omits population/family/P/safety/process noise; belief cache validates nothing (`pipeline.py`) | deliberately copy a cache between cells; add dataset/config/code hash to every cache and validate. Open |
| Unequal compute | medium/high | BA-MCTS 8×256, EVD 20×35 fitted-Q iterations, OGSRL actor+256 deployment rollouts, and PBVI/cached LBFGS differ | aggregate `fit_seconds,row_seconds,planner_seconds` on the same machine; publish budgets. Unclear from copied CSV |
| Unreachable method hyperparameters | high | General `EnsembleValueDisagreementPolicy.__init__()` fixes ensemble size 20, CQL alpha .5, disagreement penalty .1 and 35 iterations; `OGSRLPolicy.__init__()` fixes safety budget .02, OOD budget .05 and 30 training iterations. `pipeline.build_method()` passes neither set, only `seed`. The accepted manifest therefore cannot tune or compute-match these values; `config.py:PlannerConfig` documents the same earlier hard-coded-budget failure class for BA-MCTS/OGSRL search settings | audit every registered policy constructor for scalar defaults absent from `ModelConfig`/`PlannerConfig`, route load-bearing values through validated config, and add resolved values to summaries/manifests before claiming matched tuning. Open |
| One dataset/fit, 20 episodes | critical for uncertainty claims | accepted manifests fix collection seed 116 and one fit; evaluator SD is episode-only | repeat collection and fit seeds; paired CIs/hierarchical analysis. Open |
| S2 oracle defects | mitigated | current `run_s2.py` uses `values[:,gi,gi]-values[:,fi,gi]` and 11-action feasible belief-MDP candidates | `make verify-standalone-s2`; current code fixes reviewed sign/baseline defects |
| Two-track comparison | high but mitigated | 14 path divergences; common env/evaluator/reward/actions byte-identical; accepted receipt hashes them | `diff -rq` + `scripts/verify_integrity.py`; retain shared-file hashes. Mitigated |
| Misnamed Delphic | high for attribution | general `delphic.py` aliases bootstrap ridge-Q EVD; accepted table names EVD | forbid `delphic` in new manifests/artifacts and describe as EVD only. Mitigated in accepted CSV |
| Accepted PLUS tests no four-family posterior | critical | accepted key is ecological Ricker-only, candidate family `ricker`, count 8 | state claim as within-Ricker uncertainty or run four-form faithful PLUS under matched protocol. Open |
| Holdout/leakage | medium | generic split is episode-disjoint; normalizers/surrogate spec fit on fit mask; policy fit receives train subset | test every method’s normalization and faithful internal split. Mostly mitigated; EVD terminal target uses `~dones` correctly |
| Discount/objective mismatch | medium | evaluator and planner use .95, but horizons differ (5-step planners, OGSRL 25, evaluator 50) | horizon sensitivity; do not call short-horizon objective identical to evaluator return. Open |

## Specific interpretations

The public reward formula is invertible in its smooth state term if K_ref,
penalty status and P are known:
`u=s'/(s'+K)` implies `s'=K*u/(1-u)`. Hidden `MethodContext` withholds K and
the private indicator, so exact row-by-row inversion is not directly available,
but reward remains an abundance/safety proxy learned by the surrogate.

The collector’s behavior is privileged while fitted behavior models see only
public belief features. Consequently, their action likelihood is an
approximation to a non-realizable public policy; coverage may be useful, but
standard logged-policy overlap interpretations are weakened.

For Egyptian vulture, the accepted table shows five distinct returns per
family/sigma, not identical values. PLUS and MOOR coincide in each checked cell,
while general methods differ largely through actions/cost/reward despite all
being unable to attain safety. These cells should not count as evidence that
family uncertainty improves recovery.

The accepted overall method means can be computed from the CSV, but averaging
across infeasible sink and recoverable species is descriptive only. No
cross-dataset statistical uncertainty accompanies it.

## Highest-priority invalidation points

The most plausible headline-invalidating issues are: accepted PLUS not actually
representing four-family uncertainty; surrogate-versus-scoring reward mismatch;
family degeneracy on deployed nonpositive-rate actions; infeasible sink cells;
and single-dataset/single-fit statistics. Cache-key gaps could additionally
invalidate individual cells if provenance paths were misrouted.

Smallest first checks:

```bash
diff -rq --exclude=__pycache__ src/tracks/ecological src/tracks/general
python scripts/verify_constants.py
make verify-integrity verify-negative-gate
make verify-cell-general
```

Then aggregate external `summary.json` fields for surrogate errors,
fallback counts, and timing before interpreting any ranking.
