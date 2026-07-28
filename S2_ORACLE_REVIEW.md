# Independent read-only line review: S2 oracle solver

Date: 2026-07-29  
Repository reviewed: `/home/hphung/ce25_scratch2/DeepRL_Population_Models`  
Review mode: read-only. The only review artifacts created are this report and `.review/`.

## Verdict

**The requested fox transfer-policy return differences are reproducible and trustworthy as
descriptive rollout differences. The headline fox VPI is not trustworthy as VPI.**

The independent reconstruction reproduces the archived 321-bin fox numbers to floating-point
precision, and both repository entry points reproduce the oracle CSVs byte-for-byte. However,
the reported `best_common_policy_value` is only the best of five hand-picked candidates (the
four family-oracle policies and a majority vote), not the optimum over family-agnostic policies.
For fox, a simple admissible common policy closes the entire reported gap: all four family
oracles choose action 6 initially; that action's exact next abundance uniquely identifies the
family for every seed; the policy can then follow the matching family oracle. Its independently
evaluated value is `11.839420699045172`, equal within roundoff to the reported perfect-family-
information value `11.839420699045174`. Thus the reported `0.127927936351867` is candidate-set
shortfall, not value of perfect family information.

## Findings

### BLOCKER — the VPI baseline is not a genuine optimum

`run_s2.py:457-470` forms five candidates:

1. the Ricker oracle transferred across all four truth families;
2. the Allee oracle transferred across all four;
3. the theta oracle transferred across all four;
4. the regime oracle transferred across all four; and
5. a per-state majority vote among those four oracles.

It then uses `max(candidates)`. There is no optimization over common policies, belief states,
history-dependent policies, or even all Markov policies. `run_s2_convergence.py:43-48` repeats
the same restriction.

This restriction is material, not hypothetical. At 321 bins, independently evaluated fox
values are:

| common-policy candidate | value |
|---|---:|
| Ricker oracle | 11.556467395111790 |
| Allee oracle | 11.619849959160430 |
| theta oracle | 11.708562879560294 |
| regime oracle | 11.591620788982330 |
| majority vote | 11.711492762693302 |
| **simple one-step-revelation common policy** | **11.839420699045172** |
| perfect-family-information average | 11.839420699045170 (independent) |

The common policy uses action 6 at `t=0`, exactly as each family oracle does. Across the 20
seeds, the smallest pairwise separation between the four deterministic next abundances is
`0.018230457856909`, so the true next state uniquely reveals the family. This information is
available under S2's own evaluator information structure: `evaluate()` selects actions from
`env.state` (`run_s2.py:338-353`), not from a noisy observation. From `t=1` onward the common
policy follows the corresponding oracle. Direct rollout, rather than interpolated Bellman
values, gives the value above.

The reported fox VPI therefore fails invariant 3. Under the solver's deterministic,
true-state-observed information structure, the demonstrated VPI is zero to numerical
precision, not `0.127928`.

### BLOCKER — the reported “regret” has the opposite sign, and oracle dominance is not universal

The stated invariant defines

`R[F,G] = V(pi*_G in G) - V(pi*_F in G) >= 0`.

The implementation instead computes `values[F,G] - values[G,G]`
(`run_s2.py:387-403`; convergence script `:51-60`). Consequently the requested fox row is
stored as negative transfer-minus-own differences. The magnitudes are reproducible, but the
field named `mean_regret` has the opposite sign from the stated definition.

This is not only a naming issue at 41 bins. Approximate grid policies can outperform the
nominal own-family oracle when both are rolled out in continuous dynamics. In the archived
41-bin fox matrix, Allee-policy-in-theta minus theta-own is `+0.046800128924817`; under the
stated regret definition this is `-0.046800128924817`, violating nonnegativity. Tiger has
analogous `+0.007531097846455` entries. The diagonal is exactly zero only because the code
explicitly overwrites it (`run_s2.py:391-394`).

The fox Ricker-row loss conclusion itself is robust in sign after applying the stated
convention: Ricker transfer is worse than each non-Ricker own policy at 321 bins. But the CSV
schema/sign and the blanket “all off-diagonals nonnegative” claim must not be accepted.

### SHOULD-FIX — “exact oracle” is exact only for the discretized/interpolated model

The recursion is a correct 50-stage backward induction on its grid, but continuous rollouts
choose actions by interpolating each action's grid Q surface. This need not preserve the
grid-point dominance ordering and explains transferred policies occasionally beating the
nominal own oracle. The only dominance assertion checks 11 constant-action policies
(`run_s2.py:406-425`), which is much weaker than checking against transferred policies or a
continuous-state optimum.

Receipts and prose should call this a grid/interpolation oracle rather than an unqualified
exact full-information optimum, or add a rollout-level optimality bound.

### NICE-TO-HAVE — clean-repo receipt does not disclose missing visitation support

The default replay root is `.verification/diagnostic_replay` (`scripts/diagnostics/repo_paths.py:
75-79`). It is absent in this clean checkout. `visitation_support()` silently returns no rows
when its glob finds nothing (`run_s2.py:145-170`), so the clean rerun's non-oracle
`identifiability.csv` omits `accepted_policy_visitation` rows and is not byte-identical to the
archived file. The oracle/regret/VPI outputs are unaffected and do reproduce exactly.

The receipt still has the same structure (apart from elapsed time). It should report missing
optional sources and input hashes/counts so a partial S2 rerun cannot look complete.

## Bellman recursion and environment audit

**Pass, subject to the grid-approximation qualification above.**

- Horizon and terminal condition: `q` has 50 stages; the loop runs `t=49,...,0` with
  `V_50=0` (`run_s2.py:313-327`). Evaluation runs `t=0,...,49` and applies `gamma^t`
  (`:338-354`). There is no horizon off-by-one.
- Discount: continuation uses `0.95 * V_{t+1}` and rollout uses `0.95^t`; terminal reward is
  undiscounted relative to its stage and has no continuation.
- Reward: planning uses the true deterministic `x'`,
  `x'/(x'+K_ref) - cost - 10*1[x'<=s_safe]` (`run_s2.py:322-326`). The `<=` boundary agrees
  with `safety_penalty_indicator()` (`reward.py:75-95`) and the environment computes the
  reward from true `state_next` (`envs.py:322-329`).
- Terminal zero state: environment reset/step makes exact zero terminal. The solver zeros
  reward and continuation at grid state zero (`run_s2.py:324-325`), consistent with the
  terminal convention even though action 10 otherwise has direct stocking.
- Dynamics: all four maps match `ContinuousEcologyEnv.transition_value()` including
  `r_pos=max(r_eff,0)`, unconditional `exp(r_mort)` for a negative set point, theta growth,
  Allee threshold, and regime threshold/multiplier (`envs.py:229-269`;
  `run_s2.py:265-292`).
- Capacity: 9 linear capacity bins span `[K_base,K_max]`; action `delta_K` accumulates and
  clips identically in planning and evaluation.
- Interpolation: linear abundance interpolation via `np.interp`, then linear interpolation
  between adjacent capacity bins, with endpoint clamping (`run_s2.py:295-302`). The abundance
  grid is zero plus log spacing through `1.5*K_max` (`:260-262`).
- Actions: the resolved real-ecology table contains all 11 contiguous actions. Planning loops
  over `len(actions)` and Q lookup loops over the resulting last dimension. Independent input
  loading from `configs/ecology/action_effects_long.csv` also found exactly 11 actions for
  each reviewed population/family.

## Regret construction and common randomness

**Rollout and common-seed construction pass; sign/nonnegativity reporting fails as above.**

- A transferred `pi*_F` is planned once under F and passed unchanged to `evaluate()` under G
  (`run_s2.py:376-385`). `evaluate()` performs action selection from that fixed Q policy; it
  does not call the planner. This is rollout, not replanning.
- Each matrix cell for a seed uses the same integer seed for the F policy's episode draw and
  G evaluation (`:370-385`). The 20 seeds match the receipt.
- The diagonal is forced to bit-exact `0.0` (`:391-394`).
- With the stated own-minus-transfer definition, the requested 321-bin fox Ricker
  off-diagonals are positive `0.357989365`, `0.179763141`, and `0.594060710`. The archived CSV
  stores their negatives under its reversed convention.

## Per-episode draws

**Pass for consistency, with an important interpretation note.**

`realized()` resets a fresh environment with the episode seed and records C, theta, and the
entire 50-step regime path (`run_s2.py:240-257`). `backward_oracle()` uses that realized draw
at every Bellman stage. Evaluation resets the target-family environment with the same seed;
independent reconstruction of NumPy's spawned parameter/regime RNG streams confirms identical
C/theta values, initial regime, and transition sequence.

Thus there is no matrix inconsistency: policy F and truth G are paired by seed, and each
family's own oracle and evaluation use the same episode draw. This is a highly privileged
oracle—the future regime path and episode parameters are effectively known—but it is applied
consistently. The common-policy candidates are also seed-specific, so they receive the same
within-episode parameter knowledge; the failed VPI comparison is not caused by unequal draw
handling.

## Independent 321-bin recomputation

The independent implementation is `.review/independent_s2.py`. It imports no S2 or benchmark
module. It reads only the species/action CSV tables, independently reconstructs NumPy seed
streams, implements the four maps, performs backward induction, and directly rolls policies
out for 50 discounted stages.

| quantity | archived receipt/CSV | independent | difference |
|---|---:|---:|---:|
| fox restricted-candidate VPI | 0.127927936351867 | 0.127927936351869 | +1.78e-15 |
| fox Ricker in Allee minus Allee own | -0.357989365157204 | -0.357989365157202 | +1.78e-15 |
| fox Ricker in theta minus theta own | -0.179763140770484 | -0.179763140770483 | +3.61e-16 |
| fox Ricker in regime minus regime own | -0.594060709805843 | -0.594060709805840 | +3.00e-15 |

This reproduces the requested rounded values: `0.127928`, `-0.357989`, `-0.179763`,
and `-0.594061`.

## Grid convergence and tiger floor

Independent recomputation (`.review/independent_convergence.py` plus the independent 321-bin
run) reproduces the archived restricted-candidate VPI series:

| bins | fox | tiger |
|---:|---:|---:|
| 41 | 0.111657763910475 | -0.005648323384837 |
| 81 | 0.104725034419817 | 0.000259088187751 |
| 161 | 0.127975178216939 | 0.000073857174286 |
| 321 | 0.127927936351869 | 0.000095423809555 |

The tiger `-0.00565` is a coarse-grid/interpolated-policy numerical artifact, not a sign error
in `perfect - common`: at 41 bins the transferred theta candidate rolls out slightly better
than the approximate own-family policies; the value becomes nonnegative and about `1e-4` at
all finer grids. The 161-to-321 changes are `4.72419e-5` for fox and `2.15666e-5` for tiger.
The convergence computations are real and reproducible, although they converge the restricted
candidate metric, not genuine VPI. Visited-action flip fractions remain `7.9%` (fox) and `3.0%`
(tiger) from 161 to 321, so value convergence should not be described as complete policy
stability.

## Repository regression rerun

Commands were run with `DEEPRL_S2_OUTPUT` pointing to `.review/repo_rerun/`:

```text
python scripts/diagnostics/followups/run_s2.py
python scripts/diagnostics/followups/run_s2_convergence.py
```

Results:

- `regret_matrix_fox.csv`, `regret_matrix_tiger.csv`, `vpi.csv`, `grid_stability.csv`,
  `grid_convergence_vpi.csv`, and `grid_convergence_ricker_row.csv` are byte-for-byte
  identical to the archived files.
- Both rerun receipts equal the archived JSON after removing only `elapsed_seconds`.
- Therefore the standalone repoint has not changed the oracle numbers.
- As noted above, `identifiability.csv` is not identical because the clean repository lacks
  the optional accepted-policy replay logs; this does not enter the oracle calculation.

## Exact paths located

- Main solver: `scripts/diagnostics/followups/run_s2.py`
- Convergence solver: `scripts/diagnostics/followups/run_s2_convergence.py`
- Shared path module: `scripts/diagnostics/repo_paths.py`
- Config/manifest bridge: `scripts/ecological/run_real_manifest_row.py`
- Environment and family maps:
  `src/tracks/ecological/real_ecology_benchmark/envs.py`
- Control/set-point mapping:
  `src/tracks/ecological/real_ecology_benchmark/controls.py`
- Reward:
  `src/tracks/ecological/real_ecology_benchmark/reward.py`
- Action resolution:
  `src/tracks/ecological/real_ecology_benchmark/actions.py`
- Configuration:
  `src/tracks/ecological/real_ecology_benchmark/config.py`
- Species table: `configs/ecology/species.csv`
- Action-effects table: `configs/ecology/action_effects_long.csv`
- Action-name table: `configs/ecology/actions.csv`
- Archived S2 directory: `results/followups/S2/`
- Main receipt: `results/followups/S2/S2_RECEIPT.json`
- Convergence receipt: `results/followups/S2/S2_GRID_CONVERGENCE_RECEIPT.json`
- Fox regret CSV: `results/followups/S2/regret_matrix_fox.csv`
- Tiger regret CSV: `results/followups/S2/regret_matrix_tiger.csv`
- VPI CSV: `results/followups/S2/vpi.csv`
- Convergence VPI CSV: `results/followups/S2/grid_convergence_vpi.csv`
- Convergence Ricker row: `results/followups/S2/grid_convergence_ricker_row.csv`

The scripts' default live output is `.verification/followups/S2/`, not
`results/followups/S2/`; the latter is the archived result location in this checkout.
