# Diagnostic replay: implementation and execution specification

Date: 26 July 2026
Purpose: close §11 of `UNCERTAINTY_TAXONOMY_AND_STAGED_DIAGNOSTIC_PLAN.md`.
Prepared for the code-server agent. Two parts: a zero-compute extraction pass
(Part 0) and an instrumented replay (Part 1).

---

## Standing rules

1. **The accepted result is immutable.** `MATCHED_P10_144_METHOD_CELLS.csv`
   (SHA `7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c`) and
   `MATCHED_P10_144_RECEIPT.json` remain the sole controlling artifacts. Nothing
   in this work supersedes, replaces, or re-ranks them.
2. **New root, new identity.** Write everything under a new immutable root, e.g.
   `real_ecology_runs/three_species_diagnostic_replay_20260726_v1/`, with its
   own manifest, config hashes and receipt. Register the method versions as
   `*_diagnostic_replay` so they can never be confused with the accepted rows.
3. **This is not a new experiment.** It is a *replay of already-accepted
   policies with logging added*. It opens no new scientific claim. No
   hyperparameter, reward, or method setting may be selected or modified using
   anything produced here.
4. **No refitting.** Reuse the 216 reward-independent fit-cache slots by exact
   hash (192 PLUS candidates + 24 MOOR models), as the accepted run did.
   `recomputed_fits` must be 0.
5. **Return-blindness does not apply and must not be claimed.** These returns
   are already open, and the parity gate in §1.4 deliberately uses them as the
   invariant to match. Say so explicitly in the receipt.

---

# PART 0 — Free extraction pass (no compute, no re-run)

Answers §11 items 4–11 from stored artifacts. Do this first; it may change the
scope of Part 1.

| ID | Task | Source |
|---|---|---|
| P0-1 | Dump the full 11-action table for **Amur tiger** and **Crab-eating fox**: `r_setpoint_ricker`, `r_setpoint_lgm`, `dK_step`, `dN`, `cost_step` | `action_effects_long.csv` |
| P0-2 | For each of the four general methods, identify **which reward signal the planner/actor consumes**: the shared `surrogate.predict`, or `dataset.rewards` directly. Give file+line per method | GEN code |
| P0-3 | **Surrogate fidelity.** For each of the 24 cells: R² and RMSE of the ridge reward fit on its training rows; then stratify the residuals by `true_next_state > s_safe` vs `≤ s_safe`, reporting count, mean signed error and RMSE in each stratum | `public_surrogate.py` + dataset |
| P0-4 | Per-cell `planner_seconds` (mean, total) for all six methods; and measured per-cell resources for the completed general run | `episodes.csv`; 576 `validity_receipt.json` |
| P0-5 | `observation_scale` for **all four families**, not only Ricker, for the three species | public context per cell |
| P0-6 | Confirm whether `observation_noise_sigma` is consumed by OGSRL, BA-MCTS and EVD (file+line, or NOT USED) | GEN code |
| P0-7 | Sampling point for the Allee threshold `C` and the theta exponent `θ`: per episode, per cell, or fixed at registration | `config.py`, `envs.py:159` |
| P0-8 | What threshold `danger_action_*_fraction` uses — true `s_safe` or a proxy | evaluator |
| P0-9 | For all 24 eco cells, dump the free CSV fields `pbvi_argmax_actions` and `pbvi_action_margin_min`, and compute per-cell **candidate unanimity** = fraction of candidates agreeing with the modal action | accepted CSV |
| P0-10 | For all 144 rows compute `economic_cost_mean / 50` and match against the per-species `cost_step` table to name the **deployed constant action** wherever the policy is constant | accepted CSV + P0-1 |

**P0-3 is the highest-value item in this document.** It sizes the
planner/evaluator objective mismatch, on which the blocking decision turns.

---

# PART 1 — Instrumented diagnostic replay

Answers §11 items 1–3, and yields several ablations at no extra compute.

## 1.1 Scope

**Tier A — adapted PLUS and adapted MOOR, 6 cells** (12 runs):

| # | Species | Family | σ | Why |
|---|---|---|---|---|
| A1 | Crab-eating fox | Ricker | 0.1 | In-family control on the discriminating species |
| A2 | Crab-eating fox | Allee | 0.1 | MOOR wins (11.711 vs 11.394) |
| A3 | Crab-eating fox | Regime | 0.2 | Largest PLUS margin (10.591 vs 10.142) |
| A4 | Crab-eating fox | Theta | 0.1 | A general method wins; MOOR saturated (SD 0.000, cost −5.000) |
| A5 | Amur tiger | Ricker | 0.1 | Constant policy despite 100× growth spread in the bank |
| A6 | Egyptian vulture | Ricker | 0.1 | Degenerate control — expect zero information; confirms the instrument |

**Tier B — all six methods, 3 cells** (reuse Tier A runs where they overlap):

| # | Species | Family | σ | Why |
|---|---|---|---|---|
| B1 | Crab-eating fox | Regime | 0.2 | = A3 |
| B2 | Amur tiger | Ricker | 0.1 | = A5; also the EVD collapse cell (−13.831, collapse-entry 1.000) |
| B3 | Amur tiger | Allee | 0.2 | The only cells where OGSRL wins (4.716) |

Do **not** extend to all 144. If Tier A shows the instrument works and the
findings are clear, stop.

## 1.2 Reuse and determinism

- Same dataset path and content hash per cell; same episode-preserving 80/20
  split; same 4,000-transition budget.
- Same 20 evaluation seeds: `7001–7004, 7051–7054, 7101–7104, 7151–7154,
  7201–7204`.
- Same `σ_obs`, horizon 50, γ = 0.95, `P = 10`, 11-action table, deterministic
  `N0`.
- **One numerical thread**, matching the accepted run
  (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`). A
  different thread count can change floating-point reduction order and break
  parity.
- **If solved PBVI policies / α-vectors are cached, load them.** Only replay the
  20 evaluation episodes. If they are not cached, re-solve deterministically and
  verify against the stored `pbvi_policy_diagnostics.npz['last_action_values']`
  and `pbvi_argmax_actions` before evaluating.

## 1.3 Side-effect-free logging — mandatory

Instrumentation must not perturb behaviour. Specifically:

1. **Never consume the environment or policy RNG streams.** Any diagnostic
   needing randomness uses a separate, independently seeded generator.
2. **Log by copying already-computed arrays.** Never recompute a quantity for
   logging, and never introduce a code path that changes the order of
   floating-point operations in the decision path.
3. **Write logs after the action is selected and applied**, never between a draw
   and its use.
4. Buffer in memory and flush per episode; do not interleave file I/O with the
   decision loop if that could alter timing-dependent behaviour (it should not,
   but keep it clean).

## 1.4 Parity gate — the run is void if this fails

Because `process_noise_sigma = 0.0` and all seeds are fixed, the replay should
reproduce the accepted run **exactly**. For every replayed (cell, method),
compare all seven accepted aggregate fields:

`return_mean`, `return_sd`, `unsafe_fraction_mean`, `persistence_mean`,
`collapse_entry_mean`, `min_population_mean`, `economic_cost_mean`.

- **Pass:** absolute difference ≤ 1e-9 on every field.
- **Investigate:** ≤ 1e-6 — report the field, the magnitude, and the suspected
  source (BLAS reduction order, thread count, library version) before continuing.
- **Fail:** > 1e-6 on any field. **Halt.** Do not analyse the logs. A failure
  means either the instrumentation perturbed the run or the accepted run was not
  deterministic; both are findings that must be reported before anything else.

Run **A6 (vulture × Ricker × 0.1) first as a single pilot**, verify parity, then
proceed. It is the cheapest cell and has a constant policy, so any perturbation
shows up immediately.

## 1.5 Per-step log schema

One row per `(cell, method, episode_seed, t)` for `t = 0..49`.

### Common columns (all six methods)

| Field | Notes |
|---|---|
| `cell_id, method, seed, t` | keys |
| `x_true_t`, `x_true_next` | latent true abundance before/after |
| `obs_t` | observation supplied to the policy |
| `action_t` | **§11 item 2** |
| `cost_t` | `cost_step(action_t)` |
| `utility_t` | `x_true_next/(x_true_next + K_ref)` |
| `penalty_flag_t` | `1[x_true_next ≤ s_safe]` |
| `reward_t` | `utility_t − cost_t − 10·penalty_flag_t` |
| `r_setpoint_t`, `r_pos_t`, `r_mort_t` | **critical**: `r_pos_t = max(r_eff,0)` tests the family-degeneracy claim directly |
| `k_t`, `m_t` | capacity and post-intervention abundance |
| `regime_z_t`, `regime_switched_t` | **§11 item 3** |
| `allee_C_cell`, `theta_exponent_episode` | realised draws |

### PLUS-only columns

| Field | Notes |
|---|---|
| `w_t[0..7]` | **§11 item 1** — full candidate posterior |
| `q_cand[j][a]` | 8 × 11 candidate action values at the current beliefs |
| `q_weighted[a]` | 11 posterior-weighted action values |
| `belief_mean[j]`, `belief_entropy[j]` | per-candidate state belief summary |
| `argmax_weighted` | the deployed action |
| `argmax_map_only` | argmax of `q_cand[0]` — **the MOOR-like counterfactual** |
| `argmax_uniform` | argmax under fixed `w = 1/8` — isolates *updating* from *averaging* |
| `argmax_cand[j]` | each candidate's greedy action |
| `margin_top1_top2` | on `q_weighted` |

### MOOR-only columns

`belief_mean`, `belief_entropy`, `q[a]` (11), `argmax`, `margin_top1_top2`.

### General-method columns (Tier B)

| Method | Fields |
|---|---|
| RefPlan | member posterior `w_m[0..4]`; best-sequence posterior-weighted mean return and return SD; `argmax` under `λ_ref = 0` (free ablation) |
| OGSRL | predicted `C_25` for the chosen action; safety budget; slack; both duals; OOD/guardian flag; fallback indicator |
| BA-MCTS | root visit counts per action; max depth reached; node count; root model belief |
| EVD | `Q̄[a]` (11); `Var_m Q[a]` (11); `argmax` under `λ_V = 0` — **the λ_V=0 ablation, free** |

### Free counterfactual ablations — and their honest limit

`argmax_map_only`, `argmax_uniform`, `λ_ref = 0` and `λ_V = 0` are computed from
action values that are already being calculated, so they cost nothing.

**They are one-step counterfactuals.** They answer "would the alternative rule
have chosen differently *at the states actually visited*", not "what return
would the alternative rule have obtained". A rule that diverges once may visit
entirely different states thereafter. Treat a non-zero switch fraction as
motivation for a full deployment run of that rule, and a zero switch fraction as
strong evidence the alternative is behaviourally identical **along this
trajectory distribution**.

### Optional, off by default

Repeated-search stability for BA-MCTS (two searches per decision, agreement
rate) **doubles compute and consumes RNG**. If enabled, it must use an
independent generator and its result must not influence the chosen action. Run
it only as a separate flagged arm, never in the parity-gated primary replay.

## 1.6 Derived metrics to compute from the logs

Report per (cell, method).

**M1 — posterior movement (PLUS).** Entropy `H_t = −Σ_j w_t^j ln w_t^j`
(max `ln 8 = 2.079`); effective candidate count `exp(H_t)`; total-variation
distance from uniform; `‖w_49 − w_0‖₁`; step at which `max_j w^j` first exceeds
0.5. *Answers: does the posterior move at all?*

**M2 — action-switch fractions (PLUS), over all 1,000 decisions per cell.**
- `switch_vs_MAP` = fraction where `argmax_weighted ≠ argmax_map_only`
- `switch_vs_uniform` = fraction where `argmax_weighted ≠ argmax_uniform`

The pair separates two mechanisms: **averaging** over candidates
(`switch_vs_MAP`) and **updating** the weights (`switch_vs_uniform`). If both
are ≈0, PLUS is behaviourally a point-model method on this trajectory
distribution — which explains equal PLUS/MOOR returns mechanically, without any
claim of algorithmic equivalence.

**M3 — candidate policy agreement.** Mean pairwise agreement over the 28 pairs
of `argmax_cand[j]`; fraction of steps with unanimous agreement; the modal
action's share.

**M4 — margins.** Distribution of `margin_top1_top2` (min, 5th percentile,
median); fraction below ε for preregistered ε ∈ {1e-3, 1e-2}. Compare with the
accepted per-cell minimum (`pbvi_action_margin_min`).

**M5 — centred vs raw disagreement.** For each step:
`raw = Var_j[q_cand[j][a]]` averaged over `a`;
`centred = Var_j[q_cand[j][a] − mean_a q_cand[j][a]]` averaged over `a`.
Report both and the ratio. **A large raw with a small centred value is the
signature of action-independent model disagreement** — models that differ about
the world but not about what to do.

**M6 — reward decomposition.** Per episode, the discounted sums of `utility_t`,
`cost_t`, and `10·penalty_flag_t` separately. Verify
`Σ_t γ^t reward_t` reproduces the accepted `return_mean`. Report the three
components as fractions of the total. *This settles §8.1 quantitatively.*

**M7 — family degeneracy.** Fraction of steps with `r_pos_t = 0`.
*Predicted: 1.000 for vulture (all methods), 1.000 for tiger PLUS/MOOR, and
substantially below 1 for fox PLUS/MOOR.* This is the direct test of the central
structural claim.

**M8 — depensation.** In Allee cells, fraction of steps with `m_t < allee_C`.
*Predicted: 0.000 for fox and for tiger under PLUS/MOOR.*

**M9 — regime.** Switches per episode (mean, SD); fraction of steps in the weak
regime; whether the action changed within one step of a switch.

**M10 — noise sensitivity.** Where both σ levels exist, per-step action
agreement between matched seeds at σ = 0.1 and σ = 0.2.

**M11 — EVD.** Switch fraction between `argmax(Q̄ − 0.1·Var)` and `argmax(Q̄)`;
and, on the tiger cell, the ranking of the recovery actions under `Q̄` — is the
collapse caused by the pessimism penalty or by the underlying conservative Q?

**M12 — OGSRL.** Fraction of steps where the safety constraint binds; mean
slack; comparison of `s_low` against the true `s_safe` (35.91 vs 25 for tiger,
31.90 vs 10.25 for fox, 22.93 vs 81.25 for vulture).

## 1.7 Outputs

```
{replay_root}/
  manifest.json                    # cells, methods, seeds, parent artifact hashes
  parity/PARITY_REPORT.json        # 7 fields × each (cell,method), abs diffs
  logs/{cell_id}__{method}.parquet # per-step schema of §1.5
  derived/{cell_id}__{method}.json # M1–M12
  DIAGNOSTIC_REPLAY_RECEIPT.json   # hashes, thread config, library versions,
                                   # recomputed_fits=0, parity verdict
  SUMMARY.md                       # M1–M12 tables across all replayed cells
```

Log volume is small: 1,000 rows per (cell, method) at ~120 floats for PLUS —
about 1 MB per run, well under 30 MB total.

## 1.8 Acceptance criteria

1. `recomputed_fits == 0`; all parent fit-cache hashes validated.
2. Parity gate passes at ≤ 1e-9 on all seven fields for every replayed run.
3. Every replayed episode has exactly 50 steps and 51 states.
4. Per-step `reward_t` reconstructs the accepted `return_mean` to ≤ 1e-9 (M6).
5. All logged actions are legal members of the 11-action table.
6. Deterministic repeatability: re-running one cell twice gives byte-identical
   logs.
7. No private field (`s_safe`, `K_ref`, family label) appears in any method
   input — only in the evaluator and in the logs.

## 1.9 Failure modes and what each would mean

| Observation | Meaning |
|---|---|
| Parity fails | Instrumentation perturbed the run, or the accepted run was nondeterministic. Halt and report — this is itself a finding about the accepted result. |
| M7 gives `r_pos = 0` in 100% of tiger PLUS/MOOR steps | The central structural claim is confirmed at cell level: the ecological policy switches off the family distinction. |
| M2 gives both switch fractions ≈ 0 | PLUS is behaviourally a point-model method here; the eight-candidate bank buys nothing on this trajectory distribution. |
| M5 gives large raw, small centred | Model disagreement is action-independent — the cleanest possible explanation for diverse candidates producing an identical policy. |
| M1 shows the posterior never leaves uniform | The candidate likelihoods are uninformative; sequential Bayesian discrimination is not operating. |
| M1 shows strong movement **and** M2 ≈ 0 | Inference succeeds and decisions are unaffected — the most interesting negative result available, and directly relevant to the family-uncertainty gate. |
| M6 shows the penalty term is 0 (fox, tiger) or a constant −184.6 (vulture) | Confirms §8.1: the safety term has no discriminating power. |

## 1.10 Compute estimate

From the accepted measured run: PLUS ≈ 4.26 core-h/cell, MOOR ≈ 0.54
core-h/cell (`102.15 / 24` and `12.98 / 24`), fitting excluded.

- **Evaluation-only replay with cached policies:** a small fraction of the
  above — likely under 5 core-h total.
- **Worst case, re-solving PBVI for all 6 Tier-A cells:** ≈ 29 core-h
  (PLUS) + 3 core-h (MOOR) + Tier B general methods ≈ 5 core-h → **under 40
  core-h**.

Either way this is small against the 115.13 core-h already spent. Run the A6
pilot first.

---

## What this replay does **not** establish

- It does not measure what an alternative decision rule would have *achieved* —
  only whether it would have chosen differently at the visited states (§1.5).
- It does not address `Var_data` or `Var_fit`; it is one dataset and one fit per
  cell, exactly as the accepted run was.
- It does not test any new method, reward, observability condition, or data
  budget. Those are stages S1–S9 of the staged plan.
- It cannot make the accepted returns more or less valid. It explains them.
