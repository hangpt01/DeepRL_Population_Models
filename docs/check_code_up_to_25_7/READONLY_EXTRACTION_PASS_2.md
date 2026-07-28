# Read-only extraction pass 2 — actions, reward signals, surrogate fidelity, compute, diagnostics

Date: 26 July 2026 (Australia/Melbourne). Read-only: no compute jobs launched, no re-run, no
modification of `MATCHED_P10_144_METHOD_CELLS.csv` (sha `7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c`),
its receipt, any frozen manifest, or any accepted artifact. Numbers in P0-3/P0-4/P0-9/P0-10 are computed
locally from the frozen artifacts (arithmetic/aggregation only). Items absent from the accepted artifacts
are marked **NOT FOUND**.

Sources: **ECO** = `real_ecology_runs/ricker_only_plus_72_20260720/runtime_snapshot_routing_fix/src/real_ecology_benchmark/`;
**GEN** = `general_rl_phase2_iso/…/general_phase2e_full_sigma01_02_20260720_v1/code/src/real_ecology_benchmark/`;
**P10** = `real_ecology_runs/three_species_ecological_p10_correction_20260723_v1/`;
**GENRUN** = `general_rl_phase2_iso/…/general_phase2e_full_sigma01_02_20260720_v1/`;
action table = `real_ecology_data/action_effects_long.csv`.

---

## P0-1. Full 11-action table (Amur tiger, Crab-eating fox)

Source: `action_effects_long.csv`. `cost_step` is **identical across all species** (verified):
`[0.0, −0.05, −0.1, 0.25, 0.5, 0.1875, 0.5, 0.4375, 0.75, 1.0, 0.3125]`.

**Amur tiger**

| action | r_setpoint_ricker | r_setpoint_lgm | dK_step | dN | cost_step |
|---|---|---|---|---|---|
| a0 | −0.0458 | −0.0448 | 0.0 | 0.0 | 0.0 |
| a1 | −0.1857 | −0.1695 | 0.0 | 0.0 | −0.05 |
| a2 | −0.4753 | −0.3783 | 0.0 | 0.0 | −0.1 |
| a3 | 0.0275 | 0.0279 | 0.0 | 0.0 | 0.25 |
| a4 | 0.0707 | 0.0733 | 0.0 | 0.0 | 0.5 |
| a5 | −0.0458 | −0.0448 | 25.0 | 0.0 | 0.1875 |
| a6 | −0.0458 | −0.0448 | 75.0 | 0.0 | 0.5 |
| a7 | 0.0275 | 0.0279 | 25.0 | 0.0 | 0.4375 |
| a8 | 0.0275 | 0.0279 | 75.0 | 0.0 | 0.75 |
| a9 | 0.0707 | 0.0733 | 75.0 | 0.0 | 1.0 |
| a10 | −0.0458 | −0.0448 | 0.0 | 20.0 | 0.3125 |

**Crab-eating fox**

| action | r_setpoint_ricker | r_setpoint_lgm | dK_step | dN | cost_step |
|---|---|---|---|---|---|
| a0 | 0.323 | 0.3813 | 0.0 | 0.0 | 0.0 |
| a1 | 0.2098 | 0.2334 | 0.0 | 0.0 | −0.05 |
| a2 | −0.0131 | −0.013 | 0.0 | 0.0 | −0.1 |
| a3 | 0.3779 | 0.4592 | 0.0 | 0.0 | 0.25 |
| a4 | 0.4418 | 0.5555 | 0.0 | 0.0 | 0.5 |
| a5 | 0.323 | 0.3813 | 4.1 | 0.0 | 0.1875 |
| a6 | 0.323 | 0.3813 | 12.3 | 0.0 | 0.5 |
| a7 | 0.3779 | 0.4592 | 4.1 | 0.0 | 0.4375 |
| a8 | 0.3779 | 0.4592 | 12.3 | 0.0 | 0.75 |
| a9 | 0.4418 | 0.5555 | 12.3 | 0.0 | 1.0 |
| a10 | 0.323 | 0.3813 | 0.0 | 4.1 | 0.3125 |

(dK_step = 0.10/0.30·K_base; dN(a10) = 0.10·N0.)

---

## P0-2. Reward signal consumed by each general method (all 144 cells are `expose_rk=hidden`)

| method | reward signal | file : line |
|---|---|---|
| refplan | **shared `surrogate.predict`** (hidden path uses `PublicParticlePlanner`) | `GEN/methods/refplan.py:60-62,97` → `GEN/public_models.py:145 class PublicParticlePlanner`, `:268,366 reward, risk = self.surrogate.predict(...)` |
| ogsrl | **shared `surrogate.predict`** | `GEN/methods/ogsrl.py:776 self.surrogate = self.public_context.surrogate`, `:607 reward, _ = self.surrogate.predict(...)` |
| bamcts | **shared `surrogate.predict`** | `GEN/methods/bamcts.py:119` (comment "public reward is produced by the *shared* surrogate"), `:162 reward, _risk = self.public_context.surrogate.predict(...)` |
| ensemble_value_disagreement_pessimism (EVD) | **`dataset.rewards` directly** (bootstrap fitted-Q) | `GEN/methods/ensemble_value_disagreement.py:84 rewards = dataset.rewards[rows]` |

For comparison, the ecological PBVI methods also consume the shared surrogate:
`ECO/faithful_pomdp.py:209-229 expected_public_reward` → `:221 surrogate.predict(...)`.
**So five of six methods (RefPlan, OGSRL, BA-MCTS, PLUS, MOOR) optimize the linear public surrogate reward;
EVD is the only method trained on the raw logged `dataset.rewards`.**

---

## P0-3. Surrogate reward fidelity (highest priority) — per cell, on training rows, stratified by true s_safe

Reconstructed from the frozen surrogate (`public_surrogate.py:199-223 predict`, coefficients fit at `:299-302`)
and the cell's `public.npz`; residual = prediction − `dataset.rewards`; fit rows via
`_episode_split(dataset, surrogate.split_seed)`; stratified by private `truth.npz safety_penalty_applied`
(= occupancy indicator `true_next_state ≤ s_safe`; s_safe = tiger 25.0, fox 10.25, vulture 81.25).
**Validation:** my RMSE_fit exactly matches stored `diagnostics.reward_fit_rmse` (tiger/ricker 1.17163=1.17163;
vulture/ricker 0.90954=0.90954).

`me` = mean signed error (prediction − actual); positive ⇒ surrogate over-predicts reward.

| cell | R²(fit) | RMSE(fit) | n_safe | me_safe | rmse_safe | n_unsafe | me_unsafe | rmse_unsafe |
|---|---|---|---|---|---|---|---|---|
| tiger/ricker/0.1 | 0.498 | 1.172 | 2845 | −0.309 | 0.861 | 355 | +2.476 | 2.535 |
| tiger/ricker/0.2 | 0.488 | 1.183 | 2845 | −0.314 | 0.863 | 355 | +2.520 | 2.579 |
| tiger/allee/0.1 | 0.561 | 1.340 | 2627 | −0.438 | 1.105 | 573 | +2.009 | 2.106 |
| tiger/allee/0.2 | 0.553 | 1.352 | 2627 | −0.445 | 1.109 | 573 | +2.040 | 2.138 |
| tiger/theta/0.1 | 0.505 | 1.092 | 2897 | −0.263 | 0.792 | 303 | +2.517 | 2.569 |
| tiger/theta/0.2 | 0.490 | 1.109 | 2897 | −0.271 | 0.793 | 303 | +2.589 | 2.641 |
| tiger/regime/0.1 | 0.563 | 1.402 | 2555 | −0.493 | 1.180 | 645 | +1.952 | 2.056 |
| tiger/regime/0.2 | 0.556 | 1.412 | 2555 | −0.500 | 1.184 | 645 | +1.979 | 2.083 |
| fox/ricker/0.1 | 0.461 | 0.328 | 3186 | −0.021 | 0.076 | 14 | +4.827 | 4.827 |
| fox/ricker/0.2 | 0.454 | 0.330 | 3186 | −0.021 | 0.077 | 14 | +4.858 | 4.858 |
| fox/allee/0.1 | 0.681 | 1.049 | 2825 | −0.244 | 0.846 | 375 | +1.840 | 1.998 |
| fox/allee/0.2 | 0.662 | 1.080 | 2825 | −0.257 | 0.863 | 375 | +1.936 | 2.083 |
| fox/theta/0.1 | 0.412 | 0.381 | 3181 | −0.028 | 0.106 | 19 | +4.747 | 4.747 |
| fox/theta/0.2 | 0.395 | 0.386 | 3181 | −0.029 | 0.102 | 19 | +4.837 | 4.838 |
| fox/regime/0.1 | 0.725 | 1.145 | 2652 | −0.310 | 0.992 | 548 | +1.502 | 1.700 |
| fox/regime/0.2 | 0.709 | 1.176 | 2652 | −0.326 | 1.009 | 548 | +1.575 | 1.776 |
| vulture/ricker/0.1 | 0.310 | 0.910 | 142 | −3.715 | 3.723 | 3058 | +0.173 | 0.471 |
| vulture/ricker/0.2 | 0.302 | 0.915 | 142 | −3.754 | 3.763 | 3058 | +0.174 | 0.467 |
| vulture/allee/0.1 | 0.310 | 0.910 | 142 | −3.715 | 3.723 | 3058 | +0.173 | 0.471 |
| vulture/allee/0.2 | 0.302 | 0.915 | 142 | −3.754 | 3.763 | 3058 | +0.174 | 0.467 |
| vulture/theta/0.1 | 0.319 | 0.913 | 145 | −3.670 | 3.678 | 3055 | +0.174 | 0.481 |
| vulture/theta/0.2 | 0.312 | 0.918 | 145 | −3.706 | 3.715 | 3055 | +0.176 | 0.478 |
| vulture/regime/0.1 | 0.310 | 0.910 | 142 | −3.715 | 3.723 | 3058 | +0.173 | 0.471 |
| vulture/regime/0.2 | 0.302 | 0.915 | 142 | −3.754 | 3.763 | 3058 | +0.174 | 0.467 |

**Findings:** the linear surrogate is a modest reward fit (R² 0.30–0.73) and is **systematically biased in the
minority safety stratum** — it cannot represent the −10 occupancy cliff at s_safe. For the mostly-safe
recoverable species (tiger, fox) it strongly **over-predicts reward on the rare unsafe rows** (me +1.5 to +4.9),
i.e. it under-penalises danger; for the mostly-unsafe sink (vulture) it strongly **under-predicts reward on the
rare safe rows** (me −3.7). Vulture ricker/allee/regime rows are bit-identical (same surrogate) — cross-validates
the family degeneracy; theta differs marginally.

---

## P0-4. Compute per method-cell (measured)

**Planner time** (measured, `episodes.csv planner_seconds`; mean over 20 episodes, and per-cell total over 20;
means over the 24 cells per method):

| method | mean planner_s / episode | mean planner total / cell (s) |
|---|---|---|
| RefPlan | 2.543 | 50.86 |
| OGSRL | 20.469 | 409.38 |
| BA-MCTS | 42.310 | 846.21 |
| EVD | 0.0115 | 0.23 |
| PLUS | 765.225 | 15304.50 |
| MOOR | 96.923 | 1938.46 |

**Full-task time, measured per method-cell:**
- General (`GENRUN/…/validity_receipt.json → resources`; mean over 24 cells): RefPlan row 52.42 s
  (fit 0.178, eval 51.56); OGSRL 430.15 (fit 19.64, eval 409.92); BA-MCTS 847.26 (fit 0.007, eval 846.84);
  EVD 2.50 (fit 1.467, eval 0.54).
- Eco (`P10/…/summary.json manifest_row_seconds`; mean over 24 cells): PLUS 15 314.56 s → **102.10 core-h total**;
  MOOR 1 940.64 s → **12.94 core-h total**. (PLUS 102.10 + MOOR 12.94 ≈ the measured 115.13 core-h.)

**Measured eco-vs-general per method-cell:** the ecological PBVI planners dominate cost — PLUS ≈ 15 305 s/cell
(≈ 4.25 h) and MOOR ≈ 1 938 s/cell — versus BA-MCTS 847, OGSRL 430, RefPlan 52, EVD ≈ 2.5 s/cell. Measured
general total for the three species (96 cells) ≈ 8.9 core-h, vs measured eco 115.0 core-h for 48 cells.
(Recall the general 43.7052 core-h figure is a *projection*, not this measurement.)

---

## P0-5. `observation_scale` for all four families × three species (24 cells)

From each cell's saved surrogate npz `observation_scale` (median positive observations; format `famσ=value`):

| species | ricker σ0.1/0.2 | allee σ0.1/0.2 | theta σ0.1/0.2 | regime σ0.1/0.2 |
|---|---|---|---|---|
| Amur tiger | 72.65 / 70.38 | 64.59 / 61.40 | 76.12 / 74.99 | 61.44 / 59.54 |
| Crab-eating fox | 40.64 / 41.45 | 41.22 / 41.44 | 46.14 / 47.52 | 39.29 / 38.28 |
| Egyptian vulture | 35.57 / 35.66 | 35.57 / 35.66 | 36.13 / 36.59 | 35.57 / 35.66 |

(True K_ref = K_base = 250 / 41 / 325. Vulture ricker=allee=regime `observation_scale` identical → family degeneracy.)

---

## P0-6. σ_o consumption by OGSRL, BA-MCTS, EVD

**NOT USED** by any of the three: no reference to `observation_noise_sigma` in `GEN/methods/ogsrl.py`,
`GEN/methods/bamcts.py`, or `GEN/methods/ensemble_value_disagreement.py`. Only **RefPlan** consumes it
(`GEN/methods/refplan.py:81,146`, via its `LogNormalObservationModel` and posterior-variance term). σ_o is
nonetheless *available* to all methods as a constant (`MethodContext.observation_noise_sigma`, `pipeline.py:228`).

---

## P0-7. Sampling point for Allee threshold C and theta exponent θ

**Per episode** (re-drawn at every environment reset, from the episode's process RNG):
`ECO/envs.py:158 self._C = p.uniform(cfg.C_low, cfg.C_high)`; `ECO/envs.py:159 self._theta = p.uniform(cfg.theta_low, cfg.theta_high)`
inside `reset()`. Not per cell, not fixed at registration. (`C_low/C_high = 0.18/0.30·K_base`; `theta_low/high = 3.0/6.0`, `config.py:76-77`.)

---

## P0-8. Threshold used by `danger_action_*_fraction`

The **true private s_safe** (evaluator has the real environment). `ECO/evaluator.py:129`:
`if env_cfg.safety_threshold < state_previous <= 4 * env_cfg.safety_threshold: danger_action_counts[action] += 1`.
So the danger band is `(s_safe, 4·s_safe]` on the true pre-action state, with s_safe = 25 / 10.25 / 81.25. Not a proxy.

---

## P0-9. Per-cell `pbvi_argmax_actions`, `pbvi_action_margin_min`, and candidate unanimity (PLUS = 8 candidates)

Unanimity = fraction of the 8 candidates whose greedy action equals the modal action.

| species | family | σ | pbvi_argmax_actions | margin_min | modal | unanimity | constant |
|---|---|---|---|---|---|---|---|
| Amur tiger | ricker | 0.1 | 10;3;10;10;10;10;10;10 | 0.191 | a10 | 0.88 | True |
| Amur tiger | ricker | 0.2 | 10;10;10;10;10;10;10;10 | 0.242 | a10 | 1.00 | True |
| Amur tiger | allee | 0.1 | 10;10;10;10;5;10;5;10 | 0.217 | a10 | 0.75 | True |
| Amur tiger | allee | 0.2 | 10;5;10;10;5;10;10;10 | 0.158 | a10 | 0.75 | True |
| Amur tiger | theta | 0.1 | 10;5;10;10;1;10;10;5 | 0.145 | a10 | 0.62 | True |
| Amur tiger | theta | 0.2 | 10;10;3;10;10;10;10;10 | 0.067 | a10 | 0.88 | True |
| Amur tiger | regime | 0.1 | 10;9;10;10;8;10;10;10 | 0.199 | a10 | 0.75 | True |
| Amur tiger | regime | 0.2 | 10;10;10;10;10;10;10;10 | 0.058 | a10 | 1.00 | True |
| Crab-eating fox | ricker | 0.1 | 1;1;1;1;1;1;1;1 | 0.015 | a1 | 1.00 | False |
| Crab-eating fox | ricker | 0.2 | 1;1;2;2;2;1;1;1 | 0.001 | a1 | 0.62 | False |
| Crab-eating fox | allee | 0.1 | 2;1;2;2;2;2;2;2 | 0.074 | a2 | 0.88 | False |
| Crab-eating fox | allee | 0.2 | 7;1;2;2;2;6;2;1 | 0.055 | a2 | 0.50 | False |
| Crab-eating fox | theta | 0.1 | 1;1;2;2;2;2;2;2 | 0.061 | a2 | 0.75 | False |
| Crab-eating fox | theta | 0.2 | 1;1;1;2;2;2;2;1 | 0.020 | a1 | 0.50 | False |
| Crab-eating fox | regime | 0.1 | 2;2;8;2;1;2;2;2 | 0.066 | a2 | 0.75 | False |
| Crab-eating fox | regime | 0.2 | 2;2;8;2;1;2;2;2 | 0.091 | a2 | 0.75 | False |
| Egyptian vulture | ricker | 0.1 | 5;5;5;5;5;5;5;5 | 0.173 | a5 | 1.00 | True |
| Egyptian vulture | ricker | 0.2 | 5;5;5;5;5;5;5;5 | 0.224 | a5 | 1.00 | True |
| Egyptian vulture | allee | 0.1 | 5;5;5;5;5;5;5;5 | 0.173 | a5 | 1.00 | True |
| Egyptian vulture | allee | 0.2 | 5;5;5;5;5;5;5;5 | 0.224 | a5 | 1.00 | True |
| Egyptian vulture | theta | 0.1 | 5;5;5;5;5;5;5;5 | 0.248 | a5 | 1.00 | True |
| Egyptian vulture | theta | 0.2 | 5;5;5;5;5;5;5;5 | 0.282 | a5 | 1.00 | True |
| Egyptian vulture | regime | 0.1 | 5;5;5;5;5;5;5;5 | 0.173 | a5 | 1.00 | True |
| Egyptian vulture | regime | 0.2 | 5;5;5;5;5;5;5;5 | 0.224 | a5 | 1.00 | True |

MOOR (1 candidate ⇒ trivially unanimous): tiger→a10 (all cells), fox→a1 (ricker) / a2 (allee,theta,regime),
vulture→a5 (all cells). `pbvi_action_margin_min` for MOOR ranges 0.012 (fox/theta/0.1) to 0.780 (tiger/regime/0.2).

---

## P0-10. `economic_cost_mean / 50` matched to a deployed constant action (all 144 rows)

`cost/50` = mean per-step cost. Where the policy is constant it equals a single `cost_step`. Matched actions
(cost table: a0=0, a1=−0.05, a2=−0.1, a3=0.25, a4/a6=0.5, a5=0.1875, a7=0.4375, a8=0.75, a9=1.0, a10=0.3125):

**Constant policies with an exact match:**

| species | family | σ | method | cost/50 | matched action |
|---|---|---|---|---|---|
| Amur tiger | all 4 | 0.1 & 0.2 | PLUS & MOOR (16 cells) | +0.3125 | **a10** (translocation) |
| Egyptian vulture | all 4 | 0.1 & 0.2 | PLUS & MOOR (16 cells) | +0.1875 | **a5** (moderate restoration) |
| Crab-eating fox | theta | 0.2 | MOOR | −0.1000 | **a2** (aggressive harvest) |

(These agree with the P0-9 modal actions: tiger a10, vulture a5, fox a2.)

**Non-constant policies (the other 112 rows):** `cost/50` does not equal a single `cost_step` (the policy mixes
actions), so no single action is deployed. Representative averages: the four general methods on Amur tiger give
`cost/50` ∈ {RefPlan ≈0.29–0.47, OGSRL ≈0.13–0.20, BA-MCTS ≈0.47–0.64, EVD ≈0.06–0.12}; on Egyptian vulture
RefPlan ≈0.22–0.24, OGSRL ≈0.31–0.34, BA-MCTS ≈0.32–0.34, EVD ≈0.14; Crab-eating fox PLUS/MOOR/EVD/OGSRL hover
near −0.05…−0.10 (mixed a1/a2 harvest, occasionally exactly −0.10 = a2 but flagged non-constant). None of the
144 policies deploys a constant a4 or a6 (the ambiguous 0.5-cost actions).

---

## Cross-references / flags
- P0-3 quantifies the concern raised earlier (the planner reward is a learned public surrogate): the surrogate
  is a biased, low-R² approximation that mis-scores the safety cliff, over-rewarding danger for recoverable
  species and under-rewarding safety for the sink.
- P0-2 asymmetry: EVD alone trains on raw `dataset.rewards`; the other five methods (incl. PLUS/MOOR) share the
  same biased surrogate reward.
- P0-4: measured eco compute (115 core-h/48 cells) ≫ measured general (≈8.9 core-h/96 cells for the 3 species);
  PLUS's 8-candidate PBVI dominates. The general "43.7052 core-h" is a projection, not this measurement.
- Family degeneracy (Egyptian vulture ricker=allee=regime) reappears in P0-3 (fidelity), P0-5 (obs_scale), and
  P0-9 (identical argmax/margins) — all consistent.
