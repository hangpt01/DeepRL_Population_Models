# Read-only code-to-artifact audit — information symmetry, reward metrics, environment structure, artifacts

Date: 26 July 2026 (Australia/Melbourne). Read-only: no code, artifacts, manifests, or the accepted
`MATCHED_P10_144_METHOD_CELLS.csv` / receipt were modified. Every claim below is backed by a file+line
or an artifact hash. Items that are not present in the accepted artifacts are marked **NOT FOUND**.

## Frozen sources inspected

- **ECO code** = `real_ecology_runs/ricker_only_plus_72_20260720/runtime_snapshot_routing_fix/src/real_ecology_benchmark/`
  (named by the P=10 reward-independence audit as the code that ran the ecological cells).
- **GEN code** = `general_rl_phase2_iso/…/general_phase2e_full_sigma01_02_20260720_v1/code/src/real_ecology_benchmark/`
  (snapshot `snapshot_tree_sha256 = f615d363…`).
- **P10 root** = `real_ecology_runs/three_species_ecological_p10_correction_20260723_v1/`.
- Data tables: `real_ecology_data/species.csv` (sha256 `10d3e099d0ad0635970198d23b64b60860f09c521093d40797ebf78b8d2c01ab`),
  `real_ecology_data/action_effects_long.csv`.
- All 144 cells are `expose_rk=hidden`, safe reward, `collapse_penalty=10`, horizon 50, γ=0.95.

---

## A. Information symmetry (blocking)

### A1 — Planner unsafe-penalty threshold for PLUS & MOOR: a public LEARNED reward, no threshold, no private s_safe

The PBVI planner reward is delegated to the one shared public surrogate:

- `faithful_pomdp.py:209-229 expected_public_reward(...)` → `return float(np.dot(predicted, rewards))`,
  where `rewards, _risks = surrogate.predict(prev_obs, cur_obs, following_states, action, timestep, pop_id)`.
- `public_surrogate.py:199-223 PublicRewardRiskSurrogate.predict` → `reward = design @ self.reward_coefficients`.
- `reward_coefficients` is a **ridge regression on logged public rewards**:
  `public_surrogate.py:299-302` solves `(designᵀdesign + ridge) β = designᵀ · dataset.rewards[fit_mask]`.
- Features (`transition_design`, `public_surrogate.py:138-181`): `log1p(obs/observation_scale)`, action one-hot,
  `cost`, `τ`, opaque `pop_id` one-hot. **No s_safe term anywhere in the planner reward.**

**Conclusion:** the planner uses neither the true private s_safe nor an explicit public threshold proxy.
The safety penalty is only *implicitly* captured by the linear fit to logged `dataset.rewards`. MOOR uses
the identical `faithful_pomdp`/surrogate path.

Per-species: no threshold value exists inside the planner. (For reference, the evaluator's private s_safe is
25.0 / 10.25 / 81.25 for Amur tiger / Crab-eating fox / Egyptian vulture — see B2 — but the planner never sees it.)

### A2 — Planner abundance scale: public `observation_scale` (median positive obs), not true K_ref

- `public_surrogate.py:93-94 _z = log1p(values / observation_scale)`.
- `public_surrogate.py:66-67 observation_scale = median(positive observations)`.
- True K_ref: `config.py:332 K_ref = pop.K_base`.

| species | planner `observation_scale` (ricker cell) | true K_ref (=K_base) |
|---|---|---|
| Amur tiger | 70.38 | 250 |
| Crab-eating fox | 41.45 | 41 |
| Egyptian vulture | 35.66 | 325 |

The planner abundance scale ≠ true K_ref (fox coincidentally close; vulture 35.66 vs 325 is far off).

### A3 — OGSRL s_low = 20th percentile of positive training observations (public); same privilege level as A1/A2

- `ogsrl.py:552-557`: `positive = observations[observations>0]; self.s_low = np.quantile(positive, ogsrl_low_abundance_quantile)`.
- `config.py:540 ogsrl_low_abundance_quantile = 0.20`.
- Scale: `ogsrl.py:410,425 public_context.observation_scale`.

Numeric s_low (general fit_diagnostics, ricker σ0.1 / σ0.2): tiger 35.91 / 35.15, fox 31.90 / 29.82,
vulture 22.93 / 22.41.

**Privilege verdict — SAME level (public offline data only).** `MethodContext` (`config.py:228-236`)
exposes only `observation_noise_sigma, observation_scale, action_costs, opaque pop_id, surrogate, horizon,
reward_mode` — **no `safety_threshold`, no `K_ref`.** All 144 cells are hidden, so no method (ecological or
general) receives private s_safe or K_ref. Note a difference in *value*, not privilege: OGSRL's public
s_low ≈ 22.9 for the vulture ≪ its true s_safe 81.25 (the vulture lives entirely below its true floor).

### A4 — σ_o is a known public constant for all six methods; none estimates it

- `pipeline.py:228 MethodContext.observation_noise_sigma = dataset.metadata["observation_noise_sigma"]`.
- PLUS/MOOR filter: `beliefs.py:427,483,599` use `env_cfg/context.observation_noise_sigma`.
- General: `refplan.py:81,146` use `observation_noise_sigma`.

PLUS, MOOR and all four general methods receive the true σ_o (0.1/0.2) as a fixed constant; none fits it.

---

## B. Evaluator reward constants and mechanics

### B1 — `R_t = α·s'/(s'+K_ref) − cost(a) − P·1[s' ≤ s_safe]`, α=1.0, all terms on the LATENT true next state

- `reward.py:36-38 utility = alpha * x/(x+K_ref)`; `reward.py:32 alpha=1.0`.
- `reward.py:52-63 state_reward(next_state,…)`; `envs.py:323-329` (set-point branch)
  `reward = state_reward(state_next,…); reward_true = reward`.
- Penalty indicator on the true next state, occupancy: `reward.py:89-90 next_state <= safety_threshold`.
- `operational == true` verified in 288/288 general safe summaries (max diff 0.0) and in eco episodes.

### B2 — CONFIRMED correction: K_ref and s_safe are species-specific, NOT global 500/25

- `config.py:332 K_ref = pop.K_base`; `config.py:337 safety_threshold = safety_fraction * K_base`.
- `config.py:258-277 default_safety_fraction` with constants `config.py:34-41`
  (`SMALL_K=75, MID_K=120, ENDANGERED_N=50, SEVERE_DEPLETION=0.20, MODERATE_DEPLETION=0.50,
  HEALTHY=0.10, MID=0.20, HIGH_RISK=0.25`).

Verified against actual run summaries:

| species | K_ref (=K_base) | s_safe | safety_fraction | mvp_threshold |
|---|---|---|---|---|
| Amur tiger | 250 | **25.0** | 0.10 | 50 |
| Crab-eating fox | 41 | **10.25** | 0.25 | 50 |
| Egyptian vulture | 325 | **81.25** | 0.25 | 50 |

Your reading holds: s_safe(fox)=10.25 (<21.3), s_safe(vulture)=81.25 (≥41). **A global s_safe=25 / K_ref=500
does NOT hold.** (The 25-July handoff's *symbolic* formula `N/(N+K_ref) − cost − 10·1[N≤s_safe]` is correct;
an earlier addendum in this thread that substituted global 500/25 was wrong and is corrected here.)

### B3 — No early termination; always exactly 51 states; N=0 is an untriggered absorbing state

- `envs.py:344 terminated = state_next == 0.0`.
- State clipped ≥0: `envs.py:216, 221, 272 max(..., 0.0)`; 0 is absorbing: `envs.py:206-207 if state==0.0: return 0.0`.
- **Distinct `n_steps` across all 144 cells (eco + general episodes) = {50}** → every episode ran 50 steps
  (51 states); the absorbing clip never fired. No extinction/early stop in any cell.

### B4 — CSV field definitions (verified equal to the source summaries)

- `economic_cost_mean` = mean over 20 episodes of the **UNDISCOUNTED** sum `Σ_t cost(a_t)`
  (`evaluator.py:133 economic_cost += cost`; no γ). Not discounted, not a per-step mean.
- `min_population_mean` = mean over 20 episodes of `min_true_state` = **min over the 51-state trace of the
  latent true N** (`evaluator.py:183 np.min(true_values)`).
- Cross-check (tiger/ricker/σ0.1 PLUS): CSV 15.625 / 200.0 == summary 15.625 / 200.0.
- A per-step return decomposition into abundance / cost / penalty needs the per-step true-state and
  penalty-flag traces, which are **not** in the 144-CSV (only the aggregates above).

---

## C. Environment structure

### C1 — 11-action table per species (`action_effects_long.csv`: r_setpoint, capacity `dK_step`, translocation `dN`, `cost_step`)

Set-point r is clipped to `[r_min, r_max]` per family (Ricker columns here).

**Egyptian vulture** — r_setpoint_ricker (a0…a10): −0.0912, −0.2179, −0.2558, −0.0198, **−0.0098**, −0.0912,
−0.0912, −0.0198, −0.0198, −0.0098, −0.0912; dK_step: 0,0,0,0,0,32.5,97.5,32.5,97.5,97.5,0; dN: only a10=**4.1**;
cost: 0, −0.05, −0.1, 0.25, 0.5, 0.1875, 0.5, 0.4375, 0.75, 1.0, 0.3125.
**max_a r_a = −0.0098 < 0 ⇒ g_a = r_pos = max(r_eff, 0) = 0 for every action** (`envs.py:240`). Intrinsic growth
is impossible; only a10 (translocation) adds 4.1 individuals/step.

**Amur tiger** — max r_setpoint_ricker = **+0.0707** (a4/a9); dN(a10)=20.0. Growth possible.

**Crab-eating fox** — max r_setpoint_ricker = **+0.4418** (a4/a9); dN(a10)=4.1. Strong growth.

### C2 — Families share the species base but each has its own K-scaled/derived parameters (not one set with only the map changed)

`config.py real_environment:319-352`, `realdata.caps:68-72`:

- ricker/allee/regime use the **Ricker r-columns**; **theta uses the LGM r-columns** (different r_base/r_min/r_max).
- Allee threshold `C ∈ [0.18, 0.30]·K_base`; theta exponent `θ ~ U(3.0, 6.0)` per episode
  (`config.py:76-77`, `envs.py:159`); regime thresholds `0.18/0.36·K_base`; weak-regime multiplier **0.65**
  (`config.py:81`); `regime_persistence = 0.90` (`config.py:78`); initial regime `~U{0,1}` (`envs.py:168`).
- Per-species K_base (250 / 41 / 325) gives, e.g., Allee C∈[45,75]/[7.4,12.3]/[58.5,97.5];
  regime thresholds [45,90]/[7.4,14.8]/[58.5,117]. r-cap bounds from `species.csv`
  (vulture ricker r_max=−0.0098, lgm r_max=−0.0098; tiger ricker r_max=+0.0707).

### C3 — Vulture Ricker≡Allee≡regime identity is an ALGEBRAIC DEGENERACY; theta differs by parameters (LGM columns), not only the exponent

For the vulture `r_max<0 ⇒ r_pos=0`, so the family growth exponent is 0 for ricker (`envs.py:244`),
allee (`:247`), and regime (`:262`) alike ⇒ each reduces to `value = managed·exp(r_mort)` with r_mort from the
**shared Ricker columns** (`caps` maps all three to ricker) ⇒ bit-identical trajectories ⇒ identical returns
for all six methods. `theta` (`:249-252`) also has zero growth (r_pos=0) but takes r_mort from the **LGM
columns** (`caps:70-71`; vulture r_base_lgm=−0.0872 vs r_base_ricker=−0.0912) ⇒ a slightly different decline
⇒ the 4th-significant-figure difference. `process_noise_sigma=0.0` (`config.py:82`), so this is not a noise
artifact. **Answer: yes, the theta environment differs from Ricker in the r_base/r_min parameters (via the
LGM column set), in addition to the θ exponent.**

### C4 — Regime switches occur; the regime changes the growth mechanism only where r_pos>0

- `envs.py:338-339`: each step flips the regime with probability `1 − regime_persistence = 0.10`
  (expected ≈5 switches per 50-step episode).
- Active regime scales the positive growth exponent by 0.65 (`envs.py:261`).
- **Amur tiger** (r_pos>0 on growth actions): switches materially weaken growth.
- **Egyptian vulture** (r_pos=0 always): switches occur but are **inert** (multiplier × 0 = 0), which is why
  its regime cells equal its ricker cells.
- **NOT FOUND:** the observed per-episode switch count — no per-step regime trace is stored in the accepted
  per-cell summaries / `episodes.csv`. Only the mechanism/rate is verifiable without a re-run.

---

## D. Artifact availability for return-blind diagnostics

### D1 — Yes: the 8 fitted Ricker candidate parameter vectors are stored per cell

`{P10}/plus/fit_cache/{key}.npz` arrays: `growth, mortality, capacity_increment, stocking, capacity_ceiling,
depensation_thresholds, theta_exponent, regime_multipliers, regime_matrix, …` (param_dim = 25). Dumped
(growth[0] per candidate; pairwise Euclidean distance over the flattened parameter vector):

- **fox × regime × 0.2**: growth {0.181, 0.124, 0.176, 0.298, 0.181, 0.149, 0.215, 0.152};
  dist min 0.29 / max 1.67; near-dup(<1e-3) = **0/28**.
- **fox × Allee × 0.1**: growth {0.328, 0.221, 0.251, 0.098, 0.307, 0.360, 0.112, 0.086};
  dist min 0.26 / max 2.22; near-dup = **0/28**.
- **tiger × Ricker × 0.1**: growth {0.002, 0.115, 0.006, 0.027, 0.007, 0.210, 0.034, 0.044};
  dist min 0.27 / max 1.43; near-dup = **0/28**.

The banks are genuinely distinct (no near-duplicate candidates).

### D2 — NOT FOUND: the candidate posterior trajectory w_tʲ over the 50 steps is not stored

Only `pbvi_policy_diagnostics.npz = ['last_action_values']` is saved. Whether the posterior left uniform 1/8
during rollout is **not recoverable from accepted artifacts** (needs a re-run or a separate per-step capture).

### D3 — Per-decision availability

- **Free (stored):** per-candidate greedy action — CSV `pbvi_argmax_actions` (e.g. `10;3;10;10;10;10;10;10`,
  one per candidate at the representative belief); minimum top-action margin — CSV `pbvi_action_margin_min`
  (0.191); per-episode action entropy — `episodes.csv action_entropy`; danger-zone action fractions per action —
  `episodes.csv danger_action_0..10_fraction`; per-episode planner time — `episodes.csv planner_seconds`.
- **NOT stored (needs re-run):** the chosen action per step (full ordered sequence) and per-decision top-1/top-2
  margins (only the per-cell **minimum** margin exists). `episodes.csv` is per-episode aggregate only.

### D4 — Compute

- **Eco 115.13 core-h is MEASURED** (sacct: PLUS array `58493916` 102.15 + MOOR array `58493918` 12.98 +
  two acceptance tasks ≈0). Because `recomputed_fits = 0` (216 reused; `FIT_REWARD_INDEPENDENCE_AUDIT.json`),
  it **excludes candidate demographic fitting**; it **includes** dataset load, public-surrogate fit/load,
  POMDP build, PBVI solve, and 50-step evaluation.
- **General 43.7052 core-h is a PROJECTION, not a measurement.** It equals the sum of `projected_144_task_hours`
  over the four methods (bamcts 29.219 + ogsrl 12.845 + refplan 1.574 + EVD 0.067 = 43.7052) in
  `provenance/resource_projection_from_64_canary_receipts.json` (formula `manifest_row_seconds + 19·evaluation_seconds`,
  linear-scaling from 16-canary receipts). It bundles fitting + actor training + eval per task (no breakdown).
  Its scope label is "144"; treat it as **not comparable** to the measured eco 115.13, and note it is not a
  direct measurement of the completed 576-row general run (whose per-cell measured `resources` are in the 576
  `validity_receipt.json`).
- **Decision-time per method-cell: FREE** — `episodes.csv planner_seconds` (per episode), both sides.

---

## E. Flags vs the 25-July handoff / merged TeX

1. **K_ref/s_safe are species-specific** (250/25, 41/10.25, 325/81.25), not global — corrects an earlier
   addendum in this thread; the handoff's symbolic reward formula stays valid.
2. **The planner (PLUS/MOOR) reward is a learned public linear surrogate with no explicit safety threshold**;
   the handoff/TeX evaluator formula is *not* the planner's objective. The planner optimizes the belief-expected
   value of `surrogate.predict` reward, which only implicitly encodes the penalty.
3. **A1/A2/A3 share the public-privilege level**; no hidden method accesses private s_safe or K_ref
   (sharpens the handoff's "neither method receives the hidden family").
4. **Vulture family degeneracy (C3) and vulture uncontrollability (C1, max r ≤ 0)** are structural facts
   consistent with the handoff's "recoverability/controllability failure," now mechanistically pinned.
5. **NOT FOUND / re-run required:** per-step posterior trajectory (D2); full per-step action sequence and
   per-decision margins (D3); observed regime-switch counts (C4).
