# Problem & Fix Brief — the hidden ecological parameters (r, K) are leaked to the agents

**For the code-aware server agent (Claude/Codex).** This is the single biggest problem with the
current experiment. Read it, verify it against the code, then implement the fix in §5 while
respecting the critical-settings checklist in §6. Do **not** start coding before confirming §2
against the actual code.

---

## 0. FIRST: find and report the root cause (before fixing)

Before implementing anything, trace **where the exposure of r, K (and the known true-family) was
decided**, so this class of error is understood and prevented. Produce a short **root-cause note**:

- **Which document decided it.** Check the design/spec docs — especially the implementation plan's
  **E1** (*"Expose population identity and (r_eff, K_eff) as observed inputs (known)"*; *"λ profile,
  K_base and caps are observed inputs"*) and **E8** (*"r_eff, K_eff … the methods only need to read
  the (already known) (r_eff, K_eff)"*) — and the algorithm/feature spec (public-control terms in the
  feature vector). Quote the exact lines.
- **Which code enforces it.** The env's observed-input construction, the offline tuple, the feature
  builders (refplan/bamcts/ogsrl), and the native-solver table reads. Give file/function + **git
  blame** (commit / author / date).
- **Was it a flawed instruction faithfully implemented, or a code deviation from the doc?** State
  which. *(Working hypothesis: the plan itself specified exposing r_eff/K_eff/K_base/caps/λ as
  "known context" — conflating "known population identity" with "known demographics" — and the code
  followed the plan. i.e. a **design/spec error propagated to code**, not a coding bug.)*
- **Why it was not caught.** There was a reward-leakage guard but **no parameter-leakage guard**, and
  **no explicit public/private schema derived from the motivation**.
- **One-line prevention rule** for next time (e.g. *"every observed input must pass: does exposing it
  collapse the uncertainty the experiment is meant to test? — and identity ≠ demographics"*).

Report this as deliverable §10.0 **before/with** the fix.

## 1. The problem (in one paragraph)

The ecological parameters that were meant to be **hidden** — the **per-action growth rate r(a)
/ r_eff** and the **per-population carrying capacity K (K_base / K_eff)** — are in fact **exposed
to the agents as observed inputs**, because they are published in the action/species tables and
fed to the models. So every method (both the general learners and the native ecological solvers)
effectively **reads the dynamics**. This turns the experiment into a **"structure-known"** test
(the demographics are given) instead of the **hidden-parameter / uncertainty** setting the paper
is supposed to study. Because parameter uncertainty is absent, a native mechanistic solver
becomes **near-oracle**, which is why it dominates general offline MBRL. That result is a
**valid finding for a structure-known / data-rich regime** (where "data-rich" means
demographics/action effects are well-characterized and public, not that the transition buffer is
large) — but it is **not** the intended
hidden-parameter (uncertainty) experiment.

## 2. Where the leak is (verify against the code)

The server usage audit has already confirmed the broad exposure pattern; re-verify these paths at
fix time so the implementation matches the current code rather than a stale summary.

- **Implementation plan E1/E8** states the design exposes them: *"expose … (r_eff, K_eff) as
  observed inputs (known)"* and *"r_eff a known observed input."* So this is a **design/setup
  leak**, likely not a stray coding error — the code is doing what the spec said. **[VERIFY]** in
  `discrete_action_cont_obser/` env + method code.
- **Public offline tuple** includes `ρ_{t+1}` (= r_eff set-point, since d_r=1) and `K_eff_{t+1}`.
  **[VERIFY]** the dataset schema.
- **General learners' feature vector** includes the "public-control terms" (`ρ_{t+1}`,
  `κ_{t+1}/K_ref`, `K_eff/K_ref`) — so refplan/bamcts/ogsrl receive r_eff and K as features.
  **[VERIFY]** the dynamics-model feature construction.
- **Native solvers read the tables directly:** `moor_native` reads the action-specific growth
  set-points and species K to build a table-driven mechanistic model (near-exact on Ricker;
  misspecified but still strong off-Ricker); `plus_native` reads them too and
  only keeps a posterior over the functional *form*. **[VERIFY]** `moor_native` / `plus_native`
  model construction — confirm they read `action_effects_long.csv` / `species.csv` r, K rather
  than estimating them from data.

> **Grep starting points:** `r_eff`, `delta_r`, `r_setpoint`, `K_base`, `K_eff`, `action_effects`,
> `species.csv`, and the dynamics-model / feature-vector construction for refplan/bamcts/ogsrl and
> the model build for moor_native/plus_native.

## 3. Clarification — last week did NOT fix this (do not treat as a duplicate)

> last week ≠ "hid r, K." Last week = same public-table environment, weaker ecological baseline.
> So it isn't a duplicate of the redesign experiment; it's a different point on the same
> environment. The redesign (hiding r, K) is the only one of the two weeks' setups — actually of
> neither week's setups — that would make the intended uncertainty bite.

Both prior runs (the `psafe_overnight_*` run and the `motivation_native_*` run) used the **same
public-table environment**; they differed only in the *strength of the ecological solver*
(adapted particle-MPC vs. native tabular). **Neither** hid r or K. So this fix is genuinely new
work, not a re-run of anything already done.

## 4. Why it matters

- It removes the **parameter-uncertainty** layer the paper claims to test.
- It hands the native solvers table-derived mechanistic models → the "native beats general" result
  is a **valid structure-known / data-rich comparison**, but **not** evidence for the intended
  uncertainty setting. Here, "data-rich" means demographically well-characterized/public tables, not
  a large transition budget.
- The general learners look "model-limited" partly because they compete against an opponent that
  was handed the demographics. **Keep the current run as a labelled *data-rich* comparison arm** —
  do not discard it, but do not present it as the uncertainty experiment.

---

## 4.1 Current vs corrected method-facing information flow

Use this table as the quick implementation target. The left column is what the current
public-demographics / parameter-known run allows. The right column is what the corrected
hidden-demographics / structure-unknown run must enforce. The simulator/evaluator may still use the
real tables privately; the methods may not.

Here, **structure-unknown** means both (i) hidden r/K and action-effect parameters on the parameter
axis and (ii) private `true_family` on the model-form axis. The schema fix is environment-level and
applies to every method in the codebase, including `mopo`, `delphic`, and adapted `moor`/`plus`;
the headline hidden-r,K run uses the five-method comparison: `refplan`, `bamcts`, `ogsrl`,
`moor_native`, and `plus_native`.

| Scope / method | Current run: public-demographics / parameter-known | Corrected hidden-r,K run |
|---|---|---|
| General public schema | Public dataset, belief cache, and configs may expose `rho`, `kappa`, `K_eff`, `K_base`, `K_ref`, `K_max`, action set-points, `dK`, `dN`, K-derived `s_safe`, population table lookups, and family/config identity. | Methods may consume only sanitized public data: `pop_id` as categorical token if chosen, `o_t`, `a_t`, `cost(a_t)`, `R_t` as label only, `o_{t+1}`, and `done`. All r/K aliases, table effects, K-derived safety/normalization terms, and true-family metadata stay private. |
| `refplan` | Learns dynamics with `rho`, `kappa/K_ref`, and `K_eff/K_ref`; MPC carries public controls and uses `K_ref`/safety reward terms. | Fit dynamics and plan from sanitized belief/observation/action features only. If action effects matter, infer them from data; do not pass exact `rho`, `kappa`, `K_eff`, `K_ref`, or K-derived safety features. |
| `bamcts` | Uses the same control-aware learned dynamics as `refplan`; tree keys and rollouts condition on `rho` and `kappa`. | Tree state and rollouts must be indistinguishable for histories with the same `(o,a)` but different private r/K. Learned dynamics cannot receive public-control aliases or exact table-derived action effects. |
| `ogsrl` | Uses `rho`, `kappa/K_ref`, `K_eff/K_ref`, `s_safe`, and unsafe/safety features in dynamics, actor features, guardian features, rollouts, and feasibility checks. | Remove r/K-derived features from dynamics, actor, guardian, rollout, and safety inputs. Safety-aware behavior must learn risk from public trajectories or use a public threshold not derived from hidden `K_base`. |
| `mopo` | Uses the same learned-dynamics public-control channel as the other general planners; planning reward uses `K_ref` and safety configuration. | Use sanitized features only. Model uncertainty should include uncertainty about hidden action effects and capacity rather than conditioning on exact public controls. |
| `delphic` | Does not build an ecological transition model, but receives shared belief features containing `rho`, `kappa/K_ref`, `K_eff/K_ref`, and safety terms. | Build compatible-world and Q features from sanitized public beliefs only. Remove hidden-parameter aliases from the shared feature vector so `delphic` cannot inherit the leak indirectly. |
| adapted `moor` | Uses exact action set-points, `dK`, `dN`, and species capacity structure in a mechanistic Ricker proposal; mainly fits a K-like scale in set-point mode. | Mechanistic proposal cannot call table-derived exact action effects or species capacity. It must fit required Ricker parameters/effects from public data, or be reported only in the public-demographics comparison arm. |
| adapted `plus` | Uses exact table-derived action effects and species capacity ranges inside a candidate mechanistic bank; candidate axis is mostly K because growth set-points are public. | Candidate parameters and priors must be derived from sanitized public data, not species/action tables. If retained, report separately from the naive headline if it performs closed-world model selection. |
| `moor_native` | Fits `K_hat`, but reads exact action growth set-points, `dK`, `dN`, costs, `K_ref`, and `s_safe` to build its native Ricker solver. | Headline naive baseline. Fit/estimate Ricker parameters and action effects from sanitized public data, then solve a single-Ricker model. Do not read species/action tables or K-derived safety/normalization values. |
| `plus_native` | Builds one native solver per candidate family using exact table-derived r/K, action effects, reward, and safety structure; updates mainly the family posterior. | Separate closed-world model-selection baseline. It may keep a family posterior, but all parameter candidates/priors must be fit from sanitized public data. Do not redesign open-world PLUS in this first hidden-r,K repair. |

Recommended first-fix decisions unless the PI overrides them:

1. Use **known population label, unknown demographics**. `pop_id` is a category only, never a table
   lookup path.
2. Use `expose_rk=hidden` as the intended default; keep `expose_rk=full` to reproduce the
   current public-demographics / data-rich comparison arm. **Keep BOTH regimes runnable behind one
   interface — do not delete or overwrite `full` (see §5.5).** Do not add `noisy` until the hidden
   setting is stable.
3. Keep `K_ref` private/evaluator-only.
4. Keep any `s_safe` derived from `K_base` private/evaluator-only. If a public safety threshold is
   needed later, define it independently of hidden `K_base`.
5. Use `moor_native` as the headline naive ecological baseline.
6. Report `plus_native` separately as closed-world form selection; postpone open-world PLUS.

---

## 5. The fix — hide r and K from ALL agents (instructions)

**Goal:** move from the *structure-known* regime to the *structure-unknown* regime, where the
per-action growth rate and per-population carrying capacity are **latent** and must be **inferred
from the offline data** by every method.

1. **Stop feeding r_eff / K to the agent's inputs.**
   - Remove `ρ_{t+1}` (r_eff set-point) and `K_eff / K_base` from the **general learners' feature
     vectors** and from any observed-input/context passed to the policy/belief. The agent should
     know **which action it took (a_t)** but **not the growth rate that action induces** and
     **not the population's K** — those must be learned.
   - **[DECISION]** keep the *cost* of an action observed (cost does not reveal r, K), but the
     action→(r, K-effect) mapping must be hidden.

2. **Make the NATIVE ecological solvers ESTIMATE r, K from data — do not read the tables.**
   This is the crux. If only the general side loses r, K while `moor_native`/`plus_native` keep
   reading the tables, the asymmetry gets **worse**, not fixed.
   - `moor_native` must **fit** its Ricker r and K from the offline `(o, a, o')` data (its
     original least-squares/POMDP-learning behaviour), **not** read `r_eff`/`K_base` from the tables.
   - `plus_native` must likewise estimate r, K (and may keep its form posterior) **from data**,
     not from the tables, while keeping its existing form-candidate set for this first fix. Its
     per-candidate parameters/priors must be fit from the sanitized data. Do **not** change the
     candidate axis itself — adding r/K as explicit candidates, adding distractor forms, or making
     an open-world candidate set is the separate later experiment (§6.7 scope note).
   - **Concrete contract — `fit_from_public_data()`.** Give both native solvers an explicit
     estimation entry point that takes ONLY the sanitized public dataset (§6.2) + belief/observation
     features and returns the model parameters. No argument may be, or be derived from,
     `species.csv` / `action_effects_long.csv` / any true r, K. `plus_native` may keep its form
     posterior, but its parameter candidates/priors must be **built from the data**, never
     table-initialized. (Verified by the table-corruption test in §7.)

3. **Keep r, K available only to the simulator/evaluator (sidecar), never to any method** — the
   same private/public boundary already used for the true state s and the structural latents
   (C, θ, z).

4. **[DECISION] Population identity.** Choose one and document it:
   - (a) *Known species, unknown demographics* — the agent knows the population label but not its
     r, K (must infer). Recommended: keeps it a per-population problem while restoring uncertainty.
   - (b) *Hidden population* — also hide identity (harder variant; there is already a
     `hide_population` flag per the plan).
   - **Regardless of the choice, `pop_id` is a categorical label only.** No code path (method,
     feature builder, or native fit) may look up `pop_id → species.csv` to recover r, K, caps, or
     `s_safe`.

5. **Implement the fix as an explicit REGIME SPLIT — do NOT delete or globally overwrite the
   current r,K-known behaviour.** Route both regimes through **one shared interface** behind a flag
   (`expose_rk ∈ {full, hidden}`, or equivalently `info_regime ∈ {public_demographics,
   hidden_demographics}`):
   - `full` = **keep the current run as-is** (public-demographics / parameter-known). This is a
     deliberate **ablation / upper-bound** arm — the "knowing r,K" run is still scientifically
     useful (it may reveal what the model *could* do with perfect demographics, informing later
     ablations), so it must stay **runnable**, not be destroyed by the fix. Modify it only as much
     as needed to **label** it and route it through the shared interface.
   - `hidden` = the new corrected setting (§5.1–§5.4, §6) — the **intended default / headline**
     uncertainty test.
   - **Record the regime** in output directories, configs, manifests, and result summaries, so
     `full` and `hidden` results can never be confused.
   - *(Optional, later)* a `noisy` variant (noisy r, K estimates given as priors) — add only once
     `hidden` is stable. This is the only genuinely optional part of the flag set.

---

## 6. Public / private schema — the single most important part (MUST verify)

A side-channel anywhere here reopens the problem. Audit every method's input path against it.

### 6.1 HIDDEN — private evaluator sidecar ONLY; never an input/feature to any method
- [ ] true abundance `s_t`; structural latents `C` (Allee), `θ`, `z` (regime)  *(already hidden)*
- [ ] **all growth quantities (aliases too):** `r(a) / r_eff`, `r_setpoint_ricker/lgm`, `ρ_t`,
      `ρ_{t+1}`, `r_min`, `r_max`, `r_base`, and the per-population **λ profile**  ← NEW
- [ ] **all capacity quantities:** `K_base`, `K_eff_t`, `K_eff_{t+1}`, `K_max`, `K_ref`, `κ_t`,
      `κ_{t+1}`, `dK_step`  ← NEW
- [ ] **direct-state action magnitude:** translocation `ΔN / dN` (reveals the state authority)  ← NEW
- [ ] any table encoding the above: `species.csv`, `action_effects_long.csv`, `species_lambda.csv`, caps columns
- [ ] **shared belief cache / feature builder:** must not contain or derive any quantity in this
      hidden list; sanitize at the source so no method (especially `delphic`) inherits the leak
      indirectly

### 6.2 OBSERVED — the sanitized public dataset (methods may load ONLY these)
`( pop_id [categorical label only], o_t, a_t, cost(a_t), R_t, o_{t+1}, done )` plus
belief/observation features derived **only** from these. Remove `ρ, κ, K_eff` from the offline
tuple that methods consume (they may remain in the private sidecar for the evaluator). Nothing in
the loader may be, or be derived from, §6.1.

### 6.3 Indirect capacity leaks — decide explicitly (both leak K)
`K_ref = K_base` and `s_safe = c_safe · K_base`, so exposing either leaks the capacity scale.
- [ ] `K_ref`: make it **private** (evaluator-only) — it appears in the reward benefit, but the
      benefit is a reward term, not a policy input (§6.4).
- [ ] `s_safe`: choose and document — **(a)** a public management threshold set **independently of
      the hidden K** (e.g. an absolute count), or **(b)** private/evaluator-only, with safety-aware
      methods **learning risk from data**. Do **not** pass a `c_safe·K_base`-derived `s_safe` as a feature.

### 6.4 Reward guard — stronger than the old one
`R_t` may remain an **offline learning label** only. **None** of these may enter any policy /
belief / model feature: the reward **decomposition** (benefit / cost / penalty), the **benefit
term** `α s'/(s'+K_ref)`, the **unsafe/collapse flag**, the **true state** `s`, or any `s`- or
`K`-based diagnostic. (The benefit is invertible to `s_{t+1}` given `K_ref`, and `K_ref` is now hidden.)

### 6.5 Symmetry
Neither the general learners nor `moor_native`/`plus_native` may read exact r, K; **both** estimate
from the sanitized dataset (§6.2).

### 6.6 Keep UNCHANGED (controlled comparison)
- [ ] 9 populations, 4 families, σ ∈ {0, 0.1, 0.2, 0.4}; 2 reward modes (safe P=5, yield P=0)
- [ ] 4000 transitions/cell; eval 5 × 4 × horizon 50; planner budget (h5 / 96 seq / 32 particles)
- [ ] the 5 headline methods (`refplan`, `bamcts`, `ogsrl`, `moor_native`, `plus_native`) for the
      corrected run; the schema fix itself applies to every method in the codebase
- [ ] reward computed on the **true** next state (in the evaluator); the private/public boundary

### 6.7 Structural-uncertainty axis — do not leak true family

This fix hides r, K, but the structural-form setting must also be preserved.

- The true family — `true_family` / `family` / `dynamics_family` / `model_family` / `map_family`
  / `env.family` / `cfg.kind` — is **private evaluator/simulator metadata only** (including any
  manifest/config column or filename/path-derived label).
- `moor_native` is the headline naive ecological baseline: it must always fit/solve a **single
  Ricker** model, even when the simulator truth is Allee, theta-logistic, or regime-switching.
- `plus_native` is **not** the naive baseline. If its candidate set is `{Ricker, Allee,
  theta-logistic, regime}`, report it separately as a **closed-world model-selection** ecological
  solver, not as the motivation headline. Note: once r, K are hidden it **may or may not** still
  identify the true form — **do not assume it stays near-oracle**; let the data decide.
- No method may receive the true family as an input, feature, config-conditioned model choice,
  table lookup, manifest shortcut, or filename/path-derived switch.
- **Acceptance test:** with the same public dataset but relabelled **private** `true_family`
  metadata, method training/fitting/planning artifacts must be unchanged. Only evaluator
  grouping/reporting may change.
- **Reporting:** the headline naive comparison is **best general vs `moor_native`**, reported
  **Ricker vs non-Ricker families separately** (`moor_native` is correctly specified only on Ricker).

**Scope note (this first fix only):** do NOT redesign `plus_native`'s candidate set now. First run
the corrected hidden-r,K setting with (1) `moor_native` = main naive baseline, (2) `plus_native` =
separate closed-world model-selection baseline, (3) the current structure-known run = labelled
*data-rich* comparison arm. An "open-world PLUS" variant (distractor forms, or the truth sometimes
absent from the candidate set) is a valid **later** experiment — mixing it into the r,K repair
would muddy the causal story.

## 7. Acceptance tests

- [ ] **No-r/K-access test:** grep + a runtime check proving no method receives `r_eff`, `K_base`,
      or `K_eff` (or any §6.1 alias) as an input/feature; two rollouts identical in `(o, a)` but
      different hidden r/K must be indistinguishable to the policy input.
- [ ] **Table-independence test (strongest, covers the whole schema):** corrupt / shuffle
      `species.csv`, `action_effects_long.csv`, `species_lambda.csv` and confirm **every method's
      training, fit, and belief features are byte-identical** — nothing reads the hidden tables.
- [ ] **Family-relabel test (structural axis):** with the same public dataset but relabelled
      **private** `true_family` metadata (and all its aliases), every method's training/fit/planning
      artifacts are unchanged (only evaluator grouping/reporting changes) — proves no method reads
      the true family.
- [ ] **Native solvers now estimate:** confirm `moor_native`/`plus_native` fit r, K from data
      (not table reads); their model should degrade when data are scarce/noisy.
- [ ] **Identifiability sanity:** check whether r, K are recoverable from the 4000-transition,
      noisy-observation budget — if not, report it (this is the real uncertainty, but flag if the
      budget makes it hopeless).
- [ ] **Expected outcome, not a gate:** native performance should drop relative to the leaked run
      because they are no longer near-oracle. If it does not, report that as a result (r/K were
      identifiable from the public data), not as a failed test.
- [ ] **Re-run the headline comparison** (general vs native ecological) in the `hidden` setting,
      across all cells; report per-σ and per-family, recoverable/sink split.
- [ ] **Regime split intact (both runs live):** `full` mode reproduces the current method-facing
      fields/behaviour (values match the prior run where reproducible); `hidden` mode removes every
      §6.1 field from all methods; the simulator/evaluator use the private true tables in **both**;
      and the table-corruption / family-relabel tests affect `hidden`-mode agents only through data,
      never a direct table lookup. Confirm the regime is recorded in every output/config/manifest so
      the two arms can't be confused.
- [ ] Everything in §6 "keep unchanged" verified identical to the prior run.

## 8. Do NOT confuse this with the planning-model-source ablation

- **This fix** changes the **environment/observation**: it stops publishing r, K so demographics
  become latent for everyone.
- **The planning-model-source ablation** (separate, earlier to-do) changes only what dynamics the
  *general* planners roll through, on the *current* leaked env.

They are different experiments. This one is the priority, because it fixes the setup so the paper
tests what it claims. After it is done, the model-source ablation (if still needed) runs on the
corrected env.

## 9. Open decisions for the PI (answer before running)

1. Population identity: known-species/unknown-demographics (recommended) vs. fully hidden.
2. Whether/when to add the `noisy` variant (noisy r, K priors) — off by default until `hidden` is
   stable. **(Keeping BOTH `full` and `hidden` regimes behind one interface is already decided —
   see §5.5 — it is a requirement, not an open decision.)**
3. Open-world PLUS variant (candidate set with distractor forms, or the true form sometimes absent)
   — a valid **later** experiment, **not** part of this first fix (would muddy the causal story).

## 10. Deliverables — report back

0. **Root-cause note (§0)** — where the r,K/true-family exposure was decided (doc line + git blame),
   whether it was a flawed spec faithfully implemented or a code deviation, why it wasn't caught, and
   the one-line prevention rule. **Deliver this first.**
1. Answers to every **[VERIFY]** and to §9's decisions, with exact file/function paths.
2. The **sanitized public dataset loader** (§6.2) and the **private evaluator sidecar**, as a diff.
3. `moor_native` and `plus_native` `fit_from_public_data()` implementations (§5.2).
4. **All acceptance tests (§7) passing** — especially the table-independence and family-relabel tests.
5. The corrected **hidden-r,K run**: headline **best general vs `moor_native`**, split **Ricker vs
   non-Ricker**, per-σ, recoverable/sink — with the current structure-known run kept as a labelled
   **data-rich** comparison arm.
6. A short note of anything in this brief that **conflicts with the actual code** (so we correct the
   brief, not silently deviate).
