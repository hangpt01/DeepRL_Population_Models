# Phase 2 — Corrective Implementation Report (Hidden General-RL Baselines)

Controlled by `CODE_SERVER_PHASE2_GENERAL_RL_CORRECTIVE_IMPLEMENTATION_AUTHORIZATION.md`;
continues `PHASE1B_…` and `PHASE1_…`. **Nothing was submitted, no performance returns were
inspected, the running PLUS/MOOR jobs and the main dirty tree were not touched.** All work was
done in an **isolated git worktree**.

**Approved direction implemented:** fix RefPlan and OGSRL; add genuine in-tree belief updating to
BA-MCTS; retain Delphic as **Delphic-inspired** with a real observational-compatibility gate;
archive (not delete) MOPO; keep the matched **4,000-transition** budget.

**Headline result:** all four defining-mechanism gates pass, with **one honestly-reported
benchmark degeneracy**: OGSRL's *safety-cost* channel (`P(terminated)`) is constant across actions
benchmark-wide, so that sub-constraint cannot bind. Per the authorization I **measured and reported
this rather than inventing a new safety target**; OGSRL's defining **OOD-guarded** constrained
optimization does train and bind. See §9 and the decision table (§12).

---

## 1. Files, commits, digests

**Isolation.** Worktree `/fs04/scratch2/ce25/general_rl_phase2_iso` at detached base
`5f9cf32` ("Add the known r,K model - Before hiding them"); the current hidden pipeline was imported
into it. Main worktree left dirty and **verified untouched** (`methods/refplan.py` main-tree
sha256 `5248b797…` unchanged before/after Phase 2; the corrected worktree copy is `2c8c4ece…`).
`git worktree list` shows both trees; 32 running + 2 pending `adapt32-plus` (PLUS/MOOR) jobs
unchanged throughout.

**Provenance manifests** (in `docs/fix_implement_general_RL/PHASE2_provenance/`):
`pre_correction_sha256.txt`, `pre_correction_gitprovenance.txt`, `worktree_status_pre.txt`,
`post_correction_sha256.txt`, `registration_general_corrected.json`, and the two canary JSONs.

**Files changed (general-RL only) vs `5f9cf32`:**

| File | Δ | Nature |
|---|---|---|
| `src/real_ecology_benchmark/methods/refplan.py` | +45/−3 | hidden posterior-marginalized planning |
| `src/real_ecology_benchmark/methods/ogsrl.py` | +299/− | hidden ConOpt (actor+critic+dual) + degeneracy probe |
| `src/real_ecology_benchmark/methods/bamcts.py` | +144/− | in-tree Eq-4 ensemble-belief update |
| `src/real_ecology_benchmark/methods/delphic.py` | +85/−12 | calibrated behavior head + compatibility gate wiring |
| `src/real_ecology_benchmark/public_models.py` | +~100 | `PublicParticlePlanner.plan_marginalized` |
| `src/real_ecology_benchmark/delphic_compat.py` | **new (199)** | observational-compatibility + diversity gate |
| `tests/real/test_general_paper_mechanisms.py` | **new (403)** | 16 constructed-mechanism tests |
| `tests/real/test_general_privacy.py` | **new (151)** | value-leak / relabel / determinism |
| `scripts/general_adequacy_probe.py` | **new (221)** | return-blind adequacy + runtime probe |
| `scripts/make_general_corrected_manifest.py` | **new** | prepared 4,000 manifest (not submitted) |

MOPO (`mopo.py`) was **not modified in identity** and **not deleted** (it retains its hidden branch
from the pre-Phase-2 tree; excluded from the retained manifest).

**Snapshot / registration (Stage 10, prepared):**
`real_ecology_runs/general_corrected_prepared/code/` (runnable `src/` copy) +
`registration_general_corrected.json` (base commit, methods, budgets, HP, per-file sha256,
preregistered sensitivities, `submitted:false`).

---

## 2. Implementation account per method

### RefPlan — Stage 2 (identity restored)
- New `PublicParticlePlanner.plan_marginalized(belief, dynamics, posterior, pessimism)`
  (`public_models.py`): scores every shared candidate sequence **under each ensemble member**
  with common random numbers, then combines with the deployment posterior
  `mean = posterior·M`, `reflected = mean − pessimism·√(posterior·(M−mean)²)`. This *marginalizes*
  the model belief and applies the return-variance penalty (Sikchi 2021), mirroring full mode and
  the paper's belief-marginalized MB planning.
- Hidden `act` now calls it (`refplan.py`); the posterior maintained in `observe()` (residual
  lognormal likelihood) **materially drives** planning. Disclosed: the paper's conservative prior
  policy is omitted (random/enumerated candidate sequences) → RefPlan-**inspired**.

### OGSRL — Stage 3 (ConOpt trained in hidden mode)
- New public helpers (`ogsrl.py`): `_public_state_features`, `_public_policy`,
  `_sample_cached_public`, `_public_rollouts`, `_public_belief_action_risks`, and
  `_fit_hidden_actor`. Hidden `fit` no longer returns early: it runs the **Lagrangian constrained
  policy optimizer** (linear softmax actor via policy gradient over public guarded-model rollouts;
  duals `lambda_ood`, `lambda_safety` via constraint-violation ascent) over the
  `PublicDynamicsEnsemble` + `PublicKNNGuardian` + public reward/OOD/risk channels. Paper permits
  non-CPO optimizers, so this is a legitimate `ConOpt` variant.
- Hidden `act` now deploys the **trained actor** ranked by belief-averaged action probability,
  filtered by the guardian/safety **feasibility mask** with least-violating fallback.
- **Return-blind degeneracy probe** built into fit: reports
  `safety_channel_action_spread`/`safety_channel_degenerate`. Theoretical safety-guarantee language:
  none present in code (`grep` clean).

### BA-MCTS — Stage 4 (in-tree Bayes-adaptive belief)
- `_simulate_public` rewritten (`bamcts.py`): the Bayes-adaptive state is `(observation, log-belief
  over members)`. At each node it **samples the successor model from the current node belief**, then
  **updates the belief by the public transition likelihood** (`_public_member_loglik`, Eq. 4) before
  recursing — no fixed root model. `_belief_from_logweights` gives explicit numerically-stable
  normalization, zero-likelihood → uniform reset, non-finite-max guard. Uses only public
  observations + public ensemble (no private state/reward/`r`/`K`/safety). Action progressive
  widening omitted (discrete actions, documented); continuous observation handled by log-bucket
  aggregation. Hidden `act` seeds each simulation with `log(posterior)`.

### Delphic-inspired — Stage 5 (observational compatibility)
- New module `delphic_compat.py` + wiring in `delphic.py`. Behavior heads are now **calibrated**
  (feature standardization; 400-iter, lr 0.3, L2 1e-2 logistic) so worlds can reproduce the observed
  action distribution. `fit` builds candidate worlds, fits a **latent-free reference behavior
  model**, runs `gate_worlds`, retains only surviving worlds for the delphic-uncertainty channel,
  and reports the gate. Cross-world Q variance and the `r̃ = r − λ·u_d` pessimism are preserved.

### MOPO — archived
- Unchanged and excluded from the retained manifest (`make_general_corrected_manifest.py` routes only
  the four). Still registered/importable for provenance.

---

## 3. Paper-mechanism vs adaptation matrix (post-correction, hidden mode)

Classes: **FR** faithful · **BA** benchmark adaptation · **CA** computational approximation ·
**PA** privacy adaptation · **MR** material replacement · **MI** missing.

| Method | Defining mechanism | Status now | Class | Evidence |
|---|---|---|---|---|
| RefPlan | deployment belief over dynamics | maintained + updated | FR | `refplan.py:observe` |
| RefPlan | **marginalize belief into planning** | **restored** | FR | `plan_marginalized`; mechanism tests |
| RefPlan | conservative prior policy | omitted (random seqs) | MR (disclosed) | docstring |
| RefPlan | model class (neural) | ridge-linear ensemble | CA | shared model |
| OGSRL | OOD guardian (support) | kNN vs α-quantile (paper uses kNN/KDE) | FR/CA | `PublicKNNGuardian` |
| OGSRL | **ConOpt (constrained policy opt.)** | **trained (Lagrangian actor+dual)** | CA | `_fit_hidden_actor`; tests |
| OGSRL | safety-cost constraint | present but **channel degenerate** | MI-on-this-benchmark (reported) | §9; probe |
| OGSRL | theoretical guarantee | removed | MI (disclosed) | code clean |
| BA-MCTS | **in-tree Bayes-adaptive belief (Eq.4)** | **implemented** | FR | `_simulate_public`; tests |
| BA-MCTS | successor model from updated belief | implemented | FR | `bamcts.py` |
| BA-MCTS | outer PI distillation | omitted (eval-time planner) | MI (approved D2) | — |
| BA-MCTS | double progressive widening | action-PW omitted (discrete) | BA | log-bucket state |
| Delphic | observationally-compatible worlds | **gated (marginal-TV)** | CA | `delphic_compat.py`; tests |
| Delphic | delphic uncertainty = Var_w Q_w | preserved | FR | `_uncertainty` |
| Delphic | pessimism `r̃=r−λu_d` | preserved | FR | `delphic.py` |
| Delphic | ELBO neural world model | random-proj + calibrated heads | MR (disclosed → `-inspired`) | module docstring |

---

## 4. Delphic compatibility: definition, thresholds, evidence

**Definition (preregistered in `delphic_compat.py`, before any return opened).** A candidate world
survives iff the **total-variation distance between its predicted action marginal and the empirical
action distribution** (held-out) is `≤ COMPAT_TV_TOLERANCE = 0.15`. TV is used instead of absolute
conditional NLL because the *privileged* behavior depends on hidden state, so public-feature
conditional NLL is pathologically high for all worlds (≈24 nats) and non-discriminative — TV of the
marginal robustly separates worlds that reproduce the behavior distribution from those that do not.
The report **passes** only if `≥ MIN_SURVIVING_WORLDS = 3` survive **and** their held-out cross-world
counterfactual Q variance exceeds `DIVERSITY_FLOOR = 1e-8`. Diagnostics retained: per-world
conditional NLL, latent-free reference NLL, per-world TV, diversity.

**Required elements — all implemented:** (1) fitted behavior likelihood per world
(`behavior_prob`, now calibrated); (2) transition/reward diagnostics (`bellman_mse`, behavior NLL,
reference NLL); (3) rejection rule (TV gate); (4) diversity requirement; (5) constructed test where
an incompatible world fails; (6) constructed test where compatible worlds agree observationally but
vary counterfactually; (7) explicit `-inspired` disclosure (module + docstrings).

**Measured evidence (return-blind, 4,000/25):**

| Cell | σ_obs | compatible | world TV (max) | diversity | passes |
|---|---|---|---|---|---|
| Amur tiger | 0.2 | 20/20 | 0.017 | 3.4e7 | **True** |
| Egyptian vulture (sink) | 0.2 | 20/20 | 0.038 | 239 | **True** |
| Amur tiger | **0.0** | 20/20 | 0.022 | **3.9e-23** | **False** ("agree counterfactually") |

The σ_obs=0 row is the key scientific demonstration: with the observation gap (hidden-confounding
channel) removed, compatible worlds still reproduce behavior but **agree counterfactually**, so the
gate correctly refuses to interpret the vanishing variance as delphic uncertainty. Calibrated
behavior NLL fell from ≈25 → ≈2.0 and marginal TV from ≈0.83 → ≈0.02 after standardization.

---

## 5. Shared-model matching decision (D3, refinement 2)

RefPlan, OGSRL, and BA-MCTS all use the **same `PublicDynamicsEnsemble`** as their world-model
component, with a **common primary `ensemble_size = 5`**, identical `ridge = 1e-3`, identical public
belief features, the same bootstrap procedure, and the seed schedule already fixed in the pipeline.
RefPlan is **not** given a larger base ensemble (Phase 1B's proposed 7 for RefPlan alone was
rejected here). `ensemble_size = 7` is registered as a **shared** sensitivity (applied to all three
or none). Delphic's `world_count = 20` is a distinct algorithmic object and legitimately differs.
Disclosed limitation preserved (§9): the shared linear model means method differences are
planner/policy differences over one model class.

---

## 6. Test commands and results

Environment: `PYTHONPATH=src`, `MPLCONFIGDIR` → scratch. All run in the isolated worktree.

| Command | Result |
|---|---|
| `pytest tests/real/test_general_paper_mechanisms.py` | **16 passed** (RefPlan 4, BA-MCTS 4, OGSRL 4, Delphic 4) |
| `pytest tests/real/test_general_privacy.py` | **5 passed** |
| `pytest tests/real/test_general_paper_mechanisms.py -W error::RuntimeWarning` | **16 passed** (no numerical warnings) |
| `pytest tests/real/test_hidden_rk.py` | **9 passed** (unchanged privacy/routing gates) |
| `pytest tests/synthetic/test_methods.py` | **4 passed** |
| **`pytest tests/` (full suite)** | **187 passed** |

Constructed-mechanism highlights: RefPlan action-scores move with the posterior and diverge from
MOPO under a concentrated posterior; ≥2 members prefer different actions. OGSRL actor norm
0.11→2.17, `lambda_ood` 1.0→3.5 (binds), guardian overrides. BA-MCTS belief concentrates on the
generating member, moves off the root (not root-sampling), and `_belief_from_logweights` is stable
on `−inf`/extreme inputs. Delphic rejects the degenerate always-one-action world, passes at σ>0,
fails the diversity gate at σ=0.

---

## 7. Privacy and return-blindness evidence

- **Forbidden-name + forbidden-import** gates (`test_hidden_rk.py`) still pass: hidden methods fit
  with `pops_for`/`effects_for`/`actions_for`/`resolve_actions`/`NativeSolver.build` patched to raise.
- **Value-payload leak** (`test_general_privacy.py`): recursive scan of fitted state (excluding
  public config passthroughs) finds **no array/scalar equal to `K_ref` or `safety_threshold`** for
  any of the four; `observation_scale` (70.4) ≠ `K_ref` (250). A false positive
  (`native_vi_iterations=250` coinciding with a K_base of 250) was correctly excluded as a public
  config value, not fitted state.
- **Opaque pop_id**, **deterministic fit under same seed** (byte-identical arrays), and
  **structural-label relabel invariance** all pass for the four.
- **Return-blindness:** every diagnostic touches only model internals (dynamics RMSE, behavior NLL,
  guardian OOD rate, `terminated` prevalence, ensemble disagreement, belief KL, Delphic TV/diversity,
  timing). No `operational_return`/`true_return`/survival/ranking field was read; the probe imports no
  results file.

---

## 8. Canary runtime and resource measurements

Two deterministic canaries at 4,000/25, D5 primary HP (BA-MCTS 256/8, Delphic 20 worlds), 128-core
node; **peak RSS 158 MB/process**, CPU ≈125% (numpy threads).

**Per-step act latency (measured):** RefPlan 35.1 ms · OGSRL 64.2 ms · **BA-MCTS 608 ms (p95 667)**
· Delphic 0.9 ms. **Fit:** RefPlan 0.004 s · OGSRL 2.9 s · BA-MCTS 0.004 s · Delphic 7.1 s.

**Per method-cell wall** (eval = 5 seeds × 4 eps × 50 = 1,000 act calls): RefPlan 35 s · OGSRL 67 s ·
**BA-MCTS 608 s (10 min, 85% of all compute)** · Delphic 8 s.

**Full retained arm (288 cells × 4 methods = 1,152 method-cells):**

| Quantity | Value |
|---|---|
| Aggregate execution time | **57.5 task-hours** |
| Aggregate CPU-hours (×1.25) | **≈72 CPU-hours** |
| Allocated core-hours (1 core/task) | ≈57 core-hours |
| Peak memory / task | 158 MB (128 tasks ≈ 20 GB) |
| Wall @ 32 / 64 / 128 concurrent tasks | **1.8 h / 0.9 h / 0.45 h** |
| Concurrency for a 2 h / 4 h / 8 h window | **≥29 / ≥14 / ≥7** tasks |
| Queue delay | separate, cluster-dependent (not estimated) |

**Sensitivity arms (prepared, not launched):** BA-MCTS 512/12 ≈ **146 task-hours** alone (schedule
separately, off the primary arm); Delphic `{10,20,30}` and ensemble `{5,7}` are cheap. Recommendation:
run BA-MCTS 256/8 as primary; keep 512/12 to a small preregistered sensitivity subset.

---

## 9. Remaining deviations and scientific limitations

1. **OGSRL safety-cost channel is degenerate benchmark-wide.** `terminated` (state==0) has
   prevalence **0.0** on both a healthy and a depleted/sink population → `constant_single_class`
   surrogate, risk **identical across all 11 actions**. The safety-cost sub-constraint therefore
   cannot bind. Per the authorization I **did not invent a new safety target**; I measured/reported
   it. OGSRL's **OOD-guarded** ConOpt (its headline mechanism) *does* bind (`lambda_ood` moves,
   guardian overrides). **Decision required (§12):** keep the safety-cost constraint present-but-inert
   (disclose), drop it, or (separately authorized) redefine the public risk target.
2. **Shared linear model.** The three planners share one ridge-linear ensemble (matched) → method
   differences are planner/policy differences, not model-class differences (disclosed; RMSE reported).
3. **RefPlan prior policy omitted** — candidate sequences are enumerated/random, not drawn from a
   conservative offline policy → RefPlan-*inspired*.
4. **Delphic worlds are random-projection + calibrated heads, not ELBO-trained** → Delphic-*inspired*;
   the compatibility gate is a lightweight marginal-TV surrogate for `P^{π_b}(τ)` compatibility.
5. **BA-MCTS has no outer policy/value distillation** (approved as an evaluation-time planner, D2);
   per-step cost (608 ms) makes it 85% of compute.
6. **Adequacy of 4,000 not asserted from returns** — coverage is good (min action support 68–181,
   no action < 50) and diagnostics are healthy, but return-based sufficiency remains for the
   preregistered return-blind sensitivities.

---

## 10. Proposed frozen 4,000-transition manifests (NOT submitted)

`real_ecology_runs/general_corrected_prepared/manifest_general_hidden.csv` — **1,152 rows** = 288
cells × {refplan, ogsrl, bamcts, delphic}, `expose_rk=hidden`, `filter=learned`,
`target_rows=4000`, episode-preserving, MOPO excluded. Generator:
`scripts/make_general_corrected_manifest.py`. Registration + code snapshot + per-file digests in
`real_ecology_runs/general_corrected_prepared/`. Nothing was launched.

---

## 11. Acceptance gates and GO/NO-GO

| Gate | Status |
|---|---|
| 1 isolated corrected snapshot + digest | **PASS** (worktree, pre/post sha256, registration) |
| 2 full tests + new privacy/mechanism | **PASS** (187 passed) |
| 3 RefPlan action depends on deployment posterior | **PASS** |
| 4 OGSRL ConOpt trains + constraint non-degenerate | **PARTIAL**: ConOpt trains; **OOD** channel binds; **safety-cost channel degenerate** (reported, not invented away) |
| 5 BA-MCTS belief updates inside simulated histories | **PASS** |
| 6 Delphic passes compatibility + diversity gates | **PASS** |
| 7 no private-information access | **PASS** |
| 8 matched 4,000 inputs + matched shared-model HP | **PASS** (ensemble 5 across three) |
| 9 deterministic canaries + measured resources | **PASS** |
| 10 no returns inspected, no PLUS/MOOR interference | **PASS** |

**Verdict: `GO` for a limited corrected general-RL canary, conditional on the §12 OGSRL safety-cost
decision.** RefPlan, BA-MCTS, and Delphic-inspired are unconditionally ready; OGSRL is ready as an
**OOD-guarded** method with the safety-cost sub-constraint disclosed as benchmark-inert. This is a
GO for a *limited canary*, not for the full experiment (which needs separate authorization).

---

## 12. Decisions still requiring approval

| # | Decision | Options | Recommendation |
|---|---|---|---|
| A | **OGSRL safety-cost channel** (degenerate) | (a) keep constraint present-but-inert + disclose; (b) drop the safety-cost constraint, keep OOD guard only; (c) [separate authorization] redefine the public risk target (e.g. occupancy below a public band) | **(a)** for the canary (honest, no new target); consider (c) later under explicit authorization |
| B | **BA-MCTS compute** (85% of budget) | (a) 256/8 primary + 512/12 small sensitivity; (b) 128/5 primary to cut cost | **(a)** — 256/8 primary, 512/12 as a preregistered subset only |
| C | **Merge/commit of the isolated worktree** | (a) keep in worktree until canary reviewed; (b) commit the general-only changeset onto a branch now | **(a)** — leave isolated; commit only after canary review |
| D | **Naming** | reader-facing "paper-inspired adaptation" for all four incl. Delphic-inspired | already reflected in code docstrings + `04_…tex`; confirm |

**Stop.** No full experiment submitted; no comparative performance returns inspected. Awaiting review
and the §12 decisions (especially A) before any corrected general-RL canary is launched.
