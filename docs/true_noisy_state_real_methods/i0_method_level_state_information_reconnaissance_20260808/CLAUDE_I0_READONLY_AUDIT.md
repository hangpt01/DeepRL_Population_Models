# Claude independent read-only audit of I0 reconnaissance

Date: 2026-08-08 (Australia/Melbourne)
Auditor: Claude (independent read-only pass)
Audited stage: I0 method-level state-information reconnaissance
Controlling plan SHA-256: `238f238fca99b5dd043856ef90f5eb86b757e9a39249b25bf67eb235bc5cdf3f` (Revision 2, 661 lines) — **recomputed and matching**

Collision check: `CLAUDE_I0_READONLY_AUDIT.md` did not exist in the I0 directory before this pass. No existing file was overwritten.

---

## 1. Executive verdict

**I0 is trustworthy. Its two headline findings are confirmed. The plan cannot advance unamended: it requires material change, and the required change is larger than I0 states.**

Four conclusions:

1. **I0 integrity holds.** All three recorded checksums verify. The manifest is valid JSON and internally consistent with the report. HEAD, branch, worktree cleanliness, both frozen-track hashes, all 12 accepted episode artifacts, all 9 source hashes, both dataset files, the CPU profile, the diagnostic receipt and every compute figure reproduce exactly. Three defects were found — two wrong filenames (content hashes correct) and one under-specified hash recipe — plus two incomplete searches. None overturns any I0 finding.

2. **The observation-noise discrepancy is real, and I0 is right that Revision 2 is wrong.** OGSRL, BA-MCTS and EVD contain no direct `observation_noise_sigma` reference, but all three consume `PublicObservationFilter` outputs whose particle spread is set by sigma, and that dependence reaches their offline fitting as well as their online action selection. Revision 2's claim is literally correct about direct code references and **scientifically misleading** as an information-flow statement.

3. **I0 stopped one level too early, and the missing level reverses the strategic implication.** I0 describes the general methods as receiving "cached and runtime `BeliefState` objects", which reads as though they hold beliefs comparable to PLUS/MOOR. They do not. `PublicObservationFilter` is memoryless: it discards the prior belief every step, never reweights (weights are hard-coded uniform), uses no transition model, and ignores the action. Its particle cloud is a one-shot dispersion around the current observation. The accepted receipts confirm this quantitatively in all eight general cells: `filter_ess_fraction_mean = 1.00` exactly, and `filter_log_rmse_mean = 0.196–0.202` against a true sigma of 0.200 — **zero denoising**. So the general methods consume a *sigma-scaled observation-space feature*, not a calibrated latent-abundance belief. The general-versus-ecological hypothesis is therefore **not** undermined; it is sharpened. But the plan's wording, and its Arm T design, must be corrected to say so.

4. **Arm T needs material redesign for reasons I0 did not surface.** Three previously unrecorded blockers: the existing oracle primitive zeroes the observation-history context that four methods consume as features; OGSRL's safety budget is computed from the *dataset*, not from beliefs, so it stays calibrated to noisy observations under any belief-only Arm T; and rebuilding the public dynamics ensemble on exact states shrinks `residual_sigma`, which silently sharpens RefPlan's and BA-MCTS's model posteriors and so confounds the information axis with the transition-model-quality axis.

The OGSRL short-episode fix genuinely does not exist locally, in any ref including six Codex checkpoint refs that I0 did not enumerate. It does **not** block the registered pilot: both accepted datasets are exactly 160 × 25 and the cost horizon is exactly 25, so the raising branch is unreachable — but with a margin of exactly zero transitions.

I0's own verdict of `FAIL — PLAN OR PROVENANCE DISCREPANCY` was directed at the plan, not at itself. On the audit vocabulary required here, I0's provenance is sound and the plan needs material change: **REVISE**.

---

## 2. I0 integrity result

### 2.1 Checksums

`sha256sum -c I0_HASHES.sha256` run from the repository root:

| File | Result |
|---|---|
| `I0_RECONNAISSANCE_REPORT.md` | **OK** |
| `I0_READONLY_MANIFEST.json` | **OK** |
| `docs/true_noisy_state_real_methods/DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN.md` | **OK** |

Re-verified unchanged at the end of this audit. `I0_READONLY_MANIFEST.json` parses as valid JSON with 18 top-level keys and no schema contradiction against the report body.

### 2.2 Repository state

| Item | I0 claim | Independent result |
|---|---|---|
| Git HEAD | `77cd38adb11970d56ce4c96bf84de15fac3108ac` | **matches** |
| Branch | `e1-phase1-parity` tracking `origin/e1-phase1-parity` | **matches** |
| Worktree | clean apart from the new I0 directory | **matches** — `git status --porcelain` returns only `?? docs/.../i0_..._20260808/`; `git diff HEAD` is empty |
| Local branches | `e1-phase1-parity`, `main` | **incomplete** — see §2.5 |
| Plan line count | 661 | **matches** |
| Tags | (not recorded) | none exist |

**I0 modified only its own new planning artifacts.** No tracked file differs from HEAD. No accepted artifact, frozen track, config, registration or source file was touched.

### 2.3 Frozen tracks

| Track | Files | I0 hash | Independent recompute |
|---|---:|---|---|
| `src/tracks/ecological` | 149 | `951365d7…fb01` | **matches under `LC_ALL=C`** |
| `src/tracks/general` | 162 | `614524d7…e35b` | **matches under `LC_ALL=C`** |
| `provenance/frozen_tracks.sha256` | — | `319cd428…6ffc` | **matches** |

**Defect (recipe under-specification).** The recorded recipe `find TRACK -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum` does *not* record the locale. Under the ambient locale it produces `338d6d91…` and `044c9686…`; only `LC_ALL=C` reproduces I0's published values. Any future verifier following the recipe verbatim will get a spurious mismatch.

**Defect (hash scope).** Only 52 of 149 ecological files and 55 of 162 general files are git-tracked; the remaining 97 + 107 are `__pycache__/*.pyc`. The "frozen track" hash is therefore majority-bytecode and will change if any process imports these modules with bytecode writing enabled. This audit ran every Python probe with `PYTHONDONTWRITEBYTECODE=1` and re-verified both track hashes afterwards — both unchanged.

### 2.4 Path errors in the manifest (content correct, filenames wrong)

Two recorded paths do not resolve. In both cases the recorded **content hash is correct** and identifies a real file under a different name:

| Manifest path | Status | Actual file with that hash |
|---|---|---|
| `results/accepted/MATCHED_P10_144_METHOD_CELLS_RECEIPT.json` | **does not exist** | `results/accepted/MATCHED_P10_144_RECEIPT.json` (`a198c70f…3163` ✔) |
| `requirements-paper-faithful.txt` | **does not exist** | `requirements.txt` (`e3da5657…a2ea` ✔) |

Same error in the report body (§2 line 73 and the environment table). These are transcription defects, not fabrications — I0 hashed real bytes and mislabelled them.

### 2.5 Incomplete searches

- **Refs.** `local_branches` lists two branches. `git for-each-ref` shows eleven refs, including six `refs/codex/turn-diffs/checkpoints/…` working checkpoints. I0's fix search did not enumerate them. **I searched them; the conclusion is unchanged** (§6).
- **Documentation.** I0 reports no local fix documentation. `docs/fix_implement_general_RL/PHASE2E_OGSRL_FIX_AND_CANARY_PACKAGE_REPORT.md` exists, documents an OGSRL fix package with passing test evidence, and is directly relevant. It is a *different* fix (§6), but omitting it weakened I0's disposition.

### 2.6 Everything else verified

All matched exactly: 12/12 accepted episode-artifact hashes; 9/9 source hashes; both dataset file hashes and both embedded dataset hashes; `pyproject.toml`; `configs/cluster/accepted-8452Y.yaml`; `provenance/diagnostic_replay/DIAGNOSTIC_REPLAY_RECEIPT.json`; three of four accepted-table hashes (fourth is the mislabelled receipt, content correct); every compute figure in §8 including the cold-fit scaling (`18587.53`, `74350.12`, `20.652811 h`, `8682.653422321`, `4.8236963457 h`, `25.4765074568 h`).

**Missing required I0 output:** none. All three declared artifacts exist and verify.

---

## 3. Line-level observation-noise trace

### 3.0 The filter itself

`src/tracks/general/real_ecology_benchmark/beliefs.py`

```
586  class PublicObservationFilter:
589      def __init__(self, context: MethodContext, filter_cfg: FilterConfig):
595      def _particles(self, observation: float) -> np.ndarray:
597          if observation <= 0.0 or self.context.observation_noise_sigma == 0.0:
598              return np.full(n, max(float(observation), 0.0), dtype=np.float64)
599          sigma = max(self.context.observation_noise_sigma, 0.03)
600          values = np.exp(np.log(max(float(observation), 1e-12))
602                          + self._rng.normal(0.0, sigma, n))
625          log_weights=np.full(n, -np.log(n), dtype=np.float64),   # uniform, always
635      def update(self, belief, action, observation):
636          del action                                              # action ignored
637          return self._belief(observation, belief.observation, belief.timestep + 1)
```

**Selection.** `pipeline.py:246-259` — when `hides_rk(cfg.environment)`, every filter mode (`raw`, `learned`, `native_discrete`, `faithful_internal`) collapses to `PublicObservationFilter`, and `{"reference","ricker","true_family","oracle"}` are rejected outright. All four general methods run `expose_rk=hidden` with `filter: learned` (confirmed in each accepted `summary.json`), so all four receive this filter.

**Character of the estimate.** Line 637 rebuilds the belief from the current observation alone: the prior particle cloud is discarded, no transition model is applied, the action is deleted at line 636, and line 625 pins the weights uniform so no likelihood reweighting ever occurs. Since the environment emits `y = x · exp(η)`, `η ~ N(0,σ)` (`observation.py:26-27`), sampling `y · exp(ε)`, `ε ~ N(0,σ)` is the exact single-observation posterior **under an improper flat prior on log x** — its *width* is calibrated, but its point estimate is the raw observation. Verified by direct execution of the real filter (`PYTHONDONTWRITEBYTECODE=1`, tiger observations, `observation_scale = 61.40286116280366`, 256 particles):

| σ | feature[0] mean | feature[1] sd | mean_state trajectory vs raw observations |
|---:|---:|---:|---|
| 0.2 | 1.492342 | 0.139285 | ×1.004, ×1.027, ×1.001, ×1.016, ×0.998, ×1.009 |
| 0.1 | 1.494849 | 0.069883 | — |
| 0.02 | 1.497461 | 0.021006 | — |
| 0.0 | 1.498794 | 2.22e-16 | **exactly the raw observations** |

Feature-by-feature sigma dependence of `BeliefState.public_features` (`types.py:121-151`):

- **σ-dependent:** `mean`, `sd`, `q10`, `q50`, `q90` (all computed from `self.states`)
- **σ-invariant:** `extinct`, `ctx_prev_obs`, `ctx_obs`, `ctx_t`, `ess_frac` (`ess_frac ≡ 1.0` because weights are uniform)

**Corroborated by the accepted receipts** (all eight general cells, `summary.json`):

| Cell | method | `filter_log_rmse_mean` | `filter_coverage90_mean` | `filter_ess_fraction_mean` |
|---|---|---:|---:|---:|
| tiger | refplan | 0.20156 | 0.891 | 1.00 |
| tiger | ogsrl | 0.20075 | 0.891 | 1.00 |
| tiger | bamcts | 0.20186 | 0.891 | 1.00 |
| tiger | EVD | 0.19646 | 0.891 | 1.00 |
| fox | refplan | 0.19952 | 0.891 | 1.00 |
| fox | ogsrl | 0.19571 | 0.891 | 1.00 |
| fox | bamcts | 0.19976 | 0.891 | 1.00 |
| fox | EVD | 0.19685 | 0.891 | 1.00 |

Log-RMSE against true state equals σ = 0.2 to two significant figures in every cell. The filter's *stated uncertainty* is calibrated (coverage ≈ nominal 0.89–0.90, because it uses the true σ), but its *point estimate carries no information beyond the raw observation*. ESS = 1.00 exactly confirms weights are never updated.

### 3.1 RefPlan

`src/tracks/general/real_ecology_benchmark/methods/refplan.py`

1. **Direct sigma read: YES.** `refplan.py:146-147` — `obs_var = self.public_context.observation_noise_sigma ** 2 if self.hidden else self.env_cfg.observation_noise_sigma ** 2`. (`refplan.py:81` constructs a `LogNormalObservationModel` on the non-hidden path only.)
2. **Filter reads sigma: YES** (`beliefs.py:597-599`).
3. **Output changes with sigma: YES** (§3.0).
4. **Outputs consumed:** `belief.mean_state()` (`refplan.py:141`); the whole `belief` into `PublicParticlePlanner.plan_marginalized` (`refplan.py:97-98`); offline `beliefs.features` via `policy_prior.probabilities` (`refplan.py:63`).
5. **Classification:** *heuristic public-state estimate* (σ-scaled observation-space particle cloud) **plus** *explicit observation-likelihood modelling* — but the likelihood at `refplan.py:155-157` is over **model identity θ**, not over latent abundance. **Not** a calibrated latent-abundance belief.
6. **Conditions on:** offline fitting (behavior prior from `beliefs.features`), online updates (`observe` → `self.posterior`), planning, action selection. **All four.**
7. Revision 2 already lists RefPlan as a sigma consumer, so no correction is needed for RefPlan — but the plan should record that its sigma use is a *model-posterior* variance inflation, not latent-state filtering.

### 3.2 OGSRL

`src/tracks/general/real_ecology_benchmark/methods/ogsrl.py`

1. **Direct sigma read: NO.** `grep observation_noise_sigma ogsrl.py` → zero hits, in both tracks. I0 is correct.
2. **Filter reads sigma: YES.**
3. **Output changes with sigma: YES.**
4. **Outputs consumed:**
   - `ogsrl.py:928` — `self._public_state_features(belief.states)` (action selection, σ-dependent particles)
   - `ogsrl.py:930` — `belief.weights @ particle_probabilities`
   - `ogsrl.py:725` — `starts = np.asarray(belief.states)[start_indices]` (deployment risk rollouts)
   - `ogsrl.py:746` — `self.guardian.ood_probability(belief.states, actions)`
   - `ogsrl.py:420-425` — `_sample_cached_public` draws offline rollout starts as `rng.normal(features[:,0], features[:,1])`; **`features[:,1]` is the σ-controlled dispersion**
   - `ogsrl.py:769-775` — `PublicKNNGuardian.fit(beliefs.mean_states, …)`
5. **Classification:** *σ-dependent observation-space feature*. No latent-abundance belief; no observation-likelihood model anywhere in the file.
6. **Conditions on:** offline policy fitting (**yes** — σ literally sets the spread of the ConOpt actor's training start distribution at line 424), guardian fitting (yes), online action selection (yes), online belief update (n/a — the actor is fixed).
7. **"OGSRL does not use sigma" is correct only for direct code references and is scientifically misleading.** Sigma materially scales both the training start distribution and the deployment particle cloud.

### 3.3 BA-MCTS

`src/tracks/general/real_ecology_benchmark/methods/bamcts.py`

1. **Direct sigma read: NO.** Zero hits.
2. **Filter reads sigma: YES.** 3. **Output changes with sigma: YES.**
4. **Outputs consumed:**
   - `bamcts.py:239` — `belief.sample_indices(self.simulations, self.rng)` (uniform, since weights are uniform)
   - `bamcts.py:252` — `float(belief.states[index])` as the **tree root state** of every simulation
   - `bamcts.py:242` — `belief.contexts[:, 0]` (previous observation; **σ-invariant**)
   - `bamcts.py:299` — `belief.mean_state()` drives the model-posterior update
5. **Classification:** *σ-dependent observation-space feature* used as search-root dispersion, plus a categorical belief over **model identity**. Note `bamcts.py:305` uses `scale = residual_sigma + 0.05` — a hard-coded constant, **not** sigma. Not a latent-abundance belief.
6. **Conditions on:** offline dynamics ensemble (via the cache), online model posterior, planning (root dispersion), action selection. No separately fitted policy.
7. Same disposition as OGSRL: **literally correct for direct references, misleading as an information statement.**

### 3.4 EVD

`src/tracks/general/real_ecology_benchmark/methods/ensemble_value_disagreement.py`

1. **Direct sigma read: NO.** Zero hits.
2. **Filter reads sigma: YES.** 3. **Output changes with sigma: YES.**
4. **Outputs consumed:**
   - `ensemble_value_disagreement.py:182` — `belief.public_features(self.public_context.observation_scale)` (action selection)
   - `:113`, `:120`, `:135-141` — `beliefs.features` / `beliefs.next_features` fit the behavior reference and all 20 bootstrap Q members
5. **Classification:** *σ-dependent observation-space feature vector*. No state estimate, no online update, no observation-likelihood model.
6. **Conditions on:** offline value fitting (**yes** — all 20 Q members and the behavior reference), action selection (yes). No online update; rewards come from `dataset.rewards` unchanged.
7. Same disposition: **literally correct for direct references, misleading as an information statement.**

### 3.5 PLUS and MOOR (contrast case)

1. **Direct sigma read: YES.** `faithful_fit.py:563` — `observation_scale=float(context.observation_noise_sigma)`; consumed by `faithful_ecology.py:236-252 observation_likelihood()`. Also `methods/plus.py:76`, `methods/moor.py:104`.
2. n/a — they do not use `PublicObservationFilter`.
3. Yes — the likelihood width is sigma.
4. **Outputs consumed:** `CandidateBelief.probabilities` over `pomdp.abundance_grid` — eight of them plus a candidate posterior for PLUS (`plus_faithful.py:123,132,155,165`), one for MOOR (`moor_faithful.py:93-94,108-109`).
5. **Classification: calibrated likelihood-based belief over latent abundance.** `faithful_pomdp.py:186-207` is a genuine recursive Bayes filter: `predict` → `observation_vector` → `_normalize(predicted * likelihood)`. This is categorically different from `PublicObservationFilter`.
6. **Conditions on:** offline fitting, online updates, PBVI planning, action selection. **All four.**

### 3.6 The four required distinctions, kept separate

| Distinction | PLUS | MOOR | RefPlan | OGSRL | BA-MCTS | EVD |
|---|---|---|---|---|---|---|
| Consumes noisy observation **values** | yes | yes | yes | yes | yes | yes |
| Consumes a **σ-dependent filtered feature** | n/a (own filter) | n/a | yes | yes | yes | yes |
| **Explicitly models the observation likelihood** | yes (over abundance) | yes (over abundance) | yes (over **model identity**) | no | no (hard-coded 0.05) | no |
| Maintains a **calibrated latent-abundance belief** | **yes** | **yes** | no | no | no | no |

---

## 4. Corrected six-method uncertainty-role table

| Method | Direct sigma in policy file | Sigma reaches it via the shared filter | State estimate it actually holds | Denoising achieved | Where sigma enters | Correct one-line role |
|---|---|---|---|---|---|---|
| **adapted PLUS** | **Yes** — `faithful_fit.py:563` → `faithful_ecology.py:244-251` | n/a — uses `faithful_internal` POMDP beliefs | **Calibrated likelihood-based belief over latent abundance**, one per Ricker candidate, marginalized over an 8-candidate posterior | Recursive Bayes with dynamics | offline fit, online update, planning, action selection | Joint latent-state + structural-uncertainty ecological baseline |
| **adapted MOOR** | **Yes** — same path | n/a | **Calibrated likelihood-based belief over latent abundance**, single misspecified Ricker model | Recursive Bayes with dynamics | offline fit, online update, planning, action selection | Single-model latent-state ecological baseline |
| **RefPlan-inspired** | **Yes** — `refplan.py:146-147` | **Yes** | **Heuristic public-state estimate** (σ-scaled observation cloud) + categorical posterior over **model identity** | **None** (log-RMSE 0.200/0.200) | offline prior fit, online model posterior, planning, action selection | General method that models the observation likelihood **over models, not over abundance** |
| **OGSRL-inspired** | **No** | **Yes** | **σ-dependent observation-space feature**; no belief | **None** (0.196–0.201) | offline actor + guardian fitting, action selection | Guarded fixed actor whose training start spread and deployment cloud are both σ-scaled |
| **BA-MCTS-inspired** | **No** | **Yes** | **σ-dependent observation-space feature** as search-root dispersion + model-identity belief | **None** (0.200/0.202) | offline ensemble, online model posterior, planning, action selection | Online model-belief search over σ-dispersed observation roots |
| **EVD pessimism** | **No** | **Yes** | **σ-dependent observation-space feature**; no state estimate, no online update | **None** (0.196/0.197) | offline Q + behavior fitting, action selection | Value-disagreement control on σ-scaled observation features |

**MOBILE:** confirmed **not implemented** — no executable registration or implementation in either track. I0 correct.

**The load-bearing correction.** Revision 2's binary "uses sigma / does not use sigma" is wrong in both directions. Every method's inputs are sigma-dependent, so the "does not use sigma" column must go. But the *kind* of estimate differs sharply and that difference survives: **only PLUS and MOOR maintain a calibrated latent-abundance belief**; the four general methods receive an observation-space feature whose dispersion is scaled by sigma and whose point estimate is the raw observation. The architectural hypothesis is intact — it just has to be stated as a contrast between *state representations*, not between *presence and absence of sigma*.

---

## 5. Episode-count reproduction

Independently recomputed from the two accepted NPZ files (file hashes re-verified against the manifest before reading: `9e9c3a6d…f771c` tiger, `7d1b2fc8…87e19` fox — both match).

Validity criteria applied independently: contiguous row block per episode; `timestep` running 0…L−1 with no gaps; `next_observations[i] == observations[i+1]` exactly within an episode; no `done` before the final row; exactly one of `terminated`/`truncated` set on the final row.

| Quantity | Tiger × Allee × σ0.2 | Fox × Allee × σ0.2 |
|---|---:|---:|
| Rows | 4,000 | 4,000 |
| **Total episodes** | **160** | **160** |
| **Full 25-step episodes** | **160** | **160** |
| **Genuine short terminated episodes** | **0** | **0** |
| **Truncated (short) episodes** | **0** | **0** |
| **Invalid / incomplete episodes** | **0** | **0** |
| **Minimum episode length** | **25** | **25** |
| **Maximum episode length** | **25** | **25** |
| Length distribution | `{25: 160}` | `{25: 160}` |
| Final `terminated` / `truncated` | 0 / 160 | 0 / 160 |
| `terminated` flags anywhere | 0 | 0 |
| Contiguity / timestep / chaining / early-done checks | all pass | all pass |

**Exact reproduction of I0's table.** Independently corroborated by the accepted receipts: `fit_diagnostics.behavior_cost_episode_count = 128.0` with `training_split.train_episodes = 128` and `holdout_episodes = 32` (= 160), and `terminated_prevalence = 0.0`.

### Is the OGSRL short-episode path reachable from these datasets?

**No — but the margin is exactly zero.**

`ogsrl.py:551-568` raises when `len(cost) < horizon`. `horizon = self.planner_cfg.ogsrl_cost_horizon`, which is `25` by default (`config.py:535`) and `25` in the registered config (`configs/tracks/general/general_phase2e_full_sigma01_02.yaml:32`), confirmed as-run by `fit_diagnostics.cost_horizon = 25.0`. Every episode has exactly 25 transitions. The predicate is `25 < 25` → False.

The path is unreachable by a margin of **one transition**. A single 24-step episode in either cell would abort OGSRL fitting outright. I0 reported "zero short episodes" correctly but did not record that the safety margin is nil — material for anything that touches episode boundaries.

### Would an Arm T reconstruction reuse these episode boundaries?

**It depends entirely on a design choice the plan has not yet made, and the two options differ sharply:**

- **Belief-only Arm T** (replace the belief cache / runtime state, reuse `public.npz` unchanged): boundaries are **identical**. `_fit_public_safety_scale` takes only `dataset` — it reads `dataset.observations`, `dataset.next_observations` and `dataset.episode_id`, and never touches `beliefs`. No new short-episode exposure. The missing fix stays irrelevant.
- **Dataset-regenerating Arm T** (collect new exact-state trajectories): **creates new exposure**. New seeds can produce genuine terminated collapses shorter than 25, which would hit the unfixed raise immediately. The fix would become a hard prerequisite.

The plan must state which of these Arm T is. It currently does not.

---

## 6. OGSRL-fix disposition

Searched: working tree; all 11 refs including six `refs/codex/turn-diffs/checkpoints/…`; every reachable commit (7 total); every blob version of `ogsrl.py` in the object database; `git stash` (empty); dangling objects; `docs/`; `archive/`; `tests/`; `*.patch` / `*.diff` (none exist). No network access, no branch changes.

| Question | Finding |
|---|---|
| Does the fix exist anywhere locally? | **No.** Exactly **one** blob version of `src/tracks/general/…/methods/ogsrl.py` exists in the entire object database (`268a9b6d…`), and it contains both the raise and the `len(cost) < horizon` guard. |
| Applied, proposed only, or lost? | **Neither applied nor recoverable.** No branch, commit, ref, stash, patch or handoff carries it. On the evidence it was described in planning prose but never committed to this repository. |
| Do its three stated tests exist? | **No.** 42 test files searched. No short-episode tolerance test exists. Existing tests *avoid* the guard by setting `ogsrl_cost_horizon` to 2, 4 or 6 (`test_hidden_rk.py:130,201`; `test_general_paper_mechanisms.py:335`; `test_general_privacy.py:42`) rather than testing short-episode behaviour. |
| Any passing record for those three tests? | **No** — for those three. **But see the conflation below.** |
| Is the current code unfixed? | **Yes.** `ogsrl.py:563-567` raises `ValueError("OGSRL behavior budget needs {horizon} transitions per episode; found {len(cost)}")`. |

### The conflation I0 missed

`docs/fix_implement_general_RL/PHASE2E_OGSRL_FIX_AND_CANARY_PACKAGE_REPORT.md` documents a **real, applied, passing OGSRL fix** — but a *different* one: the pathwise-cost / common-random-numbers / `ogsrl_deployment_rollouts=256` package. Its recorded source hash for `ogsrl.py` is `da965bc320f3cbfaf381a46aaa77a7eb0ef7c3b6b7b493bee02e338544e6993b`, which is **byte-identical to the current working-tree file** and to I0's `general_ogsrl` manifest entry. It records **31 passed** targeted and **203/203 passed** full-suite.

Critically, that report states the cost horizon is **unchanged** at 25, that deployment "always simulates exactly 25 public-model steps … never silently shortened", and that the behavior budget equals "the exact train-only **complete-episode** mean". It therefore *depends on* complete 25-step episodes rather than tolerating short ones.

**Disposition:** Revision 2 appears to have conflated the accepted Phase 2E OGSRL fix package (real, applied, tested, passing) with a short-episode/absorbing-terminal fix (never implemented here). The current file is the accepted, tested Phase 2E version — and it raises on short episodes.

### Do the current pilot datasets require the fix?

**No.** Both cells are 160 × 25 with horizon 25 (§5). Zero accepted rows would change.

### Does its absence block anything?

| Scope | Blocked? |
|---|---|
| I1 / I2 (registration, adapter design, unit tests) | **No** |
| The registered two-cell pilot on the existing accepted datasets | **No** — the raising branch is unreachable |
| Gate G1 as currently written | **Yes** — G1 (plan lines 259-267) mandates running "the OGSRL regression tests" that do not exist. G1 is unsatisfiable as written. |
| Gate G2's `EXPLAINED FIX DELTA — REBASELINE REQUIRED` branch | **Moot** — plan line 288 conditions on a fix that was never applied, so no such delta can arise. |
| Any future new-seed dataset generation | **Yes** — this is the only genuine blocker, and only if Arm T regenerates data. |

**Net: absence of the fix does not affect the registered pilot; it blocks only future new-seed data, and it makes two gate clauses unsatisfiable as written.**

---

## 7. Arm T feasibility audit

### 7.1 Verification of I0's component-change rows

| Method | I0 row | Audit verdict |
|---|---|---|
| PLUS — eight Ricker fits stay frozen | Same | **Confirmed.** Fits are offline from `dataset` + `context` (`faithful_fit.py`); the proposed direct belief assignment is strictly post-fit and runtime-only. 39-file artifact tree hash recorded for both cells. |
| MOOR — one Ricker fit stays frozen | Same | **Confirmed**, same argument, 9-file tree. |
| RefPlan — exact abundance without refitting | Secondary diagnostic only | **Confirmed as stated**, with an added blocker (§7.3 item 1). |
| OGSRL / BA-MCTS / EVD — require retraining | Rebuild | **Confirmed**, with an added confound (§7.3 item 3). |
| Sigma-zero / epsilon route is invalid | Verified | **Confirmed at source**, and strengthened (§7.2). |
| Oracle primitive exists but is unusable for hidden methods | Requires implementation | **Confirmed** — `pipeline.py:249-253` rejects `oracle` when hidden; `pipeline.py:565-568` `run_oracle_state_ablation` raises "oracle-state beliefs are evaluator-private and unavailable to hidden methods". Evaluator hooks exist at `evaluator.py:44` and `evaluator.py:144`. |

### 7.2 Why sigma-zero is wrong — confirmed, plus one new argument

Every I0 source claim verifies:

- `faithful_ecology.py:238` — `y = max(float(observation) / self.survey_scale, 0.0)` ✔
- `faithful_ecology.py:244-245` — `observation_scale <= 1e-12` returns `np.isclose(x, y)`, an all-zero vector for off-grid truth ✔
- `faithful_pomdp.py:18-23` — `_normalize` maps zero/non-finite total to **uniform** ✔
- `faithful_pomdp.py:171-172` — `initial_belief` normalizes `prior * likelihood` → all-zero becomes uniform ✔
- `faithful_pomdp.py:192-194` — `update` retains the **predicted** belief when `evidence <= LIKELIHOOD_FLOOR` (`= 1e-300`, line 15) ✔

**New supporting argument.** `MechanisticModel.parameter_hash()` (`faithful_ecology.py:254-283`) includes `observation_scale` in its hashed scalars. Driving sigma or `observation_scale` to zero would change the parameter hash, so the PLUS/MOOR fitted artifacts could **not** remain byte-identical. The sigma-zero route fails the frozen-artifact requirement independently of the likelihood-collapse argument. This strengthens the plan's existing prohibition.

### 7.3 Three Arm T blockers I0 did not surface

**1. The oracle primitive silently zeroes the observation-history context.**

`beliefs.py:770-772` — `OracleStateFilter.set_true_state` constructs `BeliefState(np.full(n, state), np.zeros((n, 3)), …)`. That second argument is `contexts`. But `public_features` (`types.py:137-140`) emits `context_mean` as features 7–9 (`prev_obs`, `obs`, `timestep`), and `bamcts.py:242` reads `belief.contexts[:, 0]` directly. Routing Arm T through the existing oracle primitive would therefore not merely replace the state — it would **zero the previous-observation, current-observation and timestep features that OGSRL, BA-MCTS, EVD and RefPlan consume**, including the timestep signal. That is a severe, silent distribution shift on top of the intended intervention. Additionally, `OracleStateFilter` subclasses `RawObservationFilter(ParticleFilter)` and requires `env_cfg`, a simulator configuration hidden methods must not hold — a privacy question, not just an engineering one.

**Consequence:** Arm T needs a purpose-built exact-state adapter that *preserves* the public observation-history context. The existing primitive must not be reused as-is.

**2. OGSRL's safety budget is dataset-derived, so a belief-only Arm T leaves it mis-calibrated.**

`_fit_public_safety_scale` (`ogsrl.py:551-590`) computes `s_low` from `dataset.observations`, the step cost from `dataset.next_observations`, and the budget from `dataset.episode_id` — **never from beliefs**. So:

- **Belief-only Arm T:** OGSRL receives exact abundance at deployment while its `s_low`, `safety_budget` and `deployment_safety_limit` remain calibrated to the *noisy observation* scale. Because `E[y] = x·exp(σ²/2)`, the observation scale is biased ≈ +2% relative to abundance (accepted receipts: tiger `mean_observation_mean` 121.55 vs `mean_true_state_mean` 119.49). The constraint would be evaluated against the wrong scale.
- **Dataset-substituting Arm T:** `s_low` and the entire behavior budget change → this **silently moves the safety axis**, not the information axis.

Neither option is neutral. The plan must register which, and declare the consequence.

Related: `_public_low_abundance_cost` (`ogsrl.py:427-430`) applies the shortfall to `following`, the *predicted next observation*. Under exact-state inputs the ensemble predicts abundance, so the cost channel changes meaning even with `s_low` held fixed.

**3. Rebuilding the dynamics ensemble on exact states confounds the transition-model-quality axis.**

Arm T requires rebuilding `PublicDynamicsEnsemble` for OGSRL, BA-MCTS and RefPlan. Fitting on exact abundance instead of noisy observations removes observation noise from the regression residual, so each member's `residual_sigma` **shrinks**. That value is load-bearing downstream:

- `refplan.py:155` — `sigma = max(member.residual_sigma + obs_var ** 0.5, 0.03)`
- `bamcts.py:305` — `scale = residual_sigma + 0.05`

Both are model-posterior likelihood widths. A smaller `residual_sigma` sharpens both posteriors, changing model-selection behaviour for reasons unrelated to state information. **Arm T would therefore move the transition-model-quality axis and the information axis simultaneously** — precisely the confound the pilot exists to avoid. This must be measured and reported (record `residual_sigma` per member per arm), or neutralized by registration.

### 7.4 Route-by-route flags

| Proposed Arm T route | Distribution shift | Hidden-info leak | Silently moves transition-model-quality axis |
|---|---|---|---|
| PLUS/MOOR direct grid-delta assignment (`raw_true / survey_scale` → nearest `abundance_grid` bin, no likelihood call) | Low — belief becomes a point mass; must not double-update | **Check required:** regime dimension handling; must not expose family, `r`, `K`, private safety state | **No** — fits stay frozen and hash-provable |
| Sigma / `observation_scale` → 0 or ε | **Severe** — uniform or predicted-belief fallback | no | **Yes** — changes `parameter_hash`, breaks frozen artifacts |
| Reuse `OracleStateFilter` for hidden methods | **Severe** — zeroes contexts 7–9 (§7.3.1) | **Yes** — requires `env_cfg`, evaluator-private by design | indirect |
| RefPlan frozen-fit deployment-only exact particles | Moderate — train/deploy feature mismatch; must be labelled secondary | manageable | No |
| General-method end-to-end rebuild on exact states | Moderate–high | manageable | **Yes** (§7.3.3) |
| Regenerate datasets with exact-state trajectories | High | manageable | **Yes** — plus reactivates the OGSRL short-episode raise (§5) |

### 7.5 Does a true-state input alter fitting, or only online representation?

| Method | Transition fitting | Reward fitting | Policy fitting | Online state representation only? |
|---|---|---|---|---|
| PLUS | No — frozen | No | No | **Yes** |
| MOOR | No — frozen | No | No | **Yes** |
| RefPlan | **Yes** (primary arm) | No, if the reward contract holds | **Yes** (behavior prior) | No |
| OGSRL | **Yes** | No | **Yes** (actor, guardian, safety scale/budget) | No |
| BA-MCTS | **Yes** | No | n/a (online search) | No |
| EVD | n/a (no transition model) | n/a (raw rewards) | **Yes** (20 Q members + behavior reference) | No |

**The deployment-only estimand is available for exactly two of six methods (PLUS, MOOR).** I0 states this; it is correct and it is the single most important structural constraint on the pilot's interpretation.

---

## 8. Accepted-anchor verification

All six required anchors independently confirmed against primary accepted records.

| Anchor | Required value | Independent result | Source |
|---|---|---|---|
| Tiger/Allee/0.2 OGSRL return | `4.71629840155769` | **exact** | `MATCHED_P10_144_METHOD_CELLS.csv`; corroborated by `summary.json` `operational_return_mean` at source |
| Tiger fixed-action-10 return | `4.224420899026084` | **exact** | `constant_reward_screen.csv`, `reward == "Current"` |
| Fox/Allee/0.2 MOOR return | `11.610936360689793` | **exact** | `m1_m15_comparison.csv`, `A2_crab_eating_fox_allee_s0p2` / moor / M6 `return_mean` |
| Fox fixed-action-1 return | `10.153605271956147` | **exact** | `constant_reward_screen.csv`, `reward == "Current"` |
| Fox PLUS headroom | `0.8131401783020511` | **exact** — `10.966745450258198 − 10.153605271956147` recomputed in float64 | derived |
| Fox PLUS switch-from-MAP fraction | `0.709` | **exact** — `{"switch_vs_MAP":0.709,"switch_vs_uniform":0.131,"n_decisions":1000}` | `m1_m15_comparison.csv` / plus / M2 |

**Serialization note confirmed.** The accepted CSV serializes fox MOOR as `11.61093636068979`; the replay record carries `11.610936360689793` and its own `parity_delta` of `3.552713678800501e-15`. I0's note is exactly right.

**Precision note.** Tiger's "best fixed action 10" holds under the accepted `Current` reward mode. Under the `OptionA` variant, action 0 scores higher (`4.938492770324094`). I0's anchor is correct but the report does not state the reward-variant qualifier; it should.

### Activity-cell labelling

| Cell / method | Accepted return | Headroom vs best fixed | Independent note |
|---|---:|---:|---|
| Tiger RefPlan | 3.454471 | −0.76995034 | active, no headroom |
| **Tiger OGSRL** | **4.716298** | **+0.49187750** | **only tiger method with positive headroom** |
| Tiger BA-MCTS | 1.278086 | −2.94633528 | active, no headroom |
| Tiger EVD | −11.637269 | −15.86168979 | active, no headroom |
| Tiger PLUS | 4.224421 | **+0.00000000** | **exactly equals fixed action 10** — constant policy |
| Tiger MOOR | 4.224421 | **+0.00000000** | **exactly equals fixed action 10** — constant policy |
| Fox RefPlan | 7.948142 | −2.20546303 | active, no headroom |
| Fox OGSRL | 10.130383 | −0.02322270 | **exactly equals fixed action 2** — constant policy |
| Fox BA-MCTS | 10.201364 | +0.04775857 | active, marginal headroom |
| Fox EVD | 10.397429 | +0.24382404 | active + headroom |
| **Fox PLUS** | **10.966745** | **+0.81314018** | **active + headroom** |
| **Fox MOOR** | **11.610936** | **+1.45733109** | **active + headroom** |

All twelve I0 headroom values reproduce; residual differences are last-digit display rounding (≤ 2e-8).

**Independent strengthening of I0's classification.** I0 inferred inactivity from action entropy. I confirmed it more strongly by **exact return equality with the fixed-action screen**: tiger PLUS and tiger MOOR return *bit-identically* the fixed-action-10 value, and fox OGSRL returns *bit-identically* the fixed-action-2 value. These are constant policies, not merely low-entropy ones.

**Verdict on the labels:**

- **Tiger as the general-method activity cell — SUPPORTED.** Both ecological methods are provably constant-action there (zero headroom, exact equality), and OGSRL is the sole method with positive headroom.
- **Fox as the ecological-method activity cell — SUPPORTED.** PLUS (+0.813) and MOOR (+1.457) are the two largest headrooms and both are decision-active (accepted `action_entropy_mean` 0.448599 / 0.258507; PLUS switches from MAP on 70.9% of replay decisions), while OGSRL degenerates to a constant policy and BA-MCTS's +0.048 is marginal.

**Reproducibility gap.** The per-episode action counts and entropies in I0 §7 for the four *general* methods cannot be reproduced from the artifacts I0 lists: the accepted `episodes.csv` files are episode-level summaries with no action column (`model, episode, seed, block_seed, operational_return, true_return, collapse_entry, collapse_entries, collapse_entry_timestep, unsafe_fraction, mvp_fraction, mvp_breach`), and `action_entropy_mean` is `NaN` for all general rows in the accepted CSV. I0 did not record which file those numbers came from. The *classification* is independently confirmed by the exact-equality argument above, so this is a citation defect rather than a data defect — but I1 should record the source path.

---

## 9. Required plan amendments

Exact amendments for Codex to apply to `DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN.md`. **This audit did not apply them.**

### Which outcome is supported

Of the five candidate outcomes, **two apply jointly**:

- ✅ **The Arm T design needs material revision** (§7.3 — three unaddressed blockers).
- ✅ **The pilot remains useful but must be reframed as comparing different state representations rather than presence versus absence of state-uncertainty handling.**

Explicitly **not** supported:

- ❌ *"The plan remains valid with wording corrections."* Wording alone does not fix §7.3.
- ❌ *"The hypothesis is already undermined because all methods receive meaningful sigma-dependent state estimates."* **This is the reading to avoid.** The general methods receive sigma-dependent *features*, but with `filter_log_rmse ≈ σ` and `ESS ≡ 1.0` they achieve **zero denoising** and hold no latent-abundance belief. The PLUS/MOOR versus general-method distinction is real and measurable. Sigma-dependence is not the same as state estimation.
- ❌ *"The pilot is blocked."* All required local evidence was available; both accepted datasets are clean; anchors verify; PLUS/MOOR frozen-fit reuse is sound.

### A1 — Line 67, §2 record-verification table

Replace the row:

> `| Only PLUS, MOOR and RefPlan explicitly consume the observation-noise model | **VERIFIED** | ... OGSRL, BA-MCTS and EVD receive the constant in context but do not use it. |`

with:

> `| Only PLUS, MOOR and RefPlan reference the observation-noise model in their policy files | **PARTLY CONTRADICTED** | Correct for direct references only: `grep observation_noise_sigma` returns zero hits in `ogsrl.py`, `bamcts.py` and `ensemble_value_disagreement.py`. But in hidden mode `pipeline.py:246-259` routes every filter mode to `PublicObservationFilter`, whose particle spread is set by `context.observation_noise_sigma` (`beliefs.py:597-599`). All four general methods consume the resulting σ-dependent features both offline (cached `beliefs.features`) and online. The filter is memoryless, never reweights (`beliefs.py:625`, ESS ≡ 1.0) and achieves no denoising (accepted `filter_log_rmse_mean` 0.196–0.202 against σ = 0.200), so this is a σ-scaled observation-space feature, **not** a calibrated latent-abundance belief. |`

### A2 — Lines 88-90, §3 method-roles table

Replace the `observation_noise_sigma` characterizations:

- **Line 88, OGSRL:** replace `public observation-space belief/proxies; does not use sigma explicitly` with
  `σ-dependent public observation-space features from the shared filter; no direct sigma reference in ogsrl.py; no latent-abundance belief. Sigma scales both the offline actor's rollout start dispersion (ogsrl.py:420-425) and the deployment particle cloud (ogsrl.py:928, 725, 746).`
- **Line 89, BA-MCTS:** replace `bucketed public observation, not a calibrated latent-abundance belief; does not use sigma` with
  `σ-dependent public observation particles as search roots (bamcts.py:252); not a calibrated latent-abundance belief; no direct sigma reference. Its model-posterior likelihood width is a hard-coded 0.05, not sigma (bamcts.py:305).`
- **Line 90, EVD:** replace `current public features; no latent-state filter and no sigma use` with
  `σ-dependent public observation features (ensemble_value_disagreement.py:182) used for both offline Q/behavior fitting and action selection; no latent-state filter; no direct sigma reference.`

Add a footnote under the table:

> All six methods' inputs depend on `observation_noise_sigma`. The discriminating property is **not** sigma consumption but the kind of state estimate maintained: PLUS and MOOR run a recursive Bayes filter over latent abundance (`faithful_pomdp.py:186-207`); the four general methods hold a memoryless, uniformly weighted observation-space cloud that achieves no denoising. RefPlan models an observation likelihood, but over **model identity**, not over abundance.

### A3 — §1, lines 47-52, motivating hypothesis

Replace the second bullet:

> `- several general methods represent model/value/support uncertainty but do not use the registered observation-noise model;`

with:

> `- the general methods represent model/value/support uncertainty over an observation-space state summary whose dispersion is σ-scaled but which performs no denoising and maintains no latent-abundance belief;`

### A4 — Line 198, §5 Arm O

Replace:

> `OGSRL, BA-MCTS and EVD retain their accepted method logic and therefore do not begin consuming sigma merely for this pilot.`

with:

> `OGSRL, BA-MCTS and EVD retain their accepted method logic. Their inputs already depend on sigma through the shared `PublicObservationFilter`; this pilot introduces no new sigma consumption for them.`

### A5 — §5 Arm T, insert after line 192 (new mandatory subsection)

> **Arm T exact-state adapter requirements (from the I0 audit).**
>
> 1. **Do not reuse `OracleStateFilter` for hidden methods.** `beliefs.py:770-772` sets `contexts = np.zeros((n, 3))`, which would zero the previous-observation, current-observation and timestep features that `public_features` exposes as elements 7–9 and that `bamcts.py:242` reads directly. It also requires `env_cfg`, a simulator configuration hidden methods must not hold. The Arm T adapter must preserve the public observation-history context unchanged and alter only the state particles.
> 2. **Register the Arm T dataset decision explicitly.** State whether Arm T reuses the accepted `public.npz` unchanged (belief-only) or regenerates trajectories. Belief-only preserves the 160 × 25 episode boundaries and keeps the OGSRL short-episode path unreachable; regeneration reactivates it, since `ogsrl_cost_horizon = 25` equals the episode length exactly and the guard at `ogsrl.py:563` has a margin of one transition.
> 3. **Register OGSRL's safety-scale treatment.** `_fit_public_safety_scale` (`ogsrl.py:551-590`) derives `s_low`, the behavior budget and `deployment_safety_limit` from `dataset` alone, never from beliefs. Under a belief-only Arm T these stay calibrated to the noisy observation scale (biased ≈ +2%, since `E[y] = x·exp(σ²/2)`) while the state input is exact; substituting true abundance into the dataset instead moves the safety axis. Neither is neutral: choose one, declare it, and report the consequence.
> 4. **Measure the transition-model-quality confound.** Rebuilding `PublicDynamicsEnsemble` on exact states shrinks each member's `residual_sigma`, which feeds RefPlan's model-posterior width (`refplan.py:155`) and BA-MCTS's (`bamcts.py:305`). Record `residual_sigma` per member per arm and report it alongside the primary estimand; an unreported shift confounds the information axis with the transition-model-quality axis.
> 5. **PLUS/MOOR frozen-artifact proof.** Direct grid assignment leaves the fitted artifacts untouched, but the prohibition on driving sigma or `observation_scale` toward zero is now doubly binding: `MechanisticModel.parameter_hash()` (`faithful_ecology.py:254-283`) hashes `observation_scale`, so any such change breaks byte-identity. Prove the 39-file PLUS and 9-file MOOR trees identical by hash in I2.

### A6 — §7 Gate G1, lines 259-267

G1 is unsatisfiable as written: the three OGSRL regression tests do not exist. Replace the gate body with:

> Run the repository self-tests. The three OGSRL short-episode regression tests named in earlier revisions **do not exist in this repository** and no passing record for them exists (audited 2026-08-08 across all 11 refs, 7 commits, every `ogsrl.py` blob, and 42 test files). Prior references to a passing OGSRL fix appear to conflate the accepted **Phase 2E pathwise-cost package** (`docs/fix_implement_general_RL/PHASE2E_OGSRL_FIX_AND_CANARY_PACKAGE_REPORT.md`, 31 targeted + 203/203 full-suite passed, `ogsrl.py` = `da965bc3…6e993b`, byte-identical to the current tree) with a short-episode/absorbing-terminal fix that was never implemented here.
>
> G1 therefore requires only the existing suite to pass. **If and only if** Arm T regenerates datasets, add as a prerequisite: implement the short-episode handling on a named branch, add tests for (a) bit-identical behavior on full-length data, (b) correct absorbing-terminal padding for legitimate one- and two-step collapses, (c) rejection of short nonterminal truncations, and record a passing receipt. For the registered belief-only pilot this prerequisite does not apply: both accepted cells are exactly 160 × 25 with cost horizon 25, so `ogsrl.py:563` is unreachable and zero accepted rows are affected.

### A7 — §7 Gate G2, lines 286-291

Delete or neutralize the `EXPLAINED FIX DELTA — REBASELINE REQUIRED` branch. The fix was never applied, so no fix-attributable delta can arise. Replace with:

> The documented OGSRL absorbing-terminal fix is **not present** in this repository and was never applied to the accepted code path. No fix-attributable delta is possible. Any difference from the accepted artifact is therefore unexplained and is an unconditional stop condition.

### A8 — Line 73, §2

Replace `It also documents a branch-isolated OGSRL short-episode fix.` with:

> Earlier handoffs referred to a branch-isolated OGSRL short-episode fix. The I0 audit found no such branch, commit, ref, patch or test anywhere locally; the reference appears to conflate the accepted Phase 2E OGSRL pathwise-cost package with an unimplemented short-episode fix. HEAD is `77cd38adb11970d56ce4c96bf84de15fac3108ac` on `e1-phase1-parity`, worktree clean.

### A9 — §14 audit questions

- **Question 3** (line 653) is **answered and closed:** no live branch contains the fix; it does not exist locally; its three tests do not exist; the "passing" record belongs to the different Phase 2E package. Replace the question with that finding.
- **Question 4** (line 655) is **answered and closed:** zero genuine short terminated episodes in both cells; both are exactly 160 × 25; minimum = maximum = 25.
- **Add Question 7:** Does Arm T reuse the accepted `public.npz` unchanged, or regenerate trajectories? This determines whether the OGSRL short-episode fix is a prerequisite and whether episode boundaries are preserved.
- **Add Question 8:** For OGSRL Arm T, is the safety budget recomputed on exact abundance (moving the safety axis) or held at the accepted noisy-scale value (mis-calibrated against an exact-state input)?

### A10 — Two provenance corrections carried into I1

- `results/accepted/MATCHED_P10_144_METHOD_CELLS_RECEIPT.json` → the real path is `results/accepted/MATCHED_P10_144_RECEIPT.json` (hash `a198c70f…3163` correct).
- `requirements-paper-faithful.txt` → the real path is `requirements.txt` (hash `e3da5657…a2ea` correct).
- Amend the frozen-track recipe to `LC_ALL=C find TRACK -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum`, and record that the hash currently covers 204 `__pycache__/*.pyc` files; consider excluding bytecode so the frozen-track hash is stable under import.
- Record the source file for the general-method action-entropy figures in I0 §7 (not reproducible from the listed `episodes.csv` artifacts, which contain no action column).

---

## 10. Authorization status

**Performed under read-only audit authorization:**

- Read `I0_RECONNAISSANCE_REPORT.md`, `I0_READONLY_MANIFEST.json`, `I0_HASHES.sha256` and the controlling plan in full.
- Recomputed all three declared checksums, both frozen-track aggregates, 9 source hashes, 12 accepted episode-artifact hashes, 2 dataset file hashes, and 4 config/receipt/table hashes.
- Read-only Git inspection: `status`, `rev-parse`, `for-each-ref`, `branch -a`, `tag`, `log --all`, `rev-list`, `cat-file`, `fsck`, `stash list`, `diff HEAD`. **No branch was changed, created or deleted; nothing was checked out, fetched or committed.**
- Read-only source inspection of both frozen tracks, `tests/`, `configs/`, `docs/`, `scripts/`.
- Read-only NPZ/CSV/JSON reads of the two accepted datasets, twelve accepted episode files, twelve accepted `summary.json` receipts, and three accepted result tables.
- Executed the real `PublicObservationFilter` and `BeliefState.public_features` as a pure-function sensitivity probe on already-public observation values, under `PYTHONDONTWRITEBYTECODE=1`, using the `.review/venv` interpreter I0 registered for read-only NPZ inspection.
- Created exactly one new file: `CLAUDE_I0_READONLY_AUDIT.md` in the I0 directory.

**Verified side-effect free.** Both frozen-track hashes were recomputed after all probes and are unchanged (`951365d7…fb01`, `614524d7…e35b`). `git status --porcelain` shows only the untracked I0 directory; `git diff HEAD` is empty. The three I0 checksums still verify.

**Not performed, and still unauthorized:** editing any I0 file, source file, plan, registration or configuration; modifying accepted artifacts or frozen tracks; implementing adapters or fixes; creating datasets; running scientific evaluations; submitting Slurm jobs; running I1 or any later stage; rebaselining any result; network access.

---

## Verdict

**REVISE — I0 VERIFIED BUT PLAN REQUIRES MATERIAL CHANGES**
