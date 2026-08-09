# Revision 3 changelog — method-level state-information pilot

Date: 2026-08-08 (Australia/Melbourne)  
Revision 2 source: `docs/true_noisy_state_real_methods/DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN.md`  
Revision 2 SHA-256: `238f238fca99b5dd043856ef90f5eb86b757e9a39249b25bf67eb235bc5cdf3f`  
Revision 3 target: `docs/true_noisy_state_real_methods/DRAFT_METHOD_LEVEL_STATE_INFORMATION_PILOT_PLAN_REV3.md`  
Revision 3 SHA-256: `164a536bb9a0afc4d78b42189ac119a089677bf6b83129d739ecb7b7ba103290`  
Controlling Claude audit: `docs/true_noisy_state_real_methods/i0_method_level_state_information_reconnaissance_20260808/CLAUDE_I0_READONLY_AUDIT.md`  
Claude audit SHA-256: `3e9ec9e68f705598b194c27487fb8ce1e92191e990fc001273a2b2db282b2ffc`

Revision 2 was copied to the Revision 3 filename before amendments were applied. Revision
2, the I0 directory, the Claude audit, source code, configurations, frozen tracks and
accepted outputs were not edited.

## Amendment status summary

| Amendment | Status | Revision 2 location | Revision 3 result |
|---|---|---|---|
| A1 | **APPLIED** | Section 2, line 67 | Section 2, line 70 |
| A2 | **APPLIED** | Section 3, lines 88-90 | Section 3, lines 89-113 |
| A3 | **APPLIED** | Section 1, lines 47-52 | Section 1, lines 47-55 |
| A4 | **APPLIED** | Section 5 Arm O, line 198 | Section 5 Arm O, lines 273-278 |
| A5 | **APPLIED** | Section 5, after line 192 | Section 5, lines 192-271; component table lines 257-264 |
| A6 | **APPLIED** | Section 7 G1, lines 259-267 | Section 7 G1, lines 375-399; anchor checks lines 403-408 |
| A7 | **APPLIED** | Section 7 G2, lines 286-291 | Section 7 G2, lines 401-422 |
| A8 | **APPLIED** | Section 2, line 73 | Section 2, line 76 |
| A9 | **APPLIED** | Section 14, lines 653-655 | Section 14, lines 795-819 |
| A10 | **APPLIED** | Section 2/provenance and I1 carry-forward | Sections 2, 6.1 and 7 G4: lines 78-80, 314-349 and 455-474 |

No amendment was partially applied or blocked.

## A1 — observation-noise record-verification row

> `| Only PLUS, MOOR and RefPlan reference the observation-noise model in their policy files | **PARTLY CONTRADICTED** | Correct for direct references only: grep observation_noise_sigma returns zero hits in ogsrl.py, bamcts.py and ensemble_value_disagreement.py. But in hidden mode pipeline.py:246-259 routes every filter mode to PublicObservationFilter, whose particle spread is set by context.observation_noise_sigma (beliefs.py:597-599). All four general methods consume the resulting sigma-dependent features both offline (cached beliefs.features) and online. The filter is memoryless, never reweights (beliefs.py:625, ESS ≡ 1.0) and achieves no denoising (accepted filter_log_rmse_mean 0.196–0.202 against σ = 0.200), so this is a sigma-scaled observation-space feature, **not** a calibrated latent-abundance belief. |`

- Revision 2 changed: Section 2 record-verification table, line 67.
- Revision 3 result: Section 2, line 70.
- Status: **APPLIED**.
- Application note: The full audited source references, ESS value and log-RMSE range are
  retained. ASCII `sigma` is used in prose where necessary for Markdown portability; the
  scientific meaning is unchanged.

## A2 — method-role characterizations

> **Line 88, OGSRL:** replace `public observation-space belief/proxies; does not use sigma explicitly` with `sigma-dependent public observation-space features from the shared filter; no direct sigma reference in ogsrl.py; no latent-abundance belief. Sigma scales both the offline actor's rollout start dispersion (ogsrl.py:420-425) and the deployment particle cloud (ogsrl.py:928, 725, 746).`
>
> **Line 89, BA-MCTS:** replace `bucketed public observation, not a calibrated latent-abundance belief; does not use sigma` with `sigma-dependent public observation particles as search roots (bamcts.py:252); not a calibrated latent-abundance belief; no direct sigma reference. Its model-posterior likelihood width is a hard-coded 0.05, not sigma (bamcts.py:305).`
>
> **Line 90, EVD:** replace `current public features; no latent-state filter and no sigma use` with `sigma-dependent public observation features (ensemble_value_disagreement.py:182) used for both offline Q/behavior fitting and action selection; no latent-state filter; no direct sigma reference.`
>
> All six methods' inputs depend on `observation_noise_sigma`. The discriminating property is **not** sigma consumption but the kind of state estimate maintained: PLUS and MOOR run a recursive Bayes filter over latent abundance (`faithful_pomdp.py:186-207`); the four general methods hold a memoryless, uniformly weighted observation-space cloud that achieves no denoising. RefPlan models an observation likelihood, but over **model identity**, not over abundance.

- Revision 2 changed: Section 3 table, lines 88-90, plus text below the table.
- Revision 3 result: Section 3, lines 89-113.
- Status: **APPLIED**.
- Application note: The added framing explicitly separates noisy observations, upstream
  `BeliefState`, history, denoising and calibrated latent-abundance inference.

## A3 — motivating hypothesis

> Replace the second bullet with: “the general methods represent model/value/support
> uncertainty over an observation-space state summary whose dispersion is sigma-scaled but
> which performs no denoising and maintains no latent-abundance belief.”

- Revision 2 changed: Section 1, lines 47-52.
- Revision 3 result: Section 1, lines 47-55, specifically lines 51-53.
- Status: **APPLIED**.
- Application note: Exact substantive replacement.

## A4 — Arm O sigma statement

> Replace with: “OGSRL, BA-MCTS and EVD retain their accepted method logic. Their inputs
> already depend on sigma through the shared `PublicObservationFilter`; this pilot
> introduces no new sigma consumption for them.”

- Revision 2 changed: Section 5 Arm O, line 198.
- Revision 3 result: Section 5 Arm O, lines 273-278.
- Status: **APPLIED**.
- Application note: Exact substantive replacement.

## A5 — mandatory Arm T adapter subsection

> 1. **Do not reuse `OracleStateFilter` for hidden methods.** `beliefs.py:770-772` sets `contexts = np.zeros((n, 3))`, which would zero the previous-observation, current-observation and timestep features that `public_features` exposes as elements 7-9 and that `bamcts.py:242` reads directly. It also requires `env_cfg`, a simulator configuration hidden methods must not hold. The Arm T adapter must preserve the public observation-history context unchanged and alter only the state particles.
>
> 2. **Register the Arm T dataset decision explicitly.** State whether Arm T reuses the accepted `public.npz` unchanged (belief-only) or regenerates trajectories. Belief-only preserves the 160 x 25 episode boundaries and keeps the OGSRL short-episode path unreachable; regeneration reactivates it, since `ogsrl_cost_horizon = 25` equals the episode length exactly and the guard at `ogsrl.py:563` has a margin of one transition.
>
> 3. **Register OGSRL's safety-scale treatment.** `_fit_public_safety_scale` (`ogsrl.py:551-590`) derives `s_low`, the behavior budget and `deployment_safety_limit` from `dataset` alone, never from beliefs. Under a belief-only Arm T these stay calibrated to the noisy observation scale (biased approximately +2%, since `E[y] = x·exp(sigma²/2)`) while the state input is exact; substituting true abundance into the dataset instead moves the safety axis. Neither is neutral: choose one, declare it, and report the consequence.
>
> 4. **Measure the transition-model-quality confound.** Rebuilding `PublicDynamicsEnsemble` on exact states shrinks each member's `residual_sigma`, which feeds RefPlan's model-posterior width (`refplan.py:155`) and BA-MCTS's (`bamcts.py:305`). Record `residual_sigma` per member per arm and report it alongside the primary estimand; an unreported shift confounds the information axis with the transition-model-quality axis.
>
> 5. **PLUS/MOOR frozen-artifact proof.** Direct grid assignment leaves the fitted artifacts untouched, but the prohibition on driving sigma or `observation_scale` toward zero is now doubly binding: `MechanisticModel.parameter_hash()` (`faithful_ecology.py:254-283`) hashes `observation_scale`, so any such change breaks byte-identity. Prove the 39-file PLUS and 9-file MOOR trees identical by hash in I2.

- Revision 2 changed: Section 5, after line 192, plus its component-change requirements.
- Revision 3 result: Section 5 lines 192-271; method table lines 257-264.
- Status: **APPLIED**.
- Application note: The attached revision request required an explicit resolution rather
  than leaving two design choices open. Revision 3 chooses no trajectory regeneration: it
  derives an exact-state view over the same immutable rows, actions, rewards, splits and
  160x25 episode boundaries. It chooses exact-abundance safety-budget recalibration for
  primary OGSRL Arm T and explicitly labels the safety-axis change. The table covers, for
  every method, current state, history, offline view, dynamics, residual noise, posterior
  sharpness, safety budget, reward surrogate, policy training and online updating.

## A6 — executable OGSRL regression gate

> Run the repository self-tests. The three OGSRL short-episode regression tests named in earlier revisions **do not exist in this repository** and no passing record for them exists (audited 2026-08-08 across all 11 refs, 7 commits, every `ogsrl.py` blob, and 42 test files). Prior references to a passing OGSRL fix appear to conflate the accepted **Phase 2E pathwise-cost package** (`docs/fix_implement_general_RL/PHASE2E_OGSRL_FIX_AND_CANARY_PACKAGE_REPORT.md`, 31 targeted + 203/203 full-suite passed, `ogsrl.py` = `da965bc3...6e993b`, byte-identical to the current tree) with a short-episode/absorbing-terminal fix that was never implemented here.
>
> G1 therefore requires only the existing suite to pass. **If and only if** Arm T regenerates datasets, add as a prerequisite: implement the short-episode handling on a named branch, add tests for (a) bit-identical behavior on full-length data, (b) correct absorbing-terminal padding for legitimate one- and two-step collapses, (c) rejection of short nonterminal truncations, and record a passing receipt. For the registered belief-only pilot this prerequisite does not apply: both accepted cells are exactly 160 x 25 with cost horizon 25, so `ogsrl.py:563` is unreachable and zero accepted rows are affected.

- Revision 2 changed: Section 7 G1, lines 259-267.
- Revision 3 result: Section 7 G1, lines 375-399, with noisy-anchor reproduction retained
  at G2 lines 403-408.
- Status: **APPLIED**.
- Application note: The user-required gate is stricter about local feasibility: it
  rechecks 160x25, horizon 25, guard non-entry, the exactly one-transition margin, and
  stops on any future episode shorter than 25. Revision 3's no-regeneration decision makes
  the unavailable fix irrelevant to the registered pilot without claiming it exists.

## A7 — remove fix-delta rebaseline branch

> “The documented OGSRL absorbing-terminal fix is not present in this repository and was
> never applied to the accepted code path. No fix-attributable delta is possible. Any
> difference from the accepted artifact is therefore unexplained and is an unconditional
> stop condition.”

- Revision 2 changed: Section 7 G2, lines 286-291.
- Revision 3 result: Section 7 G2, lines 401-422, especially 417-420.
- Status: **APPLIED**.
- Application note: The architecture-unavailable rebaseline alternative and G3's inherited
  fix-delta exception were also removed. Regression and rebaseline remain unauthorized.

## A8 — correct the live-fix provenance row

> Earlier handoffs referred to a branch-isolated OGSRL short-episode fix. The I0 audit found no such branch, commit, ref, patch or test anywhere locally; the reference appears to conflate the accepted Phase 2E OGSRL pathwise-cost package with an unimplemented short-episode fix. HEAD is `77cd38adb11970d56ce4c96bf84de15fac3108ac` on `e1-phase1-parity`, worktree clean.

- Revision 2 changed: Section 2, line 73.
- Revision 3 result: Section 2, line 76.
- Status: **APPLIED**.
- Application note: Exact substantive replacement, with the regression question left open
  because no authorized regression has run.

## A9 — close and add audit questions

> **Question 3** is **answered and closed:** no live branch contains the fix; it does not exist locally; its three tests do not exist; the "passing" record belongs to the different Phase 2E package.
>
> **Question 4** is **answered and closed:** zero genuine short terminated episodes in both cells; both are exactly 160 x 25; minimum = maximum = 25.
>
> **Add Question 7:** Does Arm T reuse the accepted `public.npz` unchanged, or regenerate trajectories? This determines whether the OGSRL short-episode fix is a prerequisite and whether episode boundaries are preserved.
>
> **Add Question 8:** For OGSRL Arm T, is the safety budget recomputed on exact abundance (moving the safety axis) or held at the accepted noisy-scale value (mis-calibrated against an exact-state input)?

- Revision 2 changed: Section 14, lines 653-655.
- Revision 3 result: Section 14, lines 795-819.
- Status: **APPLIED**.
- Application note: Questions 7 and 8 are included and answered as registered Revision 3
  decisions: no trajectory regeneration, and exact-abundance recalibration with the
  safety-axis shift reported.

## A10 — provenance corrections carried into I1

> `results/accepted/MATCHED_P10_144_METHOD_CELLS_RECEIPT.json` -> the real path is `results/accepted/MATCHED_P10_144_RECEIPT.json` (hash `a198c70f...3163` correct).
>
> `requirements-paper-faithful.txt` -> the real path is `requirements.txt` (hash `e3da5657...a2ea` correct).
>
> Amend the frozen-track recipe to `LC_ALL=C find TRACK -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum`, and record that the hash currently covers 204 `__pycache__/*.pyc` files; consider excluding bytecode so the frozen-track hash is stable under import.
>
> Record the source file for the general-method action-entropy figures in I0 Section 7 (not reproducible from the listed `episodes.csv` artifacts, which contain no action column).

- Revision 2 changed: Section 2/provenance requirements and I1 carry-forward.
- Revision 3 result: corrected paths at lines 78-79; bytecode/source-only safeguards and
  hashes at lines 314-349; action-entropy sources at lines 455-474.
- Status: **APPLIED**.
- Application note: The old and corrected paths and complete unchanged hashes are both
  recorded. Revision 3 requires `PYTHONDONTWRITEBYTECODE=1`, `LC_ALL=C`, source-only
  hashes, `git diff --exit-code`, untracked-path reporting and read-only treatment of
  accepted directories. It retains the historical hashes without relying on them alone.

  The action-entropy citation gap was resolved rather than guessed. Column 25
  (`action_entropy`) in the eight full accepted general-track files at
  `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/quarantine/evaluation/regime_hidden/<species>/allee/sigma_0p2/data_real/backend_numpy/regime_hidden/reward_safe/<method>/learned/episodes.csv`
  reproduces all I0 means and matches all eight I0 hashes. Claude inspected shorter,
  distinct source-repository summaries when identifying the gap. Revision 3 records the
  exact template and eight complete hashes.

## Additional required Revision 3 changes

- Compute language now distinguishes PLUS with eight frozen fits/runtime belief-only
  intervention from any unmeasured fit or planning rebuild. The 20.65 core-hour estimate
  is explicitly not an adapter-development or refit budget (Revision 3 lines 618-643).
- The fox ecological activity cell, tiger general-method activity cell, raw paired loss,
  no cross-species pooling, EVD's separate objective status, constant-policy gate,
  sigma-0.2 screening status, Stage C stop/separate authorization, immutable outputs and
  independent-audit gates are retained.
- Authorization now states I0 complete and independently verified; I1, I2,
  regression/rebaseline, Stage B and Stage C not authorized; no scientific jobs submitted
  (Revision 3 lines 772-791).

## Unresolved blockers and authorization boundary

No A1-A10 amendment is blocked. The remaining implementation question is whether each
context-preserving adapter can be implemented without hidden-information leakage and with
the registered feature preservation. That is future I2 work, not performed here.

I1, I2, regression, rebaseline, Stage B and Stage C remain unauthorized. No adapter,
source, test, configuration, registration, frozen track, accepted output or I0/audit file
was modified; no scientific job was submitted.
