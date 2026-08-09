# Draft plan — method-level state-information pilot under Ricker misspecification

**Status:** REVISION 3.1 — DRAFT FOR CLAUDE CONFIRMATION AUDIT  
**Execution status:** NOT AUTHORIZED; NOT IMPLEMENTED; NO JOBS SUBMITTED  
**Regression gate status:** NOT RUN  
**Proposed server repository:** `/fs04/scratch2/ce25/DeepRL_Population_Models`  
**Accepted outputs:** immutable; never modify, overwrite, re-score, or append to them

This document is an implementation plan for a later Codex server task. It is not an
authorization to edit code, generate datasets, submit Slurm jobs, or interpret new
results. I0 is complete and independently verified. The user and Claude must audit and
explicitly approve a later frozen version before I1, I2, regression/rebaseline work,
Stage B, Stage C, or any scientific execution. Claude's approval of I0 does not authorize
a later stage.

---

## 1. Scientific purpose

The accepted A1–A4 experiment diagnoses state information and fitted-model effects under
one special planner convention. It does not test whether the six accepted benchmark
methods differ in their ability to operate when ecological abundance is hidden.

This pilot asks an end-to-end method question across two deliberately complementary
Allee cells:

> For the accepted ecological and general-RL method implementations, how much performance
> is lost when exact abundance information is replaced by noisy surveys in out-of-family
> Allee environments, and is that loss consistent with each method's treatment of hidden
> abundance?

The experiment deliberately combines two challenges:

1. **structural transition-model misspecification:** the true environment is Allee while
   adapted PLUS and MOOR remain restricted to Ricker models; and
2. **state information:** exact abundance versus noisy survey information.

The primary estimand is **end-to-end method robustness to hidden abundance**. It is
therefore a **method-bundle stress test**, not a pure causal estimate of state
representation. The methods also differ in transition models, planners, objectives,
support/risk treatment, compute, and policy class.

Where a method demonstrably supports frozen fitted artifacts with only its online state
update changed, report a narrower deployment-only diagnostic as secondary. Do not assume
that this narrower estimand is available uniformly across methods.

The motivating architectural hypothesis is complementary rather than ordinal:

- ecological methods explicitly filter latent abundance but impose restrictive
  mechanistic transition structure;
- the general methods represent model/value/support uncertainty over an observation-space
  state summary whose dispersion is sigma-scaled but which performs no denoising and
  maintains no latent-abundance belief;
- a future method may need both a calibrated latent-state belief and flexible model
  uncertainty.

---

## 2. Record verification before plan revision

The following points were checked against the local audit record.

| Proposed correction | Local verdict | Controlling evidence |
|---|---|---|
| MOBILE is not implemented | **VERIFIED** | `00_START_HERE.md` identifies MOBILE as context only; the implemented general methods are RefPlan, OGSRL, BA-MCTS and EVD. |
| Replace MOBILE with OGSRL | **VERIFIED as method membership**, but OGSRL is not a model-belief substitute | `GENERAL_RL_CURRENT_IMPLEMENTATION_AND_SETTINGS.md`; OGSRL is a guarded constrained actor using public OOD/low-abundance proxies and is fixed after offline training. |
| Add EVD | **VERIFIED** | `READONLY_EXTRACTION_PASS_2.md` P0-4: measured full-task cost about 2.50 s/cell. EVD uses bootstrap fitted-Q disagreement, not a calibrated latent-state or Bayesian model belief. |
| PLUS registration is `plus_adapted_ricker_only_pbvi` with eight Ricker candidates and uniform prior | **VERIFIED** | `CHAT_HANDOFF_THREE_SPECIES_MATCHED_P10_ECOLOGICAL_BASELINES.md`; `PLUS_MOOR_PAPER_ALIGNED_IMPLEMENTATION_EXPLANATION.md`. |
| MOOR registration is `moor_adapted_ricker_misspec_pbvi` with one fitted Ricker model | **VERIFIED** | Same controlling files. |
| Only PLUS, MOOR and RefPlan reference the observation-noise model in their policy files | **PARTLY CONTRADICTED** | Correct for direct references only: `grep observation_noise_sigma` returns zero hits in `ogsrl.py`, `bamcts.py` and `ensemble_value_disagreement.py`. But in hidden mode `pipeline.py:246-259` routes every filter mode to `PublicObservationFilter`, whose particle spread is set by `context.observation_noise_sigma` (`beliefs.py:597-599`). All four general methods consume the resulting sigma-dependent features both offline (cached `beliefs.features`) and online. The filter is memoryless, never reweights (`beliefs.py:625`, ESS exactly 1.00) and achieves effectively no denoising (accepted `filter_log_rmse_mean` 0.196-0.202 against sigma 0.200), so this is a sigma-scaled observation-space feature, **not** a calibrated latent-abundance belief. |
| Tiger × Allee × sigma 0.2 has accepted OGSRL return 4.716 and positive headroom | **VERIFIED with exact values** | Accepted OGSRL return `4.71629840155769`; best fixed action-10 return `4.224420899026084`; headroom `0.4918775025316062`. The fixed action is not “doing nothing.” |
| Fox × Allee × sigma 0.2 is decision-discriminating for ecological methods | **VERIFIED from accepted/replay records** | MOOR return `11.610936360689793` versus best fixed action 1 at `10.153605271956147`, headroom `1.4573310887336461`; PLUS headroom `0.8131401783020511`; PLUS differs from its MAP-only action on `0.709` of replay decisions. |
| Five of six methods optimize the shared learned surrogate; EVD uses raw logged rewards | **VERIFIED** | `READONLY_EXTRACTION_PASS_2.md` P0-2. Surrogate fit R² is about 0.30–0.73 and it cannot represent the hard safety cliff. |
| Measured per-cell costs are roughly PLUS 15,305 s, MOOR 1,938 s, BA-MCTS 847 s, OGSRL 430 s, RefPlan 52 s, EVD 2.5 s | **VERIFIED** | `READONLY_EXTRACTION_PASS_2.md` P0-4. |
| The old cross-family PLUS route contains the five-hand-picked-policy baseline bug | **NOT VERIFIED; record indicates a conflation** | The five-policy bug belongs to the separate S2 full-state VPI baseline (`in new repo/S2_ORACLE_REVIEW.md`). The frozen cross-family PLUS design has 16 candidates. It still must not be used here because it includes the true family and would defeat the intended Ricker-misspecification stress test. |
| The live/current server code differs from the accepted code in a way that changes this cell | **OPEN pending an authorized regression** | Earlier handoffs referred to a branch-isolated OGSRL short-episode fix. The I0 audit found no such branch, commit, ref, patch or test anywhere locally; the reference appears to conflate the accepted Phase 2E OGSRL pathwise-cost package with an unimplemented short-episode fix. HEAD is `77cd38adb11970d56ce4c96bf84de15fac3108ac` on `e1-phase1-parity`; before I0 artifacts the tracked worktree was clean. |
| A zero/epsilon-scale likelihood is unsafe for Arm T | **VERIFIED, with scope qualification** | `AUDIT_IMPLEMENTATION_BRIEF.md` documents the ecological all-zero-likelihood fallback for off-grid positive observations when `observation_scale <= 1e-12`, plus related narrow-likelihood/underflow traps. Arm T must use direct state/belief assignment; neither sigma nor observation scale is driven toward zero. |
| Accepted-table receipt path | **CORRECTED** | The real file is `results/accepted/MATCHED_P10_144_RECEIPT.json`, SHA-256 `a198c70f34ed0191a6580c49d52ab675ea7b6768f87076994775b223624d3163`; the I0 manifest/report incorrectly named `results/accepted/MATCHED_P10_144_METHOD_CELLS_RECEIPT.json`. |
| Paper-faithful dependency path | **CORRECTED** | The real file is `requirements.txt`, SHA-256 `e3da565735841215d6d987a67278e26ac7b3aba2f5afee0876879ec512f6a2a2`; the I0 manifest/report incorrectly named `requirements-paper-faithful.txt`. |
| Frozen-track hash stability | **QUALIFIED** | Historical full-tree hashes are retained, but they cover 204 `__pycache__/*.pyc` files and reproduce only with `LC_ALL=C`. Future integrity uses source-only hashes plus Git checks and disables bytecode writes. |

---

## 3. Correct method roles

Use the accepted names and fidelity labels. Do not relabel the methods as exact
implementations of their source papers.

| Method | Role in this pilot | State/observation treatment | Model/value uncertainty | Reward used in planning/training |
|---|---|---|---|---|
| adapted PLUS | joint ecological belief baseline | candidate-specific latent-abundance beliefs; explicitly uses the survey likelihood and sigma | posterior over eight fitted Ricker candidates | shared learned linear surrogate |
| adapted MOOR | ecological state-belief baseline | one latent-abundance belief; explicitly uses the survey likelihood and sigma | one fitted Ricker point model | shared learned linear surrogate |
| RefPlan-inspired | partial/conflated general-method control | public particles/history-conditioned prior; explicitly uses `LogNormalObservationModel` and sigma | five-member observation-space posterior/ensemble | shared learned linear surrogate |
| OGSRL-inspired | guarded fixed-policy general method | sigma-dependent public observation-space features from the shared filter; no direct sigma reference in `ogsrl.py`; no latent-abundance belief. Sigma scales both the offline actor's rollout-start dispersion and the deployment particle cloud. | OOD/support and predictive ensemble machinery; fixed actor after training | shared learned linear surrogate |
| BA-MCTS-inspired | online model-belief general method | sigma-dependent public observation particles as search roots; not a calibrated latent-abundance belief; no direct sigma reference. Its model-posterior likelihood width uses a hard-coded 0.05, not sigma. | categorical model belief updated inside tree search | shared learned linear surrogate |
| EVD pessimism | low-cost value-disagreement control | sigma-dependent public observation features used for offline Q/behavior fitting and action selection; no latent-state filter; no direct sigma reference | variance across 20 bootstrap conservative Q estimators; no online update | raw logged rewards |

RefPlan is a decisive control. A comparison containing only OGSRL, BA-MCTS and EVD
would make the observation-model asymmetry partly true by construction.

All six methods' inputs depend on `observation_noise_sigma`. The discriminating property
is **not** sigma consumption but the kind of state estimate maintained. PLUS and MOOR run
a recursive Bayes filter over latent abundance (`faithful_pomdp.py:186-207`). The four
general methods receive a memoryless, uniformly weighted observation-space cloud:
`PublicObservationFilter` discards the prior on every step, deletes the action, uses
uniform weights, has accepted ESS exactly 1.00, and has log RMSE 0.196-0.202, approximately
the registered observation noise 0.200. It therefore provides effectively no denoising.
RefPlan models an observation likelihood over **model identity**, not over abundance.

Revision 3 consequently frames the pilot as a comparison of **different state
representations and their end-to-end robustness**. It keeps separate: receipt of noisy
observations; receipt of an upstream sigma-dependent `BeliefState`; preservation of
temporal/action history; actual denoising; and maintenance of a calibrated
likelihood-based latent-abundance belief.

---

## 4. Scope

### 4.1 Stage A — mandatory regression gate

One current-code rerun only:

- species: Amur tiger;
- true environment family: Allee;
- observation noise: sigma 0.2;
- reward/evaluator: accepted safe P=10 convention;
- method: OGSRL-inspired;
- information arm: accepted/noisy-survey convention;
- exact accepted dataset, action table, horizon, discount, evaluation identities, and
  configuration wherever available.

Target accepted mean for comparison:

```text
4.71629840155769
```

This gate must finish and be audited before any scientific pilot task is submitted.

### 4.2 Stage B — two-cell sigma 0.2 pilot

Run all six methods under both information arms in:

- Amur tiger × Allee × sigma 0.2; and
- crab-eating fox × Allee × sigma 0.2.

This is 24 method-arm tasks. The cells have complementary roles fixed before results:

- tiger is the general-method activity/stress cell and the OGSRL regression anchor;
- fox is the ecological-method decision-discriminating cell.

The accepted Arm O record must be used before returns open to label which method/cell
pairs were previously decision-active. This label is descriptive provenance, not a
substitute for the new Arm T/Arm O activity gate.

Stage B at sigma 0.2 is **screening only**. It cannot support a claim that requires
consistency across noise levels.

### 4.3 Stage C — sigma 0.1 confirmation

Run the same two cells, six methods and two information arms at sigma 0.1 only after:

- Stage B passes all validity gates;
- the results are not floor/ceiling saturated;
- at least one compared method is decision-active;
- Stage C is separately authorized.

If Stage B is non-discriminating for both PLUS and MOOR in both cells, do not run Stage C.

No claim requiring consistency at both noise levels may be made from Stage B alone.

### 4.4 Explicit exclusions

- no Ricker, theta-logistic, regime-switching, or vulture expansion;
- no fox cell other than fox × Allee at the registered pilot noise level(s);
- no new collection seeds, fit seeds, or evaluation identities;
- no reward redesign;
- no old cross-family PLUS route;
- no A1–A4 rerun;
- no E2 or joint-controller implementation;
- no modification of any accepted output or frozen track;
- no slide changes as part of execution.

---

## 5. Information arms

Both arms must be generated under the same audited code version. Do not reuse accepted
noisy-survey returns as one arm.

### Arm T — exact abundance information, end-to-end

The method is trained/planned and evaluated under a registered method-specific exact-
abundance information contract. It must not receive hidden Allee parameters, future
process draws, the private safety threshold, evaluator internals, or the hidden family
label. Retraining/refitting is permitted only where required by the method's input
contract and must be declared in the component-change table.

**Arm T is a direct state/belief intervention, never a zero-noise likelihood.** Do not
implement exact abundance by setting `observation_noise_sigma`, `observation_scale`, or
another likelihood-scale parameter to zero or an epsilon. The audited ecological path can
silently produce an all-zero likelihood for off-grid positive observations and then retain
the predicted or uniform belief, which would imitate a false “robustness” result.

For adapted PLUS and MOOR, use the audited direct-assignment convention through a
purpose-built adapter: convert current raw truth to latent units with `survey_scale` and
set the internal abundance belief to a delta on the registered nearest latent bin
immediately before every decision. Keep the original observation model and sigma
unchanged; do not call the base Bayesian update for runtime truth delivery. Do not reuse
`OracleStateFilter` wholesale.

For RefPlan, OGSRL, BA-MCTS and EVD, use method-specific, context-preserving exact-state
adapters. They must supply current true abundance while retaining the temporal
observation/action context required by the accepted method. In particular,
`OracleStateFilter.set_true_state` is prohibited as-is: it zeroes the three context fields,
including previous observation, current observation and timestep, and requires evaluator
configuration that hidden methods must not receive. The adapters must record every input
feature changed and preserved. They must never expose the family label, private safety
threshold, evaluator-only reward constants, future state, future process randomness,
hidden `r`, or hidden `K`.

#### Arm T exact-state adapter requirements from the I0 audit

1. **Dataset decision and exact-state source.** The registered pilot does **not**
   regenerate trajectories. It reuses the immutable accepted `public.npz` row identities,
   actions, rewards, splits, episode boundaries and 160 x 25 structure. The accepted
   `public.npz` contains no true-abundance field: its 13 keys are public-only. For the
   four general methods, the offline Arm T dataset **view changes** while row identities,
   row ordering, splits and episode structure do not. A dedicated future extraction stage
   constructs that view from the cell-specific accepted `truth.npz`, subject to the
   strict allowlist and isolated override below, while preserving already-public temporal
   observation/action context. The accepted NPZ files are never modified.

   The two registered source identities are:

   | Accepted pilot cell | Exact `truth.npz` path and SHA-256 | Rows and public binding | Verified observation/state check |
   |---|---|---|---|
   | Amur tiger × Allee × sigma 0.2 | `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/quarantine/private/regime_hidden/reward_safe/amur_tiger/allee/sigma_0p2/truth.npz`; `1658f587cc2b1144362bd90182efce1bd33b65d35c47c1e9dae4320d705b8668` | 4,000 rows, row-aligned with `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/quarantine/datasets/regime_hidden/reward_safe/amur_tiger/allee/sigma_0p2/public.npz` (file SHA-256 `9e9c3a6d4bcee3c27c785f46ce17b66e1c9099c446213f46ebcfa3c3851f771c`); embedded `public_dataset_sha256` = `7e71172af4bc95b2c31291414af53371c1ea94393e01675d269b2b7c360324c9`, matching the accepted Amur-tiger dataset identity | `sd[log(observation/state)] = 0.19942`, consistent with registered sigma 0.2 |
   | Crab-eating fox × Allee × sigma 0.2 | `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/quarantine/private/regime_hidden/reward_safe/crab_eating_fox/allee/sigma_0p2/truth.npz`; `4149e293d70a59b000a3b947ce663fb3de282fb404cea857f02be00718a467e5` | 4,000 rows, row-aligned with `/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/quarantine/datasets/regime_hidden/reward_safe/crab_eating_fox/allee/sigma_0p2/public.npz` (file SHA-256 `7d1b2fc8a1949dda6f53f39eff1dbfe220719d0cbad9f8956ab7ca2d08e87e19`); embedded `public_dataset_sha256` = `688d580f1e47a61dcad9b8a6cb96f1d62835bfbd0b835554cf31c8ab8bdd0c13`, matching the accepted crab-eating-fox dataset identity | `sd[log(observation/state)] = 0.19655`, consistent with registered sigma 0.2 |

   Claude verified both row alignments, bindings and noise checks. I1 must reverify and
   freeze every absolute source/public path, file hash, embedded binding, cell identity,
   row count, row order and unit convention before construction. Any mismatch aborts.

   **Strict truth-field allowlist.** Methods and training pipelines must never open or
   read the original `truth.npz` directly. The dedicated external extraction stage is
   the sole permitted reader. The only scientific source fields it may dereference for
   an Arm T offline view are:

   - `states`, the current true abundance; and
   - `next_states` only for methods whose primary end-to-end Arm T requires offline
     model refitting and only as the supervised one-step next-state target.

   `next_states` may be paired only with the same row's current `states` value and
   already-public action. It must never reach an online adapter, deployment feature,
   action-selection input, planning root or a frozen-fit Arm T diagnostic that does not
   require refitting. No later row, future realized trajectory information or future
   state may be exposed to a decision.

   Every other truth field is forbidden, including hidden family or `kind`; safety or
   MVP thresholds; true/evaluator reward; reward components; `r_base`, `r_eff_true`,
   `C`, `K`, `theta` or other hidden Ricker/Allee parameters; `regime`,
   `next_regime` or other latent regime identity; `entry`, `initially_unsafe`,
   `safety_penalty_applied`; future environmental randomness; private evaluator
   constants; any part of `metadata_json.environment`; and every field not explicitly
   allowlisted. The extractor may inspect only the `public_dataset_sha256` binding
   value in verification metadata; no metadata content may enter the derived artifact or
   method-visible process.

   **Fail-closed schema and alignment check.** I1 must register the exact complete source
   archive key set because Claude's audit did not enumerate it; no extractor may be
   implemented before that registration. The future extractor must check the archive
   against that exact registered source schema, abort on any unexpected or missing key,
   and permit scientific dereference only of `states` and the conditionally allowed
   `next_states`. It must also require exactly 4,000 finite rows in exact public row
   order, no NaN or nonfinite values, verified current/next-state units, and an exact
   `public_dataset_sha256` match. The derived artifact must have a separately registered
   exact schema containing only the allowed state field(s) plus already-public action,
   context and identity fields. Any missing, extra, reordered, nonfinite, unit-mismatched
   or binding-mismatched value aborts construction.

   **Registered private-truth invariant override.** The invariant near `pipeline.py:432`
   states that private truth is evaluator-only and must never be cached for training.
   Arm T deliberately overrides that invariant, narrowly and only for this preregistered
   diagnostic's two registered pilot cells and exact-state arms. The override:

   - is performed only by a dedicated external extraction/adapter stage outside
     `src/tracks/**`;
   - does not modify accepted `public.npz`, `truth.npz`, datasets, frozen tracks or
     caches;
   - writes only a new derived allowlisted artifact in a future, separately authorized
     namespace and records source and derived hashes plus row-level alignment;
   - prevents all method code and training pipelines from accessing original
     `truth.npz`;
   - makes the derived truth artifact inaccessible to noisy Arm O;
   - excludes every evaluator-private field from the derived artifact; and
   - destroys or retains the derived artifact only according to a provenance policy
     registered in I1 before construction.

   Outside this isolated diagnostic path, the original no-private-cache invariant remains
   absolute. I1 must register the override and retention/destruction policy. I2 must prove
   the information boundary with leakage tests before any extraction or scientific use.

2. **Context preservation.** At reset and every step, the adapter supplies current true
   abundance and retains the method's original previous/current observation, timestep and
   action history. I2 must fail closed if any required context becomes zero-filled,
   reordered or replaced accidentally.
3. **OGSRL safety calibration.** `_fit_public_safety_scale` derives `s_low`, the behavior
   budget and `deployment_safety_limit` from dataset observations, not beliefs. The
   primary end-to-end Arm T must recompute these quantities on the derived exact-abundance
   view. This is necessary for a correctly scaled true-state bundle, but it also moves the
   safety-calibration axis and must be reported. Holding the accepted noisy-scale budget
   is not the primary Arm T and is not a valid frozen-fit diagnostic for OGSRL.
4. **Transition-model-quality confound.** Rebuilding `PublicDynamicsEnsemble` on exact
   states can shrink each member's `residual_sigma`. That sharpens RefPlan's and BA-MCTS's
   model posteriors. Record every member's `residual_sigma` and model-posterior sharpness
   by arm. This changes transition-model quality as well as state information.
5. **PLUS/MOOR frozen proof.** Direct grid assignment leaves fitted artifacts untouched.
   `MechanisticModel.parameter_hash()` includes `observation_scale`, so zeroing or
   shrinking sigma would independently break byte identity as well as risk the all-zero
   likelihood fallback. I2 must prove the existing 39-file PLUS and nine-file MOOR trees
   byte-identical by hash.

#### Registered Arm T component changes

“Preserve” means the accepted value/history is retained exactly. “Rebuild” means a new
artifact is created only in the future registered namespace; it never overwrites an
accepted artifact.

| Method | Current-state representation | Observation/action history | Offline dataset view | Dynamics fitting | Residual-noise estimation | Model-posterior sharpness | Safety-budget calibration | Reward-surrogate fitting | Policy training | Online updating |
|---|---|---|---|---|---|---|---|---|---|---|
| **adapted PLUS** | **Change:** each of eight latent-abundance beliefs is a point mass at `raw_true / survey_scale` on the nearest grid bin | Preserve registered action history/timing; there is no general-method public-history feature; bypass the noisy likelihood only for truth delivery | Preserve accepted dataset | Preserve eight Ricker fits | Preserve | Candidate posterior algorithm is preserved, but exact state may change realized sharpness; report it | n/a | Preserve accepted surrogate | Preserve POMDP/PBVI artifacts | **Change:** direct assignment before every action; no double Bayesian update |
| **adapted MOOR** | **Change:** one latent-abundance point mass | Preserve registered action history/timing; there is no general-method public-history feature; bypass the noisy likelihood only for truth delivery | Preserve accepted dataset | Preserve one Ricker fit | Preserve | n/a (one model) | n/a | Preserve accepted surrogate | Preserve POMDP/PBVI artifacts | **Change:** direct assignment before every action; no double update |
| **RefPlan** | **Change:** exact current-abundance particles | **Preserve** previous/current noisy-observation context, timestep and action history | **Change:** derived exact-state view over the same rows/splits/boundaries | **Rebuild** public ensemble for primary Arm T | **Re-estimate and report** per member | **Expected to change/sharpen; measure** | n/a | Preserve accepted surrogate | **Rebuild** behavior prior | **Change:** exact state plus preserved context; register model-posterior likelihood semantics |
| **OGSRL** | **Change:** exact current-abundance particles/features | **Preserve** original temporal/action context | **Change:** derived exact-state view over identical rows/splits/boundaries | **Rebuild** public ensemble | **Re-estimate and report** | n/a (fixed actor) | **Recompute on exact abundance and report safety-axis shift** | Preserve accepted surrogate | **Retrain** actor, guardian, low-abundance scale and budget | **Change:** exact state with preserved context; actor remains fixed after training |
| **BA-MCTS** | **Change:** exact search-root particles/features | **Preserve** previous/current observation, timestep and actions | **Change:** derived exact-state view over identical rows/splits/boundaries | **Rebuild** public ensemble | **Re-estimate and report** | **Expected to change/sharpen; measure** | n/a | Preserve accepted surrogate | n/a; search remains online | **Change:** exact root state; preserve context and register posterior update |
| **EVD** | **Change:** exact current-state features | **Preserve** temporal/action context | **Change:** derived exact-state view over identical rows/splits/boundaries | n/a | n/a | n/a | n/a | n/a; raw logged rewards remain fixed | **Retrain** behavior reference and all 20 Q members | **Change:** exact state features; no new online estimator |

The primary Arm T versus Arm O contrast remains an **end-to-end method-bundle
comparison**. Rebuilt dynamics, residual estimates, safety calibration and policies are
part of that bundle. A frozen-fit/state-input-only diagnostic may be reported only where
it is genuinely implementable: required for PLUS/MOOR and optional, explicitly secondary,
for RefPlan after context-preservation tests. Neither contrast may be called a pure causal
state-uncertainty effect without additional controls.

### Arm O — noisy survey information

The method receives the accepted public survey information. PLUS, MOOR and RefPlan may
use their existing observation-likelihood machinery. OGSRL, BA-MCTS and EVD retain their
accepted method logic. Their inputs already depend on sigma through the shared
`PublicObservationFilter`; this pilot introduces no new sigma consumption for them.

### Mandatory design audit before coding

The six methods do not share one state interface. Before implementation, verify the
registered component table above against the concrete adapter design and record both the
changed and preserved feature names.

The primary estimand is an **end-to-end information-regime effect**. It must not be
described as a pure online-filtering effect. If fits can be frozen and only the online
estimator changed—plausibly for PLUS and MOOR—report that narrower diagnostic separately,
and prove artifact identity by hash in I2.

Never feed true states at evaluation time into an unchanged policy whose input contract
was trained only on noisy observations unless the adapter and its distributional validity
are explicitly justified and tested.

---

## 6. Common controls

Within each sigma cell and information pair, hold fixed:

- underlying trajectory identities and row ordering;
- training-data volume and split identities;
- action set and action effects;
- true Allee evaluator/environment distribution;
- reward evaluator, P=10 penalty, horizon 50 and gamma 0.95;
- 20 evaluation identities and their pairing order;
- method hyperparameters and compute budget, except the registered information adapter;
- software commit, dependency lock and CPU architecture;
- random seeds for collection, fitting, planning and evaluation;
- true-state reward used for final evaluation.

Any unavoidable difference must be registered before results are opened.

### 6.1 Integrity and bytecode safeguards

Every future reconnaissance command, test, adapter check, regression and scientific task
must run with:

```text
PYTHONDONTWRITEBYTECODE=1
LC_ALL=C
```

Before and after each stage:

- run `git diff --exit-code` for tracked content and report all untracked paths explicitly;
- treat or mount frozen accepted directories read-only where practical;
- compute source-only track hashes excluding `__pycache__`, `.pyc`, `.pyo`,
  `.pytest_cache` and other generated files;
- never treat an import-generated bytecode change as evidence that scientific source code
  changed; investigate it separately and verify source-only hashes;
- keep the historical full-track hashes for provenance, but never use them as the sole
  integrity check.

The historical full-tree recipe is:

```text
LC_ALL=C find TRACK -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
```

It yields ecological `951365d7874a417d7e66b14538dc275a9f325ac4643df8ea1af4d1d24877fb01`
over 149 files and general `614524d7b058418a3ff3f370e7e5b57582213c2c518766d8d88f1ec093e7a35e`
over 162 files, but includes 204 bytecode files across the two tracks. The I0 source-only
recipe excluded `*/__pycache__/*`, `*/.pytest_cache/*`, `*.pyc` and `*.pyo`; it produced:

- ecological: 52 files, `2b3b8ae6d2f8ff5ffb17c4885ded9e8f1f6b3c0cb662f393186fe4b4706a884e`;
- general: 55 files, `f90cea6f28dcacb910b5e036bf9e09958715d00a2418fd0856a3d5a12856bdbd`.

I1 must freeze the exact source-only recipe, file list and hashes in its registration.

---

## 7. Regression and validity gates

### Gate G0 — provenance

Before running anything, reverify the I0 finding that each accepted pilot dataset contains
4,000 rows arranged as exactly 160 complete 25-transition episodes. There are no short
episodes, terminated episodes, short truncations or invalid/incomplete records; all 160
episodes per cell end with the registered full-horizon truncation. Then record:

- live server path, Git HEAD, branch and dirty status;
- hashes of both frozen tracks;
- exact dataset, registration, action-table and evaluator hashes;
- Python/dependency environment;
- Slurm partition, CPU model and requested resources;
- that the OGSRL short-episode fix is not present locally and must not be claimed;
- `results/accepted/MATCHED_P10_144_RECEIPT.json` and `requirements.txt` at their corrected
  paths and unchanged hashes;
- the required `PYTHONDONTWRITEBYTECODE=1` and `LC_ALL=C` environment, historical and
  source-only hashes, `git diff --exit-code`, and every untracked path.

Abort if the accepted artifacts or frozen tracks would be modified.

### Gate G1 — tests

Run the existing repository self-tests. The three OGSRL short-episode regression tests
named in earlier revisions **do not exist in this repository**, and no passing record for
them exists. The prior claim conflated the accepted Phase 2E pathwise-cost package
(`docs/fix_implement_general_RL/PHASE2E_OGSRL_FIX_AND_CANARY_PACKAGE_REPORT.md`, 31 targeted
and 203/203 full-suite tests passed, current `ogsrl.py` SHA-256
`da965bc320f3cbfaf381a46aaa77a7eb0ef7c3b6b7b493bee02e338544e6993b`) with an
unimplemented short-episode/absorbing-terminal fix. The passing package is byte-identical
to the current file and is a separate issue, not evidence of short-episode handling.

G1 must verify from the pinned inputs that:

- both accepted datasets remain exactly 160 x 25;
- `ogsrl_cost_horizon` remains exactly 25;
- the `len(cost) < horizon` guard is not reached (`25 < 25` is false), with a margin of
  exactly one transition because a 24-transition episode would reach it;
- no future pilot input contains an episode shorter than 25.

The registered pilot does not regenerate trajectories, so no short-episode implementation
is a prerequisite. If any future dataset contains an episode shorter than 25, stop before
fitting OGSRL until a separately reviewed fix exists with tests for bit-identical
full-length behavior, absorbing-terminal handling for legitimate one- and two-step
collapses, and rejection of short nonterminal truncations. Abort G1 on any existing-suite
failure or dataset/horizon mismatch.

### Gate G2 — accepted-cell regression anchor

Rerun noisy-survey OGSRL for tiger × Allee × sigma 0.2 and compare with the accepted
artifact at three levels:

1. per-episode returns;
2. action sequences and ecological event counts;
3. aggregate return mean and SD.

Use the pinned accepted architecture and require bit parity or the repository's existing
exact-parity criterion. If that architecture is unavailable, stop; do not invent a
tolerance or create a new baseline under this plan.

Any unexplained difference stops the study. Create a regression report and do not submit
Stages B or C.

The documented OGSRL absorbing-terminal fix is **not present** in this repository and was
never applied to the accepted code path. No fix-attributable delta is possible. Any
difference from the accepted artifact is unexplained and is an unconditional stop
condition. Regression and rebaseline execution both require separate user authorization.

**Current result:** NOT RUN. No regression conclusion is available.

### Gate G3 — paired-run integrity

For every method and sigma:

- Arm T and Arm O must contain the same 20 ordered evaluation identities;
- environment process draws must be paired where the method interface permits;
- reward reconstruction must equal stored return within the registered tolerance;
- no task may silently fall back to another method or information mode.

For each Stage B Arm O rerun, also compare per-episode returns, action sequences and event
counts with its accepted counterpart on the same architecture. Any mismatch invalidates
that method-cell and stops the study; no missing-fix exception or silent rebaseline exists.

### Gate G4 — preregistered decision-activity / non-inertness

The accepted record shows PLUS and MOOR used constant action 10 in every tiger cell. A
zero true-to-noisy loss from a constant policy is not evidence that its state belief
handled uncertainty.

Before new returns are opened, label activity expected from the accepted Arm O record.
After acceptance, for each new method-arm report:

- number of distinct deployed actions;
- fraction of decisions differing across paired histories/states;
- adaptive headroom over the best fixed action;
- action dependence on belief/observation summaries where recoverable.

A method may support a state-representation claim only if it is decision-active and has
meaningful adaptive headroom. Otherwise label its state-information comparison
**non-discriminating**.

The I0 general-method action-entropy figures come from column 25, `action_entropy`, in the
accepted full general-track episode files at
`/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/quarantine/evaluation/regime_hidden/<species>/allee/sigma_0p2/data_real/backend_numpy/regime_hidden/reward_safe/<method>/learned/episodes.csv`.
The source paths follow the template above; their exact hashes are:

| Cell | Method | Episode-file SHA-256 |
|---|---|---|
| tiger | RefPlan | `8d546adb928b13a9824f07f5b362bafa5133dc772ba34c823bd3946b2a8870a8` |
| tiger | OGSRL | `c1db405698660f9cad3bb352aa575d2a2f4499141c5a6a1beae09a92cd9c6d02` |
| tiger | BA-MCTS | `20608a3c90eb1d438a372284d5ebe6965eaed40108f9915e0b27159b7e6cde75` |
| tiger | EVD | `8d35854f0d02fcc112a992caeae6a1693514c75893e4759fd0fb85de5a4add07` |
| fox | RefPlan | `7e8c6d5812bc6b238092728b54734b6cd92e9ec58011ae3fa934380d32f24536` |
| fox | OGSRL | `bb732bac5946e162c2179b3b13a461d5a7c6dbd5d80fcb78c829c639ab98c540` |
| fox | BA-MCTS | `1c75ec2e9b6cc299ec020dbe2fa17a8b0ab201c7e98b04ef07eef099a77e5997` |
| fox | EVD | `fb581f8d89f72be2f2ffed6765adca01202d88af70faed9b678ccfda3349eb10` |

Claude's prior audit inspected these same accepted files and hashes but displayed only
the first 12 of their 61 columns, then incorrectly reported the action column absent.
`action_entropy` is present at column 25, all eight I0 activity-entropy values reproduce
exactly, and the I0 activity table and G4 activity labels are verified. This provenance
correction does not change the decision-activity gate. I1 must record the full paths and
hashes, not only the accepted aggregate table whose general-method
`action_entropy_mean` cells are `NaN`.

### Gate G5 — ecological viability

Positive accepted OGSRL headroom establishes that the accepted noisy sigma 0.2 cell is
not universally floor-saturated. It does not guarantee that both new information arms or
all methods are informative. Report collapse, unsafe occupancy and action diversity before
interpreting return gaps.

---

## 8. Outcomes and estimands

### 8.1 Primary within-method estimand

Because all methods in one cell use the same true-reward evaluator and return units, use
the paired raw loss as primary:

```text
state_information_loss_m = mean_return_true_m - mean_return_noisy_m
```

Positive values mean that the method performs better with exact abundance information.
Report paired episode-level intervals using the registered pairing keys.

Raw losses are directly comparable across methods **within one species/cell** because the
evaluator and units are shared. They are not quantitatively comparable across fox and
tiger. Cross-cell synthesis is restricted to preregistered qualitative features: sign,
within-cell ordering, activity status and whether the same interpretation category holds.
Do not pool, average or rank raw losses across species.

### 8.2 Guarded normalized diagnostic

Claude proposed dividing by each method’s true-state return. That ratio can reverse sign or
explode when the true-state return is negative or near zero. Report a normalized diagnostic
only if its denominator and exclusion threshold are preregistered before results are opened,
for example:

```text
normalized_loss_m = (R_true_m - R_noisy_m) / max(abs(R_true_m), epsilon)
```

The raw paired loss remains primary. Never rank methods using an unstable ratio.

### 8.3 Required ecological/safety outcomes

For both arms and every method report:

- absolute discounted true-reward return;
- paired true-minus-noisy return loss;
- collapse-entry rate and first-collapse time;
- exact unsafe-occupancy step count and discounted safety-penalty contribution;
- minimum abundance and time below the safety threshold;
- action frequencies, switching and economic action cost;
- adaptive headroom over the best fixed action;
- planning/training time and failures.

Cross-method absolute return is secondary and must be labelled an
**architecture-bundle comparison**.

### 8.4 Reward-confound report

Five methods plan against the shared learned linear surrogate while EVD trains on raw
logged rewards. For tiger × Allee, the local audit reports surrogate R² about 0.561 at
sigma 0.1 and 0.553 at sigma 0.2, with systematic under-penalisation of unsafe rows.

Every output must state:

> State-information effects remain confounded with each method’s planning objective and
> reward approximation. EVD is additionally not objective-matched to the other five.

Where possible, report surrogate error along each arm’s visited trajectories. Do not
credit any method with explicit private-safety optimisation solely from evaluator return.

The direction of the surrogate's unsafe-row error is established: for recoverable species
it under-penalises unsafe states. This plausibly compresses the observed true-versus-noisy
loss if the noisy arm visits more unsafe states, but the direction of the **contrast bias**
is not guaranteed. Estimate it from arm-specific visited-state errors rather than assuming
it. EVD is objective-unmatched and must be reported in a separate block; it cannot break a
tie in the primary state-representation decision rule.

### 8.5 Planned but unrun objective-matched follow-up

Stage B cannot support a causal architectural claim. Register a future follow-up concept,
but do not implement it here: use the same pilot cells with a common planning/training
objective for all methods, or demonstrate that the within-cell loss ordering survives
restriction to arms with comparable visited-trajectory surrogate error.

---

## 9. Predeclared interpretation rules

These are tiered rules, not a single winner declaration.

### Strong support for an ecological state-belief advantage

Only if, at both sigma 0.1 and 0.2:

- PLUS and MOOR are decision-active rather than constant-policy controls;
- both have smaller paired raw state-information loss than RefPlan, OGSRL and BA-MCTS;
- the result is not explained solely by collapse saturation, reward-surrogate failure,
  or invalid true-state adapters.

EVD is reported separately because it alone trains on raw logged rewards and cannot
determine this rule.

This is intentionally stricter than a one-cell pilot. Stage B alone cannot satisfy it.

### Evidence that a model-identity likelihood plus public representation may be sufficient

If RefPlan is materially more robust than OGSRL, BA-MCTS and EVD, and is comparable to
active PLUS/MOOR, then its model-identity likelihood and history-conditioned public
representation may explain much of the gap. This is not evidence that all sigma-dependent
preprocessing is equivalent, and it weakens—but does not eliminate—the case for a new
joint architecture.

### Support for a broader belief-representation limitation

If active PLUS/MOOR remain robust while RefPlan degrades similarly to the other general
methods, then sigma-scaled preprocessing without denoising is insufficient. A calibrated
recursive latent-abundance belief becomes a stronger candidate mechanism.

### Transition/reward/planner limitation instead

If methods perform poorly with exact abundance, state information is not the binding
constraint for that method. Examine transition-model fit, reward surrogate, planner and
safety formulation before proposing a new belief representation.

### No state-representation conclusion

Declare the pilot non-discriminating if:

- ecological methods remain on any single constant action (tiger reference action 10;
  fox reference action 1);
- true-state policies have no adaptive headroom;
- most arms collapse immediately or remain at a floor/ceiling;
- regression/parity fails;
- the true-state adapters change uncontrolled components;
- an Arm T adapter enters a zero/epsilon-noise or degenerate all-zero observation-
  likelihood path instead of directly assigning the true-state representation;
- results depend on unstable normalization rather than raw paired loss.

---

## 10. Compute estimate

Using the measured mean full-task times from `READONLY_EXTRACTION_PASS_2.md` for paths
that reuse existing fits/artifacts:

```text
one arm, six methods, one cell       ≈ 18,588 s ≈ 5.16 core-hours
two arms, two cells, one sigma       ≈ 20.65 core-hours
two arms, two cells, both sigmas     ≈ 41.31 core-hours
```

Add the OGSRL regression anchor, existing-suite test time, analysis and failed-task
contingency. With eight frozen Ricker fits and only the registered runtime belief
intervention, PLUS dominates wall time at approximately 4.25 hours per method-cell.
Parallel execution can reduce elapsed time, but every task must remain separately
receipted and the experiment must be pinned to the approved Xeon Platinum 8452Y profile.

The **20.65 core-hour figure is an existing-fit execution estimate only**. It excludes
adapter design, implementation and audit, all unmeasured exact-state rebuild work for the
general methods, and any PLUS refitting or replanning caused by a latent-state dataset
view. The nearest receipt-based contingency for cold-fitting eight PLUS candidates is
about 2.41 hours per affected cell (2.15-2.53 hours), but this scaling does not measure
replanning or adapter work. I1/I2 scheduling must first confirm the component-change
result: if PLUS keeps its eight fits and changes runtime belief only, use the frozen-fit
estimate with margin; if any PLUS fit or planning artifact would change, stop, measure the
actual path, revise the compute plan, and obtain authorization before execution.

---

## 11. Implementation stages for the Codex server agent

The server agent must stop after each stage and return the listed audit artifact. It must
not continue automatically across a failed or unaudited gate.

### I0 — read-only reconnaissance

**COMPLETE AND INDEPENDENTLY VERIFIED.** I0 produced its report, manifest and hashes; Claude
verified their integrity and required Revision 3. I0 submitted no scientific job and did
not authorize any later stage.

### I1 — freeze registration and predictions

- Convert the audited plan into a machine-readable registration.
- Freeze scope, hypotheses, arms, seeds, metrics, gates, compute limits and output root.
- Seal expected qualitative outcomes before returns are opened.
- Create a new unique namespace; never write beneath accepted E1 or accepted 144-cell roots.

**Stop for explicit authorization.**

### I2 — adapter design and unit tests

- Implement arm selection outside frozen tracks, using wrappers/adapters only.
- Do not edit `src/tracks/**`.
- Add tests that detect leakage of family label, safety threshold, evaluator state beyond
  current abundance, or future randomness.
- Add method-specific tests for shape, units, filtering calls and identical Arm T/Arm O
  non-information configuration.
- Assert the general-method adapters preserve previous/current observation, timestep and
  required action history; fail if the oracle primitive's zero-filled contexts appear.
- Assert the primary general Arm T uses the identical accepted row/split/episode identities
  and does not regenerate trajectories.
- Assert the dedicated extractor is the only process able to read original `truth.npz`;
  method code and training pipelines cannot access it directly.
- Assert each Arm T derived artifact contains only the allowlisted current state and,
  where required for offline one-step supervision, next state, plus already-public
  action/context/identity fields; fail on any unexpected field.
- Assert Arm O cannot access any derived truth artifact.
- Assert no method receives hidden family/kind, safety threshold, true/evaluator reward,
  reward components, hidden parameters, latent regime, private constants, future
  environmental randomness or any other non-allowlisted truth.
- Assert runtime action selection receives only current abundance and permitted historical
  public context. Separately assert `next_states`, later rows and future realized
  trajectory information cannot enter deployment, action selection or a planning root.
- Assert the original no-private-cache invariant remains true everywhere outside the
  isolated, registered two-cell diagnostic path.
- For OGSRL, assert the exact-state safety budget is recalibrated from the derived exact
  view and receipted as a known safety-axis change.
- For RefPlan and BA-MCTS, record per-member `residual_sigma` and posterior-sharpness
  diagnostics for both arms.
- Assert Arm T never changes `observation_noise_sigma`, `observation_scale` or another
  likelihood scale to zero/epsilon.
- For PLUS/MOOR, assert raw truth is converted through `survey_scale`, the selected latent
  bin maps back to the registered discretisation of raw truth, and the belief immediately
  before every action is a point mass at that bin.
- For RefPlan, assert its exact-state representation is concentrated at current truth by
  the registered adapter while the observation-model parameters remain identical to Arm O.
- Add a negative test demonstrating that any attempted zero-scale/all-zero-likelihood Arm T
  construction fails closed rather than silently retaining a predicted or uniform belief.
- Produce the mandatory component-change table from section 5.

**Stop for independent read-only code audit.**

### I3 — regression gate

- Run G0–G2 only.
- Produce `REGRESSION_GATE_REPORT.md` with exact accepted/current deltas, episode-level
  parity, action/event parity, hardware and hashes.
- If any gate fails, set verdict `FAIL — STOP`; rebaseline is not authorized by this plan.

**Stop for user + Claude audit.**

### I4 — sigma 0.2 pilot

- Run both arms for all six methods and both preregistered cells under one Slurm
  submission plan.
- Keep returns sealed until structural completeness and receipt validation pass.
- Produce raw immutable artifacts and an acceptance report before interpretation.

**Stop for independent audit.**

### I5 — analysis

- Open results only after acceptance.
- Produce the paired raw-loss table, guarded normalized diagnostic, safety outcomes,
  activity gate, reward-confound analysis and compute report.
- Do not pool methods or noise levels and do not convert bundle comparisons into causal
  representation claims.

**Stop for scientific interpretation audit.**

### I6 — optional sigma 0.1 confirmation

- Requires a new explicit authorization after Stage B interpretation.
- Do not run if both ecological methods were non-discriminating in both Stage B cells.
- Otherwise repeat the frozen two-cell design at sigma 0.1 without tuning.
- Apply the two-noise-level decision rules.

---

## 12. Required artifact tree

The final registered output root should contain at least:

```text
registration/
  human_plan.md
  registration.json
  sealed_predictions.json
provenance/
  code_manifest.json
  data_manifest.json
  environment_manifest.json
  hashes.sha256
regression_gate/
  REGRESSION_GATE_REPORT.md
  episode_parity.csv
pilot_sigma_0p2/
  raw/<species>/<method>/<arm>/...
  receipts/<species>/<method>/<arm>.json
  structural_acceptance.json
analysis/
  paired_return_loss.csv
  safety_outcomes.csv
  method_activity.csv
  normalized_diagnostics.csv
  reward_confounds.csv
  compute.csv
  PILOT_RESULTS.md
audit/
  independent_code_review.md
  independent_results_review.md
```

If Stage C is authorized, add a separate `confirmation_sigma_0p1/` subtree. Never merge
raw Stage B and Stage C artifacts.

---

## 13. Authorization ledger

| Action | Current status |
|---|---|
| I0 read-only server reconnaissance | **Complete and independently verified** |
| Revision 3 plan and changelog | **Authorized and completed as planning documents only; unchanged by Revision 3.1** |
| Revision 3.1 plan and changelog | **Authorized and completed as documentation only** |
| Truth-data extraction or derived artifact creation | **Not authorized** |
| I1 registration and predictions | **Not authorized** |
| I2 adapter design and unit tests | **Not authorized** |
| Create/edit experiment code | **Not authorized** |
| Create a registration/output namespace | **Not authorized** |
| Regression gate or rebaseline | **Not authorized** |
| Stage B two-cell sigma 0.2 execution | **Not authorized** |
| Stage C sigma 0.1 execution | **Not authorized; separately conditional even after Stage B** |
| Scientific jobs submitted | **None** |
| Modify accepted outputs/frozen tracks | **Permanently forbidden** |
| Edit slides from pilot results | **Not authorized** |

The audited plan must be explicitly approved before the server agent performs any later
action. Claude's approval of I0 does not authorize I1, I2, regression, rebaseline, Stage B
or Stage C. Approval of Stage B does not automatically authorize Stage C.

---

## 14. Audit questions for the user and Claude

The prior audit resolved the estimand as end-to-end robustness, added the fox activity
cell, accepted raw paired loss as primary, and retained sigma 0.2 as screening only.
The remaining server-factual or authorization questions and closed audit findings are:

1. Can each Arm T contract be implemented without hidden-information leakage, and which
   narrower frozen-fit diagnostics are genuinely available?
2. **Closed by I0:** strict accepted replay uses Intel Xeon Platinum 8452Y under
   `xenon-8452Y`; visible nodes included `m3h100` and `m3h101` at I0.
3. **Closed by I0 + Claude:** no live branch contains the OGSRL short-episode fix; it does
   not exist locally, its three tests do not exist, and the passing record belongs to the
   separate byte-identical Phase 2E pathwise-cost package.
4. **Closed by I0 + Claude:** both accepted cells are exactly 160 x 25, with zero genuine
   short terminated, short-truncated or invalid episodes; minimum and maximum are 25.
5. Regression and rebaseline remain unauthorized. Any anchor mismatch is an unconditional
   stop under Revision 3.
6. Stage C remains unauthorized and requires separate approval under its
   non-discrimination stop rule.
7. **Closed as a Revision 3 design decision:** Arm T reuses the immutable accepted
   `public.npz` row identities, actions, rewards, splits and episode boundaries; it creates
   a derived exact-state view and does not regenerate trajectories.
8. **Closed as a Revision 3 primary-arm decision:** OGSRL recalibrates its safety budget on
   exact abundance. This moves the safety axis and must be reported; retaining the noisy
   budget would be mis-calibrated and is not the primary Arm T.

Revision 3.1 registers the truth source and information boundary but does not resolve
question 1 by implementation. Until that question is resolved in an independently
audited and explicitly authorized I2, the correct server action is
**planning/read-only audit only**.
