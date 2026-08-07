# Draft plan — method-level state-information pilot under Ricker misspecification

**Status:** REVISION 2 — DRAFT FOR FINAL USER + CLAUDE AUDIT  
**Execution status:** NOT AUTHORIZED; NOT IMPLEMENTED; NO JOBS SUBMITTED  
**Regression gate status:** NOT RUN  
**Proposed server repository:** `/fs04/scratch2/ce25/DeepRL_Population_Models`  
**Accepted outputs:** immutable; never modify, overwrite, re-score, or append to them

This document is an implementation plan for a later Codex server task. It is not an
authorization to edit code, generate datasets, submit Slurm jobs, or interpret new
results. The user and Claude must audit and explicitly approve a revised, frozen version
before implementation.

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
- several general methods represent model/value/support uncertainty but do not use the
  registered observation-noise model;
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
| Only PLUS, MOOR and RefPlan explicitly consume the observation-noise model | **VERIFIED** | `READONLY_EXTRACTION_PASS_2.md` P0-6 and the ecological method documentation. OGSRL, BA-MCTS and EVD receive the constant in context but do not use it. |
| Tiger × Allee × sigma 0.2 has accepted OGSRL return 4.716 and positive headroom | **VERIFIED with exact values** | Accepted OGSRL return `4.71629840155769`; best fixed action-10 return `4.224420899026084`; headroom `0.4918775025316062`. The fixed action is not “doing nothing.” |
| Fox × Allee × sigma 0.2 is decision-discriminating for ecological methods | **VERIFIED from accepted/replay records** | MOOR return `11.610936360689793` versus best fixed action 1 at `10.153605271956147`, headroom `1.4573310887336461`; PLUS headroom `0.8131401783020511`; PLUS differs from its MAP-only action on `0.709` of replay decisions. |
| Five of six methods optimize the shared learned surrogate; EVD uses raw logged rewards | **VERIFIED** | `READONLY_EXTRACTION_PASS_2.md` P0-2. Surrogate fit R² is about 0.30–0.73 and it cannot represent the hard safety cliff. |
| Measured per-cell costs are roughly PLUS 15,305 s, MOOR 1,938 s, BA-MCTS 847 s, OGSRL 430 s, RefPlan 52 s, EVD 2.5 s | **VERIFIED** | `READONLY_EXTRACTION_PASS_2.md` P0-4. |
| The old cross-family PLUS route contains the five-hand-picked-policy baseline bug | **NOT VERIFIED; record indicates a conflation** | The five-policy bug belongs to the separate S2 full-state VPI baseline (`in new repo/S2_ORACLE_REVIEW.md`). The frozen cross-family PLUS design has 16 candidates. It still must not be used here because it includes the true family and would defeat the intended Ricker-misspecification stress test. |
| The live/current server code differs from the accepted code in a way that changes this cell | **OPEN** | The local handoff says the clean repository preserves byte-identical accepted tracks and that one accepted cell reproduced bit-for-bit. It also documents a branch-isolated OGSRL short-episode fix. The exact live HEAD/branch and behavior of tiger × Allee × 0.2 must be checked on the server. |
| A zero/epsilon-scale likelihood is unsafe for Arm T | **VERIFIED, with scope qualification** | `AUDIT_IMPLEMENTATION_BRIEF.md` documents the ecological all-zero-likelihood fallback for off-grid positive observations when `observation_scale <= 1e-12`, plus related narrow-likelihood/underflow traps. Arm T must use direct state/belief assignment; neither sigma nor observation scale is driven toward zero. |

---

## 3. Correct method roles

Use the accepted names and fidelity labels. Do not relabel the methods as exact
implementations of their source papers.

| Method | Role in this pilot | State/observation treatment | Model/value uncertainty | Reward used in planning/training |
|---|---|---|---|---|
| adapted PLUS | joint ecological belief baseline | candidate-specific latent-abundance beliefs; explicitly uses the survey likelihood and sigma | posterior over eight fitted Ricker candidates | shared learned linear surrogate |
| adapted MOOR | ecological state-belief baseline | one latent-abundance belief; explicitly uses the survey likelihood and sigma | one fitted Ricker point model | shared learned linear surrogate |
| RefPlan-inspired | partial/conflated general-method control | public particles/history-conditioned prior; explicitly uses `LogNormalObservationModel` and sigma | five-member observation-space posterior/ensemble | shared learned linear surrogate |
| OGSRL-inspired | guarded fixed-policy general method | public observation-space belief/proxies; does not use sigma explicitly | OOD/support and predictive ensemble machinery; fixed actor after training | shared learned linear surrogate |
| BA-MCTS-inspired | online model-belief general method | bucketed public observation, not a calibrated latent-abundance belief; does not use sigma | categorical model belief updated inside tree search | shared learned linear surrogate |
| EVD pessimism | low-cost value-disagreement control | current public features; no latent-state filter and no sigma use | variance across 20 bootstrap conservative Q estimators; no online update | raw logged rewards |

RefPlan is a decisive control. A comparison containing only OGSRL, BA-MCTS and EVD
would make the observation-model asymmetry partly true by construction.

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

For adapted PLUS and MOOR, use the audited direct-assignment convention: route current
truth through an external oracle-state interface, convert raw abundance to the model's
latent units using `survey_scale`, and set the internal abundance belief to a delta on the
registered nearest latent bin immediately before every decision. Keep the original
observation model and sigma unchanged; do not call the base Bayesian update for runtime
truth delivery.

For RefPlan, design the equivalent exact-state adapter for its public particle/history
representation without changing sigma or observation scale. Whether this can be done with
frozen fitted artifacts is an I0/I2 question, not an assumption. OGSRL, BA-MCTS and EVD
use separately audited end-to-end exact-state input contracts.

### Arm O — noisy survey information

The method receives the accepted public survey information. PLUS, MOOR and RefPlan may
use their existing observation-likelihood machinery. OGSRL, BA-MCTS and EVD retain their
accepted method logic and therefore do not begin consuming sigma merely for this pilot.

### Mandatory design audit before coding

The six methods do not share one state interface. Before implementation, produce a
method-by-method table stating whether the information change affects:

- offline training inputs;
- fitted transition/value artifacts;
- reward-surrogate fitting;
- policy fitting;
- online state/belief update;
- action selection only.

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

---

## 7. Regression and validity gates

### Gate G0 — provenance

Before running anything, first count and report accepted training episodes shorter than
the registered behavior horizon in both pilot cells, separated into genuine terminated
collapses, truncations and invalid/incomplete records. Then record:

- live server path, Git HEAD, branch and dirty status;
- hashes of both frozen tracks;
- exact dataset, registration, action-table and evaluator hashes;
- Python/dependency environment;
- Slurm partition, CPU model and requested resources;
- whether the OGSRL short-episode fix is present and on which branch.

Abort if the accepted artifacts or frozen tracks would be modified.

### Gate G1 — tests

Run the repository self-tests plus the OGSRL regression tests requiring:

- bit-identical behavior on full-length data;
- correct absorbing-terminal padding for legitimate one- and two-step collapses;
- rejection of short nonterminal truncations.

Abort on any failure.

### Gate G2 — accepted-cell regression anchor

Rerun noisy-survey OGSRL for tiger × Allee × sigma 0.2 and compare with the accepted
artifact at three levels:

1. per-episode returns;
2. action sequences and ecological event counts;
3. aggregate return mean and SD.

If the accepted CPU architecture and behavior-preserving code path are reproduced, require
bit parity or the repository’s existing exact-parity criterion. If that architecture is
unavailable, do not invent a tolerance after seeing results: either obtain it or rerun an
explicit baseline and the new arms on one pinned architecture, then independently approve
the new baseline.

Any unexplained difference stops the study. Create a regression report and do not submit
Stages B or C.

If the only differences are localized to genuine short terminated episodes affected by
the documented OGSRL absorbing-terminal fix, and all three OGSRL regression tests pass,
the gate outcome is `EXPLAINED FIX DELTA — REBASELINE REQUIRED`, not immediate scientific
failure. On one pinned architecture, rerun an explicit current-code noisy baseline,
independently audit it, and obtain user approval before it becomes the reference. Any
delta outside the documented fix path remains a stop condition.

**Current result:** NOT RUN. No regression conclusion is available.

### Gate G3 — paired-run integrity

For every method and sigma:

- Arm T and Arm O must contain the same 20 ordered evaluation identities;
- environment process draws must be paired where the method interface permits;
- reward reconstruction must equal stored return within the registered tolerance;
- no task may silently fall back to another method or information mode.

For each Stage B Arm O rerun, also compare per-episode returns, action sequences and event
counts with its accepted counterpart on the same architecture. An unexplained mismatch
invalidates that method-cell. A documented behavior-preserving-fix delta follows the same
independently approved rebaseline path as G2; it is never silently accepted.

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

### Evidence that consuming the observation model may be sufficient

If RefPlan is materially more robust than OGSRL, BA-MCTS and EVD, and is comparable to
active PLUS/MOOR, then explicit use of sigma/observation likelihood may explain much of
the gap. This weakens the case that an entirely new joint architecture is already needed.

### Support for a broader belief-representation limitation

If active PLUS/MOOR remain robust while RefPlan degrades similarly to the other general
methods, then merely consuming sigma is insufficient. A calibrated latent-abundance
belief becomes a stronger candidate mechanism.

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

Using the measured mean full-task times from `READONLY_EXTRACTION_PASS_2.md`:

```text
one arm, six methods, one cell       ≈ 18,588 s ≈ 5.16 core-hours
two arms, two cells, one sigma       ≈ 20.65 core-hours
two arms, two cells, both sigmas     ≈ 41.31 core-hours
```

Add the OGSRL regression anchor, test time, analysis and failed-task contingency.
PLUS dominates wall time at approximately 4.25 hours per method-cell. Parallel execution
can reduce elapsed time, but every task must remain separately receipted and the experiment
must be pinned to one approved CPU architecture.

The estimate assumes existing pipelines can support the information adapters. Adapter
implementation and audit—not compute—may be the dominant schedule risk.

---

## 11. Implementation stages for the Codex server agent

The server agent must stop after each stage and return the listed audit artifact. It must
not continue automatically across a failed or unaudited gate.

### I0 — read-only reconnaissance

- Read repository `README.md`, `REPRODUCE.md`, `docs/INDEX.md`, registrations and both
  frozen-track manifests.
- Identify exact current implementations and input contracts for all six methods.
- Identify accepted tiger × Allee and fox × Allee datasets and output artifacts.
- Count short terminated, truncated and incomplete training episodes before any code work.
- Report the adapted PLUS Arm T component-change row first: whether eight Ricker fits can
  be reused unchanged and only its runtime beliefs collapsed. This determines the dominant
  compute and wall-time risk.
- Produce `I0_RECONNAISSANCE_REPORT.md` containing paths, hashes, live HEAD, dirty status,
  available CPU architecture and discrepancies from this plan.
- Make no code change and submit no job.

**Stop for user + Claude audit.**

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
- If any gate fails, set verdict `FAIL — STOP AND RE-BASELINE` and submit nothing else.

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
| Read-only server reconnaissance | **Not yet authorized by this draft** |
| Create/edit experiment code | **Not authorized** |
| Create a registration/output namespace | **Not authorized** |
| Run regression gate | **Not authorized** |
| Conditional pinned-architecture rebaseline after a documented-fix delta | **Claude recommends pre-authorization; user authorization still required** |
| Submit Stage B two-cell sigma 0.2 jobs | **Not authorized** |
| Submit Stage C sigma 0.1 jobs | **Not authorized** |
| Modify accepted outputs/frozen tracks | **Permanently forbidden** |
| Edit slides from pilot results | **Not authorized** |

The audited plan must be explicitly approved before the server agent performs any action.
Approval of Stage B does not automatically authorize Stage C.

If the user pre-authorizes the conditional rebaseline path, it is valid only when all four
conditions hold:

1. all three documented OGSRL regression tests pass;
2. every delta is localized to genuine terminated short episodes;
3. the new baseline receives an independent per-episode/action/event diff audit, not only
   an aggregate-mean check; and
4. any supervisor-deck figure compared with post-rebaseline results is labelled explicitly
   as an old-code value until regenerated under the new baseline.

---

## 14. Audit questions for the user and Claude

The prior audit resolved the estimand as end-to-end robustness, added the fox activity
cell, accepted raw paired loss as primary, and retained sigma 0.2 as screening only.
The remaining server-factual or authorization questions are:

1. Can each Arm T contract be implemented without hidden-information leakage, and which
   narrower frozen-fit diagnostics are genuinely available?
2. What CPU architecture produced each accepted anchor, and is it currently available?
3. Which live branch contains the OGSRL short-episode fix, and has it passed its three
   documented regression tests?
4. How many genuine short terminated episodes occur in each accepted pilot dataset?
5. If the regression gate produces only documented-fix deltas, does the user authorize
   the pinned-architecture rebaseline path?
6. After Stage B, is Stage C separately authorized under its non-discrimination stop rule?

Until these questions are resolved and the registration is frozen, the correct server
action is **read-only inspection only**.
