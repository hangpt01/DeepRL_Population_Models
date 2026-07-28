# Handoff: three-species matched P=10 ecological-baseline experiment

Date: 24 July 2026 (Australia/Melbourne)

## Copy-paste starter prompt for the next chat

Read this handoff completely, then read
`merged_results_three_species_matched_general_and_ecological.tex`. Treat the
P=10 three-species comparison as the current accepted result. I now want to
design follow-up experiments that determine which uncertainties the ecological
baselines already handle (parameter, process, observation, partial
observability, and fitted-model uncertainty) before motivating a new method for
uncertainty over the underlying population-model family. Do not assume that the
current results prove those uncertainties are solved: first identify
decision-discriminating ablations, oracle controls, independent fit/data
replicates, and acceptance criteria. Keep recoverable and sink species
separate, preserve the common evaluator, and distinguish policy degeneracy from
genuine uncertainty handling.

## What was completed

A matched safe-reward comparison was completed for:

- Species: Amur tiger, Crab-eating fox, and Egyptian vulture.
- Hidden simulator families: Ricker, Allee, theta-logistic, and
  regime-switching.
- Observation noise: 0.1 and 0.2.
- Methods: RefPlan-inspired, OGSRL-inspired, BA-MCTS-inspired,
  EVD/value-disagreement pessimism, adapted PLUS, and adapted MOOR.
- Final scope: 3 species x 4 families x 2 noise levels x 6 methods = 144
  method-cells.
- Evaluation: 20 episodes per method-cell, 50 steps, discount 0.95.

The four general-RL methods contributed 96 already accepted safe-reward cells.
The correction experiment planned and evaluated 24 PLUS and 24 MOOR cells under
the same safe-reward penalty, giving 48 new ecological method-cells.

The ecological implementations are registered adaptations, not exact paper
reproductions:

- PLUS: `plus_adapted_ricker_only_pbvi`, with eight Ricker candidates and a
  uniform candidate prior.
- MOOR: `moor_adapted_ricker_misspec_pbvi`, with one fitted Ricker model.

Neither ecological method receives the hidden family label. Ricker is the
in-family control; Allee, theta-logistic, and regime-switching are
out-of-family stress tests.

## Why the P=10 correction was necessary

The first merged report exposed a material mismatch:

- General RL used unsafe-occupancy penalty P=10.
- Historical PLUS and MOOR used P=5.

Those returns were not fairly comparable. The valid correction did not merely
re-score policies planned for P=5. It:

1. reused reward-independent fitted demographic artifacts after identity and
   hash validation;
2. rebuilt reward-dependent POMDP components with P=10;
3. re-solved every PLUS and MOOR planner under P=10; and
4. evaluated the new policies under the common P=10 evaluator.

The historical P=5 outputs remain unchanged and are retained only as unmatched
historical results.

## Common evaluation definition

All six methods use:

```text
R_t = N_{t+1}/(N_{t+1}+K_ref)
      - cost(a_t)
      - 10 * 1[N_{t+1} <= s_safe]
```

They also share:

- 50-step horizon and discount 0.95;
- deterministic registered population-specific initial state;
- exact registered hidden simulator and demographic inputs;
- log-normal observation model at sigma 0.1 or 0.2;
- identical ordered 11-action table and intervention effects;
- 20 evaluation seeds:
  7001-7004, 7051-7054, 7101-7104, 7151-7154, and 7201-7204;
- the same definitions of unsafe occupancy, persistence, collapse entry, and
  abundance;
- one exact authoritative dataset hash for all six methods in each matched
  species-family-noise cell.

## Execution and acceptance

- Immutable experiment root:
  `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_ecological_p10_correction_20260723_v1/`
- PLUS array: Slurm `58493916`, 24/24 completed, exit 0:0.
- MOOR array: Slurm `58493918`, 24/24 completed, exit 0:0.
- PLUS acceptance: Slurm `58493967`,
  `PASS_LIMITED_STRUCTURAL_ACCEPTANCE`, 24/24.
- MOOR acceptance: Slurm `58493968`,
  `PASS_LIMITED_STRUCTURAL_ACCEPTANCE`, 24/24.
- Returns were sealed during structural acceptance and opened only after both
  gates passed.
- Total allocated computation: 115.13 core-hours.
- All 216 reward-independent fitted-model slots were reused:
  192 PLUS candidate slots and 24 MOOR model slots.
- No demographic fit was recomputed.

Frozen identifiers reported by the server:

- Launch-package digest:
  `13f6fcdf12450efb09f3c62c60db05c4c41007f64356a3999eb5d38d7e34368d`
- PLUS manifest:
  `cfe43f35ce4ff1658958a2173026ca70c7164878f2408b4b246673016dc0d2a6`
- PLUS configuration:
  `10a064344e07cac16a7b2d5bd717bdce109c8970148de4295d6ff3532297fa68`
- MOOR manifest:
  `f890c9a7a535fe903bf7a40ecf992a9881341b84103b9e0dc7e10be002fc6d57`
- MOOR configuration:
  `a3655a846d951439177da350ddf40ddc6d4962b42b317582dd29e72ca05323bd`
- Fit-reuse ledger:
  `5410b0e4fd71312af601cbdd4c8f11115469483780f26c202fbe232916187ac3`
- Accepted matched CSV:
  `MATCHED_P10_144_METHOD_CELLS.csv`
- Matched CSV SHA-256:
  `7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c`
- Outcome/validation receipt:
  `MATCHED_P10_144_RECEIPT.json`

## Main result pattern

The report ranks mean return descriptively within each family-noise cell. Bold
is the highest mean; underline is the second-highest distinct mean; ties share
rank. These marks do not establish statistical significance.

### Amur tiger: declining but recoverable

- Across the eight cells, PLUS and MOOR tie for the largest mean return at
  4.234; OGSRL is next at 3.607.
- PLUS and MOOR are constant-policy in all eight cells and have essentially
  identical returns.
- All methods except EVD have zero unsafe occupancy and full persistence.
- EVD has unsafe occupancy around 0.24-0.31 and zero persistence.

Interpretation guardrail: the ecological result may be explained by a dominant
fixed intervention. It does not yet show that PLUS candidate uncertainty or
MOOR model fitting is the causal reason for good performance.

### Crab-eating fox: strong-growth recoverable

- Across eight cells, PLUS has mean 10.590 and MOOR 10.584.
- Cell-level winners vary:
  - MOOR is largest in the two Allee cells.
  - BA-MCTS is largest in the two theta-logistic cells.
  - PLUS is largest in the two regime-switching cells.
  - PLUS and MOOR split the Ricker-noise cells.
- The ecological policies are mostly non-constant, so this species is more
  decision-discriminating than Amur tiger.
- RefPlan has occasional collapse/unsafe behavior and very high return
  dispersion in regime-switching at noise 0.2.

Interpretation guardrail: the tiny aggregate PLUS-MOOR difference is not
evidence that eight-candidate uncertainty is superior to the MOOR point model.
Independent fit/data replicates are absent.

### Egyptian vulture: demographic sink

- RefPlan has the least-negative return across the eight cells.
- Every method has unsafe fraction 1 and persistence 0 in every cell.
- Zero collapse-entry rate is not success because the initial state is already
  unsafe.
- PLUS and MOOR are constant-policy and tied in every cell.

Interpretation guardrail: this is an apparent recoverability/controllability
failure under the registered horizon and action set. Return ranking reflects
abundance and cost differences among policies that all fail the biological
safety objective.

## What is not yet demonstrated

The completed experiment does **not** establish that:

- ecological baselines have solved parameter uncertainty;
- PLUS's multiple candidates improve decisions over MOOR's point model;
- model-form misspecification is harmless globally;
- the three selected species represent all nine species;
- 4,000 transitions are adequate in rare or dangerous state-action regions;
- rankings persist across independently collected datasets or refitted models;
- zero within-cell SD means robustness;
- any method can recover the Egyptian-vulture sink;
- observed mean differences are statistically resolved.

Within-cell SD is variation across 20 evaluation episodes conditional on one
registered dataset and one fitted policy. It is not uncertainty across datasets,
fit seeds, or candidate banks.

## Recommended next scientific step

Before introducing uncertainty over the underlying population-model family,
test whether the ecological baselines genuinely handle the uncertainties that
are presumed to be easier.

Use a staged, matched ablation on decision-discriminating cells:

1. Oracle full dynamics and state information.
2. Known Ricker family and known demographic parameters, with observation and
   process randomness retained.
3. Known Ricker family but fitted parameters.
4. MOOR's fitted point-model route.
5. PLUS's fitted Ricker-candidate route.
6. Only after these comparisons, a method with explicit uncertainty over
   Ricker/Allee/theta-logistic/regime-switching family identity.

For every stage, preserve the same evaluator and dataset identities and report:

- return, unsafe occupancy, persistence, and abundance;
- action frequencies and action entropy;
- state/belief occupancy;
- top-two action-value margins;
- disagreement among candidate or oracle policies;
- independent dataset and fit replicates.

Amur tiger alone is insufficient for this ablation because its current
ecological policies are constant. Crab-eating fox is presently the most useful
of the three for detecting changes in decisions. Egyptian vulture should first
receive an oracle reachability/controllability analysis; it should not be used
as ordinary evidence of uncertainty resolution if no registered policy can
recover it.

## Data and hyperparameter adequacy work

High-priority checks:

- Nested transition budgets, beginning with 4,000 versus 8,000.
- Multiple independent collection seeds and refits.
- Coverage of low-abundance states and rare actions, not transition count alone.
- PLUS candidate count/diversity and actual candidate-policy disagreement.
- PBVI belief-grid size, iterations, observation discretization, convergence,
  and action-value margins.
- Return-blind sensitivity checks for MCTS budget, EVD penalty, OGSRL
  constraint/guardian settings, and RefPlan prior strength.
- A registered reward-sensitivity analysis to test whether the binary P=10
  occupancy penalty creates policy saturation.

Do not select hyperparameters by reading the final test returns. Use predefined
structural diagnostics, training/validation cells, or a separate development
domain.

## Local report files

- `merged_results_three_species_matched_general_and_ecological.tex`
- `merged_results_three_species_matched_general_and_ecological.pdf`

The local TeX has been extended with:

- bold best and underlined second-best descriptive return rankings;
- explicit tie handling;
- a detailed interpretation and possible-mechanisms section;
- prioritized improvement and follow-up experiments.

Final local SHA-256:

- TeX:
  `24b1feb51057e50d94f1934b0f3b94d86e12dfe4914e0efc46d3da68a9bb302e`
- PDF:
  `48162293db20904a528a69d2d696f5c6ee330093242d3ff02646038bd09d31c6`

## Reporting rules to preserve

- Always say “adapted PLUS” and “adapted MOOR,” not exact reproductions.
- Keep Ricker in-family and the other three families out-of-family.
- Keep species separate and never pool the sink into a recoverable-species
  headline.
- Keep safe and yield rewards separate; PLUS and MOOR have no matched yield
  experiment here.
- Do not mix historical P=5 ecological returns with the fair P=10 table.
- Treat bold/underline as descriptive only.
- Do not claim algorithmic equivalence from equal returns.
- Do not claim uncertainty resolution from a constant policy.
- Do not interpret zero collapse entry as safety when the population starts
  below the threshold.
