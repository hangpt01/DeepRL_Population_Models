# Phase 1B: Deep paper verification and corrective plan for the general-RL baselines

## Purpose

Continue from `PHASE1_GENERAL_RL_BASELINE_PAPER_ALIGNMENT_AUDIT.md`.

The Phase 1 source audit is accepted as strong evidence about the local implementation. In
particular, treat these as confirmed local-code findings unless a direct recheck disproves them:

- hidden RefPlan computes a posterior but does not use it for planning, making its hidden action
  path behaviorally equivalent to the hidden MOPO planner;
- hidden OGSRL returns before actor/critic/dual training and is therefore a one-step constrained
  greedy method rather than a trained guarded policy;
- hidden BA-MCTS samples a model at the root but does not update its model belief inside the tree;
- Delphic exists but is absent from the hidden comparison manifest;
- the hidden implementations share lightweight public-state infrastructure and a 4,000-transition
  offline budget;
- privacy tests currently pass, but provenance, relabel-invariance, and value-leakage coverage need
  strengthening.

However, do **not** implement the Phase 1 recommendations yet. The Phase 1 report acknowledges that
some paper-mechanism judgments were based on abstracts or first-page evidence. Before prescribing
paper-aligned corrections, verify the complete methods from the full papers, appendices,
supplements, and official author code where available.

## Scientific standard

These four methods are ingredients and comparison methods from which the eventual proposal will be
developed. They do not need the near-reproduction standard required of the ecological baselines
PLUS and MOOR. They must nevertheless:

1. retain each paper's defining algorithmic mechanism and remain recognizably that method;
2. disclose every material departure from the paper;
3. adapt legitimately to this benchmark's offline data, discrete management actions, partial
   observations, finite horizon, safety objective, and hidden demographics;
4. never access private `r`, `K`, private safety quantities, privileged simulator state, or exact
   private reward information in hidden mode;
5. use a matched primary offline-data budget of 4,000 transitions, unless a preregistered matched
   sensitivity study later changes the budget for all compared methods;
6. distinguish a necessary framework adaptation from a computational approximation and from a
   replacement that destroys paper identity.

The retained set is **RefPlan, OGSRL, BA-MCTS, and Delphic**. Use the canonical name **OGSRL**.
MOPO should remain in the repository for provenance and archival reproducibility but should remain
excluded from the retained headline comparison. Do not delete it.

## Constraints for this phase

- **Plan and audit only. Do not edit runtime code, tests, manifests, registrations, or result
  documents.**
- Do not launch, cancel, alter, or inspect any experiment jobs.
- Do not inspect performance returns.
- Do not disturb the active paper-aligned PLUS/MOOR jobs or their artifacts.
- Preserve the current dirty working tree. Do not reset or overwrite uncommitted work.
- Do not present a mechanism as paper-required unless it is supported by the full paper,
  supplement, or official code.
- If a full source is unavailable, mark the claim unresolved rather than inferring it from an
  abstract.

## Task 1: Establish authoritative paper evidence

For each retained method, identify and read:

- the complete paper;
- all appendices or supplementary material;
- official author code and configuration files, if publicly available;
- any official errata or project documentation needed to interpret the algorithm.

Record exact bibliographic identity, URLs or local paths, paper sections, algorithm numbers,
equation numbers, appendix pages, and official-code file/function/config references. Separate:

- `PAPER`: stated in the paper or supplement;
- `OFFICIAL-CODE`: demonstrated by author code;
- `LOCAL-CODE`: demonstrated only by this repository;
- `INFERENCE`: your reasoned interpretation;
- `UNVERIFIED`: inaccessible or not established.

Do not use a README, abstract, or first page as sufficient evidence for algorithm internals.

## Task 2: Define the minimum identity-preserving core

For each method, extract the smallest set of mechanisms that must survive adaptation for the local
method to carry the paper name. Answer at least the following.

### RefPlan

- What precisely is reflected on, what posterior or uncertainty object is formed, and how must it
  affect deployment planning?
- What role does the base offline policy play?
- Which conservative or pessimistic components are defining rather than optional?
- Does the paper require posterior-weighted planning, posterior sampling, or another specific
  integration rule?

### OGSRL

- Which actor, critic, guard, support/OOD model, safety cost, dual, and rollout-training components
  are defining?
- What is learned from offline data, and what can legitimately be replaced by a public surrogate
  in hidden mode?
- Under what conditions, if any, could a one-step constrained policy still be called OGSRL?
- Which safety claims must be removed when the benchmark's private safety objective is unavailable?

### BA-MCTS

- Is belief/posterior updating inside simulated tree histories required by the published method?
- What is sampled at the root, what is updated at each node, and what constitutes the Bayes-adaptive
  state?
- Is an outer policy-improvement/value-learning loop essential, or can the planner itself be a
  faithful evaluation-time baseline?
- Which mechanisms are needed for discrete actions, and which continuous-action machinery can be
  omitted legitimately?

### Delphic

- How are compatible worlds constructed in the paper and official code?
- What hidden-confounding assumptions are essential?
- Can this benchmark's hidden demographics/partial observation instantiate the paper's ambiguity
  set without an explicit unobserved action-outcome confounder?
- If not, what honest adaptation is possible, and must the method be named `Delphic-inspired`
  rather than `Delphic`?

For every answer, cite the full source evidence.

## Task 3: Audit both known and hidden implementations

Trace the actual implementation of each method in both regimes:

- the committed known-`r,K` path at `5f9cf32`;
- the current uncommitted hidden-`r,K` path;
- the copied hidden-run snapshot, only as provenance evidence.

Produce one matrix per method with these columns:

| Paper-defining component | Full/known implementation | Hidden implementation | Evidence | Classification | Required action |
|---|---|---|---|---|---|

Use these classifications:

- faithful/retained;
- necessary benchmark adaptation;
- privacy-required adaptation;
- computational approximation;
- material replacement;
- missing/inert;
- not applicable;
- unverified.

Check whether the earlier known-`r,K` repair retained the method's paper identity. Do not assume the
full implementation is correct merely because it is committed or previously ran.

## Task 4: Reassess the Phase 1 findings against full paper evidence

For every Phase 1 finding N1-N10, return one of:

- confirmed unchanged;
- confirmed, but correction needs modification;
- downgraded to disclosure/verification only;
- disproved;
- unresolved.

Pay special attention to whether the proposed simple fixes are genuinely sufficient:

- RefPlan: merely passing posterior weights into the existing particle planner may not reproduce
  the paper's defining reflection/planning procedure;
- OGSRL: merely enabling the existing full-mode actor path may not preserve the paper's guard,
  safety-cost, and support semantics under hidden demographics;
- BA-MCTS: a depth/simulation sensitivity cannot compensate for a missing Bayes-adaptive belief
  update if that update is defining;
- Delphic: adding it to the manifest cannot cure a mismatch between hidden demographic uncertainty
  and hidden confounding.

## Task 5: Design justified adaptations for this benchmark

For each method, describe the proposed adapted algorithm in enough detail to implement and audit:

- offline inputs and the exact 4,000-transition train/holdout split;
- public observation/history representation;
- learned model, posterior, ambiguity set, policy/value model, and safety/support model;
- training stages and losses;
- planning or action-selection equations;
- online belief/history update during evaluation;
- use of the public reward and public safety/risk surrogate;
- prohibited private inputs;
- deterministic seeding and artifact boundaries.

For every departure from the original paper, provide:

| Adaptation | Why the framework requires it | Effect on interpretation | Alternative considered | Naming consequence |
|---|---|---|---|---|

The implementation may use benchmark-compatible lightweight models for deadline and compute
reasons, but identify when a shared ridge-linear model makes the methods only planner/policy
variants. Recommend method-specific model classes only where needed to preserve algorithm identity;
do not add neural complexity merely for cosmetic similarity.

## Task 6: Hyperparameters, data fairness, and compute

Create a method-by-method hyperparameter table containing:

- paper value/range and exact source;
- official-code default, if different;
- current known-mode value;
- current hidden-mode value;
- proposed adapted value;
- provenance: paper, official code, benchmark adaptation, or computational limit;
- expected sensitivity and scientific risk;
- whether changing it requires refitting or only replanning.

Keep 4,000 offline transitions as the matched primary budget. Also propose a return-blind adequacy
and sensitivity design that can determine whether 4,000 is insufficient without selecting the
budget from performance returns. Include difficult sink populations and action-coverage diagnostics.

For expensive paper defaults, propose:

1. a paper-near value;
2. a compute-scaled primary value;
3. a small preregistered sensitivity that establishes the scaled value is not qualitatively
   collapsing the algorithm.

Estimate fit time, planning time, memory, CPU/GPU needs, and total wall time under the server limits.

## Task 7: Corrective implementation plan

Return a staged, file-specific plan, but do not execute it. Include:

1. provenance isolation of the current hidden changes;
2. RefPlan identity correction;
3. OGSRL identity correction;
4. BA-MCTS belief-state and distinctness correction/verification;
5. Delphic scientific-scope decision and manifest treatment;
6. privacy and relabel-invariance tests;
7. paper-mechanism unit tests on small constructed problems;
8. return-blind data/hyperparameter checks;
9. deterministic runtime canaries;
10. snapshot/digest/registration procedure;
11. matched experiment manifests at 4,000 transitions;
12. criteria for deciding whether each method keeps its paper name or must use an `-inspired`
    label.

For each stage list files, functions, tests, dependencies, expected runtime, rerun scope, and a
rollback strategy that does not disturb unrelated dirty-tree work.

## Task 8: Decisions that require user approval

End with a short decision table. At minimum decide or ask for approval on:

- whether Delphic is scientifically meaningful in this benchmark as implemented/adapted;
- whether BA-MCTS must include in-tree belief updates and an outer learning loop;
- whether lightweight shared models are acceptable for the primary comparison;
- whether to add behavior cloning and/or CQL/IQL only as contextual diagnostic references rather
  than as members of the retained four;
- the compute-scaled hyperparameters and sensitivity budget;
- whether every retained method should be reported explicitly as paper-inspired.

## Required output

Write:

`PHASE1B_GENERAL_RL_DEEP_PAPER_VERIFICATION_AND_CORRECTIVE_PLAN.md`

The report must contain:

1. source/provenance ledger;
2. paper-mechanism summaries with exact citations;
3. known-versus-hidden implementation matrices;
4. reassessment of Phase 1 N1-N10;
5. adaptation register;
6. hyperparameter and data-budget tables;
7. staged corrective implementation plan;
8. tests and acceptance gates;
9. runtime estimate;
10. user decision table;
11. final verdict for each method: `retain`, `fix`, `rename`, `exclude`, or `unresolved`.

Stop after returning the report. Do not implement anything until the report is reviewed and the
user explicitly authorizes the selected plan.
