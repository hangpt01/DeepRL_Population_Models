# Code-Server Audit Questions: General-RL Baselines Under Hidden Demographics

## Purpose

Audit the current implementations of the general-RL baselines before any further fixes or experiment runs.

The retained baseline set is provisionally:

1. RefPlan
2. OGSRL/OSGRL -- first resolve the canonical name
3. BA-MCTS
4. Delphic

MOPO is being removed from the retained comparison because its principal paper is from 2020. Do not delete or modify MOPO during this audit; only identify whether removing it would affect shared code, configurations, manifests, reports, or tests.

The intended scientific standard differs from the ecological-baseline audit:

- These general-RL baselines should preserve the defining algorithmic ideas of their cited papers as closely as reasonably possible.
- Necessary adaptation to the conservation benchmark, hidden demographics, discrete management actions, partial observations, offline episodes, and shared reward must be explicit.
- They need not reproduce every paper-specific domain detail or solver exactly, because they are general methods from which the later proposal will be developed.
- They must not be given hidden `r`, `K`, true dynamics family, private action effects, latent states, private safety thresholds, or evaluation returns.
- They should receive the same registered offline-data opportunity as the ecological baselines: 4,000 target transitions per environment cell while preserving complete episodes, with the exact train/holdout availability reported.
- No paper-fidelity claim may be based only on a method name, comment, or configuration label.

## Mandatory sequencing

### Phase 1: inspect and report only

Read the current repository, historical known-`r,K` implementation, hidden-`r,K` fixes, method registry, tests, configurations, manifests, and relevant local paper/audit documents.

Do **not**:

- edit code or documentation;
- remove MOPO;
- launch, resume, cancel, or alter jobs;
- regenerate data;
- inspect `operational_return`, `true_return`, survival return, rankings, or comparative result summaries;
- tune hyperparameters;
- infer paper alignment from previous performance.

Return a detailed evidence-based audit using the requested structure, then stop and wait for review.

---

# Questions and Required Evidence

## A. Repository, version, and job state

1. What repository path, branch, commit, runtime digest, and working-tree state are currently active?
2. Which snapshot or commit produced the existing known-`r,K` general-baseline implementation?
3. Which snapshot or working-tree changes produced the hidden-`r,K` fixes?
4. Are corrected general-baseline changes committed, frozen only by digest, or still mixed with unrelated work?
5. Are any general-RL jobs active, pending, completed, failed, or cancelled?
6. Which result directories belong to known-`r,K`, hidden-`r,K`, provisional, superseded, or void runs?
7. Which artifacts may be inspected without opening performance returns?

Provide exact paths, commit hashes, digests, job IDs/states, and artifact labels.

## B. Canonical method and paper identity

For each retained method, report:

1. canonical method name and acronym;
2. current registered method ID;
3. exact paper title, authors, venue, and year currently claimed as provenance;
4. DOI, arXiv identifier, or official paper URL recorded in the repository;
5. whether the repository includes the paper, supplement, official code, archived code, or only a secondary description;
6. official implementation repository/version, if one is claimed;
7. current implementation files, registry entry, configuration, runner, and tests;
8. whether the method is actually paper-derived, inspired by a paper, a project-specific baseline, or a hybrid.

Resolve whether `OGSRL` or `OSGRL` is correct. Search code, documentation, paper metadata, report labels, and artifacts. Recommend one canonical spelling and list every inconsistent occurrence, but do not edit it yet.

If RefPlan, BA-MCTS, or another name has no single matching paper algorithm, say so directly. Do not invent provenance.

## C. Original-paper algorithm summary

For each method, summarize the original paper's defining algorithmic components independently of the current code.

At minimum cover:

1. learning setting: online/offline, model-based/model-free, fully/partially observed;
2. state, action, transition, observation, and reward assumptions;
3. learned objects: dynamics model, uncertainty model, value function, policy, belief, confounder model, or ensemble;
4. training objective and regularization;
5. uncertainty representation;
6. planning or policy-optimization procedure;
7. treatment of out-of-distribution actions/states;
8. safety mechanism, if any;
9. data collection assumptions and paper data budgets;
10. architecture and optimization defaults;
11. principal ablations required to establish the paper's claimed mechanism;
12. components that define the method versus components that are replaceable implementation details.

Distinguish facts verified from the paper/supplement/code from facts inferred from local documents.

## D. Current implementation flow

For each retained method, trace the real current execution path from sanitized offline data to selected action.

Show:

1. exact input fields read;
2. normalization and preprocessing;
3. train/holdout splitting;
4. learned model/policy/belief structures;
5. loss functions and equations;
6. optimizer and stopping criteria;
7. uncertainty estimation;
8. planner or actor operation;
9. safety/OOD logic;
10. state filtering or use of observation history;
11. how episode boundaries and temporal order are used;
12. outputs saved for diagnostics;
13. any shared implementation such as `ContinuousDynamicsEnsemble`;
14. any fallback, shortcut, oracle, table lookup, or native solver path.

Provide file paths, function/class names, and line numbers. Include equations or pseudocode where needed.

## E. Paper-to-code fidelity matrix

Create one component matrix per method:

| Original-paper component | Current implementation | Evidence | Classification | Consequence |
|---|---|---|---|---|

Use exactly these classifications:

- **Faithfully retained**
- **Necessary benchmark adaptation**
- **Disclosed computational approximation**
- **Optional extension**
- **Material replacement**
- **Missing**
- **Not applicable**
- **Unverified**

For every material replacement or missing defining component, explain whether the current method name remains defensible. Recommend paper-faithful, paper-aligned, paper-inspired, or project-specific wording.

## F. Known-`r,K` to hidden-`r,K` change audit

For each retained method:

1. What did the known-`r,K` implementation receive directly or indirectly?
2. Which code paths, features, priors, normalization values, planner inputs, reward inputs, or filenames depended on true `r`, `K`, dynamics family, action effects, latent state, or safety information?
3. What exact changes were made for hidden demographics?
4. Were hidden quantities removed, estimated, marginalized, replaced by public proxies, or accidentally retained?
5. Did the hidden fix alter only information access, or did it also change architecture, optimization, planning, or compute budgets?
6. Can the known and hidden versions be considered a clean information ablation?
7. Are any old known-parameter table lookups or environment objects reachable from the hidden method IDs?
8. Can population/family identity, path names, manifest ordering, or configuration choices indirectly reveal private structure?

Give source-level evidence and tests. Do not characterize the known-`r,K` implementation as wrong merely because it is information-rich.

## G. Privacy and leakage audit

For every method, prove whether hidden mode blocks:

- true `r` and `K` or their benchmark analogues;
- true family;
- true action-effect magnitudes;
- latent true abundance/state;
- regime state and transition matrix;
- private safety threshold/objective;
- evaluator reward components unavailable to the method;
- future observations or next states not present in an allowed training transition;
- evaluation seeds or evaluation trajectories;
- result summaries and return fields;
- population/family leakage through filenames, array positions, caches, or method-specific configuration.

Report existing privacy tests and missing tests. Include relabelling, table-independence, forbidden-import, and manifest/path leakage checks where applicable.

## H. Offline-data budget and fairness

The inherited target is 4,000 transitions per environment cell while preserving complete 25-step episodes.

For each retained method report:

1. actual collected transitions and complete episodes;
2. actual fitting/training transitions and episodes after splitting;
3. holdout transitions and whether they influence model/hyperparameter selection;
4. whether all methods consume exactly the same permitted training episodes;
5. whether any method pools data across population, family, noise, reward mode, or cell;
6. whether reward modes reuse the same fitted dynamics correctly;
7. whether next observations, rewards, or terminal flags differ between methods;
8. per-action transition and episode coverage;
9. whether sequence order is preserved where the method requires it;
10. whether 4,000 means total logged transitions or transitions actually supplied to training;
11. whether the current general methods use more or less data than corrected PLUS/MOOR;
12. whether any replay, synthetic rollout, model-generated transition, or online evaluation observation is counted as additional training data.

Do not claim that 4,000 is sufficient. Report existing learning-curve or adequacy evidence, if any, and identify what remains untested.

## I. Hyperparameter provenance and adequacy

For each method produce a complete hyperparameter table containing:

- hyperparameter name;
- implemented value;
- original-paper value/range;
- source file and line;
- provenance: paper / benchmark adaptation / computational constraint / inherited local default;
- selected before or after any returns were observed;
- sensitivity evidence;
- expected effect if too small or too large;
- runtime effect.

Include, where applicable:

- model architecture, ensemble size, hidden width/depth, activation;
- optimizer, learning rate, batch size, epochs/steps, regularization;
- bootstrap method;
- uncertainty calibration;
- rollout horizon;
- planning horizon;
- sequence count;
- particle count;
- tree depth, simulations, exploration constant, progressive widening;
- belief or posterior particles;
- actor/value architecture;
- support/OOD thresholds;
- k-nearest-neighbour settings;
- safety margins and penalties;
- Delphic ambiguity/confounding-set settings;
- random seeds and replicate count;
- evaluation episodes/horizon.

Answer explicitly:

1. Which values are much smaller than the paper because of runtime constraints?
2. Which defining algorithmic mechanisms may be ineffective at the current budget?
3. Which values were inherited from the old known-`r,K` run without new justification?
4. Which values were changed after seeing any historical results?
5. Which focused, return-blind sensitivities are required before a fair comparison?
6. Are computational budgets reasonably comparable across RefPlan, OGSRL/OSGRL, BA-MCTS, Delphic, and corrected PLUS/MOOR?

Method-appropriate compute need not be numerically identical, but no method should be deliberately crippled by a clearly inadequate setting.

## J. Method-specific questions

### J1. RefPlan

1. What does “reference” mean: reference dynamics, reference policy, reference planner, or local project terminology?
2. Is RefPlan a published algorithm or a locally constructed model-based planner?
3. Does it learn dynamics from the 4,000-transition dataset, and if so what model class and uncertainty representation?
4. Does planning use mean dynamics, ensemble sampling, pessimism, uncertainty penalties, or support constraints?
5. Is it a valid independent baseline or merely the shared model/planner underlying other methods?
6. What paper-facing name is scientifically defensible?

### J2. OGSRL/OSGRL

1. What is the exact expansion and paper?
2. Which components define the original method?
3. Does the current implementation retain its guarded/support/safety mechanism, or is it a low-capacity local actor with heuristics?
4. How are OOD support, nearest neighbours, safety, and uncertainty computed?
5. Does hidden safe mode require information unavailable to the method?
6. Is the current method genuinely the paper algorithm, an adaptation, or an inspired proposal precursor?

### J3. BA-MCTS

1. What is the exact paper/provenance for the implemented BA-MCTS?
2. What latent uncertainty defines the Bayes-adaptive state?
3. Is uncertainty updated along simulated histories or only sampled once at the root?
4. Does the tree branch on observations/beliefs, states, models, or actions only?
5. Are root sampling, posterior updates, UCT, particle filtering, and progressive widening implemented as required by the cited method?
6. If the current planner is only belief-rooted finite-action MCTS over a learned ensemble, is “BA-MCTS” still accurate?
7. Are tree depth and simulation counts large enough for the method to differ meaningfully from RefPlan?

### J4. Delphic

1. What exact Delphic paper and implementation are claimed?
2. What source of hidden confounding or ambiguity exists in this conservation benchmark?
3. How is the Delphic uncertainty/ambiguity set constructed from the offline data?
4. Does the current code implement the paper's identifiability assumptions, world-model ensemble, pessimistic objective, and policy learning?
5. If the benchmark has partial observation but not the paper's confounding structure, what is the scientific interpretation?
6. Is Delphic currently implemented, registered, tested, and runnable, or only planned?
7. Does its compute/data budget match its original design sufficiently to be meaningful?

### J5. MOPO removal

1. Where is MOPO registered or referenced?
2. Does any retained method inherit MOPO-specific model, rollout, uncertainty, or penalty code?
3. Would removing MOPO from manifests/results leave shared training code unchanged?
4. Which tests and documents would need updating later?
5. Can MOPO remain archived but excluded from the retained comparison without affecting reproducibility?

## K. Shared-model concern

Determine whether RefPlan, OGSRL/OSGRL, BA-MCTS, and Delphic all use the same `ContinuousDynamicsEnsemble` or equivalent learned model.

If they do, report:

1. which differences remain between methods after fitting;
2. whether the shared model is faithful to each paper;
3. whether model misspecification could dominate all method comparisons;
4. whether the methods are truly independent baselines or planner/policy variants over one local model;
5. whether the shared model was designed for known `r,K` and later patched for hidden mode;
6. whether each paper requires a different uncertainty, latent, confounding, or model representation;
7. which shared components are appropriate framework reuse versus inappropriate algorithm collapse.

## L. Tests and evidence

For each retained baseline inventory existing tests for:

- paper-defining mechanism;
- hidden-information blocking;
- known/hidden mode routing;
- episode order and split integrity;
- deterministic reproducibility;
- uncertainty calibration;
- planner or policy correctness on a small exactly solvable case;
- OOD/support behaviour;
- safety behaviour without private information;
- artifact provenance and method ID;
- runtime and memory canary;
- data-budget and hyperparameter sensitivity.

Report exact recent test commands/results. Do not run expensive tests in Phase 1. Quick read-only or existing unit-test reruns are allowed only if they do not write experiment artifacts or open returns; state exactly what was executed.

## M. Adaptation register

Create a consolidated table:

| Method | Departure from paper | Classification | Why required | Scientific consequence | Must fix? |
|---|---|---|---|---|---|

Distinguish:

1. necessary conservation-domain adaptations;
2. necessary hidden-demographics/privacy adaptations;
3. partial-observation adaptations;
4. discrete-action adaptations;
5. finite-horizon/episodic adaptations;
6. reward replacement;
7. computational approximations;
8. optional extensions;
9. material replacements or missing defining components.

## N. Severity-ordered findings and recommendations

Classify findings as:

- **Blocking scientific identity**
- **Blocking hidden-mode validity/privacy**
- **Blocking fair comparison**
- **Required verification missing**
- **Hyperparameter/data adequacy risk**
- **Runtime/engineering risk**
- **Documentation/naming only**

For each finding provide:

1. method affected;
2. exact evidence;
3. why it matters;
4. recommended correction;
5. tests required;
6. expected implementation and runtime effect;
7. whether correcting it would require a new method ID or rerun.

## O. Required final verdicts

For each method choose exactly one current verdict:

- **Paper-aligned and sufficiently verified**
- **Paper-aligned but insufficiently verified**
- **Paper-inspired adaptation**
- **Project-specific baseline with defensible provenance**
- **Materially mislabelled/incomplete**
- **Not implemented**

Then answer:

1. Which methods can be retained unchanged?
2. Which require implementation fixes before any new run?
3. Which require only naming/documentation corrections?
4. Which require paper or official-code verification before judgment?
5. Is the proposed retained set of RefPlan, OGSRL/OSGRL, BA-MCTS, and Delphic scientifically coherent?
6. Is any omitted general baseline necessary to avoid a misleading comparison?
7. What is the minimum defensible corrected baseline suite for the next experiment?

End with one recommendation:

- `NOT READY: paper identity or privacy fixes required`
- `NOT READY: verification/hyperparameter evidence required`
- `READY FOR A CORRECTIVE IMPLEMENTATION PLAN`
- `READY FOR LIMITED GENERAL-BASELINE CANARIES`

Then stop. Do not edit files, launch jobs, remove MOPO, or inspect performance returns.

