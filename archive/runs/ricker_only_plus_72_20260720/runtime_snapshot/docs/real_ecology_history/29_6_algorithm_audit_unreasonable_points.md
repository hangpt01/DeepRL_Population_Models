# Points From The Audit I Think Are Overstated Or Not Appropriate To Fold In

Date: 2026-07-05

This file records audit points that are either unreasonable as immediate
algorithm updates, overstated relative to the live code, or mismatched with the
user-requested purpose of the algorithm note.

## U1. Do Not Add A Code-Verification Checklist To The Algorithm Note

The audit asks the TeX to include a code-verification checklist. I do not think
that belongs in `29_6_algorithm_method_notes.tex`.

Reason:

- The user's latest instruction was that the file should not contain audit
  questions. It should explain the algorithms, equations, training/evaluation
  flow, implementation adaptations, and deviations from the papers.
- Code checks are useful, but they should live in a separate test/audit planning
  file, not in the paper-facing algorithm explanation.

Recommended handling:

- Keep this todo file and the test suite as the place for verification tasks.
- Keep `29_6_algorithm_method_notes.tex` explanatory.

## U2. H1 Is A Real Documentation Bug, But It Is Not Evidence Of A Current Filter Leak

The audit is right that the TeX history currently includes rewards and that this
would be a leakage problem if used by the filter or policy. But as a claim about
the current code, it is overstated.

Code evidence:

- The evaluator passes reward inside `PublicTransition` at
  `src/real_ecology_benchmark/evaluator.py:107-116`.
- The actual filter update uses only action and next observation at
  `src/real_ecology_benchmark/evaluator.py:139-147`.
- Offline belief caching also uses observations/actions only at
  `src/real_ecology_benchmark/beliefs.py:662-682`.
- The default policy `observe` is a no-op at
  `src/real_ecology_benchmark/methods/base.py:61-67`.
- The online posterior-update methods inspected here use `result.observation`,
  not `result.reward`: RefPlan at
  `src/real_ecology_benchmark/methods/refplan.py:86-100`, BA-MCTS at
  `src/real_ecology_benchmark/methods/bamcts.py:136-147`, and PLUS at
  `src/real_ecology_benchmark/methods/plus.py:114-162`.

Reasonable action:

- Fix the TeX history and add a regression test.

Unreasonable framing:

- Treating this as already-proven reward leakage in the live filter/policy path.
  Delphic-CQL uses `dataset.rewards` in Bellman targets, but that is an offline
  value-learning target, not a belief update from hidden state.

## U3. Full Paper-Faithful Reimplementations Are Not A Reasonable "Fix" To The Current Benchmark

The audit correctly notes that BA-MCTS, Delphic-CQL, RefPlan, MOPO, and PLUS are
heavily adapted. However, implementing the original papers' full mechanisms is a
new-baseline project, not a small correction.

Examples:

- BA-MCTS paper-shaped actor/critic distillation and Continuous BAMCP/DPW would
  replace the current online tree-search baseline.
- Delphic compatible hidden-confounding generative worlds would replace the
  current random-latent-world linear CQL baseline.
- RefPlan's latent epistemic encoder/decoder/control-as-inference stack would
  replace the current ensemble-posterior planner.
- MOPO neural synthetic-rollout policy optimization would replace the current
  ridge-ensemble MPC baseline.

Reasonable action:

- Rename/framing: call them `style`, `inspired`, or `benchmark adaptation`
  variants.

Unreasonable action without a new experimental decision:

- Requiring full paper-faithful implementations before continuing with this
  adapted benchmark.

## U4. Reward Mode Should Not Be Required In The OGSRL Support Guardian

The audit suggests the guardian may need reward mode or safety context. I agree
with adding safety/public-control/belief context, but not with making
`reward_mode` a support feature by default.

Reason:

- `reward_mode` changes the scalar objective/penalty, not the transition support
  or offline behavior distribution.
- The same collected transition dataset can be valid for both `yield` and
  `safe`; the agent is trained separately per mode, but state-action support is
  still a property of public state/action coverage.

Reasonable action:

- Expand the guardian beyond `[state, action]` to include public belief/context,
  uncertainty, safety, and cumulative controls.

Unreasonable action:

- Treating `reward_mode` itself as necessary for state-action support unless a
  future dataset collection policy differs by reward mode.

## U5. PLUS Does Not Need To Add All Paper Axes Unless We Want A New Stronger Baseline

The audit is right that the wording "21-candidate count is preserved" is
misleading. The current code uses a 21-point capacity-only candidate set in the
real set-point setting:

- `src/real_ecology_benchmark/methods/plus.py:21-37`.

Reasonable action:

- Document it as a deliberately reduced K-only PLUS-style prior and an
  intentionally misspecified mechanistic baseline.

Unreasonable action as a required fix:

- Forcing PLUS to include the paper's broader `r/K/noise/structure` candidate
  axes before any experiment. That would change the baseline's scope, runtime,
  and scientific interpretation.

## U6. "MOPO Proper" And "RefPlan Proper" Should Be Framing Notes, Not Blocking Bugs

The audit's MOPO and RefPlan comments are fair as paper-comparison notes. They
are not evidence that the current algorithms are internally broken.

Current code intentionally uses:

- transparent ridge dynamics at `src/real_ecology_benchmark/dynamics.py:17-58`;
- bootstrap linear ensembles at `src/real_ecology_benchmark/dynamics.py:92-127`;
- shared particle MPC for several methods, described in the note at
  `docs/29_6_algorithm_method_notes.tex:168-199`.

Reasonable action:

- Make the names and limitations explicit.

Unreasonable action:

- Treating the absence of neural policies or original-paper architecture as a
  correctness failure for this audit-friendly ecological benchmark.

