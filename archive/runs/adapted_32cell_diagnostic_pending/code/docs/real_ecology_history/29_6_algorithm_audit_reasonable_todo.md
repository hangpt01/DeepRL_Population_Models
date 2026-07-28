# To-Do From `AUDIT_algorithm_paper_implementation.md`

Date: 2026-07-05

Scope: this file records audit points I think are reasonable after checking the
current algorithm note and the runnable code under `src/real_ecology_benchmark`.
It is a to-do list, not an implementation patch.

## Priority 0: Fix Leakage-Framing In The Algorithm Note

1. Rewrite the POMDP history in `29_6_algorithm_method_notes.tex`.
   - Current text includes reward in public belief history:
     `h_t=(o_0,a_0,R_0,o_1,...,R_{t-1},o_t)` at
     `docs/29_6_algorithm_method_notes.tex:49-54`.
   - The corrected decision/filter history should be:
     `(o_0,a_0,o_1,...,a_{t-1},o_t)` plus public controls/context.
   - Reward should be described as a logged training/evaluation target, not a
     filter or online policy input.

2. Add a leakage guard test for online filtering/policy updates.
   - Code evidence: the evaluator passes `reward` inside `PublicTransition`
     at `src/real_ecology_benchmark/evaluator.py:107-116`, but then the filter
     update uses only `(belief, action, observation)` at
     `src/real_ecology_benchmark/evaluator.py:139-147`.
   - Code evidence: cached beliefs are built from observations/actions only at
     `src/real_ecology_benchmark/beliefs.py:662-682`.
   - Test idea: for online methods with `observe`, two transitions with the
     same observation/action/public controls but different rewards should leave
     model posterior/action choice identical. Exempt offline value fitting that
     legitimately uses `dataset.rewards` as a Bellman target.

## Priority 1: Make Method Names Honest About Adaptation Level

3. Add an explicit "paper-faithfulness versus benchmark adaptation" paragraph
   near the start of `29_6_algorithm_method_notes.tex`.
   - The current final sentence already says these are not exact paper
     reimplementations at `docs/29_6_algorithm_method_notes.tex:788-793`.
   - Strengthen this earlier and make it visible before the method sections.

4. Rename or soften the most heavily adapted methods in prose/tables.
   - Use wording like `MOPO-style`, `RefPlan-inspired`, `BA-MCTS-inspired online
     Bayes tree search`, `PLUS-style K-only Ricker prior`, and
     `Delphic-inspired random-world CQL`.
   - Current summary table is too generous for BA-MCTS, PLUS, and Delphic-CQL:
     `docs/29_6_algorithm_method_notes.tex:755-770`.

5. For each method section, add a compact "Main deviations from the paper"
   paragraph.
   - MOPO: ridge bootstrap dynamics + MPC instead of neural ensemble rollouts
     and policy optimization.
   - RefPlan: ensemble posterior with residual likelihood instead of learned
     latent epistemic encoder/control-as-inference machinery.
   - BA-MCTS: online tree search only; no paper-style offline search target
     distillation into actor/critic.
   - MOOR: Ricker expert benchmark, not the original catch/effort fishery
     pipeline.
   - PLUS: deliberately reduced 21-point capacity-only prior, not the paper's
     broader `r/K/noise/structure` prior.
   - Delphic-CQL: random-feature latent worlds, not verified compatible
     hidden-confounding worlds.
   - OGSRL: linear actor + KNN support guard, not the full paper-scale
     guardian/critic stack.

## Priority 2: Correct Specific Algorithm Descriptions

6. Fix shared MPC wording.
   - Current text calls the planner a "continuous-action-evaluation engine" at
     `docs/29_6_algorithm_method_notes.tex:197-199`.
   - The benchmark has 11 finite actions, so describe it as a continuous-state,
     particle-belief finite-action sequence sampler.

7. Document the zero-abundance observation rule in RefPlan/BA-MCTS/PLUS.
   - The TeX currently writes `log(o_{t+1}) - log(s_hat)` without a zero rule at
     `docs/29_6_algorithm_method_notes.tex:306-316`.
   - Code uses epsilon floors for RefPlan and BA-MCTS:
     `src/real_ecology_benchmark/methods/refplan.py:86-100` and
     `src/real_ecology_benchmark/methods/bamcts.py:136-147`.
   - Code uses a point-mass-at-zero log-normal observation model at
     `src/real_ecology_benchmark/observation.py:32-55`.
   - Update the equations to show either `log(max(o, eps))` or the point-mass
     zero case.

8. State the theta-logistic rate convention in the algorithm note.
   - Code/spec evidence: Ricker/Allee/regime use `ln(lambda)`, theta uses
     `lambda - 1` at `src/real_ecology_benchmark/realdata.py:14-16` and
     `src/real_ecology_benchmark/config.py:125-131`.
   - Add this next to the real-action/set-point equations.

9. Mark the negative-growth split as a benchmark convention.
   - Current note already gives the split at
     `docs/29_6_algorithm_method_notes.tex:92-118` and mentions sign pathology
     at `docs/29_6_algorithm_method_notes.tex:432-438`.
   - Add a sentence that this is not the standard Ricker/Allee map for negative
     intrinsic rate; it is the benchmark's exploitation-mortality convention.

10. Fix PLUS wording.
    - Current text says "The 21-candidate count is preserved" at
      `docs/29_6_algorithm_method_notes.tex:500-506`.
    - Replace it with: "we use a deliberately reduced 21-point K-only
      PLUS-style prior in the real set-point setting."
    - Code evidence: PLUS uses `candidate_count=21` and spans only
      `K_base..K_max` in real mode at
      `src/real_ecology_benchmark/methods/plus.py:21-37`.

11. Fix Delphic-CQL wording and target equation.
    - Current note calls random-feature worlds "compatible worlds" at
      `docs/29_6_algorithm_method_notes.tex:512-533` and again at
      `docs/29_6_algorithm_method_notes.tex:600-603`.
    - Rephrase as random latent worlds / Delphic-inspired worlds unless the code
      actually trains compatible hidden-confounding world models.
    - The TeX target uses uncertainty for a greedy `a*_t` at
      `docs/29_6_algorithm_method_notes.tex:567-578`. The code builds a greedy
      target action but subtracts the uncertainty column for observed actions at
      `src/real_ecology_benchmark/methods/delphic.py:192-204`.
    - Decide and document one consistent rule. If this is meant to be CQL on
      observed Bellman targets, prefer an action-indexed penalty
      `U(phi_t, a_t)` for the updated action.

12. Clarify BA-MCTS leaf values.
    - Code evidence: terminal/depth leaf returns zero at
      `src/real_ecology_benchmark/methods/bamcts.py:61-63`; recursive value is
      only simulated rollout return at `src/real_ecology_benchmark/methods/bamcts.py:87-104`.
    - Update the TeX so it does not imply a learned critic leaf value.
    - Optional experiment: run a leaf-value ablation (`zero` versus fitted value)
      before claiming BA-MCTS underperformance is algorithmic rather than a leaf
      approximation artifact.

## Priority 3: Algorithm-Code Improvements Worth Considering

13. Expand OGSRL's support guardian feature space.
    - Code evidence: the guardian currently embeds only
      `[log1p(state/500), action/(A-1)]` and hard-codes `500` at
      `src/real_ecology_benchmark/methods/ogsrl.py:32-58` and
      `src/real_ecology_benchmark/methods/ogsrl.py:73-78`.
    - This is weaker than the policy features, which include safety and public
      controls at `src/real_ecology_benchmark/methods/ogsrl.py:105-116`.
    - Update the guardian to use `env_cfg.K_ref`, action, belief-state summary
      or particle-derived public context, `rho`, `kappa/K_ref`, `K_eff/K_ref`,
      and uncertainty/safety features.
    - After changing this, update the OGSRL equations in the TeX.

14. Separate training-time budgets from deployment thresholds in the OGSRL text.
    - Code already has both training budgets and deployment limits at
      `src/real_ecology_benchmark/methods/ogsrl.py:84-103`.
    - The note should explicitly distinguish the discounted training CMDP
      budgets from one-step deployment feasibility gates.

15. Report MOOR filter context clearly.
    - MOOR fits to cached public belief means at
      `src/real_ecology_benchmark/methods/moor.py:61-85`.
    - If result tables include both learned-filter and mechanistic-filter MOOR
      rows, report them separately. If only learned-filter MOOR is used, say the
      Ricker baseline benefits from the shared learned belief front-end.

16. Keep the episode-level holdout split claim.
    - Code evidence: `split_train_holdout` shuffles and splits by episode IDs at
      `src/real_ecology_benchmark/training_monitor.py:80-125`.
    - No fix needed except referencing this in documentation if training
      diagnostics are described.

## Priority 4: Optional Larger Baselines, Not Quick Fixes

17. Consider separate paper-faithful variants only if they become scientific
    goals.
    - BA-MCTS paper-shaped variant: Continuous BAMCP/DPW, search-generated
      targets, actor/critic distillation.
    - Delphic paper-shaped variant: latent compatible generative worlds with
      observational-distribution checks.
    - RefPlan paper-shaped variant: latent encoder/decoder and trajectory
      posterior control-as-inference.
    - PLUS stronger variant: axes over `K`, noise, and structural family.
    - These are new baselines, not small corrections to the current benchmark
      adaptation.

