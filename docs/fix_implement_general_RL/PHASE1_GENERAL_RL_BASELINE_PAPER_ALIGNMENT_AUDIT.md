# Phase 1 Audit — General-RL Baselines Under Hidden Demographics

**Scope:** inspection-only evidence report answering
`docs/fix_implement_general_RL/CODE_SERVER_GENERAL_RL_BASELINE_PAPER_ALIGNMENT_AUDIT.md`.
No code/docs edited, no jobs launched or altered, MOPO not removed, no performance
returns (`operational_return`, `true_return`, survival return, rankings, comparative
summaries) inspected, no hyperparameters tuned.

**Retained set under audit:** `refplan`, `ogsrl`, `bamcts`, `delphic` (general-RL);
`mopo` inspected for removal impact only.

**One-line bottom line:** the implementations are honestly documented linear/mechanistic
*re-implementations* of the four papers' *ideas* (already labelled "-inspired"), privacy
blocking is real and tested, but (a) the hidden-r,K fixes are **uncommitted and mixed with
unrelated work**, (b) in the hidden mode that is actually run, **RefPlan collapses onto MOPO**
and **OGSRL trains no policy**, and (c) **Delphic is not in the hidden manifest**. See
Section N/O.

---

## A. Repository, version, and job state

**A.1 Active repo / branch / commit / runtime / tree state**

| Item | Value |
|---|---|
| Working dir | `/fs04/scratch2/ce25/Claude_DeepRL_Population_Models` (also reachable as `/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models`) |
| Branch | `main` |
| HEAD commit | `5f9cf32a69d47e2d31b2ed6ee30ebbcfe84b8536` — *"Add the known r,K model - Before hiding them"* |
| `git describe --always --dirty` | `5f9cf32-dirty` |
| Runtime | Python 3.10.14, NumPy 2.2.6, PyTorch 2.13.0+cpu (`src/` general methods use **no** torch — see K) |
| Package layout | `pyproject.toml` → `package-dir = {"" = "src"}`; tests need `PYTHONPATH=src` |
| Tree state | dirty: ~50 tracked files modified, ~20 untracked (methods, configs, `real_ecology_runs/…`) |

**A.2 Snapshot that produced the known-`r,K` general baselines.**
The known-`r,K` general baselines are the **committed HEAD** (`5f9cf32`). The method files
themselves first entered git history at `e95b000` *"Promote ecology benchmark to repo root"*
(a path move); their known-`r,K` form is frozen at `5f9cf32`. Proof: `git show
HEAD:src/real_ecology_benchmark/methods/ogsrl.py` contains **0** references to
`self.hidden` / `PublicDynamicsEnsemble` / `PublicKNNGuardian`; the working tree has 6.

**A.3 Snapshot/working-tree changes that produced the hidden-`r,K` fixes.**
The hidden-`r,K` fixes are **entirely uncommitted working-tree changes** on top of `5f9cf32`:

| File | Change | Nature |
|---|---|---|
| `methods/bamcts.py` | `+97` lines | hidden branch added |
| `methods/ogsrl.py` | `+148` lines | hidden branch added |
| `methods/refplan.py` | `+38` lines | hidden branch added |
| `methods/mopo.py` | `+18` lines | hidden branch added |
| `methods/delphic.py` | `+8` lines | hidden branch added |
| `public_models.py` | **new untracked** | `PublicDynamicsEnsemble`, `PublicParticlePlanner` |
| `public_surrogate.py` | **new untracked** | `PublicRewardRiskSurrogate` (shared reward/risk) |
| `privacy.py` | **new untracked** | forbidden-name leakage guard |

**A.4 Committed vs frozen vs mixed.** The corrected hidden general-baseline changes are
**not committed** and **not isolated** — they are interleaved in the same dirty tree as
unrelated ecology-baseline work (PLUS/MOOR "faithful/adapted" files, ~15 configs, ~10
scripts, `docs/…`). They *are* **frozen-by-copy** inside the run directory
`real_ecology_runs/hidden_rk_comparison_20260716/code/` (that snapshot contains the hidden
branches and all three new modules), and the run's parameters are registered in
`real_ecology_runs/hidden_rk_comparison_20260716/manifests/registration.json`. There is **no
commit or content digest** binding the live tree to that snapshot; provenance rests on the
copied directory only.

**A.5 Job state.** `squeue -u $USER`: the only active/pending jobs are the **ecology**
adapted-PBVI diagnostic (`adapt32-plus-fit` ×32 running as array `58391584`;
`adapt32-plus-plan`, `adapt32-accept-v2` pending as `58391586`, `58391658`). **No general-RL
job is active, pending, or queued.** Prior general-RL runs are all completed on disk (below).

**A.6 Result directories (labels only — returns not opened).**

| Directory | Label |
|---|---|
| `real_ecology_runs/general_audit_20260712/` | **known-`r,K` (full)** provisional — search-budget audit; manifests are `expose_rk=full`, `filter=learned`, large budgets (`h20_seq256`). Superseded for the hidden question. |
| `real_ecology_runs/hidden_rk_comparison_20260716/` | **current hidden-`r,K`** comparison (both `full` + `hidden` arms) with frozen `code/` snapshot + `registration.json`. Methods = `bamcts, ogsrl, refplan, moor_native, plus_native`. |
| `real_ecology_runs/motivation_native_20260711/` | native motivation run (ecology natives). |
| `real_ecology_runs/adapted_*`, `paper_faithful_*`, `psafe_overnight_*` | **ecology** PLUS/MOOR faithful/adapted + P_safe grid — unrelated to general-RL. |

**A.7 Artifacts inspectable without opening returns:** manifests/routing CSVs, `registration.json`,
frozen `code/` snapshot, dataset/surrogate `.npz` schemas (field names only), config YAMLs,
`fit_diagnostics` structure, and the docs bundle. **Not** inspected: `results*.jsonl`/aggregate
reward fields.

---

## B. Canonical method and paper identity

**OGSRL vs OSGRL — resolved.** Canonical spelling is **`OGSRL`** = **O**ffline **G**uarded
**S**afe **RL**. `OSGRL` appears **only inside the audit-request document itself** (6 hits, all
in `CODE_SERVER_GENERAL_RL_BASELINE_PAPER_ALIGNMENT_AUDIT.md`); it appears **nowhere** in code,
configs, other docs, manifests, or artifacts (`OGSRL/ogsrl`: 93 hits). Registered method ID is
`ogsrl`. **Recommendation: adopt `OGSRL`; the only inconsistent occurrences are in the request
doc and require no code change.**

| # | RefPlan | OGSRL | BA-MCTS | Delphic |
|---|---|---|---|---|
| 1 canonical name/acronym | Reflect-then-Plan (RefPlan) | Offline Guarded Safe RL (OGSRL) | Bayes-Adaptive MCTS (BA-MCTS) | Delphic offline RL |
| 2 registered ID | `refplan` (`RefPlanPolicy`) | `ogsrl` (`OGSRLPolicy`) | `bamcts` (`BAMCTSPolicy`) | `delphic` (`DelphicCQLPolicy`) |
| 3 paper (verified from PDF p.1) | *Reflect-then-Plan: Offline Model-Based Planning through a Doubly Bayesian Lens*, Jeong, Wang, Wang, Sanner, Poupart — **ICML 2025** | *Offline Guarded Safe RL for Medical Treatment Optimization Strategies*, Yan, Shen, Wachi, Gros, Zhao, Hu — **NeurIPS 2025 Spotlight** | *Bayes Adaptive Monte Carlo Tree Search for Offline Model-Based RL*, Chen, Xu, Chen, Schneider — **ICLR 2026** | *Delphic Offline RL under Nonidentifiable Hidden Confounding*, Pace, Yèche, Schölkopf, Rätsch, Tennenholtz — **ICLR 2024** |
| 4 DOI/arXiv recorded | none in repo (PDF only) | none (PDF only) | none (PDF only) | none (PDF only) |
| 5 repo holds | **PDF** at `docs/references/general_model_based_offline_RL_papers/ICML25_Reflect-then-Plan-…pdf` | **PDF** at `docs/references/discrete_action_continuous_obser_state/NeurIPS25_Spotlight_Offline Guarded Safe…pdf` | **PDF** at `docs/references/general_model_based_offline_RL_papers/ICLR26_Bayes Adaptive…pdf` | **PDF** at `docs/references/discrete_action_continuous_obser_state/ICLR24_Delphic…pdf` |
| 6 official code claimed | no | no | no | no |
| 7 files/registry/config/runner/tests | `methods/refplan.py`; registry `__init__.py:5,18`; configs `hidden_rk*.yaml`, planner block; runner `scripts/make_hidden_rk_manifest.py`; tests `tests/synthetic/test_methods.py`, `tests/real/test_hidden_rk.py` | `methods/ogsrl.py`; registry `__init__.py:12,25`; same configs/runner; tests same | `methods/bamcts.py`; registry `__init__.py:6,19`; same; tests same | `methods/delphic.py`; registry `__init__.py:11,24`; **not** in `make_hidden_rk_manifest.py` routing; tests `test_methods.py` (+ hidden-fit loop in `test_hidden_rk.py`) |
| 8 provenance class | **paper-inspired** (idea-level) | **paper-inspired** (idea-level) | **paper-inspired** (idea-level) | **paper-inspired** (idea-level) |

No name lacks a matching paper — every mapping is 1:1 and verified against the PDF first pages.
The repository already declares all four as "-inspired" with explicit valid/invalid claim
boundaries in `docs/benchmark/04_algorithm_adaptations_and_claims.tex:27–32, 83–264`.

---

## C. Original-paper algorithm summaries (from the PDFs / abstracts; independent of the code)

Facts below marked **[P]** are read from the paper PDF (first pages/abstract); **[I]** are
inferred from local docs. Deep-method internals beyond the abstract are **[P-abstract]**
(claimed by the paper but not line-verified here in Phase 1).

**RefPlan (ICML25) [P].** Offline, model-based, (typically) fully-observed continuous control.
A *doubly Bayesian* planner: recasts planning as Bayesian posterior estimation and, **at
deployment**, updates a belief over environment dynamics from real-time observations, folding
that uncertainty into model-based planning **via marginalization**. Sits on top of a base
conservative offline policy. Defining components: (i) deployment-time posterior over dynamics;
(ii) marginalized (posterior-weighted) planning; (iii) resilience to changing dynamics.
Replaceable: exact base policy, network architecture, planner internals.

**OGSRL (NeurIPS25 Spotlight) [P].** Offline, **model-based**, POMDP-flavoured (uses "the full
patient state history"). Two defining constraints: (1) an **OOD guardian** that specifies
clinically-validated in-distribution regions and restricts optimization to them; (2) a
**safety-cost constraint** encoding domain safety boundaries. Theoretically grounded; improves
on CQL's action-only regularization by also regulating *downstream state trajectories*.
Defining: learned dynamics + OOD-support guardian + safety-cost constraint + constrained
policy optimization (dual). Replaceable: network class, support estimator family.

**BA-MCTS (ICLR26) [P].** Offline model-based RL cast as a **Bayes-Adaptive MDP (BAMDP)** to
handle "various MDPs that behave identically on the offline dataset." A **Bayes-adaptive MCTS**
planner for continuous state/action, stochastic transitions, used as a **policy-improvement
operator inside policy iteration** ("RL+Search", AlphaZero-style). Defining: (i) posterior over
which MDP is true, **updated along simulated histories**; (ii) MCTS with continuous
progressive widening; (iii) policy/value learning in the outer PI loop. Replaceable: exact
network sizes, D4RL/tokamak domain detail.

**Delphic (ICLR24) [P].** Offline RL under **nonidentifiable hidden confounding**. Introduces
**delphic uncertainty** = variation of value over the set of **world models compatible with the
observed data** (distinct from epistemic/aleatoric). Builds a **pessimistic** offline algorithm
that penalizes delphic uncertainty; does **not** assume confounder identifiability. Defining:
(i) confounded (PO)MDP with an unobserved variable driving both action and outcome; (ii)
compatible-world set; (iii) delphic-uncertainty penalty in value learning. Replaceable: the
world-model class (paper uses richer/variational world models).

**MOPO (2020, for removal context) [P-abstract/I].** Offline model-based RL with a probabilistic
dynamics **ensemble** and a **reward penalty proportional to model uncertainty**; policy
optimized on penalized model rollouts.

---

## D. Current implementation flow (sanitized data → action). File/line evidence.

Common substrate for all four (hidden mode): dataset → `PublicObservationFilter`
(`beliefs.py:586`) → `PublicBeliefCache` (`beliefs.py:820`, features via
`types.py:BeliefState.public_features:121`) → method `fit` → method `act`. The public filter is
**memoryless**: `update()` ignores the action and prior belief and just re-jitters the current
observation lognormally (`beliefs.py:635–641`, particles at `595–604`); "history" is only the
previous observation carried in `contexts[:,0]`.

**RefPlan (`methods/refplan.py`).** *Hidden:* `fit` builds `PublicDynamicsEnsemble`
(`refplan.py:42`), sets a uniform `posterior` over members, and a `PublicParticlePlanner`
(`:51`). `act` (`:73`) delegates to `self.planner.plan(belief, self.dynamics,
pessimism=…)` — **the posterior is used only for a diagnostic `model_entropy`, never to weight
planning**. `observe` (`:113`) does a lognormal-likelihood Bayesian update of `posterior`, but
because hidden `act` ignores `posterior`, that "reflect" update **does not affect actions in
hidden mode**. *Full:* `act` (`:85`) samples ≤`max_planning_members=5` members by posterior,
scores shared MPC sequences per member, and returns `mean − pessimism·std` ("reflected"):
here the posterior **is** used. Model = `ContinuousDynamicsEnsemble`; reward = `build_reward`;
planner = `ParticleMPC`.

**OGSRL (`methods/ogsrl.py`).** *Hidden:* `fit` (`:364`) builds `PublicDynamicsEnsemble` +
`PublicKNNGuardian` (`:372`) + attaches the shared surrogate, then **`return diagnostics` at
`:388` — no actor, no critics, no dual variables are trained**. `act` (`:517`) does a **one-step**
constrained greedy: for each action, surrogate reward − `pessimism·mean(sqrt(var))`, feasibility
= `risk ≤ deployment_safety_limit(0.02)` AND `ood ≤ deployment_ood_limit(0.05)`; if none feasible,
least-violating fallback (`:552–560`). *Full:* trains a **linear softmax actor via policy
gradient over `_rollouts`** with Lagrangian `lambda_safety`/`lambda_ood` dual ascent
(`:410–447`) + linear reward/safety/OOD critics (`:457–472`); deployment applies the feasibility
mask over posterior particles (`_belief_action_risks:482`). KNN guardian: standardized
k=5 NN distance vs an α=0.05 anchor-quantile threshold (`KNNGuardian.fit:80`).

**BA-MCTS (`methods/bamcts.py`).** *Hidden:* `fit` (`:50`) builds `PublicDynamicsEnsemble` +
uniform `posterior`. `act` (`:192`) samples `simulations(128)` belief particles + `simulations`
member indices from `posterior`; each `_simulate_public` (`:94`) rollout **fixes one member for
the whole depth-`5` rollout**, buckets the observation (`_public_key:84`), UCB action selection
(`:106`), reward from surrogate, value −= `pessimism·disagreement`. `observe` (`:251`) updates
`posterior` by lognormal likelihood. *Full:* `_simulate` (`:139`) buckets **true state** finer
near `safety_threshold`, uses `env` reward + control fields. **In neither mode is the model
posterior updated *inside* a simulated trajectory** — it is sampled once at the tree root.

**Delphic (`methods/delphic.py`).** `fit` (`:182`) builds `world_count=10` `CompatibleWorld`s;
each world = a random projection of belief features, **scaled by `_posterior_ambiguity`**
(`:28` — posterior log-state std / quantile spread; collapses to 0 when `sigma_obs=0`), a
softmax behavior head and a ridge Q head (`_fit_world:102`). Delphic uncertainty = **variance of
counterfactual Q across worlds** (`_uncertainty:157`). Final Q = 35 iterations of a **CQL-style
ridge** fitted-Q with conservative pseudo-targets (`cql_alpha=0.5`) minus `delphic_lambda(0.1)·
uncertainty` (`:200–212`). `act` (`:244`): hidden uses `public_features`, full uses
`features(K_ref, safety_threshold)`; greedy on `Q − delphic_lambda·uncertainty`.

**Shared implementations reached:** `ContinuousDynamicsEnsemble` / `PublicDynamicsEnsemble`
(dynamics), `ParticleMPC` / `PublicParticlePlanner` (planning), `PublicRewardRiskSurrogate`
(reward+risk for hidden bamcts/ogsrl/refplan/mopo), `PublicObservationFilter` (belief),
`build_reward` (full). **No fallback/oracle/table-lookup/native-solver path is reachable from
the four general method IDs** — verified by `tests/real/test_hidden_rk.py:114`
(patches `pops_for`/`effects_for`/`actions_for`/`resolve_actions`/`NativeSolver.build` to raise;
all methods still fit).

---

## E. Paper-to-code fidelity matrices

Classifications: **FR** Faithfully retained · **BA** Necessary benchmark adaptation ·
**CA** Disclosed computational approximation · **OE** Optional extension · **MR** Material
replacement · **MI** Missing · **NA** Not applicable · **UV** Unverified.

### RefPlan
| Paper component | Implementation | Evidence | Class | Consequence |
|---|---|---|---|---|
| Deployment-time posterior over dynamics | ensemble posterior via lognormal likelihood in `observe` | `refplan.py:113–134` | FR | present in both modes |
| Marginalized planning under posterior | full: posterior-weighted member scoring | `refplan.py:85–111` | FR (full) | defining mechanism active |
| …same, **hidden mode** | **posterior ignored; plain `PublicParticlePlanner.plan`** | `refplan.py:73–84` | **MR (hidden)** | **RefPlan ≡ MOPO in hidden mode** |
| Base conservative offline policy | none (planning-only) | whole file | MI | no policy layer; planner only |
| Doubly-Bayesian latent dynamics | ridge-linear ensemble | `dynamics.py`/`public_models.py` | MR | model-class change (disclosed) |
| Continuous MB planner | finite-action particle MPC | `planning.py` | BA | discrete mgmt actions |

### OGSRL
| Paper component | Implementation | Evidence | Class | Consequence |
|---|---|---|---|---|
| OOD guardian (support region) | kNN(5) distance vs α-quantile threshold | `ogsrl.py:24–138` | CA | reasonable support proxy |
| Safety-cost constraint | surrogate risk / unsafe-occupancy ≤ budget | `ogsrl.py:549`, `290–299` | BA | domain-mapped |
| Constrained policy optimization (dual) | full: linear softmax actor + λ dual ascent | `ogsrl.py:410–447` | CA | present in full |
| …**hidden mode** | **no actor trained; one-step constrained greedy** | `ogsrl.py:388` (early return), `517` | **MR (hidden)** | **hidden OGSRL is not a learned guarded policy** |
| Model-based, uses full state history | learned linear dynamics + **memoryless** belief | `public_models.py`, `beliefs.py:586` | MR | "history" = previous obs only |
| Theoretical safety guarantee | none transferred | — | MI (disclosed) | doc invalid-claim boundary states this |

### BA-MCTS
| Paper component | Implementation | Evidence | Class | Consequence |
|---|---|---|---|---|
| BAMDP / belief over MDP | ensemble posterior sampled at root | `bamcts.py:194–197` | CA | coarse posterior |
| Posterior updated **along simulated history** | **not updated in-tree; one member per rollout** | `bamcts.py:94–137,139–189` | **MI** | not Bayes-*adaptive* inside search |
| MCTS + progressive widening | UCB over finite actions + continuous state bucketing | `bamcts.py:106,151`,`_key:74` | BA | discrete actions ⇒ no widening needed |
| Outer policy-iteration / value net | none (decision-time search only) | whole file | MI | no learned policy/value |
| Continuous stochastic transitions | ridge-linear stochastic member | `dynamics.py:75` | MR | model-class change (disclosed) |

### Delphic
| Paper component | Implementation | Evidence | Class | Consequence |
|---|---|---|---|---|
| Delphic uncertainty = value var over compatible worlds | variance of counterfactual Q across 10 worlds | `delphic.py:152–159` | FR (idea) | core mechanism present |
| Compatible-world set (same obs-distribution) | random-projection latents scaled by posterior ambiguity | `delphic.py:102–133,28–39` | MR | not proven observationally-equivalent (doc admits) |
| Pessimistic value learning | CQL-style ridge fitted-Q − λ·uncertainty | `delphic.py:200–212` | CA | linear head |
| Hidden-confounding structure | benchmark has partial-obs, **no explicit confounder** | see J4.2 | UV/BA | interpretation caveat |
| World-model ensemble (neural/variational) | ridge/linear | `delphic.py` | MR | model-class change (disclosed) |

**Defensibility of names:** every method has ≥1 **MR/MI** on a *defining* component, so
**paper-faithful is not defensible for any of the four**. The repo's existing
**"-inspired"** wording is correct for RefPlan, BA-MCTS, Delphic. For OGSRL the guarded/safe
*structure* is retained in **full** mode, so "OGSRL-inspired" holds there; in **hidden** mode the
policy-learning core is absent, so hidden OGSRL is closer to **project-specific constrained
greedy** than "OGSRL-inspired" until an actor is trained. Recommended wording overall:
**paper-inspired** (all four), with a hidden-mode asterisk on RefPlan and OGSRL.

---

## F. Known-`r,K` → hidden-`r,K` change audit

**F.1 What the known-`r,K` (full) baselines received.** Directly/indirectly: `K_ref = K_base`
(true carrying capacity; `config.py:332`), `safety_threshold = safety_fraction·K_base`
(`config.py:337`), public cumulative controls `rho/kappa/K_eff` derived from the action table &
`K_base` (`controls.py`, `dynamics._design:40–57`), the **exact** reward via `build_reward(env_cfg)`,
and a mechanistic **learned/reference** particle filter that knows the env. Belief `features()`
are normalized by true `K_ref` and include `unsafe`/`regime` mass (`types.py:85–119`).

**F.2 Code paths that depended on true `r/K`/family/effects.** `ContinuousDynamicsEnsemble`
(K_ref normalization + control design), `ParticleMPC` (`env_cfg.safety_threshold`, `K_ref`,
`build_reward`), `KNNGuardian`/`_state_features` (K_ref, safety_threshold), BA-MCTS `_key`
(K_ref, safety_threshold, rho/kappa buckets), Delphic full `features(K_ref, safety_threshold)`,
RefPlan/BA-MCTS `observe` obs-variance from `env_cfg.observation_noise_sigma`.

**F.3 Exact hidden changes.** For each method a `self.hidden` branch (set in `base.py:26`
by passing a `MethodContext` instead of `EnvironmentConfig`) swaps in:
`PublicDynamicsEnsemble` (observation_scale = **data-median** of positive obs,
`public_surrogate.py:66`; no controls), `PublicObservationFilter`, `PublicRewardRiskSurrogate`
(reward = ridge on public features, risk = logistic on `terminated`), `PublicKNNGuardian`
(observation features), opaque `pop_id`, and `public_features` (no K_ref/threshold/regime).

**F.4 Were hidden quantities removed / estimated / retained?** Removed from the method surface:
`r`, `K`, family, action effects, latent state, regime, private safety threshold. **Replaced by
public proxies**: `observation_scale` (median obs) for K_ref; surrogate reward for exact reward;
surrogate risk (public `terminated`) for the private safety objective. No accidental retention
found (Section G). The safety *threshold* itself is **not** exposed in hidden mode — hidden
safety is expressed only through the surrogate risk channel.

**F.5 Did the hidden fix change only information access?** **No.** It also changed **model class
normalization** (K_ref→observation_scale, controls dropped), **belief filter** (mechanistic
particle filter → memoryless observation jitter), **reward/risk** (exact → fitted surrogate),
and for **OGSRL** the **training architecture** (trained Lagrangian actor → untrained one-step
greedy) and for **RefPlan** the **planning objective** (posterior-weighted → posterior-ignored).

**F.6 Clean information ablation?** **No.** Known vs hidden are **two different implementations**,
not a pure information toggle. This is partly *necessary* (a K_ref-normalized model is impossible
when K is hidden) and therefore defensible — but it means "full vs hidden" deltas conflate
information loss with architecture change, and must not be reported as a clean ablation. (The one
genuinely clean piece: `test_full_and_hidden_legacy_arrays_are_identical`,
`test_hidden_rk.py:100`, shows the *underlying logged arrays* are identical across modes.)

**F.7 Old known-parameter lookups reachable from hidden IDs?** **No** — proven by
`test_hidden_rk.py:114` (private tables/solver patched to raise; hidden fits succeed) and by
`base.py:31–34` (a `hidden` env that still hides r,K cannot construct a full-mode policy).

**F.8 Indirect leakage via identity/paths/ordering?** `pop_id` is a salted SHA-256 token
(`opaque_population_id`, `config.py:218`) with no species string; hidden manifests use it.
Residual channels (filenames, array position, manifest ordering, config family label) are
discussed in G.

---

## G. Privacy and leakage audit

Mechanism: `privacy.py` provides `forbidden_method_name` + `forbidden_paths` +
`assert_hidden_method_artifact` — a **name-based** recursive guard over a 60-name blocklist
(`r`, `K`, `rho`, `kappa`, `safety_threshold`, `family`, `population`, `action_effects`, …). It
inspects **field/attribute names**, not numeric array payloads (`privacy.py:88` returns early on
`np.ndarray`).

| Private quantity | Blocked in hidden mode? | Evidence |
|---|---|---|
| true `r`, `K` / analogues | **Yes** — methods see `observation_scale` (median obs), never `K_ref`/`K_base` | `pipeline.py:230`, `public_surrogate.py:66` |
| true family | **Yes** — relabel-invariance test | `test_hidden_rk.py:175` (native); general methods never read `kind` in hidden branch |
| action-effect magnitudes | **Yes** — `resolve_actions`/`effects_for` patched-to-raise test passes | `test_hidden_rk.py:114` |
| latent true abundance/state | **Yes** — belief particles are observation-space jitter | `beliefs.py:595–604` |
| regime state / transition matrix | **Yes** — `public_features` omits regime; hidden regimes set to 0 | `types.py:121`, `beliefs.py:624` |
| private safety threshold/objective | **Yes** — not exposed; only surrogate risk on public `terminated` | `ogsrl.py` hidden branch; `public_surrogate.py:304` |
| evaluator-only reward components | **Yes** — surrogate is fit from public rewards; evaluator strata computed separately | `public_surrogate.py:392` (`evaluator_only_…`) |
| future obs / next states not in a transition | **Yes** — only `next_observations` within logged transitions used | `dataset.py` schema |
| eval seeds / eval trajectories | **Yes** — eval seeds live in config eval block, not method inputs | `configs/hidden_rk.yaml` |
| result summaries / returns | **Yes** — not passed to methods | n/a |
| identity leakage via names/positions/caches | **Mostly** — opaque `pop_id`; serialized caches/surrogate name-checked | `test_hidden_rk.py:255` |

**Existing privacy tests (all pass — see L):** `test_hidden_schema_has_cost_identity_events_and_no_controls`,
`test_full_and_hidden_legacy_arrays_are_identical`,
`test_hidden_environment_config_cannot_construct_policy_directly`,
`test_all_hidden_methods_fit_without_table_or_exact_native_access` (forbidden-import + artifact
name-scan over **all** registered methods incl. delphic/mopo),
`test_private_family_relabel_does_not_change_hidden_native_fit` (native only),
`test_private_safety_diagnostics_cannot_change_surrogate`,
`test_serialized_hidden_caches_and_surrogate_have_no_private_names`.

**Missing / weak tests (leakage):**
1. **Relabel-invariance for the general methods** — the relabel test covers only `moor_native`;
   there is **no** test asserting `refplan/bamcts/ogsrl/delphic` hidden fits are byte-identical
   under a private family relabel.
2. **Payload (value) leakage** — `privacy.py` checks names, not values; nothing asserts that no
   method-facing array *equals* `K_base`/`safety_threshold` numerically. `observation_scale` is a
   median (safe), but this is untested as a guarantee.
3. **Manifest/path leakage** — no automated check that hidden output paths and manifest ordering
   don't encode species order.
4. **Delphic hidden** is exercised by the generic hidden-fit loop but has **no dedicated
   hidden-mode privacy/relabel test**.

---

## H. Offline-data budget and fairness

**Target:** `DatasetConfig.transitions = 4000`, `episode_length = 25`
(`configs/hidden_rk.yaml`). Collection preserves **complete** episodes: `collect_dataset`
(`collector.py:234`) loops whole episodes until `len ≥ target` and asserts overshoot
`< episode_length` (`collector.py:347`) ⇒ 160 complete 25-step episodes (occasionally 161 →
4025; cf. known "4005≠4000" overshoot note).

| Q | Finding |
|---|---|
| 1 collected transitions/episodes | ~4000 / 160 complete episodes (≤1-episode overshoot) |
| 2 fitting transitions/episodes after split | episode-disjoint 80/20 (`training_monitor.py:88`, `holdout_fraction=0.2`) ⇒ ~128 train / ~32 holdout episodes (~3200 / ~800 transitions) |
| 3 does holdout drive selection? | **No** — holdout only feeds `log_training(…, "holdout", …)` curves; every fit loop runs a **fixed** iteration count (no early stop / model select) |
| 4 same permitted episodes for all methods? | **Yes** — all shared methods receive the same `train_dataset`+`train_cache` (`pipeline.py:369`, `build_method`) built from the same filter+seed |
| 5 any cross-cell pooling? | **No** — one opaque `pop_id` per cell enforced (`pipeline.py:221–223`); surrogate/model fit per cell |
| 6 reward modes reuse fitted dynamics? | dynamics is reward-mode-independent; reward differs via surrogate/`build_reward` — consistent |
| 7 next-obs/reward/terminal differ by method? | **No** — identical logged arrays; `test_full_and_hidden_legacy_arrays_are_identical` |
| 8 per-action coverage | not separately reported per method (untested) |
| 9 sequence order preserved where needed? | episode order preserved in dataset; **but** hidden belief filter is memoryless so sequence order is largely unused at inference (D) |
| 10 does 4000 = logged or supplied-to-training? | **logged**; supplied-to-fit is the ~80% train split |
| 11 more/less data than corrected PLUS/MOOR? | **same** target/split — PLUS/MOOR (native + faithful) share the 4000/25 + 0.2 holdout pipeline |
| 12 replay/synthetic/model rollouts counted as data? | **No** — OGSRL/BA-MCTS/RefPlan model rollouts are planning-time only, never added to the fit set |

**Adequacy is untested and not claimed.** No return-blind learning-curve/data-adequacy evidence
for the general methods was found (holdout curves exist but were not opened, and would show
surrogate loss, not task return). Whether ~3200 transitions suffice for a 10-world Delphic head,
a 128-sim tree, or a Lagrangian actor is **unknown**.

---

## I. Hyperparameter provenance and adequacy

Sources: `config.py` `ModelConfig` (`:511`), `PlannerConfig` (`:521`); method `__init__`
defaults; `configs/hidden_rk.yaml`; `registration.json`. **All values were set before the
hidden run and reproduce the pre-existing literals** (`config.py:528–536` documents that the
per-method budgets were lifted verbatim from in-method literals, so surfacing them "changes no
result"). No evidence any value was changed *after* seeing hidden returns (returns not opened);
provenance is **inherited local default / benchmark adaptation**, **not paper-derived**.

| HP | Value | Paper value | Source | Provenance | If too small | Runtime |
|---|---|---|---|---|---|---|
| ensemble_size | 5 | — (paper uses NN ensembles) | `config.py:513`, `hidden_rk.yaml` | local default | weak posterior/disagreement | low |
| ridge | 1e-3 | n/a | `config.py:514` | local default | — | low |
| planner horizon | 5 | task-dependent | `config.py:523` | local default | myopic MPC | ↑ w/ horizon |
| sequences | 96 | n/a (CEM/grad in papers) | `config.py:524` | local default | poor action coverage | ↑ |
| particles | 32 | n/a | `config.py:525` | local default | noisy value est. | ↑ |
| discount | 0.95 | task | `config.py:526` | benchmark | — | — |
| pessimism | 0.5 | MOPO λ tuned | `config.py:527` | local default | over/under-conservative | — |
| bamcts_depth | 5 | large (MCTS) | `config.py:534` | local default | **shallow tree ≈ greedy** | ↑↑ |
| bamcts_simulations | 128 | ≫ (AlphaZero-scale) | `config.py:535` | local default | **posterior barely explored** | ↑↑ |
| ogsrl_rollout_horizon | 6 | task | `config.py:536` | local default | myopic constraint est. | ↑ |
| ogsrl train_iterations | 30 | many | `ogsrl.py:213` | local default | under-fit actor (full only) | ↑ |
| ogsrl safety/ood budget | 0.02 / 0.05 | domain | `ogsrl.py:210–211` | benchmark | feasibility too tight/loose | — |
| delphic world_count | 10 | ensemble-dependent | `delphic.py:81` | local default | unstable delphic var | ↑ |
| delphic cql_alpha / λ | 0.5 / 0.1 | tuned in paper | `delphic.py:82–83` | local default | too/insufficiently conservative | — |
| eval seeds / episodes | 5 seeds × 4 eps, H=50 | — | `hidden_rk.yaml` | benchmark | high-variance estimate | — |

**Answers.** (1) **Much smaller than paper:** BA-MCTS depth(5)/simulations(128) and ensemble
size(5) vs AlphaZero-scale search / neural ensembles; Delphic 10 linear worlds vs
neural/variational. (2) **Mechanisms possibly inert at budget:** BA-MCTS Bayes-adaptive search
(shallow + no in-tree posterior update → close to RefPlan/greedy — see J3); RefPlan reflect
(hidden: inert *by construction*, not budget). (3) **Inherited from known-`r,K` without new
justification:** essentially **all** (config documents verbatim reuse of the old literals). (4)
**Changed after historical results:** none evidenced (returns not opened). (5) **Return-blind
sensitivities required before a fair comparison:** BA-MCTS depth×simulations (does it diverge
from RefPlan?), Delphic world_count×λ, OGSRL horizon×train_iterations, ensemble_size. (6)
**Comparable compute across methods?** Roughly, but **uneven**: OGSRL-full trains an actor while
OGSRL-hidden does not; BA-MCTS spends 128×5 model calls while RefPlan/MOPO spend 96×32 — no
method is *deliberately crippled*, but the budgets were never tuned for the hidden setting.

---

## J. Method-specific questions

**J1 RefPlan.** (1) "Reflect" = deployment-time **posterior over dynamics** ("reflect") then
**plan** — local naming matches the paper's doubly-Bayesian idea. (2) Published algorithm
(ICML25), here idea-level. (3) Yes — learns a ridge-linear ensemble from the 4000-split; posterior
over members. (4) Full: ensemble sampling + posterior weighting + `mean−pessimism·std`. **Hidden:
plain pessimistic particle MPC — no posterior weighting.** (5) **In hidden mode it is *not*
independent of MOPO**: `refplan.act` hidden delegates to the same `PublicParticlePlanner.plan`
MOPO uses, over the same `PublicDynamicsEnsemble`, same pessimism → same action distribution;
the posterior is diagnostic-only. (6) Defensible name: **"posterior-ensemble / RefPlan-inspired
planner"** in full mode; in hidden mode it must be described as **pessimistic MPC (== MOPO)** or
fixed so the posterior feeds planning.

**J2 OGSRL.** (1/2) Offline Guarded Safe RL (NeurIPS25); defining = OOD guardian + safety-cost
constraint + constrained policy learning over a model. (3) **Hidden mode does NOT train the
guarded actor** (`ogsrl.py:388` early return) — it is a one-step constrained greedy over surrogate
one-step predictions with a kNN guardian and risk/OOD budgets. Full mode **does** train the
Lagrangian softmax actor + critics. (4) OOD = standardized k=5 NN distance vs α=0.05 quantile
threshold; safety = surrogate risk (hidden) / model unsafe-occupancy (full); dual λ updated in
full only. (5) Hidden safe-mode needs **no** private info — risk comes from the public surrogate.
(6) **Full: OGSRL-inspired adaptation; Hidden: project-specific constrained greedy** (guarded, but
not a learned safe policy).

**J3 BA-MCTS.** (1) ICLR26 BA-MCTS. (2) Bayes-adaptive latent = which ensemble member/MDP is
true. (3) **Sampled once at the root, not updated along simulated histories** (`bamcts.py`
fixes `member_idx` for the whole rollout; posterior updated only at real `observe`). (4) Tree
branches on **actions only**; continuous state is **bucketed/aggregated** (`_key`/`_public_key`),
not branched on beliefs/models. (5) Root member sampling ✓, UCT ✓; **posterior update in-tree ✗,
particle filtering in-tree ✗, progressive widening ✗** (finite actions). (6) Given (3)–(5) it is a
**belief-rooted finite-action MCTS over a learned ensemble** — "BA-MCTS-inspired" is defensible;
"BA-MCTS" (unqualified) is **not**. (7) At depth 5 / 128 sims with no in-tree posterior update and
discrete actions, it is **at risk of being behaviourally close to RefPlan/greedy**; a return-blind
depth×sims sensitivity is needed to show it differs (Section I).

**J4 Delphic.** (1) ICLR24 Delphic; class `DelphicCQLPolicy`. (2) Benchmark confounding source is
**weak/indirect**: partial observation (lognormal survey noise) + hidden demographics, **not** the
paper's action-outcome confounder; `_posterior_ambiguity` collapses to 0 when `sigma_obs=0`
(`delphic.py:28–39`), so with no obs-noise it is a genuine **negative control**. (3) Ambiguity set
= 10 random-projection latent worlds scaled by posterior ambiguity (`_fit_world:102`). (4)
Implements the **pessimistic penalty + CQL-style value head** and cross-world variance, but **not**
the paper's identifiability treatment or neural world-model ensemble. (5) With partial-obs-only,
the scientific reading is "**value pessimism to observation-induced ambiguity**", not
"hidden-confounding correction". (6) **Implemented, registered, and tested** (`test_methods.py`
incl. `test_delphic_uncertainty_responds_to_hidden_state_ambiguity`; hidden-fit loop in
`test_hidden_rk.py`) — but **absent from the hidden headline manifest**
(`make_hidden_rk_manifest.py` routes only `refplan/bamcts/ogsrl` + 2 natives;
`registration.json` methods list confirms). (7) 10 linear worlds ≪ the paper's design; adequacy
untested.

**J5 MOPO removal.** (1) Registered `mopo` (`__init__.py:4,17`); referenced in `manifest.py:33`
default list, `cli.py:207` (a probe `run_method("mopo")`), `scripts/{make_real_experiment_manifests,
make_cumulative_controls_12h_manifests,extract_report_tables,capture_trajectories,check_docs}.py`,
and tests (`test_hidden_rk.py:112`, `test_real_ecology.py` backend/aggregate cases,
`test_evaluator_gate.py`). (2) **No retained method inherits MOPO-*specific* code** — the shared
code (`ContinuousDynamicsEnsemble`, `PublicDynamicsEnsemble`, `ParticleMPC`,
`PublicParticlePlanner`, `build_reward`) is owned by the framework, not MOPO; MOPO is itself just a
thin caller. (3) Removing MOPO from **manifests/results** leaves shared training code
**unchanged**. (4) Later updates would touch: `manifest.py` default list, `cli.py` probe,
`__init__` registry, `scripts/check_docs.py:127` documented set, the doc tables
(`04_algorithm_adaptations…:26`), and the MOPO-referencing tests. (5) **Yes** — MOPO can remain
archived/registered but excluded from the retained comparison with no reproducibility impact; the
hidden headline manifest **already excludes it**. **(Do not delete during this audit — none of the
above was modified.)**

---

## K. Shared-model concern

**Do the four share one model?** **Yes.** Hidden: `refplan/bamcts/ogsrl/mopo` all fit
`PublicDynamicsEnsemble` (ridge-linear bootstrap in `log1p(obs/observation_scale)` space,
`public_models.py:53`); `delphic` instead fits ridge Q-heads over the **same** `PublicBeliefCache`
features. `refplan/mopo` (hidden) additionally share the identical `PublicParticlePlanner`. Full
mode: the first four share `ContinuousDynamicsEnsemble` + `ParticleMPC`. The whole `src/` general
stack uses **no torch/jax/d3rlpy** (verified: 0 torch imports in the four method files + shared
model/planner/surrogate).

1. **Remaining differences after fitting:** RefPlan = posterior-weighted MPC (full) / plain MPC
   (hidden); MOPO = plain MPC; BA-MCTS = tree search vs MPC; OGSRL = constrained actor(full)/greedy
   (hidden) + guardian; Delphic = CQL Q-head + delphic penalty (different estimator). So the
   *control strategy* differs, but the **dynamics model class is identical**.
2. **Faithful to each paper?** No single ridge-linear ensemble matches any paper's model (each
   assumes a different neural/variational/confounded world model) — disclosed in `04_…tex`.
3. **Could model misspecification dominate comparisons?** **Yes, plausibly** — all methods inherit
   the same linear-Gaussian bias, so differences reduce to planner/penalty choices on a common
   (possibly wrong) model. This is a first-order risk for interpreting any comparison.
4. **Truly independent baselines?** More accurately **planner/policy variants over one shared local
   model**, except Delphic which varies the value estimator.
5. **Designed for known `r,K`, patched for hidden?** **Yes** — the ensemble was K_ref-normalized
   with control features (known mode) and re-expressed in observation space for hidden mode.
6. **Does each paper need a different representation?** **Yes** — Delphic needs a compatible-world
   set, BA-MCTS a BAMDP posterior with in-tree update, RefPlan a deployment dynamics posterior,
   OGSRL a support/safety model — collapsing them onto one ridge ensemble is the central
   adaptation.
7. **Appropriate reuse vs algorithm collapse:** belief cache / MPC engine / dataset split are
   *appropriate* framework reuse; **RefPlan≡MOPO (hidden)** and **BA-MCTS≈greedy at current
   budget** are the cases at risk of *inappropriate collapse*.

---

## L. Tests and evidence

**Executed in Phase 1 (read-only unit tests; no experiment artifacts written, no returns opened),
`PYTHONPATH=src`, `MPLCONFIGDIR` redirected to scratch:**
- `python -m pytest tests/real/test_hidden_rk.py -q` → **9 passed**.
- `python -m pytest tests/synthetic/test_methods.py -q` → **4 passed** (incl.
  `test_delphic_uncertainty_responds_to_hidden_state_ambiguity`, `…_all_methods_fit_and_act`).

**Inventory (general baselines):**

| Test dimension | Present? | Evidence / Gap |
|---|---|---|
| paper-defining mechanism | partial | Delphic ambiguity-response test ✓; **no** test that BA-MCTS search ≠ RefPlan, that RefPlan posterior changes hidden actions (it can't), or that OGSRL constraints bind |
| hidden-information blocking | **yes** | `test_hidden_rk.py:114,255` |
| known/hidden routing | **yes** | `test_hidden_rk.py:100,110` |
| episode order / split integrity | partial | `test_episode_preserving_4000_target_bound`; split logic in `training_monitor.py` (no dedicated general-method split test) |
| deterministic reproducibility | partial | seeds fixed; **no** explicit "same seed → same action" test for the four |
| uncertainty calibration | **no** | none for ensemble disagreement / delphic var |
| planner correctness on solvable case | **no** | none |
| OOD/support behaviour | **no** | guardian threshold untested behaviourally |
| safety without private info | partial | `test_private_safety_diagnostics_cannot_change_surrogate` |
| artifact provenance / method ID | **yes** | `test_serialized_hidden_caches…`, manifest default test |
| runtime/memory canary | **no** (general) | ecology canary tooling exists (`test_corrected_canary_tooling.py`) |
| data-budget / HP sensitivity | **no** | none — the key adequacy gap (Section I/H) |

---

## M. Adaptation register

| Method | Departure from paper | Class | Why required | Scientific consequence | Must fix? |
|---|---|---|---|---|---|
| all | neural world model → ridge-linear ensemble | comp. approx. | auditable, small, no-torch benchmark | shared misspecification bias | disclose (done) |
| all | continuous/large action → 11 discrete mgmt actions | discrete-action adapt. | conservation action menu | no progressive widening needed | no |
| all | infinite/long horizon → H=50 episodic | finite-horizon adapt. | benchmark episodes | myopic-ish planning | no |
| all (hidden) | exact reward → fitted surrogate; true safety threshold → surrogate risk | reward replacement / privacy | r,K hidden | risk channel is near-inert (public extinction ~0) | verify risk channel is meaningful |
| all (hidden) | mechanistic filter → **memoryless** obs jitter | partial-obs adapt. | no simulator access | methods needing history lose it | consider a real filter |
| RefPlan | posterior **ignored** in hidden planning | **material replacement** | — (not required) | **RefPlan ≡ MOPO in hidden mode** | **yes** |
| OGSRL | **no actor trained** in hidden mode | **material / missing** | — (not required) | hidden OGSRL = one-step greedy, not guarded RL | **yes** |
| BA-MCTS | no in-tree posterior update; shallow tree | missing / comp. approx. | discrete actions; runtime | may not differ from RefPlan/greedy | verify (sensitivity) |
| Delphic | random-proj worlds; linear head; no confounder | material replacement | benchmark has partial-obs not confounding | "obs-ambiguity pessimism" not confounding-correction | reinterpret + (optionally) fix |
| Delphic | absent from hidden manifest | scheduling gap | — | not in headline comparison | decide inclusion |

---

## N. Severity-ordered findings and recommendations

**N1 — Blocking scientific identity — RefPlan collapses onto MOPO in hidden mode.**
Method: RefPlan. Evidence: `refplan.py:73–84` delegates to `PublicParticlePlanner.plan` (same
object MOPO uses) with the posterior used only for a diagnostic; `observe` updates a posterior
that hidden `act` never reads. Why it matters: removing MOPO while keeping RefPlan would ship the
*same* pessimistic-MPC algorithm under two names; RefPlan's defining "reflect" is inert exactly in
the mode being run. Fix: feed the posterior into hidden planning (posterior-weighted sequence
scoring, as in full mode). Tests: "hidden RefPlan action changes when posterior changes"; "RefPlan
≠ MOPO on a constructed cell". Effect: RefPlan diverges from MOPO; modest runtime ↑. **New run
required for RefPlan-hidden; no new ID.**

**N2 — Blocking scientific identity — OGSRL trains no policy in hidden mode.**
Method: OGSRL. Evidence: `ogsrl.py:388` early `return diagnostics` before actor/critic/dual
training (`:410–447`); hidden `act` is one-step constrained greedy. Why: the paper's defining
guarded *policy learning* is absent; hidden OGSRL is a constrained greedy planner. Fix: train the
Lagrangian softmax actor over surrogate rollouts in hidden mode (mirror full path), or rename
hidden OGSRL honestly. Tests: "hidden OGSRL actor weights are trained/non-trivial"; "constraint
binds". Effect: OGSRL-hidden becomes a learned guarded policy; runtime ↑ (rollout training). **New
run required; no new ID.**

**N3 — Blocking fair comparison — shared linear model may dominate.**
Methods: all. Evidence: Section K. Why: differences may reflect planner choices on one possibly
misspecified model rather than method merit. Fix (verification, not code): report dynamics RMSE on
holdout per cell; a return-blind check that method rankings aren't explained by shared model error.
**No new ID.**

**N4 — Blocking fair comparison / naming — BA-MCTS may not differ from RefPlan at current budget.**
Method: BA-MCTS. Evidence: no in-tree posterior update (`bamcts.py`), depth 5 / 128 sims, discrete
actions. Fix: return-blind depth×simulations sensitivity; if inseparable from RefPlan, add in-tree
posterior update or relabel. Tests: "BA-MCTS action distribution ≠ RefPlan on a constructed cell".

**N5 — Required verification missing — Delphic not in the hidden headline manifest.**
Evidence: `make_hidden_rk_manifest.py` ROUTING + `registration.json` methods list exclude
`delphic`. Why: the audit's retained set includes Delphic, but the current hidden run does not
produce it. Fix: decide inclusion; if included, add routing + a hidden Delphic privacy/relabel test.

**N6 — Blocking hidden-mode validity — surrogate risk channel likely inert.**
Methods: OGSRL (and any risk-using planner). Evidence: risk head is logistic on public
`terminated` (state==0), which fires extremely rarely (prior audits note ~0.002% ⇒
`constant_single_class` fallback path in `public_surrogate.py:309`). Why: OGSRL's safety constraint
may never bind in hidden mode ⇒ its "guarded safe" identity is vacuous. Fix: verify termination
prevalence per cell; consider an occupancy-based public risk target. Verification first.

**N7 — Required verification missing — general-method relabel & payload-leakage tests.**
Evidence: relabel test covers only natives; `privacy.py` checks names not values (G). Fix: add
family-relabel invariance for `refplan/bamcts/ogsrl/delphic` and a numeric assertion that no
method-facing array equals `K_base`/`safety_threshold`.

**N8 — Hyperparameter/data adequacy risk.**
Evidence: Section I — all budgets inherited from the known-`r,K` literals; ensemble 5, worlds 10,
tree 128×5. Fix: the return-blind sensitivities in I(5). No naming impact.

**N9 — Runtime/engineering risk — hidden fixes uncommitted and mixed.**
Evidence: A.3/A.4. Fix: commit the hidden general-baseline changes as an isolated changeset (or a
content digest) before any corrective run, so provenance is not "5f9cf32-dirty + copied dir".

**N10 — Documentation/naming only — OSGRL typo confined to the request doc; "-inspired" wording
already correct.** Evidence: B. Fix: standardize on `OGSRL`; keep "-inspired" language.

---

## O. Required final verdicts

| Method | Verdict |
|---|---|
| **RefPlan** | **Paper-inspired adaptation** in full mode; **Materially mislabelled/incomplete in hidden mode** (reflect inert ⇒ ≡ MOPO). Insufficiently verified. |
| **OGSRL** | **Paper-inspired adaptation** in full mode; **Materially incomplete in hidden mode** (no policy trained). Insufficiently verified. |
| **BA-MCTS** | **Paper-inspired adaptation**, **insufficiently verified** (must show it differs from RefPlan/greedy). |
| **Delphic** | **Paper-inspired adaptation** (partial-obs reinterpretation), implemented/registered/tested but **not scheduled** in the hidden run; insufficiently verified. |
| **MOPO** | **Paper-inspired (MOPO-style); safely excludable** — already out of the hidden manifest; do not delete. |

**Answers.**
1. **Retain unchanged:** none as-is for the hidden setting. (In *full* mode, RefPlan/OGSRL/BA-MCTS/
   Delphic are coherent paper-inspired baselines.)
2. **Require implementation fixes before any new run:** **RefPlan** (posterior→hidden planning),
   **OGSRL** (train hidden actor or rename). 
3. **Require only naming/doc corrections:** the `OSGRL→OGSRL` typo (request doc); "-inspired"
   already correct. 
4. **Require paper/official-code verification before judgment:** BA-MCTS (in-tree posterior
   necessity), Delphic (confounding vs partial-obs interpretation).
5. **Is {RefPlan, OGSRL, BA-MCTS, Delphic} scientifically coherent?** **Conditionally.** As
   *idea-level* baselines over a shared model, yes — *provided* RefPlan-hidden and OGSRL-hidden are
   fixed so they are not (respectively) MOPO-in-disguise and an untrained greedy, and provided BA-
   MCTS is shown to differ from RefPlan.
6. **Any omitted general baseline needed to avoid a misleading comparison?** A **behavior-cloning /
   dataset-return floor** and an explicit **model-free conservative baseline (e.g. CQL/IQL)** would
   contextualize whether the shared linear model, not the algorithms, drives results. MOPO's
   *removal* is fine; but with RefPlan≡MOPO (hidden) unfixed, removing MOPO removes the only
   honestly-named instance of that algorithm.
7. **Minimum defensible corrected suite for the next experiment:** RefPlan (posterior wired into
   hidden planning), OGSRL (hidden actor trained **or** renamed to a constrained-greedy baseline),
   BA-MCTS (verified-distinct or in-tree posterior added), Delphic (included in the manifest with a
   corrected partial-obs interpretation), plus a return-floor/model-free reference — all over the
   shared 4000/25 + 0.2-holdout budget with dynamics-RMSE reporting.

---

## Final recommendation

`NOT READY: paper identity or privacy fixes required`

**Primary blockers are scientific-identity, not privacy** (privacy blocking is implemented and
passes its tests). Specifically: **RefPlan's defining "reflect" is inert in the hidden mode being
run (RefPlan ≡ MOPO)**, and **hidden OGSRL trains no guarded policy** — both must be corrected (or
the methods renamed) before a hidden-`r,K` comparison can carry the paper names. Secondary work
(BA-MCTS distinctness, Delphic inclusion + reinterpretation, risk-channel and relabel verification,
and committing the currently-uncommitted hidden changeset) is required before the suite is fair and
reproducible. These are concrete, known edits — i.e. this is ripe for a **corrective
implementation plan** immediately after review.

*Phase 1 stops here. No files were edited, no jobs launched/altered, MOPO was not removed, and no
performance returns were opened.*
