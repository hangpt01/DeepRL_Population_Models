# Phase 1B — Deep Paper Verification & Corrective Plan (General-RL Baselines)

Continues `PHASE1_GENERAL_RL_BASELINE_PAPER_ALIGNMENT_AUDIT.md`. **Plan/audit only** — no
runtime code, tests, manifests, registrations, results, or jobs were modified; no performance
returns inspected; the dirty working tree and active PLUS/MOOR jobs were left untouched.

Retained set: **RefPlan, OGSRL, BA-MCTS, Delphic** (canonical **OGSRL**). MOPO stays archived/
excluded, not deleted.

Evidence tags per Task 1: **[PAPER]** = stated in paper/supplement · **[OFFICIAL-CODE]** =
author code · **[LOCAL-CODE]** = this repo only · **[INFERENCE]** = reasoned · **[UNVERIFIED]** =
inaccessible/not established.

> **Headline correction to Phase 1.** Deep reading changed two judgments: (i) **Delphic is more
> applicable than Phase 1 credited** — the offline behavior policy is *privileged* (acts on true
> hidden abundance) while methods see only noisy observations, so the data carries **genuine
> hidden confounding** (unobserved `s` → both action and outcome), which is exactly Delphic's
> setting. (ii) **BA-MCTS's gap is deeper than "just add a sensitivity"** — the cited paper
> *explicitly rejects* the root-sampling scheme the local code uses. Details below.

---

## 1. Source / provenance ledger

| Method | Bibliographic identity | Local PDF | Official code | Algorithm/eq. anchors read |
|---|---|---|---|---|
| RefPlan | Jeong, Wang, Wang, Sanner, Poupart, *Reflect-then-Plan: Offline Model-Based Planning through a Doubly Bayesian Lens*, **ICML 2025** | `docs/references/general_model_based_offline_RL_papers/ICML25_Reflect-then-Plan-….pdf` (21 pp) | **[UNVERIFIED]** — PDF says "code available upon acceptance"; `github.com/iclavera/` is a cited *baseline*, not RefPlan | §2–3 (prior policy, belief `b_t=p(M\|τ_:t)`, latent `m`), Eq (10) MPPI, Alg. 2 (appendix ref) |
| OGSRL | Yan, Shen, Wachi, Gros, Zhao, Hu, *Offline Guarded Safe RL for Medical Treatment Optimization Strategies*, **NeurIPS 2025 Spotlight** | `docs/references/discrete_action_continuous_obser_state/NeurIPS25_Spotlight_….pdf` (42 pp) | **[OFFICIAL-CODE]** `github.com/Runz96/SafeRL-OGSRL` (fetched): `guardian.py` (Gaussian-kernel KDE), `models/cpo.py`/`cpo_guard.py` (CPO), `transition_model.ipynb` (kNN) | Alg. 1 (guardian→guarded model→ConOpt), §3.1 (GCL min-volume set), Def. 2, Goal (safety-cost constraint) |
| BA-MCTS | Chen, Xu, Chen, Schneider, *Bayes Adaptive Monte Carlo Tree Search for Offline Model-Based RL*, **ICLR 2026** | `docs/references/general_model_based_offline_RL_papers/ICLR26_Bayes Adaptive….pdf` (33 pp) | **[UNVERIFIED]** — OpenReview only | Eq (1) BAMDP posterior `b'(θ)∝b(θ)P_θR_θ`, Eq (2) Bayes-optimal Q over `x=(s,b)`, Eq (4) deep-ensemble posterior update, §4.1 ensemble, App. A (root-sampling rejected under DPW), PUCT+DPW, Dirichlet root noise |
| Delphic | Pace, Yèche, Schölkopf, Rätsch, Tennenholtz, *Delphic Offline RL under Nonidentifiable Hidden Confounding*, **ICLR 2024** | `docs/references/discrete_action_continuous_obser_state/ICLR24_Delphic….pdf` (29 pp) | **[UNVERIFIED]** for the algorithm; `github.com/clinicalml/gumbel-max-scm` is the *sepsis simulator*, not Delphic's code | Alg. 1 (compatible worlds factorising to `P^{π_b}(τ)`; `u^π_d=Var_w Q^π_w`; pessimism), §5.1 ELBO, §5.2 `r̃=r−λu_d`, Fig. 10 (`\|W\|>30`) |

**Repo provenance (unchanged from Phase 1, rechecked):** HEAD `5f9cf32` = known-`r,K`; hidden
fixes are **uncommitted** (`bamcts +97`, `ogsrl +148`, `refplan +38`, `mopo +18`, `delphic +8`
+ new untracked `public_models.py`, `public_surrogate.py`, `privacy.py`); frozen-by-copy at
`real_ecology_runs/hidden_rk_comparison_20260716/code/`; `git describe = 5f9cf32-dirty`.

**Benchmark confounding provenance (new, decisive for Delphic):** `collector.py:211`
`abundance = true_state if self.privileged else observation`; default
`privileged_behavior=True` (`config.py:105`); `MixedDangerZonePolicy.act` (`collector.py:204–231`)
branches on thresholds of `abundance` (`0.3/0.5/0.7·K`). **[LOCAL-CODE]** ⇒ logged actions depend
on the **unobserved true abundance**; methods (hidden) see only the lognormal observation ⇒ a real
back-door `a ← s → s'`.

---

## 2. Paper-mechanism summaries and the minimum identity-preserving core

### RefPlan (ICML 2025)
**[PAPER]** "Doubly Bayesian": (a) a **prior/base offline policy** `π_p` (a conservative offline-RL
policy) supplies the plan prior `p(τ)` and generates `N̄` prior plans with the learned model; (b) a
**belief over dynamics** `b_t=p(M|τ_:t)` (prior `b_0=p(M)`) is **updated at deployment** from
real-time observations, approximated by a latent `m` à la VariBAD/Dorfman, with dynamics models
**conditioned on `m_t`**; (c) planning is **probabilistic inference** — the belief is
**marginalized** into an **MPPI** plan optimizer, Eq (10)
`a*_{t+h}=Σ_n exp(κR^n_H)a^n_{t+h} / Σ_n exp(κR^n_H)`; (d) a **return-variance uncertainty penalty**
(Sikchi 2021).
**Minimum identity core:** *belief over dynamics updated at deployment* **AND** *its marginalization
into planning*. Integration rule = posterior-/belief-weighted plan scoring (MPPI weighting), not a
single sampled model. The **prior policy** is defining to the "doubly" framing but may be a
**disclosed simplification** for an idea-level baseline; the return-variance penalty is expected.
**Answer to Task 2:** posterior-weighted (marginalized) planning is required; a belief that is
computed but ignored does **not** satisfy identity.

### OGSRL (NeurIPS 2025 Spotlight)
**[PAPER]** Alg. 1: (1) learn a **guardian classifier `ĝ`** partitioning `S×A` into ID/OOD via a
**minimum-volume support set** (GCL, coverage threshold `α_c`); (2) build a **guarded model
`M̂_ĝ`** (learned dynamics restricted to ID region, Def. 2); (3) `π̂ ← ConOpt(M̂_ĝ)` — a
**constrained policy optimizer** (they use **CPO**; paper states "other constrained RL algorithms
are not prohibited") maximizing `V^π_{r,T}` s.t. **safety-cost constraints** `V^π_{c_j,T}(ρ_0)≤α_c`
and the **OOD constraint**. **[OFFICIAL-CODE]** guardian = Gaussian-kernel KDE; transitions = kNN;
ConOpt = CPO (`cpo.py`, `cpo_guard.py`).
**Minimum identity core:** guarded/support-restricted **learned model** + **OOD guardian** +
**multi-step constrained policy optimization** with a safety-cost constraint. **A one-step greedy is
NOT ConOpt** ⇒ cannot be called OGSRL. Because the paper permits any constrained optimizer, the
local **Lagrangian softmax actor is a legitimate ConOpt variant** — the defect is that hidden mode
runs **no** ConOpt. When the private safety objective is unavailable, the **theoretical safety
guarantee must be dropped**; the guardian + surrogate-risk constraint may remain.

### BA-MCTS (ICLR 2026)
**[PAPER]** Offline MBRL as a **BAMDP** with augmented state `x=(s,b)`, `b` a belief over world
model `θ`. Eq (1): `b'(θ)∝b(θ)P_θ(s'|s,a)R_θ(r|s,a)` — the belief is **updated to the Bayesian
posterior along the trajectory**. Eq (4): practical **deep-ensemble** posterior update — `b` is a
categorical over `K` members, reweighted multiplicatively by transition+reward **likelihood at each
node**. Planner = **PUCT + double progressive widening** (continuous `S,A`), Q normalized to `[0,1]`,
**Dirichlet root noise** `a∼ηx_d+(1−η)π`. Search results are **distilled into policy+value networks**
inside **policy iteration** ("RL+Search"). **App. A: root sampling (fix `θ` per rollout) is
explicitly rejected — "the rationale of root sampling (Lemma A.1) does not hold when applying DPW".**
**Minimum identity core:** the **in-tree Bayes-adaptive belief update (Eq 1/4)** — the belief over
models must evolve *inside simulated histories*; that is what makes it *Bayes-adaptive*. The outer
distillation/PI loop is for real-time execution and can be omitted for an **evaluation-time planner
baseline**, *provided the in-tree update is present*. For **discrete actions**, action-DPW is
unnecessary; only state handling (widening/aggregation) + the belief update are needed. **Answer to
Task 2:** in-tree belief updating is required; root sampling is a *different* (older BAMCP) method
the cited paper rejects for its regime.

### Delphic (ICLR 2024)
**[PAPER]** Alg. 1: (1) learn **compatible world models** `{Z_w,ν_w,ρ_{0,w},P_{r,w},T_w,π_{b,w}}`
that **all factorise to the same behavior distribution `P^{π_b}(τ)`** (trained by ELBO, §5.1, with a
**latent confounder `z`** driving both `π_{b,w}(a|s,z)` and outcome via `T_w,P_{r,w}`); (2)
counterfactual `Q^π_w`; (3) **`u^π_d(s,a)=Var_w(Q^π_w(s,a))`**; (4) pessimism `r̃=r−λu_d` (or
sample weighting). Essential assumption: an **unobserved variable confounding action↔outcome**;
worlds differ only in unobservable counterfactuals, agreeing on observables. `|W|>30` in experiments.
**Minimum identity core:** a set of **observationally-compatible worlds** + **delphic uncertainty =
cross-world Q variance** + **pessimism penalty**. **Answer to Task 2 (can this benchmark instantiate
the ambiguity set?):** **Yes, partially** — the privileged behavior policy creates the required
action↔outcome confounding through the observation gap; ambiguity vanishes as `σ_obs→0`, matching
the paper's structure. But the **local worlds are not ELBO-trained to reproduce `P^{π_b}(τ)`** ⇒ the
"compatible" guarantee is unproven ⇒ **name it `Delphic-inspired`**, not `Delphic`.

---

## 3. Known-vs-hidden implementation matrices

Classes: **FR** faithful/retained · **BA** necessary benchmark adaptation · **PA** privacy-required
adaptation · **CA** computational approximation · **MR** material replacement · **MI** missing/inert ·
**NA** · **UV** unverified.

### RefPlan
| Paper-defining component | Full/known (`5f9cf32`) | Hidden (uncommitted) | Evidence | Class | Required action |
|---|---|---|---|---|---|
| Belief over dynamics, deployment update | ensemble posterior updated in `observe` | same posterior updated | `refplan.py:113–134` | FR | keep |
| Marginalize belief into planning | posterior-weighted member scoring `mean−pess·std` | **posterior ignored; plain `PublicParticlePlanner.plan`** | `refplan.py:85–111` vs `:73–84` | **MR (hidden)** | **wire posterior into hidden planning** |
| Prior/base offline policy `π_p` | none (random sequences) | none | `planning.py:40–46`; `public_models.py:139–148` | MI (both) | disclose; optional prior |
| MPPI weighting (Eq 10) | argmax over sequences (not softmax-κ) | same | `planning.py:150` | CA | disclose (argmax≈MPPI κ→∞) |
| Return-variance penalty | present | present | `refplan.py:105`,`planning.py:150` | FR | keep |
| Model class (neural) | ridge-linear ensemble | ridge-linear (obs-space) | `dynamics.py`/`public_models.py` | CA | disclose |

### OGSRL
| Paper-defining component | Full/known | Hidden | Evidence | Class | Required action |
|---|---|---|---|---|---|
| OOD guardian (support envelope) | kNN(5) dist vs α=0.05 quantile | kNN over obs features | `ogsrl.py:24–138,142–201` | CA (KDE≈kNN, paper uses kNN) | keep |
| Guarded/support-restricted model | ridge-linear ensemble | obs-space ensemble | `ogsrl.py:389`,`365` | CA | keep |
| ConOpt (constrained policy optimization) | linear softmax actor + λ-dual PG | **none — early `return` before actor** | `ogsrl.py:410–447` vs `:388` | **MI (hidden)** | **run ConOpt over surrogate model in hidden** |
| Safety-cost constraint | model unsafe-occupancy ≤ budget | surrogate-risk ≤ budget (one-step) | `ogsrl.py:549`,`290–299` | BA/PA | keep; verify risk not inert (N6) |
| Uses full state history | posterior particles | **memoryless obs jitter** | `beliefs.py:586,635` | MR | acknowledge; optional filter |
| Theoretical safety guarantee | none transferred | none | — | MI (disclosed) | keep removed |

### BA-MCTS
| Paper-defining component | Full/known | Hidden | Evidence | Class | Required action |
|---|---|---|---|---|---|
| In-tree Bayes-adaptive belief update (Eq 1/4) | **absent — one member fixed per rollout (root sampling)** | absent (same) | `bamcts.py:139–189,94–137,224–233` | **MI (both)** | **add per-node ensemble-posterior reweight, or relabel BAMCP-inspired** |
| Root model sampling from belief | present (sample member ∝ posterior) | present | `bamcts.py:196,224` | FR (BAMCP-style) | keep |
| PUCT + DPW | UCB + state bucketing; no DPW | UCB + obs bucketing | `bamcts.py:106,151`,`_key:74` | BA (discrete actions) | keep (justify) |
| Outer PI distillation to policy/value nets | none (decision-time only) | none | whole file | MI (acceptable as planner) | disclose |
| Dirichlet root exploration | none (unvisited-first) | none | `bamcts.py:104,147` | CA | optional |
| Model class (ensemble) | ridge-linear | obs-space ridge | `dynamics.py`/`public_models.py` | CA | disclose |

### Delphic
| Paper-defining component | Full/known | Hidden | Evidence | Class | Required action |
|---|---|---|---|---|---|
| Worlds factorise to `P^{π_b}(τ)` (ELBO, latent confounder) | random-proj latents scaled by ambiguity; fitted behavior/Q heads | same (obs features) | `delphic.py:102–133,28–39` | **MR** | rename `-inspired`; optionally add behavior-compat objective |
| Delphic uncertainty `Var_w Q^π_w` | present | present | `delphic.py:152–159` | FR | keep |
| Pessimism `r̃=r−λu_d` | present (CQL head − λ·unc) | present | `delphic.py:200–212` | FR | keep |
| Confounding present in data | via privileged behavior (σ_obs-driven) | same | `collector.py:211` | **BA/FR** | **document as the ambiguity source** |
| `\|W\|` worlds | 10 | 10 | `delphic.py:81` | CA (paper >30) | raise/sensitivity |
| Registered in hidden manifest | n/a | **absent** | `make_hidden_rk_manifest.py:17` | MI | add if retained |

**Did the known-`r,K` repair keep paper identity?** Full mode is materially closer for OGSRL
(actor+dual present) and RefPlan (marginalization present); BA-MCTS lacks the in-tree update in
*both* modes (not a hidden-only defect); Delphic's compatible-world gap exists in *both* modes. So
BA-MCTS and Delphic identity issues are **not** artifacts of the hidden repair.

---

## 4. Reassessment of Phase 1 findings N1–N10

| # | Phase 1 finding | Verdict vs full paper | Note |
|---|---|---|---|
| N1 | RefPlan reflect inert in hidden ⇒ ≡ MOPO | **Confirmed, but correction needs modification** | Fix must restore **marginalization** (Eq-10-style belief-weighted scoring), not merely attach posterior as a diagnostic; also disclose missing prior policy. |
| N2 | OGSRL trains no policy in hidden | **Confirmed unchanged** | Paper Alg. 1 requires ConOpt; one-step greedy ≠ OGSRL. Paper permits non-CPO optimizers, so the local Lagrangian actor is an acceptable ConOpt once actually run in hidden. |
| N3 | Shared linear model may dominate | **Confirmed, downgraded to verification** | Add per-cell holdout dynamics-RMSE reporting; not a code blocker. |
| N4 | BA-MCTS may not differ from RefPlan; add sensitivity | **Confirmed, but correction needs modification** | A depth/sims sensitivity is **insufficient**: the defining in-tree belief update (Eq 1/4) is *missing*, and the paper **rejects** the root sampling the code uses (for continuous DPW). Either implement Eq 4 or relabel BAMCP-inspired + justify discrete-action root sampling. |
| N5 | Delphic absent from hidden manifest | **Confirmed, reframed** | Not just a scheduling gap: Delphic **is** applicable (privileged-behavior confounding), so **retain + add**, but rename `-inspired` and fix world-compatibility, not merely list it. |
| N6 | Surrogate risk channel likely inert | **Confirmed unchanged** | `public_surrogate.py:309` `constant_single_class` fallback fires when `terminated` never varies; OGSRL/RefPlan safety constraint then never binds. Verify prevalence per cell; consider occupancy risk target. |
| N7 | Missing general-method relabel/value-leak tests | **Confirmed unchanged** | `privacy.py` checks names not values; relabel test covers only natives. |
| N8 | HP/data adequacy risk | **Confirmed unchanged** | Delphic `\|W\|=10`≪ paper 30; ensemble 5; budgets inherited. Return-blind sensitivities required. |
| N9 | Hidden fixes uncommitted/mixed | **Confirmed unchanged** | Provenance isolation is Stage 1 below. |
| N10 | OSGRL typo; "-inspired" wording | **Confirmed unchanged** | Standardize `OGSRL`; "-inspired" is correct for all four. |

**No finding disproved. One unresolved:** exact RefPlan/BA-MCTS/Delphic *official-code* defaults
(**[UNVERIFIED]** — repos inaccessible); paper values used instead.

---

## 5. Adaptation register (with alternatives)

| Adaptation | Why the framework requires it | Effect on interpretation | Alternative considered | Naming consequence |
|---|---|---|---|---|
| Neural world models → ridge-linear ensemble (all) | auditable, no-torch, small-data, deadline | shared model ⇒ methods risk being planner variants | per-method neural models | keep **-inspired**; note shared-model caveat |
| Continuous action/DPW → 11 discrete + bucketing | discrete management menu | removes action-DPW machinery legitimately | discretize finer | no name change |
| RefPlan: drop prior policy `π_p`, use random+enumerated seqs | no shipped conservative offline policy | loses one "Bayes" of doubly-Bayesian | add BC/CQL prior policy (Decision D4) | `RefPlan-inspired` |
| RefPlan hidden: marginalize posterior into planning | otherwise ≡ MOPO | restores defining mechanism | posterior sampling of members | required for identity |
| OGSRL: CPO → Lagrangian softmax actor | simpler, paper permits any constrained optimizer | acceptable ConOpt variant | port CPO/trust region | `OGSRL-inspired` |
| OGSRL hidden: run ConOpt over surrogate model | one-step greedy ≠ OGSRL | restores policy-optimization identity | keep greedy + rename | required for identity |
| OGSRL: drop theoretical safety guarantee (hidden) | private safety objective unavailable | remove safety-guarantee claims | occupancy-based public risk | disclose |
| BA-MCTS: add in-tree ensemble-posterior reweight (Eq 4) | defining Bayes-adaptive mechanism | makes it genuinely BA-MCTS | relabel BAMCP-inspired (root sampling) | identity vs rename (Decision D2) |
| Delphic: worlds via random projection, not ELBO | no variational world-model infra; deadline | "compatible" unproven | add behavior-likelihood compat term | `Delphic-inspired` |
| Delphic: confounding via privileged behavior + obs noise | benchmark's built-in ambiguity source | scientifically valid instantiation | inject explicit latent confounder | supports retaining Delphic |
| All hidden: memoryless obs filter | no simulator access | history-dependent methods weakened | lightweight learned recurrent filter (Decision D3) | disclose |
| All hidden: fitted reward/risk surrogate | exact reward is private | reward is approximate; risk may be inert | occupancy risk target | disclose; N6 fix |

---

## 6. Hyperparameters, data fairness, compute

### 6.1 Method HP table (paper vs official vs current vs proposed)

| Method · HP | Paper value/source | Official default | Known-mode | Hidden-mode | Proposed primary | Provenance | Sensitivity/risk | Refit or replan? |
|---|---|---|---|---|---|---|---|---|
| RefPlan · belief update | posterior over M (Eq, §3) | UV | ensemble posterior | ensemble posterior | keep | paper | low | replan |
| RefPlan · plan integration | MPPI softmax κ (Eq 10) | UV | argmax seq | **argmax, posterior unused** | **posterior-weighted score** | paper→adapt | **high (identity)** | replan |
| RefPlan · ensemble K | ≥7 typical (deep ens.) | UV | 5 | 5 | 7 (scaled) | benchmark | med | refit |
| RefPlan · prior policy | conservative offline π | UV | none | none | none (disclose) / BC opt. | adapt | med | refit if added |
| OGSRL · guardian | KDE, α_c coverage | Gaussian KDE | kNN k=5, α=0.05 | kNN | keep | paper≈code | med | refit |
| OGSRL · ConOpt | CPO | CPO | λ-dual PG, 30 it | **none** | **λ-dual PG in hidden** | paper permits | **high (identity)** | refit |
| OGSRL · safety budget | domain | domain | 0.02/0.05 | 0.02/0.05 | keep; verify | benchmark | med (N6) | replan |
| BA-MCTS · in-tree belief | Eq 1/4 required | UV | **absent** | absent | **add Eq-4 reweight** | paper | **high (identity)** | replan |
| BA-MCTS · simulations | AlphaZero-scale | UV | 128 | 128 | 256 primary + {128,512} sens. | comp-limit | high | replan |
| BA-MCTS · depth | large | UV | 5 | 5 | 8 primary + {5,12} sens. | comp-limit | high | replan |
| BA-MCTS · UCB c / Dirichlet η | c,η in App. table | UV | c=1.25, η=0 | same | c=1.25; η opt. | paper/adapt | low | replan |
| Delphic · `\|W\|` | >30 (Fig. 10) | UV | 10 | 10 | 20 primary + {10,30} sens. | comp-limit | **high** | refit |
| Delphic · λ (pessimism) | tuned | UV | 0.1 | 0.1 | 0.1 + {0.03,0.3} sens. | adapt | med | refit |
| Delphic · cql_alpha | n/a (their ORL algo) | UV | 0.5 | 0.5 | 0.5 | local | med | refit |
| All · offline budget | task | task | 4000/25 | 4000/25 | **4000/25 (matched)** | preregistered | — | refit |

### 6.2 Return-blind data-adequacy & sensitivity design (no returns opened)
- **Learning curves vs budget** at {1000, 2000, 4000, (8000 stress)} transitions, reporting
  **holdout dynamics RMSE**, **behavior NLL**, **delphic-uncertainty magnitude**, **guardian OOD
  rate**, **tree Q-normalization stats** — all **model-internal**, no task return.
- **Action-coverage diagnostic** per cell: transitions & episodes per discrete action; flag actions
  with `<K` support (drives ridge instability, guardian threshold, Delphic per-action Q).
- **Sink/hard populations**: include the known sink cells and `Amur tiger` band case; report whether
  surrogate `terminated` prevalence >0 (else N6 makes risk inert there).
- **Compute-scaled defaults** (per Task 6): for each expensive paper default give (1) paper-near,
  (2) compute-scaled primary, (3) a small preregistered sensitivity proving the scaled value does not
  *qualitatively* collapse the mechanism (e.g., BA-MCTS Q-spread across depth; Delphic `u_d` across
  `|W|`).

### 6.3 Compute estimate (server limits: `m3h`, `MaxArraySize=1001`, pack rows/task)
All four are **NumPy/CPU** (no torch in `src/` general stack). Per-cell fit (ridge/kNN) ≈ seconds;
per-decision planning dominates. **[INFERENCE]** from model sizes:

| Method | Fit/cell | Plan/step | Eval/cell (H=50×20 eps) | Mem |
|---|---|---|---|---|
| RefPlan (posterior-weighted) | <2 s | ~96×32 scores | ~1–3 min | <0.5 GB |
| OGSRL (hidden ConOpt) | +rollout train (30 it × 128 × 6) ≈ 10–40 s | one-step | ~1–2 min | <0.5 GB |
| BA-MCTS (256 sims × depth 8 + Eq-4) | <2 s | 256×8 model calls | **~5–15 min** (largest) | <0.5 GB |
| Delphic (`\|W\|=20`) | 20 worlds × 25 it ridge ≈ 10–30 s | linear | ~1 min | <0.5 GB |

At `registration.json` scale (~288 cells/arm, hidden arm only for retained 4 ≈ ~1150 rows):
**~1–2 CPU-days wall** with row-packing under `MaxSubmit`; BA-MCTS at 512 sims is the cost driver
(keep 512 to the sensitivity arm only). No GPU required.

---

## 7. Staged corrective implementation plan (do **not** execute yet)

Each stage: files · functions · tests · deps · runtime · rerun scope · rollback. Rollback for all
code stages = `git stash`/revert of the *named new* hunks only; **never** `git reset`/checkout that
would touch the ~50 unrelated dirty files or the PLUS/MOOR jobs' artifacts.

**Stage 1 — Provenance isolation (do first).**
Files: (git) the hidden hunks in `methods/{refplan,ogsrl,bamcts,mopo,delphic}.py` + new
`public_models.py`, `public_surrogate.py`, `privacy.py`. Action: create a dedicated branch
`hidden-rk-general` and commit **only** these paths as one changeset; record `git rev-parse` +
`sha256sum` of each file into a `SNAPSHOT_general_hidden.json`. Tests: none. Runtime: <1 min. Rerun
scope: none. Rollback: delete branch; working tree unchanged. **Depends on:** nothing. Gate: hidden
general changes are addressable by commit hash, not "dirty".

**Stage 2 — RefPlan identity correction.**
Files: `public_models.py` (`PublicParticlePlanner.plan`), `methods/refplan.py` (hidden `act`).
Functions: add posterior-/belief-weighted scoring — pass `member_posterior` into `plan` and score
each candidate sequence as a **posterior-weighted mean over members** minus `pess·std` (marginalize
`m_t`), instead of uniform `dynamics.sample_next`. Keep the `observe` posterior update. Disclose the
absent prior policy in docstring. Tests (Stage 6/7): "hidden RefPlan action changes when posterior
changes"; "RefPlan ≠ MOPO on a constructed 2-model cell". Deps: Stage 1. Runtime: replan only.
Rerun scope: RefPlan-hidden cells. Rollback: revert the two hunks.

**Stage 3 — OGSRL identity correction.**
Files: `methods/ogsrl.py` (hidden `fit`). Function: replace the early `return diagnostics` (`:388`)
with the **existing** actor+critic+dual training loop run over **surrogate/PublicDynamicsEnsemble
rollouts** (reuse `_rollouts`/`_rollout_training_metrics` with public reward/risk); keep the
deployment feasibility mask. Remove any residual safety-*guarantee* wording. Tests: "hidden OGSRL
actor weights are trained/non-trivial"; "constraint binds when risk>budget". Deps: Stage 1, N6
check. Runtime: +10–40 s/cell fit. Rerun scope: OGSRL-hidden cells. Rollback: restore early return.

**Stage 4 — BA-MCTS belief-state correction/verification (decision-gated: D2).**
Option A (identity): in `methods/bamcts.py` `_simulate*`, carry a per-simulation categorical
`b` over members and **reweight by member likelihood** `P_θ(s'|s,a)` at each node (Eq 4), selecting
the member per step from `b` instead of a fixed root member. Option B (rename): keep root sampling,
relabel **BAMCP-inspired**, and add App.-A justification that root sampling is valid for
discrete-action/bucketed-state MDPs. Tests: "BA-MCTS action dist ≠ RefPlan on a constructed cell";
(A) "in-tree belief concentrates on the data-consistent member". Deps: Stage 1, D2. Runtime: replan
(+belief bookkeeping). Rerun scope: BA-MCTS-hidden cells. Rollback: revert hunk / label.

**Stage 5 — Delphic scope + manifest (decision-gated: D1).**
Files: `scripts/make_hidden_rk_manifest.py` (`ROUTING`), docs. Action: if retained, add
`"delphic": ("learned",)` to routing; document the **privileged-behavior confounding** as the
ambiguity source; rename reader-facing label to **Delphic-inspired**; (optional) add a
behavior-likelihood compatibility term to `_fit_world`. Tests: hidden Delphic privacy + relabel
(Stage 6). Deps: Stage 1, D1. Runtime: n/a (manifest). Rerun scope: adds Delphic-hidden cells.
Rollback: revert routing line.

**Stage 6 — Privacy & relabel-invariance tests.**
Files: `tests/real/test_hidden_rk.py`. Add: (a) family-relabel invariance for
`refplan/bamcts/ogsrl/delphic` (fit under `kind` relabel → identical fitted arrays); (b) numeric
value-leak assertion (no method-facing array equals `K_base`/`safety_threshold`); (c) manifest/path
ordering independence from species order. Deps: Stages 2–5. Runtime: <1 min. Rerun: tests only.
Rollback: remove added tests.

**Stage 7 — Paper-mechanism unit tests (small constructed problems).**
Files: new `tests/real/test_general_paper_mechanisms.py`. Cases: RefPlan marginalization
(2-model toy), OGSRL constraint binding, BA-MCTS belief update / distinctness, Delphic `u_d`→0 at
`σ_obs=0` and ↑ with ambiguity (extend existing synthetic test). Deps: Stages 2–4. Runtime: <2 min.
Rollback: remove file.

**Stage 8 — Return-blind data/HP checks.** Files: new `scripts/general_adequacy_probe.py` (writes
to a *new* `real_ecology_runs/general_adequacy_*/` dir only). Emits §6.2 diagnostics. Deps: Stages
2–5. Runtime: minutes/cell. Rerun: probe only, **no headline artifacts**. Rollback: delete dir.

**Stage 9 — Deterministic runtime canaries.** Reuse `tests/real/test_corrected_canary_tooling.py`
pattern for the four methods (seed→same action; wall/mem bound). Deps: Stages 2–4. Runtime: minutes.

**Stage 10 — Snapshot/digest/registration.** Copy corrected `src/` into
`real_ecology_runs/general_corrected_<date>/code/`, write `registration.json` (methods, budgets,
seeds, HP, code hashes). Deps: Stages 2–7 green. Rollback: delete dir.

**Stage 11 — Matched manifests at 4000.** Files: `scripts/make_hidden_rk_manifest.py` (or a new
`make_general_corrected_manifest.py`) routing `refplan,ogsrl,bamcts,delphic` (+ diagnostic
references if D5) at 4000/25, matched seeds. Deps: Stage 10. Runtime: n/a. Rollback: revert.

**Stage 12 — Naming decision application.** Update `docs/benchmark/04_algorithm_adaptations_and_claims.tex`
reader-facing labels per §criteria below. Deps: all. Rollback: revert doc.

**Keep-name vs `-inspired` criteria:** keep paper name only if the **minimum identity core** (§2) is
present in the *mode being run*; else `-inspired`. Under this rule (pre-fix): all four →
`-inspired`. Post-fix: RefPlan/OGSRL may claim "faithful-in-spirit adaptation" of their core;
BA-MCTS keeps name only if Stage 4-A lands; Delphic stays `-inspired` unless ELBO-compatible worlds
are added.

---

## 8. Tests and acceptance gates

| Gate | Assertion | Stage |
|---|---|---|
| G1 provenance | hidden general changes committed + hashed; unrelated dirty tree untouched | 1 |
| G2 RefPlan identity | hidden action responds to posterior; ≠ MOPO on constructed cell | 2,7 |
| G3 OGSRL identity | hidden actor trained; safety/OOD constraint binds | 3,7 |
| G4 BA-MCTS | distinct from RefPlan; (A) belief update concentrates correctly | 4,7 |
| G5 Delphic | `u_d`→0 at σ_obs=0, ↑ with ambiguity; hidden privacy passes | 5,6,7 |
| G6 privacy | forbidden-name + value-leak + relabel invariance (all 4) | 6 |
| G7 adequacy | action-coverage & holdout-RMSE reported; sink prevalence known (N6) | 8 |
| G8 canary | deterministic seed→action; within wall/mem | 9 |
| G9 no-returns | none of G1–G8 reads `operational/true` return fields | all |

---

## 9. Runtime estimate (summary)
CPU-only; per-cell fit seconds–tens-of-seconds; eval 1–15 min (BA-MCTS dominant). Full corrected
hidden arm for the four ≈ **1–2 CPU-days** wall with row-packing; sensitivity arms (BA-MCTS 512,
Delphic `|W|=30`, budget 8000) add a comparable amount and should be scheduled separately. No GPU.
Fits under `m3h`/`MaxArraySize=1001` with the Phase-1 row-packing recipe.

---

## 10. User decision table

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Is Delphic scientifically meaningful here? | (a) retain as **Delphic-inspired** + add to manifest; (b) exclude | **(a)** — privileged-behavior confounding instantiates the ambiguity set; fix world-compatibility + rename |
| D2 | Must BA-MCTS have in-tree belief update + outer loop? | (a) add Eq-4 in-tree update, keep name; (b) keep root sampling, rename **BAMCP-inspired**; (c) add outer PI distillation too | **(a)** for identity (cheap); outer PI loop **not** required for an eval-time baseline |
| D3 | Lightweight shared models acceptable for the primary comparison? | (a) yes + disclose shared-model caveat + RMSE reporting; (b) per-method models | **(a)** for deadline; report dynamics RMSE so misspecification is visible |
| D4 | Add BC and/or CQL/IQL? | (a) as **contextual diagnostic references** only; (b) as retained members; (c) none | **(a)** — return floor + model-free context, not part of the four |
| D5 | Compute-scaled HP + sensitivity budget | (a) primary {BA-MCTS 256/d8, Delphic |W|=20} + small sens.; (b) paper-near (512+/|W|30+) primary | **(a)** primary, **(b)** as preregistered sensitivity only |
| D6 | Report every retained method as paper-inspired? | (a) yes for all four now; (b) keep names for those passing identity core post-fix | **(a)** now; revisit per §criteria after fixes |

---

## 11. Final verdict per method

| Method | Verdict | Basis |
|---|---|---|
| **RefPlan** | **fix** (then `-inspired`) | belief+marginalization core exists but is **inert in hidden** (≡ MOPO); Stage-2 fix restores identity; prior policy remains a disclosed simplification |
| **OGSRL** | **fix** (then `-inspired`) | guardian+guarded-model+constrained-actor core present in full but **not run in hidden**; Stage-3 fix restores ConOpt over the surrogate; drop safety-guarantee claims |
| **BA-MCTS** | **fix or rename** (Decision D2) | in-tree Bayes-adaptive update (Eq 1/4) **missing in both modes**; paper rejects the local root sampling for its regime; add Eq-4 (keep name) or relabel BAMCP-inspired |
| **Delphic** | **rename + retain** (`Delphic-inspired`, add to manifest) | uncertainty=Var_wQ and pessimism faithful; confounding genuinely present; worlds not ELBO-compatible ⇒ `-inspired`; `|W|` under budget |
| **MOPO** | **exclude (archive)** | already out of hidden manifest; shares no unique code; do not delete |

**Unresolved:** official-code defaults for RefPlan/BA-MCTS/Delphic (**[UNVERIFIED]** — repos
inaccessible; paper values used). Flag for a later recheck if those repos become available.

---

### Final recommendation
`READY FOR A CORRECTIVE IMPLEMENTATION PLAN` — the deep paper evidence **confirms** the Phase-1
identity defects (and shows the RefPlan/BA-MCTS fixes must be deeper than "attach posterior"/"add a
sensitivity"), while **upgrading Delphic** from "likely mismatched" to "applicable but mislabelled."
The staged plan in §7 is specific enough to implement once the user approves the §10 decisions
(especially D1, D2, D5). **No code, tests, manifests, jobs, or results were modified; no returns were
inspected.**
