# Server Handoff — New Experiment Setting: Hidden-Dynamics (Option B)

**For:** the next implementation agent (fresh chat, has repo access, does NOT have this
conversation's history)
**Prereq reading, in order:** `SERVER_HANDOFF_motivation_experiment.md` (the original task),
`SERVER_RESULTS_motivation_native_run.md` (the negative result), `SERVER_RESULTS_general_method_audit.md`
(the model-limited verdict), `09_motivation_native_results.tex` (the standalone report),
`SERVER_METHOD_ADAPTATIONS_motivation_native.md` (what the methods are).

This file specifies a **new benchmark setting**, not a re-run of the old one. It exists because the
completed motivation experiment answered its question — and the answer exposed a flaw in the
*environment*, not in any method.

---

## 1. Why a new setting is needed (the diagnosis, in one paragraph)

The motivation experiment asked whether general offline MBRL beats native ecological solvers under
structural uncertainty. **It found the opposite (natives win 860/864), but the result does not test
the intended question, because the current environment hands the native solvers a near-exact model.**
Confirmed three independent ways in this project:

1. **The generals are not under-tuned.** Search-budget audit: 4× budget closes only 21 % of the gap,
   and *deeper search makes them worse* (h5 > h10 > h20 for all three) — the fingerprint of a wrong
   *model*, not a shallow *search*. (`SERVER_RESULTS_general_method_audit.md`.)
2. **`plus_native` is a form-oracle.** Its posterior over the four mechanistic forms concentrates on
   the *true* form within one 50-step episode (P = 0.90–0.999). It does not face structural
   uncertainty; it *resolves* it. (`09_motivation_native_results.tex`, §Methods.)
3. **The leak is in the code, not a subtlety.** The environment generates dynamics from
   `r_setpoint(family)` and `dK_step` in `real_ecology_data/action_effects_long.csv`; the native
   solvers read the **same values** via `native_solver.assumption_config → config.real_environment_like
   → realdata.RealPopulation.caps`. The general methods only ever see the 4000-transition offline
   dataset (`ContinuousDynamicsEnsemble.fit`). **The natives get the generative parameters for free;
   the generals must estimate them.** That informational asymmetry — not planning, not search — is
   what the natives win on.

So the current benchmark measures *"exact mechanistic model + tabular solver vs learned model +
planner."* The natives should win that, and they do. **The paper's premise is untested.**

---

## 2. The job, in one paragraph

Build a **hidden-dynamics** variant of the real-ecology setting in which the per-action growth
set-points (and, in the stronger variant, the capacity effects) are **latent**: the environment
still generates from the true per-action parameters, but **no solver may read them from the tables.
Every method — general and ecological — must estimate its dynamics model from the same public
offline dataset.** This removes the native oracle and makes parameter + structural uncertainty
genuinely bite, which is what the paper claims to test. Keep the environment, action set, reward
modes, and evaluation protocol otherwise identical, so results remain comparable to the completed
motivation run.

---

## 3. **Phase 0 — a 50-CPU-h falsification BEFORE you build anything**

Do not build the new environment until you have run this. It is cheap and it can save the whole
effort.

**The model-source ablation.** Give the *general* methods the mechanistic model the natives get, and
see whether the gap closes.

- Add `model.dynamics_source ∈ {learned, ricker, true_family}` and, inside each general method's
  `fit()`, swap `ContinuousDynamicsEnsemble` for a `MechanisticProposal` when the source is not
  `learned`. (`refplan.py:40`, `bamcts.py:49`, `ogsrl.py:300` are the fit sites; `MechanisticProposal`
  already exists in `beliefs.py` and is what the natives/adapted PLUS use.)
- **Gate it with a bit-identical no-op proof** (like G1 in the search-budget audit): at
  `dynamics_source=learned` the three methods must reproduce the completed motivation run **to the
  last decimal**, or the change is not inert and the run is invalid.
- Run on the 72 safe cells (or all 144), reusing the frozen datasets. ~50 CPU-h, ~15 min at 240 cores.

**Interpretation, decided in advance:**

- **If the gap closes** (generals ≈ natives once they get the mechanistic model) → the deficit *is*
  the model. The hidden-dynamics redesign in §4 will work: hide the model from the natives and the
  gap closes from the other side. **Proceed to build.**
- **If the gap does NOT close** → something other than the model explains it (planning, reward
  shaping, the discretization advantage). Hiding the model from the natives would *not* rescue the
  motivation, and §4 is a waste. **Report that and stop.**

This is the only experiment that tells you whether §4 is worth building. Run it first.

---

## 4. The new setting — design (primary is settled; two knobs are open)

**Primary design (recommended, minimal, preserves comparability): hidden per-action parameters.**

Everything the environment *generates* stays the same. What changes is what the solver's *model
builder* is allowed to read.

- **Environment:** unchanged. It still reads `action_effects_long.csv` / `species.csv` and generates
  true dynamics from `r_setpoint`, `dK_step`, `K_base`. The truth is unchanged so trajectories stay
  comparable.
- **Offline dataset:** unchanged schema (`PUBLIC_FIELDS` + `PUBLIC_CONTROL_FIELDS`); still the only
  thing any solver sees. The dataset already contains everything needed to *estimate* the per-action
  effects — `(observation, action, next_observation, rho, kappa)` — because `rho`/`kappa` are public
  controls. Estimation is genuinely possible from data; it is just not free.
- **The one real change — the native model builder must FIT, not READ:**
  - `moor_native` currently reads `r_setpoint` from the table (via `assumption_config →
    real_environment_like → caps`) and only grid-searches `K`. It must instead **estimate the
    per-action growth effect from the offline `(o,a,o')` data**, exactly as it already estimates `K`.
  - `plus_native` must do the same per candidate form, so its form posterior is scored on
    *estimated-parameter* models, not oracle-parameter models. Expect its form-identification edge to
    shrink — that shrinkage is the point.
  - The general methods are **unchanged** — they already estimate everything from data. That is the
    equalization.

**Open knob 1 — how much is hidden.** Two levels, pick one (or run both):
  - **(a) growth only:** hide `r_setpoint`; leave `K_base`/`dK_step` public. Smaller change, isolates
    the growth-rate leak, which is the dominant one (6/11 actions are growth-only).
  - **(b) growth + capacity:** hide both. Fuller test; larger build. Recommend (a) first as the
    minimal fix, then (b) if (a) still leaves the natives ahead.

**Open knob 2 — does the solver know the candidate FORM set?** This is the definition of "structural
uncertainty" and it is a scientific call, not an engineering one:
  - **known form set** (Ricker/Allee/theta/regime) with unknown parameters → tests *parameter*
    uncertainty within a known structural family. `plus_native` still enumerates forms but must fit
    each. Milder.
  - **unknown / wrong form set** (e.g. give the solver only Ricker, or a form not containing the
    truth) → tests genuine *structural* misspecification. Harder, and closest to the paper's claim.
  - **Recommendation:** run the known-form-set version first (it is the honest minimal change and
    `plus_native` already enumerates forms). Escalate to wrong-form-set only if the paper needs the
    stronger structural claim.

Flag both knobs to the PI before building; they change what the result *means*.

---

## 5. Exact code paths

| concern | file / symbol |
|---|---|
| **the leak to close** | `native_solver.py :: assumption_config` (line ~35) → `config.py :: real_environment_like` → `realdata.py :: RealPopulation.caps` (line 67). This chain hands the native solver the exact `r_min/r_max/r_setpoint`. |
| native model fit (already estimates K) | `methods/moor_native.py :: fit` (grid-searches K via `predict_ricker_next`); extend to estimate the per-action growth effect too. |
| native candidate bank | `methods/plus_native.py :: fit` (builds a `NativeSolver` per form via `NativeSolver.build`). Each must fit params from data. |
| general model fit (the reference — do not change) | `dynamics.py :: ContinuousDynamicsEnsemble.fit`; called at `refplan.py:40`, `bamcts.py:49`, `ogsrl.py:300`. |
| Phase-0 ablation hook | add `model.dynamics_source` to `config.py :: ModelConfig`; branch in the three general `fit()`s; `MechanisticProposal` already in `beliefs.py`. |
| public vs private data (leak boundary) | `dataset.py :: PUBLIC_FIELDS`, `PUBLIC_CONTROL_FIELDS`, `PRIVATE_FIELDS`. The solver may read public + control; never private. |
| env generation (leave truth intact) | `envs.py :: transition_value`; `controls.py :: private_r_eff`, `advance_public_controls`. |
| manifest / runner / output tags | `scripts/run_real_manifest_row.py` (has `--dataset-root`, planner overrides, `config_tag` output path — reuse the pattern for a `dynamics_source` / `hidden_level` tag); `scripts/make_general_audit_manifest.py` is the template. |
| aggregation | `manifest.py :: aggregate_summaries` (cross-filter path); `general_audit_20260712/analyze_budget.py` is a clean per-cell template. |
| reward / eval protocol (do not change) | `reward.py :: build_reward`; `EvaluationConfig` defaults (5 seeds × 4 episodes × horizon 50). |

---

## 6. Invariants / guardrails (each is a way to silently invalidate the run)

1. **The environment generates from the true parameters. Never estimate the truth.** Hiding is a
   *solver-side* restriction; the env is untouched. If you change the env's generation, you break
   comparability with the completed run and with the reward calibration.
2. **The native model must be fit from the SAME public dataset the generals use.** Log a
   `model_source` field per row and assert it is `estimated` (not `table`) for the hidden-dynamics
   rows. A leak here silently restores the oracle and the whole experiment is void.
3. **No private-field access.** The solver reads `PUBLIC_FIELDS` + `PUBLIC_CONTROL_FIELDS` only. Add a
   test that the native fit does not touch `PRIVATE_FIELDS` (true `r_base`, `C`, `theta`, `regime`,
   true state).
4. **Leak-proof gate (analogous to G1/G2).** Before the sweep: (a) assert the hidden-dynamics native
   fit produces *different* parameters than the table would give (if identical, hiding did nothing);
   (b) assert both sides share one `dataset_sha256` per cell; (c) assert the *unhidden* path still
   reproduces the completed run bit-for-bit (so the hiding is the only change).
5. **Do not touch the eval protocol.** 5 seeds × 4 episodes × horizon 50 is locked for comparability
   with the motivation run and every prior audited run.
6. **Do not tune toward the desired answer.** This experiment can *rescue* the paper's premise, which
   is exactly why the result must be pre-registered and reported whichever way it falls. If the
   natives *still* win with estimated parameters, that is a real, publishable finding (mechanistic
   inductive bias helps under data scarcity) — not a failure to be tuned away.
7. **CPU-only.** No method here has a working CuPy path (`backend.py`: GPU covers only
   `mechanistic_transition` / `method:plus` / `oracle_ablation:plus`). Requesting GPU buys nothing and
   queues behind fair-share. `MaxSubmit=1000`, `MaxArraySize=1001`, `cpu=250` on this cluster;
   per-array `%` caps do NOT pool (size each to its own cost); size `--time` against the *worst* task,
   not the mean. Measure per-config cost with a canary before sizing the sweep — the analytic cost
   model was 1.38× off last time and qualitatively wrong for OGSRL.

---

## 7. Acceptance tests

- [ ] Phase-0 ablation run, with a bit-identical no-op proof at `dynamics_source=learned`, and its
      pre-registered reading recorded.
- [ ] Native model builder fits per-action growth from the offline data; `model_source=estimated`
      logged; fitted params provably differ from the table values (guardrail 4a).
- [ ] Leak test: native fit touches no `PRIVATE_FIELDS` (guardrail 3).
- [ ] Both sides share one `dataset_sha256` per cell (guardrail 4b).
- [ ] Unhidden path reproduces the completed motivation run bit-for-bit (guardrail 4c).
- [ ] `plus_native` form posterior under hidden params is measured and reported — expect it to be
      *less* concentrated than the 0.90–0.999 oracle case; quantify the shrinkage.
- [ ] Full grid runs with 0 failures, 0 silent fallbacks (`fallback_count == 0`), native metrics in
      the existing schema.
- [ ] Result reported per-σ, per-family, recoverable-vs-sink, never pooled across reward mode.

---

## 8. First smoke test

The single most diagnostic cell, run before any sweep:

> **Amur tiger / Ricker / σ = 0 / safe, hidden-level (a) growth-only, known-form-set.**

Ricker is where the native assumption is *correct*, so any failure here is an implementation bug, not
the phenomenon. Check:

1. `moor_native` fits a per-action growth effect from data, `model_source=estimated`, and the fitted
   value is close-but-not-equal to the table `r_setpoint` (estimation works, and it is genuinely
   estimating).
2. `fallback_count == 0`; belief weights finite; action histogram not a point mass.
3. `operational_return` is **lower** than the oracle `moor_native` on the same cell from the motivation
   run (5.671) — because it no longer reads the exact rate. If it is *equal*, the hiding did not take
   effect; the leak is still open.
4. The general methods (`refplan` etc.) are **unchanged** on this cell vs the motivation run
   (bit-identical), since only the native path changed.

Then repeat on **Iberian lynx / Allee / σ = 0.2** (where the forms most disagree) before trusting any
noisy or misspecified cell.

---

## 9. First actions (ordered)

1. **Read the five prereq docs.** The negative result and its diagnosis are the entire reason this
   file exists; do not re-derive them.
2. **Run Phase 0** (§3). If the ablation says the gap is not model-driven, stop and report.
3. Confirm the leak chain in §5 by tracing `assumption_config → real_environment_like → caps` in the
   live code, and confirm the generals read only the dataset.
4. Get the PI's call on the two open knobs (§4).
5. Implement the native model-fitting path + `model_source` logging + the leak guardrails (§6).
6. Canary one cell, measure per-config cost, run the §8 smoke, check the §7 gates.
7. Size the sweep from the *measured* cost (not a model), launch CPU-only, aggregate with the
   cross-filter path, report per-σ / per-family / recoverable-vs-sink.

---

## 10. What this run can conclude

- **If, with estimated parameters, the general methods catch or beat the natives** → the original
  motivation is *recovered*: the natives only won because the old environment leaked the model. The
  paper's premise holds in a fair setting. This is the outcome the paper hoped for, now earned rather
  than assumed.
- **If the natives still win with estimated parameters** → mechanistic structure is a genuinely strong
  inductive bias under data scarcity, even without oracle parameters. That is a *different* and still
  publishable finding — and it means the paper's original framing ("naive ecological solvers are
  fragile") is simply false for this domain, and should be dropped rather than redesigned around.

Either way, report it straight. The one outcome that must not happen is tuning the setting until it
produces the desired ordering.
