# Zero-standard-deviation feature rule — read-only design memo

Date: 2026-08-08 (Australia/Melbourne)
Author: Claude (read-only rule design; no implementation)
Trigger: I2A correctly stopped — no controlling document selects an exact transformation for a feature that becomes constant under exact-state Arm T.
Environment: all probes run with `PYTHONDONTWRITEBYTECODE=1`, `LC_ALL=C`; no bytecode written, no truth array opened, no dataset constructed.

Collision check: `CLAUDE_ZERO_SD_RULE_MEMO.md` did not exist. No file was overwritten.

---

## 1. Source trace

### 1.1 The feature vector

`BeliefState.public_features(observation_scale)` — `src/tracks/general/real_ecology_benchmark/types.py:121-151`. Ten float64 columns, produced in this fixed order:

| Idx | Name | Source line | Units | Meaning |
|---:|---|---|---|---|
| 0 | `mean` | `types.py:134` | log1p(abundance / `observation_scale`), dimensionless | weighted mean of transformed particles |
| 1 | **`sd`** | `types.py:135,144` | same log1p space | `sqrt(max(var,0))` — **the feature at issue** |
| 2 | `q10` | `types.py:131-133` | log1p space | 10th percentile of particles |
| 3 | `q50` | `types.py:131-133` | log1p space | median |
| 4 | `q90` | `types.py:131-133` | log1p space | 90th percentile |
| 5 | `extinct` | `types.py:136` | probability | `sum(w[states == 0.0])` |
| 6 | `ctx_prev_obs` | `types.py:137-140` | **raw survey units** | previous observation |
| 7 | `ctx_obs` | `types.py:137-140` | **raw survey units** | current observation |
| 8 | `ctx_t` | `types.py:137-140` | steps | timestep |
| 9 | `ess_frac` | `types.py:148` | ratio | `ess / n` |

Particles come from `PublicObservationFilter._particles` (`beliefs.py:595-604`); weights are uniform by construction (`beliefs.py:625`).

### 1.2 What actually happens under Arm T — measured, not assumed

Under a point-mass Arm T belief all particles equal the true abundance, so from `types.py:127-144`: `x` is a single repeated value, `mean = sum(w*x)`, `var = sum(w*(x-mean)^2)`, and all three quantiles select the same element.

I replicated that arithmetic exactly in float64 across both cells' abundance ranges:

| `observation_scale` | state | `x` | computed `sd` | `sd == 0.0`? |
|---:|---:|---|---:|:--:|
| 61.40286116280366 | 0.02 | 0.000325664686486 | 0.000e+00 | **True** |
| 61.40286116280366 | 1.07 | 0.017275808235665 | 0.000e+00 | **True** |
| 61.40286116280366 | 30.10 | 0.398913808521292 | **5.551e-17** | **False** |
| 61.40286116280366 | 119.50 | 1.080503775733605 | 0.000e+00 | **True** |
| 61.40286116280366 | 359.17 | 1.924161304274940 | **2.220e-16** | **False** |
| 20.0 | 359.17 | 2.942252379773154 | **4.441e-16** | **False** |

**The premise that the feature becomes *exactly* zero is false.** It is exactly `0.0` for some abundances and up to **4.441e-16** (≈ 2–4 ulp) for others, because `mean = sum(w*x)` is not bit-exactly `x` for every `x`. Any rule or unit test keyed on `sd == 0.0` would pass on round-numbered synthetic fixtures and silently misclassify real rows. This is the single most important finding in this memo.

Verified separately: the uniform weights *are* exact — `exp(-log n)` normalised gives `w[0] == 1.0/n` exactly and `sum(w) - 1.0 == 0.0`. The residue therefore originates solely in the `sum(w*x)` reduction, not in the weights.

### 1.3 Other columns that go constant or collinear

Measured directly on the two **accepted Arm O** belief caches (`public.regime_hidden.learned.beliefs.npz`, shape `(4000, 10)`, dtype float64):

| Idx | tiger std | fox std | Arm O state | Arm T state |
|---:|---:|---:|---|---|
| 0 `mean` | 3.410e-01 | 3.074e-01 | variable | variable |
| 1 `sd` | 3.482e-02 | 3.426e-02 | variable | **→ constant ≈ 0 (≤ 4.441e-16)** |
| 2 `q10` | 2.981e-01 | 2.650e-01 | variable | **→ exactly equals col 0** |
| 3 `q50` | 3.406e-01 | 3.067e-01 | variable | **→ exactly equals col 0** |
| 4 `q90` | 3.849e-01 | 3.511e-01 | variable | **→ exactly equals col 0** |
| 5 `extinct` | **0.000e+00** | **0.000e+00** | **already exactly constant** | unchanged |
| 6–8 context | variable | variable | variable | unchanged (preserved by registration) |
| 9 `ess_frac` | **0.000e+00** | **0.000e+00** | **already exactly constant** | unchanged |

Two consequences the candidate rule does not cover:

1. **Zero-SD columns are not new.** Columns 5 and 9 are *already* exactly constant in the accepted Arm O pipeline, in both cells, and the existing code already handles that case (§3). Arm T adds a third such column; it does not create the condition.
2. **Rank collapse, not just constancy.** The state-derived block `[0..4]` has measured rank **5 of 5** in Arm O; under Arm T it collapses to **rank 1** (cols 2–4 become bit-identical to col 0, col 1 becomes ≈0). A constant-mask rule alone does not address collinearity.

### 1.4 The convention already in the code

`PublicParticlePlanner._proposal_features` — `src/tracks/general/real_ecology_benchmark/public_models.py:180-200` — **already constructs exactly this degenerate vector today**, inside accepted Arm O:

```
189    features = np.repeat(root_features[None, :], len(current), axis=0)
190    x = np.log1p(np.maximum(current, 0.0) / self.context.observation_scale)
191    features[:, 0] = x
192    features[:, 1] = 0.0                 # sd assigned exactly 0.0
193    features[:, 2:5] = x[:, None]        # q10 = q50 = q90 = mean
194    features[:, 5] = (current <= 0.0).astype(np.float64)
195    features[:, 6] = previous; [:, 7] = current; [:, 8] = timestep
```

This is the defensible in-repository convention the task asks for: for a degenerate belief the codebase **assigns** `sd = 0.0` and **assigns** the quantiles equal to the mean. It does not detect, threshold, drop, mask, or noise. Adopting assignment removes the 4.441e-16 residue by construction and requires no invented threshold on the primary path.

---

## 2. Affected methods and features

All six pipelines were checked; the issue is confined to four.

| Method | Consumes `public_features`? | Where | Fitting | Policy training | Online update | Action selection | Affected |
|---|---|---|:--:|:--:|:--:|:--:|:--:|
| **adapted PLUS** | **No** — 0 occurrences in `plus_faithful.py` | uses `CandidateBelief.probabilities` over the abundance grid | – | – | – | – | **No** |
| **adapted MOOR** | **No** — 0 occurrences in `moor_faithful.py` | same | – | – | – | – | **No** |
| **RefPlan** | Yes | `refplan.py:56,63` (prior fit + apply); `public_models.py:216` (planner root) | ✔ | ✔ | – | ✔ | **Yes** |
| **OGSRL** | Yes | `ogsrl.py:630,694` → `_sample_cached_public` (`ogsrl.py:420-425`) reads cols 0 and 1 | ✔ | ✔ | – | – | **Yes** |
| **BA-MCTS** | Indirect | via `PublicDynamicsEnsemble` (uses `mean_states`, not `features`) and the shared cache | ✔ | – | ✔ | ✔ | **Partly** |
| **EVD** | Yes | `ensemble_value_disagreement.py:81-82,113,120,136,182` — all 20 Q members, behavior reference, every action selection | ✔ | ✔ | – | ✔ | **Yes** |

**PLUS and MOOR — the only frozen-fit-eligible methods — are entirely unaffected.** This scopes the whole problem: every affected method is already registered as *not eligible* for the frozen-fit label, i.e. already end-to-end bundle-only. The lone exception is RefPlan's optional secondary frozen-fit diagnostic (`I1_REGISTRATION.json`: `"optional secondary only; eligibility unresolved"`), the only place the frozen-fit branch can ever bind.

Current transformation state of the feature: **raw** as emitted by `public_features` (log1p-scaled, but not centered, standardized, clipped or masked). Standardization happens downstream, per consumer (§3).

Two further Arm T behavioural changes in the same neighbourhood, neither a preprocessing issue but both requiring receipts:

- **OGSRL rollout-start collapse.** `_sample_cached_public` (`ogsrl.py:420-425`) computes `rng.normal(features[:,0], features[:,1])`. With col 1 ≈ 0 the start distribution degenerates to a point. Verified: `rng.normal(loc, 0.0)` returns `loc` **exactly** and **still advances the RNG state**, so no cross-arm stream desynchronisation occurs. Semantically correct under Arm T (exact state ⇒ no start uncertainty), but a component change.
- **RefPlan pre-existing train/runtime offset.** The prior is *fitted* on `beliefs.features` where col 1 is variable, but *applied* at runtime to `_proposal_features` where col 1 is hard-set to `0.0`. Measured standardized runtime value of that coordinate today: **tiger −2.7018, fox −2.7348**. Under Arm T the fitted column also becomes constant, so this coordinate moves to **0.0**. That is a ≈2.7σ shift in RefPlan's proposal input caused by the preprocessing regime, *not* by state information. It must be receipted or RefPlan's arm contrast will be partly an artefact.

---

## 3. Current preprocessing behaviour

| Consumer | Site | Statistics fitted | Scope | Zero/near-zero handling | Existing floor |
|---|---|---|---|---|---|
| Behavior reference / RefPlan policy prior | `behavior_model.py:44-48` | mean + std of augmented design | **per method, per cell** | `scale = np.maximum(std, 1e-8)`; intercept forced `mean[0]=0.0, scale[0]=1.0` | **1e-8** |
| Public reward surrogate | `public_surrogate.py:89-90` | mean + std of continuous block | per cell, shared across methods | `std = np.maximum(std, 1e-8)` | **1e-8** |
| OGSRL kNN guardian | `ogsrl.py:210-211` (public), `143-145` (non-hidden) | mean + std of anchor features | per method, per cell | `scale = np.maximum(std, 1e-6)` | **1e-6** |
| EVD Q members | `ensemble_value_disagreement.py:22-23,81-82` | **none** — `augment_features` then `_ridge` | per member | ridge `+ ridge*I` regularises | ridge |
| `PublicDynamicsEnsemble` | `public_models.py:93-118` | none on features; uses `mean_states` | per member | `residual_sigma = max(std(residual), 0.02)` | **0.02** |

Key facts established by measurement:

- **No code path divides by zero or produces NaN.** Every standardizing consumer already carries a floor; EVD relies on ridge. For a constant column, `(c − c)/1e-8` evaluates to **exactly `0.0`** — verified. Existing behaviour is already numerically safe.
- **The hazard is amplification, not failure.** With `scale = 1e-8`, a runtime deviation from the fitted constant is multiplied by 1e8:

  | runtime deviation | z with `scale=1e-8` | z with `scale=1.0` |
  |---:|---:|---:|
  | 1e-12 | 1.0e-04 | 1.0e-12 |
  | 1e-09 | 1.0e-01 | 1.0e-09 |
  | 1e-06 | **1.0e+02** | 1.0e-06 |

  A 1e-6 discrepancy between the fitting and runtime code paths would inject a 100σ feature. This is the real risk, and it is why `scale = 1.0` (the candidate rule) is correct.
- **Training and runtime use the same fitted transform** for the behavior model (`CalibratedBehaviorModel` stores `mean`/`scale`), but **not the same feature constructor** for RefPlan (§2, the −2.70 offset).
- **Frozen-fit Arm T would reuse Arm O preprocessing** byte-for-byte — that is what makes it "frozen-fit".
- **End-to-end Arm T would refit preprocessing** on the Arm T training view, changing the serialized artifact and therefore triggering the existing model-fit-axis label.

**Compatibility constraint (critical).** Columns 5 and 9 already hit the 1e-8 floor in the accepted Arm O runs. Changing the constant-column convention *inside* `behavior_model.py` would leave transformed values identical (both give exactly `0.0`) but would change the serialized `scale` entries, altering the Arm O preprocessing artifact hash and breaking G2/G3 byte-parity against accepted artifacts. **Therefore the new rule must be implemented only in the external Arm T preprocessing artifact, never by editing `src/tracks/**`.** I2 is already forbidden from editing frozen tracks; the addendum must say so explicitly, because "fixing" `behavior_model.py` is the most likely way an implementer breaks parity while believing they are compliant.

---

## 4. Frozen-fit / state-input-only rule — **APPROVED as written, with scope correction**

| Clause | Verdict |
|---|---|
| Reuse the exact Arm O preprocessing artifact byte-for-byte | **APPROVED** — `CalibratedBehaviorModel` already carries `mean`/`scale`; reuse is a load, not a refit |
| Feed the Arm T raw constant value through the unchanged Arm O transform | **APPROVED** — for col 1 this yields the in-range, non-degenerate value `(0 − mean_O)/scale_O` = **−2.7018 tiger / −2.7348 fox**. No division hazard: `scale_O ≈ 3.5e-2`, not the floor |
| Do not refit means, scales, masks or feature order | **APPROVED** |
| Record the transformed constant value | **APPROVED**, strengthened: record per column, not only col 1 |
| Any preprocessing change removes frozen-fit eligibility | **APPROVED** — consistent with the frozen `residual_sigma` rule in `I1_REGISTRATION.md` §5 |

**Scope correction.** PLUS and MOOR never touch `public_features`, so for them this branch is vacuous. It binds **only** on RefPlan's optional secondary frozen-fit diagnostic. The addendum must say this, otherwise an implementer may believe a frozen-fit path exists for OGSRL/BA-MCTS/EVD — it does not.

**Added requirement.** Because the Arm O transform maps the Arm T constant to ≈ −2.70σ, the frozen-fit diagnostic is a genuine out-of-distribution extrapolation of the fitted prior. That must be receipted as an interpretive caveat, not silently reported as "same preprocessing".

---

## 5. End-to-end Arm T rule — **APPROVED with three corrections**

| Clause | Verdict |
|---|---|
| Fit preprocessing on the Arm T training view | **APPROVED** |
| Detect constant/near-constant features using one exact preregistered threshold | **CORRECTED** — see (a) |
| `constant_mask=true`, `offset=c`, `scale=1.0` | **APPROVED**; `offset` defined precisely in (b) |
| transformed value exactly `0.0` | **CORRECTED** — achievable only by explicit assignment, see (b) |
| Do not drop the column or change feature order | **APPROVED** |
| Do not add artificial noise | **APPROVED** |
| Runtime uses the same stored offset, scale and mask | **APPROVED** |
| Runtime value differing beyond tolerance fails closed | **APPROVED**; tolerance fixed in §6 |
| Label the preprocessing artifact changed ⇒ end-to-end bundle only | **APPROVED** — already implied by `I1_REGISTRATION.md` §5 |
| — | **MISSING** — collinearity, see (c) |

**(a) Construct, do not detect, on the primary path.** The Arm T feature constructor must *assign* the degenerate entries exactly as the accepted code already does at `public_models.py:191-193`: `features[:,1] = 0.0` and `features[:,2:5] = features[:,0][:,None]`. This makes the value exactly `0.0` by construction, eliminates the 4.441e-16 residue, and reuses an accepted convention instead of inventing a threshold. Threshold-based detection is retained **only** as a fail-closed guard on the constructed view.

**(b) `offset` and the exact-zero guarantee.** Because a data-fitted column is not bit-exactly constant, `(v − offset)/1.0` cannot be relied upon to give exactly `0.0`. Define: `offset = float64 mean of the fitted column`; `scale = 1.0`; and for any column with `constant_mask=true` the transform **emits the literal `0.0`** rather than computing the subtraction. Exactness then holds by construction, and the stored `offset` remains available for the runtime tolerance check.

**(c) Collinearity must be handled explicitly.** Cols 2–4 become bit-identical to col 0 (rank 5 → 1). Ridge keeps every solve well-posed (`_ridge` adds `ridge*I`; `behavior_model` adds `ridge*weights`; `PublicDynamicsEnsemble` adds a penalty with `penalty[0,0]=0`), so nothing fails — but the coefficient split across collinear columns becomes arbitrary-though-deterministic and must not be interpreted. Require the receipt to record design-matrix rank and condition number per arm, and require that collinear columns are **not** dropped, merged or reordered.

Both rules are compatible with the actual code and with the registered estimand: every affected method is already end-to-end/bundle-only, so applying the changed-preprocessing label costs no eligibility the registration grants.

---

## 6. Exact constants

| Quantity | Value | Justification |
|---|---|---|
| dtype | `numpy.float64` throughout | measured: accepted feature caches are `float64`; every consumer casts with `dtype=np.float64` |
| Constant-detection threshold | `std(column) <= 1e-8` | **existing repository convention**, not invented: `behavior_model.py:45` and `public_surrogate.py:90` both use `np.maximum(std, 1e-8)`. The rule activates on exactly the set where the pre-existing floor would bind |
| Comparison convention | **`<=`** (inclusive) | matches `np.maximum(std, 1e-8)`, whose floor binds when `std <= 1e-8`. Using `<` would leave a measure-zero gap at exactly 1e-8 where old and new paths diverge |
| Margin over numerical noise | 1e-8 vs measured worst-case residue **4.441e-16** | ~7.4 orders of headroom; and 1e-8 is ~3.4e-7 × the smallest real Arm O signal in col 1 (std 3.426e-02), so no genuine variation can be misclassified |
| Fox/tiger scale insensitivity | satisfied | threshold applies in log1p space, where measured values span 0.000326–2.942 across both cells; it is **not** applied to raw abundance (fox min 0.02, tiger max 359.17), so the 6× abundance-scale difference cannot affect classification |
| Relative scaling term | **none** | deliberately absolute. A relative criterion (`std <= rel * abs(mean)`) is unsafe here: col 0 reaches 0.000326 for fox's smallest abundances, where any relative rule becomes ill-conditioned |
| Fitted `scale` for a masked column | **`1.0`** exactly | avoids the measured 1e8 amplification of the existing 1e-8 floor (1e-6 deviation → z = 100.0 vs 1e-6) |
| Transformed constant value | **`0.0`** exactly, by explicit assignment | not by computed subtraction — see §5(b) |
| Runtime constant-value tolerance | `abs(v − offset) <= 1e-6`, else **fail closed** | 9 orders above the 4.441e-16 construction residue; 5 orders below the smallest real Arm O signal (3.426e-02). Absolute, in log1p space, hence cell-scale insensitive |
| Finite-value checks | `numpy.isfinite(...).all()` on every column, fit and runtime; any NaN/±inf aborts | matches `I1_SOURCE_SCHEMA.json` `fail_closed_requirements` item 4 |
| Serialization precision | float64 **round-trip exact**: `repr`/`float(...)` or `numpy.save` binary; never a fixed-decimal string | offset and scale must reload bit-identically or the runtime tolerance check is meaningless |
| Unchanged existing floors | `1e-8` (behavior, surrogate), `1e-6` (kNN guardian), `0.02` (`residual_sigma`), `0.03` (filter sigma) | the addendum introduces **no** change to any existing floor inside `src/tracks/**` |

---

## 7. Offline / runtime parity invariants

Mandatory, all fail closed:

1. **Feature names and order** — identical byte-for-byte between the fitted artifact and every runtime call; the ten-column order of `types.py:121-151` is fixed; no drop, insert, merge or reorder.
2. **Offset, scale, constant mask** — identical arrays; loaded from the serialized artifact, never recomputed at runtime.
3. **dtype** — `float64` at fit and runtime; no float32 downcast anywhere in the path.
4. **Transformation code** — one function, one module, one call site for both fit-time and runtime; no reimplementation.
5. **Serialized preprocessing artifact** — one file; runtime loads exactly the file the fit wrote.
6. **Hash identity** — SHA-256 of the serialized artifact recorded at fit and re-verified at runtime; mismatch aborts.
7. **Fail closed on** — missing column, reordered column, nonfinite value, a column marked constant whose runtime value deviates by more than 1e-6, or a column *not* marked constant whose runtime `std <= 1e-8` (unexpected degeneracy).

How the invariants differ by regime:

| Regime | Preprocessing artifact | Constant mask | Expected transformed col 1 | Label |
|---|---|---|---|---|
| **Arm O (accepted)** | the accepted artifact, unmodified; `src/tracks/**` untouched | cols 5 and 9 only, via the existing `1e-8` floor | variable (fit) / **−2.7018 tiger, −2.7348 fox** at the RefPlan planner root | accepted baseline; must reproduce bit-exactly under G2/G3 |
| **Frozen-fit Arm T** (RefPlan secondary only) | **byte-identical to Arm O**; hash must match | inherited from Arm O; **not** recomputed | **−2.7018 / −2.7348** (OOD extrapolation — receipt it) | frozen-fit eligible only if every artifact hash matches |
| **End-to-end Arm T** | **new** external artifact, refit on the Arm T view | cols 1, 5, 9 masked; `scale=1.0`; cols 2–4 collinear-flagged | **exactly `0.0`** | preprocessing artifact changed ⇒ `MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY` |

---

## 8. Receipt schema

One receipt per (method × species × cell × arm), emitted before returns open.

```json
{
  "schema_version": "i1_zero_sd_preprocessing_receipt_v1",
  "method": "ogsrl",
  "species": "Amur tiger",
  "cell": "amur_tiger__allee__sigma_0p2",
  "arm": "T",
  "regime": "end_to_end",
  "feature_names": ["mean","sd","q10","q50","q90","extinct",
                    "ctx_prev_obs","ctx_obs","ctx_t","ess_frac"],
  "feature_order_hash": "<sha256 of the joined name list>",
  "dtype": "float64",
  "per_column": [
    {"index": 1, "name": "sd",
     "raw_mean": 0.0, "raw_std": 0.0, "raw_min": 0.0, "raw_max": 0.0,
     "constant_mask": true, "offset": 0.0, "scale": 1.0,
     "transformed_constant_value": 0.0,
     "collinear_with": null}
  ],
  "collinear_groups": [[0, 2, 3, 4]],
  "design_rank": 1,
  "design_condition_number": null,
  "constant_threshold": 1e-8,
  "threshold_comparison": "<=",
  "runtime_constant_tolerance": 1e-6,
  "finite_check_passed": true,
  "preprocessing_artifact_sha256": "<sha256>",
  "fit_view_sha256": "<sha256 of the Arm T derived view>",
  "runtime_transform_sha256": "<sha256; must equal preprocessing_artifact_sha256>",
  "arm_o_artifact_sha256": "<sha256 of the accepted Arm O artifact>",
  "frozen_fit_eligible": false,
  "model_fit_axis_label": "MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY",
  "zero_variance_events": [
    {"index": 1, "stage": "fit", "std": 0.0, "action": "masked"},
    {"index": 5, "stage": "fit", "std": 0.0, "action": "masked_preexisting"},
    {"index": 9, "stage": "fit", "std": 0.0, "action": "masked_preexisting"}
  ],
  "parity_result": "PASS",
  "parity_failures": []
}
```

Every listed field is mandatory. `parity_result` must be `PASS` before any scientific execution; `FAIL` is a hard stop under `I1_REGISTRATION.md` §10.

---

## 9. Exact controlling amendment text

The addendum **supplements** and does not overwrite the frozen I1 registration. `I1_REGISTRATION.md`, `I1_REGISTRATION.json`, `SEALED_PREDICTIONS.json`, `I1_SOURCE_SCHEMA.json`, `I1_HASHES.sha256` and `I1_FREEZE_REPORT.md` all keep their existing hashes; the two new files are added alongside, covered by their own manifest entry.

### 9.1 `I1_ZERO_SD_RULE_ADDENDUM.md`

```markdown
# I1 addendum — degenerate (zero-standard-deviation) feature rule

**Status:** ADDENDUM TO THE FROZEN I1 REGISTRATION; SUPPLEMENTS, DOES NOT OVERWRITE
**Supplements:** I1_REGISTRATION.md (SHA-256 d877fbcf74db7b2a0005d74e19f168e52de9b60ee3df613646e3017d21acb4f3)
**Trigger:** I2A halted because no controlling document fixed a transformation for
features that become degenerate under exact-state Arm T.
**Execution status:** NOT AUTHORIZED. This addendum creates no data and no code.

## A1. Scope

Applies only to `BeliefState.public_features` (`types.py:121-151`), consumed by RefPlan,
OGSRL, BA-MCTS and EVD. adapted PLUS and adapted MOOR do not consume it and are
unaffected; their frozen-fit eligibility is untouched.

## A2. Measured facts

Under a point-mass Arm T belief, column 1 (`sd`) is NOT exactly zero: it is 0.0 for some
abundances and up to 4.441e-16 for others. Columns 2-4 become bit-identical to column 0,
collapsing the state-derived block from rank 5 to rank 1. Columns 5 (`extinct`) and 9
(`ess_frac`) are already exactly constant in the accepted Arm O caches for both cells.
No existing code path divides by zero: floors of 1e-8 (behavior model, public surrogate),
1e-6 (kNN guardian) and ridge regularisation already prevent it.

## A3. Construction rule (primary path)

The Arm T feature constructor SHALL assign degenerate entries directly, mirroring the
accepted convention at `public_models.py:191-193`:

    features[:, 1]   = 0.0
    features[:, 2:5] = features[:, 0][:, None]

Detection by threshold is used only as the fail-closed guard in A6. No artificial noise
may be added. No column may be dropped, merged or reordered.

## A4. Frozen-fit / state-input-only Arm T

Reuse the Arm O preprocessing artifact byte-for-byte; verify by SHA-256. Do not refit
means, scales, masks or feature order. Feed the Arm T value through the unchanged Arm O
transform and record the resulting value per column. Any preprocessing change removes
frozen-fit eligibility. This branch binds only on RefPlan's optional secondary
diagnostic; it is vacuous for PLUS and MOOR and unavailable to OGSRL, BA-MCTS and EVD.
Because the Arm O transform maps the Arm T constant to approximately -2.7018 (tiger) and
-2.7348 (fox) standard units, the diagnostic is an out-of-distribution extrapolation of
the fitted prior and MUST be receipted as such.

## A5. End-to-end Arm T

Fit preprocessing on the Arm T training view. For every column with fitted
`std <= 1e-8` store `constant_mask=true`, `offset = float64 fitted column mean`,
`scale = 1.0`, and EMIT THE LITERAL 0.0 rather than computing `(v - offset)/scale`.
Record collinear groups, design rank and condition number. Runtime must load the same
stored offset, scale and mask. The preprocessing artifact is changed, so the result is
labelled `MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY`.

## A6. Constants (all float64)

    constant_threshold                = 1e-8      comparison `<=`
    fitted scale for a masked column  = 1.0
    transformed constant value        = 0.0       (assigned, not computed)
    runtime constant tolerance        = 1e-6      absolute, log1p space
    finite check                      = numpy.isfinite(...).all()
    serialization                     = float64 round-trip exact

1e-8 is the pre-existing repository convention (`behavior_model.py:45`,
`public_surrogate.py:90`); it is ~7.4 orders above the measured 4.441e-16 construction
residue and ~3.4e-7 of the smallest real Arm O signal in column 1. Both thresholds are
absolute in log1p space and therefore insensitive to the fox/tiger abundance scale.

## A7. Prohibition on editing frozen tracks

This rule SHALL be implemented only in the external Arm T preprocessing module.
`src/tracks/**` MUST NOT be edited. Columns 5 and 9 already reach the 1e-8 floor in the
accepted Arm O runs; changing that code would alter the Arm O preprocessing artifact hash
and break G2/G3 byte-parity against accepted artifacts, even though the transformed
values are identical.

## A8. Parity invariants and receipts

The invariants of the controlling memo section 7 and the receipt schema of section 8 are
mandatory. `parity_result` must be `PASS` before any scientific execution; `FAIL` is a
hard stop under I1_REGISTRATION.md section 10.

## A9. Effect on the frozen registration

No frozen I1 artifact is modified. The sealed predictions remain valid and unchanged.
```

### 9.2 `I1_ZERO_SD_RULE_ADDENDUM.json`

```json
{
  "schema_version": "i1_zero_sd_rule_addendum_v1",
  "status": "addendum; supplements and does not overwrite the frozen I1 registration",
  "supplements": {
    "path": "docs/true_noisy_state_real_methods/i1_registration_freeze_20260808/I1_REGISTRATION.md",
    "sha256": "d877fbcf74db7b2a0005d74e19f168e52de9b60ee3df613646e3017d21acb4f3"
  },
  "feature_source": "src/tracks/general/real_ecology_benchmark/types.py:121-151",
  "feature_names": ["mean","sd","q10","q50","q90","extinct","ctx_prev_obs","ctx_obs","ctx_t","ess_frac"],
  "affected_methods": ["refplan","ogsrl","bamcts","ensemble_value_disagreement_pessimism"],
  "unaffected_methods": ["plus_adapted_ricker_only_pbvi","moor_adapted_ricker_misspec_pbvi"],
  "measured_facts": {
    "arm_t_sd_exactly_zero": false,
    "arm_t_sd_max_residue": 4.441e-16,
    "arm_o_already_constant_columns": [5, 9],
    "arm_t_additional_constant_columns": [1],
    "arm_t_collinear_group": [0, 2, 3, 4],
    "arm_o_state_block_rank": 5,
    "arm_t_state_block_rank": 1,
    "existing_floors": {"behavior_model": 1e-8, "public_surrogate": 1e-8, "knn_guardian": 1e-6, "residual_sigma": 0.02},
    "no_division_by_zero_in_existing_code": true,
    "refplan_armo_runtime_z_of_sd": {"tiger": -2.7018, "fox": -2.7348}
  },
  "construction_rule": {
    "mode": "assign, do not detect",
    "precedent": "src/tracks/general/real_ecology_benchmark/public_models.py:191-193",
    "assignments": {"col_1_sd": 0.0, "cols_2_3_4": "equal to col 0"},
    "artificial_noise_permitted": false,
    "column_drop_or_reorder_permitted": false
  },
  "frozen_fit_rule": {
    "reuse_arm_o_artifact_byte_for_byte": true,
    "verify_by_sha256": true,
    "refit_permitted": false,
    "record_transformed_value_per_column": true,
    "binds_only_on": "refplan optional secondary diagnostic",
    "vacuous_for": ["plus_adapted_ricker_only_pbvi","moor_adapted_ricker_misspec_pbvi"],
    "ood_extrapolation_receipt_required": true,
    "any_preprocessing_change_removes_eligibility": true
  },
  "end_to_end_rule": {
    "fit_on_arm_t_view": true,
    "constant_mask_criterion": "fitted std <= 1e-8",
    "offset": "float64 fitted column mean",
    "scale": 1.0,
    "transformed_value": 0.0,
    "transformed_value_is_assigned_not_computed": true,
    "record_collinear_groups_rank_condition": true,
    "runtime_uses_stored_offset_scale_mask": true,
    "label": "MODEL-FIT AXIS CHANGED — END-TO-END BUNDLE ONLY"
  },
  "constants": {
    "dtype": "float64",
    "constant_threshold": 1e-8,
    "threshold_comparison": "<=",
    "relative_scaling_term": null,
    "runtime_constant_tolerance": 1e-6,
    "finite_check": "numpy.isfinite(...).all()",
    "masked_column_scale": 1.0,
    "transformed_constant_value": 0.0,
    "serialization": "float64 round-trip exact"
  },
  "parity_invariants": [
    "identical feature names and order",
    "identical offset, scale and constant mask",
    "identical dtype float64",
    "single shared transformation code path",
    "identical serialized preprocessing artifact",
    "sha256 identity between fitted and runtime transform",
    "fail closed on missing, reordered, nonfinite, out-of-tolerance or unexpectedly degenerate features"
  ],
  "frozen_tracks_edit_permitted": false,
  "receipt_schema_version": "i1_zero_sd_preprocessing_receipt_v1",
  "sealed_predictions_affected": false,
  "authorization": {
    "i2a": "not authorized by this addendum",
    "data_created": false,
    "code_created": false
  }
}
```

### 9.3 Future I2A authorization text

```text
I2A is authorized to design and implement the degenerate-feature transformation
described in I1_ZERO_SD_RULE_ADDENDUM.md, in a NEW external module outside
src/tracks/**, against SYNTHETIC fixtures only.

Permitted: adapter and preprocessing design; implementation in the new external module;
synthetic fixtures reproducing the ten-column float64 schema; reading truth METADATA
(key set, dtype, shape, public_dataset_sha256); unit tests, including negative tests that
must fail closed on a reordered column, a nonfinite value, an out-of-tolerance runtime
constant, and an unexpectedly degenerate column; emission of the receipt from synthetic
inputs.

Prohibited: editing src/tracks/** or any accepted artifact; opening real truth arrays;
constructing a real derived overlay; regression; rebaseline; scientific evaluation;
Slurm submission.

I2A must stop and return an independent-audit artifact demonstrating: source-only
frozen-track hashes unchanged; git diff --exit-code clean; every parity invariant
implemented; every negative test failing closed; and a synthetic-input receipt whose
transformed constant value is exactly 0.0 and whose parity_result is PASS.
```

---

## 10. Impact on I1 and the sealed predictions

**No frozen I1 artifact requires modification.** The addendum is additive. All six frozen hashes stand: `I1_REGISTRATION.md` `d877fbcf…`, `I1_REGISTRATION.json` `d4a9b99a…`, `SEALED_PREDICTIONS.json` `82a7fcb3…`, `I1_SOURCE_SCHEMA.json` `4de0b45f…`, `I1_HASHES.sha256` `5c483370…`, `I1_FREEZE_REPORT.md` `e335019a…`. All were re-verified during this pass and are unchanged, as are Revision 3.1 (`701b4509…`), both frozen tracks (historical and source-only) and `git diff --exit-code`.

**The sealed predictions remain valid and unchanged.** Checked against all eight:

- The rule creates no numerical prediction and changes no tiered outcome.
- **P7** already states that any non-identical fitted artifact restricts the result to end-to-end bundle interpretation. A refitted Arm T preprocessing artifact is exactly such an artifact, so the end-to-end branch of this rule is *already* covered by the sealed set — no widening.
- **P1/P2** are unaffected: PLUS and MOOR do not consume the feature.
- **P3** (RefPlan decisive control) gains a caveat that must be receipted rather than predicted — the measured −2.70σ preprocessing offset — which is an interpretive requirement, not a prediction change.
- **P8** (no causal claim) is reinforced.

**One consequence for the registration's reach.** Because every method that consumes this feature is already registered as ineligible for the frozen-fit label, adopting the changed-preprocessing label costs no eligibility the registration grants. The only live frozen-fit exposure is RefPlan's optional secondary diagnostic, and §4 keeps that branch byte-identical to Arm O.

**Residual item for the I2A audit, not for I1.** The RefPlan −2.7018 / −2.7348 offset is a *pre-existing* Arm O condition (the prior is fitted on variable `sd` but applied to a hard-zeroed `sd` at the planner root). Arm T removes it. That is a genuine cross-arm change unrelated to state information and must appear in RefPlan's component-change receipt; it does not require a plan revision.

---

## Verdict

**PASS — EXACT ZERO-SD RULE READY FOR I1 ADDENDUM**
