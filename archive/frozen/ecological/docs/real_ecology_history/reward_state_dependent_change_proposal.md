# Proposed change: state-dependent reward + two `reward_mode` settings

**Status:** IMPLEMENTED (A + B approved and applied) plus Codex post-audit
hardening. All 16 tests pass (`tests/test_real_ecology.py`: 10 E10 + 4 reward
T1–T4 + 2 cache/P-safe hardening tests). This document is kept as the audit trail:
the sections below describe exactly what was changed and why, for an independent
review against the two updated spec files and the code.

**Scope rule (unchanged):** all edits are confined to
`discrete_action_cont_obser/real_ecology_cont_obser/`. The original
`src/tier2_benchmark/` package is not touched.

---

## 1. Source of truth (the two updated specs)

Both live in `discrete_action_cont_obser/docs/`:

- **`29_6_Real_Ecological_Data_Actions_and_Costs.tex`**, section **"Reward: two
  state-dependent settings"** (Eq. `eq:reward`, lines ~185–227):

  > `R^{(m)}_t = α · s_{t+1}/(s_{t+1}+K_ref(p)) − cost(a_t) − P_m · 1[s_t>s_safe(p) ∧ s_{t+1}≤s_safe(p)]`, with `K_ref(p)=K_base(p)`.
  >
  > "The reward is defined on the **true** latent abundance, not the noisy survey
  > `o_t`: the manager acts on `o_t` but accrues reward from `s`, which the
  > simulator logs, so every method consumes identical tuples
  > `(o_t, a_t, R_t, o_{t+1})`."
  >
  > Two settings, identical benefit and cost, differing **only** in the penalty:
  > `R_yield` (`P=0`, baseline-style, matches PLUS/MOOR) and `R_safe` (`P>0`,
  > collapse-aware). "A separate agent is trained under each setting … their raw
  > returns are **not** compared … Both are evaluated on a common, reward-agnostic
  > battery — persistence / final abundance, quasi-extinction (collapse)
  > probability, minimum abundance, fraction of steps with `s≤s_safe`, and
  > economic cost."

- **`29_6_Real_Ecology_Setting_Implementation_Plan.tex`**, transition "reward:"
  line (L171) and paragraphs **(E6)** (L233–252) and **(E6′)** (L254–273), plus
  acceptance test **(E10.4)** (L310–314):

  > (E6) "Benefit on the true state. Use `α·s_{t+1}/(s_{t+1}+K_ref)`, **not**
  > `o_t/(o_t+K_ref)`." Four reasons: coherence with the true-state collapse term;
  > baseline comparability (PLUS/MOOR reward the latent state); no σ-noise confound
  > (`E[o|s]=s·e^{σ²/2}`); conservation semantics.
  >
  > (E6′) "Expose a `reward_mode` flag; both variants use the **same** benefit and
  > cost and differ **only** in the penalty weight `P_m`. … Train a **separate**
  > agent (per method) under each mode. **Do not compare their raw returns** …
  > evaluate **both** on a common, reward-agnostic battery … Cross this reward
  > factor with the recoverability split of E9 and sweep σ within each cell."
  >
  > (E10.4) "`reward_mode=yield` forces `P_yield=0`, `safe` uses `P_safe>0`; both
  > log identical benefit (`s_{t+1}`-based, never `o_t`) and cost; the evaluation
  > battery is computed independently of `reward_mode`."

---

## 2. What is already correct (no change)

- Per-population safety floor `s_safe(p) = 0.1·K_base(p)`
  (`config.py:safety_threshold = safety_fraction*K_base`, default 0.1). ✅
- Collapse indicator `1[s_t>s_safe ∧ s_{t+1}≤s_safe]`
  (`envs.py` `entered_now`, lines ~302–307). ✅
- `K_ref(p)=K_base(p)`, fixed per episode (`config.real_environment`). ✅
- Dynamics, action table, cost table, filter, seed protocol. ✅

## 3. What is WRONG vs. the updated spec (must change)

1. **Benefit uses the observation, and the wrong time index.**
   `envs.py:309` computes the environment reward from `o_t` via
   `reward_model.operational(observation_prev, …)`, and the eval-only
   `reward_true` (`envs.py:310`) uses `s_t` (`state_prev`). The spec requires a
   single reward on the **true next state** `s_{t+1}` (`state_next`).
2. **No `reward_mode ∈ {yield, safe}`.** The env has a single
   `collapse_penalty`; the `reward_mode` field exists but means
   `{observed, belief_expected}` (a dead Tier-3 C2 flag, never wired).
3. **Eval battery incomplete.** `evaluator.py` logs return / collapse /
   `unsafe_fraction` / `mean_true_state`, but not `economic_cost`,
   `min_true_state`, `final_true_state`, or `persistence`.
4. **The planners/critics reconstruct the benefit from the observation**, so even
   after fixing the env reward the five planning methods would optimize the
   obs-based proxy and would not respond to `reward_mode` (see Part B).

---

## 4. Proposed changes

Two parts. **Part A** is the env/dataset/eval reward (unambiguously required).
**Part B** aligns the method/planner internal reward so `reward_mode` actually
reaches all seven methods and the σ-noise confound is removed end-to-end.

### Part A — env / dataset / eval reward (core, required)

#### A1. `reward.py` — add a state reward, a penalty switch, and a factory

Current (`reward.py:12–42`):

```python
@dataclass(frozen=True)
class ContinuousReward:
    actions: tuple[ActionSpec, ...]
    alpha: float = 1.0
    K_ref: float = 500.0
    collapse_penalty: float = 20.0

    def utility(self, abundance): ...
    def operational(self, observation, action, entered) -> float: ...  # utility(o)
    def true(self, state, action, entered) -> float: ...               # utility(s)
    def expected(self, observations, actions, entered): ...
```

Add:

```python
    def state_reward(self, next_state, action, entered) -> float:
        """R = alpha * s'/(s'+K_ref) - cost(a) - collapse_penalty*entered."""
        value = float(self.utility(next_state)) - self.actions[action].cost
        if entered:
            value -= self.collapse_penalty
        return float(value)
```

New module-level helpers (import `resolve_actions` from `.actions`, cfg type from
`.config`):

```python
def effective_collapse_penalty(cfg) -> float:
    # yield => P=0; safe (and any non-real mode) => configured penalty.
    if cfg.control_mode == "real_setpoint" and cfg.reward_mode == "yield":
        return 0.0
    return cfg.collapse_penalty

def build_reward(cfg) -> ContinuousReward:
    return ContinuousReward(
        resolve_actions(cfg), cfg.alpha, cfg.K_ref, effective_collapse_penalty(cfg)
    )
```

Import-cycle check: `reward → {actions, config}`; `actions → realdata`
(+ TYPE_CHECKING config); `config → realdata`. Neither `actions` nor `config`
imports `reward`, so no cycle.

#### A2. `config.py` — `reward_mode ∈ {yield, safe}` for real cells

- Field default (`config.py:75`): `reward_mode: str = "observed"` → `"safe"`
  (the dataclass default `control_mode` is already `"real_setpoint"`, so a bare
  `EnvironmentConfig()` must carry a real-valid `reward_mode`).
- `validate()` (`config.py:109–110`): accept the union
  `{"yield", "safe", "observed", "belief_expected"}` (real semantics only act on
  `yield`/`safe`). This avoids breaking `MechanisticProposal`'s
  `EnvironmentConfig(**overrides)` round-trips.
- `real_environment(...)`: set `reward_mode` in the `values` dict (default
  `"safe"`, overridable). `reward_mode` is **already** in `_CARRYOVER_FIELDS`
  (`config.py:277`), so `real_environment_like` and the CLI `--population` switch
  preserve it.

#### A3. `envs.py` — build the reward via the factory; reward on `s_{t+1}`

- `ContinuousEcologyEnv.__init__` (`envs.py:70`):
  `self.reward_model = ContinuousReward(self.actions, cfg.alpha, cfg.K_ref, cfg.collapse_penalty)`
  → `self.reward_model = build_reward(cfg)`.
- `step()` (`envs.py:309–310`):

  ```python
  # before
  reward = self.reward_model.operational(observation_prev, action_id, entered_now)
  reward_true = self.reward_model.true(state_prev, action_id, entered_now)
  # after (real setting)
  reward = self.reward_model.state_reward(state_next, action_id, entered_now)
  reward_true = reward
  ```

  Guarded by `control_mode == "real_setpoint"`; the non-real branch keeps the old
  operational/true split so nothing else in the vendored package shifts.

Consequence: `result.reward` (accumulated by the evaluator as
`operational_return`) and `reward_true` are now identical and both state-based.
The offline `dataset.rewards` (collector appends `result.reward` unchanged at
`collector.py`) become state-based automatically — no collector edit. The public
schema guard is unaffected (reward is a public scalar; `states`/params remain
private).

#### A4. `evaluator.py` — reward-agnostic battery

In the per-episode loop (`evaluator.py:92–153`) accumulate:

- `economic_cost += env.actions[action].cost` (per step),
- `min_true_state = min(min_true_state, s_{t+1})`,
- `final_true_state = last true state`,
- `persistence = int(final_true_state > env_cfg.safety_threshold)`.

Add these to the row dict (`evaluator.py:158–195`) and to the `numeric` tuple in
`summarize` (`evaluator.py:203`). All are functions of true states / actions
only, hence independent of `reward_mode` (test A/T4). `collapse_entry`
(quasi-extinction) and `unsafe_fraction` already exist.

#### A5. configs / CLI / manifest

- `configs/real_default.yaml`, `configs/real_smoke.yaml`: add
  `environment.reward_mode: safe`.
- `cli.py`: add `--reward-mode {yield,safe}` to the common parser; apply as an
  env override in `_config`.
- `manifest.py:make_manifest`: add a `reward_mode` axis (`["safe","yield"]`) and a
  `reward_mode` column, so cells cross reward_mode × population × family × σ
  (E6′). `evaluator` should also record `reward_mode` per row; `aggregate` key
  gains `reward_mode` so the two settings are never pooled.

### Part B — align method/planner internal reward (required for E6′ to hold)

Rationale: the environment/dataset reward is only half the loop. Five methods
(MOPO, RefPlan, BA-MCTS, MOOR, PLUS) plan with an **internally reconstructed**
reward; two (Delphic, and OGSRL's critic) read `dataset.rewards`. If B is skipped:
(i) the five planning methods keep optimizing `utility(o)` (the σ-confound E6
forbids), and (ii) they ignore `reward_mode` entirely, so "train a separate agent
per mode" (E6′) is a no-op for them. B changes only the **benefit input**
(observation → predicted next latent state) and the **penalty wiring**; no method
structure/logic changes.

| File:line | Current | Proposed |
|---|---|---|
| `planning.py:108` | `step_reward = self.reward.utility(current_observations) - costs` | `... = self.reward.utility(next_states) - costs` (belief-expected next-state benefit). `current_observations` plumbing at `planning.py:79,143` becomes dead → remove. |
| `methods/value.py:44,48` | `expected_obs = states*exp(0.5σ²)`; `immediate = reward.utility(expected_obs) - cost` | `immediate = reward.utility(next_states) - cost` |
| `methods/ogsrl.py:159,161` | `expected_obs = states*exp(0.5σ²)`; `reward = utility(expected_obs) - costs - P*entry` | `reward = utility(next_states) - costs - P*entry` |
| `methods/bamcts.py:92,93` | `observation_mean = state*exp(0.5σ²)`; `reward = self.reward.operational(observation_mean, a, entry)` | `reward = self.reward.state_reward(next_state, a, entry)` |
| 6 `ContinuousReward(...)` sites: `methods/{mopo:32, refplan:46, bamcts:44, moor:85, plus:47, ogsrl:188}` + `gate.py:80` | `ContinuousReward(resolve_actions(cfg), α, K_ref, cfg.collapse_penalty)` | `build_reward(cfg)` — so the planner penalty honors `reward_mode` (P=0 under `yield`). |

Note the collapse-penalty terms in `planning.py`/`ogsrl.py`/`value.py` read
`self.reward.collapse_penalty`, which becomes `effective_collapse_penalty(cfg)`
once the reward models are built via `build_reward` — automatically 0 under
`yield`.

---

## 5. Invariants preserved

- **Same tuples for every method:** `dataset.rewards` = `R_t(s,a,s')` is written
  once by the collector; all methods read the same public
  `(o_t, a_t, R_t, o_{t+1})`. (I1 leakage: reward is a public scalar; true
  `states`/params stay private and guarded by `assert_public_schema`.)
- **Dynamics/action/cost/filter/seed protocol unchanged.**
- **`reward_mode` differs only in `P_m`;** benefit and cost identical between
  settings (E6′). Raw returns across modes are never compared — the manifest keys
  and aggregation separate them, and the headline is the reward-agnostic battery.

## 6. Test plan (`tests/test_real_ecology.py`)

- **T1 benefit at capacity:** with `s_{t+1}=K_base`, benefit `= α/2`
  (⇒ `state_reward(K_base, a0, False) = 0.5 - cost(a0) = 0.5`).
- **T2 reward tracks `s_{t+1}`, not `o_t`:** hold the true next state fixed
  (deterministic map from a chosen `state_override`, action `a0`) and vary
  `observation_noise_sigma ∈ {0, 0.4}`; `result.reward` is identical (depends on
  `s'`, not the survey).
- **T3 `reward_mode` switch:** on a step that crosses `s_safe`,
  `yield` reward − `safe` reward `= P_safe` (i.e. `yield` adds no penalty, `safe`
  subtracts `collapse_penalty`); away from a crossing the two are equal.
- **T4 battery is reward-mode-agnostic:** run one fixed policy (constant action)
  under `reward_mode=yield` and `reward_mode=safe`; assert
  `min_true_state / final_true_state / economic_cost / collapse_entry /
  unsafe_fraction` are identical (same trajectory), while `operational_return`
  differs iff a crossing occurred.
- Existing E10 tests (env reproduces data; caps; N0 start; collapse band; schema
  guard) must still pass; the abundance-term test updates to assert the reward's
  benefit uses `s_{t+1}`.

---

## 7. Open scope question for the reviewer

**Recommendation: implement A + B together.** B is what makes the two
`reward_mode` settings meaningful for all seven methods and removes the
σ-dependent obs-noise confound that E6 explicitly eliminates. B touches the shared
planner + four method internals, but only the benefit input (obs → predicted
state) and the penalty wiring — no change to method structure.

Alternative: **A only** — env/dataset/eval reward become state-based, but the five
planning methods keep optimizing the obs-based proxy and do not respond to
`reward_mode`. Not recommended (E6′ would be a no-op for those methods), listed
for completeness.

## 8. File-touch summary

Part A: `reward.py`, `config.py`, `envs.py`, `evaluator.py`, `manifest.py`,
`cli.py`, `configs/real_default.yaml`, `configs/real_smoke.yaml`,
`tests/test_real_ecology.py`.
Part B: `planning.py`, `methods/value.py`, `methods/ogsrl.py`,
`methods/bamcts.py`, `methods/mopo.py`, `methods/refplan.py`, `methods/moor.py`,
`methods/plus.py`, `gate.py`.
All inside `real_ecology_cont_obser/`. No file outside the subpackage is modified.
