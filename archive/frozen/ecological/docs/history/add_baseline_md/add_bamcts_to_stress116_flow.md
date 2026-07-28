You are a rigorous Principal Research Software Engineer working in `claude_build/`.
Goal: run **BA-MCTS (ICLR'26, "Bayes-Adaptive Monte Carlo Tree Search for Offline
Model-based RL")** as a 5th method in the **stress-116 final benchmark**, matched
apples-to-apples against MOPO, RefPlan, PLUS, MOOR.

---

## 0. READ THIS FIRST — the situation is not what it looks like

**The BA-MCTS adapter already exists and is registered. Do NOT reimplement it.**

- Adapter: `claude_build/src/models/bamcts_adapter.py` → class `BayesAdaptiveMCTSAdapter`
- Config: `claude_build/config/model/bamcts.yaml`
- Registry: `src/models/__init__.py` registers key `"bamcts"`
- Tests: `claude_build/tests/test_general_mbrl_baselines.py`

It implements the full `BaseModel` interface (`fit_offline`, `predict`,
`select_action`, `update`, `evaluate`, `save`, `load`) as a tabular BA-MCTS and is
already used by the generic experiment harness. **It was simply never wired into
the stress-116 benchmark flow** (`stress_pomdp_116.sh` / `make_stress116_manifest.py`
/ `aggregate_stress116_phase116.py`), which currently runs only `mopo, refplan,
plus, moor_ricker`.

So your job is: **(A) fix one real correctness blocker, (B) wire `bamcts` into the
stress-116 dispatch + aggregator, (C) run 60 matched cells on CPU reusing the
existing datasets, (D) re-aggregate.** Do not rebuild the algorithm.

**PDF note:** the paper PDF could not be auto-extracted in this environment — no
`pdftotext`/poppler and no `pymupdf`/`pdfplumber`/`PyPDF2`. To read it yourself,
`pip install pymupdf` (or `apt-get install poppler-utils`) then extract. The
algorithm summary below is reconstructed from the adapter docstring + config +
standard BA-MCTS; confirm against the PDF if you want, but the adapter is already a
faithful finite-state adaptation.

---

## 1. Algorithm core (BA-MCTS) and how the repo realizes it

Paper core: train a probabilistic **dynamics ensemble** from the offline data; treat
the unknown true model as a latent variable so planning happens in a **Bayes-Adaptive
MDP** where the agent maintains a **belief over which ensemble member is the true
model**, updated by Bayes' rule as transitions are observed *within* the search;
plan with **MCTS (PUCT/UCB)** over belief-augmented states; add an **ensemble
target-disagreement pessimism penalty** for offline conservatism; (optionally distill
search into a policy/value net).

Repo realization (finite-state, faithful — already in `bamcts_adapter.py`):

| Paper concept | Repo mechanism |
|---|---|
| Probabilistic dynamics ensemble | `TabularDynamicsEnsemble` (bootstrapped Dirichlet count models), `src/models/tabular_mbrl.py` |
| Belief over models (BAMDP state) | `posterior_from_history()` / `update_posterior()` (Bayes update `b'(m) ∝ b(m)·P_m(s'|s,a)`) |
| MCTS in belief-augmented space | `_simulate_tree()` with node key `(state, depth, rounded belief)` (`_node_key`, `tree_belief_decimals`) |
| PUCT/UCB action selection | `_select_tree_action()` (UCB1 with `exploration_c`) |
| Ensemble pessimism penalty | `_ensemble_target_penalty()` = `pessimism_lambda · std_m( R + γ·P_m·V_m )` |
| Belief-weighted bootstrap/terminal value | `_terminal_value()` = `belief @ V_members`, member `V` from `value_iteration` |

Hyperparameters live in `config/model/bamcts.yaml`: `ensemble_size=15`,
`mcts_simulations=128`, `mcts_depth=5`, `exploration_c=1.25`, `pessimism_lambda=0.10`,
`tree_belief_decimals=2`, `discount=0.95`. For the baseline run, **use these defaults**
(treat BA-MCTS like PLUS/MOOR — a fixed, untuned config). Tuning is optional and out
of scope unless requested.

---

## 2. ⚠️ BLOCKER — BA-MCTS will CRASH under `collapse_sensitive` as written. Fix first.

The stress-116 headline mode is `collapse_sensitive`, where the env exposes a
**next-state** reward via `env.planning_reward_fn` → a `NextStateRewardFn` instance
with `uses_next_state = True` (see `src/environments/stress_pomdp_envs.py`). Two
problems in `bamcts_adapter.py`:

**2a. (BLOCKER) `_validate_reward_fn` rejects the next-state reward_fn.**
`env.planning_reward_fn` returns a **new** `NextStateRewardFn` object on every access,
so the instance passed to `fit_offline` differs (by `id`) from the one the evaluator
passes to `select_action`. BA-MCTS's `_validate_reward_fn` then recomputes a reward
table with `dataset=None` for a `uses_next_state` fn — which falls back to
`state/num_states` — and raises *"BA-MCTS reward_fn does not match…"*. RefPlan avoids
this by **early-accepting any `uses_next_state` reward_fn**; BA-MCTS does not.

→ **Fix:** make `BayesAdaptiveMCTSAdapter._validate_reward_fn` mirror
`RefPlanAdapter._validate_reward_fn` (`src/models/refplan_adapter.py`): add at the top,
after the `None` check —
```python
if bool(getattr(reward_fn, "uses_next_state", False)):
    self._reward_fn_id = id(reward_fn)
    return
```
Without this, **every collapse_sensitive cell fails** (base mode would still run).

**2b. (ALSO REQUIRED — fairness blocker) rollout reward ignores the actual next state.**
In `_simulate_tree` (`bamcts_adapter.py:299–304`), BA-MCTS computes
`reward = self._reward(state, action, reward_fn)` *before* sampling `next_state`, and
`_reward` (`bamcts_adapter.py:338`) calls `reward_fn(state, action, state)` — passing
`state` as next_state. For `collapse_sensitive` this penalizes *being in* a low bin
instead of *entering* one — different semantics from RefPlan/PLUS/MOOR, which apply the
entry penalty on the true next bin ("option (a)"). **Do not treat this as optional:**
reorder `_simulate_tree` to **sample `next_state` first**, then compute `reward` with
the real next bin (mirror `RefPlanAdapter._rollout_sequence` / `_reward`, which pass the
actual `next_state`). Without it, BA-MCTS plans against a *different* reward than the
other four methods, so its collapse_sensitive numbers are not comparable and the
headline claim would be biased. (Base mode is unaffected — `reward_fn` there is the
plain `RewardContract`.)

**Verify the fix with a single-cell smoke before launching all 60** (see §6).

---

## 3. Wire `bamcts` into the stress-116 flow (3 edits)

### 3.1 Method dispatch — `scripts/slurm/stress_pomdp_116.sh`, `run_method()`
Add a `bamcts)` case alongside `moor_ricker)` (BA-MCTS is CPU/tabular → `device=cpu`;
`num_actions` must follow the env for 5a/10a cells). It has `fit_offline`, so
`run_pipeline.py` handles it without the `env=/env_cfg=` kwargs that only PLUS/MOOR need:
```bash
        bamcts)
            mkdir -p "$OUT_ROOT/bamcts/ds" "$OUT_ROOT/bamcts/ck" "$OUT_ROOT/bamcts/eval"
            run_monitored bamcts \
                env WANDB_RUN_GROUP="$WB_GROUP" WANDB_NAME="stress116_${ENV_KEY}_${REWARD_MODE}_bamcts_seed${SEED}" \
                "$PYTHON" scripts/core/run_pipeline.py \
                "${COMMON_OVERRIDES[@]}" \
                "+dataset_path=$DATASET_PATH" \
                device=cpu \
                active_model=bamcts \
                model=bamcts \
                "model.num_actions=$NUM_ACTIONS" \
                "hydra.run.dir=$OUT_ROOT/hydra_bamcts" \
                "paths.dataset_dir=$OUT_ROOT/bamcts/ds" \
                "paths.checkpoint_dir=$OUT_ROOT/bamcts/ck" \
                "eval.output_dir=$OUT_ROOT/bamcts/eval" \
                eval.csv_name=bamcts.csv \
                eval.json_name=bamcts.json
            ;;
```

### 3.2 Aggregator — `scripts/analysis/aggregate_stress116_phase116.py`
**Decision (locked): fold BA-MCTS into the headline "learned beats baselines" claim**
as a third general method. Do all of:
- Add to `METHOD_LABELS` (≈ line 20): `"bamcts": "BA-MCTS",` — plain name, matching the
  learned-method convention (`RefPlan`/`MOPO` carry no venue tag). Do **not** label it
  "BA-MCTS (ICLR26)": the adapter is an explicit finite-state adaptation, not the
  paper's full neural stack (continuous-control, reward models, double progressive
  widening, policy distillation), so a venue tag would overclaim a reproduction.
  Confirm the report's methodology section flags the general methods as tabular
  adaptations (it already does for RefPlan); add BA-MCTS to that note.
- Add `"bamcts"` to the learned-method list in **BOTH** hardcoded masks (they are
  separate — patching one is not enough; verified in code):
    - `_claim_tests` win-fraction block (≈ line 224):
      `learned = g[g["method"].isin(["mopo", "refplan", "bamcts"])]`
    - `_attach_advantage` mask (≈ line 177):
      `mask = g["method"].isin(["mopo", "refplan", "bamcts"])`
  Leave the baseline set `["plus", "moor_ricker"]` unchanged in both. (There is **no**
  generic "not baseline" path — `_attach_advantage` only fills `learned_advantage` for
  rows in its explicit mask, so BA-MCTS gets an advantage column only after you add it
  there.)
- Update the **claim-description text** so it no longer says "MOPO or RefPlan":
  change the summary/`.tex` strings (≈ lines 284–286 and 383–390) from
  *"at least one learned method (MOPO or RefPlan) beats both fixed-family baselines"*
  to *"…(MOPO, RefPlan, or BA-MCTS)…"*. The metric itself (`best_learned >
  best_baseline`, where `best_learned = max` over the three) is unchanged in form;
  since `max` can only rise, the existing 1.0 win fraction will hold or improve — but
  the claim now legitimately covers all three general methods.
- **Also report BA-MCTS on its own footing** so the fold-in is honest, not just a
  `max` trick: add a per-method "beats both baselines" rate (i.e., for each of mopo /
  refplan / bamcts, the fraction of cells where that single method > best baseline).
  This shows whether BA-MCTS individually clears the baselines or only helps via the
  max. Put it in the summary alongside the primary fraction.

### 3.3 Manifest — `scripts/experiments/make_stress116_manifest.py`
No code change required: `--methods` has no allow-list, and `_tag()` returns the bare
method name → `config_tag="bamcts"`. (Optional: append `,bamcts` to the `--methods`
default at line 246 for convenience.) The unused `mopo_*`/`refplan_*` columns ride
along harmlessly.

**Do NOT touch** `claude_build/.wandb_env` (git-ignored secret).

---

## 4. Run it — 60 matched cells, CPU, reuse existing datasets

The 60 `final116` shared datasets already exist
(`outputs/stress_pomdp_116/shared_datasets/final116/*.npz`), so BA-MCTS needs **no
dataset phase and no `afterok`** — submit it straight to the `comp` CPU pool with
codex's CPU worker (`stress_pomdp_116_manifest_worker_cpu.sh`, `partition=comp`,
`cpus-per-task=8`, no GRES). Matched grid = 6 cells × 2 reward modes × 5 held-out
seeds 7001–7005 = 60 rows.

```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/claude_build
FROZEN=outputs/stress_pomdp_116/frozen/phase116_frozen_1day.env

# (1) BA-MCTS-only method manifest (60 rows)
python scripts/experiments/make_stress116_manifest.py \
    --kind methods --phase final116 \
    --output outputs/manifests/stress116_final_methods_bamcts.tsv \
    --seeds 7001,7002,7003,7004,7005 \
    --reward-modes collapse_sensitive,base \
    --methods bamcts \
    --data-n 75000 --eval-episodes 50 --eval-horizon 50 \
    --frozen-env "$FROZEN"
rows=$(($(wc -l < outputs/manifests/stress116_final_methods_bamcts.tsv)-1))   # expect 60

# (2) submit on the CPU pool (no GPU). normal QOS = 250 CPU/user; 8 cpus/task → %30.
sbatch --qos=normal --partition=comp --array=0-$((rows-1))%30 \
    --export=ALL,BASE="$PWD",WB_MODE=online,WB_PROJECT=deeprl_population_models,WB_GROUP=stress_pomdp_116_final,REQUIRE_GATE_PASS=true,REQUIRE_DATASET=true \
    scripts/slurm/stress_pomdp_116_manifest_worker_cpu.sh \
    outputs/manifests/stress116_final_methods_bamcts.tsv
```
Expected wall ≈ **~1–1.5 h** (BA-MCTS ≈ RefPlan order, ~30–90 min/run on CPU, ~30 wide).
Gate summaries already pass for all 6 cells, so `REQUIRE_GATE_PASS=true` is fine.

---

## 5. Re-aggregate over all 5 methods

The aggregator reads methods from whatever manifest it's pointed at. Build a combined
300-row manifest (original 240 + 60 BA-MCTS) and re-run the CPU report:
```bash
cd /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/claude_build
cp outputs/manifests/stress116_final_methods.tsv outputs/manifests/stress116_final_methods_with_bamcts.tsv
tail -n +2 outputs/manifests/stress116_final_methods_bamcts.tsv >> outputs/manifests/stress116_final_methods_with_bamcts.tsv
# rows should be 300
python scripts/analysis/aggregate_stress116_phase116.py \
    --manifest outputs/manifests/stress116_final_methods_with_bamcts.tsv \
    --output-dir outputs/stress_pomdp_116/reports --tag phase116_final_bamcts
```
Outputs land at `outputs/stress_pomdp_116/reports/phase116_final_bamcts_{summary.md,aggregate.csv,seed_level.csv,hidden_buckets.csv}`.
Confirm each of the 60 BA-MCTS runs produced `…/bamcts/eval/bamcts.csv` with **exactly
50 rows** (mirror the per-run check used for the other 240) before trusting the table.

---

## 6. Tests / smoke (run before the 60-cell launch)

1. Existing unit tests: `pytest tests/test_general_mbrl_baselines.py -q`. **Coverage gap
   to know:** these exercise BA-MCTS only at `num_actions ∈ {2, 4}` (generic),
   fit/predict/select/evaluate/save/load, and the lower-cost / tree-reuse behaviors.
   They do **not** cover the 10-action env or the next-state (collapse) reward — the
   file's 10-action test targets RefPlan, not BA-MCTS. So existing green ≠ stress-116
   readiness.
2. **Add two targeted BA-MCTS unit tests** for the gaps above (in
   `tests/test_general_mbrl_baselines.py`):
   - **10-action:** construct `BayesAdaptiveMCTSAdapter` with `num_actions=10`, fit a
     synthetic 10-action dataset, assert `select_action` returns `0 ≤ a < 10`, and that
     `validate_action_space` passes against a 10-action env (`num_actions=10`).
   - **Next-state reward:** pass a `uses_next_state=True` reward_fn (a small stub, or a
     `collapse_sensitive` stress env's `env.planning_reward_fn`) to `fit_offline`, then a
     **different instance** of the same to `select_action`; assert **no `ValueError`**
     (locks in §2a) and that a collapse penalty actually lowers the penalized action's
     value (locks in §2b).
3. **Reward-fix smokes** — one cell each, tiny budget; both must finish without the
   `reward_fn` ValueError:
   ```bash
   # 5-action (model.num_actions defaults to 5 in bamcts.yaml → matches the 5a env)
   python scripts/core/run_pipeline.py env=allee_ricker_pomdp_116 active_env=allee_ricker_pomdp_116 \
       env.reward.mode=collapse_sensitive active_model=bamcts model=bamcts device=cpu \
       +dataset_path=outputs/stress_pomdp_116/shared_datasets/final116/allee_ricker_pomdp_116_collapse_sensitive_seed7001_n75000.npz \
       eval.n_episodes=3 model.mcts_simulations=16 model.mcts_depth=3 wandb.mode=disabled
   # 10-action — MUST pass model.num_actions=10. run_pipeline does NOT infer it from the
   # env, and bamcts.yaml defaults to 5 → without this, action-space validation fails.
   python scripts/core/run_pipeline.py env=allee_ricker_pomdp_116_10a active_env=allee_ricker_pomdp_116_10a \
       env.reward.mode=collapse_sensitive active_model=bamcts model=bamcts device=cpu \
       model.num_actions=10 \
       +dataset_path=outputs/stress_pomdp_116/shared_datasets/final116/allee_ricker_pomdp_116_10a_collapse_sensitive_seed7001_n75000.npz \
       eval.n_episodes=3 model.mcts_simulations=16 model.mcts_depth=3 wandb.mode=disabled
   ```

---

## 7. Definition of done
- **Both §2a and §2b fixes applied** (the reward_fn early-accept AND the next-state
  rollout-reward reorder); the two new unit tests (10-action, next-state reward) added
  and passing; both smokes pass.
- `bamcts` dispatch case added; aggregator updated in **all three** spots — label,
  `_claim_tests` mask (line 224), and `_attach_advantage` mask (line 177).
- 60 BA-MCTS runs COMPLETED, all with 50-episode eval CSVs, both reward modes, 5 seeds.
- Combined 5-method report regenerated; BA-MCTS appears with its label and is
  **folded into the learned set** (§3.2): the headline claim now reads "MOPO, RefPlan,
  or BA-MCTS beats both PLUS and MOOR," and the summary reports each general method's
  own beats-baselines rate.
- Report which methods win under collapse_sensitive now that BA-MCTS is included, and
  whether it changes the headline (currently: learned MOPO/RefPlan beat PLUS/MOOR in
  1.0 of cells; MOOR collapses, PLUS fails on theta). State explicitly whether BA-MCTS
  individually clears both baselines or only contributes via the `max`. Keep `base`
  framed as the degenerate/decision-irrelevant mode.

Scope: edits confined to `claude_build/` (+ this guide). Do not modify
`DeepRL_Population_Models/` (the clean repo) or `.wandb_env`.
