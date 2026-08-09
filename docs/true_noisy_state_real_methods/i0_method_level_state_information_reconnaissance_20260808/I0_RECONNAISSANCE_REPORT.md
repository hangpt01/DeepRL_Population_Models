# I0 method-level state-information reconnaissance

Date: 2026-08-08 (Australia/Melbourne)  
Scope: read-only I0 reconnaissance for the Revision 2 method-level state-information pilot  
Controlling plan SHA-256: `238f238fca99b5dd043856ef90f5eb86b757e9a39249b25bf67eb235bc5cdf3f`

## 1. Executive verdict

I0 is complete, but the pilot is not ready for implementation or execution. The requested local evidence is sufficient to design I1/I2, and the two accepted datasets contain no short episodes. The accepted anchors and the two-track provenance are recoverable. Two material discrepancies prevent a pass:

1. Revision 2 says that OGSRL, BA-MCTS and EVD do not consume `observation_noise_sigma`. Their policy files do not reference it directly, but their accepted hidden-state pipeline does: all three receive cached and runtime `BeliefState` objects from `PublicObservationFilter`, whose particle construction uses `MethodContext.observation_noise_sigma`. The same indirect dependency applies to RefPlan in addition to RefPlan's direct posterior-update dependency.
2. Revision 2 describes an already prepared branch-isolated OGSRL short-episode fix and three regression tests. The current tree still raises on any episode shorter than the configured cost horizon. No fix branch/commit, three tests, or prior passing receipt exists in the local refs or repository evidence inspected at I0.

The existing exact-state infrastructure is only partial. `OracleStateFilter` and evaluator hooks exist, but the normal factory forbids oracle filtering when `expose_rk="hidden"`, and the oracle-ablation runner rejects hidden methods. PLUS/MOOR therefore need a deliberate direct-belief adapter; RefPlan and the other general methods need an explicit exact-state contract. No adapter was implemented.

No accepted or frozen artifact was modified. No test, experiment, evaluation, dataset creation, or Slurm submission was performed.

## 2. Provenance

### Repository identity

| Item | I0 result |
|---|---|
| Requested server path | `/fs04/scratch2/ce25/DeepRL_Population_Models` |
| Resolved repository path | `/fs04/scratch2/ce25/DeepRL_Population_Models` |
| `/home` path resolution | `/home/hphung/ce25_scratch2/DeepRL_Population_Models` resolves to the same physical path |
| Git HEAD | `77cd38adb11970d56ce4c96bf84de15fac3108ac` |
| Branch | `e1-phase1-parity`, tracking `origin/e1-phase1-parity` |
| Initial dirty status | Clean; `git status --short --branch` contained only the branch/tracking line |
| Local refs relevant to fix search | `e1-phase1-parity` and `main`; no OGSRL short-episode fix branch/ref found |

There were no pre-existing working-tree changes to attribute. The only I0 changes are the new files in this new planning-only directory.

### Frozen tracks

The track hashes use this deterministic recipe: hash every regular file with SHA-256 in null-delimited sorted path order, then SHA-256 the resulting checksum stream. Paths are intentionally part of the checksum stream as emitted by `sha256sum`.

| Track | Files | Aggregate SHA-256 |
|---|---:|---|
| `src/tracks/ecological` | 149 | `951365d7874a417d7e66b14538dc275a9f325ac4643df8ea1af4d1d24877fb01` |
| `src/tracks/general` | 162 | `614524d7b058418a3ff3f370e7e5b57582213c2c518766d8d88f1ec093e7a35e` |

The repository's frozen-track declaration, `provenance/frozen_tracks.sha256`, hashes to `319cd42884da74cbdd54b228ecf4fbb11688e40293d376a333bf1b3be6ffcaaf`. The repository documentation correctly treats these as separate tracks and does not authorize merging them.

### Dependency and environment identity

| Identity | Value |
|---|---|
| Registered paper-faithful Python | 3.10.14 |
| Registered numerical stack | NumPy 2.2.6; PyTorch 2.13.0+cpu; single-thread OpenBLAS 0.3.29 DYNAMIC_ARCH in the accepted diagnostic receipt |
| `requirements-paper-faithful.txt` SHA-256 | `e3da565735841215d6d987a67278e26ac7b3aba2f5afee0876879ec512f6a2a2` |
| `pyproject.toml` SHA-256 | `032bb9db59b03e73058e5cdeae898fd14c78ecea8802c61b7d38bbf7b8c49fca` |
| Accepted paper-faithful venv | Python 3.10.14; NumPy 2.2.6; PyTorch 2.13.0+cpu; pytest 9.1.1; complete `pip freeze --all` stream SHA-256 `92f510968afa46085b24bd8b5d7c171257997963a7d99623fdcd0e41d760df69` |
| Local review venv used only to inspect NPZ records | Python 3.10.14; NumPy 2.2.6; pandas 2.3.3; PyTorch 2.13.0+cpu; pytest 9.1.1; complete freeze stream SHA-256 `27f2c85b6c36937b80314bf44b38d26a88bff13bf1aa77ee2b9c7e8633d614f9` |
| Login-shell interpreter | `/usr/bin/python`, Python 3.9.25; not the registered execution environment |

The two Python 3.10 venvs have byte-identical `pyvenv.cfg` files (`0730a27f751d19f35e908e605120b964bd39136089196f27f9ddc12e90553da7`). The differing freeze hashes are recorded rather than normalized away.

### CPU architecture

The present host is x86_64, Intel Xeon Gold 6548Y+. The accepted strict replay profile is x86_64 Intel Xeon Platinum 8452Y on `m3h101`, with `--constraint=xenon-8452Y`; the accepted CPU profile hash is `db6dc5a6d79b1c21c111bdd54ae68dfbdca2e9b19362f2e4039d3fa5bdf3d4b6`. A read-only `sinfo` query confirmed currently visible 8452Y nodes `m3h100`/`m3h101` and AMD EPYC 9454 nodes in `m3h`. Existing evidence also records a successful A6 MOOR login-node run on the 6548Y+, but strict BLAS-level replay should use the 8452Y. The accepted diagnostic receipt explicitly records that EPYC 9454 did not meet the `1e-9` strict parity criterion. Its SHA-256 is `204d2a176689385a6061775c938f80e6446005f5415a96b4f092c30e870369f8`.

### Exact accepted tiger × Allee and fox × Allee artifact locations

Aliases used below:

- `ECO=/fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/three_species_ecological_p10_correction_20260723_v1`
- `GEN=/fs04/scratch2/ce25/general_rl_phase2_iso/real_ecology_runs/general_phase2e_full_sigma01_02_20260720_v1/quarantine`

Primary accepted tables:

- `results/accepted/MATCHED_P10_144_METHOD_CELLS.csv` — `7431318803e468c13ff3008acdc530c0d9f9a919cc29a6fe04951fc8a2cadc1c`
- `results/accepted/MATCHED_P10_144_METHOD_CELLS_RECEIPT.json` — `a198c70f34ed0191a6580c49d52ab675ea7b6768f87076994775b223624d3163`

Accepted datasets and shared general-track inputs:

| Cell/artifact | Exact location | SHA-256 |
|---|---|---|
| Tiger public dataset | `$GEN/datasets/regime_hidden/reward_safe/amur_tiger/allee/sigma_0p2/public.npz` | `9e9c3a6d4bcee3c27c785f46ce17b66e1c9099c446213f46ebcfa3c3851f771c` (embedded dataset hash `7e71172af4bc95b2c31291414af53371c1ea94393e01675d269b2b7c360324c9`) |
| Tiger cached beliefs | same directory, `public.regime_hidden.learned.beliefs.npz` | `cf725f9647bea0d9346ef1578aa6c9311a2c0773abbd191945c4cd4da7b4142a` |
| Tiger reward surrogate | same directory, `public.regime_hidden.public_surrogate.v2.seed20116.npz` | `be0f835b7f0a9e8c3c63ca681998acc900678b08776bc6277b50bdfdd4d5da7c` |
| Fox public dataset | `$GEN/datasets/regime_hidden/reward_safe/crab_eating_fox/allee/sigma_0p2/public.npz` | `7d1b2fc8a1949dda6f53f39eff1dbfe220719d0cbad9f8956ab7ca2d08e87e19` (embedded dataset hash `688d580f1e47a61dcad9b8a6cb96f1d62835bfbd0b835554cf31c8ab8bdd0c13`) |
| Fox cached beliefs | same directory, `public.regime_hidden.learned.beliefs.npz` | `3a2d621089e79d178ec92ad27dc106d9f48bb5d8a98bac9b5a7e38379b96af6a` |
| Fox reward surrogate | same directory, `public.regime_hidden.public_surrogate.v2.seed20116.npz` | `4d9305a94509e2bb4eacd4b25b8ac4ba138a29e269dd240eba81684b1ecc7a4b` |

Accepted ecological episodes:

| Cell/method | Exact location below `$ECO` | SHA-256 |
|---|---|---|
| Tiger PLUS | `plus/evaluation/regime_hidden/amur_tiger/allee/sigma_0p2/plus_three_species_matched_p10_v1/data_real/backend_numpy/regime_hidden/reward_safe/plus_adapted_ricker_only_pbvi/faithful_internal/episodes.csv` | `8b09ee8511af6e0ba90c6f3f5e78ccbbabc1f497beb34af84332380efb68f913` |
| Tiger MOOR | `moor/evaluation/regime_hidden/amur_tiger/allee/sigma_0p2/moor_three_species_matched_p10_v1/data_real/backend_numpy/regime_hidden/reward_safe/moor_adapted_ricker_misspec_pbvi/faithful_internal/episodes.csv` | `540837583fc07dd017db9f0e8766cb9a7c188bfd3ff2930e20a966e5daf5f823` |
| Fox PLUS | `plus/evaluation/regime_hidden/crab_eating_fox/allee/sigma_0p2/plus_three_species_matched_p10_v1/data_real/backend_numpy/regime_hidden/reward_safe/plus_adapted_ricker_only_pbvi/faithful_internal/episodes.csv` | `91eb0298baa5012d9fafeda38608fd0bf81eac98a64871c550800b6818ec735c` |
| Fox MOOR | `moor/evaluation/regime_hidden/crab_eating_fox/allee/sigma_0p2/moor_three_species_matched_p10_v1/data_real/backend_numpy/regime_hidden/reward_safe/moor_adapted_ricker_misspec_pbvi/faithful_internal/episodes.csv` | `34bb35079496f6ec4b5fe8eda824bb64b542fb6eb2e4825d6b200bb916a9276d` |

Accepted general episodes are all below `$GEN/evaluation/regime_hidden/<species>/allee/sigma_0p2/data_real/backend_numpy/regime_hidden/reward_safe/<method>/learned/episodes.csv`:

| Cell | RefPlan | OGSRL | BA-MCTS | EVD |
|---|---|---|---|---|
| Tiger | `8d546adb928b13a9824f07f5b362bafa5133dc772ba34c823bd3946b2a8870a8` | `c1db405698660f9cad3bb352aa575d2a2f4499141c5a6a1beae09a92cd9c6d02` | `20608a3c90eb1d438a372284d5ebe6965eaed40108f9915e0b27159b7e6cde75` | `8d35854f0d02fcc112a992caeae6a1693514c75893e4759fd0fb85de5a4add07` |
| Fox | `7e8c6d5812bc6b238092728b54734b6cd92e9ec58011ae3fa934380d32f24536` | `bb732bac5946e162c2179b3b13a461d5a7c6dbd5d80fcb78c829c639ab98c540` | `1c75ec2e9b6cc299ec020dbe2fa17a8b0ab201c7e98b04ef07eef099a77e5997` | `fb581f8d89f72be2f2ffed6765adca01202d88af70faed9b678ccfda3349eb10` |

The full exact paths, artifact-tree hashes, source hashes and checksum recipes are repeated in `I0_READONLY_MANIFEST.json`.

## 3. Method-role verification

| Executable | Current role and representation | Uncertainty representation | Planning/training reward | `observation_noise_sigma` | Noisy-arm recovery and exact-arm frozen artifacts |
|---|---|---|---|---|---|
| `plus_adapted_ricker_only_pbvi` | Adapted PLUS; eight fitted Ricker candidates. Maintains one latent abundance-grid belief per candidate plus a posterior over candidates; PBVI action values are marginalized over both. | Eight-model structural/value uncertainty plus candidate posterior. | Accepted public safe reward surrogate embedded in the faithful context/POMDP construction. | **Direct.** Fitted mechanistic models store it as `observation_scale`; likelihood updates use it. | Accepted source/episode/fit receipts exist in the current ecological track. Execution was not rerun. For a deployment-only exact-state arm, the eight Ricker fits, candidate bank, reward surrogate and PBVI/POMDP artifacts can remain byte-identical; only runtime abundance beliefs should be replaced. |
| `moor_adapted_ricker_misspec_pbvi` | Adapted MOOR; one deliberately misspecified fitted Ricker model, one latent abundance-grid belief, PBVI control. | One fitted model with pessimistic/robust planner construction; no model posterior bank. | Same accepted public safe reward surrogate contract. | **Direct.** Same mechanistic observation likelihood path as PLUS. | Accepted source/episode/fit receipts exist. One fitted Ricker model, reward surrogate and PBVI/POMDP artifacts can remain byte-identical under direct runtime belief replacement. |
| `refplan` | RefPlan-inspired reflect-then-plan policy. Hidden mode fits a public observation-space dynamics ensemble and behavior-policy prior; plans from shared public particles/history and updates a deployment model posterior. | At least five public dynamics members plus categorical deployment posterior; policy prior proposes plans. | `PublicParticlePlanner` uses the shared public reward surrogate. | **Direct and indirect.** Directly adds `observation_noise_sigma` to model-posterior update variance; indirectly receives shared-filter particles produced with sigma. | Accepted general-track artifacts and episodes exist; no I0 replay. A secondary deployment-only exact-particle diagnostic could freeze dynamics, behavior prior and surrogate, but the primary end-to-end Arm T must rebuild noisy-state-derived cache/model/prior. |
| `ogsrl` | OGSRL-inspired guarded constrained categorical actor, fixed after offline training. Fits a public dynamics ensemble, public kNN OOD guardian, train-only low-abundance proxy and constrained actor. | Bootstrap public dynamics uncertainty plus OOD and low-abundance constraint channels; no online model posterior in the deployed actor. | Hidden mode uses the shared public reward surrogate in actor rollouts; its separate safety cost is the registered public low-abundance shortfall. | **Indirect, not direct in `ogsrl.py`.** Shared cached/runtime public particles use sigma. | Accepted artifacts/episodes exist. End-to-end Arm T must rebuild exact-state belief cache, dynamics ensemble, guardian, low-abundance scale/budget and actor; surrogate may remain frozen if reward inputs/definition stay registered. |
| `bamcts` | BA-MCTS-inspired online search over public observation buckets/history. Samples public dynamics members and carries/updates a categorical model belief inside the tree and at deployment. | Public dynamics ensemble and categorical model belief. | Hidden tree rollout uses the shared public reward surrogate minus dynamics disagreement pessimism. | **Indirect, not direct in `bamcts.py`.** Root particles/cache come from the sigma-dependent shared filter. | Accepted artifacts/episodes exist. End-to-end Arm T must rebuild exact-state cache and public dynamics ensemble, and define exact-state root/bucket/posterior updates; shared reward surrogate may remain frozen. |
| `ensemble_value_disagreement_pessimism` | EVD pessimism: 20 episode-bootstrap conservative fitted-Q estimators plus fitted behavior reference; action is mean Q minus empirical Q-variance penalty. | 20-member Q ensemble; disagreement is empirical value variance. | Fits Bellman targets directly to `dataset.rewards`; no separate online planner reward. | **Indirect, not direct in its policy file.** Training and action features come from sigma-dependent cached/runtime public beliefs. | Accepted artifacts/episodes exist. End-to-end Arm T must rebuild exact-state cache, behavior reference and all 20 Q members. Raw rewards can remain unchanged. |

The direct policy-file classification in Revision 2 is therefore incomplete for OGSRL, BA-MCTS and EVD. For an end-to-end information-arm comparison, sigma enters their inputs before their policy classes are called.

MOBILE is **not implemented**. It appears only as contextual literature/documentation; it has no executable registration or method implementation in either frozen track.

Current implementation source SHA-256 values are:

- ecological PLUS `plus_faithful.py`: `5f186aca4c7c467afb0c4a62505b7736a26ec697005713e1c42419d54d34cb1e`
- ecological MOOR `moor_faithful.py`: `4cf371a4677341dea8be8229df98567d96bea7ddf019bb44bb5ab2f1fa2e8b20`
- general RefPlan: `c65865b2c9c270d499701827ad2c754e0f9cf02b31febd6916d5dd6494c0fbf1`
- general OGSRL: `da965bc320f3cbfaf381a46aaa77a7eb0ef7c3b6b7b493bee02e338544e6993b`
- general BA-MCTS: `5ffd3d8962e723d0efb195e4b461262903c2b3e1e977a0781f70a9a17f8b4f05`
- general EVD: `ab2d953f5a46ef4892830095bc8d496cc587fc0426b6e705736b5c3b905f171e`
- general shared beliefs/filter: `e8277daad30e5052925f4139ac9fbfb4e6c539882940527e827274f46448e184`

## 4. Arm T component-change table

`Same` means byte-identical reuse is technically supportable under the stated information intervention; it does not grant I1/I2 authorization. `Rebuild` means an end-to-end exact-state arm would otherwise train against a different representation than it receives at deployment.

| Method | Offline inputs | Transition/model fitting | Reward-surrogate fitting | Policy fitting | Online belief/state updates | Action selection | Cached artifacts | Expected compute |
|---|---|---|---|---|---|---|---|---|
| **PLUS** | Same public dataset | **Same eight Ricker fits** | Same | Same POMDP/PBVI construction | **Change:** after reset/each step, assign each candidate belief's abundance marginal to the nearest true latent-grid point; do not apply the noisy likelihood | Algorithm unchanged; receives collapsed beliefs | Keep all 39 faithful-artifact files byte-identical; prove hashes | Approximately accepted PLUS task time, 15,315 s (4.25 h) per cell/arm. Cold eight-fit contingency adds about 8,683 s (2.41 h) per affected cell. |
| MOOR | Same | **Same one Ricker fit** | Same | Same POMDP/PBVI construction | **Change:** direct one-belief abundance point mass | Algorithm unchanged | Keep all nine faithful-artifact files byte-identical; prove hashes | About 1,941 s (0.54 h) per cell/arm |
| RefPlan | Primary end-to-end arm: exact-state cache replaces noisy cache. Secondary frozen-fit diagnostic: same offline inputs | Primary: rebuild public dynamics ensemble. Secondary: keep it frozen | Same if reward contract/input fields remain unchanged | Primary: rebuild behavior prior. Secondary: freeze | Exact raw-abundance particles/history via a new adapter; model-posterior update contract must be specified | Planner unchanged after its input contract is settled | Primary: rebuild belief/model/prior cache. Secondary: retain and hash existing artifacts | Existing receipt about 52 s per task; rebuild expected to remain small relative to PLUS, but exact adapter timing is open |
| OGSRL | **Rebuild exact-state cache** | **Rebuild public dynamics ensemble** | Same shared surrogate if reward contract unchanged | **Retrain guardian, low-abundance scale/budget and constrained actor** | Exact-state `BeliefState`; no noisy public-particle draw | Composite actor/guardian rule unchanged | Rebuild all state-derived model/policy caches | Existing receipt about 430 s per task; exact rebuild expected similar order, implementation uncertainty dominates |
| BA-MCTS | **Rebuild exact-state cache** | **Rebuild public dynamics ensemble** | Same shared surrogate | No separately fitted policy; search is online | Exact-state root/history and an explicit model-posterior update rule | Search structure remains, but keys and likelihood update contract must match exact state | Rebuild belief/model cache | Existing receipt about 847 s per task; search remains the general-method bottleneck |
| EVD | **Rebuild exact-state cache** | No transition model | Not applicable; keep raw accepted rewards | **Retrain behavior reference and 20 Q members** | Exact-state features | Same mean-Q-minus-variance rule | Rebuild belief/behavior/Q caches | Existing receipt about 2.5 s per task |

For PLUS and MOOR, the existing Ricker fit and faithful artifact tree hashes are:

| Cell/method | Files | Artifact-tree SHA-256 |
|---|---:|---|
| Tiger PLUS | 39 | `5b3ef8ed9453646810fcf8629450c3a595271d7b3ab9d8fcf8cc4abe0e8b0dc5` |
| Tiger MOOR | 9 | `07cca42624514a4065417cd1355065ec87d0ff0dbcd494c611ba2a338a7e4eb3` |
| Fox PLUS | 39 | `b42f3e68f87e8a5174b2e804b4eccd2ed3ec925867b37590ddf5b7dbeea79c46` |
| Fox MOOR | 9 | `d2df0bd6771fb408437cab82786d4573c4cb0529e5885efdb4ffd343788e4110` |

These are aggregate hashes using the frozen-track recipe. The exact directories are the `faithful_artifacts` children of the four ecological `faithful_internal` episode directories listed in Section 2.

## 5. Exact-state route audit

### Why sigma zero/epsilon is wrong

`MechanisticModel.observation_likelihood()` first computes latent observation `y = max(raw_observation / survey_scale, 0)`. When `observation_scale <= 1e-12`, it returns an equality/is-close indicator on the discrete latent abundance grid. An off-grid true value therefore yields an all-zero likelihood. This is not an exact-state adapter.

The consequences are source-verified:

- `_normalize()` maps a zero or non-finite total to a **uniform** vector.
- `initial_belief()` normalizes `prior * likelihood`; all-zero likelihood therefore becomes uniform, not the intended point mass and not even the original nonuniform prior.
- Runtime `update()` computes a predicted belief first. If evidence is `<= 1e-300` or non-finite, it retains that **predicted belief** rather than conditioning on truth.

Setting sigma, `observation_scale`, or any likelihood scale to zero or epsilon can therefore erase information or silently fall back to prediction. Arm T must bypass the observation likelihood.

### Correct conversion and assignment point

The true abundance presented by the evaluator is in raw survey units. For each faithful mechanistic model:

1. convert to the model's latent abundance units with `latent_true = raw_true / model.survey_scale`;
2. locate the nearest value in `pomdp.abundance_grid`;
3. put all abundance probability on that grid index;
4. preserve the registered handling of any regime dimension and do not expose family label, private safety state, future state, `r`, or `K`.

For PLUS, direct assignment belongs in the runtime policy/evaluator bridge immediately before `act()`, replacing the abundance component of each entry in `PLUSRickerOnlyFaithfulPBVIPolicy.internal_beliefs`. For MOOR it belongs at the analogous single `internal_belief`. Their normal observation update must not then be applied to the same transition.

### Existing oracle route and its limit

`OracleStateFilter.set_true_state()` creates a `BeliefState` whose particles all equal the supplied true raw state. `ContinuousEvaluator` already calls it after reset and after every environment step. However, `make_filter_factory` rejects oracle filtering for `expose_rk="hidden"`, and the oracle-state ablation runner rejects hidden methods. Thus the primitive exists, but no authorized/executable hidden-method Arm T route exists today.

For RefPlan, a wrapper can populate its public particles/history with the exact raw abundance while leaving the fitted public dynamics, behavior prior and reward surrogate untouched for a clearly labelled secondary deployment-only diagnostic. Its primary end-to-end Arm T must instead rebuild the exact-state-derived inputs. The design must settle previous/current history semantics and model-posterior likelihood semantics; merely swapping particles at evaluation would otherwise train and deploy on different feature distributions.

Discrepancies from Revision 2: the zero-likelihood warning and `survey_scale` conversion are correct; the generic oracle primitive is present but is not directly available to hidden methods; and the plan's sigma-use inventory omits the general methods' shared-filter dependency.

## 6. Short-episode audit

### Fix provenance

| Question | Finding |
|---|---|
| Which branch/commit contains the fix? | **Unavailable locally.** No matching local branch, remote-tracking ref, commit message, handoff, or source change was found. |
| Applied to current working tree? | **No.** `_fit_public_safety_scale()` still raises when an episode has fewer transitions than `ogsrl_cost_horizon`. |
| Three regression-test locations? | **Unavailable locally.** No such three-test set was found; only the draft plan describes it. |
| Previously recorded passing evidence? | **Unavailable locally.** No receipt or test log for those three tests was found. |

This does not corrupt the accepted rows inspected below, but it is a provenance discrepancy that must be closed before G1.

### Existing accepted dataset classification

The audit read the two accepted NPZ files only. An episode was considered valid when its rows were contiguous, timesteps ran from 0 through the final step without gaps, observation/next-observation chaining was consistent, no early done flag appeared, and the final row had exactly one of `terminated`/`truncated` set.

| Cell | Rows | Episodes | Complete full length | Genuine short, `terminated=True` | Truncated short | Invalid/incomplete | Length distribution | Would documented fix touch an accepted row? |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Tiger × Allee × σ=0.2 | 4,000 | 160 | 160 | 0 | 0 | 0 | `{25: 160}` | No |
| Fox × Allee × σ=0.2 | 4,000 | 160 | 160 | 0 | 0 | 0 | `{25: 160}` | No |

All 320 episodes end by full-horizon truncation; none is marked terminated. The proposed short-episode normalization change would touch zero rows in either accepted dataset.

## 7. Accepted-anchor verification

Primary accepted sources:

- six-method values: `results/accepted/MATCHED_P10_144_METHOD_CELLS.csv` and its receipt;
- fixed-action screens: `results/followups/reward_screen/constant_reward_screen.csv`, SHA-256 `a09d21d26fcbd95a6279603bfffed10c2a75505a360f70cb38d9c8d96fdaed14`;
- diagnostic differences and PLUS activity: `results/diagnostic_replay/m1_m15_comparison.csv`, SHA-256 `8ac75e8b15e155436a71d96f9db934f09bd4b8d8e940b4b76054f2408947b2b9`, plus the diagnostic replay receipt.

| Required anchor | I0 verification |
|---|---|
| Tiger OGSRL return | `4.71629840155769` — exact match in accepted table |
| Tiger best fixed action 10 | `4.224420899026084` — exact match in fixed-action screen |
| Fox MOOR return | `11.610936360689793` — exact recomputation/source value in diagnostic replay; accepted CSV serializes it as `11.61093636068979` (difference about `3.55e-15`) |
| Fox best fixed action 1 | `10.153605271956147` — exact match in fixed-action screen |
| PLUS fox headroom | `10.966745450258198 - 10.153605271956147 = 0.8131401783020511` — exact match |
| PLUS fox switch-from-MAP fraction | `0.709` — exact match in diagnostic replay evidence |

### Pre-classification of accepted Arm O activity

This is descriptive classification of accepted episode records only. “Active + headroom” means at least one within-episode action switch and positive accepted return minus best fixed-action return. It is not an interpretation of a new scientific result.

| Cell/method | Episodes with >1 action | Mean episode action entropy | Fixed-action headroom | Pre-classification |
|---|---:|---:|---:|---|
| Tiger RefPlan | 20/20 | 1.69514 | -0.76995034 | Decision-active; no positive headroom |
| Tiger OGSRL | 20/20 | 1.13894 | +0.49187750 | Active + headroom |
| Tiger BA-MCTS | 20/20 | 1.14487 | -2.94633530 | Decision-active; no positive headroom |
| Tiger EVD | 20/20 | 1.19685 | -15.86168980 | Decision-active; no positive headroom |
| Tiger PLUS | 0/20 | 0 | 0 | Inactive/non-discriminating |
| Tiger MOOR | 0/20 | 0 | 0 | Inactive/non-discriminating |
| Fox RefPlan | 20/20 | 1.97367 | -2.20546300 | Decision-active; no positive headroom |
| Fox OGSRL | 0/20 | 0 | -0.02322270 | Inactive/non-discriminating |
| Fox BA-MCTS | 20/20 | 1.49684 | +0.04775860 | Active + small headroom |
| Fox EVD | 20/20 | 0.33010 | +0.24382400 | Active + headroom |
| Fox PLUS | 20/20 | 0.44860 | +0.81314018 | Active + headroom |
| Fox MOOR | 20/20 | 0.25851 | +1.45733109 | Active + headroom |

## 8. Compute estimate

Existing full-task mean receipts used by Revision 2:

| Method | Seconds per cell/arm task | Hours |
|---|---:|---:|
| PLUS | 15,314.56 | 4.254 |
| MOOR | 1,940.64 | 0.539 |
| RefPlan | 52.42 | 0.0146 |
| OGSRL | 430.15 | 0.119 |
| BA-MCTS | 847.26 | 0.235 |
| EVD | 2.50 | 0.0007 |
| **One cell × one arm** | **18,587.53** | **5.163** |

For two cells, six methods and two arms, the existing-fit/frozen-artifact estimate is `4 × 18,587.53 = 74,350.12` CPU-seconds, or **20.65 core-hours**. With sufficient independent task concurrency, the expected wall-time bottleneck is one PLUS task, about **4.25 hours**, plus queue and I/O variance.

The accepted eight-candidate PLUS receipts are cache hits and report only about 0.054 seconds of lookup time, so they cannot estimate cold fitting. The nearest cold-fit evidence consists of 32 completed 16-candidate adapted-mechanistic PLUS receipts. Their mean is 17,365.31 seconds, range 15,452.72–18,244.06. Assuming approximately linear candidate cost, eight cold candidates are **8,682.65 seconds (2.41 hours) per affected cell**, range **2.15–2.53 hours**.

If Arm T unexpectedly requires cold-fitting eight PLUS candidates in both cells, add about **4.82 core-hours** (range 4.29–5.07), producing a total estimate of **25.47 core-hours** (range 24.95–25.72). With full concurrency, the cold-fit PLUS Arm T task becomes the wall bottleneck at about **6.66 hours per cell** (range 6.40–6.79), assuming fitting and evaluation are sequential.

Uncertainty: the cold-fit estimate scales 16-candidate receipts rather than timing the exact eight-candidate code path; node load, BLAS architecture, cache warmth, filesystem traffic, and the not-yet-implemented exact-state adapters are unmeasured. The correct I1 design should avoid the contingency for PLUS/MOOR by preserving their fits byte-for-byte.

## 9. Discrepancy ledger

Status vocabulary is exactly the requested vocabulary. “Requires execution authorization” means I0 found the route but did not run it.

| Controlling-plan requirement | Status | I0 evidence / consequence |
|---|---|---|
| Plan is complete Revision 2 | verified | Entire 661-line file read; header says Revision 2 and ends at Section 14. |
| Two cells: tiger/fox × Allee × σ=0.2 | verified | Accepted datasets and six-method artifacts located. |
| Six executable methods and roles | verified | Registrations and implementation files inspected; MOBILE absent. |
| Ecological/general tracks remain separate | verified | Both track hashes recorded independently. |
| Repository/HEAD/branch/clean status | verified | HEAD and clean initial state recorded in Section 2. |
| Accepted environment and CPU profile | verified | Paper-faithful venv, dependency hashes and 8452Y receipt located. |
| Accepted noisy arm recoverable from current tracks | verified | Source, dataset, receipt and episode hashes exist for all six methods; runtime reproduction was not attempted. |
| Only PLUS, MOOR and RefPlan consume observation-noise sigma | contradicted | OGSRL, BA-MCTS and EVD consume it indirectly through shared cached/runtime `PublicObservationFilter` beliefs. |
| Arm T is not sigma/likelihood scale zero | verified | Zero-likelihood/fallback source route confirms the warning. |
| Raw abundance must pass through `survey_scale` for faithful grids | verified | Source divides raw observation by `survey_scale` before grid likelihood. |
| Generic oracle-state primitive exists | verified | `OracleStateFilter.set_true_state` plus evaluator hooks found. |
| Generic oracle route works for hidden pilot methods without changes | requires implementation | Factory and ablation runner reject oracle plus `expose_rk=hidden`. |
| PLUS exact-state runtime belief collapse | requires implementation | Correct assignment point identified; no adapter exists. |
| MOOR exact-state runtime belief collapse | requires implementation | Correct assignment point identified; no adapter exists. |
| PLUS/MOOR fitted Ricker artifacts can remain frozen | verified | Fit inputs are independent of runtime assignment; four faithful artifact trees hashed. I2 must prove equality. |
| RefPlan frozen-fit deployment-only exact particles | requires implementation | Technically feasible wrapper identified; history/posterior semantics need registration. |
| RefPlan primary end-to-end Arm T | requires implementation | Exact cache, dynamics and behavior prior must be rebuilt. |
| OGSRL end-to-end Arm T | requires implementation | Cache, dynamics, guardian, safety scale/budget and actor must be rebuilt. |
| BA-MCTS end-to-end Arm T | requires implementation | Cache/model plus exact root/key/posterior semantics required. |
| EVD end-to-end Arm T | requires implementation | Exact cache, behavior reference and 20 Q members must be rebuilt. |
| Branch-isolated OGSRL short-episode fix exists locally | unavailable locally | No branch/ref/commit/handoff located. |
| OGSRL fix is in current working tree | contradicted | Current function still raises for `len(cost) < horizon`. |
| Three OGSRL short-episode regression tests exist | unavailable locally | No matching test set found. |
| Prior passing evidence for those tests | unavailable locally | No test receipt/log found. |
| Accepted tiger/fox datasets contain affected short rows | contradicted | Both are exactly 160 × 25; zero short episodes. This contradicts any implication that accepted rows require repair, not the intended general fix. |
| G1 run the three regression tests | requires execution authorization | Tests also first require the missing implementation/provenance. |
| G2 regression anchor/rebaseline | requires execution authorization | Explicitly unauthorized at I0. Conditional rebaseline remains closed. |
| Accepted anchor values | verified | All six requested numbers traced to primary accepted records; fox MOOR serialization note recorded. |
| Pre-classify Arm O activity | verified | Existing episodes only; Section 7. |
| Common controls/seeds/action schema/evaluator settings for future arms | open | Plan specifies them, but exact-arm registration and receipts do not yet exist. |
| Primary estimands and within-seed contrasts | open | Defined by plan; no Arm T execution exists. |
| Frozen-fit compute estimate | verified | 20.65 core-hours from current receipt means. |
| Eight-candidate PLUS cold-refit contingency | verified | Receipt-based scaled estimate 4.82 added core-hours across two cells, with explicit uncertainty. |
| Stage B pilot execution | requires execution authorization | Not authorized and not run. |
| Stage C expansion | requires execution authorization | Conditional and explicitly unauthorized. |
| Create I0 planning-only artifacts only | verified | New collision-checked directory; no accepted namespace overlap. |
| Do not silently repair discrepancies | verified | No source/config/document/accepted artifact was edited. |

## 10. Explicit recommendations for I1/I2

1. Amend the information-flow inventory before I1: distinguish direct policy-file sigma use from indirect sigma use through shared belief construction. Treat all four general methods as representation-dependent in an end-to-end Arm T.
2. Recover or explicitly recreate the OGSRL short-episode fix on a named branch/commit, define the three exact test paths, and produce a passing non-scientific regression receipt before G1. Do not claim the currently absent provenance.
3. Register two clearly distinct RefPlan analyses: the primary end-to-end exact-state arm with rebuilt cache/model/prior, and an optional secondary frozen-fit deployment-only sensitivity diagnostic. Do not mix their interpretation.
4. Implement a first-class exact-state adapter. For PLUS/MOOR it must convert `raw_true / survey_scale`, select the nearest abundance-grid point, and assign beliefs directly. For general methods it must preserve public history fields without leaking family, private safety, `r`, `K`, or future state.
5. Add unit tests for off-grid truth, reset/update timing, no double update, all-zero likelihood avoidance, and byte-identical PLUS/MOOR artifacts. Test RefPlan previous/current history and posterior-update semantics explicitly.
6. In I2, hash the four existing faithful artifact trees before and after exact-arm packaging. A mismatch for PLUS/MOOR should stop the run unless a separately authorized refit contingency is invoked.
7. Size future jobs from PLUS: reserve at least 4.25 hours plus margin for frozen fits, or about 6.7 hours plus margin if an authorized cold eight-fit contingency is actually needed. Pin strict replay to Xeon Platinum 8452Y.
8. Keep the accepted datasets untouched. The short-episode fix is important for general correctness but would alter zero accepted rows in these two cells.

## 11. Authorization status

Completed under I0 authorization: repository/source/document/config/receipt inspection; hashes; existing-dataset episode classification; existing-record activity summaries; one new planning-only directory containing this report, its read-only manifest, and its checksum file.

Not performed and still unauthorized: source/config/registration edits; adapter implementation; dataset generation; tests with repository-state effects; scientific evaluation; regression anchor; rebaseline; Slurm submission; I1/I2 or later stages; freezing any new scientific artifact; conditional Stage B or Stage C.

The process stops after checksum/format verification of these three I0 files and awaits user and Claude review.

FAIL — PLAN OR PROVENANCE DISCREPANCY
