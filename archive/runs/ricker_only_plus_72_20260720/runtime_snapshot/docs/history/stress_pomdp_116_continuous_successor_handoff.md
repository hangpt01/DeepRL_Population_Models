# Stress POMDP 116 implementation handoff for a continuous-state successor

This is a code-level handoff for the current discretized ecological offline model-based RL benchmark. It records the implementation that actually ran, not only the original design. Paths are relative to the repository root; the executable project root is claude_build/.

The target successor is different: latent abundance s_t is continuous, nonnegative, and unbounded; o_t = s_t eta_t with eta_t ~ LogNormal(0, sigma_obs^2); reward is computed from o_t; a shared state-belief filter feeds every method; biological parameters remain Bayes-adaptive. Section 8 identifies every current assumption that must be replaced.

## 1. Repo map

### 1.1 Layout and configuration

| Path | Responsibility |
|---|---|
| claude_build/config/config.yaml | Root Hydra composition: environment, model, planner, seed, device, WandB, evaluation and output paths (claude_build/config/config.yaml:1-41). |
| claude_build/config/env/ | Dynamics, priors, actions, reward, discretization, collection and evaluation starts. |
| claude_build/config/model/ | MOPO, RefPlan, BA-MCTS, PLUS and MOOR settings. |
| claude_build/config/planner/pessimistic.yaml | MOPO receding-horizon planner (claude_build/config/planner/pessimistic.yaml:1-18). |
| claude_build/src/environments/ | RickerEnv, three stress POMDPs and reward contract. |
| claude_build/src/models/ | Learned-method and ecology-baseline adapters. |
| claude_build/src/planners/ | External MOPO planner. Other headline methods plan internally. |
| claude_build/src/training/ | Neural MOPO offline trainer. |
| claude_build/src/evaluation/evaluator.py | Shared episode evaluator and artifact writer. |
| claude_build/scripts/core/ | Dataset, training, evaluation and end-to-end entry points. |
| claude_build/scripts/experiments/ | Manifest generation. |
| claude_build/scripts/analysis/ | Control gate, freezing, seed aggregation and reports. |
| claude_build/scripts/slurm/ | Stress harness, workers and launch/report jobs. |
| claude_build/tests/ | Environment, POMDP-contract, action-authority, method and planner tests. |
| claude_build/outputs/manifests/ | Exact deployed TSV matrices. |
| claude_build/outputs/stress_pomdp_116/ | Shared data, gate files, run outputs, metadata and frozen settings. |

Environment keys are registered at claude_build/src/environments/__init__.py:1-24. Model keys are registered at claude_build/src/models/__init__.py:1-23. Construction uses Registry.build(name: str, **kwargs) in claude_build/src/registry.py:25-82; action counts are checked by validate_action_space(...) at claude_build/src/registry.py:108-125.

Hydra composes env, model and planner groups. Root defaults are env=ricker, model=ensemble and planner=pessimistic; registry keys are selected separately by active_env, active_model and active_planner (claude_build/config/config.yaml:1-14). The stress harness selects both env and active_env, then overrides reward mode, data, evaluation and method parameters (claude_build/scripts/slurm/stress_pomdp_116.sh:146-159 and :297-421).

### 1.2 Entry points and setup

The end-to-end Hydra entry has signature main(cfg: DictConfig) -> None at claude_build/scripts/core/run_pipeline.py:93-94. It builds the environment and data (:124-165), fits the model (:174-212), builds the planner (:237-240), and evaluates (:291-303).

A direct run is:

    cd claude_build
    python scripts/core/run_pipeline.py \
      env=allee_ricker_pomdp_116 active_env=allee_ricker_pomdp_116 \
      model=refplan active_model=refplan \
      model.num_states=100 model.num_actions=5

Other entry points are scripts/core/generate_dataset.py, train.py and evaluate.py. Dataset generation's Hydra main is at claude_build/scripts/core/generate_dataset.py:58-82.

Cluster jobs load miniforge3 and activate conda environment pytorchrl (claude_build/scripts/slurm/stress_pomdp_116.sh:26-28). Fresh setup is:

    cd claude_build
    pip install -r requirements.txt
    pytest tests/ -v

Dependencies are PyTorch >=2.0, Hydra >=1.3, OmegaConf >=2.3, WandB >=0.15, NumPy >=1.24, pandas >=2.0, SciPy >=1.10, d3rlpy >=2.6,<3.0, pytest >=7 and tqdm >=4.65 (claude_build/requirements.txt:1-26). The harness reads WandB settings from claude_build/.wandb_env and accepts only WANDB_* and WB_PROJECT variables (claude_build/scripts/slurm/stress_pomdp_116.sh:42-60).

### 1.3 Benchmark and ablation launch

Gate the six principal cells from claude_build/ with:

    sbatch --array=0-5%6 scripts/slurm/stress_pomdp_116.sh calibrate

The stage/array convention is documented at claude_build/scripts/slurm/stress_pomdp_116.sh:13-22. The final launch flow is:

    sbatch scripts/slurm/stress_pomdp_116_after_tune.sh

It freezes on tuning data, then generates final manifests with seeds 7001-7005, reward modes collapse_sensitive and base, 75,000 transitions, 50 evaluation episodes and horizon 50 (claude_build/scripts/slurm/stress_pomdp_116_after_tune.sh:53-80). Array dependencies are submitted at :82-106.

The complete five-method final manifest is claude_build/outputs/manifests/stress116_final_methods_with_bamcts.tsv: 300 rows = 6 environment/action cells x 2 rewards x 5 seeds x 5 methods. The 60 dataset cells are in stress116_final_datasets.tsv. A worker reads one row, checks its gate, exports values and delegates to the harness (claude_build/scripts/slurm/stress_pomdp_116_manifest_worker.sh:57-146).

Reusable resource wrappers are:

- CPU: stress_pomdp_116_manifest_worker_cpu.sh, comp partition, 8 CPUs, 64 GB, 8 h (claude_build/scripts/slurm/stress_pomdp_116_manifest_worker_cpu.sh:1-23).
- GPU: stress_pomdp_116_manifest_worker_gpu_any.sh, gpu partition, one unpinned GPU, 8 CPUs, 64 GB, 8 h (claude_build/scripts/slurm/stress_pomdp_116_manifest_worker_gpu_any.sh:1-24).

The Tier-1 resolution ablation is w=2, not continuous. It covers Allee and theta, 5/10 actions, collapse_sensitive only, seeds 7001-7005, 75,000 transitions and 50 x horizon-50 evaluations. Exact manifests are:

- stress116_bw2_datasets.tsv: 20 rows.
- stress116_bw2_methods.tsv: 100 rows.
- stress116_bw2_methods_cpu.tsv: 80 rows.
- stress116_bw2_methods_mopo.tsv: 20 rows.

It can be rerun with the manifest workers. The bw2 cells have no separate saved gate JSONs, so set REQUIRE_GATE_PASS=false; submit CPU/GPU method arrays only after the dataset array succeeds:

    BASE=$PWD REQUIRE_GATE_PASS=false \
      sbatch --array=0-19%20 \
      scripts/slurm/stress_pomdp_116_manifest_worker_cpu.sh \
      outputs/manifests/stress116_bw2_datasets.tsv

    BASE=$PWD REQUIRE_GATE_PASS=false REQUIRE_DATASET=true \
      sbatch --dependency=afterok:DATASET_JOB --array=0-79%40 \
      scripts/slurm/stress_pomdp_116_manifest_worker_cpu.sh \
      outputs/manifests/stress116_bw2_methods_cpu.tsv

    BASE=$PWD REQUIRE_GATE_PASS=false REQUIRE_DATASET=true \
      sbatch --dependency=afterok:DATASET_JOB --array=0-19%4 \
      scripts/slurm/stress_pomdp_116_manifest_worker_gpu_any.sh \
      outputs/manifests/stress116_bw2_methods_mopo.tsv

Manifest fields and row construction are in claude_build/scripts/experiments/make_stress116_manifest.py:54-78 and :165-234; its CLI is at :237-313.

## 2. Environment

### 2.1 API, observations and diagnostics

BaseEnvironment defines (claude_build/src/interfaces/base_env.py:47-100):

    @property
    def num_states(self) -> int

    @property
    def num_actions(self) -> int

    def reset(self, seed: Optional[int] = None) -> int

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]

    def reward(self, state: int, action: int, next_state: int) -> float

    def generate_dataset(
        self,
        n_transitions: int,
        policy: str = "random",
        seed: Optional[int] = None,
    ) -> Dataset

RickerEnv extends reset to:

    def reset(
        self,
        seed: Optional[int] = None,
        s0_low: Optional[float] = None,
        s0_high: Optional[float] = None,
        episode_len: Optional[int] = None,
    ) -> int

and adds reset_for_eval(self, seed: Optional[int] = None) -> int (claude_build/src/environments/ricker_env.py:150-198). StressPOMDPMixin keeps that reset and implements step(self, action: int) -> Tuple[int,float,bool,Dict[str,Any]] (claude_build/src/environments/stress_pomdp_envs.py:63-142).

There are no Gym space objects. Observation is an integer in {0,...,num_states-1}; action is an integer in {0,...,num_actions-1}. Hidden state and parameters are excluded. Tests assert integer observations and that datasets contain no hidden fields (claude_build/tests/test_stress_pomdp_envs.py:146-168).

Stress info contains s_continuous, collapsed, collapse_entered, hidden variables before the transition, and the same hidden keys suffixed _after (claude_build/src/environments/stress_pomdp_envs.py:135-142). Hidden keys are:

- Allee: hidden_C, hidden_C_bucket.
- Theta: hidden_theta, hidden_theta_bucket.
- Regime: hidden_regime, hidden_regime_label, hidden_regime_harsh_fraction, hidden_regime_majority.

The evaluator calls select_action before env.step and appends diagnostics only afterward, so hidden info is not in agent history (claude_build/src/evaluation/evaluator.py:200-225).

### 2.2 Shared action transform and transition order

For every model:

    r_eff(a) = r_base + Delta_r(a)
    K_eff(a) = K_base + Delta_K(a)
    s_managed = clip(s_t * (1 - h(a)) + delta(a), 0, s_max)

Effective parameters are at claude_build/src/environments/ricker_env.py:294-302; direct abundance authority is at :309-315. The exact order is:

1. Compute r_eff and K_eff from unchanged episode r_base and fixed K_base. Action effects do not accumulate into the next step's bases.
2. Apply harvest/stocking and clip.
3. Apply growth and clip.
4. Discretize.
5. Under collapse_sensitive, convert a raw next bin at/below threshold into absorbing s=0,x=0.
6. Score reward, record the transition, then advance hidden regime.

The stress step is claude_build/src/environments/stress_pomdp_envs.py:92-142. Regime z_t is used for current growth and z_{t+1} is sampled after it (:287-301).

### 2.3 Exact dynamics and priors

Ricker class/constructor:

    class RickerEnv(BaseEnvironment):
        def __init__(self, cfg: Any) -> None

(claude_build/src/environments/ricker_env.py:30-44). The coded map is:

    s_{t+1} = clip(
        s_managed * exp(r_eff * (1 - s_managed/K_eff)),
        0, s_max)

implemented by _ricker_step(self,s,r_eff,K_eff) at claude_build/src/environments/ricker_env.py:304-307. Standalone ricker.yaml samples r_base ~ Uniform(0.95,1.00) once per episode, fixes K_base=500, and uses s_max=1000 (claude_build/config/env/ricker.yaml:6-23; sampling at ricker_env.py:158-175).

The principal stress benchmark does not use that high-growth Ricker as a true-environment cell. Ricker is the deliberately misspecified family for PLUS, MOOR and the gate. Those components take r prior/range from the active stress cell; gate Ricker MPC uses that range's midpoint (claude_build/scripts/analysis/stress116_control_gap.py:59-79).

Allee-Ricker:

    class AlleeRickerPOMDPEnv(StressPOMDPMixin)

It samples C ~ Uniform(90,150) once per episode, with r_base ~ Uniform(0.12,0.30) and K_base=500 (claude_build/config/env/allee_ricker_pomdp_116.yaml:4-8). Its map is:

    exponent = clip(
        r_eff * (1 - s_managed/K_eff) * (s_managed/C - 1),
        -50, 50)
    s_{t+1} = clip(s_managed * exp(exponent), 0, s_max)

Sampling and dynamics are at claude_build/src/environments/stress_pomdp_envs.py:178-193. C buckets are equal thirds, named C_low/C_mid/C_high (:195-212).

Theta-logistic:

    class ThetaLogisticPOMDPEnv(StressPOMDPMixin)

It samples theta ~ Uniform(3,6), with r_base ~ Uniform(0.18,0.40), K_base=500 (claude_build/config/env/theta_logistic_pomdp_116.yaml:4-8). Its map is:

    ratio = max(s_managed/K_eff, 0)
    s_{t+1} = clip(
        s_managed + r_eff*s_managed*(1 - ratio**theta),
        0, s_max)

implemented at claude_build/src/environments/stress_pomdp_envs.py:228-234. Theta buckets are equal thirds of [3,6] (:244-250).

Regime-switch:

    class RegimeSwitchPOMDPEnv(StressPOMDPMixin)

At reset z_0 is uniform on {safe,harsh}. Persistence P(z_{t+1}=z_t)=0.90. Safe has C=90 and growth multiplier m=1.00; harsh has C=180 and m=0.65. r_base ~ Uniform(0.12,0.30), K_base=500 (claude_build/config/env/regime_switch_pomdp_116.yaml:4-13). Its map is:

    exponent = clip(
        m[z_t]*r_eff*(1 - s_managed/K_eff)*(s_managed/C[z_t] - 1),
        -50, 50)
    s_{t+1} = clip(s_managed*exp(exponent), 0, s_max)

Initial regime, switching and dynamics are at claude_build/src/environments/stress_pomdp_envs.py:281-301. hidden_regime_majority is harsh when harsh_fraction >=0.5 (:303-315).

### 2.4 Exact discretization

Principal geometry is w=10, N=100, s_max=1000, reward x_max=100 and collapse bin 5 (for example claude_build/config/env/allee_ricker_pomdp_116.yaml:17-26). Discretization is exactly:

    x = int(s / self._bin_width)
    return int(np.clip(x, 0, self._num_states - 1))

(claude_build/src/environments/ricker_env.py:317-320). Bin x represents [xw,(x+1)w), except top bin N-1 aliases all values from (N-1)w through s_max. Direct action and growth are separately clipped to [0,s_max] (ricker_env.py:304-315).

The resolution ablation uses w=2, N=500, s_max=1000, x_max=500 and collapse bin 25 (claude_build/config/env/allee_ricker_pomdp_116_bw2.yaml:17-26 and corresponding theta/10a files). Thus x/x_max remains approximately s/1000 and the physical collapse threshold remains about 50.

## 3. Actions

ActionSpec fields are id, name, delta_r, delta_K, cost, harvest_fraction and stocking_delta; omitted direct-authority values default to zero (claude_build/src/environments/reward.py:27-70).

These are the calibrated stress tables actually used by every 116 true model.

### Five actions

| ID | Name | Delta r | Delta K | h | delta | cost |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Do Nothing | 0.00 | 0 | 0.00 | 0 | 0.00 |
| 1 | Aggressive Harvest | -0.02 | 0 | 0.50 | 0 | -0.40 |
| 2 | Predator / Disease Control | 0.02 | 0 | 0.00 | 60 | 0.20 |
| 3 | Intensive Restoration | 0.02 | 200 | 0.00 | 100 | 0.30 |
| 4 | Flagship Conservation | 0.04 | 200 | 0.00 | 160 | 0.60 |

Source: claude_build/config/env/allee_ricker_pomdp_116.yaml:10-15.

### Ten actions

| ID | Name | Delta r | Delta K | h | delta | cost |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Do Nothing | 0.00 | 0 | 0.00 | 0 | 0.00 |
| 1 | Sustainable Harvest | -0.01 | 0 | 0.25 | 0 | -0.20 |
| 2 | Aggressive Harvest | -0.02 | 0 | 0.50 | 0 | -0.40 |
| 3 | Predator / Disease Control | 0.01 | 0 | 0.00 | 40 | 0.20 |
| 4 | Breeding / Recruitment Support | 0.02 | 0 | 0.00 | 60 | 0.40 |
| 5 | Moderate Restoration | 0.00 | 100 | 0.00 | 80 | 0.15 |
| 6 | Intensive Restoration | 0.00 | 200 | 0.00 | 100 | 0.30 |
| 7 | Integrated Conservation (light) | 0.01 | 100 | 0.00 | 80 | 0.35 |
| 8 | Adaptive Conservation Trial | 0.01 | 200 | 0.00 | 120 | 0.50 |
| 9 | Flagship Conservation Programme | 0.02 | 200 | 0.00 | 160 | 0.60 |

Source: claude_build/config/env/allee_ricker_pomdp_116_10a.yaml:10-20.

Theta and regime use identical tables. Negative cost is harvest revenue, so subtracting cost adds immediate reward. PLUS and MOOR also apply h/delta before their assumed Ricker growth.

Generic ricker.yaml/ricker_full.yaml omit h and delta, which parse as zero, and have weaker Delta r values (claude_build/config/env/ricker.yaml:11-18; ricker_full.yaml:18-29). They are not the deployed stress tables.

## 4. Reward and collapse

RewardContract signatures are:

    def immediate(self, state: int, action: int) -> float
    def __call__(
        self, state: int, action: int,
        next_state: Optional[int] = None
    ) -> float

The base reward is:

    R_base(x_t,a_t) = alpha*x_t/x_max - cost(a_t), alpha=1

It uses the current bin, not next bin (claude_build/src/environments/reward.py:73-115). Principal runs use x_max=100; w=2 uses x_max=500.

Collapse-sensitive reward is:

    R(x_t,a_t,x_{t+1}) =
        R_base(x_t,a_t)
        - 20 * I[x_{t+1} <= x_c and x_t > 0]

with x_c=5 for w=10 and x_c=25 for w=2. Code is claude_build/src/environments/stress_pomdp_envs.py:82-90; values are in the environment YAML reward blocks.

This is an entry penalty. On first raw x_next<=x_c, collapse_sensitive sets _collapsed=True and forces s=0,x=0. Later steps stay zero (stress_pomdp_envs.py:99-121). The x_t>0 condition prevents repeated -20 penalties, although normal action cost/revenue still applies in absorbing state.

All stress configs use terminate_on_extinction=false, so collapse is absorbing internally but evaluation continues to fixed horizon (stress_pomdp_envs.py:125-133). In base mode, threshold collapse conversion is disabled and raw dynamics may recover.

planning_reward_fn is next-state-aware only for collapse_sensitive (stress_pomdp_envs.py:53-61). MOPO, RefPlan and BA-MCTS score predicted/sampled next states; BA-MCTS samples before reward (claude_build/src/models/bamcts_adapter.py:286-320). PLUS and MOOR integrate collapse penalty under their own transition models (bioconserv18_plus_adapter.py:397-420; moor_adapter.py:749-770).


## 5. Data generation

### 5.1 Dataset and storage

The in-memory schema is (claude_build/src/interfaces/base_env.py:27-44):

    @dataclass
    class Dataset:
        states: np.ndarray
        actions: np.ndarray
        next_states: np.ndarray
        rewards: np.ndarray
        dones: np.ndarray

generate_dataset(self, n_transitions: Optional[int]=None, policy: str="random", seed: Optional[int]=None) -> Dataset allocates state/action/next-state int32 arrays, float32 rewards and bool dones (claude_build/src/environments/ricker_env.py:235-288). The compressed NPZ contains those five arrays plus scalar metadata_json (claude_build/scripts/core/generate_dataset.py:96-124).

Episode boundaries are retained only through dones. There is no episode_id, timestep, continuous abundance, hidden r_base, C, theta or regime in the agent dataset. MOPO uses dones to prevent a frame stack crossing an episode boundary (claude_build/src/training/offline_trainer.py:187-230).

### 5.2 mixed_danger_zone_116 policy

At each collection episode, one component is sampled (claude_build/src/environments/ricker_env.py:383-439):

| Component | Probability | Generic stress start |
|---|---:|---:|
| random | 0.25 | Uniform(60,240) |
| harvest_probe | 0.30 | Uniform(170,320) |
| rescue_dwell | 0.25 | Uniform(60,210) |
| threshold_probe | 0.20 | Uniform(60,220) |

Theta overrides the starts to (60,180), (60,200), (60,180), (60,180) respectively (claude_build/src/environments/stress_pomdp_envs.py:236-242). Component starts overwrite the YAML nominal collection start after reset; YAML still controls episode length.

Exact sub-policies are at claude_build/src/environments/ricker_env.py:327-381:

- random: uniform action.
- harvest_probe: above abundance 250, harvest with probability 0.85; otherwise support-dwell.
- rescue_dwell: at/below 250 use support; above 350 harvest with probability 0.45; otherwise do nothing with probability 0.60 and a uniform action with probability 0.40.
- threshold_probe: first eight steps alternate do-nothing and harvest; then support at/below 250, otherwise uniform.

For 10 actions, a harvest draw is ID 1 with probability 0.25 or ID 2 with 0.75. At abundance <=150, support draws IDs [3,4,5,6,8,9] with [.20,.20,.20,.20,.10,.10]; above 150 it draws [0,3,4,5,6] with [.30,.20,.20,.15,.15]. For 5 actions, low support draws [2,3,4] with [.45,.40,.15], otherwise [0,2,3] with [.35,.40,.25] (ricker_env.py:408-422).

### 5.3 Episode lengths, budgets and shared paths

Collection episode lengths are Allee 25, theta 20 and regime 30 (claude_build/config/env/allee_ricker_pomdp_116.yaml:31-36; theta_logistic_pomdp_116.yaml:31-36; regime_switch_pomdp_116.yaml:36-41). Evaluation starts are Uniform(80,250) with episode length 50 in all three files.

YAML development data size is 25,000, but final and w=2 manifests override it to 75,000. Every method in a scenario receives the identical path:

    outputs/stress_pomdp_116/shared_datasets/
      <phase>/<env>_<reward>_seed<seed>_n<data_n>.npz

Construction is at claude_build/scripts/experiments/make_stress116_manifest.py:165-188. Principal final has 60 datasets = 6 cells x 2 rewards x 5 seeds. The w=2 ablation has 20 = 4 cells x one reward x 5 seeds.

### 5.4 Calibration and seed protocol

The 116 collapse frequency is not constrained during rollout. It is an emergent result tuned through priors, starts, behavior mixture, action authority and collapse semantics, then checked on deterministic seeds.

The current Allee 116 test uses 5,000 transitions at seed 116 and requires (claude_build/tests/test_stress116_experiment.py:135-176):

- states in bins 6..20: 0.15 to 0.30;
- states in bins <=20: 0.25 to 0.45;
- states in bin 0: below 0.30;
- episode collapse rate: 0.10 to 0.35;
- one-step action next-bin spread from s=120: at least 8 bins.

An older generic test targets 0.02 to 0.25 under mixed_danger_zone, not mixed_danger_zone_116 (claude_build/tests/test_stress_pomdp_envs.py:212-234).

| Role | Seeds | Purpose |
|---|---|---|
| gate calibration | 116 | Gate; episode seeds 116..135. |
| method tuning | 1160,1161,1162 | Select one MOPO and one RefPlan setting. |
| held-out final | 7001,7002,7003,7004,7005 | Never used for tuning. |

Within top-level seed q, evaluation uses q,q+1,...,q+49 for every method (claude_build/src/evaluation/evaluator.py:95-110). This pairs methods, but adjacent top-level seeds create overlapping episode-seed ranges. That is the exact current protocol; replace it if independent final seed blocks are required.

## 6. Methods

### 6.1 MOPO

Implementation signatures:

    class CategoricalMOPOEnsemble(BaseModel):
        def __init__(self, cfg: Any) -> None
        def predict(
            self, history: List[Tuple[int,int]]
        ) -> Tuple[np.ndarray,float]
        def predict_batch(
            self, states: np.ndarray, actions: np.ndarray
        ) -> Tuple[np.ndarray,np.ndarray]

(claude_build/src/models/ensemble.py:23-67 and :99-177). Five bootstrapped categorical MLPs map two discrete state/action frames to next-bin logits. Each member uses state embedding 32, action embedding 16, frame stack 2, three ReLU layers of width 256, and an N-bin output head (claude_build/src/models/network.py:20-100). Uncertainty is variance across members of expected next-bin index.

OfflineTrainer.fit(self, dataset: Dataset) -> Dict[str,List[float]] trains each member by cross entropy on an 80% bootstrap sample with replacement (claude_build/src/training/offline_trainer.py:31-105 and :111-181). Frozen training: ensemble 5, stack 2, bootstrap 0.8, batch 512, Adam lr 3e-4, weight decay 1e-5, 50 epochs, cosine schedule, warmup 2 (claude_build/config/model/ensemble.yaml:4-32).

Planner signature:

    class PessimisticPlanner:
        def __init__(self, cfg: Any) -> None
        def plan(
            self,
            history: List[Tuple[int,int]],
            model: BaseModel,
            reward_fn: Optional[Any] = None,
        ) -> int

(claude_build/src/planners/pessimistic_planner.py:35-74). It scores every first action by model rollout and subtracts lambda*Var_member(E[x_next]) each step. Production batches candidate actions x rollouts (pessimistic_planner.py:248-320), preferably using torch.func.vmap across members (ensemble.py:179-200). Frozen planning is H=5, 25 rollouts/action, lambda=0.5, gamma=0.95, greedy and vectorized (claude_build/outputs/stress_pomdp_116/frozen/phase116_frozen_1day.env:2-5).

### 6.2 RefPlan

Signatures:

    class RefPlanAdapter(BaseModel):
        def __init__(self, cfg: Any) -> None
        def fit_offline(
            self, dataset: Any, train_cfg: Optional[Any] = None,
            wandb_cfg: Optional[Any] = None,
            seed: Optional[int] = None,
            reward_fn: Optional[Any] = None,
        ) -> Dict[str,Any]
        def predict(
            self, history: List[Tuple[int,int]]
        ) -> Tuple[np.ndarray,float]
        def select_action(
            self, history: List[Tuple[int,int]],
            planner: Any = None, reward_fn: Any = None
        ) -> int

(claude_build/src/models/refplan_adapter.py:33-158). This is a finite-state repo adaptation, not a continuous VAE/MPPI reproduction. A 15-member bootstrap transition ensemble represents latent models. Exact observed transition likelihoods update the member posterior from public history. It samples behavior-prior action sequences, rolls each under latent members, scores mean(return)-uncertainty_penalty*std(return), softmaxes kappa*score and selects the first action with greatest mass (refplan_adapter.py:126-158 and :296-333).

Frozen: M=15, bootstrap 0.8, prior count 0.05, behavior smoothing 0.1, likelihood floor 1e-12, gamma 0.95, H=8, 512 sequences, 12 latent samples/sequence, kappa=10, uncertainty penalty 0.2 and terminal-value weight 1.0 (claude_build/config/model/refplan.yaml:9-27; frozen env:6-11).

### 6.3 BA-MCTS

Signatures:

    class BayesAdaptiveMCTSAdapter(BaseModel):
        def __init__(self, cfg: Any) -> None
        def fit_offline(
            self, dataset: Any, train_cfg: Optional[Any] = None,
            wandb_cfg: Optional[Any] = None,
            seed: Optional[int] = None,
            reward_fn: Optional[Any] = None,
        ) -> Dict[str,Any]
        def predict(
            self, history: List[Tuple[int,int]]
        ) -> Tuple[np.ndarray,float]
        def select_action(
            self, history: List[Tuple[int,int]],
            planner: Any = None, reward_fn: Any = None
        ) -> int

(claude_build/src/models/bamcts_adapter.py:33-150). It shares RefPlan's bootstrapped tabular posterior, but runs posterior-state MCTS. Simulations select by UCB, sample member and next bin, update posterior, subtract Bellman-target disagreement, recurse, and use posterior-averaged member VI at leaves (bamcts_adapter.py:255-346).

Deployed untuned values: M=15, bootstrap 0.8, prior count 0.05, behavior smoothing 0.1, likelihood floor 1e-12, gamma 0.95, 128 simulations, depth 5, exploration 1.25, pessimism 0.10, tree belief rounded to 2 decimals (claude_build/config/model/bamcts.yaml:10-28). This is a finite-state adaptation, not the paper's neural stack.

### 6.4 BioConserv18 PLUS

Exact constructor:

    class BioConserv18PLUSAdapter(BaseModel):
        def __init__(
            self,
            cfg: Any,
            env: Optional[Any] = None,
            env_cfg: Optional[Any] = None,
            reward_contract: Optional[RewardContract] = None,
            planning_reward_fn: Optional[Any] = None,
        ) -> None

(claude_build/src/models/bioconserv18_plus_adapter.py:44-51). It creates 21 evenly spaced r_base candidates over the active env range with uniform prior (bioconserv18_plus_adapter.py:356-378). Each candidate yields a finite Ricker transition/Q table. Exact observed (x,a,x') likelihoods update the candidate posterior; action maximizes posterior-weighted Q (:200-253).

Stress dispatch forces use_env_transition=false, so all true models are planned as Ricker. Parameters: 21 candidates, gamma 0.95, 2,000 sub-bin points, floors 1e-12, VI tolerance 1e-8, max 1,000 iterations (claude_build/scripts/slurm/stress_pomdp_116.sh:356-373; config/model/bioconserv18_plus.yaml:12-32).

### 6.5 ExpertSys23 MOOR-Ricker

Exact constructor:

    class MOORAdapter(BaseModel):
        def __init__(
            self,
            cfg: Any,
            env: Optional[Any] = None,
            env_cfg: Optional[Any] = None,
            reward_contract: Optional[RewardContract] = None,
            planning_reward_fn: Optional[Any] = None,
        ) -> None

(claude_build/src/models/moor_adapter.py:78-85). It converts bins to midpoint proxies (x+0.5)w, fits one Ricker model's r and K by normalized least squares, sweeps 2,000 subpoints/bin into one transition table, then uses finite VI (moor_adapter.py:476-541 and :655-722). Direct h/delta is applied before fitted growth.

Stress settings: family ricker, learn [r,K], 30 L-BFGS-B restarts, 20 iterations/restart, jitter 0.25, gamma 0.95, grid 2,000, transition noise sigma 0, likelihood floor 1e-12, and best-5-restart next-bin variance as uncertainty (claude_build/config/model/moor.yaml:15-53; stress_pomdp_116.sh:375-396).

### 6.6 Shared tabular model and tuning

RefPlan/BA use:

    TabularDynamicsEnsemble.__init__(
        num_states: int,
        num_actions: int,
        ensemble_size: int = 15,
        prior_count: float = 0.05,
        bootstrap_ratio: float = 0.8,
        behavior_smoothing: float = 0.1,
        likelihood_floor: float = 1e-12,
        seed: Optional[int] = None,
    )

It allocates [M,S,A,S] counts/transitions and updates b'(m) proportional to b(m)P_m(x'|x,a) (claude_build/src/models/tabular_mbrl.py:215-320).

Declared tuning grids (claude_build/scripts/experiments/make_stress116_manifest.py:26-39):

- RefPlan: H in {8,10,12}; sequences {256,512,1024}; latent samples {8,12,16}; kappa {2,5,10}; uncertainty penalty {0.05,0.10,0.20}.
- MOPO: H {5,8}; rollouts {20,25,50}; lambda {0.5,1,2}; epochs {50,100}.

The deployed one-day tuning used a deterministic matched subset of six configs/method over all six cells and seeds 1160-1162, hence 18 seed-scenarios/config. Exact rows are outputs/manifests/stress116_tune_methods_1day.tsv; selection/default inclusion is at make_stress116_manifest.py:93-143. Frozen values were selected before final seeds and stored in outputs/stress_pomdp_116/frozen/phase116_frozen_1day.env:1-11. PLUS, MOOR and BA-MCTS were not tuned.

## 7. Evaluation and gate

### 7.1 Unified evaluator and metrics

Signatures:

    class UnifiedEvaluator:
        def __init__(
            self,
            env: BaseEnvironment,
            models: List[BaseModel],
            planner: Optional[PessimisticPlanner] = None,
            eval_cfg: Optional[Any] = None,
            wandb: Optional[Any] = None,
        ) -> None

        def run(self, seed: int = 42) -> pd.DataFrame

(claude_build/src/evaluation/evaluator.py:47-54 and :95-118). Final runs use 50 episodes/top-level seed, 5 seeds, horizon 50 and gamma=0.95. Return is sum gamma^t R_t (evaluator.py:198-232).

Every episode row contains (evaluator.py:234-266):

| Field | Meaning |
|---|---|
| model, episode, seed | Model name, zero-based episode, environment seed. |
| cumulative_reward | Discounted return. |
| mean_abundance | Mean observed bin, including initial. |
| final_abundance | Last observed bin. |
| min_abundance | Minimum observed bin. |
| catastrophic_low_abundance | float(min_abundance <= 5.0), hard-coded. |
| collapse_entry | Any info collapse_entered. |
| collapse_entries | Entry count. |
| collapse_entry_timestep | First entry step or NaN. |
| policy_entropy | Entropy of model.predict next-state probabilities for the chosen action. This is predictive transition entropy, not action-policy entropy. |
| state_coverage | Distinct visited bins / model.num_states. |
| uncertainty_mean, uncertainty_max | Mean/max method-reported uncertainty; scales differ by method. |
| n_steps | Executed transitions. |
| live_danger_fraction | Fraction of current states inside configured danger bins. |
| live_danger_action_a_frac | Conditional action fractions in danger. |
| hidden diagnostics | C/theta bucket or majority regime, collapse state/count, reward mode. |

Hidden-bucket JSON reports mean return, min abundance and collapse-entry by C/theta/regime bucket (evaluator.py:322-351). Catastrophic bin 5 is correct for w=10 but is not rescaled to bin 25 for w=2. Default danger bins 6..20 similarly have physical meaning only at w=10 (evaluator.py:78-89).

The harness samples GPU name/memory/utilization every 5 s and writes status, elapsed seconds, GPU name and peak memory to out_root/metadata/method.json (claude_build/scripts/slurm/stress_pomdp_116.sh:161-180 and :250-263). WandB receives episode aggregate statistics and summary keys (evaluator.py:353-407).

### 7.2 Seed aggregation and claims

scripts/analysis/aggregate_stress116_phase116.py averages episodes within each top-level seed and collects hidden buckets/runtime (:74-160), then reports mean/std across seeds (:190-219). Learned methods are mopo, refplan, bamcts; fixed-family baselines are plus, moor_ricker (:19-30).

Learned advantage per scenario/seed is:

    G_method - max(G_PLUS, G_MOOR-Ricker)

implemented at :172-187. For collapse_sensitive cells, claim logic tests whether the best learned method beats both baselines, also reports each learned method's individual rate, and compares RefPlan versus MOPO in Allee/regime cells (:222-271).

### 7.3 Decision-relevance control gate

scripts/analysis/stress116_control_gap.py compares:

- oracle MPC: deep-copy actual initialized env and exhaustively evaluate action sequences, using true hidden parameters/dynamics (:36-56);
- Ricker MPC: reconstruct abundance from bin midpoint, use active r prior midpoint, apply identical action authority, propagate compensatory Ricker and score identical reward (:59-104).

Both use identical episode seeds. Defaults: 20 episodes, horizon 50, gamma 0.95, exhaustive MPC H=4, seed 116, reward gap >=1.0 and collapse-rate gap >=0.05 (stress116_control_gap.py:220-285).

    reward_gap = mean(G_oracle) - mean(G_Ricker)
    collapse_gap = collapse_rate_Ricker - collapse_rate_oracle

Definitions are at :188-217. Allee/regime hard-gate at (1.0,0.05); theta is diagnostic at (0.5,0.0) and nonblocking (claude_build/scripts/slurm/stress_pomdp_116.sh:266-286).

| Cell | Reward gap | Collapse gap | Policy |
|---|---:|---:|---|
| Allee 5a | 2.4911546366 | 0.10 | hard pass |
| Allee 10a | 5.8645415719 | 0.25 | hard pass |
| Regime 5a | 2.8734346512 | 0.10 | hard pass |
| Regime 10a | 7.0536483233 | 0.30 | hard pass |
| Theta 5a | 0.6415701219 | 0.00 | diagnostic pass |
| Theta 10a | 2.7926297086 | 0.05 | diagnostic pass |

Exact JSONs are at outputs/stress_pomdp_116/<env>_collapse_sensitive_seed116/control_gap/control_gap_summary.json:1-20.


## 8. Assumptions to replace for the continuous, noisy-observation setting

### 8.1 Discrete-bin and finite-state assumptions

1. **Environment API types.** BaseEnvironment.reset/step return integer bins and reward accepts integer states (claude_build/src/interfaces/base_env.py:47-92). Return continuous observation, while latent true state stays private.
2. **State conversion.** _discretize, _bin_for_abundance, num_states and bin thresholds are finite-state machinery (claude_build/src/environments/ricker_env.py:317-325). Remove them from the true environment; use abundance-valued thresholds.
3. **Dataset.** Dataset states/next_states are integer arrays (base_env.py:27-34). Replace with observations, next_observations, actions, rewards, dones and preferably episode_id/timestep. Never expose true s_t or hidden parameters in the agent artifact.
4. **MOPO representation.** State embeddings, categorical logits, cross entropy, expected-bin uncertainty and categorical sampling assume finite bins (claude_build/src/models/network.py:20-100; ensemble.py:99-177; offline_trainer.py:140-153). Use a continuous probabilistic model over filtered latent state or shared belief features. Keep bootstrap epistemic uncertainty.
5. **MOPO planner.** Expected-bin calculations, rounding/clipping and categorical sampling assume integers (claude_build/src/planners/pessimistic_planner.py:135-177 and :248-320). Propagate continuous particles/samples and compute imagined reward from imagined observations.
6. **RefPlan/BA posterior.** TabularDynamicsEnsemble stores dense [M,S,A,S] arrays and uses exact P_m(x'|x,a) (claude_build/src/models/tabular_mbrl.py:215-320). Replace with continuous transition-model posterior likelihood marginalized through the state filter and observation kernel.
7. **RefPlan planning.** behavior_policy[state], categorical imagined states and tabular terminal values are discrete (claude_build/src/models/refplan_adapter.py:296-333). Replace with belief-conditioned proposals, continuous latent rollouts and a continuous/belief terminal value.
8. **BA-MCTS tree.** Keys contain integer state plus rounded member posterior, and leaves use finite-MDP VI tables (claude_build/src/models/bamcts_adapter.py:255-346). Use particle-belief summaries and continuous generative transitions.
9. **PLUS.** It materializes [M,S,A,S] and candidate Q tables; exact bins remove state belief in the current adapter (claude_build/src/models/bioconserv18_plus_adapter.py:35-41 and :200-253). Add the shared state filter, observation likelihood p(o_next | history,a,model), and belief-space planning. Retain Ricker structural candidates only if intentional misspecification remains the comparison.
10. **MOOR.** Bin midpoints, sub-bin grids and finite VI are core assumptions (claude_build/src/models/moor_adapter.py:476-541 and :677-722). Fit continuous latent dynamics under observation uncertainty and replace the table with a belief-space planner/approximation.
11. **Evaluator.** Mean/final/min abundance currently mean bin IDs; coverage and danger/catastrophe are bin-based (claude_build/src/evaluation/evaluator.py:234-269). Define public metrics from physical observations and diagnostic safety metrics from private latent state, clearly labeled.
12. **Gate.** Ricker MPC reconstructs a bin midpoint and clips to finite geometry (claude_build/scripts/analysis/stress116_control_gap.py:59-79). Replace it with filtered Ricker MPC receiving the same noisy observation history. Keep true-state oracle information diagnostic-only.
13. **Harness geometry inference.** The shell infers NUM_STATES from suffixes _bw2/_bw5 and NUM_ACTIONS from _10a (claude_build/scripts/slurm/stress_pomdp_116.sh:120-131). A continuous harness must not synthesize a finite state count.

### 8.2 Exact, noise-free observation assumptions

Current observation is deterministic x=floor(s/w). The evaluator feeds x directly to methods (claude_build/src/evaluation/evaluator.py:148-225). RefPlan, BA-MCTS and PLUS update model belief using exact next-bin transition likelihood and keep no belief over current ecological state. MOPO's frame stack contains exact bins. PLUS/MOOR explicitly justify finite-MDP planning because abundance is exact (claude_build/config/model/bioconserv18_plus.yaml:1-6; config/model/moor.yaml:1-9). Dataset validation also recomputes reward from stored bins (claude_build/scripts/core/run_pipeline.py:59-90).

Introduce one shared observation kernel, for example:

    class LogNormalObservationModel:
        def sample(
            self, state: float, rng: np.random.Generator
        ) -> float: ...
        def log_prob(
            self, observation: float, state: ArrayLike
        ) -> ArrayLike: ...

It must implement o_t=s_t*eta_t, eta_t~LogNormal(0,sigma_obs^2), and explicitly handle s_t=0 because a usual log-normal density has positive support. The environment stores s_t, r_base, C/theta/z privately, returns only o_t, and exposes truth only as diagnostics after the action decision.

Introduce one shared filter used by all methods:

    class StateBeliefFilter(Protocol):
        def reset(
            self, observation: float, episode_seed: int
        ) -> BeliefState: ...
        def predict(
            self, belief: BeliefState, action: int, model: Any
        ) -> BeliefState: ...
        def update(
            self, predicted: BeliefState, observation: float,
            action: int, model: Any
        ) -> BeliefState: ...
        def features(self, belief: BeliefState) -> np.ndarray: ...
        def sample(
            self, belief: BeliefState, n: int, rng: Any
        ) -> np.ndarray: ...

Separate state belief from structural/model belief. Bayes-adaptive joint posterior includes current s_t, per-episode r_base, C or theta, and current z_t for regime-switching. Every method, including PLUS and MOOR, must receive the same filter implementation, initial prior and observation history.

### 8.3 Finite-MDP/value-iteration assumptions

Shared value_iteration and dense transitions are used by RefPlan member values, BA-MCTS leaves, PLUS candidate policies and MOOR. Enlarging those tables would simply reintroduce the discretization the successor is intended to remove.

The replacement planning boundary should be belief-state-based, for example:

    def select_action(
        self,
        observation_history: Sequence[TransitionObservation],
        belief: BeliefState,
        planner: Optional[Any] = None,
        reward_fn: Optional[Any] = None,
    ) -> int

MOPO can retain sampled batched MPC with continuous ensemble rollouts. RefPlan can sample latent dynamics/parameter particles and trajectories. BA-MCTS can use particle beliefs in tree nodes. PLUS and MOOR require a real filter plus observation-model adaptation; retaining finite-MDP Q tables is not a faithful noisy-observation baseline.

Delphic and OGSRL are not currently registered. Add each as a BaseModel-compatible adapter, Hydra model YAML, registration in src/models/__init__.py, continuous harness dispatch, manifest/report labels, WandB naming and tests. Both must consume the same shared belief representation.

### 8.4 s_max ceiling assumptions

The current fixed ceiling s_max=1000 enters:

- direct-action and true-growth clipping;
- top-bin discretization;
- reward normalization via x_max;
- PLUS/MOOR sub-bin transition grids;
- MOOR fit normalization and parameter bounds;
- control-gate Ricker propagation;
- state coverage, danger/catastrophe metrics;
- plots and bin-midpoint conversions.

The target is unbounded. Remove hard clipping from action and true dynamics. Keep numerical safeguards that do not alter valid dynamics, such as exponent-domain checks or log-domain calculation. Replace x/x_max by an explicitly documented observation-scale reward because reward now uses o_t. Decide whether it saturates, normalizes by K_base, or remains unbounded. Safety thresholds must be in physical latent or observed-abundance units, never top-bin indices.

### 8.5 Concrete extension points

| Current point | Continuous successor change |
|---|---|
| src/interfaces/base_env.py | Add continuous observation/dataset protocols; separate latent state from observation. |
| src/environments/ricker_env.py | Reuse parameter sampling, actions and equation; remove discretization/ceiling; invoke observation kernel on reset/step. |
| src/environments/stress_pomdp_envs.py | Reuse hidden C/theta/regime logic; define collapse diagnostics in physical units and reward in observed units. |
| src/environments/reward.py | Replace integer RewardContract with observation-valued contract and explicit noisy-threshold semantics. |
| new src/beliefs/ | Shared state/parameter filter and log-normal observation likelihood. |
| src/models/ensemble.py and network.py | Continuous probabilistic dynamics conditioned on belief features. |
| src/planners/pessimistic_planner.py | Preserve batched ensemble x action x rollout acceleration; propagate continuous samples. |
| src/models/refplan_adapter.py | Joint state/parameter particles and non-tabular terminal value. |
| src/models/bamcts_adapter.py | Particle-belief tree nodes and continuous generative model. |
| src/models/bioconserv18_plus_adapter.py | Shared filter, p(o|s) likelihood, belief planning; preserve candidate Ricker bias if desired. |
| src/models/moor_adapter.py | Continuous latent fit under observation noise, filter and non-tabular belief planning. |
| src/evaluation/evaluator.py | Feed only observation/belief; separate observed reward metrics from private true-state diagnostics. |
| scripts/analysis/stress116_control_gap.py | Matched noisy-observation oracle/misspecified control; no midpoint/bin propagation. |
| manifests/aggregation | Add sigma_obs, filter config, continuous env key, Delphic and OGSRL; preserve shared data and held-out pairing. |

## 9. Reuse versus replace for the continuous setting

| Component | Reuse | Replace or extend |
|---|---|---|
| Biological dynamics | Four equations and per-episode hidden-parameter sampling. | Remove finite ceiling/bin conversion; continuous numerical safeguards. |
| Actions | IDs and calibrated (Delta r,Delta K,h,delta,cost); authority before growth. | Remove s_max clipping; validate unbounded behavior. |
| POMDP discipline | Hidden truth stays internal; diagnostics appear after decisions. | Add noisy observation and explicit shared state/parameter belief. |
| Reward | One contract threaded through env/planners/baselines. | Compute from o_t; replace bin normalization/thresholds. |
| Data flow | Shared dataset per scenario, done boundaries, held-out seeds. | Continuous observations and explicit episode structure; no inaccessible truth fields. |
| MOPO | Bootstrap ensemble, epistemic uncertainty and batched pessimistic MPC. | Continuous density model, belief input and continuous rollout. |
| RefPlan | History-conditioned reflection and uncertainty-aware sequences. | Particle joint posterior and non-tabular value. |
| BA-MCTS | Bayes-adaptive tree/posterior/pessimism. | Particle-belief nodes and observation branching. |
| PLUS/MOOR | Their Ricker-family inductive bias as ecology baselines. | Mandatory shared filter, observation likelihood and belief planning. |
| Evaluation | Paired methods, seed aggregation, hidden diagnostics, runtime/GPU, WandB. | Physical metrics, independent seed blocks, corrected entropy name/threshold scaling. |
| Control gate | Oracle-versus-misspecified decision relevance. | Matched noisy-observation belief controllers. |
| Registry/manifests | Hydra, registry and manifest architecture. | Continuous adapters plus Delphic/OGSRL and updated claim logic. |
