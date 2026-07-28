# Architecture

The package is intentionally self-contained inside this repository: runtime code
is under `src/real_ecology_benchmark/`, and the real data tables are the sibling
`real_ecology_data/` directory.

```text
configs/                  YAML experiment defaults
scripts/                  manifest and Slurm entry points
src/real_ecology_benchmark/
  actions.py              locked five/ten action tables
  config.py               validated dataclass configuration
  envs.py                 four continuous latent dynamics
  observation.py          exact log-normal emission kernel
  reward.py               public and true evaluator reward
  dataset.py              public trajectories/private truth sidecars
  collector.py            privileged-state behavior mixture
  beliefs.py              reference, learned, mechanistic PF proposals
  dynamics.py             bootstrap continuous dynamics ensemble
  planning.py             shared particle MPC
  methods/                 seven policy implementations
  evaluator.py            paired seed evaluator
  gate.py                 three-controller decision gate
  realdata.py             real ecology CSV accessors
  dummydata.py            in-code set-point r + cumulative K dummy profile
  manifest.py             synthetic and real matrix generation
  pipeline.py / cli.py    end-to-end orchestration
tests/                    standard-library unittest suite
```

## Benchmark Axes

`data_mode` says where the ecological cell comes from. `control_mode` says how
an action changes the population dynamics.

| `data_mode` | Data source | Valid `control_mode` | Meaning |
| --- | --- | --- | --- |
| `real` | CSV ecology tables in `real_ecology_data/` | `setpoint_cumulative` | action-specific set-point `r`; cumulative `K` |
| `dummy` | in-code dummy ecology profile | `setpoint_cumulative` | same semantics as real, but small and synthetic |
| `synthetic` | in-code continuous-state simulator | `tier2_one_step`, `cumulative_capped` | one-step or cumulative-control synthetic experiments |

Reader-facing names should describe the setting rather than use old development
labels:

| Reader-facing name | Current code/config identifier | What distinguishes it |
| --- | --- | --- |
| discretized-state synthetic benchmark | archived `claude_build/` lineage | synthetic data with discretized states |
| continuous-state one-step synthetic benchmark | `data_mode: synthetic`, `control_mode: tier2_one_step` | continuous latent abundance; one-step action effects |
| continuous-state cumulative-control synthetic benchmark | `data_mode: synthetic`, `control_mode: cumulative_capped` | continuous latent abundance; public cumulative control state |
| real-ecology set-point benchmark | `data_mode: real`, `control_mode: setpoint_cumulative` | real ecology tables; action sets `r`, accumulates `K` |
| dummy-ecology set-point benchmark | `data_mode: dummy`, `control_mode: setpoint_cumulative` | small in-code profile for quick experiments with real-like action semantics |

`real_setpoint` is accepted only as a legacy alias for old configs and metadata.
New configs and generated metadata use `setpoint_cumulative`.

## Information boundary

The public dataset contains only observation, action, reward, next observation, done, episode ID, and timestep. Truth is serialized to a separate file that no training API accepts. The collapse-entry penalty is intentionally observable through reward and is the only truth-derived public bit.

Cumulative-control synthetic rows additionally expose only manager-owned public
controls: `rho`, `kappa`, public `K_eff = clip(K_base + kappa, K_min, K_max)`,
and their next-step values. They never expose `r_eff`: it depends on hidden
`r_base`, and in theta cells the cap also depends on hidden `theta`. True
abundance, hidden episode parameters, regime, `r_eff_true`, and private noise
remain evaluator/private-only.

## Shared filter

The reference PF validates the known emission using a weak log-state proposal. The primary learned proposal is fitted once from public trajectories and is shared across methods. Mechanistic proposals and the PLUS candidate filter bank are baseline-fidelity ablations. Fixed offline belief trajectories can be cached; evaluation beliefs cannot be shared because policies choose different actions.

The NumPy learned proposal and dynamics ensemble are deliberately compact and auditable. The dynamics ridge basis includes action-by-state interactions rather than treating action as an intercept only. Their interfaces permit replacement by neural latent-state models without changing datasets, policies, the evaluator, or the gate.

Under `control_mode=cumulative_capped`, multi-step lookahead is Markov in the
augmented rollout state `(s, context, regime, rho, kappa)`, not in abundance
alone. The shared planner, BA-MCTS rollout, OGSRL rollout, and learned dynamics
thread these public accumulators through hypothetical actions. The old
`tier2_one_step` internal path remains the one-step synthetic default and is
pinned by a golden regression test. Real and dummy ecology cells share
`control_mode=setpoint_cumulative`, where actions set `r` directly and update
`K` cumulatively.

## Numerical policy

The true simulator has no abundance ceiling. It evaluates Ricker-family exponentials in log space and raises on overflow rather than silently clipping state. Theta-logistic applies only the non-negativity floor. Learned models may bound internal numerical transforms; these are approximators and do not alter simulator physics.
