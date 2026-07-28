# Benchmark Documentation Bundle

This directory is the current, reader-facing documentation for the continuous
state ecological offline-RL benchmark. It is written for a reader who starts with
zero repository history and wants to understand the problem, the data, the
algorithm adaptations, the implementation, and the recent real-ecology
experiment.

## Reading Order

1. `01_problem_setting_and_design.tex` defines the POMDP and the scientific
   design choices.
2. `02_real_ecology_data_actions_and_costs.tex` explains the six real-data CSVs,
   population/action tables, growth conversions, and cost provenance.
3. `03_state_action_reward_reference.tex` is the canonical state, action, reward,
   and metric reference.
4. `04_algorithm_adaptations_and_claims.tex` explains what each named method
   implements here and what claims are valid.
5. `05_implementation_and_code_map.md` maps the scientific concepts to code.
6. `06_experiment_protocol_and_reproducibility.md` gives the run protocol.
7. `07_recent_real_ecology_results.md` summarizes the P-safe experiment.
8. `08_limitations_tracker.md` tracks aggregate limitations and future work.

## Source-Of-Truth Hierarchy

1. Code and CSVs are authoritative: `../../src/real_ecology_benchmark/` and
   `../../real_ecology_data/`.
2. This bundle is the current reader-facing explanation.
3. Retired design specs in `../history/` and `../real_ecology_history/` may be
   mined for context, but they are not current truth. Some of them predate the
   `data_mode` split, the `setpoint_cumulative` rename, and the P-safe decision.

One source of truth per fact: do not duplicate the same table in Markdown and
LaTeX. Markdown files link to the paper-bound LaTeX reference when a table is
already defined there.

## Scope

The bundle focuses on the active continuous-state benchmark. The recent-results
document covers the real-ecology P-safe experiment in
`../../real_ecology_runs/psafe_overnight_20260705/`. Older synthetic
stress-benchmark reports are archived in `../history/22_6_Continuous_Observation_New_Baselines.tex`
and `../history/22_6_Continuous_Observation_Experiment_Results.tex`; they are
provenance, not the current results layer.

## What The Benchmark Claims

The benchmark compares control strategies under a shared ecological POMDP, data
format, evaluator, gate, and seed protocol. It supports real ecology data, a
small dummy ecology profile with the same action semantics, and continuous-state
synthetic stress settings.

The method names identify benchmark-native adaptations. The implementations use
NumPy, ridge-linear models, mechanistic proposals, particle filters, and particle
MPC. They should be read as comparisons of strategies such as pessimism,
posterior planning, finite mechanistic model banks, conservative value fitting,
and guarded safe learning under a common ecological model class.

Two additional hidden-demographics diagnostic methods are registered but have
not yet produced an accepted experiment: `plus_adapted_mechanistic_pbvi` and
`moor_adapted_ricker_misspec_pbvi`. They are explicitly adapted mechanistic
baselines, not exact reproductions of the published PLUS or MOOR algorithms.
Their corrected canary protocol is maintained under
`../fix_implement_ecology_baseline/`.

## Glossary

`data_mode`: the data source. `real` uses `../../real_ecology_data/`; `dummy`
uses the in-code profile in `dummydata.py`; `synthetic` uses the older
continuous-state simulator settings.

`control_mode`: the action-control semantics. `setpoint_cumulative` is the
canonical mode for real and dummy ecology: the action sets the public growth
set-point and accumulates carrying-capacity increments. `tier2_one_step` and
`cumulative_capped` are legacy code identifiers for synthetic settings; in prose
call them one-step synthetic control and cumulative-control synthetic.

`real_setpoint` is accepted only as a legacy config alias for `setpoint_cumulative`.

`reward_mode`: `safe` uses the configured collapse penalty; `yield` sets the
collapse penalty to zero for set-point ecology cells. Synthetic configs also
retain historical `observed` and `belief_expected` modes.

`rho`: public growth set-point carried in the dataset/control state.

`kappa`: public carrying-capacity accumulator.

`K_eff`: public effective carrying capacity,
`clip(K_base + kappa, K_min, K_max)`.

`s_safe` / `safety_threshold`: the population-specific safety floor used by the
reward penalty and unsafe-occupancy metrics.

`mvp_threshold`: the absolute minimum-viable-population diagnostic threshold, 50
individuals by default.

`collapse`: entry into or occupancy of the below-safety region, depending on the
configured penalty mode. Exact abundance zero is terminal extinction.

`occupancy penalty`: charges every step whose next true state is below the
safety floor.

`crossing penalty`: charges only a downward transition from above to at/below
the safety floor.

`public information`: observations, actions, rewards as logged targets, public
controls, done flags, episode IDs, and timesteps. Training APIs consume only the
public dataset.

`evaluator-only information`: true abundance, hidden structural parameters,
regime, true effective growth, and private simulator diagnostics.

`gate`: a decision-relevance check comparing controllers under the same
particle-MPC engine before expensive sweeps.

`manifest`: a CSV of experiment rows.

`cell` / `row`: a cell fixes a population/family/noise/reward setting; a row is a
method/filter/backend run inside a cell.

`recoverable population`: a population whose strongest recovery action gives
positive Ricker growth (`r_max_ricker > 0`).

`sink population`: a population whose strongest recovery action does not make
Ricker growth positive (`r_max_ricker <= 0`). The current real data have two:
Egyptian vulture and bottlenose dolphin.

## Reader-Facing Terms

| Reader-facing term | Code/config identifier | Meaning |
| --- | --- | --- |
| real-ecology set-point benchmark | `data_mode: real`, `control_mode: setpoint_cumulative` | CSV-backed real populations, action sets `r`, action accumulates `K` |
| dummy-ecology set-point benchmark | `data_mode: dummy`, `control_mode: setpoint_cumulative` | in-code toy profile with real-like action semantics |
| one-step synthetic control | `data_mode: synthetic`, `control_mode: tier2_one_step` | synthetic continuous abundance with one-step action effects |
| cumulative-control synthetic | `data_mode: synthetic`, `control_mode: cumulative_capped` | synthetic continuous abundance with public cumulative controls |

Verified against: README.md, docs/README.md, docs/architecture.md, src/real_ecology_benchmark/config.py, src/real_ecology_benchmark/realdata.py, src/real_ecology_benchmark/dummydata.py, real_ecology_runs/psafe_overnight_20260705/analysis/PAPER_RESULT_PACKAGE.md @ be96c36
