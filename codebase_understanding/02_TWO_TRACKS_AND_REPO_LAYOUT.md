# Two frozen tracks and repository layout

## Import rule

`src/tracks/ecological` and `src/tracks/general` each contain a package named
`real_ecology_benchmark`. Put exactly one track on `PYTHONPATH`; if both are
present, Python imports whichever path is first, silently mixing the requested
experiment with the wrong registry. The `Makefile` demonstrates the safe rule in
`test-ecological` and `test-general`.

`src/tracks/real_ecology_data` is a symlink to `../../configs/ecology`.
`realdata.py:DATA_DIR` resolves the sibling data directory used by both tracks.
Removing/breaking the symlink prevents `load_actions()`, `load_pops()`, and
`load_effects()` from resolving the scientific CSV inputs.

## Complete `diff -rq` result

Command:

```bash
diff -rq --exclude=__pycache__ src/tracks/ecological src/tracks/general
```

The command reports exactly 14 divergent/one-sided paths.

| Path | Difference and scientific consequence |
|---|---|
| `behavior_model.py` | General-only calibrated behavior prior; general RefPlan uses `fit_reference_behavior()`. |
| `general_canary_acceptance.py` | General-only receipt validation; experiment acceptance infrastructure, not ecology. |
| `methods/ensemble_value_disagreement.py` | General-only bootstrap ridge-Q ensemble used for accepted EVD. |
| `config.py` | General has OGSRL 25-step normalized-cost/deployment settings; ecological accepts the Ricker-only faithful bank. |
| `faithful_fit.py` | Ecological supports `RICKER_ONLY_CANDIDATE_CONSTRUCTION`; general permits only cross-form construction. |
| `manifest.py` | General default names EVD; ecological default names historical `delphic`. |
| `public_models.py` | General adds common-random-number sampling and posterior-marginalized/prior-guided public planning. |
| `training_monitor.py` | General imports shared `behavior_model.augment_features`; ecological imports `_augment` from its Delphic implementation. |
| `methods/__init__.py` | Ecological registers `plus_adapted_ricker_only_pbvi` and its own `DelphicCQLPolicy`; general registers EVD plus deprecated alias. |
| `methods/bamcts.py` | General hidden BA-MCTS maintains a public transition-likelihood model belief; ecological’s older hidden simulation uses a member index directly. |
| `methods/delphic.py` | Ecological contains a CQL-style historical class; general is an alias shim to EVD. This prevents attributing latent-confounder Delphic semantics to accepted EVD. |
| `methods/ogsrl.py` | General adds normalized occupancy, train-only low-abundance scale, CRN pathwise deployment risk, and the 25-step hidden actor objective. |
| `methods/plus_faithful.py` | Ecological adds the eight-candidate Ricker-only PLUS class used in accepted results; general exposes only cross-form PLUS. |
| `methods/refplan.py` | General fits a public behavior prior and makes the posterior affect marginalized plan values. |

All other shared files are byte-identical, including the load-bearing
`envs.py`, `evaluator.py`, `reward.py`, `actions.py`, `collector.py`, and
`pipeline.py`. Verify with the diff command, not documentation.

## Which track produced accepted results

`results/accepted/MATCHED_P10_144_METHOD_CELLS.csv` contains:

- ecological track: `moor_adapted_ricker_misspec_pbvi` and
  `plus_adapted_ricker_only_pbvi` (the latter is absent from the general
  registry and present in ecological `methods/__init__.py`);
- general track: `refplan`, `ogsrl`, `bamcts`, and
  `ensemble_value_disagreement_pessimism`.

The accepted receipt records hashes of the common evaluator/environment/reward
files and exact dataset hashes per physical cell
(`results/accepted/MATCHED_P10_144_RECEIPT.json:comparability_audit`).

## Top-level ownership

| Directory | Role |
|---|---|
| `src/tracks` | Frozen executable scientific packages; primary authority. |
| `configs/ecology` | Scientific input tables: populations, actions, effects, cost sources. |
| `configs`, `experiments/*/configs` | Runtime defaults/accepted overlays. |
| `experiments/*/manifests` | Actual accepted row specifications. |
| `scripts/{general,ecological}` | Grid builders, row runners, Slurm launchers. |
| `scripts/diagnostics`, `src/diagnostics` | Replay/follow-up instrumentation and M1–M15 analysis. |
| `results`, `.verification` | Copied results and local verification output. |
| `provenance` | Hash ledgers, receipts, frozen snapshot registry. |
| `tests` | Track-separated structural and scientific regression tests. |
| `docs`, `.review` | Claims/history and independent review; never code authority. |
| `archive` | Superseded code/artifacts/history; do not use for current runs. |

`methods/delphic.py` in general is a deprecated alias; synthetic control paths
in `actions.py`/`config.py` remain regression fixtures but are not accepted real
cells. Archived scripts are superseded. The apparent `plus.py`,
`plus_native.py`, and `plus_faithful.py` families are live but distinct:
observation-space fitted forms/MPC, tabular fitted solver, and mechanistic
candidate PBVI respectively.

