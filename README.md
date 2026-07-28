# DeepRL Population Models

Research code and provenance for the continuous-observation ecological offline-RL
benchmark. This repository preserves the accepted ecological and general-method
implementations as two separate import tracks. They must not be merged: accepted
PLUS/MOOR cells and accepted RefPlan/OGSRL/BA-MCTS/EVD cells were produced by
different frozen source trees.

## What is here

- `src/tracks/ecological/`: byte-preserved accepted environment and PLUS/MOOR track.
- `src/tracks/general/`: byte-preserved accepted general-method track.
- `src/diagnostics/replay_analysis/`: M1–M15 analysis and its 36 assertions
  (the Phase-1 audit described an earlier 32-assertion minimum).
- `scripts/diagnostics/`: parity replay, S1, S2, reward-screen, H12/H14/S6 tooling.
- `configs/ecology/`: the authoritative 11-action and species tables.
- `results/`: the accepted 144-cell table and approved small diagnostic artifacts.
- `provenance/`: source paths, hashes, dataset identities, and audit records.
- `archive/`: superseded code, working history, and the debugging trail; authored
  archive content is tracked by Git.

The large artifact-only archive is retained locally under `archive/artifacts/` but
is intentionally excluded from Git history. Raw datasets, fit caches, model blobs,
and active run outputs remain at their recorded scratch locations.

## Installation

The paper-faithful environment used Python 3.10.14 and CPU-only PyTorch
2.13.0+cpu. The pins in `requirements.txt` are the exact `pip freeze --all`
export from `.venv-paper-faithful`; its first line adds the official PyTorch CPU
wheel index required by the `+cpu` pin.

```bash
module load miniforge3/24.3.0-0
/apps/miniforge3/24.3.0-0/miniforge3/bin/python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e '.[analysis]'
```

Use one scientific track at a time:

```bash
export PYTHONPATH="$PWD/src/tracks/ecological"
# or
export PYTHONPATH="$PWD/src/tracks/general"
```

The duplicate package name is deliberate. Selecting both tracks simultaneously is
unsupported because it would make import order decide scientific behavior.

Both packages expect repository-root `real_ecology_data`. The tracked symlink
`src/tracks/real_ecology_data -> ../../configs/ecology` restores that location
without modifying or duplicating the frozen packages. Verification entry points
check it at startup and report a clear error if it is missing.

## Reproduction

The complete instructions are in [REPRODUCE.md](REPRODUCE.md). With the original
scratch data and fit caches mounted:

```bash
make verify \
  SCRATCH_PROJECT=/home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models \
  PYTHON=python
```

`make verify` runs:

1. the M1–M15 synthetic self-tests (all 36 current assertions);
2. the constants check, including the vulture bound and dominated-action sets;
3. a negative test proving a corrupted comparison exits non-zero;
4. an A6-MOOR accepted-cell replay with the enforced seven-field `1e-9` parity,
   cache-reuse, and measured `recomputed_fits=0` gates.

The general accepted path is independently available with:

```bash
make verify-cell-general PYTHON=python
```

It defaults to the fast B1-EVD accepted cell and uses the repository’s 576-row
general manifest.

## How the project works

The environment has four hidden demographic families—Ricker, Allee, theta-logistic,
and regime-switching—an 11-action management table, species-specific `K_ref` and
`s_safe`, a true-next-state reward, and `ContinuousEvaluator`.

The scientific workflow treats explanation as part of the implementation:

- accepted policies are replayed with side-effect-free instrumentation;
- every replay is gated on seven accepted fields;
- shadow rollouts assert that state and RNG streams are unchanged;
- demographic fits must be cache hits (`recomputed_fits=0`);
- constant-action and S1 references test whether learning adds value;
- M1–M15, S2, reward screening, H12, H14, and S6 test the proposed mechanisms.

Audits, handoffs, specifications, and the decision log remain under `docs/`. The
curated public/internal reading order is in [docs/INDEX.md](docs/INDEX.md).

## Repository status

The MIT text is present, but its adoption remains provisional pending final owner
confirmation; see [LICENSE_STATUS.md](LICENSE_STATUS.md).
