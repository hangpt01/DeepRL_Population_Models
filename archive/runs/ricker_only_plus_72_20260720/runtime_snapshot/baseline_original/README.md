# Historical baseline: universal two-state solver

This directory preserves the code for the paper *A universal 2-state n-action
adaptive management solver*. It is historical/reference code; the active
continuous-state ecology benchmark lives at the repository root.

## Requirements

- **Python 3** with packages listed in `python_port/requirements.txt`.
- **SARSOP** (`pomdpsol`): build or obtain a binary and place it under
  `sarsop/src/` (see [SARSOP](https://github.com/AdaCompNUS/sarsop)).

The original R implementation has been removed; the workflow lives in `python_port/` (see `python_port/README.md`).

## Quick start

Set the example module and run the pipeline from this directory:

```bash
cd baseline_original
export EXAMPLE_MODULE=gouldian
python "python_port/src/building hmMDP/main.py"
python "python_port/src/solving hmMDP/main.py"
python "python_port/src/simulations/main.py"
python "python_port/src/analyse results/main.py"
```

Or use helpers:

```bash
bash scripts/run_python_version.sh gouldian
bash python_port/scripts/run_python_benchmarks.sh gouldian
bash scripts/run_compare_examples.sh
```

## Folder layout

**`data/`**: inputs and solver outputs used in the original paper:

- `gouldian4Exp.pomdpx`: hmMDP instance assessed by four experts.
- **`POLICYX/`**: `*.policyx` files from SARSOP.

Example definitions for the Python code are in **`python_port/data/*.py`**; each
exports a `CONFIG` dict.

**`res/`**: generated outputs (mean parameters, POMDPX, simulation CSVs, performance summaries).

**`python_port/`**: Python source mirroring the former R layout (`src/building hmMDP`, `src/simulations`, etc.).

**`sarsop/`**: SARSOP sources / `pomdpsol` binary used by the solving step.
