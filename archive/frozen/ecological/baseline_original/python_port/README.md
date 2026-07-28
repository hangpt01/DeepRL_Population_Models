# Python Port (excluding `sarsop/`)

This folder mirrors the original project structure and ports the R code to Python while preserving the same workflow and output paths.

## Structure

- `data/*.py`: example problem definitions (`CONFIG` dicts)
- `src/building hmMDP/`: MC-UAMS model generation and POMDPX writing
- `src/solving hmMDP/`: call SARSOP (`pomdpsol`) on generated POMDPX
- `src/Dirichlet solver/`: PUBD/Dirichlet baseline solver
- `src/simulations/`: simulation pipeline and helper functions
- `src/analyse results/`: performance aggregation
- `src/utils/`: shared helpers for config loading and MDP dynamic programming

## Run Pattern

Use `EXAMPLE_MODULE` to switch examples, e.g.:

```bash
EXAMPLE_MODULE=gouldian python "python_port/src/building hmMDP/main.py"
EXAMPLE_MODULE=gouldian python "python_port/src/solving hmMDP/main.py"
EXAMPLE_MODULE=gouldian python "python_port/src/simulations/main.py"
EXAMPLE_MODULE=gouldian python "python_port/src/analyse results/main.py"
```

## Notes

- This port intentionally reuses original output locations (`res/...`, `data/POLICYX/...`) for direct comparisons.
- `sarsop/` sources are untouched as requested.

## Python-only Baseline Benchmarking

To run MC-UAMS vs baselines (PUBD + Optimal) and save all artifacts only inside `python_port/`:

```bash
bash python_port/scripts/run_python_benchmarks.sh gouldian
```

Outputs are saved in:

- `python_port/results/<example>/<example>_meanparams.csv`
- `python_port/results/<example>/<example>.pomdpx`
- `python_port/results/<example>/<example>.policyx`
- `python_port/results/<example>/<example>_sim_mcuams.csv`
- `python_port/results/<example>/<example>_sim_pubd.csv`
- `python_port/results/<example>/<example>_sim_optimal.csv`
- `python_port/results/<example>/<example>_performance_summary.csv`
