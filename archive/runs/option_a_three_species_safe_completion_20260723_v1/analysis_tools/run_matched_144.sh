#!/usr/bin/env bash
set -euo pipefail
exec /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/.venv-paper-faithful/bin/python \
  /fs04/scratch2/ce25/Claude_DeepRL_Population_Models/real_ecology_runs/option_a_three_species_safe_completion_20260723_v1/analysis_tools/build_matched_144.py
