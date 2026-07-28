.PHONY: test test-real test-dummy test-synthetic smoke smoke-real smoke-dummy smoke-synthetic dataset gate docs docs-check

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

test-real:
	PYTHONPATH=src python -m unittest discover -s tests/real -v

test-dummy:
	PYTHONPATH=src python -m unittest discover -s tests/dummy -v

test-synthetic:
	PYTHONPATH=src python -m unittest discover -s tests/synthetic -v

smoke:
	PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/synthetic_default.yaml

smoke-synthetic:
	PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/synthetic_default.yaml

smoke-real:
	PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/real_smoke.yaml

smoke-dummy:
	PYTHONPATH=src python -m real_ecology_benchmark.cli smoke --config configs/dummy_setpoint_smoke.yaml

dataset:
	PYTHONPATH=src python -m real_ecology_benchmark.cli generate --config configs/synthetic_default.yaml

gate:
	PYTHONPATH=src python -m real_ecology_benchmark.cli gate --config configs/synthetic_default.yaml

docs:
	@set -e; out="$${TMPDIR:-/tmp}/ecology_benchmark_docs"; mkdir -p "$$out"; for tex in docs/benchmark/0*.tex; do pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$$out" "$$tex" >/dev/null; done; echo "built docs into $$out"

docs-check:
	PYTHONPATH=src python scripts/check_docs.py
