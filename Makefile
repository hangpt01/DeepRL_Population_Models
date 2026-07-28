.PHONY: verify verify-self-tests verify-constants verify-integrity verify-negative-gate verify-cell \
	verify-cell-general verify-standalone-s2 test-ecological test-general

PYTHON ?= python3
SCRATCH_PROJECT ?= /home/hphung/ce25_scratch2/Claude_DeepRL_Population_Models
VERIFY_CELL ?= A6
VERIFY_SIDE ?= moor
VERIFY_GENERAL_CELL ?= B1
VERIFY_GENERAL_METHOD ?= ensemble_value_disagreement_pessimism

verify: verify-self-tests verify-constants verify-integrity verify-negative-gate verify-cell

verify-self-tests:
	cd src/diagnostics/replay_analysis && $(PYTHON) run_analysis.py --self-test

verify-constants:
	$(PYTHON) scripts/verify_constants.py

verify-integrity:
	$(PYTHON) scripts/verify_integrity.py

verify-negative-gate:
	@if $(PYTHON) scripts/verify_accepted_cell.py --gate-self-test-corrupt; then \
		echo "ERROR: corrupted parity comparison exited zero"; exit 1; \
	else \
		echo "PASS corrupted parity comparison exited non-zero"; \
	fi

verify-cell:
	$(PYTHON) scripts/verify_accepted_cell.py \
		--scratch-project "$(SCRATCH_PROJECT)" \
		--cell "$(VERIFY_CELL)" \
		--side "$(VERIFY_SIDE)"

verify-cell-general:
	$(PYTHON) scripts/diagnostics/replay/run_tier_b_replay.py \
		"$(VERIFY_GENERAL_CELL)" "$(VERIFY_GENERAL_METHOD)"

verify-standalone-s2:
	$(PYTHON) scripts/diagnostics/followups/run_s2.py --planning-only

test-ecological:
	PYTHONPATH=src/tracks/ecological $(PYTHON) -m unittest discover -s tests/ecological -v

test-general:
	PYTHONPATH=src/tracks/general $(PYTHON) -m unittest discover -s tests/general -v
