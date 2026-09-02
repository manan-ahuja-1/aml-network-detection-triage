# AML Laundering-Network Detection & Triage
#
# Every target runs through the project venv explicitly rather than relying on an
# activated shell, so `make train` behaves identically whether or not you remembered
# to `source .venv/bin/activate`.
#
# NOTE: this repo lives in a path containing spaces. All paths here are RELATIVE for
# that reason — GNU Make cannot handle spaces in target or prerequisite paths.

PY := ./.venv/bin/python

.PHONY: help check data features train eval agent-eval all clean fix-libomp

# Default target: running bare `make` prints the menu rather than doing something
# unexpected and expensive.
help:
	@echo ""
	@echo "  make check       Verify interpreter, dependencies and directories"
	@echo "  make fix-libomp  Repair LightGBM's OpenMP link (macOS, no Homebrew)"
	@echo "  make data        Download the dataset and build the parquet cache"
	@echo "  make features    Build all four feature arms"
	@echo "  make train       Train arms A-D"
	@echo "  make eval        Evaluate the engine, write results/engine.json"
	@echo "  make agent-eval  Evaluate the triage agent, write results/agent.json"
	@echo "  make all         Full pipeline from raw data to results"
	@echo "  make clean       Remove derived data and results (keeps raw downloads)"
	@echo ""

check:
	$(PY) src/check_env.py

fix-libomp:
	$(PY) scripts/fix_macos_libomp.py

# --- pipeline stages -------------------------------------------------------
# Each stage is a stub that FAILS LOUDLY until implemented. A target that silently
# succeeds without doing anything is worse than no target: `make all` would report
# success having produced nothing.

data:
	@echo "NOT IMPLEMENTED: src/make_data.py (Day 1)" && exit 1

features:
	@echo "NOT IMPLEMENTED: src/features_*.py (Days 2-4)" && exit 1

train:
	@echo "NOT IMPLEMENTED: src/train.py (Days 2-4)" && exit 1

eval:
	@echo "NOT IMPLEMENTED: src/evaluate.py (Day 5)" && exit 1

agent-eval:
	@echo "NOT IMPLEMENTED: src/agent/evaluate_agent.py (Day 8)" && exit 1

# The reproducibility claim in the README rests on this target: raw data in,
# every reported number out, no manual steps.
all: check data features train eval agent-eval

clean:
	rm -rf data/processed/* results/figures/* results/*.json
	@echo "Removed derived data and results. Raw downloads in data/raw/ kept."
