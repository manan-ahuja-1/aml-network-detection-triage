# AML Laundering-Network Detection & Triage
#
# Every target runs through the project venv explicitly rather than relying on an
# activated shell, so `make train` behaves identically whether or not you remembered
# to `source .venv/bin/activate`.
#
# NOTE: this repo lives in a path containing spaces. All paths here are RELATIVE for
# that reason — GNU Make cannot handle spaces in target or prerequisite paths.

PY := ./.venv/bin/python

.PHONY: help check data features graph embeddings train eval figures kb agent-eval all clean fix-libomp test

# Default target: running bare `make` prints the menu rather than doing something
# unexpected and expensive.
help:
	@echo ""
	@echo "  make check       Verify interpreter, dependencies and directories"
	@echo "  make fix-libomp  Repair LightGBM's OpenMP link (macOS, no Homebrew)"
	@echo "  make data        Build the parquet cache and freeze the temporal split"
	@echo "  make features    Build all four feature arms"
	@echo "  make graph       Build the cached graph-topology features (~8 min)"
	@echo "  make embeddings  Build the cached node2vec embeddings"
	@echo "  make train       Train the model arms and score on validation"
	@echo "  make eval        Freeze the engine, score TEST once, write results/engine.json"
	@echo "  make figures     Redraw the result charts (cached scores after first run)"
	@echo "  make kb          Build the ChromaDB index over kb/corpus (~1 min)"
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
	$(PY) src/make_data.py
	$(PY) src/make_splits.py

features:
	$(PY) src/build_features.py

graph:
	$(PY) src/features_graph.py

embeddings:
	$(PY) src/embeddings.py

train:
	$(PY) src/train.py --arms A,B,C

# Trains arm C (~10 min), scores val and TEST, bootstraps CIs, sweeps the cost ratio,
# computes pattern-level recall and SHAP, and retrains arms A and B for the PR-curve
# figure. The frozen booster is cached in models/, so a second run skips the training.
eval:
	$(PY) src/evaluate.py

# Redraws results/figures/. The first run retrains arms A and B (~13 min) and caches
# their validation scores; after that it is instant. --rebuild forces a retrain.
figures:
	$(PY) src/make_figures.py

# Indexes the curated AML corpus. The embedder (ONNX all-MiniLM-L6-v2) runs locally,
# so this needs no API key -- only the first run downloads the ~41 MB model.
kb:
	$(PY) src/kb_index.py --rebuild

agent-eval:
	@echo "NOT IMPLEMENTED: src/agent/evaluate_agent.py (Day 8)" && exit 1

# The reproducibility claim in the README rests on this target: raw data in,
# every reported number out, no manual steps.
all: check data features train eval figures kb agent-eval test

test:
	$(PY) -m pytest tests/ -q

# NOTE: this deletes data/processed/split_boundaries.json, the "frozen" split.
# That is safe because make_splits.py is deterministic — same input data and same
# fractions reproduce byte-identical boundaries. The split is frozen in the sense
# that nothing RECOMPUTES it mid-pipeline, not in the sense that it is unrecoverable.
# NOTE: models/ is removed too -- the frozen engine is regenerable from `make eval`,
# and a stale booster paired with fresh features is exactly the kind of mismatch the
# provenance guards exist to prevent.
clean:
	rm -rf data/processed/* data/chroma/* results/figures/* results/*.json models/*
	@echo "Removed derived data and results. Raw downloads in data/raw/ kept."
