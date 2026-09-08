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
	@echo "  make single-bank How much of the network graph features need (~1h)"
	@echo "  make single-bank-replicates  Re-run the curve under two more edge draws"
	@echo "  make fx          Source the FX table and measure whether it moves the engine"
	@echo "  make agent-eval  Evaluate the triage agent, write results/agent.json"
	@echo "  make agent-run   COSTS MONEY. Run the agent over the test case queue"
	@echo "  make demo-bundle Export the case dossiers the demo reads"
	@echo "  make readme      Regenerate README.md from results/*.json"
	@echo "  make demo        Launch the Streamlit demo locally"
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

# Scores the TEST case queue once, both retrieval configurations, and writes
# results/agent.json. Needs results/triage_cases_test*.json -- see agent-run.
agent-eval:
	$(PY) src/agent/evaluate_agent.py --split test

# COSTS MONEY. Runs the triage agent over a case queue under the $10 ledger ceiling in
# src/agent/budget.py, which prices one real call and refuses to start if the batch
# would breach the remainder. Every reported agent number is already committed in
# results/, so `make all` does NOT call this.
agent-run:
	$(PY) src/agent/run_cases.py --split test
	$(PY) src/agent/run_cases.py --split test --no-rag

# B1: how much of the network graph features need. Rebuilds the graph at four
# visibility fractions and retrains each (~1h). Writes results/single_bank.json plus
# single_bank_scores.parquet, which the paired comparisons are computed from.
single-bank:
	$(PY) src/single_bank.py

# Seed replicates: re-draw WHICH EDGES ARE VISIBLE, so the curve's shape can be told
# apart from one unlucky subsample. Only the partial fractions are re-run -- arm B has
# no graph and the 100% arm does no sampling, so neither depends on the seed, and the
# floor is read back from single_bank_scores.parquet instead of retrained.
# Both seeds run concurrently: graph construction is single-threaded networkx.
single-bank-replicates:
	$(PY) src/single_bank.py --seed 7 & \
	$(PY) src/single_bank.py --seed 2024 & \
	wait
	$(PY) src/single_bank.py --aggregate

# Sources the FX table for 2022-09-01 and measures whether correcting it changes the
# engine, at inference and after a retrain (~2.6h, the Louvain pass dominates).
# --reverdict re-applies the materiality criterion to a saved run in a second.
fx:
	$(PY) src/diagnose_fx.py

# Exports the case dossiers the Streamlit demo reads, so app/ never touches data/
# or models/ -- both gitignored and absent on a deployment host.
demo-bundle:
	$(PY) src/export_demo.py

# Regenerates README.md from results/*.json. No number in the README is typed by
# hand; results/readme_provenance.json records the source path of every one.
readme:
	$(PY) src/make_readme.py

demo:
	./.venv/bin/streamlit run app/streamlit_app.py

# The reproducibility claim in the README rests on this target: raw data in,
# every reported number out, no manual steps. agent-run is excluded because it
# spends money; its outputs are committed.
all: check data features train eval figures kb single-bank single-bank-replicates \
     agent-eval demo-bundle readme test

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
	@echo "NOTE: results/spend_ledger.json is removed too -- restore it from git," \
	      "or the budget guard forgets what has already been spent."
	@echo "Removed derived data and results. Raw downloads in data/raw/ kept."
