# AML Laundering-Network Detection & Triage — 2-Week Build Plan

**Working title:** `Laundering-Net` (rename to taste — "MuleGraph", "AlertPilot", etc.)

**One-line pitch:** A system that detects money-laundering / mule-account networks using graph-derived features on top of a gradient-boosted model, then uses an LLM agent to triage flagged alerts and auto-draft investigator-ready case narratives — with a full eval harness.

**Why this project:** Fraud/AML is *the* existential operational problem for every fintech (payments, neobanks, crypto). This build shows the whole stack an applied-AI/fintech engineer needs: real ML (not just API calls), the graph insight that laundering is relational, and the agent layer that turns a raw alert into a decision. Plus rigorous evals — the thing 90% of portfolio projects skip.

---

## The dataset

**Primary: IBM Transactions for Anti-Money Laundering (AML)**
`kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml`

- Synthetic, account-to-account transactions with a per-transaction `Is Laundering` label.
- Comes in variants: **HI-Small / HI-Medium / LI-*** (HI = higher illicit ratio, easier to work with; LI = more realistic, harder). **Use HI-Small** for a 2-week local build — it's manageable on a Mac and the illicit ratio is workable.
- It's explicitly built for GNNs and network analysis, so the graph structure is genuine, not bolted on.

**Backups if you want alternatives:** SAML-D (larger, 28 typologies) or PaySim (mobile-money framing, but labels are theft-fraud not laundering). Stick with IBM unless you hit a wall.

**⚠️ Size warning:** Even HI-Small is large. If graph construction or node2vec is slow, **subsample to the last ~500k transactions** (this is standard — the AML network-analytics literature does exactly this). Don't try to run node2vec on millions of nodes; you'll burn a day watching a progress bar.

---

## Environment (Mac)

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install pandas numpy lightgbm scikit-learn networkx node2vec \
             matplotlib seaborn anthropic chromadb streamlit kaggle
```

Get the data: set up the Kaggle API token (`~/.kaggle/kaggle.json`) and `kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, or just download the HI-Small CSV manually from the site.

**Stack:** LightGBM (engine) · NetworkX + node2vec (graph features) · scikit-learn (metrics) · Claude API + ChromaDB (agent + RAG) · Streamlit (demo) · matplotlib/seaborn (eval charts).

---

## The two components

1. **Detection engine** — LightGBM classifier that scores each transaction/account for laundering risk, using transaction features **+ graph-network features**.
2. **Triage agent** — takes a high-risk alert from the engine, pulls the entity's transaction subgraph, retrieves relevant AML typologies (RAG), classifies the pattern (fan-in / fan-out / layering / cycle), and drafts an investigator case note with a disposition recommendation.
3. **Eval harness** — the differentiator. Rigorous metrics for *both* halves.

---

## Week 1 — The engine + evals (the ML half; this is where the learning curve lives)

### Day 1 — Setup + data understanding
- Venv, install, download HI-Small, load in pandas.
- EDA: understand the schema (sender/receiver accounts, amount, currency, payment format, timestamp, `Is Laundering`).
- **Measure and note the class imbalance ratio** — you'll cite it later; it justifies every downstream choice.
- Decide your graph framing: **account nodes, transaction edges** (sender → receiver). Sketch it on paper.

### Day 2 — Baseline model (your control)
- Feature-engineer *transaction-level* features only: amount, currency, payment format (one-hot), hour/day-of-week from timestamp, simple per-account aggregates (count, sum, mean of recent transactions).
- Train LightGBM. **Handle imbalance** (class weights or `scale_pos_weight`, not naive oversampling).
- **Evaluate with PR-AUC, not ROC-AUC** — ROC-AUC lies on severe imbalance. Record baseline PR-AUC. *This number is the thing your graph features will beat.*

### Day 3 — Build the graph + classical network features
- Build the transaction graph in NetworkX.
- Compute per-account/per-transaction network features: **in/out degree, degree centrality, betweenness (on a subsample if slow), connected-component membership, community detection (Louvain)**, and simple structural flags (fan-in count, fan-out count, whether it sits in a cycle).
- These features *are* the mule/laundering signal — structuring, layering, and smurfing all show up as graph topology.

### Day 4 — node2vec embeddings + integration
- Run node2vec on the (subsampled) graph to get node embeddings; attach them as features.
- Retrain LightGBM with **transaction features + graph features + embeddings**.
- **Measure the lift over your Day-2 baseline.** This comparison is your headline result.

### Day 5 — Eval harness + operating threshold
- Full evaluation of the engine:
  - PR curve, precision/recall at your chosen threshold, confusion matrix.
  - **Cost-sensitive threshold selection** — assign a (rough, stated) cost to a missed laundering case vs a false-positive review, and pick the threshold that minimizes expected cost. This shows you think like a compliance ops team, not a Kaggler.
  - **Feature importance** — show graph features ranking high. Great visual, great story.
- Lock the engine and freeze its numbers.
- *Weekend = buffer if any of Days 3–4 overran.*

**⚠️ Time-leakage guardrail:** use a **time-based train/test split** (train on earlier transactions, test on later). If you split randomly, graph features leak future edges into training and your metrics come out fraudulently good — a reviewer will spot it instantly. A time split is both correct and something you can talk about intelligently.

---

## Week 2 — The agent layer + polish (your comfort zone; this moves fast)

### Day 6 — AML knowledge base + RAG
- Curate a small corpus of AML typologies and red-flag indicators: FATF red flags, mule-account patterns, structuring/smurfing/layering definitions. A dozen good pages is plenty.
- Index it in ChromaDB (you've built RAG before — reuse the pattern).

### Day 7 — Triage agent core
- Input: a high-risk account/transaction surfaced by the engine.
- Agent pulls the entity's **subgraph context** (counterparties, flow shape) and retrieves the relevant typologies via RAG.
- Output via **structured outputs**: risk rationale, pattern classification (fan-in / fan-out / layering / cycle), confidence, disposition recommendation.

### Day 8 — Case-narrative generation + light agentic loop
- Turn the structured triage into an **investigator-ready case note / SAR-style narrative**, citing the specific evidence (which transactions, which pattern, which typology matched).
- Add one genuine agentic step: the agent decides *what extra context to pull* before finalizing (e.g., "counterparty X also flagged — pull its subgraph"). Keep it bounded.

### Day 9 — Agent evals (rare and impressive — most agent projects have none)
- Build a small labelled set of alerts (use ground-truth laundering labels you already have).
- **Triage-decision accuracy:** does the agent's escalate/dismiss call match ground truth? Report precision/recall on *that*.
- **Narrative quality:** a short rubric — correct typology, evidence completeness, and **no hallucinated facts** (check every claimed transaction actually exists). Report the score.

### Day 10 — Integration + demo + writeup
- Wire engine → agent into one flow: model flags → agent triages → case note out.
- Minimal **Streamlit demo** (or a clean end-to-end notebook): pick a flagged account, watch it get triaged and written up.
- Write the **README**: problem, architecture diagram, the metrics, the graph-feature lift, and the story. Push to GitHub with clean commits.
- *Weekend = buffer, or start the GNN v2 stretch (below) only if you're ahead.*

---

## Metrics to report (fill in your real numbers — never fabricate)

| Metric | Why it matters |
|---|---|
| Class imbalance ratio | Shows you understand the problem shape |
| **PR-AUC: baseline vs +graph features** | **The headline. The lift is your whole thesis.** |
| Precision / recall at operating threshold | The real-world operating point |
| Cost-sensitive threshold rationale | You think like an ops team |
| Top features by importance | Graph features ranking high = visual proof |
| Agent triage-decision accuracy | You evaluated the agent, not just vibes |
| Narrative quality + hallucination rate | Production rigor on the LLM half |
| Est. analyst-time reduction | Ties it to business value |

---

## The CV bullet

Fill in `[X]` / `[Y]` with your actual results:

> Built an end-to-end AML laundering-detection system on synthetic transaction data: engineered graph-network features (node2vec, centrality, community detection) that lifted PR-AUC from **[X]** to **[Y]** over a transaction-only baseline, then wrapped the model in an LLM agent that triages flagged alerts and auto-drafts investigator case narratives — evaluated with a harness measuring precision/recall, cost-sensitive thresholds, and triage-decision accuracy.

Shorter version if space is tight:

> AML laundering-network detector: graph features (node2vec + centrality) lifted PR-AUC **[X]→[Y]** vs baseline; LLM agent triages alerts and drafts case notes, with full precision/recall + agent-accuracy evals.

---

## The 60-second interview story

1. **Problem** — Mule accounts and laundering networks are the operational nightmare for fintechs. The pain isn't just *catching* it; it's the false-positive deluge drowning compliance teams.
2. **Insight** — Laundering is inherently relational. You catch it in the *graph*, not the individual transaction. So I engineered network features.
3. **Result** — Graph features lifted PR-AUC from X to Y over a transaction-only baseline. *(This is your money quote — lead with it.)*
4. **Second half** — Detection is only half the job; a human still reviews every alert. So I built an agent that triages the alert and drafts the case note, cutting analyst time.
5. **Rigor** — I measured everything: PR-AUC not ROC because of imbalance, cost-sensitive thresholds because a missed launder ≠ a false alarm in cost, a time-based split to avoid leakage, and I evaluated the agent's decisions against ground truth.
6. **Fintech kicker** — This mirrors how real fintech fraud stacks actually work: gradient boosting + graph features + human-in-the-loop review. Pure GNNs are the exception, not the rule.

---

## Stretch (only if ahead of schedule)

**GNN v2:** swap the boosted engine for a GraphSAGE / GAT model in PyTorch Geometric and compare it head-to-head with the boosting+features approach. Now you've *earned* the GNN on top of a finished product — and "I compared classical graph features vs a GNN and here's the tradeoff" is a stronger, more senior story than either alone. Do **not** let this eat into the core 10 days.

---

## The three things that will make or break it

1. **Subsample early.** Don't fight dataset size — cap it and move on.
2. **Time-based split.** Random splits leak and produce fake-good numbers a reviewer will catch.
3. **Ship the eval harness.** It's the single feature that separates this from every "I built a chatbot" project. Non-negotiable.
