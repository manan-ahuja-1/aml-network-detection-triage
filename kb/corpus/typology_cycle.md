---
source_id: ibm-amlsim-typologies
title: "Typology: CYCLE"
chunk_type: typology
typologies: [CYCLE]
---

# CYCLE (simple cycle)

**Shape.** Money leaves an account and returns to it after passing through a chain of
intermediaries: A -> B -> C -> A.

**Real-world correspondence.** Round-tripping. The purpose is to give funds an apparent
transactional history and a plausible origin — the money comes back looking like revenue,
a loan repayment, or an investment return, which places it at the *integration* stage.

**Matching regulatory red flags.**
- FFIEC Appendix F: "Funds transfer activity is unexplained, repetitive, or shows unusual
  patterns."
- FFIEC Appendix F, lending activity: transactions that "tend to obscure the movement of
  funds (e.g., loans made to a borrower and immediately sold to an entity related to the
  borrower)."
- FFIEC Appendix F, shell companies: "Multiple high-value payments or transfers between
  shell companies with no apparent legitimate business purpose."

**Model features that fire.** This is the one typology with a direct, purpose-built graph
feature: `in_cycle` and `scc_size`, derived from strongly connected components. An account
in a non-trivial SCC is one money can return to. Only about 1.9% of nodes in the training
graph qualify, so it is a sparse, high-signal flag rather than noise.

**Caveat this project measures rather than assumes.** A *static* cycle is not the same as
money actually travelling A -> B -> C -> A in increasing time order. The graph here is
static, so `in_cycle` will also fire on accounts that merely have a mutual relationship
with a counterparty over the window. Time-respecting cycle detection was scoped and
deliberately deferred; the limitation is reported rather than papered over.
