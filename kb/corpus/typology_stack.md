---
source_id: ibm-amlsim-typologies
title: "Typology: STACK"
chunk_type: typology
typologies: [STACK]
---

# STACK

**Shape.** Layered bipartite structures in sequence: a set of accounts feeds a second set,
which feeds a third, and so on. Depth rather than breadth.

**Real-world correspondence.** Sustained layering — each additional hop puts another
institution, jurisdiction or account holder between the funds and their origin. Depth is
the point: an investigator must subpoena at every level, and if the levels span
institutions no single bank sees more than one hop.

**Matching regulatory red flags.**
- FinCEN SAR guidance: "unusual and/or complex series of transactions indicative of
  layering."
- FFIEC Appendix F: "Frequent involvement of multiple jurisdictions or beneficiaries
  located in higher-risk offshore financial centers."
- FFIEC Appendix F: "Funds transfer activity occurs to or from a financial institution
  located in a higher risk jurisdiction distant from the customer's operations."

**Model features that fire.** Intermediate-layer accounts show high `flow_through_ratio`,
low `retention_ratio` and short `median_hours_to_forward`. The multi-hop signature is
`betweenness`, which is elevated for every middle layer, and `core_number`. Cross-bank
movement shows in `is_same_bank == 0` and in `entity_n_banks`.

**Why this is the hardest typology for a single institution.** A stack is defined by hops
a single bank cannot observe. This is the motivating case for the single-bank realism
experiment: measuring how much detection power depends on network visibility no individual
institution has.
