---
source_id: ibm-amlsim-typologies
title: "Typology: SCATTER-GATHER"
chunk_type: typology
typologies: [SCATTER-GATHER]
---

# SCATTER-GATHER

**Shape.** One source disperses funds across many intermediary accounts, which then
reconsolidate into a single destination. Fan-out followed by fan-in, with a set of
parallel one-hop intermediaries in between.

**Real-world correspondence.** Deliberate layering: the split-and-rejoin exists only to
break the direct link between origin and destination. Unlike gather-scatter, the
intermediaries here are pass-throughs with a single payer and a single payee each, which
is a strong signature because almost no legitimate account behaves that way.

**Matching regulatory red flags.**
- FinCEN SAR narrative guidance names "complex layering activities" as an activity type to
  identify at the outset of a narrative.
- FFIEC Appendix F: "Payments or receipts with no apparent links to legitimate contracts,
  goods, or services are received."
- FFIEC Appendix F: "Funds transfers contain limited content and lack related party
  information."

**Model features that fire.** The intermediaries show `in_n_counterparties == 1`,
`out_n_counterparties == 1`, `flow_through_ratio` near 1 and very short
`median_hours_to_forward`. The origin and destination show high out- and in-degree
respectively. On the graph the intermediaries carry non-zero `betweenness` despite tiny
degree, and the whole structure usually lands in one `louvain_community_size` cluster.

**Disposition note.** The parallel-path structure is hard to explain innocently, which
makes this typology one of the higher-precision detections — but the individual
intermediary accounts look almost empty in isolation, so evidence has to be assembled at
the ring level rather than the account level.
