---
source_id: ibm-amlsim-typologies
title: "Typology: GATHER-SCATTER"
chunk_type: typology
typologies: [GATHER-SCATTER]
---

# GATHER-SCATTER

**Shape.** Funds are collected from many sources into one or a few intermediary accounts,
then redistributed outward to many destinations. Fan-in followed by fan-out through the
same hub.

**Real-world correspondence.** The full mule-network cycle: collection, consolidation,
re-dispersal. The intermediary is the account of interest — it is doing the laundering
work, and it is the account a case would be opened on.

**Matching regulatory red flags.**
- FFIEC Appendix F: "Many small, incoming transfers of funds are received... Almost
  immediately, all or most of the transfers or deposits are wired to another city or
  country."
- FinCEN FIN-2014-A005 funnel account definition, with the withdrawal side dispersed
  rather than consolidated.
- FFIEC Appendix F: "Funds transfer activity is unexplained, repetitive, or shows unusual
  patterns."

**Model features that fire.** This is the shape the typology features were built for. The
hub shows a `flow_through_ratio` near 1 (almost everything received leaves again), a low
`retention_ratio`, a short `median_hours_to_forward`, and high counterparty counts on
*both* sides. Graph-wise it carries high `betweenness`, because it sits on the paths
between the sources and the destinations — a quantity no per-account groupby produces,
which is why gather-scatter is one of the shapes arm C can see and arm B cannot.

**Disposition note.** Payment processors, remitters and pooled client accounts are
legitimately gather-scatter by design. Customer type is the discriminator, and a model
without KYC context cannot supply it.
