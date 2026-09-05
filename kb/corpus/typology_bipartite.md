---
source_id: ibm-amlsim-typologies
title: "Typology: BIPARTITE"
chunk_type: typology
typologies: [BIPARTITE]
---

# BIPARTITE

**Shape.** Two disjoint sets of accounts with flow only from one set to the other —
every source pays some subset of the destinations, and no account appears on both sides.

**Real-world correspondence.** A mule network operating as an organised layer: a set of
controlled source accounts feeding a set of controlled receiving accounts, with the
many-to-many wiring making any single relationship look unremarkable. It is the shape that
most resembles ordinary commercial activity, which is exactly why it is used.

**Matching regulatory red flags.**
- FFIEC Appendix F, shell companies: "Multiple high-value payments or transfers between
  shell companies with no apparent legitimate business purpose."
- FFIEC Appendix F, shell companies: "Transacting businesses share the same address,
  provide only a registered agent's address, or have other address inconsistencies" — the
  entity-level analogue of a shared-control signal.
- FinCEN FIN-2020-A003 on money mule networks.

**Model features that fire.** No single account in a bipartite structure is extreme on any
per-account statistic, which is the difficulty. Detection depends on community structure:
`louvain_community_size` and `wcc_size` capture that the participants belong to one dense
cluster, and `core_number` is elevated because the many-to-many wiring means participants
survive repeated degree-peeling. This is the clearest case where a groupby cannot
substitute for graph structure.

**Disposition note.** Supply-chain and marketplace payment flows are genuinely bipartite.
The signal is not the shape alone but the shape combined with accounts that have no
history outside the cluster.
