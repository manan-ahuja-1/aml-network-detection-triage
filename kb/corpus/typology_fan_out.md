---
source_id: ibm-amlsim-typologies
title: "Typology: FAN-OUT"
chunk_type: typology
typologies: [FAN-OUT]
---

# FAN-OUT

**Shape.** One source account distributes funds to many distinct destination accounts over
a short window. In this dataset the injected instances are labelled with their degree,
e.g. "Max 16-degree Fan-Out".

**Real-world correspondence.** Distribution to a mule network, and the disbursement half
of smurfing. A single controlled account splitting a lump sum across many recipients is
how a consolidated balance is fragmented before onward movement, and if the individual
amounts sit below a reporting threshold it is structuring in the sense of 31 USC 5324.

**Matching regulatory red flags.**
- FFIEC Appendix F, shell company activity: "Unusually large number and variety of
  beneficiaries are receiving funds transfers from one company."
- FFIEC Appendix F, funds transfers: "Funds transfers are sent or received from the same
  person to or from different accounts."
- FFIEC Appendix G: structuring need not exceed $10,000 at any one bank on any single day.

**Model features that fire.** High `from_out_n_counterparties` and `from_out_n` against a
short `from_out_active_days`; elevated `from_out_n_currencies` when the split also crosses
currencies; low `from_retention_ratio`; the sender's `pagerank` stays modest because
outbound flow does not accumulate importance, so fan-out is carried mostly by arm B
degree-like counts rather than by multi-hop topology.

**Disposition note.** Payroll, supplier settlement runs and benefit disbursement all
produce fan-out. The discriminating question is whether the destination accounts have any
relationship to each other or to the sender's stated business, and whether the funds move
on again quickly.
