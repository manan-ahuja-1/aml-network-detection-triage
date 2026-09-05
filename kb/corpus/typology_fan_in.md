---
source_id: ibm-amlsim-typologies
title: "Typology: FAN-IN"
chunk_type: typology
typologies: [FAN-IN]
---

# FAN-IN

**Shape.** Many distinct source accounts pay into one destination account over a short
window — the mirror of fan-out.

**Real-world correspondence.** This is the **funnel account**, and FinCEN has a formal
definition for it (Advisory FIN-2014-A005): "An individual or business account in one
geographic area that receives multiple cash deposits, often in amounts below the cash
reporting threshold, and from which the funds are withdrawn in a different geographic area
with little time elapsing between the deposits and withdrawals." It is also the collection
point of a mule network and the aggregation half of smurfing.

**Matching regulatory red flags.**
- FinCEN FIN-2014-A005, the whole funnel-account red-flag list.
- FFIEC Appendix F: "Many small, incoming transfers of funds are received... Almost
  immediately, all or most of the transfers or deposits are wired to another city or
  country in a manner inconsistent with the customer's business or history."
- FFIEC Appendix F, terrorist financing: "Multiple personal and business accounts... are
  used to collect and funnel funds to a small number of foreign beneficiaries."

**Model features that fire.** High `to_in_n_counterparties` and `to_in_n` over few
`to_in_active_days`; `to_in_counterparties_per_txn` near 1 (each payer appears once, which
is what distinguishes a funnel from a busy merchant); elevated `to_pagerank`, because
inbound money-weighted flow is exactly what PageRank accumulates; high
`to_in_n_banks` when the payers span institutions.

**Disposition note.** Merchants, landlords, charities and utilities all receive from many
unrelated payers. The funnel signature is the *combination* with rapid onward movement —
high `to_flow_through_ratio`, low `to_median_hours_to_forward` — and with payers who have
no other visible activity.
