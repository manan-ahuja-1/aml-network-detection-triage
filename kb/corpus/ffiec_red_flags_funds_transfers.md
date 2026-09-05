---
source_id: ffiec-appendix-f
title: "Red flags: funds transfer activity"
url: https://bsaaml.ffiec.gov/manual/Appendices/07
chunk_type: red_flag
typologies: [FAN-IN, FAN-OUT, GATHER-SCATTER, SCATTER-GATHER, STACK, BIPARTITE]
---

# Red flags: funds transfer activity

The FFIEC BSA/AML Examination Manual states that "the mere presence of a red flag is
not by itself evidence of criminal activity. Closer scrutiny should help to determine
whether the activity is suspicious or one for which there does not appear to be a
reasonable business or legal purpose."

Verbatim indicators, Appendix F, "Potentially Suspicious Activity That May Indicate
Money Laundering — Funds Transfers":

- Many funds transfers are sent in large, round dollar, hundred dollar, or thousand
  dollar amounts.
- Funds transfer activity occurs to or from a financial secrecy haven, or to or from a
  higher-risk geographic location without an apparent business reason or when the
  activity is inconsistent with the customer's business or history.
- Funds transfer activity occurs to or from a financial institution located in a higher
  risk jurisdiction distant from the customer's operations.
- Many small, incoming transfers of funds are received, or deposits are made using
  checks and money orders. Almost immediately, all or most of the transfers or deposits
  are wired to another city or country in a manner inconsistent with the customer's
  business or history.
- Large, incoming funds transfers are received on behalf of a foreign client, with little
  or no explicit reason.
- Funds transfer activity is unexplained, repetitive, or shows unusual patterns.
- Payments or receipts with no apparent links to legitimate contracts, goods, or
  services are received.
- Funds transfers are sent or received from the same person to or from different
  accounts.
- Funds transfers contain limited content and lack related party information.

The fourth indicator — small amounts in, rapid consolidation out — is the canonical
pass-through / mule signature, and is the closest regulatory language to what a high
`flow_through_ratio` combined with a low `median_hours_to_forward` measures.
