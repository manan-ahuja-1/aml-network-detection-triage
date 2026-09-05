---
source_id: ibm-amlsim-typologies
title: "The three stages: placement, layering, integration"
url: ""
chunk_type: definition
typologies: [STACK, CYCLE, SCATTER-GATHER, GATHER-SCATTER]
---

# The three stages of money laundering

The standard framing used across FATF material, supervisory guidance and industry
training. Written here as a summary rather than quoted, because it is a textbook
description rather than a specific regulatory text.

**Placement** — introducing criminal proceeds into the financial system. This is where
cash enters: deposits, purchases of monetary instruments, cash-intensive businesses. It is
the stage most exposed to reporting thresholds, which is why structuring clusters here.
In transaction data, placement shows up as cash and cheque activity rather than as
transfers between existing accounts.

**Layering** — moving the funds through a sequence of transactions to break the audit
trail between the money and its origin. Cross-border transfers, transfers between
institutions, conversion between currencies and instruments, chains of intermediary
accounts. This is the stage a transaction graph is best placed to detect, because layering
is defined by *shape*: money passes through accounts rather than resting in them.

**Integration** — returning the laundered funds to the criminal in an apparently
legitimate form: asset purchases, business revenue, loan repayments, investment returns.

Why this matters for reading the model's features: `is_same_bank == 0` and
`is_cross_currency == 1` are layering indicators, not placement ones; cash and cheque
payment formats sit at placement; and a high `flow_through_ratio` with a short
`median_hours_to_forward` is the numeric signature of an account being used *for* layering
rather than as a destination.
