---
source_id: ibm-amlsim-typologies
title: "Typology: RANDOM"
chunk_type: typology
typologies: [RANDOM]
---

# RANDOM

**Shape.** Laundering transactions injected without an imposed geometric structure — no
fan, no chain, no cycle.

**Real-world correspondence.** Laundering that does not present a recognisable topology:
opportunistic movement, or activity whose structure lies outside the observation window.
It is the honest residual category.

**Why it matters for evaluation.** RANDOM is the control within the ground truth. A model
that detects fan-in and fan-out well but RANDOM poorly is detecting *shape*, which is the
project's thesis and a legitimate finding. A model that detects RANDOM just as well is
probably keying on something else entirely — payment format, amount, or a simulator
artifact — and the graph story would be overstated.

**Model features that fire.** By construction, no topology feature should be particularly
predictive here. Detection, if it happens, comes from transaction-level and account-level
attributes: `payment_format`, amount features, and behavioural aggregates.

**Note on coverage.** Separately from the RANDOM label, only 2,554 of the 4,522 laundering
transactions surviving truncation belong to *any* named pattern. The remaining 1,968 are
laundering the generator did not group into a ring at all. Pattern-level recall is
reported over the labelled subset, and that limitation is stated wherever the number
appears.
