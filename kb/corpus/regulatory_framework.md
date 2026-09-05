---
source_id: ecfr-1020-320
title: "Regulatory framework: BSA, CTR, SAR, FATF, SR 11-7"
url: https://www.ecfr.gov/current/title-31/subtitle-B/chapter-X/part-1020/subpart-C/section-1020.320
chunk_type: regulatory
typologies: []
---

# Regulatory framework

**Bank Secrecy Act (BSA), 1970.** The statute requiring US financial institutions to keep
records and file reports that have, in the Act's own phrase, "a high degree of usefulness"
to law enforcement or national security officials.

**Currency Transaction Report (CTR).** Filed for currency transactions above **$10,000**.
Deliberately conducting transactions to keep them below this threshold is **structuring**,
an offence in itself under 31 USC 5324 — the transactions need not exceed $10,000 at any
one bank on any single day for structuring to have occurred (FFIEC Appendix G).

**Suspicious Activity Report (SAR), FinCEN Form 111.** Filed when an institution knows,
suspects, or has reason to suspect that a transaction involves funds from illegal
activity, is designed to evade BSA regulations, or lacks a business or apparent lawful
purpose. Per **31 CFR § 1020.320**:

> A bank is required to file a SAR no later than **30 calendar days** after the date of
> initial detection by the bank of facts that may constitute a basis for filing a SAR. If
> no suspect was identified on the date of the detection of the incident requiring the
> filing, a bank may delay filing a SAR for an additional 30 calendar days to identify a
> suspect

— in no case more than **60 calendar days** after initial detection. Continuing activity
is reported at least every 90 days.

**FATF.** The Financial Action Task Force, an intergovernmental body that sets the
international AML/CFT standard through its 40 Recommendations, and publishes typology
reports and red-flag indicator sets that national supervisors and institutions draw on.

**SR 11-7 (Supervisory Guidance on Model Risk Management).** Issued by the Federal Reserve
and OCC in 2011. It is the reason model explainability in AML is a compliance requirement
rather than a convenience: a model used to make regulatory decisions must be documented,
validated, and its outputs explicable to an examiner. An alert an institution cannot
justify is a model risk finding, independent of whether the alert was correct.

**The operational workflow this vocabulary describes**, and where an automated system sits
in it:

    transaction monitoring -> alert -> L1 triage -> L2 investigation -> SAR filing decision

A scoring model is the *transaction monitoring* layer. An automated triage agent is *L1*.
Neither makes the filing decision.
