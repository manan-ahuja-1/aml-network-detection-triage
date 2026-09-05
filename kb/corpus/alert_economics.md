---
source_id: bpi-getting-to-effectiveness-2018
title: "Alert economics: why precision at capacity is the constraint"
url: https://bpi.com/getting-to-effectiveness-report-on-u-s-financial-institution-resources-devoted-to-bsa-aml-sanctions-compliance/
chunk_type: context
typologies: []
---

# Alert economics

The "90-95% of AML alerts are false positives" figure is repeated constantly in vendor
marketing, almost always without attribution. The primary source most of those citations
trace back to is the Bank Policy Institute's 2018 study *Getting to Effectiveness*, a
survey of 19 US financial institutions holding roughly $50bn to over $500bn in assets.

**Figures reported by BPI, verbatim:**

- Participants employ "over 14,000 individuals, investing approximately $2.4 billion" and
  use "as many as over 20 different I.T. systems per institution" for BSA/AML compliance.
- In 2017 participants "reviewed approximately 16 million alerts, filed over 640,000
  suspicious activity reports (SAR) and more than 5.2 million currency transaction reports
  (CTR)".
- "a median of 4% of SARs and an average of 0.44% of CTRs warranted follow-up inquiries
  from law enforcement".
- "18% of their alerts related to structuring".
- Of ~2.36 million "high risk" customers, "a median of roughly 6% were subject to SAR
  filings while 0.3% of these customers were the subject of follow-up inquiries".

**Ratios derived from those figures (our arithmetic, not BPI's):**

- 640,000 SARs from 16,000,000 alerts is a **4% alert-to-SAR conversion rate** — i.e. 96%
  of alerts did not result in a SAR. This is the primary-source version of the
  "90-95% false positive" claim, and it is slightly worse than the number usually quoted.
- 4% of those SARs drawing law-enforcement follow-up puts roughly **0.16% of all alerts**
  on a path to law-enforcement interest.
- $2.4bn across 16 million alerts is about **$150 per alert** — but that $2.4bn covers the
  entire BSA/AML programme (staff, systems, KYC, CTR filing, model validation), not alert
  review alone. So $150 is an **upper bound** on the fully-loaded cost of one alert
  review, not a measured per-alert cost, and it should never be quoted as one.

**Why the cost of a MISS cannot be grounded the same way.** No published figure gives the
expected cost of one missed laundering transaction. Enforcement penalties are levied for
programme failures rather than per undetected transaction, and the counterfactual harm is
not observable. This is why the operating point in this project is reported as a
sensitivity strip across cost ratios rather than as a single tuned threshold: the
denominator can be sourced, the numerator cannot, and pretending otherwise would present
a judgement call as a measurement.

**The consequence for model design.** The binding constraint is not detection power but
**precision at review capacity**. A compliance team can work a fixed number of alerts per
day; a model that improves recall by flooding the queue makes the operation worse. That is
why this project reports precision@k at analyst-scale k, and why the account-level alert
queue — not the transaction-level score — is the unit the agent is evaluated on.
