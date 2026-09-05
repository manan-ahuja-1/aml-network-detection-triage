---
source_id: fincen-fin-2020-a003
title: "Money mules: unwitting, witting and complicit"
url: https://www.fincen.gov/system/files/advisory/2020-07-07/Advisory_%20Imposter_and_Money_Mule_COVID_19_508_FINAL.pdf
chunk_type: definition
typologies: [BIPARTITE, STACK, FAN-IN, FAN-OUT]
---

# Money mules: unwitting, witting and complicit

FinCEN Advisory FIN-2020-A003 (7 July 2020), citing the FBI's "Money Mule Awareness"
(July 2019), defines a money mule as "a person who transfers illegally acquired money on
behalf of or at the direction of another," and distinguishes three states of knowledge:

- **Unwitting / unknowing** — an individual "unaware that he or she is part of a larger
  criminal scheme," motivated by trust in an apparent romance, job offer or proposition.
- **Witting** — an individual who "chooses to ignore obvious red flags or acts willfully
  blind to his/her money movement activity," motivated by financial gain or an
  unwillingness to acknowledge the role.
- **Complicit** — an individual "aware of his/her role as a money mule and is complicit
  in the larger criminal scheme," motivated by financial gain or loyalty to a criminal
  group.

Why this distinction matters to triage rather than only to detection: all three produce
the same transaction topology. A pass-through account with high flow-through ratio and
short hold time looks identical whichever category the account holder falls into. The
distinction is drawn from context a monitoring model does not have — account opening
history, customer statements, recruitment patterns — which is precisely the judgement an
L1 analyst adds on top of a model score, and precisely the boundary an automated triage
layer must respect rather than assert past.

The advisory's standing caution applies:

> As no single financial red flag indicator is necessarily indicative of illicit or
> suspicious activity, financial institutions should consider additional contextual
> information and the surrounding facts and circumstances, such as a customer's
> historical financial activity, whether the transactions are in line with prevailing
> business practices, and whether the customer exhibits multiple indicators, before
> determining if a transaction is suspicious.
