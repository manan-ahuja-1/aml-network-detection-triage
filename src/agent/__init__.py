"""The L1 triage agent.

The engine (arm C) is the transaction-monitoring layer: it produces a ranked alert
queue. This package is the layer immediately after it in the real workflow —

    transaction monitoring -> alert -> L1 triage -> L2 investigation -> SAR decision

— and nothing here makes a filing decision. L1 triage reads the alert, gathers the
evidence, decides whether it is worth an investigator's time, and writes the case note
that an investigator would read first.
"""
