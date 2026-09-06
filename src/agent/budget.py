"""A hard ceiling on API spend, enforced rather than remembered.

WHY THIS EXISTS
---------------
Two things happened on Day 6 that this module prevents.

A 200-alert run died at alert 72 because the account ran out of credit mid-flight. It
had already billed for 72 completions, and an earlier attempt had billed 51 more that
were then discarded when one bad alert killed the batch. Nothing checked, before
spending anything, whether the run could afford to finish.

And every cost figure reported that day was ~33% too high, because two hardcoded price
constants held Sonnet 4.5's rates while the calls went to Sonnet 5. A budget tracked by
arithmetic in someone's head, against prices that are quietly wrong, is not a budget.

So: every billed call lands in a ledger on disk, and a batch prices itself before it
starts. Going over the ceiling requires typing the amount.

WHAT COUNTS
-----------
Billed calls only. A cache hit costs nothing and must not appear in the ledger, or the
recorded total drifts upward every time a run is repeated and the guard starts refusing
work that would in fact be free.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


class BudgetExceeded(RuntimeError):
    """Raised before any spending happens, never partway through a run."""


def load_ledger() -> list[dict]:
    if not config.SPEND_LEDGER.exists():
        return []
    return json.loads(config.SPEND_LEDGER.read_text())


def total_spent() -> float:
    return round(sum(e["cost_usd"] for e in load_ledger()), 4)


def remaining() -> float:
    return round(config.BUDGET_CEILING_USD - total_spent(), 4)


def record(script: str, model: str, calls: int, input_tokens: int,
           output_tokens: int, cost_usd: float, note: str = "") -> dict:
    """Append one batch's billed usage. Called after the work, with real numbers."""
    if calls == 0 or cost_usd <= 0:
        return {"skipped": "nothing billed"}

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "model": model,
        "billed_calls": int(calls),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost_usd": round(float(cost_usd), 6),
        "note": note,
    }
    ledger = load_ledger()
    ledger.append(entry)
    config.SPEND_LEDGER.write_text(json.dumps(ledger, indent=2))
    return entry


def estimate(per_call_usd: float, n_calls: int, headroom: float = 1.25) -> float:
    """Projected cost of a batch, deliberately pessimistic.

    The 25% headroom is not padding for its own sake: per-call cost varies by a factor
    of three across this alert set, because evidence volume does, and the sample used to
    price a run is usually the first case rather than a representative one.
    """
    return round(per_call_usd * n_calls * headroom, 4)


def preflight(script: str, per_call_usd: float, n_calls: int,
              confirm_spend: float | None = None) -> dict:
    """Decide whether a batch may start. Raises BudgetExceeded if it may not.

    `confirm_spend` is the deliberate override: the caller must name the amount they are
    willing to spend, so exceeding the ceiling is always a typed act rather than a
    default that quietly gives way.
    """
    projected = estimate(per_call_usd, n_calls)
    spent, left = total_spent(), remaining()

    report = {
        "script": script,
        "n_calls": n_calls,
        "per_call_usd": round(per_call_usd, 5),
        "projected_usd": projected,
        "already_spent_usd": spent,
        "ceiling_usd": config.BUDGET_CEILING_USD,
        "remaining_usd": left,
        "approved": True,
    }

    if projected <= left:
        return report

    if confirm_spend is not None and confirm_spend >= projected:
        report["approved"] = True
        report["override"] = f"--confirm-spend {confirm_spend}"
        return report

    report["approved"] = False
    raise BudgetExceeded(
        f"\n  BUDGET GUARD — refusing to start {script}\n"
        f"    {n_calls} calls at ~${per_call_usd:.5f} each\n"
        f"    projected      ${projected:.2f}  (includes 25% headroom)\n"
        f"    already spent  ${spent:.2f}\n"
        f"    ceiling        ${config.BUDGET_CEILING_USD:.2f}\n"
        f"    remaining      ${left:.2f}\n"
        f"  Short by ${projected - left:.2f}. Either reduce the batch "
        f"(--limit), or pass --confirm-spend {projected:.2f} to authorise it."
    )


def status_line() -> str:
    return (f"budget: ${total_spent():.2f} spent of ${config.BUDGET_CEILING_USD:.2f} "
            f"ceiling  (${remaining():.2f} left)")


if __name__ == "__main__":
    ledger = load_ledger()
    print(status_line())
    if not ledger:
        print("  ledger empty — no billed calls recorded yet")
    for e in ledger:
        print(f"  {e['timestamp']}  {e['script']:<22} {e['model']:<28} "
              f"{e['billed_calls']:>4} calls  ${e['cost_usd']:.4f}  {e.get('note','')}")
