"""Plain-language summaries for the marketing director."""
from __future__ import annotations


def _usd(x: float) -> str:
    return f"${abs(x):,.0f}"


def allocation_summary(strategies: list[dict], budget: float, segments: dict, use_truth: bool = False) -> str:
    key = "true_profit_gain" if use_truth and "true_profit_gain" in strategies[0] else "expected_profit_gain"
    by = {s["strategy"]: s for s in strategies}
    causal, blanket = by["EconoCausal (causal)"], by["Random blanket"]
    pred = by["Predictive targeting"]
    saved = causal[key] - blanket[key]
    lines = [
        f"This strategy saves {_usd(saved)} compared to blanket targeting"
        if saved >= 0 else f"This strategy is {_usd(saved)} worse than blanket targeting",
        f"and beats predictive targeting by {_usd(causal[key] - pred[key])}." if causal[key] >= pred[key]
        else f"but trails predictive targeting by {_usd(pred[key] - causal[key])}.",
    ]
    spend, treated = causal["expected_spend"], causal["customers_treated"]
    lines.append(
        f"It discounts {treated:,} customers and spends {_usd(spend)} of your {_usd(budget)} budget"
        + (" (extra discounts would lose money, so it stops there)." if spend < 0.97 * budget else ".")
    )
    pers = segments.get("Persuadable", 0)
    dogs = segments.get("Sleeping dog", 0)
    pers_txt = f"{pers:,} customer{'s are Persuadables who buy' if pers != 1 else ' is a Persuadable who buys'} mainly because of a discount"
    dogs_txt = (f"; {dogs:,} {'is a Sleeping dog' if dogs == 1 else 'are Sleeping dogs'} that a discount could push away."
                if dogs else ".")
    lines.append(pers_txt + dogs_txt)
    return " ".join(lines)