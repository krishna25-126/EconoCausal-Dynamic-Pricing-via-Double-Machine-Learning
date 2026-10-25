"""Compare the causal prescription against common non-causal strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MARGIN_COL
from .optimizer import option_table, allocate


def _spend_until_budget(order: np.ndarray, cost_per_user: np.ndarray, budget: float) -> np.ndarray:
    """Boolean mask: treat users in `order` until the expected spend hits the budget."""
    spent, picked = 0.0, np.zeros(len(order), dtype=bool)
    for i in order:
        if spent + cost_per_user[i] > budget:
            continue
        spent += cost_per_user[i]
        picked[i] = True
    return picked


def _summarize(name: str, picked_idx_level, gain, cost, prob, p0, truth=None) -> dict:
    n = len(p0)
    r = np.arange(n)
    est_gain = float(gain[r, picked_idx_level].sum())
    out = {
        "strategy": name,
        "customers_treated": int((picked_idx_level > 0).sum()),
        "expected_spend": float(cost[r, picked_idx_level].sum()),
        "expected_extra_purchases": float((prob[r, picked_idx_level] - p0).sum()),
        "expected_profit_gain": est_gain,
    }
    if truth is not None:                       # synthetic data only: score with the true effects
        tprob, tcost, tgain = truth
        out["true_profit_gain"] = float(tgain[r, picked_idx_level].sum())
        out["true_spend"] = float(tcost[r, picked_idx_level].sum())
    return out


def compare(audience: pd.DataFrame, tau: np.ndarray, p0: np.ndarray, budget: float,
            levels: list[int], blanket_level: int = 10, seed: int = 0) -> list[dict]:
    """Three strategies under the same budget:
      * Random blanket: a flat discount to random customers.
      * Predictive targeting: flat discount to customers least likely to buy
        (what a churn/purchase-propensity model would suggest).
      * Causal (EconoCausal): the optimizer's personalized discounts.
    """
    margin = audience[MARGIN_COL].to_numpy(float)
    prob, cost, gain = option_table(tau, p0, margin, levels)
    k = levels.index(blanket_level)
    n = len(audience)

    truth = None
    if {"true_tau", "true_p0"} <= set(audience.columns):
        truth = option_table(audience["true_tau"].to_numpy(), audience["true_p0"].to_numpy(), margin, levels)

    rng = np.random.default_rng(seed)
    blanket = _spend_until_budget(rng.permutation(n), cost[:, k], budget)
    predictive = _spend_until_budget(np.argsort(p0), cost[:, k], budget)       # lowest purchase prob first
    causal = allocate(audience, tau, p0, budget, levels)["discount"].map({d: i for i, d in enumerate(levels)}).to_numpy()

    return [
        _summarize("Random blanket", np.where(blanket, k, 0), gain, cost, prob, p0, truth),
        _summarize("Predictive targeting", np.where(predictive, k, 0), gain, cost, prob, p0, truth),
        _summarize("EconoCausal (causal)", causal, gain, cost, prob, p0, truth),
    ]