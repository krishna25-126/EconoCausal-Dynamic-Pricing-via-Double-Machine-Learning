"""Prescriptive optimization: who gets which discount, given a fixed budget.

For each customer i and discount level d we compute
    p_i(d)    = clip(p0_i + tau_i * d, 0, 1)          probability of buying
    cost_i(d) = p_i(d) * d                              expected discount spend
    gain_i(d) = p_i(d) * (margin_i - d) - p0_i * margin_i   expected extra profit
and choose exactly one level per customer to maximize total gain subject to
total expected spend <= budget (a multiple-choice knapsack, solved exactly as a
mixed-integer program with SciPy's HiGHS solver).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from .config import DISCOUNT_LEVELS, ID_COL, MARGIN_COL


def option_table(tau: np.ndarray, p0: np.ndarray, margin: np.ndarray, levels=DISCOUNT_LEVELS):
    """Return (prob, cost, gain) matrices of shape (n_users, n_levels)."""
    d = np.asarray(levels, float)[None, :]
    prob = np.clip(p0[:, None] + tau[:, None] * d, 0.0, 1.0)
    cost = prob * d
    gain = prob * (margin[:, None] - d) - (p0 * margin)[:, None]
    return prob, cost, gain


def optimize(tau, p0, margin, budget: float, levels=DISCOUNT_LEVELS, time_limit: float = 30.0) -> np.ndarray:
    """Return the chosen level index for each user."""
    n, k = len(tau), len(levels)
    prob, cost, gain = option_table(np.asarray(tau), np.asarray(p0), np.asarray(margin), levels)
    c = -gain.ravel()                                    # milp minimizes

    # exactly one level per user
    rows = np.repeat(np.arange(n), k)
    one_per_user = csr_matrix((np.ones(n * k), (rows, np.arange(n * k))), shape=(n, n * k))
    budget_row = csr_matrix(cost.ravel()[None, :])
    cons = [LinearConstraint(one_per_user, 1, 1), LinearConstraint(budget_row, -np.inf, budget)]

    res = milp(c, constraints=cons, integrality=np.ones(n * k), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-4})
    if res.x is None:                                    # fall back to "no discounts"
        return np.zeros(n, dtype=int)
    return res.x.reshape(n, k).argmax(axis=1)


def segment(tau: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Classic uplift segments from the estimated effect of a $10 discount."""
    lift10 = tau * 10
    return np.select(
        [lift10 >= 0.05, lift10 <= -0.01, p0 >= 0.5],
        ["Persuadable", "Sleeping dog", "Sure thing"],
        default="Lost cause",
    )


def allocate(audience: pd.DataFrame, tau: np.ndarray, p0: np.ndarray, budget: float,
             levels=DISCOUNT_LEVELS) -> pd.DataFrame:
    """Prescription table: one row per customer with the optimal discount."""
    margin = audience[MARGIN_COL].to_numpy(float)
    prob, cost, gain = option_table(tau, p0, margin, levels)
    pick = optimize(tau, p0, margin, budget, levels)
    r = np.arange(len(audience))
    out = pd.DataFrame({
        ID_COL: audience[ID_COL].to_numpy(),
        "segment": segment(tau, p0),
        "baseline_prob": p0,
        "lift_if_10": tau * 10,
        "lift_if_20": tau * 20,
        "discount": np.asarray(levels)[pick],
        "prob_with_discount": prob[r, pick],
        "expected_lift": prob[r, pick] - p0,
        "expected_cost": cost[r, pick],
        "expected_profit_gain": gain[r, pick],
    })
    return out