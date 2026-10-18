"""Synthetic retail data with a known ground-truth causal effect.

Why synthetic? Real data never tells us the true effect of a discount on a
specific customer, so we cannot check whether a causal model is right. Here we
generate data where the true individual effect (`true_tau`, change in purchase
probability per dollar of discount) is known, and where past discounts were
handed out in a *confounded* way (mostly to low-engagement customers), exactly
the situation that fools naive predictive models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DISCOUNT_LEVELS, FEATURES, ID_COL, MARGIN_COL, OUTCOME, RANDOM_STATE, TREATMENT


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _features(n: int, rng: np.random.Generator, shift: float = 0.0) -> pd.DataFrame:
    """Draw customer features. `shift` > 0 simulates behaviour drift."""
    engagement = np.clip(rng.beta(2, 2, n) - 0.12 * shift, 0.0, 1.0)
    income_k = np.clip(rng.lognormal(np.log(55), 0.45, n) * (1 + 0.25 * shift), 15, 250)
    return pd.DataFrame(
        {
            "age": np.clip(rng.normal(40, 12, n), 18, 75).round(0),
            "income_k": income_k.round(1),
            "tenure_months": np.clip(rng.exponential(24, n), 1, 120).round(0),
            "past_purchases": rng.poisson(3 + 2 * shift, n),
            "engagement": engagement.round(3),
            "cart_value": np.clip(rng.lognormal(np.log(110), 0.35, n), 25, 400).round(2),
        }
    )


def _truth(X: pd.DataFrame):
    """Ground-truth baseline purchase probability and per-dollar effect."""
    eng = X["engagement"].to_numpy()
    inc_z = (np.log(X["income_k"].to_numpy()) - np.log(55)) / 0.45
    past_z = (X["past_purchases"].to_numpy() - 3) / 1.8
    ten_z = (X["tenure_months"].to_numpy() - 24) / 24

    p0 = _sigmoid(-0.9 + 3.2 * (eng - 0.5) + 0.35 * past_z + 0.15 * inc_z)

    price_sens = _sigmoid(-1.3 * inc_z + 3.0 * (0.5 - eng))          # who reacts to price
    sleeping_dog = _sigmoid(10 * (eng - 0.8)) * _sigmoid(2 * ten_z)  # discount annoys loyal fans
    tau = 0.020 * price_sens - 0.008 * sleeping_dog                  # prob. change per $
    return p0, tau


def generate_observational(n: int = 6000, seed: int = RANDOM_STATE, shift: float = 0.0) -> pd.DataFrame:
    """Historical campaign data where discounts were targeted (confounded)."""
    rng = np.random.default_rng(seed)
    X = _features(n, rng, shift)
    p0, tau = _truth(X)

    # Marketing historically discounted people who looked likely to churn:
    # low engagement and few past purchases.
    eng = X["engagement"].to_numpy()
    past_z = (X["past_purchases"].to_numpy() - 3) / 1.8
    p_any = _sigmoid(0.2 - 3.5 * (eng - 0.5) - 0.4 * past_z)
    got_discount = rng.random(n) < p_any
    level_probs = np.column_stack([0.40 - 0.3 * eng, 0.3 * np.ones(n), 0.2 + 0.1 * eng, 0.10 + 0.2 * eng])
    level_probs = level_probs / level_probs.sum(axis=1, keepdims=True)
    cum = level_probs.cumsum(axis=1)
    draw = rng.random(n)[:, None]
    amounts = np.array(DISCOUNT_LEVELS[1:])[(draw > cum).sum(axis=1).clip(0, 3)]
    discount = np.where(got_discount, amounts, 0)

    p = np.clip(p0 + tau * discount, 0.01, 0.99)
    purchased = (rng.random(n) < p).astype(int)

    df = X.copy()
    df.insert(0, ID_COL, np.arange(1, n + 1))
    df[TREATMENT] = discount
    df[OUTCOME] = purchased
    df[MARGIN_COL] = (0.35 * df["cart_value"]).round(2)
    df["true_p0"] = p0
    df["true_tau"] = tau
    return df


def generate_rct(n: int = 4000, seed: int = RANDOM_STATE + 1, shift: float = 0.0) -> pd.DataFrame:
    """A randomized experiment: discounts assigned independently of features.

    Used only to *evaluate* uplift ranking honestly (Qini curve), the way you
    would use a small real A/B test as a hold-out.
    """
    rng = np.random.default_rng(seed)
    X = _features(n, rng, shift)
    p0, tau = _truth(X)
    discount = rng.choice(DISCOUNT_LEVELS, size=n)
    p = np.clip(p0 + tau * discount, 0.01, 0.99)
    df = X.copy()
    df.insert(0, ID_COL, np.arange(100001, 100001 + n))
    df[TREATMENT] = discount
    df[OUTCOME] = (rng.random(n) < p).astype(int)
    df[MARGIN_COL] = (0.35 * df["cart_value"]).round(2)
    df["true_p0"] = p0
    df["true_tau"] = tau
    return df


def campaign_audience(n: int = 2000, seed: int = RANDOM_STATE + 2, shift: float = 0.0) -> pd.DataFrame:
    """Customers the marketing director wants to target in the next campaign."""
    df = generate_rct(n, seed, shift).drop(columns=[TREATMENT, OUTCOME])
    df[ID_COL] = np.arange(200001, 200001 + n)
    return df


def required_columns() -> list[str]:
    return [ID_COL, *FEATURES, TREATMENT, OUTCOME, MARGIN_COL]