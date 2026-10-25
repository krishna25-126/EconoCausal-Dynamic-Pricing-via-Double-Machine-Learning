import numpy as np
from econocausal import data, optimizer, strategies
from econocausal.config import DISCOUNT_LEVELS


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    tau = rng.uniform(-0.005, 0.02, n)
    p0 = rng.uniform(0.1, 0.7, n)
    margin = rng.uniform(25, 50, n)
    return tau, p0, margin


def test_budget_is_respected():
    tau, p0, margin = _toy()
    for budget in (100, 500, 2000):
        pick = optimizer.optimize(tau, p0, margin, budget)
        _, cost, _ = optimizer.option_table(tau, p0, margin)
        assert cost[np.arange(len(tau)), pick].sum() <= budget + 1e-6


def test_more_budget_never_hurts():
    tau, p0, margin = _toy()
    gains = []
    for budget in (100, 500, 5000):
        pick = optimizer.optimize(tau, p0, margin, budget)
        _, _, gain = optimizer.option_table(tau, p0, margin)
        gains.append(gain[np.arange(len(tau)), pick].sum())
    assert gains[0] <= gains[1] + 1e-6 <= gains[2] + 2e-6


def test_no_discounts_for_negative_effect():
    tau, p0, margin = np.array([-0.01] * 5), np.full(5, 0.4), np.full(5, 40.0)
    assert (optimizer.optimize(tau, p0, margin, 1000) == 0).all()


def test_causal_beats_blanket_on_truth():
    aud = data.campaign_audience(800)
    res = strategies.compare(aud, aud["true_tau"].to_numpy(), aud["true_p0"].to_numpy(), 2000, DISCOUNT_LEVELS)
    by = {r["strategy"]: r for r in res}
    assert by["EconoCausal (causal)"]["true_profit_gain"] > by["Random blanket"]["true_profit_gain"]
    assert by["EconoCausal (causal)"]["true_profit_gain"] > by["Predictive targeting"]["true_profit_gain"]