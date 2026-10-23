import numpy as np
from econocausal import data, evaluation
from econocausal.dml import CausalEngine


def test_qini_beats_random():
    df, rct = data.generate_observational(5000), data.generate_rct(4000)
    e = CausalEngine(n_trees=100).fit(df)
    good = evaluation.qini_curve(rct, e.tau(rct))["qini_coefficient"]
    noise = evaluation.qini_curve(rct, np.random.default_rng(0).random(len(rct)))["qini_coefficient"]
    assert good > noise + 10


def test_oracle_ranking_is_best():
    rct = data.generate_rct(4000)
    oracle = evaluation.qini_curve(rct, rct["true_tau"].to_numpy())["qini_coefficient"]
    anti = evaluation.qini_curve(rct, -rct["true_tau"].to_numpy())["qini_coefficient"]
    assert oracle > 0 > anti