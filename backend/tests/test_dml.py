import numpy as np
import pytest
from econocausal import data
from econocausal.dml import CausalEngine


@pytest.fixture(scope="module")
def fitted():
    df = data.generate_observational(5000)
    return df, CausalEngine(n_trees=100).fit(df).fit_baseline(df)


def test_dml_recovers_heterogeneity(fitted):
    df, e = fitted
    assert np.corrcoef(e.tau(df), df["true_tau"])[0, 1] > 0.6


def test_dml_beats_naive(fitted):
    """The naive regression is biased by confounding; DML is much closer to the truth."""
    df, e = fitted
    truth = df["true_tau"].mean()
    assert abs(e.average_effect(df) - truth) < abs(e.naive_effect_ - truth)


def test_ite_scales_with_discount(fitted):
    df, e = fitted
    X = df.head(50)
    assert np.allclose(e.ite(X, 0, 20), 2 * e.ite(X, 0, 10))


def test_baseline_probability_valid(fitted):
    df, e = fitted
    p0 = e.p0(df)
    assert p0.min() >= 0.01 and p0.max() <= 0.99
    assert np.corrcoef(p0, df["true_p0"])[0, 1] > 0.8