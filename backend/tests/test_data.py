import numpy as np
from econocausal import data
from econocausal.config import FEATURES


def test_shapes_and_columns():
    df = data.generate_observational(1000)
    assert len(df) == 1000
    assert set(data.required_columns()) <= set(df.columns)
    assert df["purchased"].isin([0, 1]).all()


def test_discounts_are_confounded():
    """Past discounts went to low-engagement customers (the trap for naive ML)."""
    df = data.generate_observational(5000)
    treated = df["discount"] > 0
    assert df.loc[treated, "engagement"].mean() < df.loc[~treated, "engagement"].mean() - 0.1


def test_rct_is_randomized():
    df = data.generate_rct(5000)
    treated = df["discount"] > 0
    assert abs(df.loc[treated, "engagement"].mean() - df.loc[~treated, "engagement"].mean()) < 0.03


def test_effect_is_heterogeneous():
    df = data.generate_observational(3000)
    assert df["true_tau"].min() < 0 < df["true_tau"].max()
    assert not df[FEATURES].isna().any().any()