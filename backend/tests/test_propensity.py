from econocausal import data, propensity


def test_matching_improves_balance():
    r = propensity.match(data.generate_observational(4000))
    assert r["max_abs_smd_after"] < 0.1 < r["max_abs_smd_before"]
    assert r["n_matched_pairs"] > 0.8 * r["n_treated"]