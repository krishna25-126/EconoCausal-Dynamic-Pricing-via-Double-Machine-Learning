from econocausal import data, refutation


def test_audit_passes_on_valid_data():
    a = refutation.run_audit(data.generate_observational(4000), simulations=10)
    assert a["estimated_effect_per_dollar"] > 0
    assert len(a["tests"]) == 3
    assert a["all_passed"], a["tests"]
