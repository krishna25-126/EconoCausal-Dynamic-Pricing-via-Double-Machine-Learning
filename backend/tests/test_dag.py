from econocausal import dag, data
from econocausal.config import FEATURES


def test_graph_structure():
    g = dag.build_graph()
    assert g.has_edge("discount", "purchased")
    assert all(g.has_edge(f, "discount") and g.has_edge(f, "purchased") for f in FEATURES)


def test_backdoor_identified():
    _, est = dag.identify(data.generate_observational(500))
    assert "backdoor" in est.estimands
    assert set(est.get_backdoor_variables()) == set(FEATURES)