"""Causal graph (DAG) for the retail discount problem, built with DoWhy."""
from __future__ import annotations

import networkx as nx
import pandas as pd
from dowhy import CausalModel

from .config import FEATURES, OUTCOME, TREATMENT


def build_graph() -> nx.DiGraph:
    """Every feature is a confounder: it affects both the discount decision
    and the purchase outcome. The discount also affects the outcome."""
    g = nx.DiGraph()
    for f in FEATURES:
        g.add_edge(f, TREATMENT)
        g.add_edge(f, OUTCOME)
    g.add_edge(TREATMENT, OUTCOME)
    return g


def graph_json() -> dict:
    """Nodes and edges for display in the dashboard."""
    g = build_graph()
    roles = {TREATMENT: "treatment", OUTCOME: "outcome"}
    return {
        "nodes": [{"id": n, "role": roles.get(n, "confounder")} for n in g.nodes],
        "edges": [{"source": u, "target": v} for u, v in g.edges],
    }


def build_model(df: pd.DataFrame) -> CausalModel:
    gml = "\n".join(nx.generate_gml(build_graph()))
    return CausalModel(data=df, treatment=TREATMENT, outcome=OUTCOME, graph=gml)


def identify(df: pd.DataFrame):
    """Return (model, identified estimand). Backdoor adjustment on all features."""
    model = build_model(df)
    estimand = model.identify_effect(proceed_when_unidentifiable=False)
    return model, estimand