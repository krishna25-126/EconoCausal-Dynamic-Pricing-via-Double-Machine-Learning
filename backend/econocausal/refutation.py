"""Causal audit: DoWhy refutation tests.

A refutation test tries to *break* the causal estimate. If the estimate
survives, we have more reason to trust that it reflects a real causal effect
rather than hidden confounding or noise.
"""
from __future__ import annotations

import pandas as pd

from . import dag
from .config import FEATURES, OUTCOME, TREATMENT

METHOD = "backdoor.linear_regression"


def _p_value(res) -> float | None:
    try:
        rr = res.refutation_result
        if isinstance(rr, dict) and "p_value" in rr:
            return float(rr["p_value"])
    except Exception:
        pass
    return None


def run_audit(df: pd.DataFrame, simulations: int = 20, seed: int = 0) -> dict:
    """Estimate the average effect, then try to refute it three ways."""
    cols = [TREATMENT, OUTCOME, *FEATURES]
    data = df[cols].reset_index(drop=True)
    model, estimand = dag.identify(data)
    est = model.estimate_effect(estimand, method_name=METHOD)
    base = float(est.value)

    tests = []

    placebo = model.refute_estimate(estimand, est, method_name="placebo_treatment_refuter",
                                    placebo_type="permute", num_simulations=simulations, random_seed=seed)
    new = float(placebo.new_effect)
    tests.append({
        "test": "Placebo treatment",
        "what_it_checks": "Replace the real discount with random noise. The effect should vanish.",
        "original_effect": base, "new_effect": new, "p_value": _p_value(placebo),
        "passed": abs(new) < 0.25 * abs(base),
    })

    rcc = model.refute_estimate(estimand, est, method_name="random_common_cause",
                                num_simulations=simulations, random_seed=seed)
    new = float(rcc.new_effect)
    tests.append({
        "test": "Random common cause",
        "what_it_checks": "Add an irrelevant random confounder. The effect should not change.",
        "original_effect": base, "new_effect": new, "p_value": _p_value(rcc),
        "passed": abs(new - base) < 0.10 * abs(base),
    })

    subset = model.refute_estimate(estimand, est, method_name="data_subset_refuter",
                                   subset_fraction=0.8, num_simulations=simulations, random_seed=seed)
    new = float(subset.new_effect)
    tests.append({
        "test": "Data subset",
        "what_it_checks": "Re-estimate on random 80% subsets. The effect should stay stable.",
        "original_effect": base, "new_effect": new, "p_value": _p_value(subset),
        "passed": abs(new - base) < 0.15 * abs(base),
    })

    return {
        "estimand": "Average effect of $1 discount on purchase probability (backdoor adjustment)",
        "estimated_effect_per_dollar": base,
        "tests": tests,
        "all_passed": all(t["passed"] for t in tests),
    }