"""Data drift detector: warns when customer behaviour has changed since training.

If the customers we score today no longer look like the customers the causal
model was trained on, its effect estimates may be unreliable and it should be
retrained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import FEATURES


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate, >0.25 major shift."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


class DriftDetector:
    def __init__(self, p_threshold: float = 0.01, psi_threshold: float = 0.10):
        self.p_threshold, self.psi_threshold = p_threshold, psi_threshold
        self.reference: pd.DataFrame | None = None

    def fit(self, reference: pd.DataFrame) -> "DriftDetector":
        self.reference = reference[FEATURES].copy()
        return self

    def check(self, current: pd.DataFrame) -> dict:
        if self.reference is None:
            raise RuntimeError("DriftDetector must be fit on reference data first")
        rows = []
        for f in FEATURES:
            ref, cur = self.reference[f].to_numpy(float), current[f].to_numpy(float)
            ks = ks_2samp(ref, cur)
            p = psi(ref, cur)
            rows.append({
                "feature": f, "ks_statistic": float(ks.statistic), "p_value": float(ks.pvalue), "psi": p,
                "reference_mean": float(ref.mean()), "current_mean": float(cur.mean()),
                "drifted": bool(ks.pvalue < self.p_threshold and p >= self.psi_threshold),
            })
        drifted = [r["feature"] for r in rows if r["drifted"]]
        return {
            "status": "drift_detected" if drifted else "stable",
            "drifted_features": drifted,
            "features": rows,
            "message": (f"Customer behaviour has changed in: {', '.join(drifted)}. Retrain the causal model "
                        "before trusting new discount prescriptions.") if drifted
                       else "No significant drift detected. The model's training data still matches current customers.",
        }