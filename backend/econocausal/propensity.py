"""Propensity score matching: balance observational data to mimic an A/B test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, OUTCOME, TREATMENT


def _smd(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Standardized mean difference per feature (|SMD| < 0.1 is well balanced)."""
    pooled = np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
    return ((a.mean() - b.mean()) / pooled).fillna(0.0)


def match(df: pd.DataFrame, caliper_sd: float = 0.2, seed: int = 0) -> dict:
    """1:1 nearest-neighbour matching (with caliper) on the propensity of
    receiving any discount.

    Returns balance tables before/after matching and the matched-pair estimate
    of the average effect of getting *any* discount (ATT).
    """
    data = df.copy()
    data["_treated"] = (data[TREATMENT] > 0).astype(int)
    X = StandardScaler().fit_transform(data[FEATURES])
    model = LogisticRegression(max_iter=1000).fit(X, data["_treated"])
    ps = np.clip(model.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    logit = np.log(ps / (1 - ps))
    data["_ps"] = ps

    treated = data[data["_treated"] == 1]
    control = data[data["_treated"] == 0]
    caliper = caliper_sd * logit.std()

    nn = NearestNeighbors(n_neighbors=1).fit(logit[control.index].reshape(-1, 1))
    dist, idx = nn.kneighbors(logit[treated.index].reshape(-1, 1))
    ok = dist.ravel() <= caliper
    matched_t = treated[ok]
    matched_c = control.iloc[idx.ravel()[ok]]

    before = _smd(treated[FEATURES], control[FEATURES])
    after = _smd(matched_t[FEATURES], matched_c[FEATURES])
    naive = treated[OUTCOME].mean() - control[OUTCOME].mean()
    att = matched_t[OUTCOME].mean() - matched_c[OUTCOME].mean()
    return {
        "n_treated": int(len(treated)),
        "n_matched_pairs": int(ok.sum()),
        "naive_difference": float(naive),
        "att_matched": float(att),
        "balance": [
            {"feature": f, "smd_before": float(before[f]), "smd_after": float(after[f])} for f in FEATURES
        ],
        "max_abs_smd_before": float(before.abs().max()),
        "max_abs_smd_after": float(after.abs().max()),
    }