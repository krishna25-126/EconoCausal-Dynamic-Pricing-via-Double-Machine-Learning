"""Uplift evaluation on a randomized hold-out (Qini curve and decile table)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OUTCOME, TREATMENT


def qini_curve(df: pd.DataFrame, score: np.ndarray, points: int = 50) -> dict:
    """Qini curve: cumulative incremental purchases when customers are treated
    in order of predicted effect. Needs a randomized hold-out.

    `treated` means received any discount. Uplift at fraction k is
        Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k)
    where Y_* is the number of purchases among treated/control customers in the
    top-k share, and N_* the group sizes.
    """
    d = pd.DataFrame({"score": score, "t": (df[TREATMENT] > 0).astype(int).to_numpy(),
                      "y": df[OUTCOME].to_numpy()}).sort_values("score", ascending=False).reset_index(drop=True)
    n = len(d)
    cum_t, cum_c = d["t"].cumsum().to_numpy(), (1 - d["t"]).cumsum().to_numpy()
    y_t = (d["y"] * d["t"]).cumsum().to_numpy()
    y_c = (d["y"] * (1 - d["t"])).cumsum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(cum_c > 0, y_t - y_c * cum_t / np.maximum(cum_c, 1), 0.0)
    q = np.nan_to_num(q)
    frac = np.arange(1, n + 1) / n
    random_line = frac * q[-1]
    idx = np.unique(np.linspace(0, n - 1, points).astype(int))
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    area = float(_trapz(q - random_line, frac))
    return {
        "fraction": [0.0, *frac[idx].tolist()],
        "model": [0.0, *q[idx].tolist()],
        "random": [0.0, *random_line[idx].tolist()],
        "qini_coefficient": area,
        "total_uplift": float(q[-1]),
    }


def decile_table(df: pd.DataFrame, score: np.ndarray, bins: int = 10) -> list[dict]:
    """Observed treated-minus-control purchase rate within score deciles."""
    d = df.assign(score=score, t=(df[TREATMENT] > 0).astype(int))
    d["bin"] = pd.qcut(d["score"].rank(method="first", ascending=False), bins, labels=False)
    rows = []
    for b, g in d.groupby("bin"):
        gt, gc = g[g.t == 1], g[g.t == 0]
        rows.append({
            "decile": int(b) + 1,
            "predicted_effect": float(g["score"].mean()),
            "observed_uplift": float(gt[OUTCOME].mean() - gc[OUTCOME].mean()) if len(gt) and len(gc) else 0.0,
            "n": int(len(g)),
        })
    return rows


def synthetic_validation(df: pd.DataFrame, tau_hat: np.ndarray) -> dict | None:
    """Compare estimates to the known ground truth (synthetic data only)."""
    if "true_tau" not in df.columns:
        return None
    true = df["true_tau"].to_numpy()
    return {
        "correlation": float(np.corrcoef(tau_hat, true)[0, 1]),
        "mean_estimated": float(tau_hat.mean()),
        "mean_true": float(true.mean()),
        "rmse": float(np.sqrt(np.mean((tau_hat - true) ** 2))),
    }