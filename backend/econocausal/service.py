"""Application state and orchestration shared by the API and the scripts."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as datagen
from . import evaluation, optimizer, propensity, refutation, strategies, summary
from .config import DEFAULT_BUDGET, DISCOUNT_LEVELS, FEATURES, ID_COL, MARGIN_COL, OUTCOME, TREATMENT
from .dml import CausalEngine
from .drift import DriftDetector


@dataclass
class Session:
    history: pd.DataFrame | None = None
    holdout: pd.DataFrame | None = None
    audience: pd.DataFrame | None = None
    engine: CausalEngine | None = None
    drift: DriftDetector | None = None
    train_report: dict = field(default_factory=dict)
    audit_report: dict | None = None
    last_allocation: pd.DataFrame | None = None
    synthetic: bool = False


def validate_history(df: pd.DataFrame) -> None:
    missing = [c for c in [ID_COL, *FEATURES, TREATMENT, OUTCOME, MARGIN_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if df[[*FEATURES, TREATMENT, OUTCOME, MARGIN_COL]].isna().any().any():
        raise ValueError("Data contains missing values; please clean them first")
    if len(df) < 500:
        raise ValueError("Need at least 500 historical rows to estimate causal effects")
    if not df[OUTCOME].isin([0, 1]).all():
        raise ValueError(f"'{OUTCOME}' must be 0 or 1")
    if (df[TREATMENT] <= 0).all() or (df[TREATMENT] > 0).all():
        raise ValueError("Need both discounted and non-discounted customers")


def read_csv(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))


def load_demo(s: Session, n: int = 6000) -> dict:
    s.history = datagen.generate_observational(n)
    s.holdout = datagen.generate_rct(max(2000, n // 2))
    s.audience = datagen.campaign_audience(2000)
    s.synthetic = True
    s.engine = s.drift = s.audit_report = s.last_allocation = None
    return describe_history(s)


def load_history(s: Session, df: pd.DataFrame) -> dict:
    validate_history(df)
    s.history, s.synthetic = df.reset_index(drop=True), "true_tau" in df.columns
    s.holdout = None
    s.audience = df.drop(columns=[TREATMENT, OUTCOME]).sample(min(2000, len(df)), random_state=0)
    s.engine = s.drift = s.audit_report = s.last_allocation = None
    return describe_history(s)


def describe_history(s: Session) -> dict:
    h = s.history
    treated = h[TREATMENT] > 0
    return {
        "rows": int(len(h)), "synthetic": s.synthetic,
        "share_discounted": float(treated.mean()),
        "purchase_rate_discounted": float(h.loc[treated, OUTCOME].mean()),
        "purchase_rate_not_discounted": float(h.loc[~treated, OUTCOME].mean()),
        "columns": list(h.columns),
        "preview": h.head(8).round(3).to_dict("records"),
    }


def train(s: Session, n_trees: int = 300) -> dict:
    if s.history is None:
        raise ValueError("Load data first")
    h = s.history
    s.engine = CausalEngine(n_trees=n_trees).fit(h).fit_baseline(h)
    s.drift = DriftDetector().fit(h)
    tau_h = s.engine.tau(h)
    report = {
        "rows": int(len(h)),
        "average_effect_per_dollar": float(tau_h.mean()),
        "naive_effect_per_dollar": float(s.engine.naive_effect_),
        "ite_distribution": np.histogram(tau_h * 10, bins=30)[0].tolist(),
        "ite_bins": np.histogram(tau_h * 10, bins=30)[1].tolist(),
        "matching": propensity.match(h),
        "validation": evaluation.synthetic_validation(h, tau_h),
    }
    if s.holdout is not None:
        tau_ho = s.engine.tau(s.holdout)
        report["qini"] = evaluation.qini_curve(s.holdout, tau_ho)
        report["deciles"] = evaluation.decile_table(s.holdout, tau_ho)
    s.train_report = report
    s.audit_report = None
    return report


def audit(s: Session) -> dict:
    if s.history is None:
        raise ValueError("Load data first")
    if s.audit_report is None:
        s.audit_report = refutation.run_audit(s.history)
    return s.audit_report


def allocate(s: Session, budget: float = DEFAULT_BUDGET) -> dict:
    if s.engine is None:
        raise ValueError("Train the model first")
    aud = s.audience
    tau, p0 = s.engine.tau(aud), s.engine.p0(aud)
    table = optimizer.allocate(aud, tau, p0, budget, DISCOUNT_LEVELS)
    s.last_allocation = table
    comp = strategies.compare(aud, tau, p0, budget, DISCOUNT_LEVELS)
    segments = table["segment"].value_counts().to_dict()
    by_level = table.groupby("discount").size().reindex(DISCOUNT_LEVELS, fill_value=0)
    return {
        "budget": budget,
        "expected_spend": float(table["expected_cost"].sum()),
        "customers": int(len(table)),
        "customers_discounted": int((table["discount"] > 0).sum()),
        "segments": segments,
        "discount_counts": {str(k): int(v) for k, v in by_level.items()},
        "strategies": comp,
        "summary": summary.allocation_summary(comp, budget, segments, use_truth=s.synthetic),
        "table": table.sort_values("expected_profit_gain", ascending=False).round(4).to_dict("records"),
    }


def user_ite(s: Session, user_id: int) -> dict:
    if s.engine is None:
        raise ValueError("Train the model first")
    pool = pd.concat([s.audience, s.history.drop(columns=[TREATMENT, OUTCOME])]).drop_duplicates(ID_COL)
    row = pool[pool[ID_COL] == user_id]
    if row.empty:
        raise KeyError(user_id)
    tau = float(s.engine.tau(row)[0])
    lo, hi = s.engine.tau_interval(row)
    p0 = float(s.engine.p0(row)[0])
    return {
        "user_id": user_id, "baseline_prob": p0,
        "levels": [{"discount": d, "prob": float(np.clip(p0 + tau * d, 0, 1)), "lift": tau * d,
                    "lift_low": float(lo[0] * d), "lift_high": float(hi[0] * d)} for d in DISCOUNT_LEVELS if d > 0],
    }


def drift_check(s: Session, current: pd.DataFrame) -> dict:
    if s.drift is None:
        raise ValueError("Train the model first")
    return s.drift.check(current)