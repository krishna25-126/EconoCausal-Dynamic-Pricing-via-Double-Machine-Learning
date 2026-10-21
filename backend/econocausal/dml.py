"""Double/Debiased Machine Learning engine (EconML) for individual treatment effects.

How it works, in plain words:
  1. Predict the outcome from the features alone      -> residual outcome
  2. Predict the discount from the features alone     -> residual discount
  3. Regress residual outcome on residual discount    -> causal effect

Step 1 and 2 use flexible ML models (LightGBM) with cross-fitting, so the
confounding by features is "stripped out". A causal forest in step 3 lets the
effect differ from customer to customer (the Individual Treatment Effect, ITE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

from .config import FEATURES, OUTCOME, RANDOM_STATE, TREATMENT


def _lgbm(seed: int) -> LGBMRegressor:
    return LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1)


class CausalEngine:
    def __init__(self, n_trees: int = 300, seed: int = RANDOM_STATE):
        self.n_trees, self.seed = max(4, (n_trees // 4) * 4), seed  # forest size must be a multiple of 4
        self.est: CausalForestDML | None = None
        self.naive_effect_: float | None = None

    def fit(self, df: pd.DataFrame) -> "CausalEngine":
        X, T, Y = df[FEATURES], df[TREATMENT].to_numpy(float), df[OUTCOME].to_numpy(float)
        self.est = CausalForestDML(
            model_y=_lgbm(self.seed), model_t=_lgbm(self.seed + 1),
            discrete_treatment=False, n_estimators=self.n_trees, min_samples_leaf=25,
            max_depth=None, cv=3, random_state=self.seed,
        )
        self.est.fit(Y, T, X=X)
        # Naive predictive baseline: ignores confounding.
        self.naive_effect_ = float(LinearRegression().fit(df[[TREATMENT]], df[OUTCOME]).coef_[0])
        # Nuisance predictions used later to recover baseline purchase probability.
        self._ey = cross_val_predict(_lgbm(self.seed), X, Y, cv=5)
        self._et = cross_val_predict(_lgbm(self.seed + 1), X, T, cv=5)
        return self

    # --- predictions -----------------------------------------------------
    def tau(self, X: pd.DataFrame) -> np.ndarray:
        """Estimated change in purchase probability per $1 of discount."""
        return np.asarray(self.est.const_marginal_effect(X[FEATURES])).ravel()

    def tau_interval(self, X: pd.DataFrame, alpha: float = 0.1):
        lo, hi = self.est.const_marginal_effect_interval(X[FEATURES], alpha=alpha)
        return np.asarray(lo).ravel(), np.asarray(hi).ravel()

    def ite(self, X: pd.DataFrame, t0: float, t1: float) -> np.ndarray:
        """Individual Treatment Effect of moving the discount from t0 to t1."""
        return self.tau(X) * (t1 - t0)

    def fit_baseline(self, df: pd.DataFrame) -> "CausalEngine":
        """Learn p0(X) = E[Y|X] - tau(X) E[T|X] as a function of X (so it can
        be applied to new customers)."""
        tau = self.tau(df)
        p0_target = np.clip(self._ey - tau * self._et, 0.01, 0.99)
        self._p0_model = _lgbm(self.seed + 2).fit(df[FEATURES], p0_target)
        return self

    def p0(self, X: pd.DataFrame) -> np.ndarray:
        """Estimated probability of buying with no discount."""
        return np.clip(self._p0_model.predict(X[FEATURES]), 0.01, 0.99)

    def average_effect(self, df: pd.DataFrame) -> float:
        return float(self.tau(df).mean())