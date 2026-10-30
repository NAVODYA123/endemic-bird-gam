"""Shared helpers for the endemic-bird GAM project (imported by the notebook AND the Streamlit app).

Kept in a separate module so that the pickled model can be loaded by app.py.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from pygam import LogisticGAM, s, f, l


class GAMPreprocessor(BaseEstimator, TransformerMixin):
    """Raw sighting table -> numeric matrix for the GAM.

    * right-skewed numerics: negative sentinels -> NaN, median-impute, log1p
    * other numerics      : median-impute
    * categoricals        : integer codes (categories unseen in training -> most frequent training category)
    No scaling is needed: splines are invariant to monotone rescaling of the axis.
    """
    def __init__(self, num_log, num_plain, cat):
        self.num_log, self.num_plain, self.cat = num_log, num_plain, cat

    @property
    def columns_(self):
        return list(self.num_log) + list(self.num_plain) + list(self.cat)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.medians_ = {}
        for c in self.num_log:
            v = pd.to_numeric(X[c], errors='coerce'); self.medians_[c] = float(v.where(v >= 0).median())
        for c in self.num_plain:
            self.medians_[c] = float(pd.to_numeric(X[c], errors='coerce').median())
        self.cat_maps_, self.cat_mode_ = {}, {}
        for c in self.cat:
            sv = X[c].astype(str)
            self.cat_maps_[c] = {v: i for i, v in enumerate(sorted(sv.unique()))}
            self.cat_mode_[c] = self.cat_maps_[c][sv.mode().iloc[0]]
        return self

    def transform(self, X):
        X = pd.DataFrame(X); out = {}
        for c in self.num_log:
            v = pd.to_numeric(X[c], errors='coerce'); v = v.where(v >= 0).fillna(self.medians_[c])
            out[c] = np.log1p(v.astype(float)).values
        for c in self.num_plain:
            out[c] = pd.to_numeric(X[c], errors='coerce').fillna(self.medians_[c]).astype(float).values
        for c in self.cat:
            out[c] = X[c].astype(str).map(self.cat_maps_[c]).fillna(self.cat_mode_[c]).astype(int).values
        return pd.DataFrame(out, index=X.index)[self.columns_]


class GAMClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-style wrapper around pygam.LogisticGAM.

    Terms: spline s() for numeric features (cyclic spline for `cyclic` features such as month),
    factor f() for categorical ones.  `pos_weight` is kept at 1.0 (unweighted, calibrated
    probabilities): heavy class-weights made pyGAM's PIRLS fit unstable in our tests.
    """
    def __init__(self, feature_names, cat_names, cyclic=('month',), n_splines=12, lam=1.0, pos_weight=1.0, max_iter=100):
        self.feature_names, self.cat_names, self.cyclic = feature_names, cat_names, cyclic
        self.n_splines, self.lam, self.pos_weight, self.max_iter = n_splines, lam, pos_weight, max_iter

    def _terms(self, X):
        terms = None
        for i, name in enumerate(self.feature_names):
            if name in self.cat_names:
                t = f(i)
            elif name in self.cyclic:
                t = s(i, n_splines=min(self.n_splines, 10), basis='cp')
            else:
                nu = int(np.unique(X[:, i]).size)
                t = l(i) if nu < 6 else s(i, n_splines=int(max(5, min(self.n_splines, nu - 1))))
            terms = t if terms is None else terms + t
        return terms

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y).astype(int)
        self.gam_ = LogisticGAM(self._terms(X), lam=self.lam, max_iter=self.max_iter)
        self.gam_.fit(X, y, weights=np.where(y == 1, self.pos_weight, 1.0))
        self.classes_ = np.array([0, 1])
        self.cat_levels_ = {n: np.unique(X[:, self.feature_names.index(n)]).astype(int) for n in self.cat_names}
        return self

    def decision_function(self, X):           # log-odds (linear predictor)
        return np.asarray(self.gam_._linear_predictor(np.asarray(X, dtype=float))).ravel()

    def predict_proba(self, X):
        p = np.asarray(self.gam_.predict_mu(np.asarray(X, dtype=float)), dtype=float).ravel()
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def term_curve(model, i, n=100):
    """x-grid, effect (log-odds) and 95 % CI of term i of a fitted GAMClassifier."""
    g = model.gam_; name = model.feature_names[i]
    if name in model.cat_names:                      # categorical: one point per category code
        lv = model.cat_levels_[name]
        XX = np.zeros((len(lv), len(model.feature_names))); XX[:, i] = lv
    else:
        XX = g.generate_X_grid(term=i, n=n)
    pdep, ci = g.partial_dependence(term=i, X=XX, width=0.95)
    return XX[:, i], pdep, ci


def term_contributions(model, X_prepped_row):
    """Per-feature contribution (log-odds) of ONE prepped row + the intercept."""
    g = model.gam_; Xr = np.asarray(X_prepped_row, dtype=float).reshape(1, -1)
    contrib = np.array([float(g.partial_dependence(term=i, X=Xr)[0]) for i in range(len(model.feature_names))])
    return contrib, float(g.coef_[-1])


def term_matrix(model, X):
    """(n_rows x n_features) matrix of every term's contribution (log-odds), for prepped rows X."""
    g = model.gam_; X = np.asarray(X, dtype=float)
    return np.column_stack([np.asarray(g.partial_dependence(term=i, X=X)).ravel() for i in range(len(model.feature_names))])


def exact_shap(model, X, background):
    """Exact (interventional) SHAP values of an additive log-odds model:
    phi_j(x) = f_j(x_j) - mean_background f_j.   Returns (shap_values, base_value)."""
    Cb = term_matrix(model, background)
    return term_matrix(model, X) - Cb.mean(0), float(model.gam_.coef_[-1] + Cb.mean(0).sum())
