"""Estimators and the pipeline factory. Everything that learns from data is an estimator or
transformer, so it is refit inside each fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from catboost import CatBoostRegressor

from functions.features import CALENDAR_COLS

CLIP_PREFIX = "model__regressor__"       # grid keys of the model inside the pipeline built by build_pipeline


class SeasonalClimatology(RegressorMixin, BaseEstimator):
    """Mean water level per calendar month of the training data.

    Months absent from the training data get the overall mean. With harmonics=True a
    smooth seasonal curve (sin_doy, cos_doy) is fitted instead of monthly means.
    """

    def __init__(self, harmonics=False):
        self.harmonics = harmonics

    def fit(self, X, y):
        if self.harmonics:
            self.model_ = LinearRegression().fit(X[CALENDAR_COLS], y)
        else:
            self.monthly_ = y.groupby(X.index.month).mean()
            self.mean_ = y.mean()
        return self

    def predict(self, X):
        if self.harmonics:
            return self.model_.predict(X[CALENDAR_COLS])
        return X.index.month.map(self.monthly_).fillna(self.mean_).to_numpy()


def enet_params(gap, inner_splits, l1_ratios, max_iter):
    """Arguments of ElasticNetCV; its inner CV is temporal and keeps the same gap."""
    return {"l1_ratio": l1_ratios, "max_iter": max_iter, "cv": TimeSeriesSplit(n_splits=inner_splits, gap=gap)}


def make_linear_baseline(gap, inner_splits, l1_ratios, max_iter):
    """Scaling + ElasticNetCV on every column."""
    return Pipeline([
        ("scale", StandardScaler().set_output(transform="pandas")),
        ("enet", ElasticNetCV(**enet_params(gap, inner_splits, l1_ratios, max_iter))),
    ])


class PhysicalClipRegressor(RegressorMixin, BaseEstimator):
    """Clip the predictions of regressor to [lower, upper]; None = min / max of the training y."""

    def __init__(self, regressor, lower=None, upper=None):
        self.regressor = regressor
        self.lower = lower
        self.upper = upper

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor).fit(X, y)
        self.lower_ = y.min() if self.lower is None else self.lower
        self.upper_ = y.max() if self.upper is None else self.upper
        return self

    def predict(self, X):
        return np.clip(self.regressor_.predict(X), self.lower_, self.upper_)


class LinearPlusTrees(RegressorMixin, BaseEstimator):
    """lin predicts the level and trees predict what lin leaves (extrapolates like lin, bends like trees)."""

    def __init__(self, lin, trees):
        self.lin = lin
        self.trees = trees

    def fit(self, X, y):
        self.lin_ = clone(self.lin).fit(X, y)
        self.trees_ = clone(self.trees).fit(X, y - self.lin_.predict(X))
        return self

    def predict(self, X):
        return self.lin_.predict(X) + self.trees_.predict(X)


def select_columns(X, columns):
    return X[columns]


# Candidates: make(enet) builds the bare estimator (enet = enet_params), scale says whether it
# needs standardized inputs, grid is relative to that estimator. simplest orders the configurations from
# the most regularized: (parameter, True if smaller = more regularized), compared in that order (None = largest).
MODEL_SPECS = {
    "random_forest": {"make": lambda enet: RandomForestRegressor(n_estimators=300), "scale": False,
                      "grid": {"max_depth": [4, 8, None], "min_samples_leaf": [5, 10, 20, 40], "max_features": [0.5, 1.0]},
                      "simplest": [("min_samples_leaf", False), ("max_depth", True), ("max_features", True)]},
    "catboost": {"make": lambda enet: CatBoostRegressor(iterations=500, verbose=0, allow_writing_files=False, thread_count=1),
                 "scale": False, "grid": {"learning_rate": [0.02, 0.05], "depth": [2, 3, 4], "l2_leaf_reg": [3, 10, 30]},
                 "simplest": [("depth", True), ("l2_leaf_reg", False), ("learning_rate", True)]},
    "hybrid": {"make": lambda enet: LinearPlusTrees(
                   Pipeline([("scale", StandardScaler()), ("enet", ElasticNetCV(**enet))]),
                   RandomForestRegressor(n_estimators=300)),
               "scale": False, "grid": {"trees__max_depth": [4, 8, None], "trees__min_samples_leaf": [5, 10, 20, 40],
                                        "trees__max_features": [0.5, 1.0]},
               "simplest": [("trees__min_samples_leaf", False), ("trees__max_depth", True), ("trees__max_features", True)]},
}


def build_pipeline(model_name, enet, columns=None, limits=(None, None), seed=0, params=None):
    """Columns -> scaling -> model -> clip, with only the steps asked for.

    columns fixes the input columns (the frozen list); limits clips the predictions (None = range of the training y); params are pipeline
    parameters (as returned by tune).
    """
    spec = MODEL_SPECS[model_name]
    model = spec["make"](enet)
    model.set_params(**{k: seed for k in model.get_params() if k.endswith("random_state")})
    steps = []
    if columns is not None:
        steps.append(("columns", FunctionTransformer(select_columns, kw_args={"columns": list(columns)})))
    if spec["scale"]:
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    steps.append(("model", PhysicalClipRegressor(model, *limits)))
    return Pipeline(steps).set_params(**(params or {}))


def pipeline_grid(model_name):
    """Hyperparameter grid of a candidate, with the keys of the pipeline from build_pipeline."""
    return {CLIP_PREFIX + key: values for key, values in MODEL_SPECS[model_name]["grid"].items()}


def pipeline_simplest(model_name):
    """Regularization order of a candidate (see MODEL_SPECS), with the keys of the pipeline from build_pipeline."""
    return [(CLIP_PREFIX + key, ascending) for key, ascending in MODEL_SPECS[model_name].get("simplest", [])]


def tune(pipeline, grid, simplest, X, y, split_sets):
    """Grid search over the folds of the joint scheme with a paired one-standard-error rule.

    For each configuration, diff = its RMSE minus the RMSE of the lowest-mean configuration in the same fold;
    it is within_noise when the mean diff is no larger than the standard error of the diffs
    (std / sqrt(number of folds), column paired_se), which discounts how hard each fold is for every model; among
    them the most regularized one is chosen, following simplest = [(parameter, True if smaller = more
    regularized), ...] in that order. Returns the chosen parameters and a table with the RMSE of each fold
    (rmse_fold_1, ...), its mean and std across folds, within_noise, lowest (lowest mean) and chosen.
    """
    joint = split_sets["joint"]
    search = GridSearchCV(pipeline, grid, cv=joint, scoring="neg_root_mean_squared_error", refit=False).fit(X, y)
    params = search.cv_results_["params"]
    scores = -np.array([search.cv_results_[f"split{i}_test_score"] for i in range(len(joint))])
    summary = pd.DataFrame({"params": params, **{f"rmse_fold_{i + 1}": fold for i, fold in enumerate(scores)},
                            "rmse_mean": scores.mean(axis=0), "rmse_std": scores.std(axis=0, ddof=1)})
    lowest = summary["rmse_mean"].idxmin()
    diff = scores - scores[:, [lowest]]
    summary["paired_se"] = diff.std(axis=0, ddof=1) / np.sqrt(len(joint))
    summary["within_noise"] = diff.mean(axis=0) <= summary["paired_se"]
    order = pd.DataFrame({key: [np.inf if p[key] is None else p[key] for p in params] for key, _ in simplest},
                         index=summary.index)
    candidates = order[summary["within_noise"]]
    chosen = candidates.sort_values([k for k, _ in simplest], ascending=[a for _, a in simplest]).index[0] if simplest else lowest
    summary["lowest"] = summary.index == lowest
    summary["chosen"] = summary.index == chosen
    return params[chosen], summary
