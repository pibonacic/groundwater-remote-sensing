"""Estimators and the pipeline factory. Everything that learns from data is an estimator or
transformer, so it is refit inside each fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

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
# needs standardized inputs, grid is relative to that estimator.
MODEL_SPECS = {
    "elasticnet": {"make": lambda enet: ElasticNetCV(**enet), "scale": True, "grid": {}},
    "svr": {"make": lambda enet: TransformedTargetRegressor(SVR(), transformer=StandardScaler()), "scale": True,
            "grid": {"regressor__C": [0.3, 1, 3, 10], "regressor__epsilon": [0.05, 0.2]}},
    "random_forest": {"make": lambda enet: RandomForestRegressor(n_estimators=200), "scale": False,
                      "grid": {"max_depth": [3, 6, None], "min_samples_leaf": [5, 20]}},
    "boosting": {"make": lambda enet: HistGradientBoostingRegressor(max_iter=200), "scale": False,
                 "grid": {"learning_rate": [0.03, 0.1], "max_depth": [2, 3], "min_samples_leaf": [20, 40]}},
    "xgboost": {"make": lambda enet: XGBRegressor(n_estimators=200, tree_method="hist", n_jobs=1), "scale": False,
                "grid": {"learning_rate": [0.03, 0.1], "max_depth": [2, 3], "min_child_weight": [5, 20]}},
    "catboost": {"make": lambda enet: CatBoostRegressor(iterations=200, verbose=0, allow_writing_files=False, thread_count=1),
                 "scale": False, "grid": {"learning_rate": [0.03, 0.1], "depth": [2, 3], "l2_leaf_reg": [3, 10]}},
    "hybrid": {"make": lambda enet: LinearPlusTrees(
                   Pipeline([("scale", StandardScaler()), ("enet", ElasticNetCV(**enet))]),
                   RandomForestRegressor(n_estimators=200)),
               "scale": False, "grid": {"trees__max_depth": [3, None], "trees__min_samples_leaf": [5, 20]}},
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


def tune(pipeline, grid, X, y, split_sets):
    """Grid search over the joint folds (forward + backward together).

    Returns the best parameters (lowest mean RMSE over all folds) and a table with the mean and
    std of the RMSE across folds for each configuration, jointly and for the forward and backward
    folds separately; within_noise marks the configurations whose joint mean is no more than one
    std above the best, among which a simpler one may be preferred.
    """
    joint = split_sets["joint"]
    search = GridSearchCV(pipeline, grid, cv=joint, scoring="neg_root_mean_squared_error", refit=False).fit(X, y)
    scores = -np.array([search.cv_results_[f"split{i}_test_score"] for i in range(len(joint))])
    n_fwd = len(split_sets["fwd"])
    summary = pd.DataFrame({"params": search.cv_results_["params"]})
    for name, part in {"joint": scores, "fwd": scores[:n_fwd], "bwd": scores[n_fwd:]}.items():
        suffix = "" if name == "joint" else f"_{name}"
        summary[f"rmse_mean{suffix}"] = part.mean(axis=0)
        summary[f"rmse_std{suffix}"] = part.std(axis=0, ddof=1)
    best = summary["rmse_mean"].idxmin()
    summary["within_noise"] = summary["rmse_mean"] <= summary.loc[best, "rmse_mean"] + summary.loc[best, "rmse_std"]
    return search.cv_results_["params"][best], summary
