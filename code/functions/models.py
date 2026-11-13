"""Estimators and the pipeline factory. Everything that learns from data is an estimator or
transformer, so it is refit inside each fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from catboost import CatBoostRegressor


CLIP_PREFIX = "model__regressor__"       # grid keys of the model inside the pipeline built by build_pipeline


class SeasonalClimatology(RegressorMixin, BaseEstimator):
    """Mean water level per calendar month of the training data.

    Months absent from the training data get the overall mean.
    """

    def fit(self, X, y):
        self.monthly_ = y.groupby(X.index.month).mean()
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        return X.index.month.map(self.monthly_).fillna(self.mean_).to_numpy()


def enet_params(gap, inner_splits, l1_ratios, max_iter):
    """Arguments of the ElasticNetCV inside the hybrid model; its inner CV is temporal and keeps the same gap."""
    return {"l1_ratio": l1_ratios, "max_iter": max_iter, "cv": TimeSeriesSplit(n_splits=inner_splits, gap=gap)}


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


# Candidates: make(enet, cfg) builds the bare estimator (enet = enet_params of the hybrid, cfg = the config module),
# scale says whether it needs standardized inputs, grid(cfg) is relative to that estimator.
MODEL_SPECS = {
    "elastic_net": {"make": lambda enet, cfg: ElasticNet(max_iter=cfg.ENET_MAX_ITER), "scale": True,
                    "grid": lambda cfg: {"alpha": cfg.ENET_ALPHAS, "l1_ratio": cfg.ENET_L1_RATIOS}},
    "random_forest": {"make": lambda enet, cfg: RandomForestRegressor(n_estimators=300), "scale": False,
                      "grid": lambda cfg: {"max_depth": [4, 8, None], "min_samples_leaf": [5, 10, 20, 40],
                                           "max_features": [0.5, 1.0]}},
    "catboost": {"make": lambda enet, cfg: CatBoostRegressor(iterations=500, verbose=0, allow_writing_files=False,
                                                             thread_count=1),
                 "scale": False, "grid": lambda cfg: {"learning_rate": [0.02, 0.05], "depth": [2, 3, 4],
                                                      "l2_leaf_reg": [3, 10, 30]}},
    "hybrid": {"make": lambda enet, cfg: LinearPlusTrees(
                   Pipeline([("scale", StandardScaler()), ("enet", ElasticNetCV(**enet))]),
                   RandomForestRegressor(n_estimators=300)),
               "scale": False, "grid": lambda cfg: {"trees__max_depth": [4, 8, None],
                                                    "trees__min_samples_leaf": [5, 10, 20, 40],
                                                    "trees__max_features": [0.5, 1.0]}},
}


def build_pipeline(model_name, enet, cfg, columns=None, limits=(None, None), seed=0, params=None):
    """Columns -> scaling -> model -> clip, with only the steps asked for.

    columns fixes the input columns (the frozen list); limits clips the predictions (None = range of the training y); params are pipeline
    parameters (as returned by tune).
    """
    spec = MODEL_SPECS[model_name]
    model = spec["make"](enet, cfg)
    model.set_params(**{k: seed for k in model.get_params() if k.endswith("random_state")})
    steps = []
    if columns is not None:
        steps.append(("columns", FunctionTransformer(select_columns, kw_args={"columns": list(columns)})))
    if spec["scale"]:
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    steps.append(("model", PhysicalClipRegressor(model, *limits)))
    return Pipeline(steps).set_params(**(params or {}))


def pipeline_grid(model_name, cfg):
    """Hyperparameter grid of a candidate, with the keys of the pipeline from build_pipeline."""
    return {CLIP_PREFIX + key: values for key, values in MODEL_SPECS[model_name]["grid"](cfg).items()}


def tune(pipeline, grid, X, y, splits):
    """Grid search over the hv_blocks folds: the configuration with the lowest mean RMSE across folds is chosen.

    Returns the chosen parameters and a table with the RMSE of each fold (rmse_fold_1, ...), its mean and std
    across folds and chosen.
    """
    search = GridSearchCV(pipeline, grid, cv=splits, scoring="neg_root_mean_squared_error", refit=False,
                          n_jobs=-1).fit(X, y)
    params = search.cv_results_["params"]
    scores = -np.array([search.cv_results_[f"split{i}_test_score"] for i in range(len(splits))])
    summary = pd.DataFrame({"params": params, **{f"rmse_fold_{i + 1}": fold for i, fold in enumerate(scores)},
                            "rmse_mean": scores.mean(axis=0), "rmse_std": scores.std(axis=0, ddof=1)})
    chosen = summary["rmse_mean"].idxmin()
    summary["chosen"] = summary.index == chosen
    return params[chosen], summary
