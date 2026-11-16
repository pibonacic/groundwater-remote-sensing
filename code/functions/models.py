"""Estimators and the pipeline factory. Everything that learns from data is an estimator or
transformer, so it is refit inside each fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
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


def make_hybrid(cfg):
    """ElasticNet on standardized inputs for the level + random forest on what it leaves."""
    return LinearPlusTrees(Pipeline([("scale", StandardScaler()), ("enet", ElasticNet(max_iter=cfg.ENET_MAX_ITER))]),
                           RandomForestRegressor(n_estimators=300))


# Candidates: make(cfg) builds the bare estimator (cfg = the config module), scale says whether it needs standardized
# inputs. grid(cfg) is relative to that estimator and reads the grids of config.py (ENET_GRID, RF_GRID, CATBOOST_GRID).
# Two versions of the hybrid:
# - hybrid_joint: ElasticNet (ENET_GRID) and trees (RF_GRID) tuned together, by the RMSE of the hybrid;
# - hybrid_staged: 05 fixes its ElasticNet to the alpha / l1_ratio chosen for elastic_net (same columns, scaling,
#   clip and grid, so the same linear model) and only its trees (RF_GRID) are tuned, by the RMSE of the hybrid. It
#   depends on elastic_net coming before it in MODEL_SPECS.
MODEL_SPECS = {
    "elastic_net": {"make": lambda cfg: ElasticNet(max_iter=cfg.ENET_MAX_ITER), "scale": True,
                    "grid": lambda cfg: cfg.ENET_GRID},
    "random_forest": {"make": lambda cfg: RandomForestRegressor(n_estimators=300), "scale": False,
                      "grid": lambda cfg: cfg.RF_GRID},
    "catboost": {"make": lambda cfg: CatBoostRegressor(iterations=500, verbose=0, allow_writing_files=False,
                                                             thread_count=1),
                 "scale": False, "grid": lambda cfg: cfg.CATBOOST_GRID},
    "hybrid_joint": {"make": make_hybrid, "scale": False,
                     "grid": lambda cfg: {**{f"lin__enet__{k}": v for k, v in cfg.ENET_GRID.items()},
                                          **{f"trees__{k}": v for k, v in cfg.RF_GRID.items()}}},
    "hybrid_staged": {"make": make_hybrid, "scale": False,
                      "grid": lambda cfg: {f"trees__{k}": v for k, v in cfg.RF_GRID.items()}},
}


def build_pipeline(model_name, cfg, columns=None, limits=(None, None), seed=0, params=None):
    """Columns -> scaling -> model -> clip, with only the steps asked for.

    columns fixes the input columns (the frozen list); limits clips the predictions (None = range of the training y); params are pipeline
    parameters (as returned by tune).
    """
    spec = MODEL_SPECS[model_name]
    model = spec["make"](cfg)
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

