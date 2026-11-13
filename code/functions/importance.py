"""Out-of-fold SHAP importance and feature elimination by retraining on the hv_blocks folds."""
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_error

from functions.models import LinearPlusTrees

GROUPED_UNITS = {}   # unit -> columns kept or dropped together, e.g. {"doy": ["sin_doy", "cos_doy"]}; empty = every column alone


def feature_units(columns):
    """Unit -> columns: the groups of GROUPED_UNITS form one unit each, every other column is its own unit."""
    units, grouped = {}, {c: u for u, cols in GROUPED_UNITS.items() for c in cols}
    for c in columns:
        units.setdefault(grouped.get(c, c), []).append(c)
    return units


def with_columns(pipeline, columns):
    """Unfitted copy of a pipeline from build_pipeline that uses only the given columns."""
    return clone(pipeline).set_params(columns__kw_args={"columns": list(columns)})


def _linear_shap(model, X_train, X):
    """SHAP of a linear model (independent features, background = training rows): coef * (x - mean of x_train)."""
    X_train = np.asarray(X_train, float)
    explainer = shap.LinearExplainer(model, shap.maskers.Independent(X_train, max_samples=len(X_train)))
    return explainer.shap_values(np.asarray(X, float)), float(np.ravel(explainer.expected_value)[0])


def _tree_shap(model, X):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X), float(np.ravel(explainer.expected_value)[0])


def explain(fitted, X_train, X):
    """SHAP values (rows of X x input columns of the model) and base value of a fitted pipeline from build_pipeline.

    The regressor is explained before the physical clip (the clip is not additive); the hybrid adds the SHAP of its
    linear part (in the scaled space) and of its trees (fitted on what the linear part leaves).
    """
    transform = fitted[:-1]
    Xt_train, Xt = transform.transform(X_train), transform.transform(X)
    model = fitted[-1].regressor_
    if isinstance(model, LinearPlusTrees):
        scaler, enet = model.lin_[:-1], model.lin_[-1]
        lin, lin_base = _linear_shap(enet, scaler.transform(Xt_train), scaler.transform(Xt))
        trees, trees_base = _tree_shap(model.trees_, Xt)
        values, base = lin + trees, lin_base + trees_base
    elif hasattr(model, "coef_"):
        values, base = _linear_shap(model, Xt_train, Xt)
    else:
        values, base = _tree_shap(model, Xt)
    return pd.DataFrame(values, index=X.index, columns=list(Xt.columns)), base


def shap_oof(pipeline, X, y, splits):
    """Out-of-fold SHAP: on each fold a fresh clone is fitted on the training days and its validation block is
    explained. Returns (SHAP values dates x columns, base value per date, unclipped prediction per date)."""
    values, base, raw = [], [], []
    for train, test in splits:
        fitted = clone(pipeline).fit(X.iloc[train], y.iloc[train])
        v, b = explain(fitted, X.iloc[train], X.iloc[test])
        values.append(v)
        base.append(pd.Series(b, index=v.index))
        raw.append(pd.Series(fitted[-1].regressor_.predict(fitted[:-1].transform(X.iloc[test])), index=v.index))
    return pd.concat(values), pd.concat(base), pd.concat(raw)


def shap_by_unit(values, units):
    """SHAP of each unit = sum of the SHAP of its columns (additivity)."""
    return pd.DataFrame({u: values[cols].sum(axis=1) for u, cols in units.items() if set(cols) <= set(values)})


def oof_rmse(pipeline, columns, X, y, splits):
    """Retrain with the given columns on every fold. Returns (out-of-fold predictions, mean RMSE across folds)."""
    pipe = with_columns(pipeline, columns)
    oof = pd.Series(np.nan, index=y.index)
    rmse = []
    for train, test in splits:
        pred = clone(pipe).fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
        oof.iloc[test] = pred
        rmse.append(root_mean_squared_error(y.iloc[test], pred))
    return oof, float(np.mean(rmse))


def _columns(units, names):
    return [c for u in names for c in units[u]]


def backward_elimination(pipeline, units, X, y, splits, n_jobs=-1):
    """Greedy backward elimination with retraining: at each step every remaining unit is dropped in turn, the model
    is retrained on the folds and the unit whose removal leaves the lowest mean RMSE across folds is removed.

    Returns (path, oof): path has one row per step (step 0 = all units) with the removed unit, the number of
    units left, the units left and rmse_mean; oof has the out-of-fold predictions of each step (columns step_k).
    """
    current = list(units)
    oof0, rmse0 = oof_rmse(pipeline, _columns(units, current), X, y, splits)
    rows = [{"step": 0, "removed": "", "n_units": len(current), "rmse_mean": rmse0, "units": ", ".join(current)}]
    oofs = {"step_0": oof0}
    parallel = Parallel(n_jobs=n_jobs)
    while len(current) > 1:
        trials = parallel(delayed(oof_rmse)(pipeline, _columns(units, [u for u in current if u != drop]), X, y, splits)
                          for drop in current)
        best = int(np.argmin([rmse for _, rmse in trials]))
        removed = current.pop(best)
        step = len(rows)
        rows.append({"step": step, "removed": removed, "n_units": len(current), "rmse_mean": trials[best][1],
                     "units": ", ".join(current)})
        oofs[f"step_{step}"] = trials[best][0]
    return pd.DataFrame(rows).set_index("step"), pd.DataFrame(oofs)


def readd_check(pipeline, units, chosen, X, y, splits, n_jobs=-1):
    """Add back, one at a time, every unit left out of chosen and retrain. Returns (mean RMSE across folds per unit
    added, out-of-fold predictions with a column per unit added plus the column chosen)."""
    others = [u for u in units if u not in chosen]
    trials = Parallel(n_jobs=n_jobs)(delayed(oof_rmse)(pipeline, _columns(units, chosen + [u]), X, y, splits) for u in others)
    base, _ = oof_rmse(pipeline, _columns(units, chosen), X, y, splits)
    oofs = pd.DataFrame({"chosen": base, **{f"+{u}": oof for u, (oof, _) in zip(others, trials)}})
    return pd.Series({f"+{u}": rmse for u, (_, rmse) in zip(others, trials)}, name="rmse_mean"), oofs
