"""Temporal folds, metrics and the evaluation loop."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def assert_daily_contiguous(index):
    """Fail if the index is unsorted or has missing days (folds count rows, not dates)."""
    if not index.is_monotonic_increasing or not (index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("index must be sorted with one row per day, without gaps")


def forward_splits(index, n_splits, test_size, gap, min_train):
    """Expanding-window folds: train on the past, test on the next block."""
    assert_daily_contiguous(index)
    splits = list(TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap).split(np.arange(len(index))))
    if len(splits[0][0]) < min_train:
        raise ValueError(f"first training set has {len(splits[0][0])} rows, less than min_train={min_train}")
    return splits


def backward_splits(index, n_splits, test_size, gap, min_train):
    """Mirror of forward_splits: train on the future, test on earlier blocks (hindcast direction)."""
    n = len(index)
    return [(np.sort(n - 1 - train), np.sort(n - 1 - test))
            for train, test in forward_splits(index, n_splits, test_size, gap, min_train)]


def halves_splits(index, gap):
    """Two single-fold schemes on the halves of the period.

    halves_fwd trains on the first half and tests on the second; halves_bwd the other way round.
    The gap is dropped from the end of the training half that touches the test half.
    """
    assert_daily_contiguous(index)
    half = len(index) // 2
    first, second = np.arange(half), np.arange(half, len(index))
    return {"halves_fwd": [(first[:half - gap], second)], "halves_bwd": [(second[gap:], first)]}


def make_split_sets(index, gap, n_splits, test_size, min_train):
    """Forward and backward fold schemes with the same parameters, and both together as "joint"."""
    fwd = forward_splits(index, n_splits, test_size, gap, min_train)
    bwd = backward_splits(index, n_splits, test_size, gap, min_train)
    if set(np.concatenate([test for _, test in fwd])) & set(np.concatenate([test for _, test in bwd])):
        raise ValueError("forward and backward test folds overlap; lower N_SPLITS or TEST_SIZE")
    return {"fwd": fwd, "bwd": bwd, "joint": fwd + bwd}


def bias(y, pred):
    """Mean error (pred - y); positive = over-prediction."""
    return float(np.mean(np.asarray(pred) - np.asarray(y)))


def nrmse(y, pred, value_range):
    return root_mean_squared_error(y, pred) / value_range


def skill_score(y, pred, pred_base):
    """1 - RMSE / RMSE of the baseline; positive = better than the baseline."""
    return 1 - root_mean_squared_error(y, pred) / root_mean_squared_error(y, pred_base)


def metrics_table(y, pred, value_range, strata=None, baselines=None):
    """All metrics overall and per group of each stratum (labels aligned with y).

    baselines maps a name to baseline predictions; adds a skill_vs_<name> column.
    """
    strata, baselines = strata or {}, baselines or {}
    rows = []
    for stratum, labels in {"all": pd.Series("all", index=y.index), **strata}.items():
        for group, idx in labels.groupby(labels).groups.items():
            yt, pt = y.loc[idx], pred.loc[idx]
            row = {"stratum": stratum, "group": group, "n": len(yt), "rmse": root_mean_squared_error(yt, pt),
                   "mae": mean_absolute_error(yt, pt), "bias": bias(yt, pt),
                   "nrmse": nrmse(yt, pt, value_range), "nse": r2_score(yt, pt)}   # NSE = R2 of sklearn
            for name, base in baselines.items():
                row[f"skill_vs_{name}"] = skill_score(yt, pt, base.loc[idx])
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(estimator, X, y, splits, inspect=None):
    """Fit a fresh clone on each train set and predict its test set.

    Returns the out-of-fold predictions (NaN where never tested), metrics per fold and,
    if inspect(fitted, X_test, y_test) is given, what it returned for each fold.
    """
    oof = pd.Series(np.nan, index=y.index, name="oof")
    rows, inspected = [], []
    for i, (train, test) in enumerate(splits):
        fitted = clone(estimator).fit(X.iloc[train], y.iloc[train])
        pred = fitted.predict(X.iloc[test])
        y_test = y.iloc[test]
        oof.iloc[test] = pred
        rows.append({"fold": i, "n_test": len(test), "rmse": root_mean_squared_error(y_test, pred),
                     "mae": mean_absolute_error(y_test, pred), "bias": bias(y_test, pred),
                     "nse": r2_score(y_test, pred)})
        if inspect is not None:
            inspected.append(inspect(fitted, X.iloc[test], y_test))
    return {"oof": oof, "folds": pd.DataFrame(rows).set_index("fold"), "inspected": inspected}


def compare(models, X, y, split_sets, baseline):
    """Evaluate every model on every fold scheme.

    Returns (table, oof): one row per scheme x model with fold mean/std, the RMSE of each fold (rmse_fold_1, ...) and the skill
    against the baseline model on the same test rows; oof[scheme][model] is a Series.
    """
    rows, oof = [], {}
    for scheme, splits in split_sets.items():
        oof[scheme] = {}
        for name, estimator in models.items():
            result = evaluate(estimator, X, y, splits)
            oof[scheme][name] = result["oof"]
            folds = result["folds"]
            rows.append({"scheme": scheme, "model": name,
                         "rmse_mean": folds["rmse"].mean(), "rmse_std": folds["rmse"].std(),
                         "mae_mean": folds["mae"].mean(), "bias_mean": folds["bias"].mean(),
                         "nse_mean": folds["nse"].mean(),
                         **{f"rmse_fold_{i + 1}": v for i, v in enumerate(folds["rmse"])}})
    table = pd.DataFrame(rows)
    skill = []
    for scheme, name in zip(table["scheme"], table["model"]):
        tested = oof[scheme][baseline].notna()
        skill.append(skill_score(y[tested], oof[scheme][name][tested], oof[scheme][baseline][tested]))
    table["skill"] = skill
    return table, oof


def oof_report(oof, y, value_range, strata, baseline):
    """metrics_table of the pooled out-of-fold predictions of each scheme x model."""
    frames = []
    for scheme, by_model in oof.items():
        tested = by_model[baseline].notna()
        base = {baseline: by_model[baseline][tested]}
        labels = {name: s[tested] for name, s in strata.items()}
        for model, pred in by_model.items():
            table = metrics_table(y[tested], pred[tested], value_range, labels, base)
            frames.append(table.assign(scheme=scheme, model=model))
    return pd.concat(frames, ignore_index=True)
