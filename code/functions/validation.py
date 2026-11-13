"""hv_blocks folds, metrics and the evaluation loop."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def assert_daily_contiguous(index):
    """Fail if the index is unsorted or has missing days (folds count rows, not dates)."""
    if not index.is_monotonic_increasing or not (index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("index must be sorted with one row per day, without gaps")


def hv_block_splits(index, v, h):
    """hv-block folds: consecutive validation blocks of v days from the first day, the last block absorbing the
    remainder (len(index) // v folds). The training set of each fold is every other day except the h days on each
    side of the block (the first block has no gap before it and the last one none after it).
    """
    assert_daily_contiguous(index)
    n = len(index)
    n_folds = n // v
    if n_folds < 2:
        raise ValueError(f"development has {n} rows, fewer than two blocks of v={v}")
    rows = np.arange(n)
    splits = []
    for k in range(n_folds):
        start, end = k * v, (n if k == n_folds - 1 else (k + 1) * v)
        test = rows[start:end]
        train = rows[(rows < start - h) | (rows >= end + h)]
        splits.append((train, test))
    return splits


def hv_blocks_table(index, splits):
    """One row per fold: dates of the validation block, days of gap before / after it and size of train and test.

    The gap is every day between the block and the nearest training day; with no training day on one side, it runs
    to the edge of the development period (0 only for the first block before it and the last one after it).
    """
    rows = []
    for k, (train, test) in enumerate(splits, start=1):
        start, end = test.min(), test.max()
        before, after = train[train < start], train[train > end]
        rows.append({"fold": k, "test_start": index[start], "test_end": index[end], "n_test": len(test),
                     "gap_before": start - (before.max() + 1 if len(before) else 0),
                     "gap_after": (after.min() if len(after) else len(index)) - end - 1,
                     "n_train": len(train)})
    return pd.DataFrame(rows).set_index("fold")


def mean_error(y, pred):
    """Mean error (pred - y); positive = predicted deeper than observed."""
    return float(np.mean(np.asarray(pred) - np.asarray(y)))


def skill_score(y, pred, pred_base):
    """1 - RMSE / RMSE of the baseline; positive = better than the baseline."""
    return 1 - root_mean_squared_error(y, pred) / root_mean_squared_error(y, pred_base)


def pearson_r(y, pred):
    """Pearson correlation; NaN when either series is constant."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if y.std() == 0 or pred.std() == 0:
        return np.nan
    return float(np.corrcoef(y, pred)[0, 1])


def alpha(y, pred):
    """Variability ratio: standard deviation of the predictions over that of the observations."""
    return float(np.std(pred) / np.std(y))


def normalized_bias(y, pred):
    """B: mean error over the standard deviation of the observations, (mean(pred) - mean(y)) / std(y)."""
    return float((np.mean(pred) - np.mean(y)) / np.std(y))


def kge2(y, pred):
    """KGE'' (Kling-Gupta efficiency with a normalized bias): 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + B^2).
    1 is a perfect fit (r = 1, alpha = 1, B = 0)."""
    return float(1 - np.sqrt((pearson_r(y, pred) - 1) ** 2 + (alpha(y, pred) - 1) ** 2 + normalized_bias(y, pred) ** 2))


METRICS = ["rmse", "mae", "me", "r", "alpha", "b", "kge2", "skill"]
METRIC_LABELS = {"rmse": "RMSE (cm)", "mae": "MAE (cm)", "me": "ME, pred - obs (cm)", "r": "Pearson r",
                 "alpha": "α = σpred / σobs", "b": "B = ME / σobs", "kge2": "KGE''", "skill": "skill vs climatology"}
METRIC_REFERENCE = {"me": 0, "r": 1, "alpha": 1, "b": 0, "kge2": 1, "skill": 0}


def metrics(y, pred, base):
    """All METRICS of pred against y; skill is against the baseline predictions base."""
    return {"rmse": root_mean_squared_error(y, pred), "mae": mean_absolute_error(y, pred), "me": mean_error(y, pred),
            "r": pearson_r(y, pred), "alpha": alpha(y, pred), "b": normalized_bias(y, pred), "kge2": kge2(y, pred),
            "skill": skill_score(y, pred, base)}


def metrics_table(y, preds, baseline, strata=None):
    """METRICS of each model (column of preds) overall and per group of each stratum (labels aligned with y).

    Rows where any prediction is NaN are dropped. Returns one row per model x stratum x group.
    """
    tested = preds.notna().all(axis=1)
    y, preds = y[tested], preds[tested]
    rows = []
    for stratum, labels in {"all": pd.Series("all", index=y.index), **(strata or {})}.items():
        labels = labels[tested]
        for group, idx in labels.groupby(labels, sort=True, observed=True).groups.items():
            for model in preds:
                rows.append({"model": model, "stratum": stratum, "group": group, "n": len(idx),
                             **metrics(y[idx], preds.loc[idx, model], preds.loc[idx, baseline])})
    return pd.DataFrame(rows)


def compare(models, X, y, splits, baseline):
    """Fit a fresh clone of every model on each training set and predict its validation block.

    Returns (by_fold, oof): by_fold has one row per model x fold (dates of the block and METRICS; skill against the
    baseline on the same days); oof has the observed depth and one column of out-of-fold predictions per model.
    """
    oof = pd.DataFrame(np.nan, index=y.index, columns=list(models))
    for name, estimator in models.items():
        for train, test in splits:
            fitted = clone(estimator).fit(X.iloc[train], y.iloc[train])
            oof.iloc[test, oof.columns.get_loc(name)] = fitted.predict(X.iloc[test])
    rows = []
    for name in models:
        for k, (_, test) in enumerate(splits, start=1):
            idx = y.index[test]
            rows.append({"model": name, "fold": k, "test_start": idx.min(), "test_end": idx.max(), "n": len(idx),
                         **metrics(y[idx], oof.loc[idx, name], oof.loc[idx, baseline])})
    return pd.DataFrame(rows).set_index(["model", "fold"]), y.rename("observed").to_frame().join(oof)


def summarize_folds(by_fold):
    """Mean and standard deviation across folds of each metric, one row per model."""
    stats = by_fold[METRICS].groupby(level="model", sort=False).agg(["mean", "std"])
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    return stats


def _bootstrap_blocks(days, block_days, n_boot, seed):
    """Cut days into blocks of block_days consecutive days (inside each run of consecutive dates) and draw n_boot
    resamples of the blocks with replacement. Returns (block of each day, days per block, draws n_boot x n_blocks)."""
    rng = np.random.default_rng(seed)
    run = (pd.Series(days).diff() != pd.Timedelta(days=1)).cumsum().to_numpy()
    within = pd.Series(np.arange(len(days))).groupby(run).cumcount().to_numpy()
    block = pd.factorize(pd.Series(list(zip(run, within // block_days))))[0]
    counts = np.bincount(block)
    return block, counts, rng.integers(0, len(counts), size=(n_boot, len(counts)))


def block_bootstrap(y, preds, baseline, block_days, n_boot, level, seed):
    """Block-bootstrap confidence intervals of the pooled RMSE, ME and skill against the baseline.

    preds has one column of predictions per model. The predicted days are cut into blocks of block_days
    consecutive days (inside each run of consecutive dates) and the blocks are resampled with replacement,
    the same draws for all models so the skill intervals are paired.
    """
    tested = preds.notna().all(axis=1)
    block, counts, draws = _bootstrap_blocks(y.index[tested], block_days, n_boot, seed)
    n_boot_days = counts[draws].sum(1)
    stats = {}
    for model in preds:
        residual = (preds.loc[tested, model] - y[tested]).to_numpy()
        sse, se = np.bincount(block, weights=residual ** 2), np.bincount(block, weights=residual)
        stats[model] = {"rmse": (np.sqrt(sse[draws].sum(1) / n_boot_days), np.sqrt(sse.sum() / counts.sum())),
                        "me": (se[draws].sum(1) / n_boot_days, se.sum() / counts.sum())}
    low, high = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for model, s in stats.items():
        base = stats[baseline]["rmse"]
        s = s | {"skill": (1 - s["rmse"][0] / base[0], 1 - s["rmse"][1] / base[1])}
        row = {"model": model, "n_blocks": len(counts)}
        for name, (boot, point) in s.items():
            row |= {name: point, f"{name}_lo": np.quantile(boot, low), f"{name}_hi": np.quantile(boot, high)}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def rmse_diff_bootstrap(y, preds, reference, block_days, n_boot, level, seed):
    """Block-bootstrap interval of the pooled RMSE of each column of preds and of its difference with the column
    reference (RMSE - RMSE of reference; negative = better than reference).

    The draws are the same for every column (and, with the same seed and days, across calls), so the differences
    are paired. significant is True when the interval of the difference leaves out 0.
    """
    tested = preds.notna().all(axis=1)
    block, counts, draws = _bootstrap_blocks(y.index[tested], block_days, n_boot, seed)
    n_boot_days = counts[draws].sum(1)
    boot, point = {}, {}
    for col in preds:
        sse = np.bincount(block, weights=((preds.loc[tested, col] - y[tested]) ** 2).to_numpy())
        boot[col], point[col] = np.sqrt(sse[draws].sum(1) / n_boot_days), np.sqrt(sse.sum() / counts.sum())
    low, high = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for col in preds:
        diff = boot[col] - boot[reference]
        lo, hi = np.quantile(diff, low), np.quantile(diff, high)
        rows.append({"column": col, "rmse": point[col], "rmse_lo": np.quantile(boot[col], low),
                     "rmse_hi": np.quantile(boot[col], high), "d_rmse": point[col] - point[reference],
                     "d_rmse_lo": lo, "d_rmse_hi": hi, "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows).set_index("column")
