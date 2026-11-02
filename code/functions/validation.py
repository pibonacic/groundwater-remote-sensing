"""Temporal folds, metrics and the evaluation loop."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
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


def mirror_splits(index, splits):
    """The same folds seen from the end of the series (train on the future, test on earlier blocks)."""
    n = len(index)
    return [(np.sort(n - 1 - train), np.sort(n - 1 - test)) for train, test in splits]


def halves_splits(index, gap):
    """Two single-fold diagnostic schemes on the halves of the period.

    halves_fwd trains on the first half and tests on the second; halves_bwd the other way round.
    The gap is dropped from the end of the training half that touches the test half.
    """
    assert_daily_contiguous(index)
    half = len(index) // 2
    first, second = np.arange(half), np.arange(half, len(index))
    return {"halves_fwd": [(first[:half - gap], second)], "halves_bwd": [(second[gap:], first)]}


def make_split_sets(index, gap, n_splits, test_size, min_train):
    """joint (expanding window: the first n_splits folds look forward in time, the last n_splits backward)
    plus the diagnostic halves_fwd and halves_bwd. The forward and backward test blocks must not overlap."""
    expected = min_train + gap + n_splits * test_size
    if len(index) != expected:
        raise ValueError(f"development has {len(index)} rows, expected min_train + gap + n_splits * test_size = {expected}")
    forward = forward_splits(index, n_splits, test_size, gap, min_train)
    joint = forward + mirror_splits(index, forward)
    tests = [set(test) for _, test in joint]
    if sum(map(len, tests)) != len(set().union(*tests)):
        raise ValueError("joint: forward and backward test folds overlap; lower N_SPLITS or TEST_SIZE")
    return {"joint": joint, **halves_splits(index, gap)}


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
                   "nrmse": nrmse(yt, pt, value_range)}
            for name, base in baselines.items():
                row[f"skill_vs_{name}"] = skill_score(yt, pt, base.loc[idx])
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate(estimator, X, y, splits):
    """Fit a fresh clone on each train set and predict its test set.

    Returns the out-of-fold predictions (NaN where never tested) and the metrics per fold.
    """
    oof = pd.Series(np.nan, index=y.index, name="oof")
    rows = []
    for i, (train, test) in enumerate(splits):
        fitted = clone(estimator).fit(X.iloc[train], y.iloc[train])
        pred = fitted.predict(X.iloc[test])
        y_test = y.iloc[test]
        oof.iloc[test] = pred
        rows.append({"fold": i, "n_test": len(test), "rmse": root_mean_squared_error(y_test, pred),
                     "mae": mean_absolute_error(y_test, pred), "bias": bias(y_test, pred)})
    return {"oof": oof, "folds": pd.DataFrame(rows).set_index("fold")}


FOLD_METRICS = ["rmse", "mae", "bias", "nrmse", "skill"]


def fold_direction(scheme, i, n_folds):
    """(direction, fold number from 1) of the i-th fold of a scheme; in joint the first half looks forward."""
    if scheme == "joint":
        half = n_folds // 2
        return ("fwd", i + 1) if i < half else ("bwd", i - half + 1)
    return ("bwd" if scheme == "halves_bwd" else "fwd"), i + 1


def compare(models, X, y, split_sets, baseline, value_range):
    """Evaluate every model on every fold scheme.

    Returns (by_fold, oof): by_fold has one row per model x scheme x fold x metric (FOLD_METRICS; skill is against the
    baseline model on the same test rows) with the direction of the fold; oof[scheme][model] is a Series.
    """
    oof = {scheme: {name: evaluate(estimator, X, y, splits)["oof"] for name, estimator in models.items()}
           for scheme, splits in split_sets.items()}
    rows = []
    for scheme, splits in split_sets.items():
        for i, (_, test) in enumerate(splits):
            direction, fold = fold_direction(scheme, i, len(splits))
            idx = y.index[test]
            base = {baseline: oof[scheme][baseline][idx]}
            for model, pred in oof[scheme].items():
                m = metrics_table(y[idx], pred[idx], value_range, baselines=base).iloc[0]
                m["skill"] = m[f"skill_vs_{baseline}"]
                rows += [{"model": model, "scheme": scheme, "fold": fold, "direction": direction,
                          "metric": metric, "value": m[metric]} for metric in FOLD_METRICS]
    return pd.DataFrame(rows), oof


def summarize_folds(by_fold):
    """Mean and standard deviation across folds of each metric, one row per scheme x model."""
    stats = by_fold.groupby(["scheme", "model", "metric"], sort=False)["value"].agg(["mean", "std"]).unstack("metric")
    return stats[[(stat, metric) for metric in FOLD_METRICS for stat in ("mean", "std")]].set_axis(
        [f"{metric}_{stat}" for metric in FOLD_METRICS for stat in ("mean", "std")], axis=1)


def split_roles(index, joint):
    """Role of each day in the forward and backward folds of joint, and the blocks of each fold.

    Each day gets one role: train_initial (first training set), gap_1 (gap of the first fold) or fold_k (test
    block k). The gaps of later folds fall inside the previous test block, so they keep the fold role.
    Returns (roles with columns fwd and bwd, blocks with start, end and days of train, gap and test per fold).
    """
    half = len(joint) // 2
    roles, rows = pd.DataFrame(index=index), []
    for direction, splits in (("fwd", joint[:half]), ("bwd", joint[half:])):
        role = pd.Series("", index=index)
        role.iloc[splits[0][0]] = "train_initial"
        for k, (train, test) in enumerate(splits, start=1):
            gap = (np.arange(train.max() + 1, test.min()) if train.max() < test.min()
                   else np.arange(test.max() + 1, train.min()))
            if k == 1:
                role.iloc[gap] = "gap_1"
            role.iloc[test] = f"fold_{k}"
            row = {"direction": direction, "fold": k}
            for block, pos in (("train", train), ("gap", gap), ("test", test)):
                row |= {f"{block}_start": index[pos].min(), f"{block}_end": index[pos].max(), f"{block}_days": len(pos)}
            rows.append(row)
        roles[direction] = role
    return roles, pd.DataFrame(rows)


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


def block_bootstrap(y, oof, baseline, block_days, n_boot, level, seed):
    """Block-bootstrap confidence intervals of the pooled out-of-fold RMSE and of the skill against the baseline.

    oof[scheme][model] are out-of-fold predictions. The tested days are cut into blocks of block_days
    consecutive days (inside each run of consecutive tested dates) and the blocks are resampled with
    replacement, the same draws for all models so the skill intervals are paired.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for scheme, by_model in oof.items():
        tested = by_model[baseline].notna()
        days = y.index[tested]
        run = (pd.Series(days).diff() != pd.Timedelta(days=1)).cumsum().to_numpy()
        within = pd.Series(np.arange(len(days))).groupby(run).cumcount().to_numpy()
        block = pd.factorize(pd.Series(list(zip(run, within // block_days))))[0]
        counts = np.bincount(block)
        draws = rng.integers(0, len(counts), size=(n_boot, len(counts)))
        stats = {}
        for model, pred in by_model.items():
            sse = np.bincount(block, weights=((y[tested] - pred[tested]) ** 2).to_numpy())
            stats[model] = (np.sqrt(sse[draws].sum(1) / counts[draws].sum(1)), np.sqrt(sse.sum() / counts.sum()))
        low, high = (1 - level) / 2, 1 - (1 - level) / 2
        for model, (boot, point) in stats.items():
            skill_boot, skill_point = 1 - boot / stats[baseline][0], 1 - point / stats[baseline][1]
            rows.append({"scheme": scheme, "model": model, "n_blocks": len(counts), "rmse": point,
                         "rmse_lo": np.quantile(boot, low), "rmse_hi": np.quantile(boot, high),
                         "skill": skill_point, "skill_lo": np.quantile(skill_boot, low), "skill_hi": np.quantile(skill_boot, high)})
    return pd.DataFrame(rows)
