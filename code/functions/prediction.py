"""Predictions of the final model over the whole satellite series and their uncertainty.

Two sources of uncertainty, combined by simulation:
- model: an ensemble of the pipeline refit on block-bootstrap resamples of the training days (blocks of consecutive
  days keep the autocorrelation), so its spread says how much the fit depends on which days happened to be observed;
- prediction error: out-of-fold hv_blocks residuals (pred - obs) of the same season as the predicted day, drawn and
  subtracted from each member (obs = pred - residual).
Quantiles over the members give the median and the central intervals. The out-of-fold residuals already carry part
of the model spread, so the intervals lean conservative; their coverage is checked on a holdout (coverage_table).
The daily intervals are marginal: annual aggregates use the spread of the members only (annual_summary), because
residuals drawn day by day would ignore their autocorrelation.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from functions.features import season_of
from functions.models import SeasonalClimatology
from functions.validation import _bootstrap_blocks, compare

CALENDAR_COLUMNS = ("sin_doy", "cos_doy")   # always inside the training range; not checked for extrapolation


def _fit_member(pipeline, X, y, rows, seed, X_new):
    model = clone(pipeline)
    model.set_params(**{k: seed for k in model.get_params() if k.endswith("random_state")})
    return model.fit(X.iloc[rows], y.iloc[rows]).predict(X_new)


def bootstrap_ensemble(pipeline, X, y, X_new, n, block_days, seed, n_jobs=-1):
    """Predictions on X_new (days x n members) of the pipeline refit on n block-bootstrap resamples of (X, y).

    The training days are cut into blocks of block_days consecutive days and the blocks are drawn with replacement
    (_bootstrap_blocks, as the bootstrap of the metrics); member b uses the random state seed + b.
    """
    block, counts, draws = _bootstrap_blocks(X.index, block_days, n, seed)
    rows_of = [np.flatnonzero(block == k) for k in range(len(counts))]
    preds = Parallel(n_jobs=n_jobs)(
        delayed(_fit_member)(pipeline, X, y, np.concatenate([rows_of[k] for k in draw]), seed + b, X_new)
        for b, draw in enumerate(draws))
    return pd.DataFrame(np.column_stack(preds), index=X_new.index)


def oof_residuals(pipeline, X, y, splits):
    """Out-of-fold residuals (pred - obs) of the pipeline on the hv_blocks folds, with the season of each day."""
    _, oof = compare({"climatology": SeasonalClimatology(), "model": pipeline}, X, y, splits, "climatology")
    return pd.DataFrame({"residual": oof["model"] - oof["observed"], "season": season_of(y.index)})


def _level_names(level):
    pct = round(level * 100)
    return f"lo{pct}", f"hi{pct}"


def draw_residuals(members, residuals, seed):
    """One out-of-fold residual per day and member (same shape as members), drawn with replacement from the
    residuals of the season of the day (all residuals if the season has none). residuals is the table of
    oof_residuals."""
    rng = np.random.default_rng(seed)
    draws = np.zeros(members.shape)
    season = season_of(members.index).to_numpy()
    for s in pd.unique(season):
        rows = season == s
        pool = residuals.loc[residuals["season"] == s, "residual"].to_numpy()
        pool = pool if len(pool) else residuals["residual"].to_numpy()
        draws[rows] = rng.choice(pool, size=(rows.sum(), members.shape[1]))
    return pd.DataFrame(draws, index=members.index, columns=members.columns)


def simulate(members, draws, limits):
    """Simulated depth of each day and member: the member minus its drawn residual (obs = pred - residual),
    clipped to limits."""
    return (members - draws).clip(*limits)


def predictive_quantiles(sim, levels):
    """Median and central intervals (lo{pct} / hi{pct} for each level) of each day over the simulated members."""
    values = sim.to_numpy()
    table = {"median": np.median(values, axis=1)}
    for level in levels:
        lo, hi = _level_names(level)
        table[lo] = np.quantile(values, (1 - level) / 2, axis=1)
        table[hi] = np.quantile(values, 1 - (1 - level) / 2, axis=1)
    return pd.DataFrame(table, index=sim.index)


def residual_ranges(residuals):
    """Min, median and max of the out-of-fold residuals of each season (the pools draw_residuals draws from)."""
    return residuals.groupby("season", observed=True)["residual"].agg(["count", "min", "median", "max"]).rename(
        columns={"count": "n"})


def daily_ranges(members, draws):
    """For each day, the min, median and max over the members of their prediction and of their drawn residual."""
    stats = {"min": "min", "median": "median", "max": "max"}
    return pd.DataFrame({**{f"member_{k}": getattr(members, f)(axis=1) for k, f in stats.items()},
                         **{f"residual_{k}": getattr(draws, f)(axis=1) for k, f in stats.items()}})


def outside_training_range(X_train, X_new, columns):
    """Number of columns (calendar columns aside) of each day of X_new outside the [min, max] of X_train."""
    columns = [c for c in columns if c not in CALENDAR_COLUMNS]
    lo, hi = X_train[columns].min(), X_train[columns].max()
    return ((X_new[columns] < lo) | (X_new[columns] > hi)).sum(axis=1).rename("n_outside")


def coverage_table(y, quantiles, levels, strata):
    """Empirical coverage, mean width and interval score of each interval on the observed days y.

    strata maps a name to a boolean mask over y.index (e.g. all / extrapolates / inside). The interval score is the
    width plus 2 / (1 - level) times the distance to the interval when the observation falls outside (lower = better).
    """
    rows = []
    for name, mask in strata.items():
        obs, q = y[mask], quantiles.loc[y.index[mask]]
        for level in levels:
            lo, hi = (q[c] for c in _level_names(level))
            alpha = 1 - level
            score = (hi - lo) + 2 / alpha * ((lo - obs).clip(lower=0) + (obs - hi).clip(lower=0))
            rows.append({"stratum": name, "level": level, "n": int(mask.sum()),
                         "coverage": ((obs >= lo) & (obs <= hi)).mean() if mask.any() else np.nan,
                         "mean_width": (hi - lo).mean(), "interval_score": score.mean()})
    return pd.DataFrame(rows)


def annual_summary(pred, members, n_outside, threshold, level):
    """One row per year of the point prediction pred: days predicted, mean / min / max depth, date of the
    shallowest level, days shallower than threshold (days_shallower: depth < threshold) and share of days that extrapolate. Each
    statistic has an interval over the members of the ensemble (model uncertainty only, see the module docstring)."""
    low, high = (1 - level) / 2, 1 - (1 - level) / 2
    stats = {"mean": lambda d: d.mean(), "min": lambda d: d.min(), "max": lambda d: d.max(),
             "days_shallower": lambda d: (d < threshold).sum()}
    rows = []
    for year, days in pred.groupby(pred.index.year).groups.items():
        p, m = pred[days], members.loc[days]
        row = {"year": year, "n_days": len(p), "date_shallowest": p.idxmin().date(),
               "pct_extrapolates": 100 * (n_outside[days] > 0).mean()}
        for name, f in stats.items():
            boot = f(m)
            row |= {name: f(p), f"{name}_lo": np.quantile(boot, low), f"{name}_hi": np.quantile(boot, high)}
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")
