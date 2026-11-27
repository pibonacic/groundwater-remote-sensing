"""Exploration of the relation between the remote variables and the water level. All functions return tables.

Convention: y is a depth (cm, positive below the surface), so a rise of the water level is a decrease of y.
Correlations are Spearman; daily series are autocorrelated, so p-values are only indicative and the intervals are
block-bootstrap intervals (blocks of consecutive days inside each run of consecutive dates).
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr

from functions.features import SEASON_BY_MONTH

SEASONS = ["DJF", "MAM", "JJA", "SON"]
REGIMES = ["ponded", "shallow", "deep"]


# ---------------------------------------------------------------- calendar labels and regimes
def season_year(index):
    """Year of the season of each date: December belongs to the summer (DJF) of the next year."""
    return pd.Series(index.year + (index.month == 12), index=index)


def season_label(index):
    """Season of each date (DJF = austral summer)."""
    return pd.Series(index.month.map(SEASON_BY_MONTH), index=index)


def hydro_year(index, start_month):
    """Hydrological year of each date, labeled by the calendar year in which it starts."""
    return pd.Series(index.year - (index.month < start_month), index=index)


def regime_of(y, bins):
    """Regime of each day by the raw depth: ponded (< bins[0]), shallow (bins[0] to bins[1]), deep (> bins[1])."""
    low, high = bins
    labels = np.where(y < low, "ponded", np.where(y <= high, "shallow", "deep"))
    return pd.Series(labels, index=y.index).where(y.notna())


# ---------------------------------------------------------------- harmonic anomalies
def harmonic_design(index, K):
    """Constant plus sin / cos of k theta (k = 1..K), theta = 2 pi (day of year - 1) / days in the year."""
    days = np.where(index.is_leap_year, 366.0, 365.0)
    theta = 2 * np.pi * (index.dayofyear.to_numpy() - 1) / days
    return np.column_stack([np.ones(len(index))] + [f(k * theta) for k in range(1, K + 1) for f in (np.sin, np.cos)])


def harmonic_curve(frame, K, fit_days):
    """Harmonic fit of order K of each column of frame (DataFrame), fitted by least squares on fit_days and
    evaluated on every day of frame. fit_days must have no missing values in frame."""
    fit = frame.loc[fit_days]
    assert fit.notna().all().all(), "harmonic_curve: missing values on fit_days"
    coef, *_ = np.linalg.lstsq(harmonic_design(fit.index, K), fit.to_numpy(float), rcond=None)
    return pd.DataFrame(harmonic_design(frame.index, K) @ coef, index=frame.index, columns=frame.columns)


def harmonic_anomaly(frame, K, fit_days):
    """frame (Series or DataFrame) minus its harmonic fit of order K (harmonic_curve), on every day of frame."""
    if isinstance(frame, pd.Series):
        return frame - harmonic_curve(frame.to_frame(), K, fit_days).iloc[:, 0]
    return frame - harmonic_curve(frame, K, fit_days)


def detrended_anomaly(s, K):
    """Residual of a joint least-squares fit of a constant, a linear trend and the harmonics of order K over every
    valid day of s (the whole satellite record). Returns (anomaly, trend per year)."""
    valid = s.dropna()
    origin = valid.index[0]
    years = lambda index: ((index - origin).days / 365.25).to_numpy()
    coef, *_ = np.linalg.lstsq(np.column_stack([harmonic_design(valid.index, K), years(valid.index)]),
                               valid.to_numpy(float), rcond=None)
    fitted = np.column_stack([harmonic_design(s.index, K), years(s.index)]) @ coef
    return s - fitted, coef[-1]


# ---------------------------------------------------------------- block bootstrap
def block_draws(days, block_days, n_boot, seed):
    """n_boot resamples (positions 0..n-1) of the days: the days are cut into blocks of block_days consecutive days
    inside each run of consecutive dates, blocks are drawn with replacement and concatenated until n days, and the
    last one is cut so every resample has n days. Returns an n_boot x n array of positions."""
    days = pd.DatetimeIndex(days)
    n = len(days)
    run = (days.to_series().diff() != pd.Timedelta(days=1)).cumsum().to_numpy()
    within = pd.Series(np.arange(n)).groupby(run).cumcount().to_numpy()
    block = pd.factorize(pd.Series(list(zip(run, within // block_days))))[0]
    blocks = [np.flatnonzero(block == b) for b in range(block.max() + 1)]
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n), dtype=int)
    for i in range(n_boot):
        parts, size = [], 0
        while size < n:
            for b in rng.integers(0, len(blocks), size=len(blocks)):
                parts.append(blocks[b])
                size += len(blocks[b])
                if size >= n:
                    break
        out[i] = np.concatenate(parts)[:n]
    return out


def _rowwise_spearman(xm, ym):
    rx, ry = rankdata(xm, axis=1), rankdata(ym, axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(axis=1) / np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))


def spearman_ci(x, y, min_pairs, block_days, n_boot, level, seed):
    """Spearman rho of x and y on their common days with a block-bootstrap interval. Returns a dict with rho, n,
    rho_lo, rho_hi and significant (the interval leaves out 0), plus the bootstrap draws under _boot."""
    pair = pd.concat([x, y], axis=1).dropna()
    n = len(pair)
    if n < min_pairs:
        return {"rho": np.nan, "n": n, "rho_lo": np.nan, "rho_hi": np.nan, "significant": False, "_boot": None}
    xv, yv = pair.iloc[:, 0].to_numpy(float), pair.iloc[:, 1].to_numpy(float)
    rho = spearmanr(xv, yv)[0]
    idx = block_draws(pair.index, block_days, n_boot, seed)
    boot = _rowwise_spearman(xv[idx], yv[idx])
    lo, hi = np.nanquantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"rho": rho, "n": n, "rho_lo": lo, "rho_hi": hi, "significant": bool(lo > 0 or hi < 0), "_boot": boot}


def stratified_spearman(X, y, strata, min_pairs, block_days, n_boot, level, seed):
    """Spearman of each column of X with y inside each stratum, with block-bootstrap intervals.

    strata is a DataFrame of label columns on the index of y (for example season, or regime and year); one row per
    variable and combination of labels, with rho, n, n_years (distinct season years with data) and the interval.
    Strata with fewer than min_pairs days get rho NaN.
    """
    strata = strata.to_frame() if isinstance(strata, pd.Series) else strata
    keys = list(strata.columns)
    years = season_year(y.index)
    rows = []
    for labels, group in strata.dropna().groupby(keys, sort=False):
        labels = labels if isinstance(labels, tuple) else (labels,)
        days = group.index
        for col in X:
            x, yy = X[col].reindex(days), y.reindex(days)
            res = spearman_ci(x, yy, min_pairs, block_days, n_boot, level, seed)
            res.pop("_boot")
            n_years = years.reindex(days[x.notna() & yy.notna()]).nunique()
            rows.append(dict(zip(keys, labels)) | {"variable": col, "n_years": n_years} | res)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- signs, regimes, leads, rain, redundancy, memory
def _spearman(a, b, min_pairs):
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < min_pairs:
        return np.nan, np.nan, len(pair)
    rho, p = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return rho, p, len(pair)


def spearman_signs(X, y, min_pairs):
    """Spearman rho, p-value, pairs and sign of each column of X against y.

    Daily series are autocorrelated, so p-values are only indicative.
    """
    rows = {}
    for col in X:
        rho, p, n = _spearman(X[col], y, min_pairs)
        rows[col] = {"rho": rho, "p": p, "n": n, "sign": np.sign(rho)}
    return pd.DataFrame(rows).T


def regime_check(X, y, threshold, min_pairs, depth=None):
    """Days below/above the surface and rho of each X column with y in each regime.

    The regime is set by depth (positive below the surface; depth > threshold is below the surface), which is y
    itself unless given: the anomaly version correlates the anomalies inside the regimes of the raw depth.
    """
    depth = y if depth is None else depth
    below, above = depth > threshold, depth <= threshold
    rows = {}
    for col in X:
        rho_below, _, n_below = _spearman(X[col][below], y[below], min_pairs)
        rho_above, _, n_above = _spearman(X[col][above], y[above], min_pairs)
        rows[col] = {"n_below_surface": n_below, "n_above_surface": n_above,
                     "rho_below_surface": rho_below, "rho_above_surface": rho_above}
    return pd.DataFrame(rows).T


def cross_correlation(x, y, ks, min_pairs):
    """Spearman between x.shift(-k) and y for each k (x at day t+k against y at day t)."""
    return pd.Series({k: _spearman(x.shift(-k), y, min_pairs)[0] for k in ks}, name=x.name)


def ccf_table(X, y, ks, min_pairs):
    """CCF (cross_correlation) of each column of X against y: one column per variable, one row per k."""
    return pd.DataFrame({c: cross_correlation(X[c], y, ks, min_pairs) for c in X})


def lead_summary(ccf):
    """Per variable: k of the largest |rho| (k_max, rho_max), rho at k = 0, gain of |rho| over k = 0, best k >= 1
    and whether k_max is on the edge of the k range (at_edge: the true maximum may lie outside)."""
    edges = (ccf.index.min(), ccf.index.max())
    rows = []
    for col in ccf:
        s = ccf[col]
        if s.isna().all():
            continue
        positive = s[s.index >= 1]
        k_max = s.abs().idxmax()
        rows.append({"variable": col, "k_max": k_max, "rho_max": s[k_max], "rho_0": s[0],
                     "gain": s.abs().max() - abs(s[0]), "at_edge": k_max in edges,
                     "k_best_pos": positive.abs().idxmax(), "rho_best_pos": positive[positive.abs().idxmax()]})
    return pd.DataFrame(rows).set_index("variable")


def proposed_leads(summary, columns, n=2):
    """Best k >= 1 of the n variables (among columns) with the largest |rho| there: a starting point for LEADS."""
    top = summary.loc[[c for c in columns if c in summary.index]]
    top = top.reindex(top["rho_best_pos"].abs().sort_values(ascending=False).index).head(n)
    return sorted(int(k) for k in top["k_best_pos"].unique())


def rain_window_response(precip, moisture, windows, min_pairs):
    """Spearman between w-day accumulated rain and a moisture series (raw or anomaly), for each w."""
    rows = []
    for w in windows:
        rho, p, n = _spearman(precip.rolling(w, min_periods=w).sum(), moisture, min_pairs)
        rows.append({"window": w, "rho": rho, "p": p, "n": n})
    return pd.DataFrame(rows).set_index("window")


def redundancy_clusters(X, threshold):
    """Group the columns of X that move together; the water level is not used.

    Average-linkage clusters of distance 1 - |Spearman|, cut at 1 - threshold. Returns
    (linkage matrix, table). The table has the cluster of each column and `proposed`, the
    member with the highest mean |rho| to its cluster (a neutral starting point: which one
    becomes a feature is the user's choice).
    """
    rho = X.corr(method="spearman").abs().fillna(0)
    distance = 1 - rho.to_numpy()
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance.clip(0), checks=False), "average")
    clusters = pd.Series(fcluster(tree, t=1 - threshold, criterion="distance"), index=X.columns, name="cluster")
    mean_rho = pd.Series({c: rho.loc[c, clusters.index[clusters == clusters[c]]].mean() for c in X}, name="mean_rho_in_cluster")
    table = pd.concat([clusters, mean_rho], axis=1)
    table["proposed"] = table.index.isin(mean_rho.groupby(clusters).idxmax())   # first one if tied
    return tree, table


def acf_pacf(y, max_lag):
    """Autocorrelation (ACF) and partial autocorrelation (PACF, Durbin-Levinson) of y for lags 0..max_lag.

    y must be a series without gaps (NaN are dropped at the ends only). Returns a table indexed by lag with acf,
    pacf and conf, the half-width of the approximate 95% band of no correlation (1.96 / sqrt(n)).
    """
    x = y.loc[y.first_valid_index():y.last_valid_index()]
    if x.isna().any():
        raise ValueError("acf_pacf needs a series without gaps")
    x = x.to_numpy(float) - x.mean()
    n = len(x)
    acf = np.array([np.sum(x[:n - k] * x[k:]) for k in range(max_lag + 1)]) / np.sum(x * x)
    pacf, phi = np.ones(max_lag + 1), np.zeros(0)
    for k in range(1, max_lag + 1):
        a = (acf[k] - phi @ acf[k - 1:0:-1]) / (1 - phi @ acf[1:k])
        phi = np.r_[phi - a * phi[::-1], a]
        pacf[k] = a
    return pd.DataFrame({"acf": acf, "pacf": pacf, "conf": 1.96 / np.sqrt(n)}, index=pd.Index(range(max_lag + 1), name="lag"))


# ---------------------------------------------------------------- rise / fall of the level
def level_phase(y, window, min_rate):
    """Phase of the level on each day: "rise" when the centered rolling mean (window days) of the depth decreases
    faster than min_rate cm/day (the water table goes up), "fall" when it increases faster than min_rate (recession),
    NaN otherwise or where the window is incomplete. Also returns the rate (cm/day, centered difference)."""
    smooth = y.rolling(window, center=True, min_periods=window).mean()
    rate = (smooth.shift(-1) - smooth.shift(1)) / 2
    phase = pd.Series(np.where(rate < -min_rate, "rise", np.where(rate > min_rate, "fall", None)), index=y.index)
    return phase.where(rate.notna()), rate


def _theil_sen(x, y):
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = dx != 0
    return np.median((y[j] - y[i])[ok] / dx[ok]) if ok.any() else np.nan


def phase_relation(X, y, phase, min_pairs, block_days, n_boot, n_boot_slope, level, seed):
    """Per variable and phase (rise / fall): Theil-Sen slope of the variable against the depth (units of the
    variable per cm) with its block-bootstrap interval (n_boot_slope draws), the same slope in standard deviations
    of the variable per standard deviation of the depth (whole series), and the Spearman rho with its interval.
    Also the difference fall - rise of rho with its interval (rise and fall days are disjoint, so their draws are
    independent)."""
    low, high = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for col in X:
        boots = {}
        for name in ("rise", "fall"):
            days = phase.index[phase == name]
            pair = pd.concat([X[col].reindex(days), y.reindex(days)], axis=1).dropna()
            res = spearman_ci(pair.iloc[:, 0], pair.iloc[:, 1], min_pairs, block_days, n_boot, level, seed)
            boots[name] = res.pop("_boot")
            row = {"variable": col, "phase": name} | res
            if len(pair) >= min_pairs:
                depth, value = pair.iloc[:, 1].to_numpy(float), pair.iloc[:, 0].to_numpy(float)
                idx = block_draws(pair.index, block_days, n_boot_slope, seed)
                slope_boot = np.array([_theil_sen(depth[i], value[i]) for i in idx])
                slope = _theil_sen(depth, value)
                row |= {"slope": slope, "slope_lo": np.nanquantile(slope_boot, low),
                        "slope_hi": np.nanquantile(slope_boot, high), "slope_std": slope * y.std() / X[col].std()}
            rows.append(row)
        if boots["rise"] is not None and boots["fall"] is not None:
            diff = boots["fall"] - boots["rise"]
            lo, hi = np.nanquantile(diff, [low, high])
            rows.append({"variable": col, "phase": "fall - rise", "rho": rows[-1]["rho"] - rows[-2]["rho"],
                         "rho_lo": lo, "rho_hi": hi, "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stability of the harmonic curve
def harmonic_stability(frame, period, comparisons, K, block_days, n_boot, level, seed):
    """Does the harmonic curve of order K of each column of frame change between periods?

    frame has no missing values; period labels each of its days. comparisons maps a name to (A, B), two lists of
    period labels: the curve fitted on the days of B minus the curve fitted on the days of A, per day of the year
    (a common year). Intervals: block bootstrap inside each period label separately (each period keeps its size;
    when B contains A, the same draw of A enters both fits). Returns (summary, curves): summary has per comparison
    and variable the mean of each curve, the shift (mean difference) with its interval, the largest |difference|
    and the share of days of the year whose interval leaves out 0; curves maps (comparison, variable) to a table
    by day of the year with curve_a, curve_b, diff, diff_lo, diff_hi."""
    year = pd.date_range("2001-01-01", "2001-12-31")
    design_year = harmonic_design(year, K)
    design = harmonic_design(frame.index, K)
    values = frame.to_numpy(float)
    labels = period.to_numpy()
    groups = {lab: np.flatnonzero(labels == lab) for lab in pd.unique(labels)}
    draws = {lab: pos[block_draws(frame.index[pos], block_days, n_boot, seed + i)]
             for i, (lab, pos) in enumerate(groups.items())}
    low, high = (1 - level) / 2, 1 - (1 - level) / 2

    def fit(positions):
        coef, *_ = np.linalg.lstsq(design[positions], values[positions], rcond=None)
        return design_year @ coef   # 365 x variables

    rows, curves = [], {}
    for name, (a, b) in comparisons.items():
        pos_a = np.concatenate([groups[lab] for lab in a])
        pos_b = np.concatenate([groups[lab] for lab in b])
        curve_a, curve_b = fit(pos_a), fit(pos_b)
        diffs = np.empty((n_boot, len(year), frame.shape[1]))
        for i in range(n_boot):
            diffs[i] = fit(np.concatenate([draws[lab][i] for lab in b])) - fit(np.concatenate([draws[lab][i] for lab in a]))
        lo, hi = np.quantile(diffs, [low, high], axis=0)
        shift_boot = diffs.mean(axis=1)
        for j, col in enumerate(frame.columns):
            diff = curve_b[:, j] - curve_a[:, j]
            rows.append({"comparison": name, "variable": col, "sd": frame[col].std(), "mean_a": curve_a[:, j].mean(),
                         "mean_b": curve_b[:, j].mean(), "shift": diff.mean(),
                         "shift_lo": np.quantile(shift_boot[:, j], low), "shift_hi": np.quantile(shift_boot[:, j], high),
                         "shift_in_sd": diff.mean() / frame[col].std(), "max_abs_diff": np.abs(diff).max(),
                         "amplitude_a": np.ptp(curve_a[:, j]), "amplitude_b": np.ptp(curve_b[:, j]),
                         "frac_days_significant": float(((lo[:, j] > 0) | (hi[:, j] < 0)).mean()),
                         "n_a": len(pos_a), "n_b": len(pos_b)})
            curves[(name, col)] = pd.DataFrame({"curve_a": curve_a[:, j], "curve_b": curve_b[:, j], "diff": diff,
                                                "diff_lo": lo[:, j], "diff_hi": hi[:, j]},
                                               index=pd.Index(range(1, 366), name="doy"))
    return pd.DataFrame(rows), curves


# ---------------------------------------------------------------- slow and fast parts
def slow_fast(frame, window):
    """Slow part = centered rolling median of window days (at least half the window present); fast part = the rest."""
    slow = frame.rolling(window, center=True, min_periods=window // 2 + 1).median()
    return slow, frame - slow


def slow_fast_table(y, X, window, min_pairs):
    """Variance of the series, of its slow and of its fast part (and their share of the total variance) for y and
    each column of X, and the Spearman of y with each column for the whole series and for each part.
    Returns (table, slow, fast)."""
    frame = pd.concat([y, X], axis=1)
    slow, fast = slow_fast(frame, window)
    rows = []
    for col in frame:
        total = frame[col].var()
        row = {"variable": col, "var_total": total, "var_slow": slow[col].var(), "var_fast": fast[col].var(),
               "frac_slow": slow[col].var() / total, "frac_fast": fast[col].var() / total}
        if col != y.name:
            row |= {"rho_total": _spearman(frame[col], frame[y.name], min_pairs)[0],
                    "rho_slow": _spearman(slow[col], slow[y.name], min_pairs)[0],
                    "rho_fast": _spearman(fast[col], fast[y.name], min_pairs)[0]}
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable"), slow, fast


# ---------------------------------------------------------------- jumps and rise events of the piezometer
def _rain_sum(precip, start, end):
    """Rain from start to end (both included); NaN when any day is missing (CHIRPS ends before the piezometer)."""
    window = precip.reindex(pd.date_range(start, end))
    return window.sum() if window.notna().all() else np.nan


def jump_table(y, precip, max_jump, recovery_days, recovery_fraction):
    """Day-to-day changes of the depth larger than max_jump (cm). direction: "rise" (depth decreases: the water
    table goes up) or "fall". Rain of the day and of the 3 and 7 previous days (without the day). recovered: within
    recovery_days days after the jump the depth comes back more than recovery_fraction of the jump."""
    one = pd.Timedelta(days=1)
    delta = y.diff()
    rows = []
    for day, d in delta[delta.abs() > max_jump].items():
        after = y.loc[day + one:day + recovery_days * one]
        back = (after - y[day]) * -np.sign(d)   # positive = moving back towards the level before the jump
        hit = back[back > recovery_fraction * abs(d)]
        rows.append({"date": day, "jump_cm": d, "abs_jump_cm": abs(d), "direction": "rise" if d < 0 else "fall",
                     "depth_before_cm": y[day - one], "depth_after_cm": y[day],
                     "rain_day_mm": _rain_sum(precip, day, day),
                     "rain_prev3_mm": _rain_sum(precip, day - 3 * one, day - one),
                     "rain_prev7_mm": _rain_sum(precip, day - 7 * one, day - one),
                     "recovered": len(hit) > 0, "days_to_recover": (hit.index[0] - day).days if len(hit) else np.nan,
                     "max_return_fraction": back.max() / abs(d) if len(back) else np.nan})
    return pd.DataFrame(rows)


def rise_events(y, precip, min_rise, min_rain, lead_days):
    """Rise events: runs of consecutive days whose depth decreases, with a total rise >= min_rise (cm) and rain
    >= min_rain (mm) from lead_days before the first day to the last day. Apparent specific yield Sy = rain (mm) /
    rise (mm). ponded: the event ends above the surface (Sy is then not that of the soil)."""
    one = pd.Timedelta(days=1)
    rising = y.diff() < 0
    run = (rising != rising.shift()).cumsum()
    rows = []
    for _, days in rising[rising].groupby(run[rising]):
        start, end = days.index[0], days.index[-1]
        rise = y[start - one] - y[end]
        rain = _rain_sum(precip, start - lead_days * one, end)
        if rise >= min_rise and rain >= min_rain:
            rows.append({"start": start, "end": end, "days": len(days), "depth_start_cm": y[start - one],
                         "depth_end_cm": y[end], "rise_cm": rise, "rain_mm": rain, "sy_apparent": rain / (10 * rise),
                         "ponded": bool(y[end] < 0)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- September and annual anomalies
def by_year_spearman(X, y, years, min_pairs):
    """Spearman of each column of X with y inside each year label (no interval: one month a year is ~1 block)."""
    rows = []
    for year, days in y.groupby(years).groups.items():
        for col in X:
            rho, _, n = _spearman(X[col].reindex(days), y.reindex(days), min_pairs)
            rows.append({"year": year, "variable": col, "rho": rho, "n": n})
    return pd.DataFrame(rows)


def period_means(s, period, block_days, n_boot, level, seed):
    """Mean of s in each period label with a block-bootstrap interval of the mean (blocks inside the period)."""
    low, high = (1 - level) / 2, 1 - (1 - level) / 2
    valid = s.dropna()
    rows = []
    for label, values in valid.groupby(period.reindex(valid.index)):
        boot = values.to_numpy()[block_draws(values.index, block_days, n_boot, seed)].mean(axis=1)
        rows.append({"period": label, "n": len(values), "mean": values.mean(),
                     "mean_lo": np.quantile(boot, low), "mean_hi": np.quantile(boot, high)})
    return pd.DataFrame(rows)
