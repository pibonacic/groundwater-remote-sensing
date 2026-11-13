"""Signs, regimes, leads and rain windows. All functions return tables."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def _spearman(a, b, min_pairs):
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < min_pairs:
        return np.nan, np.nan, len(pair)
    rho, p = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return rho, p, len(pair)


def seasonal_anomaly(s, half_window):
    """Subtract the mean of the same time of the year: all the days, in every year, within half_window days
    before or after the day of the year (the window wraps around the end of the year; Dec 31 of a leap year
    counts as day 365). s is a Series or DataFrame with a daily DatetimeIndex."""
    doy = pd.Series(np.minimum(s.index.dayofyear, 365), index=s.index)

    def window_sum(x):
        x = x.reindex(range(1, 366), fill_value=0)
        wrapped = pd.concat([x.iloc[-half_window:], x, x.iloc[:half_window]])
        return wrapped.rolling(2 * half_window + 1).sum().iloc[2 * half_window:].set_axis(x.index)

    clim = window_sum(s.groupby(doy).sum()) / window_sum(s.groupby(doy).count())
    return s - clim.reindex(doy.to_numpy()).set_axis(s.index)


def spearman_signs(X, y, min_pairs):
    """Spearman rho, p-value, pairs and sign of each column of X against y.

    Daily series are autocorrelated, so p-values are only indicative.
    """
    rows = {}
    for col in X:
        rho, p, n = _spearman(X[col], y, min_pairs)
        rows[col] = {"rho": rho, "p": p, "n": n, "sign": np.sign(rho)}
    return pd.DataFrame(rows).T


def regime_check(X, y, threshold, min_pairs):
    """Days below/above the surface and rho of each X column in each regime.

    y is a depth (positive below the surface): depth > threshold is below the surface.
    """
    below, above = y > threshold, y <= threshold
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


def lead_table(spectral, y, ks, min_pairs, half_window):
    """CCF of each spectral column against y, raw and as seasonal anomalies (see seasonal_anomaly).

    Returns (summary, ccf_raw, ccf_anom, proposed_leads). The proposal takes the
    best k >= 1 (anomalies) of the two indices with the largest |rho| there; it is
    a starting point for LEADS, not a decision.
    """
    ccf_raw = pd.DataFrame({c: cross_correlation(spectral[c], y, ks, min_pairs) for c in spectral})
    y_anom = seasonal_anomaly(y, half_window)
    ccf_anom = pd.DataFrame({
        c: cross_correlation(seasonal_anomaly(spectral[c], half_window), y_anom, ks, min_pairs) for c in spectral})

    rows = []
    for kind, ccf in (("raw", ccf_raw), ("anomaly", ccf_anom)):
        for col in ccf:
            s = ccf[col]
            positive = s[s.index >= 1]
            rows.append({
                "index": col, "series": kind,
                "k_max": s.abs().idxmax(), "rho_max": s[s.abs().idxmax()], "rho_0": s[0],
                "gain": s.abs().max() - abs(s[0]),
                "k_best_pos": positive.abs().idxmax(), "rho_best_pos": positive[positive.abs().idxmax()],
            })
    summary = pd.DataFrame(rows)

    anomaly = summary[summary["series"] == "anomaly"]
    top = anomaly.reindex(anomaly["rho_best_pos"].abs().sort_values(ascending=False).index).head(2)
    proposed = sorted(int(k) for k in top["k_best_pos"].unique())
    return summary, ccf_raw, ccf_anom, proposed


def rain_window_response(precip, moisture_anom, windows, min_pairs):
    """Spearman between w-day accumulated rain and a moisture anomaly, for each w."""
    rows = []
    for w in windows:
        rho, p, n = _spearman(precip.rolling(w, min_periods=w).sum(), moisture_anom, min_pairs)
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
