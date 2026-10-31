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


def monthly_anomaly(s):
    """Subtract the calendar-month mean."""
    return s - s.groupby(s.index.month).transform("mean")


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
    """Days of y above/below threshold and rho of each X column in each regime."""
    above, below = y > threshold, y <= threshold
    rows = {}
    for col in X:
        rho_above, _, n_above = _spearman(X[col][above], y[above], min_pairs)
        rho_below, _, n_below = _spearman(X[col][below], y[below], min_pairs)
        rows[col] = {"n_above": n_above, "n_below": n_below,
                     "rho_above": rho_above, "rho_below": rho_below}
    return pd.DataFrame(rows).T


def cross_correlation(x, y, ks, min_pairs):
    """Spearman between x.shift(-k) and y for each k (x at day t+k against y at day t)."""
    return pd.Series({k: _spearman(x.shift(-k), y, min_pairs)[0] for k in ks}, name=x.name)


def lead_table(spectral, y, ks, min_pairs):
    """CCF of each spectral column against y, raw and as monthly anomalies.

    Returns (summary, ccf_raw, ccf_anom, proposed_leads). The proposal takes the
    best k >= 1 (anomalies) of the two indices with the largest |rho| there; it is
    a starting point for LEADS, not a decision.
    """
    ccf_raw = pd.DataFrame({c: cross_correlation(spectral[c], y, ks, min_pairs) for c in spectral})
    y_anom = monthly_anomaly(y)
    ccf_anom = pd.DataFrame({
        c: cross_correlation(monthly_anomaly(spectral[c]), y_anom, ks, min_pairs) for c in spectral})

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
