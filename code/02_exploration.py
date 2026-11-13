"""Steps 2-3: signs, regimes, leads, rain windows and memory of the target on the development period."""
import pandas as pd

import config as cfg
from functions.data import load_table, save_table, compute_indices
from functions.exploration import (
    seasonal_anomaly, spearman_signs, regime_check, lead_table, rain_window_response, redundancy_clusters, acf_pacf)
from functions.features import compute_partition
from functions.plots import (
    scatter_by_season, zscore_series_plot, ccf_plot, dendrogram_plot, rain_window_plot, acf_pacf_plot)

SITE = "sdh1"

out_dir = cfg.OUTPUTS_DIR / SITE / "02_exploration"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the clean table and keep only the development period (same cut as step 3,
#    without the separation days, so the holdout is never looked at).
daily = load_table(cfg.INTERIM_DIR / f"{SITE}_daily.csv")
partition = compute_partition(daily["gw_cm"].dropna().index, cfg.DEV_YEARS, 0)
daily = daily.loc[:partition.dev_end]
y = daily["gw_cm"]

# 2. Spectral indices from the NBAR bands; X = indices + the I bands (the M bands are left out of the exploration).
indices = compute_indices(daily, cfg.BANDS)
bands = daily.drop(columns=["gw_cm", "qa_low", "precip_mm"])
bands = bands.drop(columns=[b for b in bands if b.startswith("M")])
X = indices.join(bands)

# 3. Sign of the relation with the water level (Spearman) and whether it differs
#    above/below the surface (two regimes).
signs = spearman_signs(X, y, cfg.MIN_PAIRS)
regimes = regime_check(X, y, cfg.REGIME_THRESHOLD_CM, cfg.MIN_PAIRS)

# 4. Lead time: correlation between the water level and the index shifted by k days,
#    on the raw series and on seasonal anomalies (seasonality removed). Proposes LEADS.
leads, ccf_raw, ccf_anom, proposed_leads = lead_table(indices, y, cfg.CCF_KS, cfg.MIN_PAIRS, cfg.ANOMALY_HALF_WINDOW)

# 5. Rain accumulation windows that best track the moisture anomaly. Narrows WINDOWS.
rain = rain_window_response(
    daily["precip_mm"], seasonal_anomaly(indices[cfg.RAIN_MOISTURE_INDEX], cfg.ANOMALY_HALF_WINDOW),
    cfg.RAIN_WINDOWS, cfg.MIN_PAIRS)

# 6. Redundancy: bands and indices that move together (the water level is not used). Which one of
#    each cluster becomes a feature is your choice (SPECTRAL in config.py); `proposed` is only a start.
#    Computed on the days of the development period that have a water level.
tree, clusters = redundancy_clusters(X.loc[y.dropna().index], cfg.REDUNDANCY_THRESHOLD)

# 7. Memory of the target: ACF and PACF of the depth, raw and as seasonal anomaly, to choose the gap h of the
#    hv_blocks folds (HV_H): the lag where the autocorrelation becomes negligible.
memory = {"raw": acf_pacf(y, cfg.ACF_MAX_LAG), "anomaly": acf_pacf(seasonal_anomaly(y, cfg.ANOMALY_HALF_WINDOW), cfg.ACF_MAX_LAG)}
acf_table = pd.concat(memory, axis=1)
acf_table.columns = [f"{a}_{b}" for a, b in acf_table.columns]

# 8. Save every table that supports a decision, plus the figures.
for name, table in {"signs": signs, "regimes": regimes, "leads": leads,
                    "ccf_raw": ccf_raw, "ccf_anomaly": ccf_anom, "rain_windows": rain, "clusters": clusters, "acf_pacf": acf_table}.items():
    save_table(table, out_dir / f"{name}.csv")
scatter_by_season(indices.loc[y.dropna().index], y.dropna(), f"{SITE}: indices vs water level").savefig(
    out_dir / "scatter_by_season.png", dpi=150)
dev = slice(y.first_valid_index(), partition.dev_end)
zscore_series_plot(X.loc[dev], y.loc[dev], f"{SITE}: indices and bands vs water level (z-scores, development period)").savefig(
    out_dir / "zscore_series.png", dpi=150)
# Seasonal anomalies (mean of the same time of the year removed, as in the CCF), z-scored, over the development period.
zscore_series_plot(seasonal_anomaly(X, cfg.ANOMALY_HALF_WINDOW).loc[dev], seasonal_anomaly(y, cfg.ANOMALY_HALF_WINDOW).loc[dev],
                   f"{SITE}: seasonal anomalies of indices and bands vs water level (z-scores, development period)",
                   suffix=" anomaly (z)").savefig(out_dir / "anomaly_series.png", dpi=150)
ccf_plot(ccf_raw, ccf_anom, f"{SITE}: CCF with water level").savefig(out_dir / "ccf.png", dpi=150)
dendrogram_plot(tree, list(X.columns), list(clusters.index[clusters["proposed"]]), cfg.REDUNDANCY_THRESHOLD,
                f"{SITE}: bands and indices, threshold {cfg.REDUNDANCY_THRESHOLD}").savefig(out_dir / "dendrogram.png", dpi=150)
rain_window_plot(rain, f"{SITE}: rain window vs {cfg.RAIN_MOISTURE_INDEX} anomaly").savefig(
    out_dir / "rain_windows.png", dpi=150)

acf_pacf_plot(memory, cfg.HV_H, f"{SITE}: ACF and PACF of the depth (development period)").savefig(
    out_dir / "acf_pacf.png", dpi=150)

# 9. Summary. Then edit DECISIONS in config.py (regime, SPECTRAL, LEADS, WINDOWS) with date and reason.
print(f"{SITE}: development period {y.dropna().index.min().date()} to {y.dropna().index.max().date()}, "
      f"{y.notna().sum()} water level days")
print("signs (indices):\n", signs.loc[list(indices.columns)].round(3))
print("regime days below/above surface:", int(regimes["n_below_surface"].iloc[0]), "/", int(regimes["n_above_surface"].iloc[0]))
for kind, table in memory.items():
    below = table.index[(table["acf"].abs() < table["conf"]) & (table.index > 0)]
    print(f"ACF {kind}: first lag inside the 95% band = {below.min() if len(below) else '>' + str(cfg.ACF_MAX_LAG)}; "
          f"ACF at h={cfg.HV_H}: {table['acf'].loc[cfg.HV_H]:.3f}")
print("leads:\n", leads.round(3).to_string(index=False))
print("proposed LEADS:", proposed_leads)
print("rain windows:\n", rain.round(3))
print("redundancy clusters (* = proposed):\n", clusters.assign(column=[f"{c} *" if p else c for c, p in zip(clusters.index, clusters["proposed"])])
      .groupby("cluster")["column"].apply(", ".join).to_string())
