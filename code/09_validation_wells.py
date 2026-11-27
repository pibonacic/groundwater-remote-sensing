"""Step 9: validate the modeled water table back in time (08_predictions) against independent wells of the basin
(Collahuasi). Both series are compared by Spearman rank correlation on monthly values (monthly min, median and max of
the daily predictions against the monthly median of each well, on the months where both have a value) and on their
seasonal cycles, also lagged by up to +-MAX_LAG_MONTHS (cross-correlation). Depth is turned into water table altitude (m a.s.l.) with the surface altitude of each well and
site; ranks do not change, so the correlations are computed on depth and checked on altitude."""
import numpy as np
import pandas as pd

import config as cfg
from functions.data import load_piezometer, load_table, normalize_piezometer, save_table
from functions.plots import (corr_heatmap, lag_heatmap, manual_measurements_plot, seasonal_cycle_plot,
                             well_validation_plot, wells_cycle_plot, wells_series_plot)
from functions.validation import (anomalies, cross_correlation, cycle_spearman, cycle_table, load_manual,
                                  load_wells, monthly_median, site_altitude, spearman_table, well_matrix,
                                  wells_altitude, zscore)

SITE = "sdh1"
WELLS_TO_PLOT = ['HPS-11', 'HPS-16']   # wells drawn in the figure (the correlations are computed for every well)
WELLS_PATH = cfg.RAW_DIR / "groundwater" / "collahuasi_wells_cleaned_long.csv"
MANUAL_PATH = cfg.RAW_DIR / "groundwater" / "SDH_punteras_collahuasi_mediciones_manuales.xlsx"   # user's field visits
WELLS_UNIT_TO_CM = 100.0     # WATER_LEVEL is a depth in m
MIN_MONTHS = 10              # minimum coincident months to compute a correlation
ALPHA = 0.05                 # p below this is marked in the heatmap
Z_LIMIT = 3                  # the all-wells figure shows z-scores from -Z_LIMIT to Z_LIMIT ...
RAW_LIMIT_CM = 150           # ... depths down to RAW_LIMIT_CM and altitudes down to the lowest well surface minus
                             # RAW_LIMIT_CM (values beyond are triangles on the edge)
STATS = ["min", "median", "max"]   # modeled monthly statistics: correlated with the wells; figures draw the median and the min-max band
CYCLE_Z_MAX = 3              # seasonal cycle: monthly values with |z-score| above this are left out ...
CYCLE_MIN_MONTHS = 2         # ... and a calendar month needs at least this many distinct year-months
CYCLE_VALID_MONTHS = 11      # a well's seasonal cycle is compared with the predictions only with this many months
CYCLE_PERIOD = "full"        # months of the wells used for their seasonal cycle: "full" (whole Collahuasi record) or
                             # "shared" (only the months of the predictions); the modeled cycle always uses the shared months
MAX_LAG_MONTHS = 3           # cross-correlation lags: -MAX_LAG_MONTHS..MAX_LAG_MONTHS (positive: the well lags the model)

# Surface altitude (m a.s.l.) of the sites and of the Collahuasi wells (corrected_hgt of tabla_8_4_punteras, puntera
# with the prefix "HPS-", rounded to cm). Wells missing here have no altitude series.
SITE_ALTITUDE = {"sdh1": 3827.6, "sdh2": 3827.5}
WELL_ALTITUDE = {
    "HPS-01": 3829.75, "HPS-02": 3830.45, "HPS-03": 3824.25, "HPS-04": 3822.25, "HPS-05": 3825.09, "HPS-06": 3825.19,
    "HPS-07": 3824.12, "HPS-08": 3824.12, "HPS-11": 3828.3, "HPS-12": 3828.7, "HPS-13": 3830.26, "HPS-14": 3830.26,
    "HPS-17": 3822.14, "HPS-18": 3823.12, "HPS-19": 3822.85, "HPS-20": 3822.4, "HPS-21": 3823.31, "HPS-21A": 3823.11,
    "HPS-22": 3823.31, "HPS-23": 3823.01, "HPS-24": 3822.71, "HPS-25": 3822.61, "HPS-26": 3822.71, "HPS-27": 3822.8,
    "HPS-28": 3822.51, "HPS-29": 3822.7, "HPS-30": 3822.8, "HPS-31": 3822.6, "HPS-32": 3822.59, "HPS-33": 3822.55,
    "HPS-34": 3822.17, "HPS-35": 3822.39, "HPS-36": 3822.44, "HPS-37": 3822.97, "HPS-38": 3824.41, "HPS-39": 3826.29,
    "HPS-40": 3825.88,
}

assert CYCLE_PERIOD in ("full", "shared"), CYCLE_PERIOD
out_dir = cfg.out_dir(SITE, "09_validation_wells")
tables_dir, figures_dir = out_dir / "tables", out_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
LAGS = range(-MAX_LAG_MONTHS, MAX_LAG_MONTHS + 1)
well_alt = lambda depth_cm: wells_altitude(depth_cm, WELL_ALTITUDE)
site_alt = lambda depth_cm: site_altitude(depth_cm, SITE_ALTITUDE[SITE])

# 1. Independent wells: depth (cm, positive below the surface) per day and well, then the monthly median of each well
#    (1-2 visits a month). Months are labeled by their first day.
wells = load_wells(WELLS_PATH, WELLS_UNIT_TO_CM)
wells_monthly = monthly_median(wells)

# 2. Daily predictions of 08 (point prediction, scheme PARTITION) over the period of the wells, aggregated by month,
#    as depth and as altitude (aggregated from the daily altitude, so its min is the day of the deepest level).
pred = load_table(cfg.out_dir(SITE, "08_predictions") / "predictions.csv")["pred"]
first, last = wells["DATE"].min(), wells["DATE"].max()
pred = pred.loc[first:last].dropna()
by_month = pred.index.to_period("M").to_timestamp().rename("month")
pred_monthly = pred.groupby(by_month).agg(STATS)
pred_monthly_alt = site_alt(pred).groupby(by_month).agg(STATS)
wells_monthly_all = wells_monthly                 # whole record of the wells
wells_monthly = wells_monthly.loc[pred_monthly.index.min():pred_monthly.index.max()]
wells_cycle_source = wells_monthly_all if CYCLE_PERIOD == "full" else wells_monthly   # months of the wells' cycle

# 3. Spearman, for each well and statistic, on the months where both have a value (z-scores do not change ranks).
corr = spearman_table(pred_monthly, wells_monthly, STATS, MIN_MONTHS)
save_table(corr, tables_dir / "spearman_monthly.csv")
save_table(pred_monthly.join(wells_monthly, how="outer"), tables_dir / "monthly_series_depth_cm.csv")
save_table(pred_monthly_alt.join(well_alt(wells_monthly), how="outer"), tables_dir / "monthly_series_altitude_masl.csv")

# 4. Figures on every month of the period (months without a value are NaN, so the lines are cut where a well skips
#    a month). z-scores: each depth series over its own months in the period.
months = pd.date_range(pred_monthly.index.min(), pred_monthly.index.max(), freq="MS", name="month")
pred_plot, wells_plot = pred_monthly.reindex(months), wells_monthly.reindex(months).dropna(axis=1, how="all")
pred_plot_alt, wells_plot_alt = pred_monthly_alt.reindex(months), well_alt(wells_plot)
span = f"{months.min():%Y-%m} to {months.max():%Y-%m}"

# 4a. The chosen wells against the monthly min, median and max of the predictions.
missing = [w for w in WELLS_TO_PLOT if w not in wells_plot]
assert not missing, f"wells without data in the period: {missing}"
shown = wells_plot[WELLS_TO_PLOT]
well_validation_plot(pred_plot.apply(zscore), shown.apply(zscore), pred_plot, shown, pred_plot_alt,
                     wells_plot_alt[WELLS_TO_PLOT],
                     f"{SITE} ({cfg.PARTITION}): monthly modeled water table vs Collahuasi wells, {span}").savefig(
    figures_dir / "validation_series.png", dpi=150)

# 4b. Every well with data in the period, with the modeled monthly median and min-max as reference.
plotted_surfaces = [WELL_ALTITUDE[w] for w in wells_plot if w in WELL_ALTITUDE]
wells_series_plot(wells_plot.apply(zscore), wells_plot, wells_plot_alt.dropna(axis=1, how="all"),
                  pred_plot.apply(zscore), pred_plot, pred_plot_alt,
                  f"{SITE} ({cfg.PARTITION}): Collahuasi wells (monthly median) and modeled monthly median and min-max, {span}",
                  z_limit=Z_LIMIT, raw_limit=RAW_LIMIT_CM,
                  alt_limit=round(min(plotted_surfaces) - RAW_LIMIT_CM / 100, 2)).savefig(
    figures_dir / "wells_series.png", dpi=150)

# 4c. Spearman of every well with the monthly min, median and max (dot: p < ALPHA).
rho = corr.pivot(index="well", columns="prediction", values="rho")[STATS].dropna(how="all")
significant = corr.pivot(index="well", columns="prediction", values="p")[STATS].reindex(rho.index) < ALPHA
corr_heatmap({"modeled monthly statistic": (rho, significant)},
             f"{SITE} ({cfg.PARTITION})\nSpearman, modeled vs wells\n(monthly, {span})",
             dot_label=f"p < {ALPHA}").savefig(figures_dir / "spearman_heatmap.png", dpi=150)

# 5. Seasonal cycle: median of each calendar month over every year, for each well (monthly median over the months
#    set by CYCLE_PERIOD: whole record, also before the predictions, or only the shared months) and each modeled
#    monthly statistic (period shared with the wells).
#    Monthly values with |z| > CYCLE_Z_MAX (z over the series' own months) are left out; a calendar month is kept
#    only with at least CYCLE_MIN_MONTHS distinct year-months. Depth and altitude cycles are computed the same way.
cycle = cycle_table(pred_monthly, wells_cycle_source.dropna(axis=1, how="all"), STATS, CYCLE_Z_MAX, CYCLE_MIN_MONTHS)
cycle_alt = cycle_table(pred_monthly_alt, well_alt(wells_cycle_source).dropna(axis=1, how="all"), STATS,
                        CYCLE_Z_MAX, CYCLE_MIN_MONTHS)
cycle_span = f"{wells_cycle_source.index.min():%Y-%m} to {wells_cycle_source.index.max():%Y-%m}"
save_table(cycle, tables_dir / "seasonal_cycle_depth_cm.csv")
save_table(cycle_alt, tables_dir / "seasonal_cycle_altitude_masl.csv")

pred_cols = [f"pred_{stat}" for stat in STATS]
pred_cycle = cycle[pred_cols].set_axis(STATS, axis=1)
pred_cycle_alt = cycle_alt[pred_cols].set_axis(STATS, axis=1)
shown_cycle = cycle.reindex(columns=WELLS_TO_PLOT)
seasonal_cycle_plot(pred_cycle.apply(zscore), shown_cycle.apply(zscore), pred_cycle, shown_cycle, pred_cycle_alt,
                    cycle_alt.reindex(columns=WELLS_TO_PLOT),
                    f"{SITE} ({cfg.PARTITION}): seasonal cycle, modeled {span} vs Collahuasi wells "
                    f"{cycle_span} ({CYCLE_PERIOD})\n"
                    f"(|z| <= {CYCLE_Z_MAX}, >= {CYCLE_MIN_MONTHS} year-months per month)").savefig(
    figures_dir / "seasonal_cycle.png", dpi=150)

# 6. Seasonal cycle of each well against that of the modeled min, median and max: Spearman over the calendar months
#    where both have a value, for the wells with at least CYCLE_VALID_MONTHS months (constant cycles, flowing wells
#    at depth 0, are left out). Each series also gets the month of its deepest and shallowest median. Prediction
#    rows carry only these descriptors.
valid_wells = [w for w in cycle.columns.drop(pred_cols)
               if cycle[w].notna().sum() >= CYCLE_VALID_MONTHS and cycle[w].nunique() > 1]   # constant: flowing well
if not valid_wells:   # e.g. CYCLE_PERIOD = "shared": too few years per calendar month
    print(f"no well has >= {CYCLE_VALID_MONTHS} months in its seasonal cycle ({CYCLE_PERIOD}): "
          f"the cycle correlation figures are skipped")

cycle_corr = cycle_spearman(cycle, valid_wells, STATS)
save_table(cycle_corr, tables_dir / "seasonal_cycle_spearman.csv")

wells_corr = cycle_corr.set_index("series").loc[valid_wells]
if valid_wells:
    corr_heatmap({"modeled climatology": (wells_corr[[f"rho_{s}" for s in STATS]].set_axis(STATS, axis=1),
                                          wells_corr[[f"p_{s}" for s in STATS]].set_axis(STATS, axis=1) < ALPHA)},
                 f"{SITE} ({cfg.PARTITION})\nSpearman of seasonal cycles\n(wells with >= {CYCLE_VALID_MONTHS} months)",
                 dot_label=f"p < {ALPHA}").savefig(figures_dir / "seasonal_cycle_spearman.png", dpi=150)

# 6a. Seasonal cycle of every valid well (no predictions).
valid_cycle = cycle[valid_wells]
if valid_wells:
    wells_cycle_plot(valid_cycle.apply(zscore), valid_cycle,
                     cycle_alt.reindex(columns=valid_wells).dropna(axis=1, how="all"),
                     f"{SITE}: seasonal cycle of the Collahuasi wells with >= {CYCLE_VALID_MONTHS} months, "
                     f"{cycle_span} ({CYCLE_PERIOD})\n"
                     f"(monthly median, |z| <= {CYCLE_Z_MAX}, >= {CYCLE_MIN_MONTHS} year-months per month)").savefig(
        figures_dir / "wells_seasonal_cycle.png", dpi=150)

# 6b. Spearman matrix between the seasonal cycles of the valid wells, each pair on the calendar months where both
#     have a value (rho in one CSV, p in another). The figure shows the lower triangle; the diagonal is gray and the
#     repeated upper half is blank.
well_rho, well_p = well_matrix(cycle, valid_wells)
save_table(well_rho, tables_dir / "wells_seasonal_cycle_spearman.csv")
save_table(well_p, tables_dir / "wells_seasonal_cycle_spearman_p.csv")
diagonal = pd.DataFrame(np.eye(len(valid_wells), dtype=bool), index=well_rho.index, columns=well_rho.columns)
upper = pd.DataFrame(np.triu(np.ones(well_rho.shape, dtype=bool), k=1), index=well_rho.index, columns=well_rho.columns)
if valid_wells:
    corr_heatmap({"": (well_rho.mask(diagonal | upper), (well_p < ALPHA) & ~diagonal & ~upper)},
                 f"{SITE}: Spearman between the seasonal cycles of the Collahuasi wells (>= {CYCLE_VALID_MONTHS} months)",
                 dot_label=f"p < {ALPHA}", hide=upper).savefig(figures_dir / "wells_seasonal_cycle_spearman.png", dpi=150)

# 6c. Cross-correlation, lags -MAX_LAG_MONTHS..MAX_LAG_MONTHS: the well at month t against the modeled statistic at
#     t - lag (positive lag: the well follows the model lag months later). Observations: monthly series over the
#     shared months, at least MIN_MONTHS coincident months. Seasonal cycle: the modeled cycle shifted circularly
#     (December is followed by January), valid wells only. Rows also flag the lag of largest |rho| of each well and
#     statistic.
shift_months = lambda table, lag: table.reindex(months).shift(lag)
shift_cycle = lambda table, lag: table.apply(lambda col: pd.Series(np.roll(col.to_numpy(), lag), index=col.index))
xcorr = cross_correlation(pred_monthly, wells_monthly, shift_months, LAGS, STATS, MIN_MONTHS)
xcorr_cycle = cross_correlation(pred_cycle, cycle[valid_wells], shift_cycle, LAGS, STATS, CYCLE_VALID_MONTHS)
save_table(xcorr, tables_dir / "crosscorr_monthly.csv")
save_table(xcorr_cycle, tables_dir / "crosscorr_seasonal_cycle.csv")
lag_heatmap(xcorr, STATS, ALPHA, f"{SITE} ({cfg.PARTITION}): Spearman, wells at t vs modeled at t - lag (monthly, {span})").savefig(
    figures_dir / "crosscorr_monthly.png", dpi=150)
if valid_wells:
    lag_heatmap(xcorr_cycle, STATS, ALPHA, f"{SITE} ({cfg.PARTITION}): Spearman of seasonal cycles, wells at month m vs modeled at "
                f"m - lag (circular; wells {cycle_span}, {CYCLE_PERIOD})").savefig(
        figures_dir / "crosscorr_seasonal_cycle.png", dpi=150)

# 6d. Anomalies over the shared months: each monthly value minus the seasonal cycle of its series for that calendar
#     month (wells: cycle over CYCLE_PERIOD; predictions: min, median and max against their own cycle). Depth in cm
#     (positive: deeper than usual) and altitude in m (positive: higher than usual; only wells with surface altitude).
#     Months whose calendar month has no valid cycle value stay empty.
pred_anom = anomalies(pred_monthly.reindex(months).set_axis(pred_cols, axis=1), cycle)
pred_anom_alt = anomalies(pred_monthly_alt.reindex(months).set_axis(pred_cols, axis=1), cycle_alt)
wells_anom = anomalies(wells_plot, cycle).dropna(axis=1, how="all")
wells_anom_alt = anomalies(well_alt(wells_plot).dropna(axis=1, how="all"), cycle_alt).dropna(axis=1, how="all")
anom_table = pd.concat([pd.concat([pred_anom, wells_anom], axis=1).add_suffix("_depth_cm"),
                        pd.concat([pred_anom_alt, wells_anom_alt], axis=1).add_suffix("_altitude_m")], axis=1)
save_table(anom_table.dropna(how="all").round(6), tables_dir / "anomalies_monthly.csv")

# Top panel: z-scores of the monthly depth (as in validation_series), not of the anomalies; no min-max band.
shown_anom = wells_anom.reindex(columns=WELLS_TO_PLOT)
well_validation_plot(pred_plot.apply(zscore), shown.apply(zscore),
                     pred_anom.set_axis(STATS, axis=1), shown_anom, pred_anom_alt.set_axis(STATS, axis=1),
                     wells_anom_alt.reindex(columns=WELLS_TO_PLOT),
                     f"{SITE} ({cfg.PARTITION}): monthly anomalies from the seasonal cycle, modeled vs Collahuasi "
                     f"wells, {span}\n(baseline: modeled cycle {span}; wells cycle {cycle_span}, {CYCLE_PERIOD})",
                     pred_label="modeled {} anomaly", well_label="{} anomaly",
                     labels=("depth (z-score)", "depth anomaly (cm)", "altitude anomaly (m)"),
                     zero_lines=True, band=False).savefig(figures_dir / "anomalies_series.png", dpi=150)

# 6e. Spearman between the anomalies of each well and of the modeled min, median and max, on the shared months where
#     both have a value (at least MIN_MONTHS). Computed on depth; altitude anomalies are depth anomalies / -100 on both
#     sides, so the ranks and rho are the same.
corr_anom = spearman_table(pred_anom.set_axis(STATS, axis=1), wells_anom, STATS, MIN_MONTHS)
save_table(corr_anom, tables_dir / "spearman_anomalies.csv")
rho_anom = corr_anom.pivot(index="well", columns="prediction", values="rho")[STATS].dropna(how="all")
significant_anom = corr_anom.pivot(index="well", columns="prediction", values="p")[STATS].reindex(rho_anom.index) < ALPHA
corr_heatmap({"modeled monthly anomaly": (rho_anom, significant_anom)},
             f"{SITE} ({cfg.PARTITION})\nSpearman of anomalies,\nmodeled vs wells ({span})",
             dot_label=f"p < {ALPHA}").savefig(figures_dir / "spearman_anomalies.png", dpi=150)

# 6f. Recent manual measurements of the user in Collahuasi wells (depth from the ground, suelo_a_agua_cm; negative:
#     water above the ground) against the observed depth of the site piezometer, from one month before the first
#     visit to one month after the last. Top: daily; middle: monthly medians placed on day 15; bottom: the daily
#     values as altitude (surface altitude of the site / well minus depth; wells without surface altitude are left
#     out of that panel). A measurement is labeled by its Collahuasi id, or by the field id when the well has none
#     or is a second tube ("(2)"); only the wells of WELLS_TO_PLOT with a measurement are drawn.
manual_all = load_manual(MANUAL_PATH)
manual = manual_all[manual_all["well"].isin(WELLS_TO_PLOT)]   # only the listed wells; those never measured are not drawn
site_cfg = cfg.SITES[SITE]
observed = normalize_piezometer(load_piezometer(site_cfg["piezo_path"], site_cfg["piezo_column"]),
                                site_cfg["piezo_unit"], site_cfg["piezo_zero_cm"], site_cfg["piezo_sign"])
observed_record = observed.round(6)   # whole record (difference table); round: float noise of m -> cm
window = (manual["date"].min() - pd.DateOffset(months=1), manual["date"].max() + pd.DateOffset(months=1))
observed = observed_record.loc[window[0]:window[1]].asfreq("D")
observed_alt = site_alt(observed)
manual_alt = manual.assign(altitude_m=manual["well"].map(WELL_ALTITUDE) - manual["depth_cm"] / 100).dropna(subset=["altitude_m"])
# Daily difference of every manual measurement (all wells, not only WELLS_TO_PLOT) with the site piezometer on the same day: depth (cm, well - site; positive: the well is deeper) and altitude (cm, well - site; positive: the well's water table is higher). Altitude needs the surface altitude of the well (empty otherwise).
site_day = observed_record.reindex(pd.DatetimeIndex(manual_all["date"]).normalize()).to_numpy()   # site depth on the day of each visit
manual_diff = manual_all.assign(
    site_depth_cm=site_day,
    diff_depth_cm=lambda d: (d["depth_cm"] - d["site_depth_cm"]).round(4),
    well_altitude_m=lambda d: (d["well"].map(WELL_ALTITUDE) - d["depth_cm"] / 100).round(6),
    site_altitude_m=lambda d: site_alt(d["site_depth_cm"]),
    diff_altitude_cm=lambda d: ((d["well_altitude_m"] - d["site_altitude_m"]) * 100).round(4),
).rename(columns={"depth_cm": "well_depth_cm"})
manual_diff = manual_diff[["date", "well", "well_depth_cm", "site_depth_cm", "diff_depth_cm",
                           "well_altitude_m", "site_altitude_m", "diff_altitude_cm"]]
save_table(manual_diff.reset_index(drop=True), tables_dir / f"manual_vs_{SITE}_daily.csv")
day15 = lambda index: index.to_period("M").to_timestamp() + pd.Timedelta(days=14)
window_months = pd.date_range(day15(pd.DatetimeIndex([window[0]]))[0], window[1], freq="MS") + pd.Timedelta(days=14)
observed_monthly = observed.groupby(day15(observed.index)).median().reindex(window_months)
manual_monthly = (manual.assign(month=day15(pd.DatetimeIndex(manual["date"])))
                  .pivot_table(index="month", columns="well", values="depth_cm", aggfunc="median", sort=False)
                  .reindex(window_months))
manual_measurements_plot(observed, manual, observed_monthly, manual_monthly, window,
                         f"{SITE}: observed depth of the site piezometer and manual measurements in Collahuasi wells, "
                         f"{manual['date'].min():%Y-%m-%d} to {manual['date'].max():%Y-%m-%d}",
                         observed_label=f"{SITE.upper()} piezometer", observed_alt=observed_alt,
                         manual_alt=manual_alt).savefig(
    figures_dir / "manual_measurements.png", dpi=150)

# 7. Check: the same correlations on altitude (no outputs). Altitude is a decreasing linear function of depth for
#    wells and predictions alike, so ranks are reversed on both sides and rho and p do not change; the modeled
#    min/max swap (the lowest altitude is the deepest level).
SWAP = {"min": "max", "median": "median", "max": "min"}
with_altitude = [w for w in wells_monthly if w in WELL_ALTITUDE]
corr_alt = spearman_table(pred_monthly_alt, well_alt(wells_monthly[with_altitude]), STATS, MIN_MONTHS)
corr_alt["prediction"] = corr_alt["prediction"].map(SWAP)
check = corr.merge(corr_alt, on=["well", "prediction"], suffixes=("", "_alt"))
valid_alt = [w for w in valid_wells if w in WELL_ALTITUDE]
cycle_corr_alt = cycle_spearman(cycle_alt.rename(columns={f"pred_{s}": f"pred_{SWAP[s]}" for s in STATS}), valid_alt, STATS)
check_cycle = cycle_corr.set_index("series").loc[valid_alt]
rho_alt, p_alt = well_matrix(cycle_alt, valid_alt)
differences = {
    "monthly rho": (check["rho"] - check["rho_alt"]).abs().max(),
    "monthly p": (check["p"] - check["p_alt"]).abs().max(),
    "cycle rho": max((check_cycle[f"rho_{s}"] - cycle_corr_alt.set_index("series").loc[valid_alt, f"rho_{s}"]).abs().max()
                     for s in STATS),
    "cycle p": max((check_cycle[f"p_{s}"] - cycle_corr_alt.set_index("series").loc[valid_alt, f"p_{s}"]).abs().max()
                   for s in STATS),
    "well matrix rho": (well_rho.loc[valid_alt, valid_alt] - rho_alt).abs().max().max(),
    "well matrix p": (well_p.loc[valid_alt, valid_alt] - p_alt).abs().max().max(),
}

# 8. Summary.
print(f"{SITE} ({cfg.PARTITION}): predictions {pred.index.min().date()} to {pred.index.max().date()}, "
      f"{len(pred_monthly)} months; {wells_monthly.shape[1]} wells")
print(corr[corr["well"].isin(WELLS_TO_PLOT)].round({"rho": 3, "p": 3}).to_string(index=False))
print("all wells, median rho by statistic:", corr.groupby("prediction")["rho"].median().round(3).to_dict())
print("seasonal cycle (cm):")
print(cycle[pred_cols + WELLS_TO_PLOT].round(1).to_string())
print(f"seasonal cycle, {len(valid_wells)} wells with >= {CYCLE_VALID_MONTHS} months:")
print(cycle_corr.round(3).to_string(index=False))
for name, table in (("monthly", xcorr), ("seasonal cycle", xcorr_cycle)):
    best = table[table["best_lag"] & (table["prediction"] == "median")]
    print(f"cross-correlation ({name}), modeled median, lag of largest |rho|:")
    print(best[["well", "lag_months", "n_months", "rho", "p"]].round(3).to_string(index=False))
print("anomalies, Spearman with the modeled median:")
print(corr_anom[corr_anom["prediction"] == "median"].dropna(subset=["rho"])[["well", "n_months", "rho", "p"]]
      .round(3).to_string(index=False))
print("wells without surface altitude:", sorted(set(wells_monthly_all.columns) - set(WELL_ALTITUDE)))
print(f"depth vs altitude, largest |difference| ({len(check.dropna(subset=['rho']))} monthly pairs, "
      f"{len(valid_alt)} wells in the cycles):", {k: float(f"{v:.2e}") for k, v in differences.items()})
