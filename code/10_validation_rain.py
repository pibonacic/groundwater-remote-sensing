"""Step 10: validate the water table (modeled back in time by 08_predictions, and observed at the site piezometer) against
CHIRPS rainfall of the basin. Rain is accumulated over trailing windows and shown with the modeled and the observed depth
(precip_accumulated*.png); then a one-sided a priori hypothesis is tested: the rain accumulated over q months (SPI_q,
q = 2, 3, 4) raises the standardized level z (monthly mean of the daily anomaly of the level) 5 to 7 months later
(k = 5, 6, 7), that is, Pearson r > 0 between SPI_q(t - k) and z(t).
Flow: C1 SPI of the monthly rain, C2 monthly z of the level (both shared); observed level O1-O7 (the 9 cells q x k in two
scenarios, Gao et al. 2024 p-values, family test with AR(1) and random-phase surrogates, slopes of SPI-3(t-6) vs z); modeled
level M1-M5b (the single cell SPI-3, k = 6 with surrogates and slopes, and the family of 9 cells). The two scenarios are all
months and without Jan-Mar (the month t of the level)."""
import numpy as np
import pandas as pd

import config as cfg
from functions.data import load_piezometer, load_table, normalize_piezometer, save_table
from functions.plots import (correlation_matrix_figure, daily_series_figure, level_anomaly_figure, monthly_series_figure,
                             precip_accumulation_plot, rain_level_figure, scatter_grid_figure, series_cross_correlation_figure,
                             single_cell_surrogate_figure, slope_scatter_figure, spi_lag0_figure, surrogate_process_figure)
from functions.validation import (GENERATORS, SCENARIOS, accumulate_daily, analyze_anomalies, cell_correlation, cell_rows,
                                  cell_surrogate_rows, cell_surrogate_test, contiguous_window, correlation_rows, correlations,
                                  family_rows, family_test, lag0_correlations, lagged_grid, load_precip, monthly_rain,
                                  scatter_pairs, scatter_r2, scatter_r2_rows, scenario_masks, slope_amplitude, slope_rows,
                                  spi_table)

SITE = "sdh1"
PRECIP_PATH = cfg.RAW_DIR / "precipitation" / "SDH_CHIRPS_2002-2026.csv"   # CHIRPS daily rain 2002-2026 (basin)
ACCUMULATION_WINDOWS = {"15d": 15, "1m": 30, "2m": 60, "3m": 91, "6m": 183, "12m": 365}   # accumulated rain: label -> days
ACCUMULATED_PLOT_START = "2011-01-01"   # first day of the accumulated rain vs modeled depth figure (the table keeps the whole period)
OBSERVED_RAIN_START = "2023-01-01"      # first day of the rain vs observed depth figure
OBSERVED_START = "2024-05-22"           # first day of the observed level used in the hypothesis test
RAIN_LEVEL_WINDOWS = {"15d": 15, "90d": 90}   # rolling accumulations (count of days) of the rain vs level figures
SPI_Q = [2, 3, 4]                       # SPI windows (months) of the hypothesis
LAGS_K = [5, 6, 7]                      # lags (months): the level at t responds to the SPI of t - k
SCATTER_Q = [1, 2, 3, 4, 6]             # SPI windows of the modeled scatter grid
SCATTER_K = [0, 1, 2, 3, 5, 6, 7]       # lags of the modeled scatter grid
HYPOTHESIS_Q, HYPOTHESIS_K = 3, 6       # single cell of the slopes, the modeled test and the figures
SPI_REFERENCE_YEARS = (2002, 2025)      # gamma fit of every calendar month on these years
SPI_ZERO_MM = 0.1                       # accumulations up to this count as zero (mass at zero)
SPI_CLIP = 3                            # SPI limited to +-SPI_CLIP
SPI_MIN_NONZERO = 10                    # fewer non-zero values: that month has no fit and no SPI
LEVEL_HARMONICS_K = 3                   # harmonics of the seasonal mean curve of the level
SIGMA_HARMONICS_K = 3                   # harmonics of the seasonal scale (sigma) of the level
PERIOD_DAYS = 365.25                    # period of the harmonics
SIGMA_FLOOR = 0.1                       # sigma(t) is at least this fraction of the global standard deviation of the residual
MIN_COVERAGE = 0.8                      # a month needs this fraction of its days to have a monthly z
EXCLUDED_MONTHS = (1, 2, 3)             # months t of the level dropped in the scenario "without Jan-Mar"
N_SURROGATES = 20000                    # surrogate series per generator
SEED = 12345                            # AR(1) surrogates: SEED; random-phase surrogates: SEED + 1

out_dir = cfg.out_dir(SITE, "10_validation_rain")
tables_dir, figures_dir = out_dir / "tables", out_dir / "figures"
for folder in (tables_dir / "observed", tables_dir / "modeled", figures_dir / "observed", figures_dir / "modeled"):
    folder.mkdir(parents=True, exist_ok=True)
cell = f"spi{HYPOTHESIS_Q}_k{HYPOTHESIS_K}"   # name of the single cell in file names
grid = f"k{LAGS_K[0]}-{LAGS_K[-1]}_all_vs_no_jan-mar"
site_name = SITE.upper()
GENERATOR_KEY = dict(zip(GENERATORS, ("ar1", "phase")))
SCENARIO_KEY = dict(zip(SCENARIOS, ("all", "no_jan_mar")))


def save_npz(path, prefix, results, field, cor):
    """Save field of every (scenario, generator) result as <prefix>_<generator>_<scenario> next to the observed r of each scenario."""
    np.savez(path, **{f"{prefix}_{GENERATOR_KEY[g]}_{SCENARIO_KEY[s]}": results[(s, g)][field] for s in SCENARIOS for g in GENERATORS},
             **{f"r_obs_{SCENARIO_KEY[s]}": c["r"] for s, c in zip(SCENARIOS, cor)}, seed=SEED, n_surrogates=N_SURROGATES)


# 1. Observed depth of the site piezometer (whole record) and daily predictions of 08.
site_cfg = cfg.SITES[SITE]
observed_record = normalize_piezometer(load_piezometer(site_cfg["piezo_path"], site_cfg["piezo_column"]),
                                       site_cfg["piezo_unit"], site_cfg["piezo_zero_cm"], site_cfg["piezo_sign"]).round(6)
pred_full = load_table(cfg.out_dir(SITE, "08_predictions") / "predictions.csv")[["pred", "lo80", "hi80"]]

# 2. Daily rainfall accumulated over the trailing windows of ACCUMULATION_WINDOWS (15, 30, 60, 91, 183 and 365 days;
#    windows end on each day, only complete windows), shown with the daily modeled depth (the model
#    starts in 2012) from ACCUMULATED_PLOT_START and with the observed depth of the site piezometer from OBSERVED_RAIN_START.
precip = load_precip(PRECIP_PATH)
windows = ACCUMULATION_WINDOWS
accumulated = accumulate_daily(precip, windows)
days = pd.date_range(precip.index.min(), max(precip.index.max(), pred_full.index.max()), freq="D", name="date")
precip_table = pd.concat([precip, accumulated,
                          pred_full.rename(columns={"pred": "pred_cm", "lo80": "pred_lo80_cm", "hi80": "pred_hi80_cm"})],
                         axis=1).reindex(days).round(4)
save_table(precip_table, tables_dir / "precip_accumulated.csv")
precip_accumulation_plot(accumulated.loc[ACCUMULATED_PLOT_START:], pred_full, windows,
                         f"{SITE} ({cfg.PARTITION}): accumulated CHIRPS rainfall and modeled depth, "
                         f"{pd.Timestamp(ACCUMULATED_PLOT_START):%Y-%m} to {days[-1]:%Y-%m}").savefig(
    figures_dir / "precip_accumulated.png", dpi=150)
observed_since = observed_record.loc[OBSERVED_RAIN_START:].asfreq("D")   # accumulations use the rain before the start
precip_accumulation_plot(accumulated.loc[OBSERVED_RAIN_START:], observed_since.to_frame("pred"), windows,
                         f"{SITE}: accumulated CHIRPS rainfall and observed depth of the {site_name} piezometer, "
                         f"{pd.Timestamp(OBSERVED_RAIN_START):%Y-%m} to "
                         f"{max(precip.index.max(), observed_since.index.max()):%Y-%m}",
                         depth_label="observed depth").savefig(figures_dir / "precip_accumulated_observed.png", dpi=150)

# Levels in the convention of the analysis: m, negative below the ground (depth_cm positive down, so level = -depth / 100).
#   observed: the piezometer from OBSERVED_START; modeled: the daily prediction of 08 (pred, median of the ensemble).
observed_level = (-observed_record.loc[OBSERVED_START:] / 100).dropna().rename("level_m")
modeled_level = (-pred_full["pred"] / 100).dropna().rename("level_m")

# C1. SPI of the monthly rain (shared). Daily rain -> monthly totals (a month with any missing day is NaN) -> accumulations of
#     q months; for every q and calendar month a gamma (maximum likelihood, location 0) is fitted on the non-zero values of the
#     reference years (zero: accumulation <= SPI_ZERO_MM; fewer than SPI_MIN_NONZERO non-zero values: no fit, no SPI), with the
#     mass at zero of Stagge et al. (2015): p0 = n0 / (n + 1); H = (n0 + 1) / (2 (n + 1)) for a zero, else p0 + (1 - p0) G(x);
#     SPI = normal quantile of H limited to +-SPI_CLIP. The windows of the scatter grid are computed too, but only SPI_Q are saved.
monthly = monthly_rain(precip)
spi_full, gamma_parameters = spi_table(monthly, sorted(set(SPI_Q) | set(SCATTER_Q)), SPI_REFERENCE_YEARS, SPI_ZERO_MM, SPI_CLIP,
                                       SPI_MIN_NONZERO)
spi = spi_full[["rain_mm"] + [f"SPI_{q}" for q in SPI_Q]]
save_table(spi.rename_axis("date").round(4), tables_dir / "spi_monthly.csv")
gamma_parameters[gamma_parameters["q"].isin(SPI_Q)].round(4).to_csv(tables_dir / "spi_gamma_parameters.csv", index=False)
print(f"rain {precip.index.min().date()} to {precip.index.max().date()}: {int(monthly.notna().sum())} complete months; "
      f"SPI windows {SPI_Q}, reference {SPI_REFERENCE_YEARS[0]}-{SPI_REFERENCE_YEARS[1]}")
for _, row in gamma_parameters[(gamma_parameters["method"] == "no fit") & gamma_parameters["q"].isin(SPI_Q)].iterrows():
    print(f"warning: SPI-{int(row['q'])}, month {int(row['month'])} has no fit (fewer than {SPI_MIN_NONZERO} non-zero values)")
acf_x = [spi[f"SPI_{q}"].dropna().to_numpy() for q in SPI_Q]   # SPI of every q over its whole record (Gao autocorrelation)
iq_h, ik_h = SPI_Q.index(HYPOTHESIS_Q), LAGS_K.index(HYPOTHESIS_K)
rain_windows = accumulate_daily(precip, RAIN_LEVEL_WINDOWS)
analysis = dict(k=LEVEL_HARMONICS_K, k_sigma=SIGMA_HARMONICS_K, period_days=PERIOD_DAYS, sigma_floor=SIGMA_FLOOR, min_coverage=MIN_COVERAGE)


# C2. Monthly z of the level (shared method, applied to the observed and to the modeled level). Seasonal mean curve = harmonic
#     regression (LEVEL_HARMONICS_K harmonics of PERIOD_DAYS, no trend), residual r = level - curve; seasonal scale
#     sigma(t) = sqrt(pi / 2) times the harmonic fit (SIGMA_HARMONICS_K) of |r|, at least SIGMA_FLOOR times the standard deviation
#     of r; daily z = r / sigma(t); monthly z = mean of z in the months with at least MIN_COVERAGE of their days.
def save_level_tables(an, folder):
    """Daily level, seasonal curve, anomaly, sigma and z, and the monthly z with its days and coverage."""
    save_table(an["table"].rename_axis("date").round(4), folder / "harmonic_anomalies.csv")
    save_table(an["zm_table"].rename(columns={"mean": "z_monthly"}).rename_axis("date").round(4), folder / "z_monthly.csv")


# O1-O2. Observed level of the piezometer from OBSERVED_START and its monthly z.
an_obs = analyze_anomalies(observed_level, **analysis)
zm_obs = an_obs["zm"]
save_level_tables(an_obs, tables_dir / "observed")
print(f"observed level {an_obs['first']:%Y-%m-%d} to {an_obs['last']:%Y-%m-%d} ({an_obs['n']} days): {len(zm_obs)} months with "
      f">= {MIN_COVERAGE:.0%} of the days ({zm_obs.index.min():%Y-%m} to {zm_obs.index.max():%Y-%m}); "
      f"days with sigma at the floor: {an_obs['n_floor']}")

# O3-O4. The 9 cells q x k: Pearson r between SPI_q(t - k) and z(t) over the months where both exist (one-sided, r > 0), in
#        the two scenarios. p of Gao et al. (2024) of every cell: reduction factor delta = [1 + 2 sum_{j=1..M} ((n - j) / n)
#        rho_xx(j) rho_yy(j)]^-1 with M = n // 4 (rho_xx: biased ACF of SPI_q over its whole record; rho_yy: of z over the whole
#        contiguous window, before dropping months; delta = 1 if the denominator <= 0, limited to [3 / n, 1]), nu = n delta - 2,
#        t = r sqrt(nu / (1 - r^2)), one-sided Student p; Sidak correction over the lags of each q.
y_obs, dates_obs, window_obs = contiguous_window(zm_obs, spi.index)
sx_obs = lagged_grid(spi, SPI_Q, LAGS_K, window_obs)
masks_obs = scenario_masks(dates_obs, EXCLUDED_MONTHS)
cor_obs = [correlations(sx_obs, y_obs, acf_x, y_obs, masks_obs[s]) for s in SCENARIOS]
correlation_rows(cor_obs, SPI_Q, LAGS_K).round(6).to_csv(tables_dir / "observed" / "correlations_9_cells.csv", index=False)

# O5. Family test: the maximum r over the 9 cells is compared with the maxima of N_SURROGATES surrogate series of z over the whole
#     contiguous window (the real SPI does not change; each surrogate gets the mask of each scenario): AR(1) (lag-1 autocorrelation,
#     mean and standard deviation of z) and random phases of the FFT (Prichard and Theiler 1994, same periodogram as z).
#     p of the family = (1 + #{max_s >= max_obs}) / (1 + N); critical r = 95th percentile of max_s.
fam_obs, surrogates_obs = family_test(sx_obs, y_obs, masks_obs, N_SURROGATES, SEED)
family_rows(cor_obs, fam_obs, SPI_Q, LAGS_K, N_SURROGATES, SEED).round(6).to_csv(tables_dir / "observed" / "family_test.csv", index=False)
save_npz(tables_dir / "observed" / "surrogate_maxima.npz", "max", fam_obs, "max_s", cor_obs)

# O7. Amplitude of the cell (SPI-3, k = 6): OLS and RMA slopes of z on SPI with 95% intervals (Gao nu), leave-one-out slopes and
#     Cook's distance of the OLS fit. It measures the size of the relation, not extra evidence for it.
slopes_obs = {}
for scenario, c in zip(SCENARIOS, cor_obs):
    x_cell, y_cell = sx_obs[iq_h, ik_h], np.where(masks_obs[scenario], y_obs, np.nan)
    ok = np.isfinite(x_cell) & np.isfinite(y_cell)
    slopes_obs[scenario] = slope_amplitude(x_cell[ok], y_cell[ok], dates_obs[ok], float(c["nu"][iq_h, ik_h]))
slope_table, slope_loo_table = slope_rows(slopes_obs)
slope_table.round(6).to_csv(tables_dir / "observed" / f"slope_{cell}.csv", index=False)
slope_loo_table.round(6).to_csv(tables_dir / "observed" / f"slope_{cell}_loo.csv", index=False)

# Observed figures.
fig_obs = figures_dir / "observed"
level_anomaly_figure(an_obs, site_name).savefig(fig_obs / "level_anomalies.png", dpi=150)
spi_lag0_figure(spi, zm_obs, SPI_Q, LAGS_K).savefig(fig_obs / "spi_q_vs_z_lag0.png", dpi=150)
correlation_matrix_figure(cor_obs, fam_obs, SPI_Q, LAGS_K).savefig(fig_obs / f"correlation_matrix_spi_z_{grid}.png", dpi=150)
series_cross_correlation_figure(spi, zm_obs, cor_obs, fam_obs, SPI_Q, LAGS_K, HYPOTHESIS_Q, HYPOTHESIS_K, EXCLUDED_MONTHS).savefig(
    fig_obs / f"series_and_cross_correlation_{cell}_all_vs_no_jan-mar.png", dpi=150)
surrogate_process_figure(sx_obs, y_obs, dates_obs, masks_obs[SCENARIOS[1]], cor_obs[1], fam_obs, surrogates_obs, SCENARIOS[1],
                         SPI_Q, LAGS_K).savefig(fig_obs / "surrogate_process_no_jan-mar.png", dpi=150)
slope_scatter_figure(slopes_obs, HYPOTHESIS_Q, HYPOTHESIS_K, EXCLUDED_MONTHS).savefig(
    fig_obs / f"slope_scatter_{cell}_all_vs_no_jan-mar.png", dpi=150)
rain_level_figure(rain_windows, observed_level, (pd.Timestamp(an_obs["first"].year, an_obs["first"].month, 1),
                                                 pd.Timestamp(an_obs["last"].year, 11, 30)), HYPOTHESIS_K, "observed",
                  "Observed water table level: raw daily series of the piezometer (m; negative = below the ground),").savefig(
    fig_obs / f"rain_15d_90d_and_level_observed_{an_obs['first'].year}-{an_obs['last'].year}.png", dpi=150)

# M1-M2. Modeled level (daily prediction pred of 08, 2012-2026) as level = -pred / 100 and its monthly z, same method as C2
#        and fitted on the whole modeled record.
an_mod = analyze_anomalies(modeled_level, **analysis)
zm_mod = an_mod["zm"]
save_level_tables(an_mod, tables_dir / "modeled")
y_mod, dates_mod, window_mod = contiguous_window(zm_mod, spi.index)
masks_mod = scenario_masks(dates_mod, EXCLUDED_MONTHS)
print(f"modeled level {an_mod['first']:%Y-%m-%d} to {an_mod['last']:%Y-%m-%d} ({an_mod['n']} days): {len(zm_mod)} months "
      f"({zm_mod.index.min():%Y-%m} to {zm_mod.index.max():%Y-%m}); days with sigma at the floor: {an_mod['n_floor']}")

# M3. Single cell (SPI-3, k = 6) with the modeled z: Pearson r in the two scenarios and the Gao p (same delta, M = n // 4, rho_yy of
#     the whole modeled window; no Sidak correction with a single lag). Surrogate test of the cell: N_SURROGATES per generator
#     (AR(1): SEED, phase: SEED + 1), the statistic is the r of the cell (not a maximum); p = (1 + #{r_s >= r_obs}) / (1 + N).
x_mod = spi[f"SPI_{HYPOTHESIS_Q}"].shift(HYPOTHESIS_K).to_numpy()[window_mod]
cor_mod = [cell_correlation(x_mod, y_mod, masks_mod[s], acf_x[iq_h]) for s in SCENARIOS]
cell_rows(cor_mod).round(6).to_csv(tables_dir / "modeled" / f"correlation_{cell}.csv", index=False)
fam_mod, surrogates_mod = cell_surrogate_test(x_mod, y_mod, masks_mod, N_SURROGATES, SEED)
cell_surrogate_rows(fam_mod, N_SURROGATES, SEED).round(6).to_csv(tables_dir / "modeled" / "surrogate_test.csv", index=False)
save_npz(tables_dir / "modeled" / "surrogate_r.npz", "r", fam_mod, "r_s", cor_mod)

# M4-M5. Amplitude of the modeled cell: same slopes, leave-one-out and Cook's distance as O7.
slopes_mod = {}
for scenario, c in zip(SCENARIOS, cor_mod):
    y_cell = np.where(masks_mod[scenario], y_mod, np.nan)
    ok = np.isfinite(x_mod) & np.isfinite(y_cell)
    slopes_mod[scenario] = slope_amplitude(x_mod[ok], y_cell[ok], dates_mod[ok], float(c["nu"]))
slope_table, slope_loo_table = slope_rows(slopes_mod)
slope_table.round(6).to_csv(tables_dir / "modeled" / f"slope_{cell}.csv", index=False)
slope_loo_table.round(6).to_csv(tables_dir / "modeled" / f"slope_{cell}_loo.csv", index=False)

# M5b. Family of the 9 cells (SPI-2/3/4 x k = 5, 6, 7) with the modeled z: same method as O3-O5 (rho_yy over the whole modeled
#      window before the mask; surrogates of the whole modeled window, masks applied after generating them).
sx_mod = lagged_grid(spi, SPI_Q, LAGS_K, window_mod)
cor9_mod = [correlations(sx_mod, y_mod, acf_x, y_mod, masks_mod[s]) for s in SCENARIOS]
correlation_rows(cor9_mod, SPI_Q, LAGS_K).round(6).to_csv(tables_dir / "modeled" / "correlations_9_cells.csv", index=False)
fam9_mod, _ = family_test(sx_mod, y_mod, masks_mod, N_SURROGATES, SEED)
family_rows(cor9_mod, fam9_mod, SPI_Q, LAGS_K, N_SURROGATES, SEED).round(6).to_csv(tables_dir / "modeled" / "family_test.csv", index=False)
save_npz(tables_dir / "modeled" / "surrogate_maxima.npz", "max", fam9_mod, "max_s", cor9_mod)

# Scatter grid of the modeled z(t) against SPI-q(t - k) for the windows SCATTER_Q and lags SCATTER_K: OLS R^2 and sign of the slope.
cells = scatter_pairs(spi_full, zm_mod, SCATTER_Q, SCATTER_K)
r2 = scatter_r2(cells, EXCLUDED_MONTHS)
scatter_r2_rows(r2, SCATTER_Q, SCATTER_K).round(6).to_csv(tables_dir / "modeled" / "scatter_r2_z_vs_spi.csv", index=False)

# Modeled figures.
fig_mod = figures_dir / "modeled"
years = f"{zm_mod.index.min().year}–{zm_mod.index.max().year}"
spi_hyp = spi[f"SPI_{HYPOTHESIS_Q}"]
lag0 = lag0_correlations(spi_hyp, zm_mod, EXCLUDED_MONTHS)   # descriptive, same month
level_anomaly_figure(an_mod, site_name, modeled=True).savefig(fig_mod / "level_anomalies.png", dpi=150)
monthly_series_figure(spi_hyp, zm_mod, x_mod, y_mod, dates_mod, cor_mod, fam_mod, lag0, HYPOTHESIS_Q, HYPOTHESIS_K,
                      EXCLUDED_MONTHS).savefig(fig_mod / f"series_{cell}_z_all_vs_no_jan-mar.png", dpi=150)
daily_series_figure(spi_hyp, an_mod["z"], cor_mod, fam_mod, lag0, HYPOTHESIS_Q, HYPOTHESIS_K).savefig(
    fig_mod / f"series_spi{HYPOTHESIS_Q}_k0_k{HYPOTHESIS_K}_daily_z_all_vs_no_jan-mar.png", dpi=150)
modeled_z_days = an_mod["z"].dropna()
rain_level_figure(rain_windows, modeled_level, (pd.Timestamp(modeled_z_days.index.min().year, 1, 1),
                                                pd.Timestamp(modeled_z_days.index.max().year, 11, 30)), HYPOTHESIS_K, "modeled",
                  "Modeled water table level: raw column `pred` converted to m (−pred/100; negative = below the ground),").savefig(
    fig_mod / f"rain_15d_90d_and_level_modeled_{modeled_z_days.index.min().year}-{modeled_z_days.index.max().year}.png", dpi=150)
correlation_matrix_figure(cor9_mod, fam9_mod, SPI_Q, LAGS_K, f"Correlation SPI-q(t−k) vs monthly z, modeled level ({years})").savefig(
    fig_mod / f"correlation_matrix_spi_z_{grid}.png", dpi=150)
single_cell_surrogate_figure(y_mod, dates_mod, cor_mod, fam_mod, surrogates_mod, HYPOTHESIS_Q, HYPOTHESIS_K).savefig(
    fig_mod / "single_cell_surrogate_process.png", dpi=150)
slope_scatter_figure(slopes_mod, HYPOTHESIS_Q, HYPOTHESIS_K, EXCLUDED_MONTHS, small_markers=True,
                     title=f"Modeled level: SPI-{HYPOTHESIS_Q}(t−{HYPOTHESIS_K}) vs modeled z(t)").savefig(
    fig_mod / f"slope_scatter_{cell}_all_vs_no_jan-mar.png", dpi=150)
scatter_grid_figure(cells, r2, SCATTER_Q, SCATTER_K, EXCLUDED_MONTHS, f"Modeled level: z(t) vs SPI-q(t−k) ({years})").savefig(
    fig_mod / f"scatter_z_vs_spi_q{'-'.join(map(str, SCATTER_Q))}_k{'-'.join(map(str, SCATTER_K))}.png", dpi=150)

# Summary.
print(f"\nobserved, cell SPI-{HYPOTHESIS_Q} k={HYPOTHESIS_K}: " + "; ".join(
    f"{s}: r = {c['r'][iq_h, ik_h]:+.3f} (n = {int(c['n'][iq_h, ik_h])}), Gao p = {c['p_gao'][iq_h, ik_h]:.4f}"
    for s, c in zip(SCENARIOS, cor_obs)))
for scenario, c in zip(SCENARIOS, cor_obs):
    for gen in GENERATORS:
        f = fam_obs[(scenario, gen)]
        print(f"observed family, {scenario}, {gen}: best SPI-{SPI_Q[f['iq']]} k={LAGS_K[f['ik']]}, r = {f['r_max']:+.3f}, "
              f"family p = {f['p_family']:.4f}, critical r = {f['r_crit95']:.3f}, Gao-Sidak p = {c['sidak'][f['iq']]['p_sidak']:.4f}")
print("observed OLS slope: " + "; ".join(f"{s}: {p['b_ols']:+.3f} [{p['ci_ols'][0]:+.3f}, {p['ci_ols'][1]:+.3f}]" for s, p in slopes_obs.items()))
print(f"modeled, cell SPI-{HYPOTHESIS_Q} k={HYPOTHESIS_K}: " + "; ".join(
    f"{s}: r = {c['r']:+.3f} (n = {c['n']}), Gao p = {c['p_gao']:.4f}" for s, c in zip(SCENARIOS, cor_mod)))
for scenario in SCENARIOS:
    print(f"modeled surrogate p, {scenario}: " + ", ".join(f"{g} {fam_mod[(scenario, g)]['p']:.4f}" for g in GENERATORS))
for scenario, c in zip(SCENARIOS, cor9_mod):
    for gen in GENERATORS:
        f = fam9_mod[(scenario, gen)]
        print(f"modeled family, {scenario}, {gen}: best SPI-{SPI_Q[f['iq']]} k={LAGS_K[f['ik']]}, r = {f['r_max']:+.3f}, "
              f"family p = {f['p_family']:.4f}, Gao-Sidak p = {c['sidak'][f['iq']]['p_sidak']:.4f}")
print("modeled OLS slope: " + "; ".join(f"{s}: {p['b_ols']:+.3f}" for s, p in slopes_mod.items()))
common = zm_mod.index.intersection(zm_obs.index)   # informative only: modeled vs observed monthly z in the common complete months
print(f"modeled vs observed monthly z (information only): r = "
      f"{float(np.corrcoef(zm_mod.loc[common], zm_obs.loc[common])[0, 1]):+.3f} over {len(common)} months")
