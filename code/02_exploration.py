"""Step 2: exploration of the remote variables against the water level, on the whole piezometric record.

Diagnostic only: it uses development + holdout (no cut of compute_partition) and does not depend on the scheme, so
its outputs go to outputs/<site>/02_exploration/. The decisions of the model (SPECTRAL, LEADS, WINDOWS, hv_h) are
still taken on the development of each scheme. Every analysis is run on the raw series and on the anomalies (series
minus its harmonic fit of order ANOMALY_HARMONICS_K on the piezometric period); outputs end in _raw or _anomaly.
"""
import numpy as np
import pandas as pd

import config as cfg
from functions.data import compute_indices, load_table, save_table
from functions.exploration import (
    REGIMES, SEASONS, acf_pacf, by_year_spearman, ccf_table, detrended_anomaly, harmonic_anomaly, harmonic_stability,
    hydro_year, jump_table, lead_summary, level_phase, period_means, phase_relation, proposed_leads, rain_window_response,
    redundancy_clusters, regime_check, regime_of, rise_events, season_label, season_year, slow_fast_table,
    spearman_signs, stratified_spearman)
from functions.features import compute_partition, modelling_days
from functions.plots import (
    acf_pacf_plot, annual_anomaly_plot, ccf_plot, corr_heatmap, date_scatter_plot, dendrogram_plot,
    harmonic_stability_plot, jumps_plot, lagged_scatter_plot, phase_relation_plot, rain_window_plot, scatter_by_season,
    september_plot, slow_series_plot, zscore_series_plot)

SITE = "sdh1"
K = cfg.ANOMALY_HARMONICS_K
hv_h = cfg.SITES[SITE]["hv_h"]   # hv_blocks gap of this site (checked against the ACF)
BOOT = {"block_days": cfg.BOOTSTRAP_BLOCK_DAYS, "n_boot": cfg.N_BOOTSTRAP, "level": cfg.BOOTSTRAP_LEVEL, "seed": cfg.SEED}

out_dir = cfg.OUTPUTS_DIR / SITE / "02_exploration"
out_dir.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(out_dir / f"{name}.png", dpi=130)


ORDER = {"regime": REGIMES, "season": SEASONS}


def pivot(table, keys, variables):
    """rho and significant of a stratified_spearman table as variables x strata (strata joined with |, ordered by
    regime, season, then year / month; strata without any rho are left out)."""
    t = table.dropna(subset=keys).copy()
    sort_keys = [t[k].map({v: i for i, v in enumerate(ORDER[k])}) if k in ORDER else t[k].astype(int) for k in keys]
    t = t.assign(**{f"_s{i}": s for i, s in enumerate(sort_keys)}).sort_values([f"_s{i}" for i in range(len(keys))])
    t["stratum"] = t[keys].astype(str).agg("|".join, axis=1)
    has_rho = t.groupby("stratum")["rho"].apply(lambda r: r.notna().any())
    columns = [c for c in dict.fromkeys(t["stratum"]) if has_rho[c]]
    rho = t.pivot(index="variable", columns="stratum", values="rho").reindex(index=variables, columns=columns)
    sig = t.pivot(index="variable", columns="stratum", values="significant").reindex(index=variables, columns=columns)
    return rho, sig.astype("boolean").fillna(False)


# 1. Whole clean daily table; the analysis period is the piezometric one (first to last day with a water level),
#    the same for the water level and the remote variables. Variables: the spectral indices and the I bands.
full = load_table(cfg.INTERIM_DIR / f"{SITE}_daily.csv")
X_full = compute_indices(full, cfg.BANDS).join(full[list(cfg.BANDS.values())])
period = slice(full["gw_cm"].first_valid_index(), full["gw_cm"].last_valid_index())
daily = full.loc[period]
y, X, precip = daily["gw_cm"], X_full.loc[period], daily["precip_mm"]
variables = list(X.columns)
common = y.index[y.notna() & X.notna().all(axis=1)]   # days with the level and every variable

# 2. Anomalies: series minus its harmonic fit of order K, fitted on the common days (the same days for all series)
#    and evaluated on every day of the period.
X_anom, y_anom = harmonic_anomaly(X, K, common), harmonic_anomaly(y, K, common)
versions = {"raw": (X, y), "anomaly": (X_anom, y_anom)}

# Labels of each day; the regime always comes from the raw depth.
regime = regime_of(y, cfg.REGIME_BINS_CM)
labels = pd.DataFrame({"season": season_label(y.index), "month": y.index.month, "year": season_year(y.index).astype(str),
                       "regime": regime}, index=y.index)
phase, rate = level_phase(y, cfg.PHASE_SMOOTH_DAYS, cfg.PHASE_MIN_RATE_CM_DAY)

results = {}
for kind, (Xv, yv) in versions.items():
    tag = f"_{kind}"
    name = "raw series" if kind == "raw" else f"anomalies (harmonic K = {K})"
    r = results[kind] = {}

    # 3. Sign of the relation with the level and its value above / below the surface (regime of the raw depth).
    r["signs"] = spearman_signs(Xv, yv, cfg.MIN_PAIRS)
    r["regimes"] = regime_check(Xv, yv, cfg.REGIME_THRESHOLD_CM, cfg.MIN_PAIRS, depth=y)

    # 4. Lead time: CCF of every variable with the level (x at t + k against the level at t) and its summary.
    r["ccf"] = ccf_table(Xv, yv, cfg.CCF_KS, cfg.MIN_PAIRS)
    r["leads"] = lead_summary(r["ccf"])

    # 5. Rain accumulation windows against the moisture index (raw or anomaly). Rain is CHIRPS as is.
    r["rain"] = rain_window_response(precip, Xv[cfg.RAIN_MOISTURE_INDEX], cfg.RAIN_WINDOWS, cfg.MIN_PAIRS)

    # 6. Redundancy between variables (the level is not used), on the common days.
    tree, r["clusters"] = redundancy_clusters(Xv.loc[common], cfg.REDUNDANCY_THRESHOLD)

    # 7. Memory of the level: ACF / PACF (to check hv_h).
    r["acf"] = acf_pacf(yv, cfg.ACF_MAX_LAG)

    # 8. Spearman by season and by month, with block-bootstrap intervals.
    r["season"] = stratified_spearman(Xv, yv, labels[["season"]], cfg.MIN_PAIRS, **BOOT)
    r["month"] = stratified_spearman(Xv, yv, labels[["month"]], cfg.MIN_PAIRS, **BOOT)

    # 9. Spearman by regime (raw depth), regime x season, regime x year and regime x season x year.
    levels = {"regime": ["regime"], "regime_season": ["regime", "season"], "regime_year": ["regime", "year"],
              "regime_season_year": ["regime", "season", "year"]}
    r["regime_strata"] = pd.concat(
        [stratified_spearman(Xv, yv, labels[keys], cfg.MIN_PAIRS, **BOOT).assign(level=level)
         for level, keys in levels.items()], ignore_index=True)

    # 10. Relation in the rise and in the fall of the level (phase from the smoothed raw depth).
    r["phase"] = phase_relation(Xv, yv, phase, cfg.MIN_PAIRS, cfg.BOOTSTRAP_BLOCK_DAYS, cfg.N_BOOTSTRAP,
                                cfg.PHASE_SLOPE_N_BOOTSTRAP, cfg.BOOTSTRAP_LEVEL, cfg.SEED)

    # 11. Slow (rolling median) and fast parts.
    r["slow_fast"], slow, _ = slow_fast_table(yv, Xv, cfg.SLOW_WINDOW_DAYS, cfg.MIN_PAIRS)

    # 12. September of each year.
    september = yv.index[yv.index.month == 9]
    sept_frame = pd.concat([yv, Xv[cfg.SEPTEMBER_VARIABLES]], axis=1).loc[september]
    r["september"] = by_year_spearman(Xv.loc[september, cfg.SEPTEMBER_VARIABLES], yv.loc[september],
                                      pd.Series(september.year, index=september), cfg.SEPTEMBER_MIN_PAIRS)

    # Tables.
    for table_name, table in {"signs": r["signs"], "regimes": r["regimes"], "ccf": r["ccf"], "leads": r["leads"],
                              "rain_windows": r["rain"], "clusters": r["clusters"], "acf_pacf": r["acf"],
                              "corr_season": r["season"], "corr_month": r["month"], "corr_regime": r["regime_strata"],
                              "phase_relation": r["phase"], "slow_fast": r["slow_fast"],
                              "september_rho": r["september"]}.items():
        save_table(table, out_dir / f"{table_name}{tag}.csv")

    # Figures.
    on_common = Xv.loc[common], yv.loc[common]
    save_fig(scatter_by_season(*on_common, f"{SITE}: variables vs water level, {name}", ncols=5), f"scatter_by_season{tag}")
    save_fig(lagged_scatter_plot(Xv, yv, r["leads"], f"{SITE}: variable at t + k (k of the largest |CCF|) vs water level at t, {name}"),
             f"scatter_lagged{tag}")
    save_fig(date_scatter_plot(*on_common, cfg.TRAJECTORY_DAYS,
                               f"{SITE}: variables vs water level colored by date, {cfg.TRAJECTORY_DAYS}-day trajectory, {name}"),
             f"scatter_by_date{tag}")
    save_fig(zscore_series_plot(Xv, yv, f"{SITE}: variables and water level (z-scores), {name}"), f"zscore_series{tag}")
    save_fig(ccf_plot(r["ccf"], r["leads"], f"{SITE}: CCF with the water level, {name}"), f"ccf{tag}")
    save_fig(dendrogram_plot(tree, variables, list(r["clusters"].index[r["clusters"]["proposed"]]), cfg.REDUNDANCY_THRESHOLD,
                             f"{SITE}: redundancy ({name}), threshold {cfg.REDUNDANCY_THRESHOLD}"), f"dendrogram{tag}")
    save_fig(rain_window_plot(r["rain"], f"{SITE}: rain window vs {cfg.RAIN_MOISTURE_INDEX} ({kind})"), f"rain_windows{tag}")
    save_fig(acf_pacf_plot({kind: r["acf"]}, hv_h, f"{SITE}: ACF and PACF of the depth, {name}"), f"acf_pacf{tag}")
    save_fig(corr_heatmap({"season": pivot(r["season"], ["season"], variables), "month": pivot(r["month"], ["month"], variables)},
                          f"{SITE}: Spearman with the water level by season and month, {name}"), f"corr_season_month{tag}")
    rs = r["regime_strata"]
    save_fig(corr_heatmap({lvl: pivot(rs[rs["level"] == lvl], keys, variables) for lvl, keys in levels.items()
                           if lvl != "regime_season_year"},
                          f"{SITE}: Spearman with the water level by regime (raw depth), season and year, {name}"),
             f"corr_regime{tag}")
    save_fig(phase_relation_plot(r["phase"], f"{SITE}: Spearman in the rise and fall of the level, {name}"), f"phase_relation{tag}")
    save_fig(slow_series_plot(pd.concat([yv, Xv], axis=1), slow,
                              f"{SITE}: series and slow part ({cfg.SLOW_WINDOW_DAYS}-day rolling median), {name}"),
             f"slow_series{tag}")
    save_fig(september_plot(sept_frame, pd.Series(september.year, index=september), r["september"],
                            f"{SITE}: September of each year, {name}"), f"september{tag}")

# 13. Stability of the harmonic curve: first year vs the rest, and development of hold_1y vs the whole record.
partition = compute_partition(modelling_days(full, cfg.BANDS.values(), bool(cfg.DECISIONS[SITE]["WINDOWS"])),
                              cfg.PARTITIONS["hold_1y"], cfg.GAP)
frame = pd.concat([y, X], axis=1).loc[common]
first = np.where(frame.index < frame.index[0] + pd.Timedelta(days=365), "first", "rest")
dev = np.where(frame.index <= partition.dev_end, "dev", "hold")
period_label = pd.Series(first, index=frame.index) + "|" + pd.Series(dev, index=frame.index)
present = list(pd.unique(period_label))
comparisons = {"first_year_vs_rest": ([p for p in present if p.startswith("first")], [p for p in present if p.startswith("rest")]),
               "dev_hold_1y_vs_full": ([p for p in present if p.endswith("dev")], present)}
stability, curves = harmonic_stability(frame, period_label, comparisons, K, **BOOT)
save_table(stability, out_dir / "harmonic_stability.csv")
for comp, names in {"first_year_vs_rest": ("first year", "rest"),
                    "dev_hold_1y_vs_full": (f"development of hold_1y (to {partition.dev_end.date()})", "whole record")}.items():
    save_fig(harmonic_stability_plot(curves, stability, comp, names, f"{SITE}: harmonic curve K = {K}, {names[0]} vs {names[1]}"),
             f"harmonic_stability_{comp}")

# 14. Jumps of the piezometer and apparent specific yield of the rise events (raw depth).
max_jump = cfg.SITES[SITE]["max_jump_cm"]
jumps = jump_table(y, precip, max_jump, cfg.JUMP_RECOVERY_DAYS, cfg.JUMP_RECOVERY_FRACTION)
events = rise_events(y, precip, cfg.EVENT_MIN_RISE_CM, cfg.EVENT_MIN_RAIN_MM, cfg.EVENT_RAIN_LEAD_DAYS)
save_table(jumps, out_dir / "jumps.csv")
save_table(events, out_dir / "rise_events.csv")
save_fig(jumps_plot(y, precip, jumps, f"{SITE}: day-to-day jumps > {max_jump} cm and daily rain"), "jumps")

# 15. Yearly and seasonal anomalies of I3 and mlswi26 against the 2012-2026 climatology, after removing a linear
#     trend (fitted jointly with the harmonics on the whole satellite record), and the anomaly of the level.
periods = {"hydro_year": lambda idx: hydro_year(idx, cfg.HYDRO_YEAR_START_MONTH).astype(str),
           "season": lambda idx: season_year(idx).astype(str) + "-" + season_label(idx)}
rows, trends = [], {}
for var in cfg.ANNUAL_ANOMALY_VARIABLES:
    anom, trends[var] = detrended_anomaly(X_full[var], K)
    piezo_mean = anom.loc[common].mean()   # mean anomaly of the piezometric period, to recenter
    for ptype, label_of in periods.items():
        means = period_means(anom, label_of(anom.index), **BOOT)
        rows.append(means.assign(variable=var, period_type=ptype, piezo_recentered=means["mean"] - piezo_mean))
for ptype, label_of in periods.items():
    rows.append(period_means(y_anom, label_of(y_anom.index), **BOOT).assign(variable="gw_cm_anomaly", period_type=ptype))
annual = pd.concat(rows, ignore_index=True)
annual = annual.assign(_order=annual["period"].map(lambda p: (int(p[:4]), SEASONS.index(p[5:]) if len(p) > 4 else 0)))
annual = annual.sort_values(["period_type", "_order"], kind="stable").drop(columns="_order").reset_index(drop=True)
annual["trend_per_year"] = annual["variable"].map(trends)
save_table(annual, out_dir / "annual_anomalies.csv")
level_periods = set(annual.loc[annual["variable"] == "gw_cm_anomaly", "period"])
compare = []
for ptype in periods:
    part = annual[annual["period_type"] == ptype]
    lvl = part[part["variable"] == "gw_cm_anomaly"].set_index("period")
    for var in cfg.ANNUAL_ANOMALY_VARIABLES:
        v = part[part["variable"] == var].set_index("period").reindex(lvl.index)
        compare.append(pd.DataFrame({"period_type": ptype, "variable": var, "period": lvl.index,
                                     "level_mean": lvl["mean"], "level_lo": lvl["mean_lo"], "level_hi": lvl["mean_hi"],
                                     "level_n": lvl["n"], "var_mean": v["mean"], "var_lo": v["mean_lo"], "var_hi": v["mean_hi"],
                                     "var_recentered": v["piezo_recentered"]}))
compare = pd.concat(compare, ignore_index=True)
save_table(compare, out_dir / "annual_anomalies_vs_level.csv")
for ptype in periods:
    shown = annual[(annual["period_type"] == ptype)]
    save_fig(annual_anomaly_plot(shown, f"{SITE}: mean anomaly by {ptype} (indices: vs detrended 2012-2026 climatology; "
                                        f"level: vs its harmonic fit), {cfg.BOOTSTRAP_LEVEL:.0%} block-bootstrap interval"),
             f"annual_anomalies_{ptype}")

# 16. Summary.
lines = [f"{SITE}: piezometric period {y.first_valid_index().date()} to {y.last_valid_index().date()}, "
         f"{y.notna().sum()} days with level, {len(common)} common days; anomalies: harmonic K = {K}"]
for kind, r in results.items():
    s, leads = r["signs"], r["leads"]
    lines += [f"\n--- {kind} ---",
              "signs (rho): " + ", ".join(f"{c} {s.loc[c, 'rho']:+.2f}" for c in variables),
              f"leads: k_max (rho) " + ", ".join(f"{c} {int(leads.loc[c, 'k_max']):+d} ({leads.loc[c, 'rho_max']:+.2f})"
                                                + ("*" if leads.loc[c, "at_edge"] else "") for c in leads.index)
              + "   (* = on the edge of CCF_KS)",
              f"proposed LEADS (best k >= 1 of the 2 variables with the largest |rho| there): {proposed_leads(leads, variables)}",
              f"rain windows: best |rho| at w = {r['rain']['rho'].abs().idxmax()} (rho {r['rain']['rho'].loc[r['rain']['rho'].abs().idxmax()]:+.2f})"]
    acf = r["acf"]
    inside = acf.index[(acf["acf"].abs() < acf["conf"]) & (acf.index > 0)]
    lines.append(f"ACF: first lag inside the 95% band = {inside.min() if len(inside) else '>' + str(cfg.ACF_MAX_LAG)}; "
                 f"ACF at hv_h = {hv_h}: {acf['acf'].loc[hv_h]:.3f}")
    season = r["season"]
    lines.append("variables with interval excluding 0, by season: " + ", ".join(
        f"{sn} {int(season[(season['season'] == sn)]['significant'].sum())}" for sn in SEASONS))
    ph = r["phase"]
    diff = ph[ph["phase"] == "fall - rise"]
    lines.append(f"rise / fall: median |rho| rise {ph[ph['phase'] == 'rise']['rho'].abs().median():.2f}, "
                 f"fall {ph[ph['phase'] == 'fall']['rho'].abs().median():.2f}; "
                 f"fall - rise significant in {int(diff['significant'].sum())} of {len(diff)} variables")
    sf = r["slow_fast"]
    lines.append(f"fast share of the level variance {sf.loc[y.name, 'frac_fast']:.2f}; median |rho| slow "
                 f"{sf['rho_slow'].abs().median():.2f}, fast {sf['rho_fast'].abs().median():.2f}")
    lines.append("clusters (* = proposed): " + " | ".join(
        r["clusters"].assign(c=[f"{c}*" if p else c for c, p in zip(r["clusters"].index, r["clusters"]["proposed"])])
        .groupby("cluster")["c"].apply(", ".join)))
lines.append("\nharmonic stability (shift of the curve, interval, share of significant days):")
for _, row in stability[stability["variable"].isin(["gw_cm", "mlswi26", "str1", "I3"])].iterrows():
    lines.append(f"  {row['comparison']}: {row['variable']} {row['shift']:+.3g} [{row['shift_lo']:+.3g}, {row['shift_hi']:+.3g}], "
                 f"{row['frac_days_significant']:.0%}")
lines.append(f"\njumps > {max_jump} cm: {len(jumps)} (" + (f"{int((jumps['direction'] == 'rise').sum())} rise, "
             f"{int((~jumps['recovered']).sum())} not recovered, {int((jumps['abs_jump_cm'] > 20).sum())} > 20 cm" if len(jumps) else "") + ")")
if len(events):
    soil = events[~events["ponded"]]
    lines.append(f"rise events: {len(events)} ({len(soil)} not ponded); apparent Sy median "
                 f"{soil['sy_apparent'].median():.3f} (IQR {soil['sy_apparent'].quantile(0.25):.3f} to {soil['sy_apparent'].quantile(0.75):.3f})")
lines.append("annual anomalies (hydrological year, mean [interval]): " + "; ".join(
    f"{row['variable']} {row['period']}: {row['mean']:+.3g} [{row['mean_lo']:+.3g}, {row['mean_hi']:+.3g}]"
    for _, row in annual[(annual["period_type"] == "hydro_year") & annual["period"].isin(level_periods)].iterrows()))
summary = "\n".join(lines)
(out_dir / "summary.txt").write_text(summary + "\n")
print(summary)
