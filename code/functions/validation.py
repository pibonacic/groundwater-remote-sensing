"""Validation of the modeled water table against independent data: Collahuasi wells (monthly series, seasonal cycles,
cross-correlations, anomalies) and CHIRPS rain (accumulations, Standardized Precipitation Index, monthly level z-score,
lagged correlations with Gao et al. p-values, surrogate-series tests and slopes)."""
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, kstest, norm, spearmanr
from scipy.stats import t as t_dist


def zscore(s):
    return (s - s.mean()) / s.std()


# altitude (m a.s.l.), rounded to 1e-6 m so float noise does not break ties that depth has (ranks must match)
def wells_altitude(depth_cm, well_altitude):
    """Altitude of the water table of each well (columns of depth_cm, cm); wells without surface altitude are NaN."""
    return (pd.Series(well_altitude).reindex(depth_cm.columns) - depth_cm / 100).round(6)


def site_altitude(depth_cm, surface):
    """Altitude of the water table of a site with surface altitude surface (m a.s.l.)."""
    return (surface - depth_cm / 100).round(6)


def load_wells(path, unit_to_cm):
    """Long table of the wells (BHID, DATE, WATER_LEVEL) with depth_cm (positive below the surface) and the month
    (labeled by its first day)."""
    wells = pd.read_csv(path, usecols=["BHID", "DATE", "WATER_LEVEL"], parse_dates=["DATE"])
    wells["depth_cm"] = wells["WATER_LEVEL"] * unit_to_cm
    wells["month"] = wells["DATE"].dt.to_period("M").dt.to_timestamp()
    return wells


def monthly_median(wells):
    """Monthly median depth of each well (months x wells)."""
    return wells.dropna(subset=["depth_cm"]).pivot_table(index="month", columns="BHID", values="depth_cm", aggfunc="median")


def spearman_table(pred_table, wells_table, stats, min_n):
    """Spearman, for each well and statistic, on the months where both have a value (z-scores do not change ranks)."""
    rows = []
    for well in wells_table:
        for stat in stats:
            pair = pd.concat([pred_table[stat], wells_table[well]], axis=1).dropna()
            valid = len(pair) >= min_n and pair.nunique().min() > 1   # a constant well (always flowing) has no rank
            rho, p = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]) if valid else (float("nan"), float("nan"))
            rows.append({"well": well, "prediction": stat, "n_months": len(pair), "rho": rho, "p": p,
                         "first_month": pair.index.min(), "last_month": pair.index.max()})
    return pd.DataFrame(rows, columns=["well", "prediction", "n_months", "rho", "p", "first_month", "last_month"])


def climatology(series, z_max, min_months):
    """Median of each calendar month (1..12): monthly values with |z-score| above z_max are left out and a calendar
    month needs at least min_months distinct year-months."""
    kept = series[zscore(series).abs() <= z_max]
    by_calendar_month = kept.groupby(kept.index.month)
    median = by_calendar_month.median().round(6)   # float noise would break ties between months
    return median.where(by_calendar_month.size() >= min_months).reindex(range(1, 13))


def cycle_table(pred_table, wells_table, stats, z_max, min_months):
    """Seasonal cycle (climatology) of each modeled statistic (columns pred_<stat>) and of each well."""
    table = pd.concat({f"pred_{stat}": climatology(pred_table[stat], z_max, min_months) for stat in stats}
                      | {well: climatology(wells_table[well].dropna(), z_max, min_months) for well in wells_table}, axis=1)
    return table.rename_axis("month")


def cycle_spearman(table, wells_list, stats):
    """Spearman of the seasonal cycle of each well in wells_list with that of each modeled statistic (over the calendar
    months where both have a value); every series also gets the month of its deepest and shallowest median."""
    rows = []
    for name in [f"pred_{s}" for s in stats] + wells_list:
        s = table[name].dropna()
        row = {"series": name, "n_months": len(s), "month_max_depth": s.idxmax(), "month_min_depth": s.idxmin()}
        if name in wells_list:
            for stat in stats:
                pair = table[[f"pred_{stat}", name]].dropna()
                row[f"rho_{stat}"], row[f"p_{stat}"] = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
        rows.append(row)
    columns = ["series", "n_months", "month_max_depth", "month_min_depth"] + [f"{v}_{s}" for s in stats for v in ("rho", "p")]
    return pd.DataFrame(rows).reindex(columns=columns)


def well_matrix(table, wells_list):
    """Spearman rho and p between every pair of wells (columns of table), each pair on the rows where both have a value."""
    rho_m = pd.DataFrame(index=wells_list, columns=wells_list, dtype=float)
    p_m = rho_m.copy()
    for a in wells_list:
        for b in wells_list:
            pair = table[[a, b]].dropna()
            rho_m.loc[a, b], p_m.loc[a, b] = spearmanr(pair[a], pair[b]) if a != b else (1.0, 0.0)
    return rho_m.rename_axis("well"), p_m.rename_axis("well")


def cross_correlation(pred_table, wells_table, shift, lags, stats, min_n):
    """Spearman of each well with the modeled statistics shifted by every lag (shift(pred_table, lag)); rows also flag
    the lag of largest |rho| of each well and statistic (best_lag)."""
    table = pd.concat({lag: spearman_table(shift(pred_table, lag), wells_table, stats, min_n) for lag in lags},
                      names=["lag_months"]).reset_index("lag_months").reset_index(drop=True)
    table = table[["well", "prediction", "lag_months", "n_months", "rho", "p"]]
    best = table.dropna(subset=["rho"]).groupby(["well", "prediction"])["rho"].transform(lambda r: r.abs() == r.abs().max())
    table["best_lag"] = best.reindex(table.index).fillna(False).astype(bool)
    return table


def anomalies(monthly, cycle_tab):
    """Each monthly value minus the seasonal cycle of its series for that calendar month."""
    baseline = cycle_tab.reindex(columns=monthly.columns).reindex(monthly.index.month).set_axis(monthly.index)
    return monthly - baseline


def load_manual(path):
    """Manual measurements of the user (depth from the ground, suelo_a_agua_cm; negative: water above the ground) as
    date, well, depth_cm. A measurement is labeled by its Collahuasi id, or by the field id when the well has none
    or is a second tube ("(2)")."""
    manual = pd.read_excel(path).dropna(subset=["suelo_a_agua_cm"])
    return pd.DataFrame({
        "date": manual["fecha"],
        "well": manual["id_collahuasi"].where((manual["id_collahuasi"] != "no_id") & ~manual["id"].str.contains(r"\("),
                                              manual["id"]),
        "depth_cm": manual["suelo_a_agua_cm"],
    })


# ---------------------------------------------------------------- rain: loading and accumulation
def load_precip(path):
    """Daily CHIRPS rain (mm) as a Series named precip_mm with a complete daily index."""
    return (pd.read_csv(path, usecols=["date", "daily_pp"])
            .assign(date=lambda d: pd.to_datetime(d["date"], format="%Y-%m-%d"))
            .set_index("date")["daily_pp"].rename("precip_mm").sort_index().asfreq("D"))


def accumulate_daily(precip, windows):
    """Rain accumulated over the trailing windows (label -> days), ending on each day, only complete windows."""
    return pd.DataFrame({f"precip_{w}_mm": precip.rolling(d, min_periods=d).sum() for w, d in windows.items()})


# ---------------------------------------------------------------- labels shared by the rain tables and figures
SCENARIOS = ("all months", "without Jan–Mar")   # the month of the level z (t) is dropped in the second one
GENERATORS = ("AR(1)", "FFT phase")             # surrogate series generators


def scenario_masks(dates, excluded_months):
    """Boolean mask of the months kept by each scenario (SCENARIOS -> array over dates)."""
    return {SCENARIOS[0]: np.ones(len(dates), bool), SCENARIOS[1]: ~np.isin(dates.month, excluded_months)}


# ---------------------------------------------------------------- C1. SPI of the monthly rain
def monthly_rain(daily):
    """Monthly rain totals (mm), month-end index; a month with any missing day is NaN."""
    res = daily.resample("ME")
    count, total = res.count(), res.sum()
    monthly = total.where(count == count.index.days_in_month)
    return monthly.reindex(pd.date_range(count.index.min(), count.index.max(), freq="ME")).rename("rain_mm")


def spi_month(monthly, q, ref_years=(2002, 2025), zero_mm=0.1, clip=3.0, min_nonzero=10):
    """SPI-q of a monthly rain series, one gamma per calendar month (maximum likelihood, location 0) fitted on the
    non-zero values (> zero_mm) of the reference years, plus the mass at zero (Stagge et al. 2015): with n reference
    values, n0 zeros and G the gamma cdf, p0 = n0 / (n + 1), H = (n0 + 1) / (2 (n + 1)) for a zero accumulation and
    H = p0 + (1 - p0) G(x) otherwise; SPI = normal quantile of H, clipped to +-clip. A month with fewer than
    min_nonzero non-zero values has no fit and no SPI. Returns (SPI Series, list of parameter rows)."""
    acc = monthly.rolling(q, min_periods=q).sum()   # accumulation over the q trailing months
    spi = pd.Series(np.nan, index=monthly.index, name=f"SPI_{q}")
    y0, y1 = ref_years
    rows = []
    for m in range(1, 13):
        in_month = acc.index.month == m
        in_ref = in_month & (acc.index.year >= y0) & (acc.index.year <= y1) & acc.notna().to_numpy()
        values = acc[in_ref].to_numpy()
        n, n0 = len(values), int((values <= zero_mm).sum())
        nonzero = values[values > zero_mm]
        row = {"q": q, "month": m, "n_ref": n, "n_zero": n0, "alpha": np.nan, "beta": np.nan, "ks_p": np.nan,
               "spi_zero": np.nan, "method": "no fit"}
        if len(nonzero) < min_nonzero:
            rows.append(row)
            continue
        a, _, b = gamma_dist.fit(nonzero, floc=0)
        row.update(alpha=float(a), beta=float(b), method="gamma", ks_p=float(kstest(nonzero, gamma_dist(a, 0, b).cdf).pvalue))
        p0, h0 = n0 / (n + 1), (n0 + 1) / (2 * (n + 1))
        if n0 > 0:
            row["spi_zero"] = float(np.clip(norm.ppf(h0), -clip, clip))
        x = acc[in_month & acc.notna().to_numpy()]
        g = gamma_dist.cdf(x.to_numpy(), a, 0, b)
        h = g if n0 == 0 else np.where(x.to_numpy() > zero_mm, p0 + (1 - p0) * g, h0)
        spi[x.index] = np.clip(norm.ppf(h), -clip, clip)
        rows.append(row)
    return spi, rows


def spi_table(monthly, qs, ref_years=(2002, 2025), zero_mm=0.1, clip=3.0, min_nonzero=10):
    """SPI of every window q in qs: (DataFrame rain_mm + SPI_q, DataFrame of the gamma parameters of every q and month)."""
    columns, rows = {"rain_mm": monthly}, []
    for q in qs:
        columns[f"SPI_{q}"], q_rows = spi_month(monthly, q, ref_years, zero_mm, clip, min_nonzero)
        rows += q_rows
    return pd.DataFrame(columns), pd.DataFrame(rows)


# ---------------------------------------------------------------- C2. monthly z-score of the water table level
ABS_TO_SD = np.sqrt(np.pi / 2)   # standard deviation of a normal variable from the mean of its absolute value


def design_matrix(t, k, period_days=365.25):
    """Columns 1, cos(2 pi j t / P), sin(2 pi j t / P) for j = 1..k (t in days)."""
    columns = [np.ones_like(t)]
    for j in range(1, k + 1):
        w = 2 * np.pi * j * t / period_days
        columns += [np.cos(w), np.sin(w)]
    return np.column_stack(columns)


def fit_harmonic(t, y, k, k_sigma, period_days=365.25, sigma_floor=0.1):
    """Seasonal mean curve of y (least squares on k harmonics, no trend) and seasonal scale of its residual: sigma is
    sqrt(pi / 2) times the harmonic fit (k_sigma harmonics) of the absolute residual, never below sigma_floor times
    the global standard deviation of the residual."""
    coef = np.linalg.lstsq(design_matrix(t, k, period_days), y, rcond=None)[0]
    resid = y - design_matrix(t, k, period_days) @ coef
    sigma_global = float(np.std(resid, ddof=1))
    coef_sigma = np.linalg.lstsq(design_matrix(t, k_sigma, period_days), np.abs(resid), rcond=None)[0]
    return {"k": k, "k_sigma": k_sigma, "period_days": period_days, "coef": coef, "coef_sigma": coef_sigma,
            "sigma_global": sigma_global, "floor": sigma_floor * sigma_global}


def predict_harmonic(model, t):
    """(seasonal curve, seasonal sigma, number of days with sigma at the floor) of a fit_harmonic model at days t."""
    curve = design_matrix(t, model["k"], model["period_days"]) @ model["coef"]
    raw = ABS_TO_SD * (design_matrix(t, model["k_sigma"], model["period_days"]) @ model["coef_sigma"])
    return curve, np.maximum(raw, model["floor"]), int((raw < model["floor"]).sum())


def aggregate(s, min_coverage):
    """Monthly mean (month-end index) of a daily series; months whose coverage (days with a value over the days of the
    calendar month, also at the ends) is below min_coverage are dropped. Returns (kept, dropped) tables."""
    res = s.resample("ME")
    out = pd.DataFrame({"mean": res.mean(), "n_days": res.count()})
    out["coverage"] = out["n_days"] / out.index.days_in_month
    keep = (out["coverage"] >= min_coverage - 1e-12) & out["mean"].notna()
    return out[keep].copy(), out[~keep].copy()


def analyze_anomalies(daily, k=3, k_sigma=3, period_days=365.25, sigma_floor=0.1, min_coverage=0.8):
    """Standardized anomaly of a daily level series (m): z = (level - seasonal curve) / seasonal sigma (fit_harmonic),
    and its monthly mean on the months with enough coverage. Returns a dict: table (daily level, curve, anomaly,
    sigma, z), z (daily), zm (monthly z), zm_table, model, sigma_global, n_floor, sd_z (by calendar month), n, n_eff
    (effective size from the lag-1 autocorrelation of the anomaly), first, last."""
    idx = daily.index
    t = (idx - idx[0]).days.to_numpy().astype(float)
    y = daily.to_numpy()
    model = fit_harmonic(t, y, k, k_sigma, period_days, sigma_floor)
    curve, sigma, n_floor = predict_harmonic(model, t)
    resid = y - curve
    z = pd.Series(resid / sigma, index=idx)
    table = pd.DataFrame({"level_m": y, "seasonal_curve_m": curve, "anomaly_m": resid, "seasonal_sigma_m": sigma,
                          "z": z.to_numpy()}, index=idx)
    full = pd.Series(resid, index=idx).reindex(pd.date_range(idx.min(), idx.max(), freq="D"))
    r1 = float(full.autocorr(lag=1))
    zm_table, _ = aggregate(z, min_coverage)
    return {"table": table, "z": z, "zm": zm_table["mean"].rename("z_monthly"), "zm_table": zm_table, "model": model,
            "sigma_global": model["sigma_global"], "n_floor": n_floor, "sd_z": z.groupby(idx.month).std(),
            "n": len(y), "n_eff": len(y) * (1 - r1) / (1 + r1), "first": idx.min(), "last": idx.max()}


def contiguous_window(zm, index):
    """Monthly z on the SPI index, cut to its first and last month with a value: (values, month-end dates, slice)."""
    y = zm.reindex(index).to_numpy(float)
    ok = np.flatnonzero(np.isfinite(y))
    window = slice(ok[0], ok[-1] + 1)
    return y[window], index[window], window


def lagged_grid(spi, qs, ks, window):
    """Array (len(qs), len(ks), T): SPI_q(t - k) on the months of window (a slice of the SPI index)."""
    return np.stack([np.stack([spi[f"SPI_{q}"].shift(k).to_numpy()[window] for k in ks]) for q in qs])


# ---------------------------------------------------------------- correlations and Gao et al. (2024) p-value
def _corr_block(X, y2):
    """Pearson r and pair count of X[iq, ik, :] against every row of y2 (S, T), NaN pairs masked."""
    mask = np.isfinite(X)[None] & np.isfinite(y2)[:, None, None, :]
    xz = np.where(np.isfinite(X), X, 0.0)[None]
    yz = np.where(np.isfinite(y2), y2, 0.0)[:, None, None, :]
    n = mask.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = (xz * mask).sum(-1) / n
        my = (yz * mask).sum(-1) / n
        dx = (xz - mx[..., None]) * mask
        dy = (yz - my[..., None]) * mask
        r = (dx * dy).sum(-1) / np.sqrt((dx ** 2).sum(-1) * (dy ** 2).sum(-1))
    return np.where(n >= 3, r, np.nan), n


def corr_grid(X, y, chunk=2000):
    """Pearson r of X[iq, ik, :] against y with pairwise NaN masking. y is (T,) or (S, T) (surrogates, correlated chunk
    rows at a time to bound memory). Returns (R, N) of shape (nq, nk) or (S, nq, nk); r is NaN where n < 3 or a
    variance is zero."""
    y2 = np.atleast_2d(y)
    blocks = [_corr_block(X, y2[i:i + chunk]) for i in range(0, len(y2), chunk)]
    r, n = np.concatenate([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks])
    return (r[0], n[0]) if y.ndim == 1 else (r, n)


def biased_acf(x, nlag):
    """Sample autocorrelation (biased, divides by N) of x without its NaN, lags 0..nlag."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    d = x - x.mean()
    c = np.array([d[:n - k] @ d[k:] / n if k < n else 0.0 for k in range(nlag + 1)])
    return c / c[0]


def gao_delta(n, rho_x, rho_y, M):
    """Reduction factor of Gao et al. (2024), [1 + 2 sum_{k=1..M} ((n - k) / n) rho_xx(k) rho_yy(k)]^-1; 1 when the
    denominator is <= 0; limited to [3 / n, 1]."""
    ks = np.arange(1, M + 1)
    den = 1 + 2 * np.sum(((n - ks) / n) * rho_x[ks] * rho_y[ks])
    delta = 1.0 if den <= 0 else 1.0 / den
    return float(min(1.0, max(delta, 3.0 / n)))


def p_one_sided(r, nu):
    """One-sided p (H1: r > 0) of a correlation r with nu degrees of freedom (Student t)."""
    if not (np.isfinite(r) and nu > 0):
        return np.nan
    if abs(r) >= 1:
        return 0.0
    return float(t_dist.sf(r * np.sqrt(nu / (1 - r ** 2)), nu))


def r_critical(nu, alpha=0.05):
    """One-sided critical r of a single cell at level alpha: t / sqrt(nu + t^2), t the 1 - alpha Student quantile."""
    t_crit = float(t_dist.ppf(1 - alpha, nu))
    return t_crit / np.sqrt(nu + t_crit ** 2)


def correlations(sx, yw, acf_x, acf_y, keep):
    """Pearson r, pair count and Gao p of every cell (q, k) of sx for the months kept by the mask keep, with the Sidak
    correction over the lags of each q (its lowest-p cell, n_i = max(1, nk * delta)). acf_x: SPI series of every q
    (whole record); acf_y: z over its whole contiguous window (before dropping months). Returns a dict of arrays
    r, n, delta, nu, M, p_gao (nq, nk) and the list sidak (ik_pmin, n_i, p_sidak per q)."""
    nq, nk = sx.shape[:2]
    r, n = corr_grid(sx, np.where(keep, yw, np.nan))
    delta, nu, M, p_gao = (np.full(r.shape, np.nan) for _ in range(4))
    for iq in range(nq):
        for ik in range(nk):
            nn = int(n[iq, ik])
            m = max(1, nn // 4)
            d = gao_delta(nn, biased_acf(acf_x[iq], m), biased_acf(acf_y, m), m)
            delta[iq, ik], nu[iq, ik], M[iq, ik] = d, nn * d - 2, m
            p_gao[iq, ik] = p_one_sided(r[iq, ik], nn * d - 2)
    sidak = []
    for iq in range(nq):
        ik = int(np.nanargmin(p_gao[iq]))
        n_i = max(nk * delta[iq, ik], 1.0)
        sidak.append({"ik_pmin": ik, "n_i": float(n_i), "p_sidak": float(1 - (1 - p_gao[iq, ik]) ** n_i)})
    return {"r": r, "n": n, "delta": delta, "nu": nu, "M": M, "p_gao": p_gao, "sidak": sidak}


def cell_correlation(x, yw, keep, acf_x):
    """Single cell (x = SPI(t - k) against z): Pearson r, n and one-sided Gao p (no Sidak correction)."""
    r, n = corr_grid(x[None, None, :], np.where(keep, yw, np.nan))
    r, nn = float(r[0, 0]), int(n[0, 0])
    m = max(1, nn // 4)
    delta = gao_delta(nn, biased_acf(acf_x, m), biased_acf(yw, m), m)
    nu = nn * delta - 2
    return {"r": r, "n": nn, "delta": delta, "nu": nu, "M": m, "p_gao": p_one_sided(r, nu)}


def lag0_correlations(spi_q, zm, excluded_months=(1, 2, 3)):
    """Descriptive same-month correlation of SPI(t) and monthly z (t) in each scenario: list of {r, n}."""
    x = spi_q.reindex(zm.index)
    ok = x.notna() & zm.notna()
    x, z = x[ok], zm[ok]
    out = []
    for keep in scenario_masks(x.index, excluded_months).values():
        out.append({"r": float(np.corrcoef(x.values[keep], z.values[keep])[0, 1]), "n": int(keep.sum())})
    return out


# ---------------------------------------------------------------- surrogate series
def ar1_surrogates(r1, mean, sd, n, n_surr, rng, nan_mask):
    """Gaussian AR(1) series (n_surr, n) with lag-1 autocorrelation r1, given mean and standard deviation; NaN where
    nan_mask."""
    s = np.zeros((n_surr, n))
    s[:, 0] = rng.standard_normal(n_surr)
    eps = rng.standard_normal((n_surr, n))
    c = np.sqrt(1 - r1 ** 2)
    for t in range(1, n):
        s[:, t] = r1 * s[:, t - 1] + c * eps[:, t]
    s = mean + sd * s
    s[:, nan_mask] = np.nan
    return s


def phase_surrogates(y, n_surr, rng):
    """Random-phase series (Prichard and Theiler 1994): the amplitudes of the rFFT and the mean of y are kept, the
    phases are drawn in [0, 2 pi) (zero at the constant term; 0 or pi at the Nyquist frequency when N is even)."""
    finite = np.isfinite(y)
    mu = y[finite].mean()
    d = np.where(finite, y, mu) - mu
    T = len(d)
    F = np.fft.rfft(d)
    phase = rng.uniform(0, 2 * np.pi, size=(n_surr, len(F)))
    phase[:, 0] = 0.0
    if T % 2 == 0:
        phase[:, -1] = rng.integers(0, 2, size=n_surr) * np.pi
    out = np.fft.irfft(np.abs(F)[None, :] * np.exp(1j * phase), n=T, axis=1) + mu
    out[:, ~finite] = np.nan
    return out


def generate_surrogates(generator, yw, n_surr, seed):
    """Surrogates of z over its whole contiguous window (generator in GENERATORS): AR(1) uses the seed, the phase
    generator seed + 1."""
    finite = np.isfinite(yw)
    values = yw[finite]
    if generator == "AR(1)":
        r1 = float(np.clip(pd.Series(yw).autocorr(1), -0.99, 0.99))
        return ar1_surrogates(r1, values.mean(), values.std(ddof=1), len(yw), n_surr, np.random.default_rng(seed), ~finite)
    return phase_surrogates(yw, n_surr, np.random.default_rng(seed + 1))


def family_test(sx, yw, keeps, n_surr, seed, chunk=2000):
    """Maximum r over the cells of sx in every surrogate (all generated first, then correlated chunk at a time), by
    generator and scenario (keeps: scenario -> month mask). Returns ({(scenario, generator): dict with iq, ik of
    the best cell, r_max, p_family = (1 + #{max_s >= r_max}) / (1 + n_surr), r_crit95 and max_s}, {generator:
    surrogates})."""
    results, surrogates = {}, {}
    for gen in GENERATORS:
        sur = generate_surrogates(gen, yw, n_surr, seed)
        surrogates[gen] = sur
        for scenario, keep in keeps.items():
            r_obs, _ = corr_grid(sx, np.where(keep, yw, np.nan))
            rr = np.where(np.isfinite(r_obs), r_obs, -9.0)
            iq, ik = np.unravel_index(int(np.argmax(rr)), rr.shape)
            r_s, _ = corr_grid(sx, np.where(keep[None, :], sur, np.nan), chunk)
            mx = np.nanmax(r_s.reshape(n_surr, -1), axis=1)
            results[(scenario, gen)] = {
                "iq": int(iq), "ik": int(ik), "r_max": float(r_obs[iq, ik]),
                "p_family": float((1 + np.sum(mx >= r_obs[iq, ik] - 1e-12)) / (1 + n_surr)),
                "r_crit95": float(np.quantile(mx, 0.95)), "max_s": mx}
    return results, surrogates


def cell_surrogate_test(x, yw, keeps, n_surr, seed, chunk=2000):
    """Single-cell surrogate test (the statistic is the r of the cell, not a maximum over cells): by (scenario,
    generator) the observed r, the r of every surrogate (r_s), the one-sided p and the 95th percentile r_crit95.
    Returns (results, {generator: surrogates})."""
    results, surrogates = {}, {}
    for gen in GENERATORS:
        sur = generate_surrogates(gen, yw, n_surr, seed)
        surrogates[gen] = sur
        for scenario, keep in keeps.items():
            r_obs = float(corr_grid(x[None, None, :], np.where(keep, yw, np.nan))[0][0, 0])
            r_s = corr_grid(x[None, None, :], np.where(keep[None, :], sur, np.nan), chunk)[0][:, 0, 0]
            results[(scenario, gen)] = {"r_obs": r_obs, "r_s": r_s,
                                        "p": float((1 + np.sum(r_s >= r_obs - 1e-12)) / (1 + n_surr)),
                                        "r_crit95": float(np.nanquantile(r_s, 0.95))}
    return results, surrogates


# ---------------------------------------------------------------- slopes of z on SPI (amplitude correspondence)
def _slopes_xy(x, y):
    """(r, s_x, s_y, OLS slope, RMA slope) of y on x (standard deviations with ddof = 1)."""
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    r = float(np.corrcoef(x, y)[0, 1])
    return r, sx, sy, r * sy / sx, np.sign(r) * sy / sx


def slope_amplitude(x, y, dates, nu):
    """OLS and RMA (reduced major axis) slopes of y on x with 95% intervals (OLS: b +- t SE_b with the Gao nu; RMA:
    Jolicoeur and Mosimann 1968), leave-one-out slopes and Cook's distance of the OLS fit. Returns a dict."""
    n = len(x)
    r, sx, sy, b_ols, b_rma = _slopes_xy(x, y)
    xm, ym = float(x.mean()), float(y.mean())
    a_ols, a_rma = ym - b_ols * xm, ym - b_rma * xm
    t_q, t_q_n2 = float(t_dist.ppf(0.975, nu)), float(t_dist.ppf(0.975, n - 2))
    se_b = (sy / sx) * np.sqrt((1 - r ** 2) / nu)
    se_b_n2 = (sy / sx) * np.sqrt((1 - r ** 2) / (n - 2))
    B = t_q ** 2 * (1 - r ** 2) / nu
    ci_rma = (b_rma * (np.sqrt(B + 1) - np.sqrt(B)), b_rma * (np.sqrt(B + 1) + np.sqrt(B)))
    loo = np.array([_slopes_xy(x[np.arange(n) != i], y[np.arange(n) != i])[3:] for i in range(n)])   # leave one out
    e = y - (a_ols + b_ols * x)
    h = 1 / n + (x - xm) ** 2 / np.sum((x - xm) ** 2)
    s2 = np.sum(e ** 2) / (n - 2)
    cook = e ** 2 / (2 * s2) * h / (1 - h) ** 2
    imax = int(np.argmax(np.abs(loo[:, 0] - b_ols)))
    return {"n": n, "r": r, "nu": nu, "sx": sx, "sy": sy, "xm": xm, "ym": ym, "t": t_q,
            "b_ols": b_ols, "a_ols": a_ols, "se_b": float(se_b),
            "ci_ols": (b_ols - t_q * se_b, b_ols + t_q * se_b),
            "ci_ols_n2": (b_ols - t_q_n2 * se_b_n2, b_ols + t_q_n2 * se_b_n2),
            "b_rma": b_rma, "a_rma": a_rma, "ci_rma": ci_rma, "loo": loo, "cook": cook,
            "imax": imax, "month_max": dates[imax].strftime("%Y-%m"), "dmax": float(loo[imax, 0] - b_ols),
            "x": x, "y": y, "dates": dates}


def slope_rows(slopes):
    """Rows of the slope table and of the leave-one-out table (slopes: scenario -> slope_amplitude dict)."""
    rows, rows_loo = [], []
    for scenario, p in slopes.items():
        rows.append({"scenario": scenario, "n": p["n"], "r": p["r"], "gao_nu": p["nu"],
                     "slope_ols": p["b_ols"], "slope_ols_ci_low": p["ci_ols"][0], "slope_ols_ci_high": p["ci_ols"][1],
                     "slope_ols_ci_low_n2": p["ci_ols_n2"][0], "slope_ols_ci_high_n2": p["ci_ols_n2"][1],
                     "intercept_ols": p["a_ols"], "slope_rma": p["b_rma"], "slope_rma_ci_low": p["ci_rma"][0],
                     "slope_rma_ci_high": p["ci_rma"][1], "intercept_rma": p["a_rma"], "s_x": p["sx"], "s_y": p["sy"],
                     "loo_ols_min": p["loo"][:, 0].min(), "loo_ols_max": p["loo"][:, 0].max(),
                     "loo_rma_min": p["loo"][:, 1].min(), "loo_rma_max": p["loo"][:, 1].max(),
                     "most_influential_month": p["month_max"], "ols_change_most_influential": p["dmax"]})
        for i in range(p["n"]):
            rows_loo.append({"scenario": scenario, "excluded_month": p["dates"][i].strftime("%Y-%m"), "x": p["x"][i],
                             "y": p["y"][i], "slope_ols_loo": p["loo"][i, 0], "slope_rma_loo": p["loo"][i, 1],
                             "cook_d": p["cook"][i]})
    return pd.DataFrame(rows), pd.DataFrame(rows_loo)


# ---------------------------------------------------------------- R^2 of z on SPI-q(t - k) over a grid of windows and lags
def scatter_pairs(spi, zm, qs, ks):
    """Pairs (x = SPI-q(t - k), y = z(t), months) of every cell (q, k) where both exist."""
    cells = {}
    for q in qs:
        for k in ks:
            x = spi[f"SPI_{q}"].shift(k).reindex(zm.index)
            ok = x.notna() & zm.notna()
            cells[(q, k)] = (x[ok].to_numpy(float), zm[ok].to_numpy(float), x.index[ok])
    return cells


def slope_sign(b):
    """"+" for a positive slope, "−" otherwise."""
    return "+" if b > 0 else "−"


def scatter_r2(cells, excluded_months):
    """OLS R^2 (with intercept) and slope of z on SPI-q(t - k) per cell and scenario: {(q, k, scenario): (n, r2, slope)}."""
    out = {}
    for (q, k), (x, y, months) in cells.items():
        for scenario, keep in scenario_masks(months, excluded_months).items():
            xe, ye = x[keep], y[keep]
            b = float(np.sum((xe - xe.mean()) * (ye - ye.mean())) / np.sum((xe - xe.mean()) ** 2))
            a = float(ye.mean() - b * xe.mean())
            ss_res, ss_tot = float(np.sum((ye - (a + b * xe)) ** 2)), float(np.sum((ye - ye.mean()) ** 2))
            out[(q, k, scenario)] = (int(len(xe)), 1.0 - ss_res / ss_tot, b)
    return out


# ---------------------------------------------------------------- result tables
def correlation_rows(cor, qs, ks):
    """Table of the cells of correlations(): scenario (SCENARIOS order of cor), q, k, n, r and Gao delta, nu, M and one-sided p."""
    return pd.DataFrame([{"scenario": scenario, "q": q, "k": k, "n": int(c["n"][iq, ik]), "r": c["r"][iq, ik],
                          "gao_delta": c["delta"][iq, ik], "gao_nu": c["nu"][iq, ik], "gao_M": int(c["M"][iq, ik]),
                          "gao_p_one_sided": c["p_gao"][iq, ik]}
                         for scenario, c in zip(SCENARIOS, cor) for iq, q in enumerate(qs) for ik, k in enumerate(ks)])


def family_rows(cor, fam, qs, ks, n_surr, seed):
    """Table of the family test per scenario and generator: best cell, maximum r, family p, critical r and the Gao-Sidak p of
    the q of the best cell."""
    rows = []
    for scenario, c in zip(SCENARIOS, cor):
        for gen in GENERATORS:
            f = fam[(scenario, gen)]
            sidak = c["sidak"][f["iq"]]
            rows.append({"scenario": scenario, "generator": gen, "best_q": qs[f["iq"]], "best_k": ks[f["ik"]], "r_max": f["r_max"],
                         "family_p_one_sided": f["p_family"], "r_crit95": f["r_crit95"], "gao_sidak_p_best_q": sidak["p_sidak"],
                         "gao_n_i": sidak["n_i"], "n_surrogates": n_surr, "seed": seed})
    return pd.DataFrame(rows)


def cell_rows(cor):
    """Table of the single-cell correlations (cell_correlation) per scenario."""
    return pd.DataFrame([{"scenario": scenario, "n": c["n"], "r": c["r"], "gao_delta": c["delta"], "gao_nu": c["nu"],
                          "gao_M": c["M"], "gao_p_one_sided": c["p_gao"]} for scenario, c in zip(SCENARIOS, cor)])


def cell_surrogate_rows(fam, n_surr, seed):
    """Table of the single-cell surrogate test (cell_surrogate_test) per scenario and generator."""
    return pd.DataFrame([{"scenario": scenario, "generator": gen, "r_obs": fam[(scenario, gen)]["r_obs"],
                          "p_one_sided": fam[(scenario, gen)]["p"], "r_crit95": fam[(scenario, gen)]["r_crit95"],
                          "n_surrogates": n_surr, "seed": seed} for scenario in SCENARIOS for gen in GENERATORS])


def scatter_r2_rows(r2, qs, ks):
    """Table of scatter_r2(): q, k, scenario, n, r2, OLS slope and its sign."""
    return pd.DataFrame([{"q": q, "k": k, "scenario": scenario, "n": r2[(q, k, scenario)][0], "r2": r2[(q, k, scenario)][1],
                          "ols_slope": r2[(q, k, scenario)][2], "sign": slope_sign(r2[(q, k, scenario)][2])}
                         for q in qs for k in ks for scenario in SCENARIOS])
