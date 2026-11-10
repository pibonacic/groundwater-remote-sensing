"""Reading, quality control and save/load of tables."""
import pandas as pd

NBAR_REFLECTANCE = "Nadir_Reflectance_"
NBAR_QUALITY = "BRDF_Albedo_Band_Mandatory_Quality_"
UNIT_TO_CM = {"m": 100.0, "cm": 1.0, "mm": 0.1}


def load_piezometer(path, column):
    """Read one piezometer column from the in-situ CSV, untransformed."""
    return pd.read_csv(path, index_col=0, parse_dates=True)[column].rename("gw")


def load_nbar(path):
    """Read the VIIRS NBAR CSV (reflectance and quality columns), untransformed."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_chirps(path):
    """Read the CHIRPS daily precipitation CSV as a Series in mm."""
    return pd.read_csv(path, index_col=0, parse_dates=True)["precipitation_median_mm"].rename("precip_mm")


def to_daily(df):
    """Force a continuous daily index; missing days become NaN."""
    return df.sort_index().asfreq("D")


def normalize_piezometer(s, unit, zero_cm):
    """Convert to cm and subtract zero_cm. The sign of the raw data is kept."""
    return (s * UNIT_TO_CM[unit] - zero_cm).rename("gw_cm")


def apply_nbar_qa(df, rules):
    """Rename NBAR bands by suffix and flag low quality; no value is removed.

    Returns bands named by suffix (M1..M11, I1..I3) plus qa_low, which is 1.0 when
    any band has a quality flag outside rules["good_flags"], 0.0 when none, NaN when
    the day has no quality data.
    """
    bands = [c[len(NBAR_REFLECTANCE):] for c in df if c.startswith(NBAR_REFLECTANCE)]
    quality = df[[NBAR_QUALITY + b for b in bands]]
    out = df[[NBAR_REFLECTANCE + b for b in bands]].set_axis(bands, axis=1)
    out["qa_low"] = (quality.notna() & ~quality.isin(rules["good_flags"])).any(axis=1).astype(float).where(quality.notna().any(axis=1))
    return out


SAVI_L = 0.5            # soil adjustment factor of SAVI
EVI2_L, EVI2_G = 1.0, 2.5   # canopy background adjustment and gain of EVI2
NDSINW_BETA = 0.05      # offset of NDSInw


def compute_indices(df, bands):
    """Spectral indices from NBAR band columns; bands maps red/nir/swir1 to columns.

    ndvi (nir-red), ndmi (nir-swir1), ndsaii (red-swir1) are normalized differences;
    str1 is the SWIR transformed reflectance (1 - swir1)^2 / (2 * swir1); dvi = nir - red;
    mlswi26 = (1 - nir - swir1) / (1 - nir + swir1); savi and evi2 are the soil-adjusted and
    two-band enhanced vegetation indices; bai = red + swir1 - nir; s3 = nir (red - swir1) / ((nir + red) (nir + swir1)).
    csiswir = (nir + swir1) / 2; mi = (nir - swir1) / (nir swir1); ndsinw = (nir - swir1 - beta) / (nir + swir1);
    baim = 1 / ((0.05 - nir)^2 + (0.2 - swir1)^2); nli = (nir^2 - red) / (nir^2 + red);
    bai_burn = 1 / ((0.1 - red)^2 + (0.06 - nir)^2) (the burned area index, not the bai above);
    snirvswir = (nir - red - swir1^2) / (nir + red + swir1^2) * nir. The SWIR2 band is not used: baim and snirvswir take swir1.
    """
    r, n, s1 = (df[bands[k]] for k in ("red", "nir", "swir1"))

    def nd(a, b):
        return (a - b) / (a + b)

    return pd.DataFrame({
        "ndvi": nd(n, r), "ndmi": nd(n, s1), "ndsaii": nd(r, s1),
        "str1": (1 - s1) ** 2 / (2 * s1.where(s1 != 0)),
        "dvi": n - r,
        "mlswi26": (1 - n - s1) / (1 - n + s1),
        "savi": (1 + SAVI_L) * (n - r) / (n + r + SAVI_L),
        "evi2": EVI2_G * (n - r) / (n + 2.4 * r + EVI2_L),
        "bai": r + s1 - n,
        "s3": n * (r - s1) / ((n + r) * (n + s1)),
        "csiswir": (n + s1) / 2,
        "mi": (n - s1) / (n * s1),
        "ndsinw": (n - s1 - NDSINW_BETA) / (n + s1),
        "baim": 1 / ((0.05 - n) ** 2 + (0.2 - s1) ** 2),
        "nli": (n ** 2 - r) / (n ** 2 + r),
        "bai_burn": 1 / ((0.1 - r) ** 2 + (0.06 - n) ** 2),
        "snirvswir": (n - r - s1 ** 2) / (n + r + s1 ** 2) * n,
    })


def align_sources(piezo, nbar, chirps):
    """Join the three sources by date on a daily index, without interpolating."""
    return to_daily(pd.concat([piezo, nbar, chirps], axis=1))


def mask_reflectance_outliers(df, columns, max_z):
    """Set to NaN every value of the given columns whose |z-score| exceeds max_z.

    The z-score uses each column's mean and standard deviation over the whole series.
    Returns the masked table and the masked values (date, band, value, z).
    """
    z = (df[columns] - df[columns].mean()) / df[columns].std()
    outliers = z.abs() > max_z
    out = df.copy()
    out[columns] = df[columns].mask(outliers)
    masked = pd.DataFrame({"value": df[columns].stack(), "z": z.stack()})[outliers.stack()]
    masked = masked.rename_axis(["date", "band"]).reset_index(level="band")
    return out, masked


def interpolate_nbar(df, columns):
    """Fill gaps inside the series of the given columns by time-weighted linear interpolation.

    Leading and trailing NaN are left as they are, and any other column is untouched.
    """
    out = df.copy()
    out[columns] = df[columns].interpolate(method="time", limit_area="inside")
    return out


def _longest_gap(s):
    missing = s.isna()
    runs = (missing != missing.shift()).cumsum()
    return int(missing.groupby(runs).sum().max())


def quality_report(df, gw_col, physical_range, max_jump):
    """Diagnostics that mark problems without removing anything.

    Gaps are counted inside each column's own first-to-last valid date.
    Returns a dict of tables: coverage, missing_by_year, qa_by_year, jumps, out_of_range.
    """
    coverage, missing = {}, {}
    for col in df:
        s = df[col].loc[df[col].first_valid_index():df[col].last_valid_index()]
        inside_gaps = int(s.isna().sum())
        coverage[col] = {
            "first": s.index.min(), "last": s.index.max(),
            "n_valid": int(s.notna().sum()), "n_missing_inside": inside_gaps,
            "longest_gap_days": _longest_gap(s) if inside_gaps else 0,
        }
        missing[col] = s.isna().groupby(s.index.year).sum()

    qa = df["qa_low"].dropna()
    qa_by_year = pd.DataFrame({
        "days_low": qa.groupby(qa.index.year).sum().astype(int),
        "days_with_data": qa.groupby(qa.index.year).size(),
    })

    gw = df[gw_col]
    step = gw.diff()
    lo, hi = physical_range
    return {
        "coverage": pd.DataFrame(coverage).T,
        "missing_by_year": pd.DataFrame(missing).fillna(0).astype(int),
        "qa_by_year": qa_by_year,
        "jumps": step[step.abs() > max_jump].rename("jump_cm").to_frame(),
        "out_of_range": gw[(gw < lo) | (gw > hi)].to_frame(),
    }


def save_table(df, path):
    """Save a table as CSV with the date as first column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_table(path):
    """Load a CSV written by save_table."""
    return pd.read_csv(path, index_col=0, parse_dates=True)
