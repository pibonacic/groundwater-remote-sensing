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


def compute_indices(df, bands):
    """Spectral indices from NBAR band columns; bands maps green/red/nir/swir1/swir2 to columns.

    ndvi (nir-red), gndvi (nir-green), ndwi (green-nir), mndwi (green-swir1),
    ndmi (nir-swir1), nbr (nir-swir2) are normalized differences; str1 and str2 are
    the SWIR transformed reflectance (1 - swir)^2 / (2 * swir).
    """
    g, r, n, s1, s2 = (df[bands[k]] for k in ("green", "red", "nir", "swir1", "swir2"))

    def nd(a, b):
        return (a - b) / (a + b)

    return pd.DataFrame({
        "ndvi": nd(n, r), "gndvi": nd(n, g), "ndwi": nd(g, n),
        "mndwi": nd(g, s1), "ndmi": nd(n, s1), "nbr": nd(n, s2),
        "str1": (1 - s1) ** 2 / (2 * s1.where(s1 != 0)),
        "str2": (1 - s2) ** 2 / (2 * s2.where(s2 != 0)),
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
