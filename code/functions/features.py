"""Feature table (spectral columns, leads, rain windows, calendar) and development/holdout split."""
from collections import namedtuple

import numpy as np
import pandas as pd

from functions.data import compute_indices, load_table

LEAD_SUFFIX = "_lead{k}"
RAIN_PREFIX = "pr_"                      # rain columns are pr_{w}d
TARGET = "gw_cm"
SEASON_BY_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                   6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

Partition = namedtuple("Partition", ["dev_end", "holdout_start"])


def add_spectral_leads(spectral, leads):
    """Append spectral.shift(-k) for each k, named with LEAD_SUFFIX (value k days after)."""
    parts = [spectral]
    for k in leads:
        parts.append(spectral.shift(-k).add_suffix(LEAD_SUFFIX.format(k=k)))
    return pd.concat(parts, axis=1)


def add_rain_windows(precip, windows):
    """Rain accumulated over the past w days; incomplete windows stay NaN."""
    return pd.DataFrame({f"{RAIN_PREFIX}{w}d": precip.rolling(w, min_periods=w).sum() for w in windows})


def build_feature_table(bands, precip, leads, windows, band_map, columns):
    """Full feature table over the whole series: the chosen bands and indices, leads, rain windows, sin/cos of day of year.

    columns are the bands and indices that become features (leads are made only for them).

    It never uses the water level, so building it on the whole series leaks nothing.
    """
    spectral = bands.join(compute_indices(bands, band_map))[columns]
    table = add_spectral_leads(spectral, leads).join(add_rain_windows(precip, windows))
    days_in_year = np.where(table.index.is_leap_year, 366.0, 365.0)
    angle = 2 * np.pi * (table.index.dayofyear - 1) / days_in_year   # Jan 1 = 0, exactly periodic each year
    table["sin_doy"] = np.sin(angle)
    table["cos_doy"] = np.cos(angle)
    return table


def season_of(index):
    """Season label per date (DJF = austral summer), ordered DJF, MAM, JJA, SON."""
    return pd.Series(pd.Categorical(index.month.map(SEASON_BY_MONTH), categories=["DJF", "MAM", "JJA", "SON"]), index=index)


def compute_partition(piezo_index, dev_years, separation_days):
    """Cut dates for development and holdout.

    The development is the first dev_years years of the piezometer record, then
    separation_days days are dropped and the rest of the record is the holdout.
    """
    first, last = piezo_index.min(), piezo_index.max()
    dev_end = first + pd.DateOffset(years=dev_years) - pd.Timedelta(days=1)
    holdout_start = dev_end + pd.Timedelta(days=separation_days + 1)
    if holdout_start > last:
        raise ValueError(f"no holdout left: development ends {dev_end.date()} (+{separation_days} days "
                         f"of separation) but the record ends {last.date()}")
    return Partition(dev_end=dev_end, holdout_start=holdout_start)


def split_dev_holdout(X, y, partition):
    """Return (X_dev, y_dev), (X_hold, y_hold); the separation days are dropped.

    Only days with a water level and a complete feature row are kept.
    """
    valid = y.notna() & X.notna().all(axis=1)
    dev = valid & (X.index <= partition.dev_end)
    hold = valid & (X.index >= partition.holdout_start)
    return (X[dev], y[dev]), (X[hold], y[hold])


def load_development(path):
    """Load the development table as (X, y). Steps 04-07 load data only through this."""
    table = load_table(path)
    return table.drop(columns=TARGET), table[TARGET]
