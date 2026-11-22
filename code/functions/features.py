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


def feature_table(daily, decisions, band_map):
    """Feature table of the whole clean daily table of step 1 with the LEADS, WINDOWS and SPECTRAL of decisions
    (build_feature_table): the same table for training (03) and for the predictions over the whole series (08)."""
    bands = daily.drop(columns=[TARGET, "qa_low", "precip_mm"])
    return build_feature_table(bands, daily["precip_mm"], decisions["LEADS"], decisions["WINDOWS"], band_map,
                               decisions["SPECTRAL"])


def season_of(index):
    """Season label per date (DJF = austral summer), ordered DJF, MAM, JJA, SON."""
    return pd.Series(pd.Categorical(index.month.map(SEASON_BY_MONTH), categories=["DJF", "MAM", "JJA", "SON"]), index=index)


def modelling_days(daily, bands, use_rain):
    """Days of the clean daily table (step 1) with a water level and the NBAR bands the features are built from,
    plus rain when use_rain (rain windows in the features): the days that get a complete feature row in step 3.
    Steps 2 and 3 cut development / holdout on them."""
    columns = [TARGET, *bands] + (["precip_mm"] if use_rain else [])
    return daily.index[daily[columns].notna().all(axis=1)]


def compute_partition(days, scheme, separation_days):
    """Cut dates for development and holdout on days (dates with a complete row, from modelling_days).

    scheme (an entry of config.PARTITIONS) has one key:
    - dev_years: the development is the first dev_years years from the first day, then separation_days days are
      dropped and the rest is the holdout;
    - holdout_days: the holdout is the last holdout_days days, the separation_days before it are dropped and
      everything earlier is the development.
    """
    days = pd.DatetimeIndex(days).sort_values()
    if "dev_years" in scheme:
        dev_end = days.min() + pd.DateOffset(years=scheme["dev_years"]) - pd.Timedelta(days=1)
        holdout_start = dev_end + pd.Timedelta(days=separation_days + 1)
        if holdout_start > days.max():
            raise ValueError(f"no holdout left: development ends {dev_end.date()} (+{separation_days} days "
                             f"of separation) but the record ends {days.max().date()}")
    else:
        holdout_days = scheme["holdout_days"]
        if len(days) <= holdout_days + separation_days:
            raise ValueError(f"no holdout left: {len(days)} days, not enough for a holdout of {holdout_days} "
                             f"(+{separation_days} of separation) and a development period")
        holdout_start = days[-holdout_days]
        dev_end = holdout_start - pd.Timedelta(days=separation_days + 1)
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
