from typing import NamedTuple
from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================

def _normalized_difference(b1:pd.Series, b2:pd.Series) -> pd.Series:
    """
    Compute the normalized difference between two bands.
    """
    denom = b1 + b2
    return (b1 - b2) / denom.where(denom != 0)

def compute_spectral_indices(
    df: pd.DataFrame,
    green: str = 'green',
    red: str = 'red',
    nir: str = 'nir',
    swir1: str = 'swir1',
    swir2: str = 'swir2',
) -> pd.DataFrame:
    """
    Compute spectral indices from surface reflectance bands.
    
        Each band argument is the name of a column in `df`. Bands must be surface
        reflectance scaled to [0, 1].
    
        Parameters
        ----------
        df : pd.DataFrame
            Input data with one column per spectral band.
        green, red, nir, swir1, swir2 : str
            Names of the columns holding the green, red, near-infrared and the
            two shortwave-infrared bands (SWIR1 ~1.6 um, SWIR2 ~2.2 um).
    
        Returns
        -------
        pd.DataFrame
            Copy of `df` with the following columns added:
    
            ndvi  : (NIR - RED) / (NIR + RED)
            gndvi : (NIR - GREEN) / (NIR + GREEN)
            ndwi  : (GREEN - NIR) / (GREEN + NIR)
            mndwi : (GREEN - SWIR1) / (GREEN + SWIR1)
            ndmi  : (NIR - SWIR1) / (NIR + SWIR1)
            nbr   : (NIR - SWIR2) / (NIR + SWIR2)
            str1  : (1 - SWIR1)^2 / (2 * SWIR1)
            str2  : (1 - SWIR2)^2 / (2 * SWIR2)
    """
    missing = [c for c in (green, red, nir, swir1, swir2) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    out = df.copy()
    g, r, n, s1, s2 = (out[c] for c in (green, red, nir, swir1, swir2))

    out["ndvi"] = _normalized_difference(n, r)
    out["gndvi"] = _normalized_difference(n, g)
    out["ndwi"] = _normalized_difference(g, n)
    out["mndwi"] = _normalized_difference(g, s1)
    out["ndmi"] = _normalized_difference(n, s1)
    out["nbr"] = _normalized_difference(n, s2)

    out["str1"] = (1 - s1) ** 2 / (2 * s1.where(s1 != 0))
    out["str2"] = (1 - s2) ** 2 / (2 * s2.where(s2 != 0))

    return out


class PCAResult(NamedTuple):
    scores: pd.DataFrame           # PC scores, same index as input
    model: PCA                     # Fitted PCA model
    scaler: StandardScaler         # Fitted scaler (needed to project new data)
    loadings: pd.DataFrame         # Feature-component correlations
    explained_variance: pd.Series  # Explained variance ratio (%) per PC

def run_pca(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 2,
    verbose: bool = True,
) -> PCAResult:
    """
    Standardize features and fit a PCA.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. Selected columns must be numeric and free of NaN.
    feature_cols : list[str]
        Columns used as PCA features.
    n_components : int
        Number of principal components to keep.
    verbose : bool
        Print variance and loadings summary.

    Returns
    -------
    PCAResult
        scores, fitted model, fitted scaler, loadings and explained variance.
    """
    # Validate inputs
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")

    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric columns in feature_cols: {non_numeric}")
    
    subset = df[feature_cols]
    if subset.isna().any().any():
        raise ValueError("Features contain NaN values. Drop or impute them first.")

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(subset)

    # Run PCA
    pca_model = PCA(n_components=n_components, svd_solver='full')
    pc_values = pca_model.fit_transform(scaled_data)

    # Build component names from the fitted model
    pc_names = [f"PC_{i + 1}" for i in range(pca_model.n_components_)]

    # Store scores in a df
    scores = pd.DataFrame(pc_values, columns=pc_names, index=df.index)

    # Loadings as feature-component correlations (data is standardized)
    loadings = pd.DataFrame(
        pca_model.components_.T * np.sqrt(pca_model.explained_variance_),
        columns=pc_names,
        index=feature_cols,
    )

    # Explained variance per component (%)
    explained_variance = pd.Series(
        pca_model.explained_variance_ratio_ * 100, index=pc_names
    )

    if verbose:
        print("\n--- PCA Evaluation ---")
        for name, var in explained_variance.items():
            print(f"{name} explains: {var:.2f}% of the variance")
        print(f"Total variance retained: {explained_variance.sum():.2f}%")
        print("\nLoadings (feature-component correlations):")
        print(loadings.round(3))

    return PCAResult(scores, pca_model, scaler, loadings, explained_variance)


def add_cyclic_doy_features(
    df: pd.DataFrame,
    add_sin: bool = True,
    add_cos: bool = True,
) -> pd.DataFrame:
    """
    Add cyclic features for the day of year (DOY) to a DataFrame with a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index of the DataFrame must be a DatetimeIndex.")
    if not (add_sin or add_cos):
        raise ValueError("At least one of add_sin / add_cos must be True.")

    df_copy = df.copy()
    idx = df_copy.index

    total_days = np.where(idx.is_leap_year, 366.0, 365.0)
    angle = 2.0 * np.pi * (idx.dayofyear - 1) / total_days

    if add_sin:
        df_copy["doy_sin"] = np.sin(angle)
    if add_cos:
        df_copy["doy_cos"] = np.cos(angle)

    return df_copy


def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    windows: Sequence[int] = (7, 14, 21),
    stats: Sequence[str] = ('mean', 'std'),
    min_periods: int | None = None,
    shift: int = 0,
) -> pd.DataFrame:
    """
    Add trailing rolling-window statistics for the selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Data with a sorted, unique DatetimeIndex.
    feature_cols : Sequence[str]
        Columns to aggregate.
    windows : Sequence[int]
        Window sizes in days (time-based windows).
    stats : Sequence[str]
        Aggregations accepted by pandas (e.g. 'mean', 'std', 'min').
    min_periods : int | None
        Minimum observations per window. Defaults to 1 for 'mean'-like stats
        and 2 if 'std' is requested.
    shift : int
        Rows to lag the series before rolling. Use shift=1 to exclude the
        current day and avoid target leakage.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with columns '{feat}_roll_{stat}_{w}d'.
    """
    # Validate inputs
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")
    
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Index must be sorted and free of duplicates.")
    
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")
    
    if any(not isinstance(w, int) or w <= 0 for w in windows):
        raise ValueError("windows must be positive integers.")

    # Build all new columns first, concatenate once to avoid fragmentation
    new_cols = {}
    for feat in feature_cols:
        series = df[feat].shift(shift) if shift else df[feat]
        for w in windows:
            roll = series.rolling(window=f"{w}D", min_periods=min_periods or 1)
            for stat in stats:
                new_cols[f"{feat}_roll_{stat}_{w}d"] = roll.agg(stat)

    out = pd.DataFrame(new_cols, index=df.index)
    
    # Avoid silent duplicate column names
    overlap = out.columns.intersection(df.columns)
    if len(overlap):
        raise ValueError(f"Columns already exist: {list(overlap)}")

    return pd.concat([df, out], axis=1)


def add_lags(
    df: pd.DataFrame,
    features: Sequence[str],
    past_lags: Sequence[int] | None = None,
    future_lags: Sequence[int] | None = None,
    freq: str = "D",
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Add time-based lagged (past) and lead (future) copies of the features.

    Parameters
    ----------
    df : pd.DataFrame
        Data with a sorted, unique DatetimeIndex.
    features : Sequence[str]
        Columns to lag.
    past_lags : Sequence[int] | None
        Positive steps into the past: value at t - lag.
    future_lags : Sequence[int] | None
        Positive steps into the future: value at t + lag.
    freq : str
        Unit of one lag step (default: daily).
    dropna : bool
        If True, drop rows with NaN in `features` or in the new columns.
        This leaves gaps in the index, so apply it as the last step of
        the pipeline.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with columns '{feat}_pastLag_{n}' / '{feat}_futureLag_{n}'.
    """
    past_lags = list(past_lags or [])
    future_lags = list(future_lags or [])

    # Validate inputs
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")
    
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("Index must be sorted and free of duplicates.")
    
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in DataFrame: {missing}")
    
    if not (past_lags or future_lags):
        raise ValueError("Provide at least one past or future lag.")
    
    if any(not isinstance(n, Integral) or n < 1 for n in past_lags + future_lags):
        raise ValueError("Lags must be positive integers.")

    # Signed steps: positive = past, negative = future
    steps = {f"pastLag_{n}": n for n in past_lags}
    steps.update({f"futureLag_{n}": -n for n in future_lags})

    # Build all lagged columns first, concatenate once
    new_cols = {}
    for feat in features:
        for suffix, periods in steps.items():
            # Shift index labels by time, then realign to the original index,
            # so a missing day yields NaN instead of picking another row
            new_cols[f"{feat}_{suffix}"] = (
                df[feat].shift(periods, freq=freq).reindex(df.index)
            )

    lags_df = pd.DataFrame(new_cols, index=df.index)

    # Avoid silent duplicate column names
    overlap = lags_df.columns.intersection(df.columns)
    if len(overlap):
        raise ValueError(f"Columns already exist: {list(overlap)}")

    out = pd.concat([df, lags_df], axis=1)

    if dropna:
        out = out.dropna(subset=list(features) + list(lags_df.columns))

    return out


def join_datasets(
    left: pd.DataFrame | pd.Series,
    right: pd.DataFrame | pd.Series,
    how: Literal["inner", "left", "right", "outer"] = "inner",
    lsuffix: str = "",
    rsuffix: str = "",
) -> pd.DataFrame:
    """
    Join two time-indexed datasets on their DatetimeIndex.

    Parameters
    ----------
    left, right : pd.DataFrame | pd.Series
        Data with a DatetimeIndex. Series must be named.
    how : {'inner', 'left', 'right', 'outer'}
        'inner' keeps only dates present in both (default).
    lsuffix, rsuffix : str
        Suffixes for overlapping column names.

    Returns
    -------
    pd.DataFrame
        Joined data, sorted by index.
    """
    # Convert Series to DataFrame (requires a name to avoid integer columns)
    frames = []
    for obj in (left, right):
        if isinstance(obj, pd.Series):
            if obj.name is None:
                raise ValueError("Series must have a name.")
            obj = obj.to_frame()
        frames.append(obj)
    left, right = frames

    # Validate indices
    for name, obj in (("left", left), ("right", right)):
        if not isinstance(obj.index, pd.DatetimeIndex):
            raise TypeError(f"{name} must have a DatetimeIndex.")
        if obj.index.has_duplicates:
            raise ValueError(f"{name} index has duplicate timestamps.")
    if (left.index.tz is None) != (right.index.tz is None):
        raise ValueError("Cannot mix tz-aware and tz-naive indices.")

    # Overlap columns need explicit suffixes
    overlap = left.columns.intersection(right.columns)
    if len(overlap) and not (lsuffix or rsuffix):
        raise ValueError(
            f"Overlapping columns {list(overlap)}; provide lsuffix/rsuffix."
        )

    result = left.join(right, how=how, lsuffix=lsuffix, rsuffix=rsuffix).sort_index()

    if result.empty:
        raise ValueError("Join produced an empty DataFrame: no overlapping dates.")

    return result