import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.signal import savgol_filter


# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================

def load_and_prepare_data(
        filepath: str,
        index_col: str='Timestamps',
        date_format: str='%Y-%m-%d',
        sep: str=','
) -> pd.DataFrame:
    """
    Load a CSV file, set a DateimeIndex and convert all columns to numeric, coercing errors to NaN.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    index_col : str
        Column containing datetime information to be used as index.
    date_format : str, default='%Y-%m-%d'
        Format string for date parsing.
    sep : str, default=','
        Column separator in the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and numeric columns. Non-convertible values are set as NaN.
    """
    df = pd.read_csv(filepath, sep=sep)

    # Standardize index to Datetime objects
    df[index_col] = pd.to_datetime(df[index_col], format=date_format)
    df.set_index(index_col, inplace=True)

    # Coerce non-numeric strings to NaN
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    # Drop columns whose values are all NaN
    df_numeric = df_numeric.dropna(axis=1, how='all')

    return df_numeric


def slice_by_dates(
        df: pd.DataFrame,
        startDate: str | None = None,
        endDate: str | None= None
) -> pd.DataFrame:
    """
    Clip a dataframe to a specified time range.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    startDate : str, optional
        Date in %Y-%m-%d format.
    endDate : str, optional
        Date in %Y-%m-%d' format.

    Returns
    ----------
    pd.DataFrame
        TIme sliced dataframe.
    """
    df_copy = df.copy()

    # Clip data to a study period if defined
    if startDate is not None or endDate is not None:
        df_copy = df_copy.loc[startDate:endDate]
    return df_copy


def handle_duplicate_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate records with the same date by calculating the daily mean.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame without duplicated values.
    """
    df_copy = df.copy()
    return df_copy.groupby(df_copy.index.date).mean()


def reindex_daily(df:pd.DataFrame) -> pd.DataFrame:
    """
    Reindex a dataframe to a daily frequency. Fills gaps with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    ----------
    pd.DataFrame
        DataFrame daily indexed without gaps.
    """
    df_copy = df.copy()
    
    full_range = pd.date_range(
        start=df_copy.index.min(),
        end=df_copy.index.max(),
        freq='D',
        name=df_copy.index.name
    )

    df_copy = df_copy.reindex(full_range)
    return df_copy

def filter_by_reflectance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter values outside physical range of reflectance (0-1)
    """
    df_filtered = df.where(
        (df >= 0) & (df <= 1))
    return df_filtered


def handle_outliers(
        df: pd.DataFrame,
        z_thresh: float = 3.0,
        ignore_vars: list[str] | None = None
) -> pd.DataFrame:
    """
    Sets to NaN outlier values, defined by a column-wise Z-score threshold. Ignores NaNs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    z_thresh : float, default=3.0
        Z-score threshold for defining outliers.
    ignore_vars: list, optional
        Columns to exclude from outlier detection (e.g. precipitation).

    Returns
    -------
    pd.DataFrame
        DataFrame without outliers.
    """
    df_copy = df.copy()
    ignore_vars = ignore_vars or []

    # Select the columns to evaluate
    cols_to_check = df_copy.select_dtypes(include=[np.number]).columns.difference(ignore_vars)
    
    # Calculate z-scores    
    z_scores = zscore(df_copy[cols_to_check], nan_policy='omit')
    is_outlier = np.abs(z_scores) > z_thresh    # Evaluate outlier threshold
    
    # Mask the outliers, filling with NaN
    df_copy[cols_to_check] = df_copy[cols_to_check].mask(is_outlier)
    return df_copy


def calculate_stats_from_random_points(
        df: pd.DataFrame,
        minPoints: int = 10
) -> pd.DataFrame:
    """
    Calculate summary statistics for multiple spectral features across multiple points
    preserving Datetime index. At least 10 valid (non-NaN) points per date are needed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing spectral feature columns with point suffixes 
        (e.g., feature_p01, feature_p02, ..., feature_pN) and a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same DatetimeIndex, containing 7 statistical columns 
        (mean, std, median, max, min, p25, p75) for each identified feature.
    """
    # Extract unique feature names from the df columns
    columns = df.columns
    feature_names = set(col.split('_p')[0] for col in columns if '_p' in col)
    feature_names = sorted(list(feature_names))

    # Dictionary to store the calculated series
    stats_results = {}

    # Iterate over each individual feature
    for feature in feature_names:
        # Create a subset of columns that belong to the current feature (e.g ndvi_p1, ..., ndvi_pN)
        feature_columns = [col for col in df.columns if col.startswith(f'{feature}_p')]
        feature_data = df[feature_columns]

        # Identify and keep rows (dates) that have at least N valid points
        valid_rows_mask = feature_data.notna().sum(axis=1) >= minPoints
        valid_feature_data = feature_data[valid_rows_mask]

        # Calculate statistics row-wise (axis=1) across all points
        stats_results[f'{feature}_mean'] = valid_feature_data.mean(axis=1)
        stats_results[f'{feature}_std'] = valid_feature_data.std(axis=1)
        stats_results[f'{feature}_median'] = valid_feature_data.median(axis=1)
        stats_results[f'{feature}_max'] = valid_feature_data.max(axis=1)
        stats_results[f'{feature}_min'] = valid_feature_data.min(axis=1)
        stats_results[f'{feature}_p25'] = valid_feature_data.quantile(0.25, axis=1)
        stats_results[f'{feature}_p75'] = valid_feature_data.quantile(0.75, axis=1)

    # Construct the final dataframe maintaining the original datetime index
    output_df = pd.DataFrame(stats_results)
    output_df = output_df.reindex(df.index)

    return output_df


def filter_columns(
        df: pd.DataFrame,
        cols_to_keep: list[str] | str | None = None
) -> pd.DataFrame:
    """
    Filter columns of a dataframe if specified.
    """
    if cols_to_keep is None:
        return df.copy()

    # Convert input string to list
    if isinstance(cols_to_keep, str):
        cols_to_keep = [cols_to_keep]

    return df[cols_to_keep].copy()


def handle_missing_values(
        df: pd.DataFrame,
        method: str = 'drop'
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame according to a specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strategy : str, default = 'drop
        Method to handle missing values:
        - 'drop' : remove rows containing any NaN
        - 'time' : interpolate NaN using time method

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df_copy = df.copy()

    if method == 'drop':  # Removes any row with a NaN
        return df_copy.dropna()
    
    elif method == 'time':  # Interpolates nan values, then drops any nan remaining at start or end
        return df_copy.interpolate(method='time', limit_area='inside').dropna()
    
    else:
        raise ValueError('Method not supported')

    
def smooth_data(
        df: pd.DataFrame,
        cols: list[str] | None = None,
        window_length: int = 11,
        polyorder: int = 2
) -> pd.DataFrame:
    """
    Apply a Savitzky-Golay filter to smooth a subset of columns in a DataFrame while preserving trends.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame without datetime gaps nor NaN.
    cols : list, optional
        List of column names to smooth. If None, all columns are processed.
    window_length : int, default 11
        Lenght of the filter window. Must be a positive odd integer.
    polyorder : int, default 2
        Polynomial order used to fit the samples. Must be less than window_length.

    Returns
    -------
    pd.DataFrame
        DataFrame with selected columns smoothed and others preserved.
    """
    # Sum 1 if window lenght is an even number
    if window_length % 2 == 0:
        window_length += 1

    df_copy = df.copy()
    
    # Evaluate which columns to apply filter
    if cols is None:
        cols_to_smooth = df.columns.tolist()
    else:
        # Validate that requested columns exist in the df
        cols_to_smooth = [c for c in cols if c in df.columns]

    df_copy[cols_to_smooth] = df[cols_to_smooth].apply(
        # Apply Savitsky-Golay filter
        lambda x: savgol_filter(x, window_length=window_length, polyorder=polyorder)
        )
    return df_copy


def attach_spacecraft_column(
        processed_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        spacecraft_col: str = 'Spacecraft',
        index_col: str = 'Timestamps',
        date_format: str = '%Y-%m-%d'
) -> pd.DataFrame:
    """
    
    """
    output_df = processed_df.copy()
    raw_copy = raw_df.copy()

    # Verifies existence of a datetime index, otherwise it creates it
    if not isinstance(raw_copy.index, pd.DatetimeIndex):
        if index_col in raw_copy.columns:
            # Conversion to datetime format
            raw_copy[index_col] = pd.to_datetime(raw_copy[index_col], format=date_format)
            # Set datetime index
            raw_copy.set_index(index_col, inplace=True)

    # Verifies existence of spacecraft column
    if spacecraft_col not in raw_copy.columns:
        return output_df

    # Duplicate values handling
    series_daily = raw_copy[spacecraft_col].groupby(raw_copy.index.date).first()
    series_daily.index = pd.to_datetime(series_daily.index)

    # Daily reindex
    output_df[spacecraft_col] = series_daily.reindex(output_df.index)

    return output_df




# ==============================================================================
# PIPELINES
# ==============================================================================

def process_modis(
        filepath: str,
        index_col: str = 'Timestamps',
        date_format: str = '%Y-%m-%d',
        sep: str = ',',
        start_date: str | None = None,
        end_date: str | None = None,
        cols_to_keep: list[str] | None = None,
        outlier_threshold: float = 3.0,
        smooth_window: int = 25,
        smooth_polynomial: int = 2
) -> pd.DataFrame:
    return (
        load_and_prepare_data(filepath=filepath, index_col=index_col, date_format=date_format, sep=sep)
        .pipe(filter_columns, cols_to_keep=cols_to_keep)
        .pipe(slice_by_dates, startDate=start_date, endDate=end_date)
        .pipe(filter_by_reflectance)
        .pipe(handle_duplicate_values)
        .pipe(reindex_daily)
        .pipe(handle_outliers, z_thresh=outlier_threshold)
        .pipe(handle_missing_values, method='time')
        .pipe(smooth_data, window_length=smooth_window, polyorder=2)
    )

def process_landsat(
        filepath: str,
        index_col: str = 'Timestamps',
        date_format: str = '%Y-%m-%d',
        sep: str = ',',
        start_date: str | None = None,
        end_date: str | None = None,
        cols_to_keep: list[str] | None = None,
        outlier_threshold: float = 3.0,
        stats_min_points: int = 10
) -> pd.DataFrame:
    return (
        load_and_prepare_data(filepath=filepath, index_col=index_col, date_format=date_format, sep=sep)
        .pipe(slice_by_dates, startDate=start_date, endDate=end_date)
        .pipe(filter_by_reflectance)
        .pipe(handle_duplicate_values)
        .pipe(reindex_daily)    # NO ES NECESARIO CREO
        .pipe(handle_outliers, z_thresh=outlier_threshold)
        .pipe(calculate_stats_from_random_points, minPoints=stats_min_points)
        .pipe(filter_columns, cols_to_keep=cols_to_keep)
    )

def process_insitu(
        filepath: str,
        index_col: str = 'Timestamps',
        date_format: str = '%Y-%m-%d',
        sep: str = ',',
        start_date: str | None = None,
        end_date: str | None = None,
        cols_to_keep: list[str] | None = None,
        outlier_threshold: float = 3.0,
) -> pd.DataFrame:
    return (
        load_and_prepare_data(filepath=filepath, index_col=index_col, date_format=date_format, sep=sep)
        .pipe(filter_columns, cols_to_keep=cols_to_keep)
        .pipe(slice_by_dates, startDate=start_date, endDate=end_date)
        .pipe(handle_duplicate_values)
        .pipe(reindex_daily)
        .pipe(handle_outliers, z_thresh=outlier_threshold)
    )

# ALTERNATIVES

def process_modis_nbar(
        filepath: str,
        index_col: str = 'Timestamps',
        date_format: str = '%Y-%m-%d',
        sep: str = ',',
        start_date: str | None = None,
        end_date: str | None = None,
        cols_to_keep: list[str] | None = None,
        outlier_threshold: float = 3.0
) -> pd.DataFrame:
    return (
        load_and_prepare_data(filepath=filepath, index_col=index_col, date_format=date_format, sep=sep)
        .pipe(filter_columns, cols_to_keep=cols_to_keep)
        .pipe(slice_by_dates, startDate=start_date, endDate=end_date)
        .pipe(filter_by_reflectance)
        .pipe(handle_duplicate_values)
        .pipe(reindex_daily)
        .pipe(handle_outliers, z_thresh=outlier_threshold)
    )

def process_modis_no_smooth(
        filepath: str,
        index_col: str = 'Timestamps',
        date_format: str = '%Y-%m-%d',
        sep: str = ',',
        start_date: str | None = None,
        end_date: str | None = None,
        cols_to_keep: list[str] | None = None,
        outlier_threshold: float = 3.0
) -> pd.DataFrame:
    return (
        load_and_prepare_data(filepath=filepath, index_col=index_col, date_format=date_format, sep=sep)
        .pipe(filter_columns, cols_to_keep=cols_to_keep)
        .pipe(slice_by_dates, startDate=start_date, endDate=end_date)
        .pipe(filter_by_reflectance)
        .pipe(handle_duplicate_values)
        .pipe(reindex_daily)
        .pipe(handle_outliers, z_thresh=outlier_threshold)
        .pipe(handle_missing_values, method='time')
    )

def process_landsat_spacecraft(
        filepath: str,
        index_col: str = 'Timestamps',
        date_format: str = '%Y-%m-%d',
        sep: str = ',',
        start_date: str | None = None,
        end_date: str | None = None,
        cols_to_keep: list[str] | None = None,
        outlier_threshold: float = 3.0,
        stats_min_points: int = 10
) -> pd.DataFrame:

    raw_df = pd.read_csv(filepath, sep=sep)

    processed_df = (
        load_and_prepare_data(filepath=filepath, index_col=index_col, date_format=date_format, sep=sep)
        .pipe(slice_by_dates, startDate=start_date, endDate=end_date)
        .pipe(filter_by_reflectance)
        .pipe(handle_duplicate_values)
        .pipe(reindex_daily)    # NO ES NECESARIO
        .pipe(handle_outliers, z_thresh=outlier_threshold)
        .pipe(calculate_stats_from_random_points, minPoints=stats_min_points)
        .pipe(filter_columns, cols_to_keep=cols_to_keep)
    )

    return attach_spacecraft_column(
        processed_df=processed_df,
        raw_df=raw_df,
        spacecraft_col='Spacecraft',
        index_col=index_col,
        date_format=date_format
    )