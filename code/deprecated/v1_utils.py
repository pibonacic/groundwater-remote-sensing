import pandas as pd

from code.v1_preparation import load_and_prepare_data, handle_duplicate_values, handle_outliers, reindex_daily, handle_missing_values, calculate_stats_from_random_points, smooth_data, merge_datasets, slice_by_dates, add_time, add_lags

def process_remote(
        filepath: str,
        index_col: str = 'Timestamps',
        date_format: str = '%Y-%m-%d',
        sep: str = ',',
        filter_cols: bool = False,
        remote_cols: list = None,
        duplicates: bool = True,
        reindex: bool = True,
        outliers: bool = True,
        z_thresh: float = 3.0,
        ignore_vars: list = None,
        interpolation: bool = False,
        strategy: str = 'time',
        stats: bool = True,
        minPoints: int = 10,
        smooth: bool = False,
        cols: list = None,
        window_length: int = 7,
        polyorder: int = 2,
        ) -> pd.DataFrame:
    
    df = load_and_prepare_data(filepath, index_col, date_format, sep)

    if duplicates is True:
        df = handle_duplicate_values(df)

    if reindex is True:
        df = reindex_daily(df)

    if outliers is True:
        df = handle_outliers(df, z_thresh, ignore_vars)
    
    if stats is True:
        df = calculate_stats_from_random_points(df, minPoints)

    if filter_cols is True:
        df = df[remote_cols]

    if interpolation is True:
        df = handle_missing_values(df, strategy)

    if smooth is True:
        df = smooth_data(df, cols, window_length, polyorder)

    return df

def process_insitu(
        filepath: str,
        index_col: str = 'Timestamps',
        date_format: str = '%Y-%m-%d',
        sep: str = ',',
        filter_cols: bool = False,
        insitu_cols: list = None,
        duplicates: bool = True,
        reindex: bool = True,
        outliers: bool = True,
        z_thresh: float = 3.0,
        ignore_vars: list = None,
        interpolation: bool = False,
        strategy: str = 'time',
        smooth: bool = False,
        cols: list = None,
        window_length: int = 7,
        polyorder: int = 2,
        invertValues: bool = False, # Para invertir valor de profundidad (*-1)
        invertedCol: str = None
        ) -> pd.DataFrame:
    
    df = load_and_prepare_data(filepath, index_col, date_format, sep)

    if filter_cols is True:
        df = df[insitu_cols]

    if duplicates is True:
        df = handle_duplicate_values(df)

    if reindex is True:
        df = reindex_daily(df)

    if outliers is True:
        df = handle_outliers(df, z_thresh, ignore_vars)

    if interpolation is True:
        df = handle_missing_values(df, strategy)

    if smooth is True:
        df = smooth_data(df, cols, window_length, polyorder)

    if invertValues is True:
        df = df[invertedCol]*-1

    return df

def fuse_datasets(
        insitu: pd.DataFrame,
        remote: pd.DataFrame,
        startDate: str = None,
        endDate: str = None,
        dropNaN: bool = True,
        addTime: bool = False,
        addLags: bool = False,
        laggedFeatures: list = None,
        nLags: int = 32,
        intervalLags: int = 2,
        ) -> pd.DataFrame:

    df = merge_datasets(insitu, remote)

    df = slice_by_dates(df, startDate, endDate)

    if dropNaN is True:
        df = handle_missing_values(df, 'drop')

    if addTime is True:
        df = add_time(df)

    if addLags is True:
        df = add_lags(df, laggedFeatures, list(range(1, nLags, intervalLags)), list(range(1, nLags, intervalLags)))

    return df