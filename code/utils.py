import pandas as pd

from preparation import load_and_prepare_data, handle_duplicate_values, handle_outliers, reindex_daily, handle_missing_values, calculate_stats_from_random_points, smooth_data, merge_datasets, slice_by_dates

def process_remote(
        filepath: str,
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
    
    df = load_and_prepare_data(filepath)

    if duplicates is True:
        df = handle_duplicate_values(df)

    if reindex is True:
        df = reindex_daily(df)

    if outliers is True:
        df = handle_outliers(df, z_thresh, ignore_vars)

    if interpolation is True:
        df = handle_missing_values(df, strategy)
    
    if stats is True:
        df = calculate_stats_from_random_points(df, minPoints)

    if smooth is True:
        df = smooth_data(df, cols, window_length, polyorder)

    return df

def process_insitu(
        filepath: str,
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
        ) -> pd.DataFrame:
    
    df = load_and_prepare_data(filepath)

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

    return df

def fuse_datasets(
        insitu: pd.DataFrame,
        remote: pd.DataFrame,
        startDate: str = None,
        endDate: str = None,
        features: list = None
        ) -> pd.DataFrame:

    df = merge_datasets(insitu, remote)

    df = slice_by_dates(df, startDate, endDate)

    df = df[features]

    return df