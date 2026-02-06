# Este script asume que los datos de entrada están en formato csv (sep=,), que son dos: uno con datos remotos (bandas, indices)
# y otro con datos insitu (pozos, pp, temp, humedad CE de suelo), que tienen una columna de fecha con separación diaria,
# que los datos remotos tienen filas (dias) con NaN los dias de no paso, que la primera fila es encabezado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_and_prepare_data(file_path: str, index_col: str='Timestamps', date_format: str='%Y-%m-%d', sep: str=',') -> pd.DataFrame:
    """
    Load a CSV file, set a DateimeIndex and convert all columns to numeric, coercing errors to NaN.
    
    Parameters
    ----------
    file_path : str
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
    df = pd.read_csv(file_path, sep=sep)

    df[index_col] = pd.to_datetime(df[index_col], format=date_format)
    df.set_index(index_col, inplace=True)

    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def merge_datasets(insitu_df: pd.DataFrame, remote_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join in-situ measurements with remote sensing time series using their DatetimeIndices

    Parameters
    ----------
    insitu_df: pd.DataFrame
        Dataframe containing in-situ measurements.
    remote_df: pd.DataFrame
        Dataframe containing satellite-derived data.

    Returns
    ----------
    pd.DataFrame
        A merged Dataframe with satellite-derived data aligned to the in-situ measurement dates.
    """
    df = insitu_df.join(remote_df, how='left')
    return df

def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any row that contains at least one NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame without missing values.
    """
    return df.dropna()


def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Remove any row that contains at least one outlier, defined by a column-wise Z-score threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    z_thresh : float, default=3.0
        Z-score threshold for defining outliers.
    
    Returns
    -------
    pd.DataFrame
        DataFrame without outliers.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    z_scores = zscore(numeric_df)               # Calculate zscore for each column
    is_normal = np.abs(z_scores) < z_thresh     # Evaluate the zscore against the threshold
    clean_rows = is_normal.all(axis=1)          # Identify rows where all values are within normal range

    return df[clean_rows]                       # Apply the filter


def obs_data(df):
    """
    Display basic statistics and histograms for the DataFrame.
    
    Statistics include count, mean, standard deviation, min, and max for each column.
    Also plots histograms for visual inspection of distributions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe for observation.
    """
    mean_values = df.mean()
    std_values = df.std()
    min_values = df.min()
    max_values = df.max()

    statistics_table = pd.DataFrame({
        'Count': df.count(),
        'Mean': mean_values,
        'StdDev': std_values,
        'Min': min_values,
        'Max': max_values
    })

    print(statistics_table)

    print('\nHistograms')
    with plt.rc_context({'axes.formatter.useoffset': False}):
        df.hist(bins=15, figsize=(10, 10))


def preprocess_for_ML(df: pd.DataFrame, target: str, test_size: float = 0.2, random_seed: int = 42):
    return