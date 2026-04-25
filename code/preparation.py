# Este script asume que los datos de entrada están en formato csv (sep=,), que son dos: uno con datos remotos (bandas, indices)
# y otro con datos insitu (pozos, pp, temp, humedad CE de suelo), que tienen una columna de fecha con separación diaria,
# que los datos remotos tienen filas (dias) con NaN los dias de no paso, que la primera fila es encabezado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_prepare_data(filepath: str, index_col: str='Timestamps', date_format: str='%Y-%m-%d', sep: str=',') -> pd.DataFrame:
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
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


def merge_and_slice(insitu_df: pd.DataFrame, remote_df: pd.DataFrame, startDate: str = None, endDate: str = None) -> pd.DataFrame:
    """
    Join in-situ measurements with remote sensing time series using their DatetimeIndices 
    and optionally clip them to a specified time range.

    Parameters
    ----------
    insitu_df : pd.DataFrame
        Dataframe containing in-situ measurements.
    remote_df : pd.DataFrame
        Dataframe containing satellite-derived data.
    startDate : str, optional
        Date in %Y-%m-%d format
    endDate : str, optional
        Date in %Y-%m-%d' format
    Returns
    ----------
    pd.DataFrame
        A merged Dataframe with satellite-derived data aligned to the in-situ measurement dates,
        clipped to a specified time range.
    """
    # Join insitu and remote data. Left join keeps all insitu records to assure index continuity
    df = insitu_df.join(remote_df, how='left')

    # Clip data to a study period if defined
    if startDate is not None or endDate is not None:
        df = df.loc[startDate:endDate]
    return df


def handle_duplicate_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate records with the same timestamp by calculating the daily mean.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame without duplicated values.
    """
    return df.groupby(df.index).mean()


def remove_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Remove any row that contains at least one outlier, defined by a column-wise Z-score threshold.
    Ignores NaNs.
    
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

    # Calculate zscore for each column, omiting NaNs
    z_scores = zscore(numeric_df, nan_policy='omit')

    # Filter out any row containing at least one outlier
    is_normal = (np.abs(z_scores) <= z_thresh) | (np.isnan(z_scores))
    clean_rows = is_normal.all(axis=1)
    return df[clean_rows]  


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the DataFrame according to a specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strategy : str, default = 'drop
        Strategy to handle missing values:
        - 'drop' : remove rows containing any NaN
        - 'linear' : interpolate NaN using linear method

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    if strategy == 'drop':  # Removes any row with a NaN
        return df.dropna()
    
    elif strategy == 'linear':  # Fill gaps using surrounding values
        return df.interpolate(method='linear', limit_direction='both')
    
    else:
        raise ValueError('Strategy not supported')


def smooth_remote_data(df: pd.DataFrame, cols: list = None, window_length: int = 11, polyorder: int = 2) -> pd.DataFrame:
    """
    Apply a Savitzky-Golay filter to smooth a subset of columns in a DataFrame while preserving trends.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame without datetime gaps.
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


def obs_data(df):
    """
    Display basic statistics and plots for the DataFrame.
    
    Statistics include count, mean, standard deviation, min, and max for each column.
    Plots include histograms and time series for visual inspection of data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe for observation.
    """
    # Pivot to long format for faceted plotting with Seaborn
    df_long = df.reset_index().melt(id_vars=df.index.name)

    print('\n--- Dataframe info ---')
    print(f'Unique days: {len(df)}')
    print(f'Temporal range: {df.index.min()} to {df.index.max()}')

    print('\n--- Descriptive stats ---')
    print(df.describe().T[['count', 'mean', 'std', 'min', 'max']])

    print('\n--- Histograms ---')
    g_hist = sns.displot(data=df_long, x='value', col='variable', col_wrap=4, kde=True,
                         bins=15, color='#4C72B0', edgecolor='white', linewidth = 1.5,
                         height=3.5, aspect=1.2, common_bins=False, facet_kws={'sharex': False, 'sharey': False})
    plt.show()

    print('\n--- Time series ---')
    g_line = sns.relplot(data=df_long, x=df.index.name, y='value', col='variable', marker='o', markersize=3,
                         col_wrap=4, kind='line', facet_kws={'sharex': True, 'sharey': False})
    g_line.figure.autofmt_xdate()
    plt.show()


def preprocess_for_ML(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the dataset for machine learning.

    Steps:
    1. Split the dataframe into features (X) and target (y).
    2. Split the data into training and testing sets.
    3. Standardize the feature data (zero mean, unit variance).
    4. Convert target arrays to 1D numpy arrays.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target : str
        Name of the target column.
    test_size : float, default=0.2
        Fraction of the dataset to use as the test set.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train_scaled : np.ndarray
        Standardized training features.
    X_test_scaled : np.ndarray
        Standardized testing features.
    y_train : np.ndarray
        Training target values.
    y_test : np.ndarray
        Testing target values.
    scaler : StandardScaler
        Fitted scaler object for possible inverse transformations.
    """
    # Isolate features (X) and target (y) 
    X = df.drop(columns=[target])
    y = df[target]

    # Split data for training and testing. Random seed is defined for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("Training data: ")
    print("Count x: ", X_train.count())
    print("Count y: ", y_train.count())
    print("Testing data: ")
    print("Count x", X_test.count())
    print("Count y: ", y_test.count())

    # Define a scaler for data standarization (mean=0, std=1)
    scaler = StandardScaler()

    # Standarize features. Fit on training only to avoid data leakeage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Flatten target for Scikit-Learn compatibility
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler