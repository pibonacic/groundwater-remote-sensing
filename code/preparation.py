import pandas as pd
import numpy as np
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
    df_copy = df.copy()
    return df_copy.groupby(df_copy.index).mean()


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


def handle_outliers(df: pd.DataFrame, z_thresh: float = 3.0, ignore_vars: list = None) -> pd.DataFrame:
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
        - 'time' : interpolate NaN using time method

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df_copy = df.copy()

    if strategy == 'drop':  # Removes any row with a NaN
        return df_copy.dropna()
    
    elif strategy == 'time':  # Fill gaps by interpolating values
        return df_copy.interpolate(method='time', 
                                   #limit=6,
                                   limit_direction='forward'
                                   )
    
    else:
        raise ValueError('Strategy not supported')


# NO ESTA FUNCIONADO AL 100; NO FILTRA CORRECTAMENTE LAS FECHAS CON MENOS DE 10 OBS VALIDAS
def calculate_stats_from_random_points(df: pd.DataFrame) -> pd.DataFrame:
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

    # Dictionary to store the calculated series
    stats_results = {}

    # Iterate over each individual feature
    for feature in feature_names:
        # Create a subset of columns that belong to the current feature (e.g ndvi_p1, ..., ndvi_pN)
        feature_columns = [col for col in df.columns if col.startswith(f'{feature}_p')]
        feature_data = df[feature_columns]

        # Identify and keep rows (dates) that have at least N valid points
        valid_rows_mask = feature_data.notna().sum(axis=1) >= 10
        valid_rows = feature_data[valid_rows_mask]

        # Calculate statistics row-wise (axis=1) across all points
        stats_results[f'{feature}_mean'] = valid_rows.mean(axis=1)
        stats_results[f'{feature}_std'] = valid_rows.std(axis=1)
        stats_results[f'{feature}_median'] = valid_rows.median(axis=1)
        stats_results[f'{feature}_max'] = valid_rows.max(axis=1)
        stats_results[f'{feature}_min'] = valid_rows.min(axis=1)
        stats_results[f'{feature}_p25'] = valid_rows.quantile(0.25, axis=1)
        stats_results[f'{feature}_p75'] = valid_rows.quantile(0.75, axis=1)

    # Construct the final dataframe maintaining the original datetime index
    output_df = pd.DataFrame(stats_results, index=df.index)

    return output_df


def smooth_data(df: pd.DataFrame, cols: list = None, window_length: int = 11, polyorder: int = 2) -> pd.DataFrame:
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


def merge_datasets(insitu_df: pd.DataFrame, remote_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join in-situ measurements with remote sensing time series using 
    the intersection of their DatetimeIndices

    Parameters
    ----------
    insitu_df : pd.DataFrame
        Dataframe containing in-situ measurements.
    remote_df : pd.DataFrame
        Dataframe containing satellite-derived data.
    
    Returns
    ----------
    pd.DataFrame
        A merged Dataframe with satellite-derived data aligned to the in-situ measurement dates.
    """
    # Join datasets using an inner join
    return insitu_df.join(remote_df, how='inner')


def slice_by_dates(df: pd.DataFrame, startDate: str = None, endDate: str = None) -> pd.DataFrame:
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


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    df_copy = df.copy()

    doy = df_copy.index.dayofyear

    #df_copy['doy'] = doy
    df_copy['doy_sin'] = np.sin(2*np.pi*doy/365.25)
    #df_copy['doy_cos'] = np.cos(2*np.pi*doy/365.25)
    
    return df_copy


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

    # Save test data dates
    test_dates = y_test.index

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
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_dates


def preprocess_for_ML_chrono(df: pd.DataFrame, target: str, train_size: float = 0.6):
  """
  Preprocess the dataset for ML using a chronological split.

  Steps:
  1. Calculate the split point based on the specified train_size.
  2. Divide the data into training (first part) and testing (remainder) sets.
  3. Standardize features using only training statistics to avoid leakage.

    ...

  """
  # Define features (X) and target (y)
  X = df.drop(columns=[target])
  y = df[target]

  # Calculate the integer index for the chronological split
  split_idx = int(len(df) * train_size)

  # Split data maintaining temporal order
  X_train = X.iloc[:split_idx]
  X_test = X.iloc[split_idx:]
  y_train = y.iloc[:split_idx]
  y_test = y.iloc[split_idx:]

  # Store dates for future time-series visualization
  test_dates = y_test.index

  print(f"Chronological split applied at: {y_train.index[-1].date()}")
  print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

  # Standardize: Fit on training data and transform both sets
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Flatten targets for Scikit-Learn compatibility
  y_train = y_train.to_numpy().ravel()
  y_test = y_test.to_numpy().ravel()

  return X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_dates