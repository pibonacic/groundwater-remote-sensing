import pandas as pd
import numpy as np

def load_piezometer_data(filepath: str) -> pd.DataFrame:
    """
    Load raw piezometer CSV data and convert to a standardized DatetimeIndex. Assumes a specific CSV format.

    Parameters
    ----------
    filepath : str
        Path to the piezometer CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with a Datetime index.
    """
    # Skip metadata headers and set encoding for special characters
    df = pd.read_csv(filepath, header=9, encoding='latin1')

    # Merge separate date and time columns into a single temporal object
    date_str = df['Date'].astype(str)
    time_str = df['Time'].astype(str)
    df['Timestamps'] = pd.to_datetime(date_str + ' ' + time_str,
                                      format='%d-%m-%Y %H:%M:%S')
   
    df = df.set_index('Timestamps').sort_index()
    df = df.drop(columns=['Date', 'Time', 'ms'])    # Remove redundant columns

    return df

def calculate_gw_metrics(df: pd.DataFrame, piezometer_config: dict) -> pd.DataFrame:
    """
    Calculate groundwater depth and level based on sensor configuration and rename output columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw 'LEVEL' and 'TEMPERATURE' data.
    piezometer_config : dict
        Dictionary containing 'name', 'sensor_depth', 'elevation', and metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with calculated groundwater depth (m), level (masl), and renamed temperature.
    """
    # Define sensor prefix (site/ID) from the config
    prefix = piezometer_config['name']

    # Calculate actual water depth from raw 'LEVEL'. Multiply by -1 to represent a negative height.
    gw_depth = (piezometer_config['sensor_depth'] - df['LEVEL']) * -1
    # Reference depth to mean sea level
    gw_level = piezometer_config['elevation'] - gw_depth

    # Format columns names and remove raw data
    df[f'{prefix}_gw-depth_m'] = gw_depth
    df[f'{prefix}_gw-level_masl'] = gw_level
    df = df.rename(columns={'TEMPERATURE': f'{prefix}_gw-temp_degreeC'})
    df = df.drop(columns=['LEVEL'])

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.round(3)

    return df

def load_datalogger_data(filepath: str, datalogger_config: dict, datalogger_vars: dict) -> pd.DataFrame:
    """
    Parse multi-header datalogger files and extract specific variables by port. Assumes a specific CSV format.

    Parameters
    ----------
    filepath : str
        Path to the datalogger CSV file.
    datalogger_config : dict
        Dictionary containing sensor name and depth information per datalogger port and port configuration.
    datalogger_vars : list[str]
        List of specific standardized column names to retain.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame a Datetime index.
    """
    # Read only the first 3 rows to extract port and variable metadata
    headers = pd.read_csv(filepath, header=None, nrows=3)
    port_row = headers.iloc[0].astype(str).str.strip().tolist()     # ['Port1', 'Port2', ...]
    var_row = headers.iloc[2].astype(str).str.strip().tolist()      # ['mm Precipitation', 'm3/m3 Water Content' ...]

    cols_to_keep = [0]  # Index 0 is always 'Timestamps'
    rename_dict = {0: 'Timestamps'}

    # Loop through each column (starting from index 1)
    for i in range(1, len(port_row)):
        # Identify the current port
        port = port_row[i]  # 'Port1', 'Port2', ...

        if port in datalogger_config:

            # Extract the metadata from the config...
            prefix = datalogger_config[port]    # 'ATMOS41', 'TEROS12-48cm', ...
            # ... and the current row
            raw_var = var_row[i]                # 'mm Precipitation', 'm3/m3 Water Content', ...

            # Identify the base sensor from the prefix ('ATMOS41', 'TEROS12', 'TEROS21')
            base_sensor = prefix.split('-')[0]

            # Split the raw variable in unit and variable name and standarize formats
            parts = raw_var.strip().split(None, 1)

            if len(parts) > 1:
              # Standarize unit and var_name format
              unit = parts[0].replace('_', '').replace('/', '')
              var_name = parts[1].replace(' ', '-').lower()
              formatted_var = f'{var_name}_{unit}'

            # Manage variables without measurement unit (e.g. 'S2412_ndvi')
            else:
              var_name = parts[0].replace(' ', '-').lower()
              formatted_var = f'{var_name}'
              
            if base_sensor in datalogger_vars and formatted_var in datalogger_vars[base_sensor]:
                new_col_name = f'{prefix}_{formatted_var}'
                cols_to_keep.append(i)
                rename_dict[i] = new_col_name
        
    if len(cols_to_keep) == 1:
        return pd.DataFrame()   # Avoid errors if no variables matched
    
    # Read only identified relevant columns and rename with standarized names
    df = pd.read_csv(filepath, skiprows=3, header=None, usecols=cols_to_keep)
    df = df.rename(columns=rename_dict)

    df['Timestamps'] = pd.to_datetime(df['Timestamps'])
    df = df.set_index('Timestamps').sort_index()
    
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.round(3)

    return df

def remove_campaign_outliers(df: pd.DataFrame, campaigns: list[tuple], ignore_vars: list[str], z_thresh: float=3.0) -> pd.DataFrame:
    """
    Detect and mask outliers within specific date ranges using a Z-score threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a DatetimeIndex.
    campaigns : list[tuple[str, str]]
        List of (start_date, end_date) representing study campaigns.
    ignore_vars : list[str]
        Columns to exclude from outlier detection (e.g. precipitation).
    z_thresh : float, default=3.0
        Z-score threshold beyond which values are replaced by NaN.

    Returns
    -------
    pd.DataFrame
        DataFrame where identified outliers are replaced by NaN.
    """
    df_clean = df.copy()

    # Loop through each study campaign
    for start_date, end_date in campaigns:

        # Isolate the time window for the current campaign
        mask_time = (df_clean.index >= start_date) & (df_clean.index <= end_date)
        if not mask_time.any():
            continue    # Skip if no data exists for this specific date range
        
        # Select only 'normal' distributed numeric columns (e.g. excluding precipitation)
        cols_to_check = [c for c in df_clean.columns if c not in ignore_vars]
        if not cols_to_check:
            continue
        
        # Isolate the data slice for this campaign to calculate 'local' statistics
        df_slice = df_clean.loc[mask_time, cols_to_check]

        # Calculate Z-scores
        means = df_slice.mean()
        stds = df_slice.std().replace(0, 1)
        z_scores = (df_slice - means) / stds
        is_outlier = z_scores.abs() > z_thresh  # Evaluate outlier threshold

        # Loop through each column
        for col in cols_to_check:
            mask_outlier_col = is_outlier[col]  # Identify columns with outliers
            if mask_outlier_col.any():
                outlier_idx = df_slice.index[mask_outlier_col]  # Identify indexes with outliers
                df_clean.loc[outlier_idx, col] = np.nan     # Replace outlier values with NaN

    return df_clean


def aggregate_daily(df: pd.DataFrame, agg_rules: dict) -> pd.DataFrame:
    """
    Resample time series data to daily frequency using variable-specific rules.

    Parameters
    ----------
    df : pd.DataFrame
        High-frequency input DataFrame.
    agg_rules : dict
        Mapping of column names to aggregation methods (e.g, 'sum' for precipitation).

    Returns
    -------
    pd.DataFrame
        Daily aggregated DataFrame with rounded values.
    """
    agg_dict = {}

    for col in df.columns:
        # Identify the rule for each column
        rule = agg_rules.get(col, 'mean')   # Default rule is set to 'mean'

        # Applies a sum aggregation
        if rule == 'sum':
            # Ensures that if a day has only NaNs, the result is NaN (not 0.0)
            agg_dict[col] = lambda x: x.sum(min_count=1)

        # Applies a mean aggregation using only values > 0
        elif rule == 'mean_positive':
            agg_dict[col] = lambda x: x[x > 0].mean() if x[x > 0].count() > 0 else np.nan
        
        # Applies a circular mean aggregation for directions
        elif rule == 'circular_mean':
            agg_dict[col] = lambda x: (
                np.rad2deg(
                    np.arctan2(
                        np.sin(np.deg2rad(x.dropna())).mean(),
                        np.cos(np.deg2rad(x.dropna())).mean()
                    )
                ) + 360
            ) % 360 if x.dropna().count() > 0 else np.nan

        # Applies the default rule
        else:
            agg_dict[col] = rule
    
    # Resample to 'D' (daily) using the constructed dictionary of rules
    daily_df = df.resample('D').agg(agg_dict).round(3)
    
    return daily_df