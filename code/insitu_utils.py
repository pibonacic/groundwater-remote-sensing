import pandas as pd
import numpy as np
from scipy.stats import zscore

def load_piezometer_data(filepath: str) -> pd.DataFrame:
    """
    
    """
    df = pd.read_csv(filepath, header=9, encoding='latin1')

    date_str = df['Date'].astype(str)
    time_str = df['Time'].astype(str)
    df['Timestamps'] = pd.to_datetime(date_str + ' ' + time_str,
                                      format='%d-%m-%Y %H:%M:%S')

    df = df.set_index('Timestamps').sort_index()

    df = df.drop(columns=['Date', 'Time', 'ms'])

    return df

def calculate_gw_metrics(df: pd.DataFrame, piezometer_config: dict) -> pd.DataFrame:
    """
    
    """
    prefix = piezometer_config['name']

    gw_depth = (piezometer_config['sensor_depth'] - df['LEVEL']) * -1
    gw_level = piezometer_config['elevation'] - gw_depth

    df[f'{prefix}_gw-depth_m'] = gw_depth
    df[f'{prefix}_gw-level_masl'] = gw_level
    
    df = df.rename(columns={'TEMPERATURE': f'{prefix}_gw-temp_degreeC'})
    df = df.drop(columns=['LEVEL'])

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.round(3)

    return df

def load_datalogger_data(filepath: str, datalogger_config: dict, datalogger_vars: list) -> pd.DataFrame:
    """
    
    """
    headers = pd.read_csv(filepath, header=None, nrows=3)
    port_row = headers.iloc[0].astype(str).str.strip().tolist()     # Port1, Port2, ...
    var_row = headers.iloc[2].astype(str).str.strip().tolist()      # m3/m3 Water Content, mm Precipitation, ...

    cols_to_keep = [0]
    rename_dict = {0: 'Timestamps'}

    for i in range(1, len(port_row)):
        port = port_row[i]

        if port in datalogger_config:
            prefix = datalogger_config[port]
            raw_var = var_row[i]

            parts = raw_var.split(' ', 1)
            unit = parts[0].replace('_', '').replace('/', '')
            var_name = parts[1].replace(' ', '-').lower()

            new_col_name = f'{prefix}_{var_name}_{unit}'

            if new_col_name in datalogger_vars:
                cols_to_keep.append(i)
                rename_dict[i] = new_col_name
        
    if len(cols_to_keep) == 1:
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, skiprows=3, header=None, usecols=cols_to_keep)
    df = df.rename(columns=rename_dict)

    df['Timestamps'] = pd.to_datetime(df['Timestamps'])
    df = df.set_index('Timestamps').sort_index()

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.round(3)

    return df

def remove_campaign_outliers(df: pd.DataFrame, campaigns: list, ignore_vars: list, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    
    """
    df_clean = df.copy()

    for start_date, end_date in campaigns:

        mask_time = (df_clean.index >= start_date) & (df_clean.index <= end_date)
        if not mask_time.any():
            continue

        cols_to_check = [c for c in df_clean.columns if c not in ignore_vars]
        if not cols_to_check:
            continue
        
        df_slice = df_clean.loc[mask_time, cols_to_check]

        means = df_slice.mean()
        stds = df_slice.std().replace(0, 1)
        z_scores = (df_slice - means) / stds

        is_outlier = z_scores.abs() > z_thresh

        for col in cols_to_check:
            mask_outlier_col = is_outlier[col]
            if mask_outlier_col.any():
                outlier_idx = df_slice.index[mask_outlier_col]
                df_clean.loc[outlier_idx, col] = np.nan

    return df_clean


def aggregate_daily(df: pd.DataFrame, agg_rules: list) -> pd.DataFrame:
    """
    
    """
    if df.empty:
        return df
    
    agg_dict ={}

    for col in df.columns:
        rule = agg_rules.get(col, 'mean')
        
        if rule == 'sum':
            agg_dict[col] = lambda x: x.sum(min_count=1)
        else:
            agg_dict[col] = rule
    
    daily_df = df.resample('D').agg(agg_dict)
    
    return daily_df.round(3)