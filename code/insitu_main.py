import os
import glob
import pandas as pd

# Import local configuration and utility functions
from insitu_config import dirs, campaigns, piezometer_config, datalogger_config, datalogger_vars, no_outlier_check, aggregation_rules
from insitu_utils import load_piezometer_data, calculate_gw_metrics, load_datalogger_data, remove_campaign_outliers, aggregate_daily

# -----------------------------
# 1. Piezometer preprocessing
# -----------------------------

processed_piezometers = {}

# Iterate over each piezometer defined in the metadata config
for piezo_id, properties in piezometer_config.items():

    # Scan for files matching the current piezometer ID
    pattern = os.path.join(dirs['piezometers'], f'{piezo_id}_*compensated.csv')
    files = sorted(glob.glob(pattern))

    piezo_name = properties['name']
    raw_dfs = []

    # Batch load all matched files
    for filepath in files:
        df = load_piezometer_data(filepath)
        raw_dfs.append(df)

    if raw_dfs:
        # Concatenate records and apply the processing pipeline
        full_df = pd.concat(raw_dfs, axis=0).sort_index()
        metrics_df = calculate_gw_metrics(full_df, properties)
        clean_df = remove_campaign_outliers(metrics_df, campaigns, no_outlier_check)
        daily_df = aggregate_daily(clean_df, aggregation_rules)

        # Store the daily aggregated time series in a dictionary, keyed by piezo name
        processed_piezometers[piezo_name] = daily_df

        print(f'Piezometer {piezo_name} (ID: {piezo_id})')
        print(f'  - Processed files: {len(files)}')
        print(f'  - Unique days: {len(daily_df)}')
        print(f'  - Temporal range: {daily_df.index.min()} to {daily_df.index.max()}')
        print(f'  - Generated columns: {list(daily_df.columns)}')
        print('-' * 50)

    else:
        print(f'No files found for piezometer {piezo_name} (ID: {piezo_id})')
        print('-' * 50)

# Concatenate (by date) all processed piezometers into a DataFrame
all_piezos_daily_df = pd.concat(processed_piezometers.values(), axis=1)

# -----------------------------
# 2. Datalogger preprocessing
# -----------------------------

# Scan for datalogger files matching the sensor prefix ('z6')
dlog_pattern = os.path.join(dirs['dataloggers'], 'z6*.csv')
dlog_files = sorted(glob.glob(dlog_pattern))

dlog_raw_dfs = []

# Iterate over each datalogger file
for filepath in dlog_files:
    
    # Extract logger ID and Config versions from filename structure
    filename = os.path.basename(filepath)
    try:
        logger_id = filename.split('(')[0].strip()
        config_id = [p for p in filename.split('-') if 'Configuration' in p][0].strip()
    except IndexError:
        continue    # Skip files without standarized naming

    # Retrieve the specific port-to-sensor map for this logger version
    current_map = datalogger_config.get(logger_id, {}).get(config_id)

    if current_map:
        # Load and rename variables based on the active configuration map
        df = load_datalogger_data(filepath, current_map, datalogger_vars)
        dlog_raw_dfs.append(df)

        print(f'Logger {logger_id} | Config: {config_id}')
        print(f'  - Extracted variables: {list(df.columns)}')
        print("-" * 10)

    else:
        print(f'No configuration map available for {logger_id} - {config_id}')

# Concatenate records and apply the processing pipeline
if dlog_raw_dfs:
    dlog_full_df = pd.concat(dlog_raw_dfs, axis=0).sort_index()
    dlog_clean_df = remove_campaign_outliers(dlog_full_df, campaigns, no_outlier_check)
    dlog_daily_df = aggregate_daily(dlog_clean_df, aggregation_rules)

    print('Compiled datalogger file')
    print(f'  - Processed files: {len(dlog_files)}')
    print(f'  - Unique days: {len(dlog_daily_df)}')
    print(f'  - Temporal range: {dlog_daily_df.index.min()} to {dlog_daily_df.index.max()}')
    print(f'  - Generated columns: {list(dlog_daily_df.columns)}')
    print('-' * 50)

else:
    print(f'No files found in {dlog_pattern}')
    print('-' * 50)


# -----------------------------
# 3. Merge and export
# -----------------------------

# # Select only data from the target piezometer
# target_piezo = processed_piezometers['SDH1PS01']

# Join datasets by dates. Outer join preserves all available data
insitu_df = all_piezos_daily_df.merge(dlog_daily_df, left_index=True, right_index=True, how='outer')

# Export data as csv
os.makedirs(dirs['processed'], exist_ok=True)
output_filename = 'daily-insitu-data.csv'
output_path = os.path.join(dirs['processed'], output_filename)
insitu_df.to_csv(output_path)

print('Merged piezometer-datalogger file')
print(f'  - Unique days: {len(insitu_df)}')
print(f'  - Temporal range: {insitu_df.index.min()} to {insitu_df.index.max()}')
print(f'  - Generated columns: {list(insitu_df.columns)}')
print("-" * 50)