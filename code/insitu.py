import os
import glob
import pandas as pd

import insitu_config as config
import insitu_utils as utils


# -----------------------------
# 1. Piezometer preprocessing
# -----------------------------

processed_piezometers = {}

for piezo_id, properties in config.piezometer_config.items():

    pattern = os.path.join(config.dirs['piezometers'], f'{piezo_id}_*compensated.csv')
    files = sorted(glob.glob(pattern))

    piezo_name = properties['name']
    raw_dfs = []

    for filepath in files:
        df = utils.load_piezometer_data(filepath)
        raw_dfs.append(df)

    if raw_dfs:
        full_df = pd.concat(raw_dfs, axis=0).sort_index()
        metrics_df = utils.calculate_gw_metrics(full_df, properties)
        clean_df = utils.remove_campaign_outliers(metrics_df, config.campaigns, config.no_outlier_check)
        daily_df = utils.aggregate_daily(clean_df, config.aggregation_rules)

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


# -----------------------------
# 2. Datalogger preprocessing
# -----------------------------

dlog_pattern = os.path.join(config.dirs['dataloggers'], 'z6*.csv')
dlog_files = sorted(glob.glob(dlog_pattern))

dlog_raw_dfs = []

for filepath in dlog_files:
    
    filename = os.path.basename(filepath)
    try:
        logger_id = filename.split('(')[0].strip()
        config_id = [p for p in filename.split('-') if 'Configuration' in p][0].strip()
    except IndexError:
        continue

    current_map = config.datalogger_config.get(logger_id, {}).get(config_id)

    if current_map:
        df = utils.load_datalogger_data(filepath, current_map, config.datalogger_vars)
        dlog_raw_dfs.append(df)

        print(f'Logger {logger_id} | Config: {config_id}')
        print(f'  - Extracted variables: {list(df.columns)}')
        print("-" * 10)

    else:
        print(f'No configuration map available for {logger_id} - {config_id}')

if dlog_raw_dfs:
    dlog_full_df = pd.concat(dlog_raw_dfs, axis=0).sort_index()
    dlog_clean_df = utils.remove_campaign_outliers(dlog_full_df, config.campaigns, config.no_outlier_check)
    dlog_daily_df = utils.aggregate_daily(dlog_clean_df, config.aggregation_rules)

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

os.makedirs(config.dirs['outputs'], exist_ok=True)

target_piezo = processed_piezometers['SDH1PS01']

training_df = target_piezo.merge(dlog_daily_df, left_index=True, right_index=True, how='outer')

output_filename = 'insitu_training_data.csv'
output_path = os.path.join(config.dirs['outputs'], output_filename)

training_df.to_csv(output_path)

print('Merged piezometer-datalogger file')
print(f'  - Unique days: {len(training_df)}')
print(f'  - Temporal range: {training_df.index.min()} to {training_df.index.max()}')
print(f'  - Generated columns: {list(training_df.columns)}')
print("-" * 50)