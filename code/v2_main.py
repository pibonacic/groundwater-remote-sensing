"""
Author: Pedro Bonacic Vera
Description:
"""

from v2_data_preparation import process_modis, process_landsat, process_insitu
from v2_sensor_fusion import fuse_sensors
from v2_feature_engineering import build_features_dataset, join_datasets, build_features_dataset_v2
from v2_modelling_framework import train_and_evaluate_RF, predict_dataset

from v2_visualization import (
    obs_data, format_metrics_table, plot_landsat_bands, plot_residuals_boxplot, 
    plot_pred_vs_real, plot_timeseries_results, plot_timeseries_results2, plot_shap)

import shap

# -----------------------------
# 1. Study site selection and config
# -----------------------------

study_site = 'SDH2'
start_date = '2000'
end_date = '2026'

if study_site == 'SDH1':
    target = 'SDH1PS01_gw-depth_m'
    insitu_filepath = '../data/processed/in-situ/SDH1_daily-insitu-data.csv'
    modis_filepath = '../data/processed/satellites/SDH1_MT_500m_2000-03_2026-06.csv'
    landsat_filepath = '../data/processed/satellites/SDH1_L_30m_1984-07_2026-06.csv'

elif study_site == 'SDH2':
    target = 'SDH2PP01_gw-depth_m'
    insitu_filepath = '../data/processed/in-situ/SDH2_daily-insitu-data.csv'
    modis_filepath = '../data/processed/satellites/SDH2_MT_500m_2000-03_2026-06.csv'
    landsat_filepath = '../data/processed/satellites/SDH2_L_30m_1984-07_2026-06.csv'

else:
    print(f'{study_site} not supported as study site')

band_mapping = {
    'blue':  ('blue_mean',  'blue'),
    'green': ('green_mean', 'green'),
    'red':   ('red_mean',   'red'),
    'nir':   ('nir_mean',   'nir'),
    'swir1': ('swir1_mean', 'swir1'),
    'swir2': ('swir2_mean', 'swir2')
}

modis_cols = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
landsat_cols = ['blue_mean', 'green_mean', 'red_mean', 'nir_mean', 'swir1_mean', 'swir2_mean']

features = [
    'ndvi',
    'gndvi',
    'ndwi',
    'mndwi',
    'ndmi',
    'ndmi2',
    'str1',
    'str2',
]

# -----------------------------
# 2. Input data preprocessing
# -----------------------------

modis_df = process_modis(
    filepath=modis_filepath,
    start_date=start_date,
    end_date=end_date,
    cols_to_keep=modis_cols,
    outlier_threshold=3.0,
    smooth_window=31
)

landsat_df = process_landsat(
    filepath=landsat_filepath,
    start_date=start_date,
    end_date=end_date,
    cols_to_keep=landsat_cols,
    outlier_threshold=3.0,
    stats_min_points=10
)

insitu_df = process_insitu(
    filepath=insitu_filepath,
    start_date='2024-05-24',
    end_date='2026-05-14',
    cols_to_keep=target,
    outlier_threshold=3.0
    )

# -----------------------------
# 3. Daily satellite series
# -----------------------------

sensor_fusion_results = fuse_sensors(
    modis_df=modis_df,
    landsat_df=landsat_df,
    band_mapping=band_mapping,
    test_size=0.3,
    sigma_threshold=3.0,
    verbose_summary=False,
    verbose_plots=False
)

# plot_residuals_boxplot(sensor_fusion_results['residuals'])
# plot_landsat_bands(
#     daily_df=sensor_fusion_results['anchored'],
#     original_df=landsat_df,
#     residuals_df=sensor_fusion_results['residuals_interpolated'],
#     start='2021',
#     end='2026'
# )
# plot_landsat_bands(
#     daily_df=sensor_fusion_results['predicted'],
#     original_df=landsat_df,
#     residuals_df=sensor_fusion_results['residuals'],
#     start='2021',
#     end='2026'
# )

metrics_df = format_metrics_table(sensor_fusion_results['metrics'])
print(metrics_df.to_string())


# -----------------------------
# 4. Predictors and target processing
# -----------------------------

# MODIFICADO PARA NO ACEPTAR ROLLINGS
features_dataset, cc_report, pca_model = build_features_dataset(
    remote_df=sensor_fusion_results['predicted'],
    insitu_df=insitu_df,
    target_col=target,
    spectral_bands_cols=modis_cols,
    pca_components=2,
    verbose_pca=True,
    rolling_windows=[14],
    rolling_stats=['mean'],
    max_cc_lag=8,
    min_corr_thresh=0.1,
    top_k_lags=5,
    random_state=42
)

# features_dataset = build_features_dataset_v2(
#     remote_df=sensor_fusion_results['predicted'],
#     spectral_bands_cols=modis_cols,
#     pca_components=3,
#     verbose_pca=True,
#     past_lags=[1, 2],
#     future_lags=[1, 2],
#     random_state=42
# )

training_df = join_datasets(
    df1=insitu_df[[target]],
    df2=features_dataset,
    how='inner'
    )


print(f'\nTraining dataset: {training_df.shape[0]} rows, {training_df.shape[1]} columns')


# -----------------------------
# 5. Base model training and evaluation
# -----------------------------

rf_results = train_and_evaluate_RF(
    df=training_df,
    target=target,
    split_strategy='chrono_train_first',    # chrono_train_first, chrono_test_first, random
    train_size=0.5,
    scale_features=False,
    tune_hyperparameters=True,
    cv_strategy='chrono',                   # chrono, random
    cv_splits=3,
    compute_shap=True
)

# Visualizaciones directas
plot_timeseries_results(df=rf_results['predictions_df'])
plot_pred_vs_real(y_test=rf_results['y_test'], y_pred=rf_results['y_pred_test'])

# Resumen de interpretabilidad
# print(rf_results['feature_importances'].head(10))
shap.plots.beeswarm(rf_results['shap_values'])

# -----------------------------
# 5. Predictions over satellite archive
# -----------------------------

predictions_2000_2026 = predict_dataset(
    df=features_dataset,
    model=rf_results['model'],
    scaler=rf_results['scaler']
)

predictions_2000_2026.to_csv(
    f'../outputs/predictions/{study_site}_{start_date}-{end_date}_suffix.csv',
    index_label='Timestamps'
)

# plot_timeseries_results2(df=predictions_2000_2026)