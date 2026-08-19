"""
Author: Pedro Bonacic Vera
Description:
"""

from v2_preparation import process_modis, process_landsat, process_insitu, load_and_prepare_data, slice_by_dates
from v2_sensor_fusion import fuse_sensors
from v2_visualization import (
    obs_data, format_metrics_table, plot_landsat_bands, plot_residuals_boxplot, 
    plot_pred_vs_real, plot_timeseries_results, plot_timeseries_results2, plot_shap)

from v2_features_processing import(
    process_spectral_data, principal_component_analysis, cross_correlations_analysis, merge_datasets, add_lags, add_time)

from v2_modelling_framework import train_and_evaluate_RF, predict_dataset

import shap

# -----------------------------
# 1. Study site selection and config
# -----------------------------

study_site = 'SDH1'
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

nLags = 3
intervalLags = 1
lags_list = list(range(1, nLags, intervalLags))

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

indices_df = process_spectral_data(sensor_fusion_results['anchored'])

pca_df = principal_component_analysis(indices_df, feature_columns=features, n_components=2, random_state=42)

# cc_df = merge_datasets(insitu_df, pca_df, how='outer')
# cc_df2 = cross_correlations_analysis(cc_df, target_col=target, feature_cols=['PC_1', 'PC_2'], max_lag=10)

time_df = (
    pca_df
    .pipe(add_lags, features=['PC_1', 'PC_2'], past_lags=lags_list, future_lags=lags_list)
    .pipe(add_time)
)

training_df = merge_datasets(insitu_df, time_df, how='inner')


# -----------------------------
# 5. Base model training and evaluation
# -----------------------------

rf_results = train_and_evaluate_RF(
    df=training_df,
    target=target,
    split_strategy='random',
    train_size=0.5,
    scale_features=False,
    tune_hyperparameters=True,
    cv_strategy='random',
    cv_splits=3,
    compute_shap=True
)

# Visualizaciones directas
plot_timeseries_results(df=rf_results['predictions_df'])
plot_pred_vs_real(y_test=rf_results['y_test'], y_pred=rf_results['y_pred_test'])

# Resumen de interpretabilidad
print(rf_results['feature_importances'].head(10))
shap.plots.beeswarm(rf_results['shap_values'])

# -----------------------------
# 5. Predictions over satellite archive
# -----------------------------

indices_2000_2026 = process_spectral_data(sensor_fusion_results['anchored'])

pca_2000_2026 = principal_component_analysis(
    indices_2000_2026,
    feature_columns=features,
    n_components=2,
    random_state=42
)

time_2000_2026 = (
    pca_2000_2026
    .pipe(add_lags, features=['PC_1', 'PC_2'], past_lags=lags_list, future_lags=lags_list)
    .pipe(add_time)
)

predictions_2000_2026 = predict_dataset(
    df=time_2000_2026,
    model=rf_results['model'],
    scaler=rf_results['scaler']
)
plot_timeseries_results2(df=predictions_2000_2026)