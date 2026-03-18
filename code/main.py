"""
Author: Pedro Bonacic Vera
Description:
"""

# probar inputs del balance de energía (SEBS)
# evaluar lags de la pp sobre nivel freatico o humedad para determinar patron de agregacion especial
# probar gradient boosting
# probar regresion multiple entre gw y vars insitu


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preparation import load_and_prepare_data, merge_and_slice, handle_duplicate_values, remove_outliers, handle_missing_values, smooth_remote_data, obs_data, preprocess_for_ML
from training import tune_model, train_model, evaluate_model, apply_model, plot_pred_vs_real, plot_timeseries_results
from shap_analysis import calculate_shap_values, plot_shap

# -----------------------------
# 0. Feature selection
# -----------------------------

target = 'SDH1PS01_gw-depth_m'

insitu_features = [
    'SDH1PS01_gw-depth_m',
    # 'ATMOS41_precipitation_mm',
    # 'ATMOS41_solar-radiation_Wm2',
    # 'ATMOS41_wind-speed_ms',
    # 'ATMOS41_air-temperature_degreeC',
    # 'TEROS12-15cm_water-content_m3m3',
    # 'TEROS12-15cm_soil-temperature_degreeC',
    # 'TEROS12-15cm_saturation-extract-ec_mScm',
    # 'TEROS12-30cm_water-content_m3m3',
    # 'TEROS12-30cm_soil-temperature_degreeC',
    # 'TEROS12-30cm_saturation-extract-ec_mScm',
    # 'TEROS12-48cm_water-content_m3m3',
    # 'TEROS12-48cm_soil-temperature_degreeC',
    # 'TEROS12-48cm_saturation-extract-ec_mScm',
    # 'TEROS21-25cm_soil-temperature_degreeC',
    # 'TEROS21-35cm_soil-temperature_degreeC',
]

remote_features = [
    's2_B2',
    # 's2_B3',
    # 's2_B4',
    # 's2_B8',
    # 's2_B11',
    # 's2_B12',
    's2_ndvi',
    's2_ndwi',
    's2_mndwi',
    's2_ndmi',
    's2_ndmi2',
    's2_str',
    # 's2_brightness',
    # 's2_greenness',
    # 's2_wetness'
]

features = insitu_features + remote_features

# -----------------------------
# 1. Load and prepare data
# -----------------------------

insitu_df = load_and_prepare_data('../data/processed/in-situ/insitu_training_data.csv')
remote_df = load_and_prepare_data('../data/processed/satellites/SDH1G30P01_sentinel2_bands_indices_TCT_202405-202602.csv')

merged_df = merge_and_slice(insitu_df, remote_df, '2024-05-25', '2026-01-25')
features_df = merged_df[features]

no_duplicates_df = handle_duplicate_values(features_df)

no_outliers_df = remove_outliers(no_duplicates_df)

no_nan_df = handle_missing_values(no_outliers_df, 'linear')

smooth_df = smooth_remote_data(no_nan_df, remote_features, 29, 2)

obs_data(smooth_df)

X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_dates = preprocess_for_ML(smooth_df, target = target, test_size = 0.2, random_state = 42)


# -----------------------------
# 2. Model training
# -----------------------------

# best_params = tune_model(X_train_scaled, y_train)
# print(best_params)

model = train_model(X_train_scaled, y_train, 
                    # best_params=best_params
                    )

y_pred, results = evaluate_model(model, X_test_scaled, y_test)

results_df = apply_model(smooth_df, target, scaler, model)

results_df.to_csv('../outputs/obs_pred.csv')

plot_pred_vs_real(y_test, y_pred)

plot_timeseries_results(results_df)

# -----------------------------
# 3. SHAP analysis
# -----------------------------

headings = features_df.drop(columns=target).columns
shap_values = calculate_shap_values(model, X_test_scaled)
plot_shap(shap_values, X_test_scaled, headings)