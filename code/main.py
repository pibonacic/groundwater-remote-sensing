"""
Author: Pedro Bonacic Vera
Description:
"""

# probar incrementar numero de img input (HLS collection)
# probar inputs del balance de energía (SEBS)
# evaluar lags de la pp sobre nivel freatico o humedad para determinar patron de agregacion especial
# probar gradient boosting
# probar regresion multiple entre gw y vars insitu


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preparation import load_and_prepare_data, merge_datasets, remove_missing_values, remove_outliers, obs_data, preprocess_for_ML
from training import tune_model, train_model, evaluate_model, plot_pred_vs_real
from shap_analysis import calculate_shap_values, plot_shap

# -----------------------------
# 1. Load and prepare data
# -----------------------------

insitu_df = load_and_prepare_data('../data/processed/in-situ/insitu_training_data.csv')
remote_df = load_and_prepare_data('../data/processed/satellites/SDH1G30P01_sentinel2_bands_indices_TCT_202405-202601.csv')

merged_df = merge_datasets(insitu_df, remote_df)

target = 'SDH1PS01_gw-depth_m'

insitu_features = [
    'SDH1PS01_gw-depth_m',
    'ATMOS41_precipitation_mm',
    'ATMOS41_solar-radiation_Wm2',
    'ATMOS41_wind-speed_ms',
    'ATMOS41_air-temperature_degreeC',
    'TEROS12-15cm_water-content_m3m3',
    'TEROS12-15cm_soil-temperature_degreeC',
    'TEROS12-15cm_saturation-extract-ec_mScm',
    'TEROS12-30cm_water-content_m3m3',
    'TEROS12-30cm_soil-temperature_degreeC',
    'TEROS12-30cm_saturation-extract-ec_mScm',
    'TEROS12-48cm_water-content_m3m3',
    'TEROS12-48cm_soil-temperature_degreeC',
    'TEROS12-48cm_saturation-extract-ec_mScm',
    'TEROS21-25cm_soil-temperature_degreeC',
    'TEROS21-35cm_soil-temperature_degreeC',
]

remote_features = [
    's2_B2',
    's2_B3',
    's2_B4',
    's2_B8',
    's2_B11',
    's2_B12',
    's2_ndvi',
    's2_ndwi',
    's2_mndwi',
    's2_ndmi',
    's2_ndmi2',
    's2_srt',
    's2_brightness',
    's2_greenness',
    's2_wetness'
]

features = insitu_features + remote_features

filt_df = merged_df[features]
headings = filt_df.drop(columns=target).columns

clean_df1 = remove_missing_values(filt_df)
clean_df2 = remove_outliers(clean_df1)

obs_data(clean_df2)

X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_for_ML(clean_df2, target = target, test_size = 0.2, random_state = 42)


# -----------------------------
# 2. Model training
# -----------------------------

best_params = tune_model(X_train_scaled, y_train)

print(best_params)

model = train_model(X_train_scaled, y_train, best_params=best_params)

y_pred, results = evaluate_model(model, X_test_scaled, y_test)

plot_pred_vs_real(y_test, y_pred)


# -----------------------------
# 3. SHAP analysis
# -----------------------------

shap_values = calculate_shap_values(model, X_test_scaled)
plot_shap(shap_values, X_test_scaled, headings)

