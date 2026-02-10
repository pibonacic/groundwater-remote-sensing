"""
Author: Pedro Bonacic Vera
Description:
"""

# probar incluir variables in-situ (por si solas e hibrido): pp, theta, soilTemp, soilEC 15cm
# probar gradient boosting
# probar inputs del balance de energía (SEBS)
# probar incrementar numero de img input (HLS collection)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preparation import load_and_prepare_data, merge_datasets, remove_missing_values, remove_outliers, obs_data, preprocess_for_ML
from training import tune_model, train_model, evaluate_model, plot_pred_vs_real
from shap_analysis import calculate_shap_values, plot_shap

# -----------------------------
# 1. Load and prepare data
# -----------------------------

insitu_df = load_and_prepare_data('../data/processed/03_daily/piezo-data_SDH1PS01_daily.csv')
remote_df = load_and_prepare_data('../data/raw/satellites/SDH1G30P01_sentinel2_bands_indices_TCT_202405-202601.csv')

merged_df = merge_datasets(insitu_df, remote_df)

target = 'Piezometer_na_groundwater-depth_m'
features =['Piezometer_na_groundwater-depth_m', 's2_ndwi', 's2_ndvi']

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

