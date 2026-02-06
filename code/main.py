"""
Author: Pedro Bonacic Vera
Description:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preparation import load_and_prepare_data, merge_datasets, remove_missing_values, remove_outliers, obs_data, preprocess_for_ML

# -----------------------------
# 1. Load and prepare data
# -----------------------------

insitu_df = load_and_prepare_data('../data/processed/03_daily/piezo-data_SDH1PS01_daily.csv')
remote_df = load_and_prepare_data('../data/raw/satellites/SDH1G30P01_sentinel2_bands_indices_TCT_202405-202601.csv')

target = 'Piezometer_na_groundwater-depth_m'

merged_df = merge_datasets(insitu_df, remote_df)

clean_df1 = remove_missing_values(merged_df)
clean_df2 = remove_outliers(clean_df1)

obs_data(clean_df2)

X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_for_ML(clean_df2, target = target, test_size = 0.2, random_state = 42)


# -----------------------------
# 2. Model training
# -----------------------------


