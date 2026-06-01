"""
Author: Pedro Bonacic Vera
Description:
"""

from config import insitu_features, remote_features, no_outlier_check

from preparation import load_and_prepare_data, handle_duplicate_values, handle_outliers, reindex_daily, handle_missing_values, calculate_stats_from_random_points, smooth_data, merge_datasets, slice_by_dates, add_time, add_lags, preprocess_for_ML, preprocess_for_ML_chrono
from training import tune_model, train_model, evaluate_model, apply_model
from plotting import obs_data, plot_pred_vs_real, plot_timeseries_results
from shap_analysis import calculate_shap_values, plot_shap


# -----------------------------
# 1. Load and prepare data
# -----------------------------

# Insitu data preparation pipeline
insitu_df = load_and_prepare_data('../data/processed/in-situ/SDH1_daily-insitu-data.csv')
insitu_df = handle_duplicate_values(insitu_df)
insitu_df = reindex_daily(insitu_df)
insitu_df = handle_outliers(insitu_df, 3.0, no_outlier_check)
insitu_df = handle_missing_values(insitu_df, 'time')

# Remote data preparation pipeline
remote_df = load_and_prepare_data('../data/processed/satellites/Sentinel2_bands_indices_TCT_20m_202405-202605.csv')
remote_df = handle_duplicate_values(remote_df)
remote_df = reindex_daily(remote_df)
remote_df = handle_outliers(remote_df, 3.0)
remote_df = handle_missing_values(remote_df, 'time')
remote_df = calculate_stats_from_random_points(remote_df, 10)
remote_df = smooth_data(remote_df, None, 7, 2)

# Features and target selection
target = 'SDH1PS01_gw-depth_m'
features = insitu_features + remote_features

# Merged data pipeline
fused_df = merge_datasets(insitu_df, remote_df)
fused_df = fused_df[features]
fused_df = add_lags(fused_df, remote_features, [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
fused_df = add_time(fused_df)
fused_df = handle_missing_values(fused_df, 'drop')
fused_df = slice_by_dates(fused_df, '2024-05-22', '2026-01-24')

# Data inspection
obs_data(fused_df)

# Data preparation for ML
#X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_dates = preprocess_for_ML(fused_df, target = target, test_size = 0.2, random_state = 42)

# Data preparation for ML (chronological split)
X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_dates = preprocess_for_ML_chrono(fused_df, target = target, train_size = 0.6)

# -----------------------------
# 2. Model training
# -----------------------------

best_params = tune_model(X_train_scaled, y_train)
print(best_params)

model = train_model(X_train_scaled, y_train, 
                    # best_params=best_params
                    )

y_pred_test, y_pred_train, results_test, results_train = evaluate_model(model, X_test_scaled, y_test, X_train_scaled, y_train)

results_df = apply_model(fused_df, target, scaler, model)

results_df.to_csv('../outputs/obs_pred.csv')

plot_pred_vs_real(y_test, y_pred_test)

plot_timeseries_results(results_df)

# -----------------------------
# 3. SHAP analysis
# -----------------------------

headings = fused_df.drop(columns=target).columns
shap_values = calculate_shap_values(model, X_test_scaled)
plot_shap(shap_values, X_test_scaled, headings)