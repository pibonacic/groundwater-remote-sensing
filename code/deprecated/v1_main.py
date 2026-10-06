"""
Author: Pedro Bonacic Vera
Description:
"""

from code.v1_config import insitu_features1, insitu_features2, remote_features, no_outlier_check

from preparation import load_and_prepare_data, handle_duplicate_values, handle_outliers, reindex_daily, handle_missing_values, calculate_stats_from_random_points, smooth_data, merge_datasets, slice_by_dates, add_time, add_lags, preprocess_for_ML, preprocess_for_ML_chrono
from code.v1_training import tune_model, train_model, evaluate_model, apply_model
from code.v1_plotting import obs_data, plot_pred_vs_real, plot_timeseries_results
from code.v1_shap_analysis import calculate_shap_values, plot_shap


# -----------------------------
# 0. Select study site
# -----------------------------

study_site = 'SDH2'

if study_site == 'SDH1':
    insitu_features = insitu_features1
    target = 'SDH1PS01_gw-depth_m'
    insitu_filepath = '../data/processed/in-situ/SDH1_daily-insitu-data.csv'
    remote_filepath = '../data/processed/satellites/SDH1_S2_20m_2023-10_2026-06.csv'

elif study_site == 'SDH2':
    insitu_features = insitu_features2
    target = 'SDH2PP01_gw-depth_m'
    insitu_filepath = '../data/processed/in-situ/SDH2_daily-insitu-data.csv'
    remote_filepath = '../data/processed/satellites/SDH2_S2_20m_2023-10_2026-06.csv'

elif study_site == 'SDH2b':
    insitu_features = insitu_features2
    target = 'SDH2PS02_gw-depth_m'
    insitu_filepath = '../data/processed/in-situ/SDH2_daily-insitu-data.csv'
    remote_filepath = '../data/processed/satellites/SDH2b_S2_20m_2023-10_2026-06.csv'

# -----------------------------
# 1. Load and prepare data
# -----------------------------

# Insitu data preparation pipeline
insitu_df = load_and_prepare_data(insitu_filepath)
insitu_df = handle_duplicate_values(insitu_df)
insitu_df = reindex_daily(insitu_df)
insitu_df = handle_outliers(insitu_df, 3.0, no_outlier_check)
insitu_df = handle_missing_values(insitu_df, 'time')

# Remote data preparation pipeline
remote_df = load_and_prepare_data(remote_filepath)
remote_df = handle_duplicate_values(remote_df)
remote_df = reindex_daily(remote_df)
remote_df = handle_outliers(remote_df, 3.0)
remote_df = handle_missing_values(remote_df, 'time')
remote_df = calculate_stats_from_random_points(remote_df, 10)
remote_df = smooth_data(remote_df, None, 7, 2)

# Features and target selection
# target = target
features = insitu_features + remote_features

# Merged data pipeline
fused_df = merge_datasets(insitu_df, remote_df)
fused_df = fused_df[features]
fused_df = add_lags(fused_df, remote_features, list(range(1, 32, 2)))
fused_df = add_time(fused_df)
fused_df = handle_missing_values(fused_df, 'drop')
fused_df = slice_by_dates(fused_df, '2024-05-22', '2026-05-15')

# Data inspection
#obs_data(fused_df)

# Data preparation for ML
#X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_dates = preprocess_for_ML(fused_df, target = target, test_size = 0.2, random_state = 42)

# Data preparation for ML (chronological split)
X_train_scaled, X_test_scaled, y_train, y_test, scaler, test_dates = preprocess_for_ML_chrono(fused_df, target = target, train_size = 0.5)

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