# sensor_fusion.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm

from v2_visualization import plot_sensor_relationship, plot_observed_vs_modeled


# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================

def get_coincident_data(
        landsat_series: pd.Series,
        modis_series: pd.Series
) -> pd.DataFrame:
    """
    Aligns Landsat and MODIS series by their DatetimeIndex and extracts coincident 
    valid observations.
    """
    # Combine landsat and modis series and rename columns correspondingly
    aligned_df = pd.concat(
        [landsat_series, modis_series], 
        axis=1, 
        keys=['landsat', 'modis'])

    # Drop NaN values so only coincident dates remain
    return aligned_df.dropna()


def train_linear_regression(
        landsat_modis_df: pd.DataFrame, 
        test_size: float = 0.3, 
        random_state: int = 42,
        verbose_summary: bool = False
):
    """
    Splits the input data into training and testing sets, trains a linear regression 
    model and prints an OLS summary if specified.
    """
    # Define independent and dependent variables
    X = landsat_modis_df[['modis']].values
    y = landsat_modis_df['landsat'].values
    
    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Train a linear regression model with scikit-learn
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print a detailed summary of the regression with statsmodels if specified
    if verbose_summary:
        X_train_sm = sm.add_constant(X_train)
        ols_model = sm.OLS(y_train, X_train_sm).fit()
        print(ols_model.summary())

    return model, X_train, X_test, y_train, y_test


def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> dict[str, float]:
    """
    Calculates R2, MAE and RMSE for a pair of observed and predicted arrays
    """
    return {
        'r2': round(float(r2_score(y_true, y_pred)), 3),
        'mae': round(float(mean_absolute_error(y_true, y_pred)), 3),
        'rmse': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3)
    }


def predict_using_linear_regression(
       modis_series: pd.Series,
       model: LinearRegression 
) -> pd.Series:
    """
    Interpolates a series to daily frequency and applies the trained linear
    regression model.    
    """
    # Prepare the independent series
    X_predict = modis_series.values.reshape(-1, 1)

    # Apply the trained model to predict the dependent variable
    y_predict = model.predict(X_predict)

    # EVALUAR LA HERENCIA DEL NOMBRE DE LA SERIE Y SU UTILIDAD
    return pd.Series(y_predict, index=modis_series.index, name=modis_series.name)


def calculate_residuals(
        original_series: pd.Series,
        predicted_series: pd.Series,
) -> pd.Series:
    """
    Calculates the residuals between an original and a predicted series.
    """
    residuals = original_series - predicted_series
    return residuals


def anchor_predictions_with_residuals(
    predicted_series: pd.Series,
    residuals: pd.Series,
    daily_index: pd.DatetimeIndex,
    sigma_threshold: float | None = 3.0
):
    """
    Anchors daily predictions to real Landsat observations and smoothly distributes 
    residuals over time, ignoring outliers beyond a defined standard deviation threshold.
    """
    # Reindex residuals to daily frequency
    daily_residuals = residuals.reindex(daily_index)

    # Residual filering operates if sigma threshold is provided
    if sigma_threshold is not None:
        # Define the tolerance range of valid residuals using std dev
        res_mean = residuals.mean()
        res_std = residuals.std()
        lower_limit = res_mean - (sigma_threshold * res_std)
        upper_limit = res_mean + (sigma_threshold * res_std)

        # Fliter residuals out of tolerance range
        outlier_mask = (daily_residuals < lower_limit) | (daily_residuals > upper_limit)
        daily_residuals[outlier_mask] = np.nan

    # Interpolate residuals to avoid sharp changes when anchoring
    # EVALUAR CUANTO SE RELLENA HACIA EL PRINCIPIO Y FINAL CON BFILL Y FFILL
    residuals_interpolated = daily_residuals.interpolate(method='time').bfill().ffill()

    # Anchor the predicted series to the original values by adding the residuals
    anchored_series = predicted_series + residuals_interpolated

    return anchored_series, residuals_interpolated


# ==============================================================================
# PIPELINE
# ==============================================================================

def fuse_sensors(
    modis_df: pd.DataFrame,
    landsat_df: pd.DataFrame,
    band_mapping: dict,
    test_size: float = 0.3,
    sigma_threshold: float | None = 3.0,
    verbose_summary: bool = False,
    verbose_plots: bool = False
) -> dict:
    """
    Main pipeline for fusing satellite data
    """
    # Define a daily index based on the length of the modis data
    daily_index = pd.date_range(
        start=modis_df.index.min(), 
        end=modis_df.index.max(), 
        freq='D')

    # Initialize output dfs and dicts
    predicted_df = pd.DataFrame(index=daily_index)
    anchored_df = pd.DataFrame(index=daily_index)
    residuals_df = pd.DataFrame(index=daily_index)
    interp_residuals_df = pd.DataFrame(index=daily_index)
    models = {}
    metrics = {}

    # Loop through each band defined in the mapping dictionary
    for output_band_name, (landsat_col, modis_col) in band_mapping.items():

        # Safety evaluation
        if landsat_col not in landsat_df.columns or modis_col not in modis_df.columns:
            print(f"Warning: columns for {output_band_name} not found. Skipping.")
            continue

        # Column-wise data extraction
        landsat_series = landsat_df[landsat_col]
        modis_series = modis_df[modis_col]

        # 1. Alignment
        coincident_df = get_coincident_data(
            landsat_series=landsat_series, 
            modis_series=modis_series)
        print(f"\n{'='*50}\nProcessing {output_band_name} ({len(coincident_df)} coincident points)\n{'='*50}")

        # Safety evaluation
        if coincident_df.empty:
            print(f"Warning: No valid coincident dates found for {output_band_name}. Skipping.")
            continue

        # 2. Model training
        model, X_train, X_test, y_train, y_test = train_linear_regression(
            landsat_modis_df=coincident_df,
            test_size=test_size,
            random_state=42,
            verbose_summary=verbose_summary
        )
        models[output_band_name] = model

        # 3. Model evaluation
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        metrics[output_band_name] = {
            'train': calculate_regression_metrics(y_train, y_train_pred),
            'test': calculate_regression_metrics(y_test, y_test_pred)
        }
        m = metrics[output_band_name]
        print(f"Train - R2: {m['train']['r2']} | MAE: {m['train']['mae']} | RMSE: {m['train']['rmse']}")
        print(f"Test  - R2: {m['test']['r2']}  | MAE: {m['test']['mae']}  | RMSE: {m['test']['rmse']}")

        # 4. Plotting
        if verbose_plots:
            plot_sensor_relationship(X_train, y_train, output_band_name)
            plot_observed_vs_modeled(y_train, y_train_pred, y_test, y_test_pred, output_band_name)

        # 5. Model application
        predicted_series = predict_using_linear_regression(
            modis_series=modis_series,
            model=model
        )

        # 6. Residuals calculation
        residuals_series = calculate_residuals(
            original_series=landsat_series,
            predicted_series=predicted_series
        )

        # 7. Anchor predictions
        anchored_series, interp_residuals_series = anchor_predictions_with_residuals(
            predicted_series=predicted_series,
            residuals=residuals_series,
            daily_index=daily_index,
            sigma_threshold=sigma_threshold
        )

        # Accumulate results
        predicted_df[output_band_name] = predicted_series
        anchored_df[output_band_name] = anchored_series
        residuals_df[output_band_name] = residuals_series
        interp_residuals_df[output_band_name] = interp_residuals_series

    return {
        'predicted': predicted_df,
        'anchored': anchored_df,
        'residuals': residuals_df,
        'residuals_interpolated': interp_residuals_df,
        'metrics': metrics,
        'models': models
    }