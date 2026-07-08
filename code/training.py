import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from hydroeval import pbias


def tune_model(X_train_scaled: pd.DataFrame, y_train: np.ndarray,
               param_grid: dict=None, cv_splits: int=3, scoring: str='r2', 
               random_state: int=42) -> dict:
    """
    Tune hyperparameters of a RandomForestRegressor using GridSearchCV with TimeSeriesSplit.

    Parameters
    ----------
    X_train_scaled : pd.DataFrame
        Scaled training features.
    y_train : np.ndarray
        Training target values.
    param_grid : dict, optional
        Grid of hyperparameters to search. Defaults to:
        {"n_estimators": [100, 200, 300], "max_depth": [5, 10, None]}.
    cv_splits : int, default=3
        Number of cross-validation folds.
    scoring : str, default='r2'
        Scoring metric for model evaluation.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    best_params : dict
        Dictionary of best hyperparameters.
    """
    # Define a default search space to optimize tree complexity and forest size
    if param_grid is None:
        param_grid ={
            'n_estimators': [100, 200, 300, 400],         # Total trees in the forest
            'max_depth': [5, 10, 15, 20],             # Limit depth to prevent overfitting
            'min_samples_leaf': [1, 2, 3, 5],      # Minimum points required in a leaf node
            'max_features': ['sqrt', 'log2', 0.5]
        }
    
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Search across the grid with cross-validation
    grid = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_grid,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1   # Use all available CPU cores for faster processing
    )
    # Execute the cross-validation to find the best hyperparameter combination
    grid.fit(X_train_scaled, y_train)
    print(grid.best_params_)

    return grid.best_params_


def train_model(X_train_scaled: pd.DataFrame, y_train: np.ndarray,
                best_params: dict=None, random_state: int=42) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor using optional hyperparameters.

    Parameters
    ----------
    X_train_scaled : pd.DataFrame
        Scaled training features.
    y_train: np.ndarray
        Training target values.
    best_params: dict, optional
        Hyperparameters for RandomForestRegressor
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    ----------
    model : RandomForestRegressor
        Trained random forest model.
    """
    # Initialize model with optimized parameters if provided (via **kwargs unpacking)
    if best_params:
        model = RandomForestRegressor(random_state=random_state, **best_params)
    else:
        model = RandomForestRegressor(random_state=random_state)

    # Train the final random forest model on the scaled training set
    model.fit(X_train_scaled, y_train)

    return model


# def evaluate_model(model: RandomForestRegressor,
#                    X_test_scaled: pd.DataFrame, y_test: np.ndarray) -> tuple[np.ndarray, dict]:
#     """
#     Evaluate model predictions and return regression metrics.

#     Parameters
#     ----------
#     model : RandomForestRegressor
#         Trained model.
#     X_test_scaled : pd.DataFrame
#         Scaled testing features.
#     y_test : np.ndarray
#         Observed target values.

#     Returns
#     ----------
#     y_pred : np.ndarray
#         Model predictions.
#     metrics : dict
#         Dictionary containing MSE, RMSE, MAE, R2 and PBIAS
#     """
#     # Generate predictions applying the model to the unseen test features
#     y_pred = model.predict(X_test_scaled)

#     metrics = {
#         'MSE': mean_squared_error(y_test, y_pred),
#         'RMSE': root_mean_squared_error(y_test, y_pred),
#         'MAE' : mean_absolute_error(y_test, y_pred),
#         'R2': r2_score(y_test, y_pred),
#         'PBIAS': pbias(y_pred, y_test)
#     }

#     print('Model performance: ', metrics)
#     return y_pred, metrics


def evaluate_model(model: RandomForestRegressor,
                   X_test_scaled: pd.DataFrame, y_test: np.ndarray,
                   X_train_scaled, y_train) -> tuple[np.ndarray, dict]:
    """
    Evaluate model predictions and return regression metrics for 
    testing and training subsets.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model.
    X_test_scaled : pd.DataFrame
        Scaled testing features.
    y_test : np.ndarray
        Observed target values.

    Returns
    ----------
    y_pred_test : np.ndarray
        Model predictions for the testing subset.
    y_pred_train : np.ndarray
        Model predictions for the training subset.
    metrics_test : dict
        Dictionary containing MSE, RMSE, MAE, R2 and PBIAS for testing subset.
    metrics_train : dict
        Dictionary containing MSE, RMSE, MAE, R2 and PBIAS for training subset.
    """
    # Generate predictions applying the model to the unseen test features
    y_pred_test = model.predict(X_test_scaled)

    metrics_test = {
        'RMSE': round(root_mean_squared_error(y_test, y_pred_test), 4),
        'MAE' : round(mean_absolute_error(y_test, y_pred_test), 4),
        'R2': round(r2_score(y_test, y_pred_test), 4),
        'PBIAS': round(pbias(y_pred_test, y_test), 4),
        'MSE': round(mean_squared_error(y_test, y_pred_test), 4)
    }

    # Generate predictions applying the model to the train features
    y_pred_train = model.predict(X_train_scaled)

    metrics_train = {
        'RMSE': round(root_mean_squared_error(y_train, y_pred_train), 4),
        'MAE' : round(mean_absolute_error(y_train, y_pred_train), 4),
        'R2': round(r2_score(y_train, y_pred_train), 4),
        'PBIAS': round(pbias(y_pred_train, y_train), 4),
        'MSE': round(mean_squared_error(y_train, y_pred_train), 4)
    }

    print('Model performance (testing):\n', metrics_test)
    print('Model performance (training):\n', metrics_train)

    return y_pred_test, y_pred_train, metrics_test, metrics_train


def apply_model(df: pd.DataFrame, target: str, scaler, model: RandomForestRegressor) -> pd.DataFrame:
    """
    Predict over the entire dataset and create a dataframe of observed values and model predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing both features and target variable.
    target : str
        Name of the target column.
    scaler : StandardScaler
        Fitted scaler object used during model training.
    model : RandomForestRegressor
        Trained Random Forest model.

    Returns
    ----------
    results_df : pd.DataFrame
        DataFrame with observed and modeled values from the complete study period.
    """
    # Isolate features and target variables
    X = df.drop(columns=target)
    y_obs = df[target[0]]
    
    # Standarize features with scaler used during training
    X_scaled = scaler.transform(X)

    # Produce a prediction with the trained model
    y_pred = model.predict(X_scaled)
    
    # Store the observed and predicted values in a new DataFrame
    results_df = pd.DataFrame({
        'Observed': y_obs,
        'Predicted': y_pred
    }, index=df.index).sort_index()

    return results_df