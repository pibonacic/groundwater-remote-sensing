import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from hydroeval import pbias

# definir formato de los parametros de entrada a las funciones (str, int, etc)

def tune_model(X_train_scaled: pd.DataFrame, y_train: np.ndarray,
               param_grid: dict=None, cv: int=3, scoring: str='r2', random_state: int=42) -> dict:
    """
    Tune hyperparameters of a RandomForestRegressor using GridSearchCV.

    Parameters
    ----------
    X_train_scaled : pd.DataFrame
        Scaled training features.
    y_train : np.ndarray
        Training target values.
    param_grid : dict, optional
        Grid of hyperparameters to search. Defaults to:
        {"n_estimators": [100, 200, 300], "max_depth": [5, 10, None]}.
    cv : int, default=3
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
    if param_grid is None:
        param_grid ={
            'n_estimators': [100, 200],         # Number of trees
            'max_depth': [3, 5, 7],             # Max depth of each tree
            'min_samples_leaf': [3, 5],         # Minimum samples per leaf
        }
    
    grid = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

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
    if best_params:
        model = RandomForestRegressor(random_state=random_state, **best_params)
    else:
        model = RandomForestRegressor(random_state=random_state)

    model.fit(X_train_scaled, y_train)

    return model


def evaluate_model(model: RandomForestRegressor,
                   X_test_scaled: pd.DataFrame, y_test: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Evaluate model predictions and return regression metrics.

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
    y_pred : np.ndarray
        Model predictions.
    metrics : dict
        Dictionary containing MSE, RMSE, MAE, R2 and PBIAS
    """
    y_pred = model.predict(X_test_scaled)

    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'MAE' : mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'PBIAS': pbias(y_pred, y_test)
    }

    print('Model performance: ', metrics)

    return y_pred, metrics


def plot_pred_vs_real(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Plot observed vs predicted values with linear trend.

    Parameters
    ----------
    y_test : np.ndarray
        Observed target values.
    y_pred : np.ndarray
        Predicted target values.
    """
    y_test = np.array(y_test, dtype=float).flatten()
    y_pred = np.array(y_pred, dtype=float).flatten()

    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred, label='GWL', s=2, c='red', alpha=0.2)
    plt.plot(y_test, y_test, label='1:1', c='black', linewidth=0.7)

    # Linear regression
    line = Polynomial.fit(y_test, y_pred, deg=1)
    x_values = np.linspace(min(y_test), max(y_test), 100)
    y_values = line(x_values)
    plt.plot(x_values, y_values, linestyle='dashed', c='blue', linewidth=0.8)

    plt.xlabel('Observed GW level (m)')
    plt.ylabel('Predicted GW level (m)')
    plt.grid(linestyle="--", alpha=0.7, linewidth=0.4)
    plt.legend(fontsize=10)
    plt.show()