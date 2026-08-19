import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, KFold, cross_validate
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from hydroeval import pbias
import shap


# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================

def preprocess_for_ML(
        df: pd.DataFrame,
        target: str,
        split_strategy: str = 'chrono_train_first',
        train_size: float = 0.5,
        scale_features: bool = False,
        random_state: int = 42
) -> dict:
    """
    Preprocess and split a DataFrame into training and testing sets for machine learning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a datetime index.
    target : str
        Name of the target column.
    split_strategy : str, default='chrono_train_first'
        Data partitioning strategy ('chrono_train_first', 'chrono_test_first', or 'random').
    train_size : float, default=0.5
        Proportion of the dataset to include in the train split.
    scale_features : bool, default=False
        Whether to standardize features using StandardScaler.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    ----------
    dict
        Dictionary containing:
        - 'X_train' (pd.DataFrame): Training feature set.
        - 'X_test' (pd.DataFrame): Testing feature set.
        - 'y_train' (pd.Series): Training target values.
        - 'y_test' (pd.Series): Testing target values.
        - 'test_dates' (pd.Index): Index labels/dates of the test set.
        - 'scaler' (StandardScaler or None): Fitted scaler if scale_features=True, else None.  
    """
    # Ensure chronological order of datetime index
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Separate features (X) from target (y)
    X = df.drop(columns=[target])
    y = df[target]

    # Split train and test sets randomly
    if split_strategy == 'random':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )
        print(f'Random split applied')

    # Split train and test sets in chronological order, starting with train set
    elif split_strategy == 'chrono_train_first':
        split_idx = int(len(df) * train_size)   # Identify splitting point
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]    # Train: from start to split point
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]    # Test: from split point to end
        print(f'Chronological split applied at: {y_train.index[-1].date()}')
        
    # Split train and test sets in chronological order, starting with test set
    elif split_strategy == 'chrono_test_first':
        split_idx = int(len(df) * (1 - train_size))
        X_train, X_test = X.iloc[split_idx:], X.iloc[:split_idx]    # Train: from split point to end
        y_train, y_test = y.iloc[split_idx:], y.iloc[:split_idx]    # Test: from start to split point
        print(f'Chronological split applied at: {y_train.index[0].date()}')

    else:
        raise ValueError(f'{split_strategy} is not a supported splitting strategy')

    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # Save test dates
    test_dates = y_test.index

    # Optional scaling
    scaler = None
    if scale_features:
        scaler = StandardScaler().set_output(transform='pandas')
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'test_dates': test_dates,
        'scaler': scaler
    }


def tune_model(
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        cv_strategy: str = 'chrono',
        param_grid: dict | None = None, 
        cv_splits: int = 3, 
        scoring: str = 'r2', 
        random_state: int = 42,
        n_jobs: int = -1
) -> dict:
    """
    Tune hyperparameters of a RandomForestRegressor using GridSearchCV with different possible strategies.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : np.ndarray
        Training target values.
    cv_strategy : {'chrono', 'random'}, default='chrono'
        Cross-validation splitting strategy matching the train/test split.
        ('chrono' for TimeSeriesSplit or 'random' for KFold).
    param_grid : dict, optional
        Grid of hyperparameters to search.
    cv_splits : int, default=3
        Number of cross-validation splits/folds.
    scoring : str, default='r2'
        Scoring metric for model evaluation.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel during grid search (-1 uses all processors).

    Returns
    -------
    best_params : dict
        Hyperparameter combination that achieved the best score during CV.
    """
    # Define a default search space to optimize tree complexity and forest size
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300, 400],       # Total trees in the forest
            'max_depth': [5, 10, 15, 20],               # Limit depth to prevent overfitting
            'min_samples_leaf': [1, 2, 3, 5],           # Minimum points required in a leaf node
            'max_features': ['sqrt', 'log2', 0.5]
        }
    # For chronological splits, uses TimeSeriesSplit
    if cv_strategy == 'chrono':
        cv = TimeSeriesSplit(n_splits=cv_splits)

    # For random splits, uses KFold with shuffle=True
    elif cv_strategy == 'random':
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    else:
        raise ValueError(f'{cv_strategy} is not a supported cross-validation strategy.')

    # Search across the grid with cross-validation
    rf = RandomForestRegressor(random_state=random_state)
    grid = GridSearchCV(
        estimator=rf,
        param_grid= param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs
    )
    # Execute the cross-validation to find the best hyperparameter combination
    grid.fit(X_train, y_train)
    print(f'Best hyperparameters ({cv_strategy} CV): {grid.best_params_}')

    return grid.best_params_


def train_model(
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        best_params: dict | None = None, 
        random_state: int = 42
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor using optional hyperparameters.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series or np.ndarray
        Training target values.
    best_params : dict, optional
        Hyperparameters for RandomForestRegressor
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    ----------
    model : RandomForestRegressor
        Trained random forest model.
    """
    # Define best params if provided
    params = best_params or {}

    # Ensure model out-of-bag calculation
    params.setdefault('oob_score', True)
    params.setdefault('bootstrap', True)

    # Initiate model with optimized parameters if provided (via **kwargs unpacking)
    model = RandomForestRegressor(random_state=random_state, **params)

    # Train the random forest model
    model.fit(X_train, y_train)

    return model


def evaluate_model(
        model: RandomForestRegressor,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        y_test: pd.Series | np.ndarray
        
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Evaluate model predictions and return regression metrics for testing and training sets.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained random forest model.
    X_train : pd.DataFrame
        Training feature set.
    X_test : pd.DataFrame
        Testing feature set.
    y_train : pd.Series or np.ndarray
        Observed training target values.
    y_test : pd.Series or np.ndarray
        Observed testing target values.

    Returns
    ----------
    y_pred_test : np.ndarray
        Model predictions for the testing subset.
    y_pred_train : np.ndarray
        Model predictions for the training subset.
    metrics_test : dict
        Dictionary containing RMSE, MAE, R2, PBIAS, and MSE for testing subset.
    metrics_train : dict
        Dictionary containing RMSE, MAE, R2, PBIAS, and MSE for training subset.
    """
    # Generate predictions applying the model to the unseen test features
    y_pred_test = model.predict(X_test)

    metrics_test = {
        'RMSE': round(root_mean_squared_error(y_test, y_pred_test), 3),
        'MAE' : round(mean_absolute_error(y_test, y_pred_test), 3),
        'R2': round(r2_score(y_test, y_pred_test), 3),
        'PBIAS': round(float(pbias(y_pred_test, y_test.to_numpy())), 3),
        'MSE': round(mean_squared_error(y_test, y_pred_test), 3)
    }

    # Generate predictions applying the model to the train features
    y_pred_train = model.predict(X_train)

    metrics_train = {
        'RMSE': round(root_mean_squared_error(y_train, y_pred_train), 3),
        'MAE' : round(mean_absolute_error(y_train, y_pred_train), 3),
        'R2': round(r2_score(y_train, y_pred_train), 3),
        'PBIAS': round(float(pbias(y_pred_train, y_train.to_numpy())), 3),
        'MSE': round(mean_squared_error(y_train, y_pred_train), 3)
    }

    # Add out-of-bag score
    if hasattr(model, 'oob_score_'):
        metrics_train['OOB_R2'] = round(model.oob_score_, 3)

    print('\n--- Training performance ---')
    print(f'R²:   {metrics_train['R2']}')
    print(f'RMSE:   {metrics_train['RMSE']}')
    print(f'MAE:   {metrics_train['MAE']}')
    print(f'PBIAS:   {metrics_train['PBIAS']}')
    print(f'MSE:   {metrics_train['MSE']}')
    print(f'OOB_R²:   {metrics_train['OOB_R2']}')

    print('\n--- Testing performance ---')
    print(f'R²:   {metrics_test['R2']}')
    print(f'RMSE:   {metrics_test['RMSE']}')
    print(f'MAE:   {metrics_test['MAE']}')
    print(f'PBIAS:   {metrics_test['PBIAS']}')
    print(f'MSE:   {metrics_test['MSE']}')    

    return y_pred_test, y_pred_train, metrics_test, metrics_train


def evaluate_CV(
        model: RandomForestRegressor,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        cv_strategy: str = 'chrono',
        cv_splits: int = 3,
        random_state: int = 42,
        n_jobs: int = -1
) -> dict:
    """
    Evaluate a RandomForestRegressor using cross-validation and calculate summary metrics.

    Parameters
    ----------
    model : RandomForestRegressor
        Random forest regression model to evaluate.
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series or np.ndarray
        Training target values.
    cv_strategy : str, default='chrono'
        Cross-validation strategy ('chrono' for TimeSeriesSplit or 'random' for KFold).
    cv_splits : int, default=3
        Number of cross-validation splits/folds.
    random_state : int, default=42
        Random seed for reproducibility when using random KFold.
    n_jobs : int, default=-1
        Number of CPU cores to use for parallel evaluation (-1 uses all available cores).

    Returns
    -------
    metrics_cv : dict
        Dictionary containing the mean and standard deviation across folds for
        R2, RMSE, MAE, and MSE.
    """
    # For chronological splits, uses TimeSeriesSplit
    if cv_strategy == 'chrono':
        cv = TimeSeriesSplit(n_splits=cv_splits)
    # For random splits, uses KFold with shuffle=True
    elif cv_strategy == 'random':
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    else:
        raise ValueError(f'{cv_strategy} is not a supported cross-validation strategy.')
    
    # Scores to calculate
    scoring = {
        'R2': 'r2',
        'RMSE': 'neg_root_mean_squared_error',
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error'
    }    

    # Execute cross-validation
    cv_results = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train.to_numpy(),
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        return_train_score=False
    )

    # Format and print results
    metrics_cv = {}
    for metric_name in scoring.keys():
        scores = cv_results[f'test_{metric_name}']

        # Invert sign of scores (negative by default) except for R2
        if metric_name != 'R2':
            scores = -scores

        # Calculate mean and std dev of metric from all folds
        metrics_cv[f'{metric_name}_mean'] = round(float(np.mean(scores)), 3)
        metrics_cv[f'{metric_name}_std'] = round(float(np.std(scores)), 3)

    print('\n--- Cross-Validation performance (X_train) ---')
    print(f'R²:   {metrics_cv['R2_mean']} ± {metrics_cv['R2_std']}')
    print(f'RMSE: {metrics_cv['RMSE_mean']} ± {metrics_cv['RMSE_std']}')
    print(f'MAE:  {metrics_cv['MAE_mean']} ± {metrics_cv['MAE_std']}')
    print(f'MSE:  {metrics_cv['MSE_mean']} ± {metrics_cv['MSE_std']}')
    
    return metrics_cv


def predict_dataset(
        df: pd.DataFrame,
        model: RandomForestRegressor,
        scaler: StandardScaler | None = None,
        target: str | None = None
) -> pd.DataFrame:
    """
    Generate predictions for a dataset using a trained RandomForestRegressor.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing feature columns and optional target column.
    model : RandomForestRegressor
        Fitted random forest regression model.
    scaler : StandardScaler or None, default=None
        Fitted scaler to transform features before prediction, if applicable.
    target : str or None, default=None
        Name of the target column if present in df to include observed values.

    Returns
    ----------
    results_df : pd.DataFrame
        DataFrame with a sorted index containing 'predicted' values and 
        optionally 'observed' values.    
    """
    # Separate features (X) from target (y) if exists
    if target and target in df.columns:
        X = df.drop(columns=[target])
        y_obs = df[target]
    else:
        X = df.copy()
        y_obs = None

    # Scale X if trained scaler is provided
    if scaler is not None:
        X_to_pred = scaler.transform(X)
        # Ensure the scaler returns a df
        if not isinstance(X_to_pred, pd.DataFrame):
            X_to_pred = pd.DataFrame(X_to_pred, index=X.index, columns=X.columns)
    else:
        X_to_pred = X

    # Predict y with trained model
    y_pred = model.predict(X_to_pred)

    # Prepare output
    if y_obs is not None:
        results = {'observed': y_obs,
                   'predicted': y_pred}
    else:
        results = {'predicted': y_pred}

    # Return results as df with sorted datetime index
    return pd.DataFrame(results, index=df.index).sort_index()


def get_feature_importances(
        model: RandomForestRegressor, 
        feature_names: list[str]
) -> pd.DataFrame:
    """Calculate and sort feature importances."""

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    return feature_importance


def calculate_shap_values(
        model: RandomForestRegressor,
        X: pd.DataFrame, 
        sample_size: int | None = None
) -> shap.Explanation:
    """
    Calculate SHAP values for a model and a feature dataframe.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained ML model supporting SHAP (e.g., RandomForest, XGBoost).
    X : pd.DataFrame
        Model features dataframe
    sample_size : int, optional
        Number of rows to calculate SHAP values for. Defaults to all.

    Returns
    ----------
    shap_values : shap.Explanation
        SHAP values object.
    """
    explainer = shap.Explainer(model)

    if sample_size is None:
        sample_size = X.shape[0]    # If sample_size is not specified, it inputs the whole dataset

    shap_values = explainer(X[:sample_size])
    return shap_values


# ==============================================================================
# PIPELINE
# ==============================================================================

def train_and_evaluate_RF(
        df: pd.DataFrame,
        target: str,
        split_strategy: str = 'chrono_train_first',
        train_size: float = 0.5,
        scale_features: bool = False,
        tune_hyperparameters: bool = True,
        param_grid: dict | None = None,
        cv_strategy: str = 'chrono',
        cv_splits: int = 3,
        scoring: str = 'r2',
        compute_shap: bool = True,
        shap_sample_size: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1
) -> dict:
    """
    Main pipeline for random forest regressor training and evaluation.
    """
    # 1. Data split
    data_split = preprocess_for_ML(
        df=df,
        target=target,
        split_strategy=split_strategy,
        train_size=train_size,
        scale_features=scale_features,
        random_state=random_state
    )

    # Variables definition
    X_train = data_split['X_train']
    X_test = data_split['X_test']
    y_train = data_split['y_train']
    y_test = data_split['y_test']
    scaler = data_split['scaler']
    test_dates = data_split['test_dates']

    # 2. Hyperparameters tuning
    best_params = None
    if tune_hyperparameters:
        best_params = tune_model(
            X_train=X_train,
            y_train=y_train,
            cv_strategy=cv_strategy,
            param_grid=param_grid,
            cv_splits=cv_splits,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs
        )

    # 3. Model training
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        best_params=best_params,
        random_state=random_state
    )

    # 4. Model evaluation
    y_pred_test, y_pred_train, metrics_test, metrics_train = evaluate_model(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    metrics_cv = evaluate_CV(
        model=model,
        X_train=X_train,
        y_train=y_train,
        cv_strategy=cv_strategy,
        cv_splits=cv_splits,
        random_state=random_state,
        n_jobs=n_jobs
    )

    # 5. Inference
    predictions_df = predict_dataset(
        df=df,
        model=model,
        scaler=scaler,
        target=target
    )

    # 6. Feature importance
    feature_importance_df = get_feature_importances(
        model=model,
        feature_names=X_train.columns.tolist()
    )

    # 7. SHAP values
    shap_values = None
    if compute_shap:
        shap_values = calculate_shap_values(
            model=model,
            X=X_test,
            sample_size=shap_sample_size
        )

    return {
        'model': model,
        'scaler': scaler,
        'best_params': best_params,
        'metrics_test': metrics_test,
        'metrics_train': metrics_train,
        'metrics_cv': metrics_cv,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'y_pred_train': y_pred_train,
        'test_dates': test_dates,
        'predictions_df': predictions_df,
        'feature_importances': feature_importance_df,
        'shap_values': shap_values
    }