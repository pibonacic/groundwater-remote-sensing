import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from hydroeval import pbias
import shap

# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================

def split_data(
        df: pd.DataFrame,
        target: str,
        test_start_date: str,
        chronological_order: bool = False
) -> dict:
    """
    ...
    """
    # Ensure inverse chronological order of datetime index
    if df.index.is_monotonic_increasing:
        df = df.sort_index(ascending=chronological_order)

    # Separate features (X) from target (y)
    X = df.drop(columns=[target])
    y = df[target]

    # Retrieve positional index of the split date
    split_idx = df.index.get_loc(pd.to_datetime(test_start_date))

    # Split train and test sets
    # Train: from start to split point, test: from split point to end
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]    
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]    
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # Store test dates
    test_dates = y_test.index

    return {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'test_dates': test_dates
        }


def define_model_pipeline_and_grid(
    model_type: str,
) -> tuple[Pipeline, dict]:
    """
    ...
    """
    normalized_model_type = model_type.lower().strip()

    if normalized_model_type == "svr":
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", SVR(kernel="rbf")),
            ]
        )
        param_grid = {
            "regressor__C": [0.1, 1.0, 10.0, 100.0],
            "regressor__epsilon": [0.01, 0.05, 0.1, 0.2],
            "regressor__gamma": ["scale", "auto", 0.01, 0.1],
        }

    elif normalized_model_type == "lasso":
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Lasso(random_state=42)),
            ]
        )
        param_grid = {
            "regressor__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            "regressor__fit_intercept": [True, False],
        }

    elif normalized_model_type in ["random_forest", "rf"]:
        # Tree-based algorithms do not require feature scaling
        pipeline = Pipeline(
            steps=[
                ("regressor", RandomForestRegressor(random_state=42)),
            ]
        )
        param_grid = {
            "regressor__n_estimators": [50, 100, 200],
            "regressor__max_depth": [None, 5, 10, 15],
            "regressor__min_samples_split": [2, 5, 10],
        }

    else:
        supported_models = ["svr", "lasso", "random_forest"]
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Choose from {supported_models}."
        )

    return pipeline, param_grid


def tune_and_train_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    model_type: str = 'svr',
    n_splits: int = 4,
    max_train_size = None,
    test_size: int = 30,
    gap: int = 1,
    nested_cv: bool = False
) -> dict:
    """
    ...
    """
    # 1. Retrieve specific pipeline and hyperparameter search space
    modeling_pipeline, parameter_grid = define_model_pipeline_and_grid(
        model_type=model_type
    )

    # 2. Configure time-series cross-validation splitters
    cross_validator = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=max_train_size,
        test_size=test_size,
        gap=gap,
    )

    # 3. Configure grid search
    grid_search_estimator = GridSearchCV(
        estimator=modeling_pipeline,
        param_grid=parameter_grid,
        cv=cross_validator,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    # 4. Evaluate generalization error via outer loop (nested CV) if requested
    optimization_results = {}
    if nested_cv:
        nested_scores_neg_rmse = cross_val_score(
            estimator=grid_search_estimator,
            X=X_train,
            y=y_train,
            cv=cross_validator,
            scoring="neg_root_mean_squared_error",
        )
        nested_rmse_scores = -nested_scores_neg_rmse
        optimization_results.update(
            {
                "nested_rmse_per_fold": nested_rmse_scores,
                "nested_rmse_mean": float(np.mean(nested_rmse_scores)),
                "nested_rmse_std": float(np.std(nested_rmse_scores)),
            }
        )

    # 6. Fit final grid search over the entire historical training set
    grid_search_estimator.fit(X_train, y_train)

    # 7. Update dict with inner CV results
    best_cv_rmse = float(-grid_search_estimator.best_score_)
    optimization_results.update(
        {
            "model_type": model_type,
            "best_cv_rmse": best_cv_rmse,
            "best_hyperparameters": grid_search_estimator.best_params_,
            "best_pipeline": grid_search_estimator.best_estimator_,
        }
    )
    return optimization_results


def evaluate_model(
    model: BaseEstimator,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    ...
    """
    # Ensure target formats are numpy arrays for custom metric compatibility
    y_true_test = np.asarray(y_test)
    y_true_train = np.asarray(y_train)

    # Generate predictions applying the model/pipeline to the unseen test features
    y_pred_test = model.predict(X_test)

    metrics_test = {
        "RMSE": round(root_mean_squared_error(y_true_test, y_pred_test), 3),
        "MAE": round(mean_absolute_error(y_true_test, y_pred_test), 3),
        "R2": round(r2_score(y_true_test, y_pred_test), 3),
        "PBIAS": round(float(pbias(y_pred_test, y_true_test)), 3),
    }

    # Generate predictions applying the model/pipeline to the train features
    y_pred_train = model.predict(X_train)

    metrics_train = {
        "RMSE": round(root_mean_squared_error(y_true_train, y_pred_train), 3),
        "MAE": round(mean_absolute_error(y_true_train, y_pred_train), 3),
        "R2": round(r2_score(y_true_train, y_pred_train), 3),
        "PBIAS": round(float(pbias(y_pred_train, y_true_train)), 3),
    }

    # Add out-of-bag score if available on the model or final estimator in pipeline
    oob_score = getattr(model, "oob_score_", None)
    if oob_score is None and hasattr(model, "named_steps"):
        final_step = model.steps[-1][1]
        oob_score = getattr(final_step, "oob_score_", None)

    if oob_score is not None:
        metrics_train["OOB_R2"] = round(oob_score, 3)

    print("\n--- Training performance ---")
    print(f"R²:   {metrics_train['R2']}")
    print(f"RMSE:   {metrics_train['RMSE']}")
    print(f"MAE:   {metrics_train['MAE']}")
    print(f"PBIAS:   {metrics_train['PBIAS']}")
    if "OOB_R2" in metrics_train:
        print(f"OOB_R²:   {metrics_train['OOB_R2']}")

    print("\n--- Testing performance ---")
    print(f"R²:   {metrics_test['R2']}")
    print(f"RMSE:   {metrics_test['RMSE']}")
    print(f"MAE:   {metrics_test['MAE']}")
    print(f"PBIAS:   {metrics_test['PBIAS']}")

    return y_pred_test, y_pred_train, metrics_test, metrics_train


def predict_dataset(
    df: pd.DataFrame,
    model: BaseEstimator,
    scaler: StandardScaler | None = None,
    target: str | None = None,
) -> pd.DataFrame:
    """Generate predictions for a dataset using a trained model or pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing feature columns and optional target column.
    model : BaseEstimator
        Fitted model or pipeline (e.g., best_pipeline from optimize_model).
    scaler : StandardScaler or None, default=None
        Optional fitted scaler if the model does not already contain scaling inside.
    target : str or None, default=None
        Name of the target column if present in df to include observed values.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with a sorted index containing 'predicted' values and
        optionally 'observed' values.
    """
    # Separate features (X) from target (y) if target column is provided
    if target and target in df.columns:
        features = df.drop(columns=[target])
        observed_target = df[target]
    else:
        features = df.copy()
        observed_target = None

    # Apply external scaling only if explicitly provided (not needed when model is a Pipeline)
    if scaler is not None:
        features_to_predict = scaler.transform(features)
        if not isinstance(features_to_predict, pd.DataFrame):
            features_to_predict = pd.DataFrame(
                features_to_predict, index=features.index, columns=features.columns
            )
    else:
        features_to_predict = features

    # Generate predictions directly using the model or pipeline
    predicted_target = model.predict(features_to_predict)

    # Prepare output mapping
    results = {"predicted": predicted_target}
    if observed_target is not None:
        results["observed"] = observed_target

    # Return results as dataframe with sorted index
    return pd.DataFrame(results, index=df.index).sort_index()


def calculate_shap_values(
    model: BaseEstimator,
    X: pd.DataFrame,
    background_sample_size: int = 50,
    explanation_sample_size: int | None = None,
) -> shap.Explanation:
    """Calculate SHAP values for any fitted estimator or pipeline.

    Parameters
    ----------
    model : BaseEstimator
        Fitted model or pipeline (e.g., best_pipeline from optimize_model).
    X : pd.DataFrame
        Model features dataframe in original scale.
    background_sample_size : int, default=50
        Number of background samples used to compute base value expectations.
    explanation_sample_size : int or None, default=None
        Number of rows to explain. Defaults to all rows in X.

    Returns
    -------
    shap_values : shap.Explanation
        Computed SHAP values explanation object.
    """
    # Select background reference data to compute marginal expectations
    background_dataset = X.iloc[: min(background_sample_size, len(X))]

    # Use model.predict as callable to let pipelines handle internal scaling automatically
    model_callable = getattr(model, "predict", model)
    explainer = shap.Explainer(model_callable, background_dataset)

    # Determine subset to explain
    if explanation_sample_size is not None:
        samples_to_explain = X.iloc[:explanation_sample_size]
    else:
        samples_to_explain = X

    # Compute SHAP values
    shap_values = explainer(samples_to_explain)

    return shap_values
