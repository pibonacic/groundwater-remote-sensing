import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def calculate_shap_values(model, X: pd.DataFrame, sample_size: int=None) -> shap.Explanation:
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


def plot_shap(shap_values: shap.Explanation, X: pd.DataFrame, feature_names: list[str], sample_size: int=None, plot_size: float=0.2):
    """
    Generate SHAP summary and bar plots.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values object.
    X : pd.DataFrame
        Model features dataframe.
    feature_names : list[str]
        Feature names.
    sample_size : int, optional
        Number of samples to include in plot. Defaults to all.
    plot_size : float, default=0.2
        Scaling factor for summary plot.
    """
    if sample_size is None:
        sample_size = X.shape[0]

    shap.summary_plot(shap_values, features=X[:sample_size], feature_names=feature_names, plot_size=plot_size)
    shap.plots.bar(shap_values)



