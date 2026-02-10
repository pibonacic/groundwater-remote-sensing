import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

def calculate_shap_values(model, X, sample_size=None) -> shap.Explanation:
    """
    
    """
    explainer = shap.Explainer(model)

    if sample_size is None:
        sample_size = X.shape[0]    # If sample_size is not specified, it inputs the whole dataset

    shap_values = explainer(X[:sample_size])
    return shap_values


def plot_shap(shap_values, X, feature_names, sample_size=None, plot_size=0.2):
    """
    
    """
    if sample_size is None:
        sample_size = X.shape[0]

    shap.summary_plot(shap_values, features=X[:sample_size], feature_names=feature_names, plot_size=plot_size)
    shap.plots.bar(shap_values)



