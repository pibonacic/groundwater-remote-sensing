import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.polynomial import Polynomial


def obs_data(df):
    """
    Display basic statistics and plots for the DataFrame.
    
    Statistics include count, mean, standard deviation, min, and max for each column.
    Plots include histograms and time series for visual inspection of data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe for observation.
    """
    # Pivot to long format for faceted plotting with Seaborn
    df_long = df.reset_index().melt(id_vars=df.index.name)

    print('\n--- Dataframe info ---')
    print(f'Unique days: {len(df)}')
    print(f'Temporal range: {df.index.min()} to {df.index.max()}')

    print('\n--- Descriptive stats ---')
    print(df.describe().T[['count', 'mean', 'std', 'min', 'max']])

    print('\n--- Histograms ---')
    g_hist = sns.displot(data=df_long, x='value', col='variable', col_wrap=4, kde=True,
                         bins=15, color='#4C72B0', edgecolor='white', linewidth = 1.5,
                         height=3.5, aspect=1.2, common_bins=False, facet_kws={'sharex': False, 'sharey': False})
    plt.show()

    print('\n--- Time series ---')
    g_line = sns.relplot(data=df_long, x=df.index.name, y='value', col='variable', marker='o', markersize=3,
                         col_wrap=4, kind='line', facet_kws={'sharex': True, 'sharey': False})
    g_line.figure.autofmt_xdate()
    plt.show()


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
    plt.scatter(y_test, y_pred, s=2, c='#4C72B0', alpha=1)
    plt.plot(y_test, y_test, label='1:1', c='black', linewidth=0.7)

    # Linear regression
    line = Polynomial.fit(y_test, y_pred, deg=1)
    x_values = np.linspace(min(y_test), max(y_test), 100)
    y_values = line(x_values)
    plt.plot(x_values, y_values, label='Linear adjustment', linestyle='dashed', c='#4C72B0', alpha=1, linewidth=0.8)

    plt.xlabel('Observed groundwater depth (m)')
    plt.ylabel('Predicted groundwater depth (m)')
    plt.grid(linestyle="--", alpha=0.7, linewidth=0.4)
    plt.legend(fontsize=10)
    plt.show()


def plot_timeseries_results(df: pd.DataFrame):
    """
    Plot a timeseries comparison between observed values and model predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing observed and predicted target values.
    """
    y_obs = df['Observed']
    y_pred = df['Predicted']

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, y_obs, label='Observed values', color='#4C72B0', linewidth=1)
    plt.plot(df.index, y_pred, label='Model predictions', color='#C44E52', linewidth=1.5, linestyle='--')
    
    plt.ylabel('Groundwater depth (m)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()