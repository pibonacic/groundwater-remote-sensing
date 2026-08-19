import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from numpy.polynomial import Polynomial

from v2_preparation import slice_by_dates


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


# MODIS VS LANDSAT
def plot_sensor_relationship(X_train, y_train, band_name):
    """
    Plots MODIS vs Landsat training data, including the trend line 
    and the line equation.
    """
    x_flat = X_train.flatten()
    
    # Calculate linear fit parameters (m = slope, b = intercept)
    # IMPRIMIR EN GRAFICO NO EN CONSOLA
    m, b = np.polyfit(x_flat, y_train, 1)
    # Print equation to console
    print(f"Equation : y = {m:.4f}x + {b:.4f}")

    # Calculate global minimum and maximum
    global_min = min(x_flat.min(), y_train.min()) - 0.02
    global_max = max(x_flat.max(), y_train.max()) + 0.02
    
    plt.figure(figsize=(6, 5))
    
    # Scatter plot of training data
    plt.scatter(x_flat, y_train, alpha=0.6, color='#2A9D8F', label='Train set', s=20)
    
    # Plot trend line
    line_x = np.array([x_flat.min(), x_flat.max()])
    line_y = m * line_x + b
    plt.plot(line_x, line_y, color='#E76F51', linewidth=2, label='Trend line')

    plt.xlim(global_min, global_max)
    plt.ylim(global_min, global_max)
    #plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title(f'MODIS vs Landsat relationship - {band_name}')
    plt.xlabel('MODIS reflectance')
    plt.ylabel('Landsat reflectance')
    plt.legend()
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.show()

# LANDSAT ORIGINAL VS MODELADO
def plot_observed_vs_modeled(y_train, y_train_pred, y_test, y_test_pred, band_name):
    """
    Plots Observed vs modeled values, differentiating between train and test sets.
    Calculates and prints metrics for both sets.
    """
    
    plt.figure(figsize=(6, 5))
    
    # Plot training predictions
    plt.scatter(y_train, y_train_pred, alpha=0.6, color='#2A9D8F', label='Train set', s=20)
    
    # Plot testing predictions
    plt.scatter(y_test, y_test_pred, alpha=1, color='#E9C46A', label='Test set', s=30, marker='x')
    
    # Plot 1:1 reference line
    all_y = np.concatenate([y_train, y_test, y_train_pred, y_test_pred])
    min_val, max_val = all_y.min(), all_y.max()
    plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', lw=1, label='1:1 line')
    
    plt.title(f'Observed vs modeled reflectance - {band_name}')
    plt.xlabel('Observed (Landsat)')
    plt.ylabel('Modeled (from MODIS)')
    plt.legend()
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.show()


def format_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    """
    Convert a metrics dictionary into a tabular DataFrame.
    """
    metrics_df = pd.DataFrame.from_dict(
        {(band, subset): metrics
         for band, subsets in metrics_dict.items()
         for subset, metrics in subsets.items()},
         orient='index'
    )
    metrics_df.index.names = ['band', 'subset']
    return metrics_df


# RESIDUALES DEL MODELADO
def plot_residuals_boxplot(df_residuals):
    """
    Generates a global violin plot of the residuals per band, overlaying 
    the individual data points.
    """
    # Transform the DataFrame to long format for Seaborn
    df_melted = df_residuals.melt(var_name='band', value_name='residual')
    df_melted = df_melted.dropna()

    plt.figure(figsize=(10, 6))

    # Plot the violin plot with a uniform color
    sns.violinplot(
        data=df_melted, 
        x='band', 
        y='residual', 
        color='#e0e0e0',
        alpha=0.3,
        inner=None, 
        linewidth=1
    )

    # Define the specific color palette for the bands
    band_palette = ['blue', 'green', 'red', 'purple', 'orange', 'saddlebrown']
    
    # Plot the individual points (Stripplot)
    sns.stripplot(
        data=df_melted, 
        x='band', 
        y='residual',
        hue='band',
        palette=band_palette,
        alpha=0.2,
        jitter=True,
        size=4,
        legend=False
    )
    
    plt.ylabel('Residual (reflectance)')
    plt.xlabel('')
    plt.grid(True, axis='y', linestyle='-', alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_observed_modeled_timeseries(
        predicted_df: pd.DataFrame,
        original_df: pd.DataFrame,
        residuals_df: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
        variables: list[str] | None = None
):
    """
    
    """
    sns.set_theme(style='white')

    # Filter inputs by dates
    pred_df = slice_by_dates(predicted_df, start, end)
    orig_df = slice_by_dates(original_df, start, end)
    resid_df = slice_by_dates(residuals_df, start, end)

    # Handle variable selection
    if variables is None:
        variables = pred_df.columns.to_list()

    n_vars = len(variables)
    if n_vars == 0:
        print('Warning: no variables specified for plotting')
        return

    # Handle color assignation
    default_colors = ['blue', 'green', 'red', 'purple', 'orange', 'saddlebrown']
    if n_vars <= len(default_colors):
        colors = default_colors[:n_vars]
    else:
        extra_colors = sns.color_palette('tab10', n_vars - len(default_colors)).as_hex()
        colors = default_colors + extra_colors

    # Create figure adaptable to number of variables
    fig, axes = plt.subplots(n_vars, 1, figsize=(18, max(3.5, 2.3 * n_vars)), sharex=True)
    # Handle axes when only 1 variable is plotted
    axes = np.atleast_1d(axes)

    # Iterate over each ax, variable and color
    for i, (ax, var, color) in enumerate(zip(axes, variables, colors)):

        # Label only first ax
        label_predicted = 'Modeled' if i == 0 else None
        label_original = 'Original' if i == 0 else None
        label_residuals = 'Residuals' if i == 0 else None

        # Modeled series
        if var in pred_df.columns:
            sns.lineplot(
                x=pred_df.index, y=pred_df[var],
                color=color, label=label_predicted, ax=ax, linewidth=1, zorder=1
            )

        # Original series
        orig_col = var if var in orig_df.columns else f'{var}_mean'
        if orig_col in orig_df.columns:
            sns.scatterplot(
                x=orig_df.index, y = orig_df[var],
                color='black', s=15, label=label_original, ax=ax, edgecolor=None, zorder=3, alpha=1
            )

        # Residuals. Handles continuous and discrete residuals
        if var in resid_df.columns:
            # If nan values exist, plot residuals as bars
            if resid_df[var].isna().sum() > 0:
                ax.bar(
                    resid_df.index, resid_df[var],
                    color='gray', width=5, label=label_residuals, alpha=0.7, zorder=2
                )
            # Otherwise, plot residuals as a filled line
            else:
                ax.plot(
                    resid_df.index, resid_df[var],
                    color='gray', linewidth=1, label=label_residuals, zorder=2, alpha=0.5
                )
                ax.fill_between(resid_df.index, 0, resid_df[var], color='gray', alpha=0.2, zorder=1)

        # Baseline at y=0
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)

        # Labels and legend on first plot
        ax.set_ylabel('Reflectance' if i == 0 else '')
        ax.set_title(f'Variable: {var.lower()}', loc='right', fontsize=10, color='dimgray')
        if i == 0:
            ax.legend(loc='upper left', frameon=True)
        elif ax.get_legend() is not None:
            ax.get_legend().remove()

    plt.tight_layout()
    plt.show()


# OUTDATED
# 6 BANDAS LANDSAT MODELADAS + PTOS ORIGINALES + RESIDUALES
def plot_landsat_bands(daily_df, original_df, residuals_df, start, end):
    """
    Filtra los DataFrames por fechas y genera un gráfico apilado con 
    los valores modelados, originales y residuales para cada banda de Landsat.
    """
    # 1. Configurar el estilo de Seaborn
    sns.set_theme(style="white")
    
    # 2. Filtrar los DataFrames usando tu función personalizada
    df_daily = slice_by_dates(daily_df, start, end)
    df_orig = slice_by_dates(original_df, start, end)
    df_resid = slice_by_dates(residuals_df, start, end)

    # 3. Definir las bandas y sus colores asociados
    bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'saddlebrown']
    
    # 4. Crear la figura y los subgráficos
    fig, axes = plt.subplots(len(bands), 1, figsize=(20, 14), sharex=True, sharey=True)
    
    # 5. Iterar sobre cada eje, banda y color
    for i, (ax, band, color) in enumerate(zip(axes, bands, colors)):
        
        # Parámetros para la leyenda (solo se muestran en el primer gráfico)
        label_modeled = 'landsat modeled' if i == 0 else None
        label_original = 'landsat original' if i == 0 else None
        label_residuals = 'residuals' if i == 0 else None

        # Gráfico de líneas (Modelado) usando Seaborn
        sns.lineplot(
            x=df_daily.index, y=df_daily[band], 
            color=color, label=label_modeled, ax=ax, linewidth=1, zorder=1
        )
        
        # Gráfico de dispersión (Original) usando Seaborn
        sns.scatterplot(
            x=df_orig.index, y=df_orig[f'{band}_mean'], 
            color='black', s=15, label=label_original, ax=ax, edgecolor=None, zorder=3, alpha=1
        )
        
        # Gráfico de barras o linea sombrada (Residuales)
        if df_resid[band].isna().sum() > 0:
            ax.bar(
                df_resid.index, df_resid[band], 
                color='gray', width=10, label=label_residuals, alpha=0.7, zorder=2
            )
        else:
            # Si son continuos (interpolados), una línea o área sombreada funciona mucho mejor que barras
            ax.plot(
                df_resid.index, df_resid[band], 
                color='gray', linewidth=1, label=label_residuals, zorder=2, alpha=0.5
            )
            # Opcional: sombrear el área del residual
            ax.fill_between(df_resid.index, 0, df_resid[band], color='gray', alpha=0.2, zorder=1)
        
        # Línea base en y=0
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        
        # Configuraciones específicas del eje
        ax.set_ylabel('Reflectance' if i == 0 else '')
        ax.set_title(f'Band: {band.upper()}', loc='right', fontsize=10, color='dimgray')
        
        # Configurar la leyenda solo en el primer subgráfico
        if i == 0:
            ax.legend(loc='upper left', frameon=True)
        else:
            # Eliminar la leyenda automática de Seaborn en los demás subgráficos
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            
    # Ajustes finales y renderizado
    plt.tight_layout()
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
    y_obs = df['observed']
    y_pred = df['predicted']

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, y_obs, label='Observed values', color='#4C72B0', linewidth=1)
    plt.plot(df.index, y_pred, label='Model predictions', color='#C44E52', linewidth=1.5, linestyle='--')
    
    plt.ylabel('Groundwater depth (m)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_timeseries_results2(df: pd.DataFrame):
    """
    Plot a timeseries comparison between observed values and model predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing observed and predicted target values.
    """
    #y_obs = df['Observed']
    y_pred = df['predicted']

    plt.figure(figsize=(14, 6))
    #plt.plot(df.index, y_obs, label='Observed values', color='#4C72B0', linewidth=1)
    plt.plot(df.index, y_pred, label='Model predictions', color='#C44E52', linewidth=1.5, linestyle='--')
    
    plt.ylabel('Groundwater depth (m)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


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