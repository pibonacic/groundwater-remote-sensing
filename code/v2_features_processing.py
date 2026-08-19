import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_spectral_indices(df, green='green', red='red', nir='nir', swir1='swir1', swir2='swir2'):
    """
    
    """
    out_df = df.copy()

    # Auxiliar function to calculate a normalized difference
    def normalized_difference(b1, b2):
        # np.where prevents divisions by zero
        return np.where((b1 + b2) == 0, 
                        np.nan, 
                        (b1 - b2) / (b1 + b2))

    # Normalized difference indices calculation
    out_df['ndvi'] = normalized_difference(out_df[nir], out_df[red])
    out_df['gndvi'] = normalized_difference(out_df[nir], out_df[green])
    out_df['ndwi'] = normalized_difference(out_df[green], out_df[nir])
    out_df['mndwi'] = normalized_difference(out_df[green], out_df[swir1])
    out_df['ndmi'] = normalized_difference(out_df[nir], out_df[swir1])
    out_df['ndmi2'] = normalized_difference(out_df[nir], out_df[swir2])

    # STR calculation
    out_df['str1'] = np.where(out_df[swir1] == 0, np.nan, ((1 - out_df[swir1])**2) / (2 * out_df[swir1]))
    out_df['str2'] = np.where(out_df[swir2] == 0, np.nan, ((1 - out_df[swir2])**2) / (2 * out_df[swir2]))

    return out_df

def compute_tasseled_cap(df, green='green', red='red', nir='nir', swir1='swir1', swir2='swir2'):
    """
    
    """
    out_df = df.copy()

    # Series extraction
    g = out_df[green]
    r = out_df[red]
    n = out_df[nir]
    s1 = out_df[swir1]
    s2 = out_df[swir2]

    # Tasseled cap transformations based on coefficients by Zhai et al. (2022)
    out_df['brightness'] = (g * 0.4596) + (r * 0.5046) + (n * 0.5458) + (s1 * 0.4114) + (s2 * 0.2589)
    out_df['greenness']  = (g * -0.3374) + (r * -0.4901) + (n * 0.7909) + (s1 * 0.0177) + (s2 * -0.1416)
    out_df['wetness']    = (g * 0.2254) + (r * 0.3681) + (n * 0.2250) + (s1 * -0.6053) + (s2 * -0.6298)

    return out_df

# MAESTRA
def process_spectral_data(df, **kwargs):
    """
    """
    df_processed = (
        df
        .pipe(compute_spectral_indices, **kwargs)
        .pipe(compute_tasseled_cap, **kwargs)
        )
    return df_processed


def scale_and_apply_pca(df, feature_columns, n_components=2, random_state=42):
    data = df.copy()
    subset = data[feature_columns]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(subset)

    pca_model = PCA(n_components=n_components, random_state=random_state)
    pc = pca_model.fit_transform(scaled_features)

    pc_names = [f'PC_{i+1}' for i in range(n_components)]

    pca_df = pd.DataFrame(
        data=pc, 
        columns=pc_names, 
        index=data.index
    )
    return pca_df, pca_model


def evaluate_pca_performance(pca_model):
    explained_variance = pca_model.explained_variance_ratio_ * 100
    total_variance = np.sum(explained_variance)

    print("--- PCA Evaluation ---")
    for i, variance in enumerate(explained_variance):
        print(f"Component {i+1} explains: {variance:.2f}% of the variance")
    
    print(f"Total information retained: {total_variance:.2f}%")
    print("----------------------")


def get_pca_loadings(pca_model: PCA, feature_columns: list) -> pd.DataFrame:

    # Extract the components matrix from the fitted model
    # Each row is a principal component, each column is an original feature
    loadings_matrix = pca_model.components_
    
    # Transpose the matrix so features become rows and components become columns
    loadings_transposed = np.transpose(loadings_matrix)
    
    # Generate dynamic column names for the components (e.g., 'PC_1', 'PC_2')
    n_components = loadings_matrix.shape[0]
    pc_names = [f'PC_{i+1}' for i in range(n_components)]
    
    # Create a dataframe for easy visualization and interpretation
    loadings_df = pd.DataFrame(
        data=loadings_transposed,
        columns=pc_names,
        index=feature_columns
    )
    return loadings_df

# MAESTRA
def principal_component_analysis(df, feature_columns, n_components=2, random_state=42):
    """
    """
    # Run PCA
    pca_df, pca_model = scale_and_apply_pca(
        df, 
        feature_columns, 
        n_components=n_components, 
        random_state=random_state
    )
    # Print explained variance
    evaluate_pca_performance(pca_model)

    # Get and print pca loadings
    loadings_df = get_pca_loadings(pca_model, feature_columns)
    print(loadings_df)

    return pca_df


def cross_correlations_analysis(df, target_col, feature_cols, max_lag):

    differenced_df = df.copy().diff().dropna()
    results = []

    for feature in feature_cols:
        for lag in range(-max_lag, max_lag+1):
            shifted_feature = differenced_df[feature].shift(lag)
            correlation = differenced_df[target_col].corr(shifted_feature)

            if lag > 0:
                direction = 'past'
                display_lag = f'past_{lag}'
            elif lag <0:
                direction = 'future'
                display_lag = f'future_{lag}'
            else:
                direction = 'same_day'
                display_lag = 'lag_0'

            results.append({
                'feature': feature,
                'direction': direction,
                'lag_value': lag,
                'lag_name': display_lag,
                'correlation': correlation,
                'abs_correlation': abs(correlation)
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=['feature', 'abs_correlation'], 
        ascending=[True, False]
    ).reset_index(drop=True)

    print(results_df)
    return results_df


# OTHER
def add_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    df_copy = df.copy()

    # Extract the day of year from the Datetime index
    doy = df_copy.index.dayofyear

    # Calculate time representations and add them as new columns
    #df_copy['doy'] = doy
    df_copy['doy_sin'] = np.sin(2*np.pi*doy/365.25)
    #df_copy['doy_cos'] = np.cos(2*np.pi*doy/365.25)
    
    return df_copy


def add_lags(df:pd.DataFrame, features: list[str], past_lags: list[int] = None, future_lags: list[int] = None) -> pd.DataFrame:
    """
    
    """
    # Initialize list with the original df and other to store lagged cols
    dfs_to_concat = [df]
    new_lag_cols = []

    # Handle default none value
    past_lags = past_lags or []
    future_lags = future_lags or []

    all_lags = past_lags + [-lag for lag in future_lags]

    # Iterate over each feature in the list
    for feature in features:
        
        # Iterate over each lag in the list
        lagged_columns = {}
        for lag in all_lags:
            # Handle lag naming
            if lag > 0:
                col_name = f'{feature}_pastLag_{lag}'
            else:
                col_name = f'{feature}_futureLag_{abs(lag)}'

            # Generate a column for the current feature shifted by the current lag
            lagged_columns[col_name] = df[feature].shift(periods=lag)
            new_lag_cols.append(col_name)

        # Store the columns in a new df
        feature_lags_df = pd.DataFrame(lagged_columns, index=df.index)

        # Add the current lagged feature df to the list
        dfs_to_concat.append(feature_lags_df)

    # Concatenate all dataframes horizontally (axis=1)
    output_df = pd.concat(dfs_to_concat, axis=1)

    # Drop rows with NaN values
    cols_to_check_for_nans = features + new_lag_cols
    output_df = output_df.dropna(subset=cols_to_check_for_nans)

    return output_df


def merge_datasets(insitu_df: pd.DataFrame, remote_df: pd.DataFrame, how: str = 'outer') -> pd.DataFrame:
    """
    Join in-situ measurements with remote sensing time series using 
    the intersection of their DatetimeIndices

    Parameters
    ----------
    insitu_df : pd.DataFrame
        Dataframe containing in-situ measurements.
    remote_df : pd.DataFrame
        Dataframe containing satellite-derived data.
    
    Returns
    ----------
    pd.DataFrame
        A merged Dataframe with satellite-derived data aligned to the in-situ measurement dates.
    """
    # Convert to dataframe if any input is a pd Series
    if isinstance(insitu_df, pd.Series):
        insitu_df = insitu_df.to_frame()
    if isinstance(remote_df, pd.Series):
        remote_df = remote_df.to_frame()

    # Join datasets using a specified join type
    return insitu_df.join(remote_df, how=how)