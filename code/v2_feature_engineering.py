import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# AUXILIARY FUNCTIONS
# ==============================================================================

def compute_spectral_indices(
        df: pd.DataFrame,
        green: str = 'green',
        red: str = 'red',
        nir: str = 'nir',
        swir1: str = 'swir1',
        swir2: str = 'swir2',
        **kwargs
        ) -> pd.DataFrame:
    """
    
    """
    out_df = df.copy()

    # Variables definition
    GREEN = out_df[green]
    RED = out_df[red]
    NIR = out_df[nir]
    SWIR1 = out_df[swir1]
    SWIR2 = out_df[swir2]

    # Auxiliar function to calculate a normalized difference
    def normalized_difference(b1, b2):
        # np.where prevents divisions by zero
        return np.where((b1 + b2) == 0, np.nan, (b1 - b2) / (b1 + b2))

    # Normalized difference indices calculation
    out_df['ndvi'] = normalized_difference(NIR, RED)
    out_df['gndvi'] = normalized_difference(NIR, GREEN)
    out_df['ndwi'] = normalized_difference(GREEN, NIR)
    out_df['mndwi'] = normalized_difference(GREEN, SWIR1)
    out_df['ndmi'] = normalized_difference(NIR, SWIR1)
    out_df['ndmi2'] = normalized_difference(NIR, SWIR2)

    # STR calculation
    out_df['str1'] = np.where(SWIR1 == 0, np.nan, ((1 - SWIR1)**2) / (2 * SWIR1))
    out_df['str2'] = np.where(SWIR2 == 0, np.nan, ((1 - SWIR2)**2) / (2 * SWIR2))

    return out_df


def run_pca(
        df: pd.DataFrame,
        feature_cols: list[str],
        n_components: int = 2,
        random_state: int = 42,
        verbose: bool = True
) -> tuple[pd.DataFrame, PCA, pd.DataFrame, list[str]]:
    """
    
    """
    # Scale features
    subset = df[feature_cols]
    scaled_data = StandardScaler().fit_transform(subset)
    
    # Run PCA
    pca_model = PCA(n_components=n_components, random_state=random_state)
    pc_values = pca_model.fit_transform(scaled_data)

    # Build components names
    pc_names = [f'PC_{i+1}' for i in range(n_components)]

    # Store components in a df 
    pca_df = pd.DataFrame(
        data=pc_values,
        columns=pc_names,
        index=df.index
    )

    # Get PCA loadings (weights of each feature) and store in df
    loadings_matrix = np.transpose(pca_model.components_)
    loadings_df = pd.DataFrame(
        data=loadings_matrix,
        columns=pc_names,
        index=feature_cols
    )

    # Get PCA performance metrics
    explained_variance = pca_model.explained_variance_ratio_ * 100
    total_variance = np.sum(explained_variance)

    if verbose:
        print('\n--- PCA Evaluation ---')

        # Print explained variance of each component
        for i, variance in enumerate(explained_variance):
            print(f'Component {i+1} explains: {variance:.2f}% of the variance')     

        # Print total variance explained by all components
        print(f'Total variance retained: {total_variance:.2f}%')

        # Print PCA loadings
        print('\nContribution per feature:')
        print(loadings_df.round(3))

    return pca_df, pca_model, loadings_df, pc_names


def cross_correlations_analysis(
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        max_lag: int = 10
) -> pd.DataFrame:
    """
    
    """
    results = []

    # Detrend time series by differencing
    differenced_df = df.diff().dropna()

    # Iterate over each feature
    for feature in feature_cols:
        # Iterate over each lag (negative and positive)
        for lag in range(-max_lag, max_lag + 1):

            # Shift current feature n positions
            shifted = differenced_df[feature].shift(lag)
            # Calculate correlation between target and current feature at current lag
            correlation = differenced_df[target_col].corr(shifted)

            # Build lag name
            if lag > 0:
                direction = 'past'
                display_lag = f'past_{lag}'
            elif lag < 0:
                direction = 'future'
                display_lag = f'future_{lag}'
            else:
                direction = 'same_day'
                display_lag = 'lag_0'

            # Append current results to the list
            results.append({
                'feature': feature,
                'direction': direction,
                'lag_value': lag,
                'lag_name': display_lag,
                'correlation': correlation,
                'abs_correlation': abs(correlation) if pd.notnull(correlation) else 0.0
            })

    # Store results in a df and sort it by |correlation|
    results_df = pd.DataFrame(results).sort_values(
        by=['feature', 'abs_correlation'], 
        ascending=[True, False]
    ).reset_index(drop=True)

    return results_df


def extract_relevant_lags(
        cc_df: pd.DataFrame,
        min_abs_corr: float = 0.1,
        top_k_per_feature: int | None = 3,
) -> dict[str, list[int]]:
    """
    
    """
    lag_mapping = {}

    # Filter out low correlations
    condition = cc_df['abs_correlation'] >= min_abs_corr
    filtered = cc_df[condition]

    print(f'\nLags with correlations >={min_abs_corr}:')
    print(filtered)

    # Group high correlations by feature
    grouped = filtered.groupby('feature')

    for feat, group in grouped:

        # Keep top K lags if specified
        if top_k_per_feature:
            group = group.head(top_k_per_feature)

        # Store relevant lag values per feature in a dict
        lag_mapping[feat] = group['lag_value'].tolist()

    print('\nSelected lags per feature:')
    print(lag_mapping)

    return lag_mapping


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    """
    df_copy = df.copy()

    # Extract the day of year from the Datetime index
    doy = df_copy.index.dayofyear

    # Calculate time representations and add them as new columns
    df_copy['doy_sin'] = np.sin(2*np.pi*doy/365.25)
    # df_copy['doy_cos'] = np.cos(2*np.pi*doy/365.25)
    
    return df_copy


def add_rolling_features(
        df: pd.DataFrame,
        feature_cols: list[str],
        windows: list[int] = [7, 14, 21, 28],
        stats: list[str] = ['mean', 'std']
) -> pd.DataFrame:
    """
    
    """
    df_copy = df.copy()

    # Iterate over features, windows and stats
    for feat in feature_cols:
        for w in windows:
            for stat in stats:
                # Build new column name
                col_name = f'{feat}_roll_{stat}_{w}_days'
                # Calculate rolling aggregate and add as column
                df_copy[col_name] = df_copy[feat].rolling(window=w).agg(stat)

    return df_copy


def add_custom_lags(
    df: pd.DataFrame,
    lag_mapping: dict[str, list[int]]
) -> pd.DataFrame:
    """
    
    """
    df_copy = df.copy()
    new_cols = {}

    # Iterate over the best lags dictionary
    for feature, lags in lag_mapping.items():

        # Define lag names based on direction
        for lag in lags:
            if lag == 0:
                continue
            if lag > 0:
                col_name = f'{feature}_pastLag_{lag}'
            else:
                col_name = f'{feature}_futureLag_{abs(lag)}'

            # Generate a column for the current feature shifted by the current lag
            new_cols[col_name] = df_copy[feature].shift(periods=lag)

    # Store the shifted columns in a df
    lags_df = pd.DataFrame(
        data=new_cols,
        index=df_copy.index
    )

    # Concatenate the shifted features with the original df
    return pd.concat([df_copy, lags_df], axis=1)


def join_datasets(
        df1: pd.DataFrame, 
        df2: pd.DataFrame, 
        how: str = 'inner'
) -> pd.DataFrame:
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
    if isinstance(df1, pd.Series):
        df1 = df1.to_frame()
    if isinstance(df2, pd.Series):
        df2 = df2.to_frame()

    # Join datasets using a specified join type
    return df1.join(df2, how=how)


# ==============================================================================
# PIPELINE
# ==============================================================================

def build_features_dataset(
        remote_df: pd.DataFrame,
        insitu_df: pd.DataFrame,
        target_col: str,
        spectral_bands_cols: list[str],
        pca_components: int = 2,
        verbose_pca: bool = True,
        rolling_windows: list[int] = [7, 14, 21, 28],
        rolling_stats: list[str] = ['mean', 'std'],
        max_cc_lag: int = 10,
        min_corr_thresh: float = 0.1,
        top_k_lags: int | None = None,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    
    """
    # 1. Spectral indices calculation
    bands_dict = {band: band for band in spectral_bands_cols}
    indices_df = compute_spectral_indices(
        df=remote_df,
        **bands_dict
    )

    # 2. Principal component analysis
    features_to_reduce = [c for c in indices_df.columns if c not in spectral_bands_cols]
    pca_df, pca_model, pca_loadings, pc_names = run_pca(
        df=indices_df,
        feature_cols=features_to_reduce,
        n_components=pca_components,
        random_state=random_state,
        verbose=verbose_pca
    )

    # 3. Rolling aggregates
    features_df = add_rolling_features(
        df=pca_df,
        feature_cols=pc_names,
        windows=rolling_windows,
        stats=rolling_stats
    )

    # 4. Cross correlation analysis
    merged_for_cc = join_datasets(insitu_df[[target_col]], pca_df[pc_names], how='inner')
    cc_results = cross_correlations_analysis(
        df=merged_for_cc,
        target_col=target_col,
        feature_cols=pc_names,
        max_lag=max_cc_lag
    )

    # 5. Best lags selection
    lag_mapping = extract_relevant_lags(
        cc_df=cc_results,
        min_abs_corr=min_corr_thresh,
        top_k_per_feature=top_k_lags
    )
    features_with_lags = add_custom_lags(
        df=features_df,          # pca_df = no rolling aggregates | features_df = rolling aggregates
        lag_mapping=lag_mapping,
    )

    # 6. Time representations
    final_features = add_time_features(df=features_with_lags)

    # # 7. Features-target join
    # dataset = join_datasets(
    #     df1=insitu_df[[target_col]],
    #     df2=final_features,
    #     how='inner'
    #     )

    return final_features, cc_results, pca_model




def add_lags(
        df:pd.DataFrame,
        features: list[str],
        past_lags: list[int] = None,
        future_lags: list[int] = None
) -> pd.DataFrame:
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




def build_features_dataset_v2(
        remote_df: pd.DataFrame,
        # insitu_df: pd.DataFrame,
        # target_col: str,
        spectral_bands_cols: list[str],
        pca_components: int = 2,
        verbose_pca: bool = True,
        past_lags = [1, 2, 3],
        future_lags = [1, 2, 3],
        # rolling_windows: list[int] = [7, 14, 21, 28],
        # rolling_stats: list[str] = ['mean', 'std'],
        # max_cc_lag: int = 10,
        # min_corr_thresh: float = 0.1,
        # top_k_lags: int | None = None,
        random_state: int = 42
) -> pd.DataFrame:
    """
    
    """
    # 1. Spectral indices calculation
    bands_dict = {band: band for band in spectral_bands_cols}
    indices_df = compute_spectral_indices(
        df=remote_df,
        **bands_dict
    )

    # 2. Principal component analysis
    features_to_reduce = [c for c in indices_df.columns if c not in spectral_bands_cols]
    pca_df, pca_model, pca_loadings, pc_names = run_pca(
        df=indices_df,
        feature_cols=features_to_reduce,
        n_components=pca_components,
        random_state=random_state,
        verbose=verbose_pca
    )

    # 3. Lags
    lags_df = add_lags(
        df=pca_df,
        features=pc_names,
        past_lags=past_lags,
        future_lags=future_lags
    )

    # # 3. Rolling aggregates
    # features_df = add_rolling_features(
    #     df=pca_df,
    #     feature_cols=pc_names,
    #     windows=rolling_windows,
    #     stats=rolling_stats
    # )

    # # 4. Cross correlation analysis
    # merged_for_cc = join_datasets(insitu_df[[target_col]], pca_df[pc_names], how='inner')
    # cc_results = cross_correlations_analysis(
    #     df=merged_for_cc,
    #     target_col=target_col,
    #     feature_cols=pc_names,
    #     max_lag=max_cc_lag
    # )

    # # 5. Best lags selection
    # lag_mapping = extract_relevant_lags(
    #     cc_df=cc_results,
    #     min_abs_corr=min_corr_thresh,
    #     top_k_per_feature=top_k_lags
    # )
    # features_with_lags = add_custom_lags(
    #     df=features_df,          # pca_df = no rolling aggregates | features df = rolling aggregates
    #     lag_mapping=lag_mapping,
    # )

    # 4. Time representations
    final_features = add_time_features(df=lags_df)

    # # 7. Features-target join
    # dataset = join_datasets(
    #     df1=insitu_df[[target_col]],
    #     df2=final_features,
    #     how='inner'
    #     )

    return final_features










# def scale_and_apply_pca(df, feature_columns, n_components=2, random_state=42):
#     data = df.copy()
#     subset = data[feature_columns]

#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(subset)

#     pca_model = PCA(n_components=n_components, random_state=random_state)
#     pc = pca_model.fit_transform(scaled_features)

#     pc_names = [f'PC_{i+1}' for i in range(n_components)]

#     pca_df = pd.DataFrame(
#         data=pc, 
#         columns=pc_names, 
#         index=data.index
#     )
#     return pca_df, pca_model


# def evaluate_pca_performance(pca_model):
#     explained_variance = pca_model.explained_variance_ratio_ * 100
#     total_variance = np.sum(explained_variance)

#     print("--- PCA Evaluation ---")
#     for i, variance in enumerate(explained_variance):
#         print(f"Component {i+1} explains: {variance:.2f}% of the variance")
    
#     print(f"Total information retained: {total_variance:.2f}%")
#     print("----------------------")


# def get_pca_loadings(pca_model: PCA, feature_columns: list) -> pd.DataFrame:

#     # Extract the components matrix from the fitted model
#     # Each row is a principal component, each column is an original feature
#     loadings_matrix = pca_model.components_
    
#     # Transpose the matrix so features become rows and components become columns
#     loadings_transposed = np.transpose(loadings_matrix)
    
#     # Generate dynamic column names for the components (e.g., 'PC_1', 'PC_2')
#     n_components = loadings_matrix.shape[0]
#     pc_names = [f'PC_{i+1}' for i in range(n_components)]
    
#     # Create a dataframe for easy visualization and interpretation
#     loadings_df = pd.DataFrame(
#         data=loadings_transposed,
#         columns=pc_names,
#         index=feature_columns
#     )
#     return loadings_df

# # MAESTRA
# def principal_component_analysis(df, feature_columns, n_components=2, random_state=42):
#     """
#     """
#     # Run PCA
#     pca_df, pca_model = scale_and_apply_pca(
#         df, 
#         feature_columns, 
#         n_components=n_components, 
#         random_state=random_state
#     )
#     # Print explained variance
#     evaluate_pca_performance(pca_model)

#     # Get and print pca loadings
#     loadings_df = get_pca_loadings(pca_model, feature_columns)
#     print(loadings_df)

#     return pca_df

# def compute_tasseled_cap(df, green='green', red='red', nir='nir', swir1='swir1', swir2='swir2'):
#     """
    
#     """
#     out_df = df.copy()

#     # Series extraction
#     g = out_df[green]
#     r = out_df[red]
#     n = out_df[nir]
#     s1 = out_df[swir1]
#     s2 = out_df[swir2]

#     # Tasseled cap transformations based on coefficients by Zhai et al. (2022)
#     out_df['brightness'] = (g * 0.4596) + (r * 0.5046) + (n * 0.5458) + (s1 * 0.4114) + (s2 * 0.2589)
#     out_df['greenness']  = (g * -0.3374) + (r * -0.4901) + (n * 0.7909) + (s1 * 0.0177) + (s2 * -0.1416)
#     out_df['wetness']    = (g * 0.2254) + (r * 0.3681) + (n * 0.2250) + (s1 * -0.6053) + (s2 * -0.6298)

#     return out_df

# # MAESTRA
# def process_spectral_data(df, **kwargs):
#     """
#     """
#     df_processed = (
#         df
#         .pipe(compute_spectral_indices, **kwargs)
#         .pipe(compute_tasseled_cap, **kwargs)
#         )
#     return df_processed