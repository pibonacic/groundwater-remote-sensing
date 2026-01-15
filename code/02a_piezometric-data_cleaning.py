# IMPORTACIONES

import pandas as pd
import glob
import os


# DEFINICION DE VARIABLES AUXILIARES

# Diccionario con con el nombre y rango de fechas de cada campana.
# La fecha de termino es excluyente. Se consideran fechas de trabajo efectivo.
field_campaigns = {
    "May 2024": pd.date_range(start='2024-05-21', end='2024-05-23'),
    "Jul 2024": pd.date_range(start='2024-07-25', end='2024-07-28'),
    "Sep 2024": pd.date_range(start='2024-09-03', end='2024-09-07'),
    "Nov 2024": pd.date_range(start='2024-11-05', end='2024-11-12'),
    "Jan 2025": pd.date_range(start='2025-01-21', end='2025-01-23'),
    "Apr 2025": pd.date_range(start='2025-04-28', end='2025-05-02'),
    "Jul 2025": pd.date_range(start='2025-07-08', end='2025-07-16')
}


# DEFINICION DE FUNCIONES

# Funcion de lectura de datos
def read_data(file_path):
    """ Lee los datos de un csv a partir de una ruta de archivo"""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


# Funcion de identificacion de datos anomalos
def identify_campaign_outliers(df, campaign_dates):
    """
    Identifica outliers en un df dentro de un rango de fechas a partir de 
    una condicion. Devuelve los indices de las filas consideradas outliers.
    """
    # Filtra el df a las fechas de terreno
    df_campaign = df.loc[
        campaign_dates.min():campaign_dates.max()-pd.Timedelta(seconds=1)
        ].copy()
    
    # Si no hay datos durante la campana devuelve un indice vacio
    if df_campaign.empty:
        return pd.Index([])
    
    # Calcula la media y la desviacion estandar de cada columna
    mean = df_campaign.mean()
    std = df_campaign.std()

    # Calcula el zscore de las columnas numericas. 
    # Si std = 0, zscore = NaN, por lo que se reemplaza por 0
    z_scores = (df_campaign - mean).div(std).fillna(0)

    # Crea una condicion para identificar datos con z-scores > 3
    outlier_condition = (z_scores.abs() > 3).any(axis=1)

    # Obtiene los indices de los outliers
    outlier_indices = df_campaign[outlier_condition].index
    return outlier_indices


# Funcion de remocion de datos anomalos
def remove_outliers(df, outlier_indices):
    """
    Reemplaza con NaN los valores en filas identificadas como outliers.
    """
    # Crea una copia del df
    df_cleaned = df.copy()

    # Reemplaza con NaN los datos en las filas identificadas como outliers
    df_cleaned.loc[outlier_indices, df_cleaned.columns] = pd.NA
    print(f'    Se removieron {len(outlier_indices)} registros anomalos.')
    return df_cleaned


# MANEJO DE RUTAS Y ARCHIVOS

# Ruta a la carpeta con los datos formateados
formatted_data_path = '../data/processed/01_formatted'

# Patron en archivos para buscar dentro de la carpeta
file_pattern = 'piezo-data*'

# Ruta a la carpeta de salida
output_path = '../data/processed/02_cleaned'

# Crea la carpeta de salida en caso de que no exista
os.makedirs(output_path, exist_ok=True)

# Construye un patron de ruta usando el patron de file_pattern
filepath_pattern = os.path.join(formatted_data_path, file_pattern)

# Genera una lista con las rutas de las archivos que cumplen con el patron
formatted_files = glob.glob(filepath_pattern)

# Imprime los archivos encontrados en la carpeta
if formatted_files:
    print(f'{len(formatted_files)} archivos identificados:')
    for filepath in sorted(formatted_files):
        print(f'  - {filepath}')


# BUCLE DE EJECUCIÓN

# Bucle externo: itera sobre cada archivo encontrado
for filepath in formatted_files:

    # Extrae el nombre del archivo desde la ruta completa
    basename = os.path.basename(filepath)

    # Define el nombre del sensor: elimina el sufijo
    sensor_name = basename.replace('_formatted.csv', '')
    print(f'Procesando {sensor_name}')

    # Aplica la funcion de lectura de archivo
    df = read_data(filepath)
    print('  Datos leidos.')

    # Crea el objeto df_cleaned
    df_cleaned = df.copy()

    # Bucle interno: itera sobre las campanas de terreno
    for campaign_name, campaign_dates in field_campaigns.items():
        print(f'  Campana de {campaign_name}:')

        # Aplica la funcion de identificacion de outliers
        outlier_indices = identify_campaign_outliers(df_cleaned, campaign_dates)

        # Aplica la funcion de remocion de outliers
        if not outlier_indices.empty:
            df_cleaned = remove_outliers(df_cleaned, outlier_indices)
        else:
            print('    No se encontraron outliers.')

    # Construye el nombre de salida del archivo
    output_filename = f'{sensor_name}_cleaned.csv'

    # Construye la ruta de salida del archivo
    output_filepath = os.path.join(output_path, output_filename)

    # Exporta el archivo como csv
    df_cleaned.to_csv(output_filepath)
    print(f'  Datos procesados guardados en: {output_filepath}\n')