# IMPORTACIONES

import pandas as pd
import numpy as np
import glob
import os


# DEFINICION DE FUNCIONES

# Funcion de agregación diaria de datos subhorarios
def aggregate_to_daily_mean(file_path):
    """
    Lee un archivo csv, establece un indice datetime y calcula el 
    valor diario promedio para cada columna.
    """
    # Asigna el tipo float64 a todas las columnas que no sean Timestamp
    columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    numeric_columns = [col for col in columns if col != 'Timestamps']
    dtype_map = {col: np.float64 for col in numeric_columns}

    # Lee el csv y establece el indice como datetime
    df = pd.read_csv(file_path, 
                     parse_dates=['Timestamps'], 
                     index_col='Timestamps', 
                     dtype=dtype_map,
                     na_values=['#N/D'])
    
    # Resamplea el df a escala diaria usando el promedio
    daily_df = df.resample('D').mean()

    # Formatea el df a tres decimales
    df_formatted = daily_df.round(3)

    return df_formatted


# MANEJO DE RUTAS Y ARCHIVOS

# Rutas a la carpetas con los datos
formatted_data_path = '../data/processed/01_formatted'
cleaned_data_path = '../data/processed/02_cleaned'

# Patrones en archivos para buscar dentro de las carpetas
piezo_pattern = '*piezo-data*.csv'
soil_pattern = 'soil-data*.csv'

# Ruta a la carpeta de salida
output_path = '../data/processed/03_daily'

# Crea la carpeta de salida en caso de que no exista
os.makedirs(output_path, exist_ok=True)

# Construye los patrones de ruta
piezo_filepath_pattern = os.path.join(cleaned_data_path, piezo_pattern)
soil_filepath_pattern = os.path.join(formatted_data_path, soil_pattern)

# Genera listas con las rutas de los archivos que cumplen con los patrones
piezo_files = glob.glob(piezo_filepath_pattern)
soil_files = glob.glob(soil_filepath_pattern)

# Combina ambas listas de archivos
files_to_process = piezo_files + soil_files

# Imprime los archivos encontrados
if files_to_process:
    print(f'{len(files_to_process)} archivos identificados:')
    for filepath in sorted(files_to_process):
        print(f'  - {filepath}')


# BUCLE DE EJECUCION

# Itera sobre cada archivo encontrado
for filepath in files_to_process:

    # Extrae el nombre del archivo desde la ruta completa
    basename = os.path.basename(filepath)

    # Define el nombre del sensor: elimina los sufijos
    sensor_name = basename.replace('_cleaned.csv', '').replace('_formatted.csv', '')
    print(f'Procesando {sensor_name}')

    # Aplica la funcion de promedio diario de los datos
    df_daily = aggregate_to_daily_mean(filepath)
    print('  Datos agregados diariamente.')

    # Construye el nombre de salida del archivo
    output_filename = f'{sensor_name}_daily.csv'

    # Construye la ruta de salida del archivo
    output_filepath = os.path.join(output_path, output_filename)

    # Exporta el archivo como csv
    df_daily.to_csv(output_filepath)
    print(f'  Datos procesados guardados en: {output_filepath}\n')