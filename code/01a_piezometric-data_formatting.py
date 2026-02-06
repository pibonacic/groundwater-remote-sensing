# IMPORTACIONES

import pandas as pd
import glob
import os
from collections import defaultdict


# DEFINICION DE VARIABLES AUXILIARES

# Diccionario con parametros de cada piezometro
piezometer_params = {
    '2188498': {
        'name': 'SDH1PS01',
        'sensor_depth': 1.2,
        'elevation': 3827.6
    },
    '2190064': {
        'name': 'SDH1PS02',
        'sensor_depth': 1.05,
        'elevation': 3828.5     # 3828.499
    },
    '2188504': {
        'name': 'SDH1PS03',
        'sensor_depth': 0.895,   # CORROBORAR
        'elevation': 3790     # CORROBORAR
    },
    '2190049': {
        'name': 'SDH1PS04',
        'sensor_depth': 0.79,   # CORROBORAR
        'elevation': 3790     # CORROBORAR
    }
}

# Nombres de columnas de salida
# Estructura: Sensor_profundidad_variable_unidad
gw_depth_name = 'Piezometer_na_groundwater-depth_m'
gw_level_name = 'Piezometer_na_groundwater-level_masl'
gw_temp_name = 'Piezometer_na_temperature_degreeC'

# Columnas a eliminar # TEMPORAL: COLUMNAS DE ALTITUD Y TEMPERATURA
columns_list = ['ms', 'LEVEL', 'Date', 'Time', 'Piezometer_na_groundwater-level_masl', 'Piezometer_na_temperature_degreeC']


# DEFINCION DE FUNCIONES

# Funcion de agrupamiento de archivos por piezometro
def group_piezometer_files(data_path, pattern):
    """
    Genera un diccionario cuyas claves son el id de los piezometros encontrados y los
    valores son listas con las rutas a los archivos correspondientes a cada piezometro.
    """
    # Consruye un patron de ruta usando el patron de busqueda definido en file_pattern
    filepath_pattern = os.path.join(data_path, pattern)

    # Genera una lista con las rutas de los archivos que cumplen con el patron
    all_filepaths = glob.glob(filepath_pattern)

    # Si no encuentra archivos termina la funcion
    if not all_filepaths:
        print(f'No se encontraron archivos en {data_path}.')
        return
    
    # defaultdict(list) genera un diccionario que automaticamente asigna a una clave
    # no existente llamada una lista vacia como valor (previene key error)
    piezometer_files = defaultdict(list)

    # Itera sobre cada ruta para asignarla a un piezometro dentro de un diccionario
    for path in all_filepaths:

        # Extrae el nombre del archivo desde la ruta completa (path)
        filename = os.path.basename(path)

        # Extrae el nombre del piezometro desde el nombre de archivo:
        # conserva los caracteres ubicados antes del primer guion bajo
        piezometer_name = filename.split('_')[0].strip()

        # Asigna a cada piezometro una lista con las rutas a sus archivos correspondientes 
        piezometer_files[piezometer_name].append(path)
    
    return dict(piezometer_files)


# Funcion para compilar csv por piezometro
def compile_csv_files(filepaths):
    """
    Lee multiples CSVs y los concatena en un Dataframe
    """
    # Crea una lista para almacenar los dfs
    list_of_dfs = []

    # Itera sobre cada archivo en orden
    for path in sorted(filepaths):
        
        # Lee un csv y lo agrega a la lista
        df = pd.read_csv(path, header=9, encoding='latin1')
        list_of_dfs.append(df)

    return pd.concat(list_of_dfs, ignore_index=True)


# Funcion para la lectura y manejo de fechas
def configure_timestamps(df_raw):
    """
    Crea una columna Timestamps uniendo Date y Time y la establece como indice (formato datetime).
    """
    # Convierte las columnas 'Date' y 'Time' en strings
    date_str = df_raw['Date'].astype(str)
    time_str = df_raw['Time'].astype(str)

    # Crea la columna 'Timestamps'
    df_raw['Timestamps'] = pd.to_datetime(
        date_str + ' ' + time_str,
        format='%d-%m-%Y %H:%M:%S'
    )

    # Establece 'Timestamps' como indice y ordena cronologicamente
    df_raw = df_raw.set_index('Timestamps').sort_index()

    return df_raw


# Funcion para calcular la profundidad y altitud del nivel freatico
def calculate_metrics(df, piezo_id, params_dict):
    """
    Calcula la profundidad y altitud del nivel freatico
    """
    # Define las variables para el calculo a partir del diccionario de parametros
    params = params_dict[piezo_id]
    sensor_depth = params['sensor_depth']
    elev = params['elevation']

    # Calculos
    df[gw_depth_name] = (sensor_depth - df['LEVEL']) * (-1)    # Profundidad del sensor - columna de agua * -1 (prof. negativa)
    df[gw_level_name] = elev - df[gw_depth_name]               # Altitud del suelo - profundidad del agua

    return df


# Funcion para el formateo de columnas
def format_columns(df):
    """
    Renombra y elimina columnas de un df a partir de un diccionario y lista.
    Convierte las columnas restantes a tipo numerico.
    """
    # Renombra columna temperatura
    df_formatted = df.rename(columns={'TEMPERATURE': gw_temp_name})
    
    # Elimina columnas no deseadas
    df_formatted = df_formatted.drop(columns=columns_list, axis=1, errors='ignore')
    
    # Itera sobre todas las columnas excepto Timestamps para convertirlas a tipo numerico
    for col in df_formatted.columns:
        if col != 'Timestamps':
            # errors='coerce' transforma cualquier valor no numerico en NaN
            df_formatted[col] = pd.to_numeric(df_formatted[col], errors='coerce')
    
    # Formatea el df a tres decimales
    df_formatted = df_formatted.round(3)

    return df_formatted


# MANEJO DE RUTAS Y ARCHIVOS

# Rutas a las carpetas de entrada y salida
raw_data_path = '../data/raw/piezometers/'
output_path = '../data/processed/01_formatted'

# Crea la carpeta de salida en caso de que no exista
os.makedirs(output_path, exist_ok=True)

# Patron en archivos crudos para buscar dentro de la carpeta
file_pattern = '*compensated.csv'

# Aplica la funcion group_datalogger_files para almacenar todas las
# rutas a archivos en un diccionario
all_piezometers = group_piezometer_files(raw_data_path, file_pattern)

# Imprime los archivos encontrados por datalogger
if all_piezometers:
    print(f'{len(all_piezometers)} piezometros identificados:')
    for piezometer, files in all_piezometers.items():
        print(f'ID {piezometer}')
        for filepath in sorted(files):
            print(f'  - {filepath}')


## BUCLE DE EJECUCION

# Itera sobre cada piezometro en el diccionario
for piezo_id, filepaths in all_piezometers.items():

    # Extrae el nombre del piezometro
    well_name = piezometer_params[piezo_id].get('name')
    print(f'\nProcesando piezometro {well_name} (ID {piezo_id})')

    # Verifica si existen parametros para el pozo actual
    if piezo_id not in piezometer_params:
        print(f'No se encontraron parametros para {piezo_id}')
        continue

    # Compila los csv por piezometro
    df_raw = compile_csv_files(filepaths)
    print(f'  {len(df_raw)} registros compilados.')

    # Formatea fechas
    df_timestamped = configure_timestamps(df_raw)
    print('  Indice timestamp configurado.')

    # Calcula profundidad y altitud del nivel freatico
    df_metrics = calculate_metrics(df_timestamped, piezo_id, piezometer_params)
    print('  Profundidad y altitud del nivel freatico calculados.')

    # Formateo de columnas
    df_formatted = format_columns(df_metrics)
    print('  Nombres de columnas formateados.')

    # Construye el nombre y ruta de salida del archivo
    output_filename = f'piezo-data_{well_name}_formatted.csv'
    output_filepath = os.path.join(output_path, output_filename)

    # Exporta el archivo como csv
    df_formatted.to_csv(output_filepath)
    print(f'  Datos procesados guardados en: {output_filepath}\n')