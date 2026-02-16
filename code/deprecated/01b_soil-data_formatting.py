# IMPORTACIONES

import pandas as pd
import glob
import os
from collections import defaultdict


# DEFINICION DE VARIABLES AUXILIARES

# Diccionario para mapeo de puertos
datalogger_ports_map = {
    'z6-25818':{
        'z6-25818': None,   # Se incluye la primera columna para conservar los Timestamps 
        'Port1': 'TEROS12_48cm',
        'Port2': 'TEROS12_30cm',
        'Port3': 'TEROS12_15cm',
        'Port4': 'TEROS21_35cm',
        'Port5': 'TEROS21_25cm'
    },
    'z6-26092':{
        'z6-26092': None,
        'Port1': 'TEROS12_65cm',
        'Port2': 'TEROS12_45cm',
        'Port3': 'TEROS12_18cm',
        'Port4': 'TEROS21_55cm',
        'Port5': 'TEROS21_31cm'
    }}


# DEFINICION DE FUNCIONES

# Funcion de agrupamiento de archivos por datalogger
def group_datalogger_files(data_path, pattern):
    """
    Genera un diccionario cuyas claves son el nombre de los dataloggers encontrados y los
    valores son listas con las rutas a los archivos correspondientes a cada datalogger.
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
    datalogger_files = defaultdict(list)

    # Itera sobre cada ruta para asignarla a un datalogger dentro de un diccionario
    for path in all_filepaths:

        # Extrae el nombre del archivo desde la ruta completa (path)
        filename = os.path.basename(path)

        # Extrae el nombre del datalogger desde el nombre de archivo:
        # conserva los caracteres ubicados antes del primer parentesis
        datalogger_name = filename.split('(')[0].strip()

        # Asigna a cada datalogger una lista con las rutas a sus archivos correspondientes 
        datalogger_files[datalogger_name].append(path)
        
    return dict(datalogger_files)


# Funcion para filtrar y concatenar archivos de un mismo datalogger
def concatenate_datalogger_data(filepaths, port_map):
    """
    Lee y concatena los datos provenientes de multiples archivos csv, filtrando las columnas
    innecesarias mediante un diccionario de mapeo. Devuelve un df con los datos concatenados 
    (sin encabezado) y dos listas con la informacion de los encabezados.
    """    
    # Ordena los archivos previo a la lectura
    sorted_paths = sorted(filepaths)

    # Define el primer archivo para extraer informacion de los encabezados
    first_file_path = sorted_paths[0]

    # Extrae la primera fila (Ports) como una serie de Pandas
    port_row_full = pd.read_csv(first_file_path, header=None, nrows=1).iloc[0]

    # Extrae la tercera fila (unidad de medida) como una serie
    header_row_full = pd.read_csv(first_file_path, header=None, skiprows=2, nrows=1).iloc[0]

    # Genera listas para almacenar la informacion de las columnas necesarias
    columns_to_keep_indices = [] # Indices numericos de las columnas a conservar
    port_row_filtered = []       # Nombres de los puertos de las columnas filtradas
    header_row_filtered = []     # Unidades de medida de las columnas filtradas

    # Itera sobre las columnas del primer archivo usando el nombre del puerto
    for i, port_name in enumerate(port_row_full):

        # La condicion verifica si el puerto actual esta en el mapa; si no esta, no se procesa
        if port_name in port_map:
            columns_to_keep_indices.append(i)               # Guarda su indice de columna
            port_row_filtered.append(port_name)             # Guarda su nombre de puerto
            header_row_filtered.append(header_row_full[i])  # Guarda su unidad de medida

    # Crea una lista para almacenar los dfs del datalogger
    list_of_dfs = []

    # Itera sobre cada archivo en sorted_paths
    for path in sorted_paths:

        # Genera un df unicamente de las filas con datos y columnas necesarias
        df = pd.read_csv(path, header=None, skiprows=3, usecols=columns_to_keep_indices)
        # Almacena los df leidos en una lista
        list_of_dfs.append(df)

    # Si no encuentra dfs en la lista devuelve un df vacio
    if not list_of_dfs:
        return pd.DataFrame(), None, None
    
    # Concatena los datos de los df
    concatenated_data = pd.concat(list_of_dfs, ignore_index=True)

    # Devuelve el df concatenado (solo datos) y las filas con informacion asociada
    return concatenated_data, port_row_filtered, header_row_filtered


# Funcion de formateo de nombres de columnas
def format_column_names(df, port_row, header_row, port_map):
    """
    Extrae texto de un diccionario de mapeo y dos listas con informacion sobre los encabezados 
    de un df, lo formatea y asigna como nuevo nombre de columna al df.
    """
    # Crea una lista vacia para almacenar los nuevos nombres de columnas
    new_columns = []

    # Itera sobre cada columna del df
    for i in range(len(df.columns)):
    
        # Asigna 'Timestamps' al nombre de la primera columna y continua el bucle
        if i == 0:
            new_columns.append('Timestamps')
            continue

        # Extrae el nombre de puerto de la lista port_row usando el indice de columna actual (i)
        port_name = port_row[i]
        # Lo mismo, pero de la lista header_row
        header_text = header_row[i].strip()
        # Extrae los valores asociados a la clave Port del diccionario port map
        sensor = port_map[port_name]

        # Divide el texto del encabezado usando el primer espacio
        parts = header_text.split(' ', 1)
        # Almacena la unidad de medida de la columna (primer elemento) y la formatea
        unit = parts[0].replace('_', '')
        # Almacena la variable medida en la columna (segundo elemento) y la formatea
        variable = parts[1].replace(' ', '-').lower()

        # Construye el nuevo nombre para la columna
        new_name = f'{sensor}_{variable}_{unit}'
        # Agrega el nuevo nombre a la lista de columnas
        new_columns.append(new_name)
    
    # Asigna la lista con nombres de columna nuevos a las columnas del df
    df.columns = new_columns
    return df


# Funcion para transformar los tipos de datos
def transform_data_types(df):
    """
    Convierte los tipos de datos de las columnas y establece el indice de tiempo.
    """
    # Convierte la columna de Timestamps a datetime
    df['Timestamps'] = pd.to_datetime(df['Timestamps'])

    # Itera sobre todas las demas columnas para convertirlas a tipo numerico
    for col in df.columns:
        if col != 'Timestamps':
            # errors='coerce' transforma cualquier valor no numerico en NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Establece la columna 'Timestamps' como el indice del DataFrame
    df = df.set_index('Timestamps')
    return df


# MANEJO DE RUTAS Y ARCHIVOS

# Ruta a la carpeta con los datos crudos
raw_data_path = '../data/raw/soil-sensors'

# Patron en archivos crudos para buscar dentro de la carpeta
file_pattern = 'z6*'

# Ruta a la carpeta de salida
output_path = '../data/processed/01_formatted'

# Crea la carpeta de salida en caso de que no exista
os.makedirs(output_path, exist_ok=True)

# Aplica la funcion group_datalogger_files para almacenar todas las
# rutas a archivos en un diccionario
all_dataloggers = group_datalogger_files(raw_data_path, file_pattern)

# Imprime los archivos encontrados por datalogger
if all_dataloggers:
    print(f'{len(all_dataloggers)} dataloggers identificados:')
    for datalogger, files in all_dataloggers.items():
        print(f'Datalogger {datalogger}')
        for filepath in sorted(files):
            print(f'  - {filepath}')


# BUCLE DE EJECUCIÓN

# Itera sobre cada datalogger en el diccionario
for datalogger_name, file_paths_list in all_dataloggers.items():
    print(f'Procesando datalogger {datalogger_name}')

    # Verifica si existe un mapa de puertos para el datalogger actual
    if datalogger_name not in datalogger_ports_map:
        print(f'No se encontro un mapa de puertos para {datalogger_name}')
        continue

    # Extrae el mapa de puertos para el datalogger actual
    current_ports_map = datalogger_ports_map[datalogger_name]

    # Aplica la funcion concatenate_datalogger_data para unir todos los archivos
    # asociados a un mismo datalogger; genera 3 outputs
    data_df, filtered_port_row, filtered_header_row = concatenate_datalogger_data(file_paths_list, current_ports_map)
    print(f'  Datos concatenados: {len(data_df)} registros.')

    # Aplica la funcion format_column_names para renombrar las columnas del df concatenado
    formatted_data_df = format_column_names(data_df, filtered_port_row, filtered_header_row, current_ports_map)
    print('  Nombres de columnas formateados.')

    # Aplica la funcion transform_data_types
    transformed_df = transform_data_types(formatted_data_df)
    print('  Tipo de datos transformados.')

    # Construye el nombre de salida del archivo
    output_filename = f'soil-data_{datalogger_name}_formatted.csv'
    # Construye la ruta de salida del archivo
    output_filepath = os.path.join(output_path, output_filename)
    
    # Exporta el archivo como csv
    transformed_df.to_csv(output_filepath)
    print(f'  Datos procesados guardados en: {output_filepath}\n')