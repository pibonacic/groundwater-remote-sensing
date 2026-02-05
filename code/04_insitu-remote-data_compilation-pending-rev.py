# PENDIENTES
# Ajustar la función para que sea aplicable en múltiples sitios. Actualmente, solo funciona para
# los datos asociados a las rutas definidas al inicio


# IMPORTACIONES
import pandas as pd
import os

# DEFINICION DE RUTAS Y ARCHIVOS

# Definimos las rutas relativas directas basandonos en tu imagen.
# Asumimos que el script corre desde la carpeta 'code'.

# 1. Archivos In-Situ (Carpeta 03_daily)
insitu_files = {
    'soil': '../data/processed/03_daily/soil-data_z6-25818_daily.csv',
    'piezo': '../data/processed/03_daily/piezo-data_SDH1PS01_daily.csv'
}

# 2. Archivos Satelitales (Carpeta raw/satellites)
# Copiamos los nombres exactos que se ven en tu pantallazo
satellite_files = {
    'Landsat': '../data/raw/satellites/SDH1G30P01_landsat89_bands_indices_TCT_202405-202601.csv',
    'Sentinel1': '../data/raw/satellites/SDH1G30P01_sentinel1_bands_indices_202405-202601.csv',
    'Sentinel2': '../data/raw/satellites/SDH1G30P01_sentinel2_bands_indices_TCT_202405-202601.csv'
}

# 3. Ruta de Salida
output_path = '../data/processed/04_compiled'
os.makedirs(output_path, exist_ok=True)


# DEFINICION DE FUNCIONES

def read_data_standard(filepath):
    """
    Lee un archivo CSV estandarizado (sea suelo, piezo o satelite).
    - Carga el archivo.
    - Convierte 'Timestamps' a datetime.
    - Lo establece como indice.
    - Maneja duplicados promediando (necesario para satelites si hay traslape de orbitas).
    """
    if not os.path.exists(filepath):
        print(f'[ERROR] No se encontró el archivo: {filepath}')
        return pd.DataFrame()

    # Leemos el CSV
    df = pd.read_csv(filepath)
    
    # Aseguramos que Timestamps sea datetime
    df['Timestamps'] = pd.to_datetime(df['Timestamps'])
    
    # Establecemos el indice
    df = df.set_index('Timestamps').sort_index()
    
    # Limpieza de duplicados: 
    # Para datos diarios de suelo/piezo no deberia haber, pero para satelitales
    # a veces GEE exporta dos pasadas el mismo dia. Promediamos por seguridad.
    if df.index.duplicated().any():
        print(f'  Nota: Se detectaron duplicados en {os.path.basename(filepath)}. Se promediaron.')
        df = df.groupby(df.index).mean()
        
    return df


def merge_datasets(df_soil, df_piezo, sat_dfs_dict):
    """
    Realiza la fusion de los datos:
    1. Base temporal completa (Suelo + Piezo) usando Outer Join.
    2. Agregado de satelites usando Left Join.
    """
    print('Iniciando fusión de datos...')

    # 1. Fusionar datos in-situ (Outer Join)
    # Esto crea el calendario base. Si falta dato de suelo o piezo en un dia, pone NaN.
    df_base = pd.merge(df_soil, df_piezo, left_index=True, right_index=True, how='outer')
    
    # Verificacion de seguridad por si ambos archivos estuvieran vacios
    if df_base.empty:
        print('[ADVERTENCIA] La base in-situ está vacía.')
        return df_base

    print(f'  Base in-situ creada: {len(df_base)} registros diarios.')
    print(f'  Rango: {df_base.index.min().date()} a {df_base.index.max().date()}')

    # 2. Fusionar datos satelitales (Left Join)
    df_final = df_base.copy()
    
    for sat_name, df_sat in sat_dfs_dict.items():
        if df_sat.empty:
            continue
            
        # Left join: Pegamos el satelite al calendario base
        # El indice del satelite (fecha) se alineará con el indice del df_final
        df_final = df_final.join(df_sat, how='left')
        print(f'  Datos de {sat_name} agregados ({len(df_sat)} capturas).')

    return df_final


# BUCLE DE EJECUCION

print('--- CARGANDO DATOS IN-SITU ---')
# Leemos directamente usando las rutas definidas al inicio
df_soil = read_data_standard(insitu_files['soil'])
df_piezo = read_data_standard(insitu_files['piezo'])

print('\n--- CARGANDO DATOS SATELITALES ---')
# Diccionario para almacenar los DFs satelitales leidos
loaded_satellites = {}

# Iteramos sobre el diccionario de rutas satelitales
for sat_name, path in satellite_files.items():
    print(f'Leyendo {sat_name}...')
    loaded_satellites[sat_name] = read_data_standard(path)


print('\n--- COMPILANDO DATASET FINAL ---')
# Ejecutamos la fusion solo si tenemos datos base
if not df_soil.empty or not df_piezo.empty:
    
    df_merged = merge_datasets(df_soil, df_piezo, loaded_satellites)

    # Construimos el nombre de salida
    # Usamos un nombre generico o basado en el codigo del pozo/sitio
    site_code = "SDH1" # O el identificador que prefieras
    output_filename = f'dataset_compiled_{site_code}.csv'
    output_filepath = os.path.join(output_path, output_filename)
    
    # Guardamos
    df_merged.to_csv(output_filepath)
    print(f'\n[EXITO] Archivo compilado guardado en: {output_filepath}')
    print(f'Dimensiones: {df_merged.shape}')
    
    # Previsualizacion rapida
    print('\nPrimeras 5 filas del resultado:')
    print(df_merged.head())

else:
    print('\n[ERROR] No se pudieron cargar datos in-situ para generar la base.')