# Este script asume que los datos de entrada están en formato csv, que son dos: uno con datos remotos (bandas, indices)
# y otro con datos insitu (pozos, pp, temp, humedad CE de suelo), que tienen una columna de fecha con separación diaria,
# que los datos remotos tienen filas (dias) con NaN los dias de no paso, que están separados de los datos de validación
# (el dataset de validación sólo debería tener)
# 
# 
# Funciones necesarias
# Carga de datos (2 datasets) ¿bucle?
# Merge de datos remotos e insitu
# Manejo de datos perdidos
# Manejo de outliers
# Despliegue de estadisticas basicas e histogramas
# Preparación de datos para ML