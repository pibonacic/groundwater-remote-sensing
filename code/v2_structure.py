# MAIN
# Importaciones:
    # Process MODIS (prep)
    # Process Landsat (prep)
    # Process insitu (prep)
    # Fuse series (fusion)
    # FEATURES
    # MODELLING
    # Plot residuals boxplot (viz)
    # Plot modeled, orginal and residuals (viz)
    # Plot pred vs real (viz)
    # Plot timeseries results (viz)

# CONFIG
# Variables:
    # Band mapping (evaluar cambiar por algo mas input-agnostico)
    # Modis cols
    # Landsat cols (podrian insertarse en band mapping)

# PREPROCESSING/PREPARATION
# Funciones:
    # Load and prepare data
    # Slice by dates
    # Handle duplicate values
    # Reindex daily
    # Handle outliers
    # Calculate stats from random points (Landsat)
    # Filter cols (Landsat)
    # Handle missing values (Modis)
    # Smooth data (Modis)
# Pipelines:
    # Process MODIS
    # Process Landsat
    # Process insitu
# Pendientes:
    # Comentar codigos
    # Añadir prints de diagnostico

# VISUALIZATION
# Funciones:
    # Obs data
    # Plot sensor relationship (mejorar nombre)
    # Plot observed vs modeled
    # Plot residuals boxplot
    # Plot modeled, orginal and residuals (mejorar nombre, convertir en funcion?, limpiar)
    # Plot pred vs real (mejorar nombre)
    # Plot timeseries results (mejorar nombre)
    # Plot shap
# Pendientes:
    # Centralizar funciones en script
    # Hacer que funciones sean agnosticas, aplicables independientemente de los nombres de columnas
    # Comentar codigos

# FUSION
# Importaciones
    # Plot sensor relationship (mejorar nombre)
    # Plot observed vs modeled
# Funciones:
    # Get coincident data
    # Train sensor model (mencionar regresion en nombre)
    # Calculate metrics
    # Predict from modis
    # Anchor predictions with residuals
# Pipelines:
    # Fuse series
# Pendientes:
    # Comentar codigos (docstrings)

# FEATURES PROCESSING 
# Funciones:
    # Compute indices
    # Compute tasseled cap
    # PCA
    # CC
    # Add time
    # Add lags
    # Merge datasets -> preprocessing?
# Pipelines:
    # PCA?
    # CC?
    # Process features?

# Features and target processing
    # Compute indices
    # Compute tasseled cap
    # Process spectral data -> maestra

    # Scale and apply PCA
    # Evaluate PCA performance
    # Get PCA loadings
    # Principal component analysis -> maestra

    # Cross correlation analysis

    # Add time
    # Add lags
    # Merge data


# GROUNDWATER PREDICTION
# Funciones:
    # Preprocess for ML chrono
    # Tune model
    # Train model
    # Evaluate model
    # Apply model
    # Calculate shap values
# Pipeline:
    # Run model?

# Pendiente homogeneizar formato de todas las fx