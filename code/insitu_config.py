import os

# Setup absolute paths to ensure the script runs regardless of the execution folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(base_dir, 'data', 'raw')
processed_data_path = os.path.join(base_dir, 'data', 'processed')

# Directories dictionary for input and output folders
dirs = {
    'piezometers': os.path.join(raw_data_path, 'piezometers'),
    'dataloggers': os.path.join(raw_data_path, 'dataloggers'),
    'processed': os.path.join(processed_data_path, 'in-situ')
}

# Study campaign windows where sensor handling or site activity might cause data noise
campaigns = [
    ('2024-05-21', '2024-05-23'),
    ('2024-07-25', '2024-07-28'),
    ('2024-09-03', '2024-09-07'),
    ('2024-11-05', '2024-11-12'),
    ('2025-01-21', '2025-01-23'),
    ('2025-04-28', '2025-05-02'),
    ('2025-07-08', '2025-07-16'),
    ('2025-11-19', '2025-11-21'),
    ('2026-01-24', '2026-01-27')
]

# Sites metadata dictionary, keyed by site ID
sites_config = {
    'SDH1': {
        'piezometers': ['2188498', '2190064', '2188505', '2189914'],
        'dataloggers': ['z6-24392', 'z6-25818']
    },
    'SDH2': {
        'piezometers': ['2190061', '2190063', '2190049', '2188504'],
        'dataloggers': ['z6-26092', 'z6-28740']
    }
}

# Piezometer metadata dictionary, keyed by piezometer ID
piezometer_config = {
    
    # North site SDH1
    '2188498': {
        'name': 'SDH1PS01',
        'sensor_depth': 1.2,    # Installation depth in meters (below ground surface)
        'elevation': 3827.6     # Reference ground surface elevation (masl)
    },
    '2190064': {
        'name': 'SDH1PS02',
        'sensor_depth': 1.05,
        'elevation': 3828.5
    },
    '2188505': {
        'name': 'SDH1PS03',
        'sensor_depth': 0.89,
        'elevation': 3790     # Not measured with diferential GNSS
    },
    '2189914': {
        'name': 'SDH1PS04',
        'sensor_depth': 0.79,
        'elevation': 3790     # Not measured with diferential GNSS
    },

    # South site SDH2
    '2190061': {
        'name': 'SDH2PP01',
        'sensor_depth': 2.2,
        'elevation': 3827.5
    },
    '2190063': {              # Original ID was 2190064, changed to avoid conflict with SDH1PS02
        'name': 'SDH2PS01',
        'sensor_depth': 1.06,
        'elevation': 3827.34
    },
    '2190049': {
        'name': 'SDH2PS02',
        'sensor_depth': 1.46,
        'elevation': 3829.32
    },
    '2188504': {
        'name': 'SDH2PS03',
        'sensor_depth': 1.29,
        'elevation': 3829.49
    }
}

# Datalogger metadata dictionary for automated column naming, keyed by datalogger ID
datalogger_config = {
    
    # North site SDH1
    'z6-24392': {
        # Nested by config version to handle port changes over time
        'Configuration 9': {
            'Port2': 'ATMOS41'
        },
        'Configuration 10': {
            'Port1': 'S2411',
            'Port2': 'S2412',
            'Port5': 'SI411',
            'Port6': 'TEROS12-05cm'
        },
        'Configuration 11': {
            'Port2': 'S2412',
            'Port5': 'SI411',
            'Port6': 'TEROS12-05cm'
        },
        'Configuration 13': {
            'Port1': 'S2411',
            'Port2': 'S2412',
            'Port5': 'SI411'
        }
    },
    'z6-25818': {
        'Configuration 3': {
            'Port1': 'TEROS12-48cm',
            'Port2': 'TEROS12-30cm',
            'Port3': 'TEROS12-15cm',
            'Port4': 'TEROS21-35cm',
            'Port5': 'TEROS21-25cm'
        },
        'Configuration 4': {
            'Port1': 'TEROS12-48cm',
            'Port2': 'TEROS12-30cm',
            'Port3': 'TEROS12-15cm',
            'Port4': 'TEROS21-35cm',
            'Port5': 'TEROS21-25cm',
            'Port6': 'ATMOS41'
        }
    },

    # South site SDH2
    'z6-26092': {
        'Configuration 3': {
            'Port1': 'TEROS12-65cm',
            'Port2': 'TEROS12-45cm',
            'Port3': 'TEROS12-18cm',
            'Port4': 'TEROS21-55cm',
            'Port5': 'TEROS21-31cm'
        },
        'Configuration 4': {
            'Port1': 'TEROS12-65cm',
            'Port2': 'TEROS12-45cm',
            'Port3': 'TEROS12-18cm',
            'Port4': 'TEROS21-55cm',
            'Port5': 'TEROS21-31cm',
            'Port6': 'ATMOS41'
        }
    },
    'z6-28740': {
        'Configuration 2': {
            'Port1': 'SI411',
            'Port2': 'S2411',
            'Port3': 'S2412'
        }
    }
}

datalogger_vars = {
    'ATMOS41': [
        'precipitation_mm',
        'solar-radiation_Wm2',
        'wind-speed_ms',
        'air-temperature_degreeC',
        'atmospheric-pressure_kPa'
    ],
    'TEROS12': [
        'water-content_m3m3',
        'soil-temperature_degreeC',
        'saturation-extract-ec_mScm'
    ],
    'TEROS21': [
        'soil-temperature_degreeC',
        'matric-potential_kPa'
    ],
    'SI411': [
        'target-temperature_degreeC'
    ],
    'S2411': [
        '650-nm-irradiance_Wm-2nm-1',
        '810-nm-irradiance_Wm-2nm-1'
    ],
    'S2412': [
        '650-nm-radiance_Wm-2nm-1sr-1',
        '810-nm-radiance_Wm-2nm-1sr-1',
        'ndvi'
    ]
}


# List of standarized column names
# datalogger_vars = [
#     'ATMOS41_precipitation_mm',
#     'ATMOS41_solar-radiation_Wm2',
#     'ATMOS41_wind-speed_ms',
#     'ATMOS41_air-temperature_degreeC',
#     'SI411_target-temperature_degreeC',
#     'TEROS12-05cm_water-content_m3m3',
#     'TEROS12-05cm_soil-temperature_degreeC',
#     'TEROS12-05cm_saturation-extract-ec_mScm',
#     'TEROS12-15cm_water-content_m3m3',
#     'TEROS12-15cm_soil-temperature_degreeC',
#     'TEROS12-15cm_saturation-extract-ec_mScm',
#     'TEROS12-30cm_water-content_m3m3',
#     'TEROS12-30cm_soil-temperature_degreeC',
#     'TEROS12-30cm_saturation-extract-ec_mScm',
#     'TEROS12-48cm_water-content_m3m3',
#     'TEROS12-48cm_soil-temperature_degreeC',
#     'TEROS12-48cm_saturation-extract-ec_mScm',
#     'TEROS21-25cm_soil-temperature_degreeC',
#     'TEROS21-35cm_soil-temperature_degreeC',
#     'S2411_650-nm-irradiance_Wm-2nm-1',
#     'S2411_810-nm-irradiance_Wm-2nm-1',
#     'S2412_650-nm-radiance_Wm-2nm-1sr-1',
#     'S2412_810-nm-radiance_Wm-2nm-1sr-1',
#     'S2412_ndvi',
#     'TEROS12-18cm_water-content_m3m3',
#     'TEROS12-18cm_soil-temperature_degreeC',
#     'TEROS12-18cm_saturation-extract-ec_mScm',
#     'TEROS12-45cm_water-content_m3m3',
#     'TEROS12-45cm_soil-temperature_degreeC',
#     'TEROS12-45cm_saturation-extract-ec_mScm',
#     'TEROS12-65cm_water-content_m3m3',
#     'TEROS12-65cm_soil-temperature_degreeC',
#     'TEROS12-65cm_saturation-extract-ec_mScm',
#     'TEROS21-31cm_soil-temperature_degreeC',
#     'TEROS21-55cm_soil-temperature_degreeC'
# ]

# List of variables to exclude from z-score filtering
no_outlier_check = ['ATMOS41_precipitation_mm']

# Additional rules for daily aggregation (default: 'mean')
aggregation_rules = {
    'ATMOS41_precipitation_mm': 'sum',
    'S2411_650-nm-irradiance_Wm-2nm-1': 'mean_positive',
    'S2411_810-nm-irradiance_Wm-2nm-1': 'mean_positive',
    'S2412_650-nm-radiance_Wm-2nm-1sr-1': 'mean_positive',
    'S2412_810-nm-radiance_Wm-2nm-1sr-1': 'mean_positive'
}