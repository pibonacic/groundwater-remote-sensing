import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(base_dir, 'data', 'raw')
processed_data_path = os.path.join(base_dir, 'data', 'processed')
dirs = {
    'piezometers': os.path.join(raw_data_path, 'piezometers'),
    'dataloggers': os.path.join(raw_data_path, 'dataloggers'),
    'outputs': os.path.join(processed_data_path, 'in-situ')
}

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

piezometer_config = {
    '2188498': {
        'name': 'SDH1PS01',
        'sensor_depth': 1.2,    # m
        'elevation': 3827.6     # masl
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

datalogger_config = {
    'z6-24392': {
        'Configuration 9': {
            'Port2': 'ATMOS41'
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
    }
}

datalogger_vars = [
    'ATMOS41_precipitation_mm',
    'ATMOS41_solar-radiation_Wm2',
    'ATMOS41_wind-speed_ms',
    'ATMOS41_air-temperature_degreeC',
    'TEROS12-15cm_water-content_m3m3',
    'TEROS12-15cm_soil-temperature_degreeC',
    'TEROS12-15cm_saturation-extract-ec_mScm',
    'TEROS12-30cm_water-content_m3m3',
    'TEROS12-30cm_soil-temperature_degreeC',
    'TEROS12-30cm_saturation-extract-ec_mScm',
    'TEROS12-48cm_water-content_m3m3',
    'TEROS12-48cm_soil-temperature_degreeC',
    'TEROS12-48cm_saturation-extract-ec_mScm',
    'TEROS21-25cm_soil-temperature_degreeC',
    'TEROS21-35cm_soil-temperature_degreeC',
]

no_outlier_check = ['ATMOS41_precipitation_mm']

aggregation_rules = {
    'ATMOS41_precipitation_mm': 'sum'
}