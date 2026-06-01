insitu_features = [
    'SDH1PS01_gw-depth_m',
    'ATMOS41_precipitation_mm',
    # 'ATMOS41_solar-radiation_Wm2',
    # 'ATMOS41_wind-speed_ms',
    # 'ATMOS41_air-temperature_degreeC',
    # 'TEROS12-15cm_water-content_m3m3',
    # 'TEROS12-15cm_soil-temperature_degreeC',
    # 'TEROS12-15cm_saturation-extract-ec_mScm',
    # 'TEROS12-30cm_water-content_m3m3',
    # 'TEROS12-30cm_soil-temperature_degreeC',
    # 'TEROS12-30cm_saturation-extract-ec_mScm',
    # 'TEROS12-48cm_water-content_m3m3',
    # 'TEROS12-48cm_soil-temperature_degreeC',
    # 'TEROS12-48cm_saturation-extract-ec_mScm',
    # 'TEROS21-25cm_soil-temperature_degreeC',
    # 'TEROS21-35cm_soil-temperature_degreeC',
]

remote_features = [
    # 'B2_mean',
    # 'B3_mean',
    # 'B4_mean',
    # 'B5_mean',
    # 'B6_mean',
    # 'B7_mean',
    # 'B8_mean',
    # 'B8A_mean',
    # 'B11_mean',
    # 'B12_mean',
    # 'ndvi_mean',
    # 'ndwi_mean',
    # 'mndwi_mean',
    # 'ndmi_mean',
    # 'ndmi2_mean',
    # 'str_mean',
    'brightness_mean',
    'greenness_mean',
    'wetness_mean'
]

# List of variables to exclude from z-score filtering
no_outlier_check = ['ATMOS41_precipitation_mm']