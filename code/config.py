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
    's2_B2',
    # 's2_B3',
    # 's2_B4',
    # 's2_B8',
    # 's2_B11',
    # 's2_B12',
    's2_ndvi',
    's2_ndwi',
    's2_mndwi',
    's2_ndmi',
    's2_ndmi2',
    's2_str',
    # 's2_brightness',
    # 's2_greenness',
    # 's2_wetness'
]

# List of variables to exclude from z-score filtering
no_outlier_check = ['ATMOS41_precipitation_mm']