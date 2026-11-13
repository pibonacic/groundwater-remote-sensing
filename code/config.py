"""Project parameters and decisions. Scripts import this module and never modify it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root; code/ lives inside it
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

# ---------------------------------------------------------------- GENERAL
DEV_YEARS = 2               # development = first DEV_YEARS years from the first piezometer day; everything after goes to the holdout
MIN_PAIRS = 30              # minimum overlapping days to compute a correlation
CCF_KS = range(-30, 31)     # k > 0: spectral value k days AFTER the water level
ANOMALY_HALF_WINDOW = 15      # anomalies (and the window climatology of 04) use the mean of the days within this many days either side of the day of the year
RAIN_WINDOWS = [3, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 30]   # candidate accumulation windows (days) for exploration
RAIN_MOISTURE_INDEX = "str1"            # index whose anomaly responds to rain
REGIME_THRESHOLD_CM = 0     # depth above this = "below surface" regime (depth is positive below the surface)

# Validation folds (hv_blocks, counted in days): consecutive validation blocks of hv_v days from the first development
# day (the last block absorbs the remainder), with hv_h days dropped from the training set on each side of the block
# (no gap before the first block nor after the last one).
# (hv_v and hv_h are set per site in SITES).
GAP = 0                     # days dropped between development and holdout
ACF_MAX_LAG = 120           # lags (days) of the ACF / PACF of the target in 02 (to explore hv_h)
INNER_SPLITS = 3            # temporal folds inside the ElasticNetCV of the hybrid model (no gap)
ENET_L1_RATIOS = [0.01, 0.1, 0.5, 0.9, 1.0]   # 0.01 is almost a Ridge regression
ENET_ALPHAS = [0.01, 0.03, 0.1, 0.3, 1, 3]   # alpha grid of the elastic_net candidate
ENET_MAX_ITER = 200000
SEED = 0                    # random state of every stochastic step

N_BOOTSTRAP = 1000          # block-bootstrap resamples of the out-of-fold predictions
BOOTSTRAP_BLOCK_DAYS = 30           # block length of the bootstrap on the out-of-fold predictions (05)
BOOTSTRAP_BLOCK_DAYS_HOLDOUT = 15   # block length of the bootstrap on the holdout predictions (08)
BOOTSTRAP_LEVEL = 0.95      # confidence level of the intervals

REDUNDANCY_THRESHOLD = 0.6  # |Spearman| above which two bands/indices count as redundant (exploration only)

# VIIRS NBAR band behind each generic name used by the spectral indices (only the I bands).
BANDS = {"red": "I1", "nir": "I2", "swir1": "I3"}

# ---------------------------------------------------------------- SITES
SITES = {
    "sdh1": {
        "piezo_path": RAW_DIR / "groundwater" / "SDH1_daily-insitu-data.csv",
        "piezo_column": "SDH1PS01_gw-depth_m",
        "nbar_path": RAW_DIR / "reflectance" / "SDH1_VIIRS_2012_01_2026_11.csv",
        "chirps_path": RAW_DIR / "precipitation" / "SDH1_CHIRPS_2012_01_01_2026_09_30.csv",
        "piezo_start": None,                # first piezometer day kept ("YYYY-MM-DD"); None = all records
        "hv_v": 60,                         # days per hv_blocks validation block
        "hv_h": 50,                         # days dropped on each side of the validation block
    },
    "sdh2": {
        "piezo_path": RAW_DIR / "groundwater" / "SDH2_daily-insitu-data.csv",
        "piezo_column": "SDH2PP01_gw-depth_m",
        "nbar_path": RAW_DIR / "reflectance" / "SDH2_VIIRS_2012_01_2026_11.csv",
        "chirps_path": RAW_DIR / "precipitation" / "SDH2_CHIRPS_2012_01_01_2026_09_30.csv",
        "piezo_start": "2024-05-24",        # 2026-11-13: earlier records are wrong (user's review of 01)
        "hv_v": 60,
        "hv_h": 82,                         # 2026-11-13: from the ACF / PACF of 02 (user's choice)
    },
}

# Shared by both sites for now; move into SITES[site] if they diverge.
for _site in SITES.values():
    _site.update({
        "piezo_unit": "m",                  # unit of the raw piezometer column
        "piezo_zero_cm": 0.0,               # offset subtracted after converting to cm
        "piezo_sign": -1,                   # multiplies the raw value so that depth is positive below the surface
        "physical_range_cm": (-30, 100),    # plausible depth range (positive = below surface)
        "max_jump_cm": 5.0,                 # day-to-day change flagged as a jump
        "nbar_qa": {"good_flags": (0, 1)},  # other BRDF quality flags mark a day qa_low (data is kept)
        "max_reflectance_z": 3.0,           # NBAR values with |z-score| above this are masked
    })

# ---------------------------------------------------------------- DECISIONS
# Filled after reading the outputs of 02_exploration.py (record date and reason).
# 2026-10-27: rain windows chosen after step 2 (rain reference index: str1).
# SPECTRAL: bands and indices that become features (your choice from the clusters of 02; leads are made only for them).
# 2026-10-29: SPECTRAL chosen by the user from the clusters of 02 (REDUNDANCY_THRESHOLD 0.6). LEADS empty
# (the CCF of the leads reflects autocorrelation).
# 2026-11-01: no variable selection before training (permutation, stability and ablation removed); FINAL_COLUMNS
# chosen by the user. WINDOWS [14] (windows 12-14 are equivalent in 02). MODEL and MODEL_PARAMS are filled from 06 (now 05).
# 2026-11-01: SPECTRAL = the 8 indices + bands I1, I2, I3 (user's choice); FINAL_COLUMNS = SPECTRAL + rain + calendar.
# 2026-11-02: indices that need M bands removed (gndvi, ndwi, mndwi, nbr, str2); added ndsaii, dvi, mlswi26, savi, evi2,
# bai, s3. SPECTRAL = all the indices that remain + I1, I2, I3, pending the redundancy review of 02 (M bands excluded there).
# 2026-11-02: evi2, s3 and I3 dropped from SPECTRAL / FINAL_COLUMNS (user's choice; 06 compared before and after).
# 2026-11-02: SPECTRAL reduced to ndvi, dvi, mlswi26, str1, ndsaii, ndmi (no raw bands; user's choice, 06 compared).
# 2026-11-03: SPECTRAL = all 17 indices (vibi dropped; baim and snirvswir use I3) + I1, I2, I3; FINAL_COLUMNS adds rain + calendar.
# 2026-11-03: evi2, baim, csiswir, bai and s3 dropped from SPECTRAL / FINAL_COLUMNS (user's choice; 06 compared).
# 2026-11-03: sdh1 WINDOWS emptied: no accumulated rain in the feature table (user's choice; 04 and 06 compared).
# 2026-11-03: rain window 14 restored in sdh1 (WINDOWS [14], pr_14d in FINAL_COLUMNS).
DECISIONS = {
    "sdh1": {"LEADS": [], "WINDOWS": [],
             "SPECTRAL": ["ndvi", "ndmi", "ndsaii", "str1", "dvi", "mlswi26", "savi", "mi", "ndsinw", "nli", "bai_burn",
                          "snirvswir", "I1", "I2", "I3"],
             "FINAL_COLUMNS": ["savi", "I1", "I2", "sin_doy", "cos_doy"], 
             "MODEL": None, 
             "MODEL_PARAMS": None
             },
    "sdh2": {"LEADS": [], "WINDOWS": [],
             "SPECTRAL": ["ndvi", "ndmi", "ndsaii", "str1", "dvi", "mlswi26", "savi", "mi", "ndsinw", "nli", "bai_burn",
                          "snirvswir", "I1", "I2", "I3"],
             "FINAL_COLUMNS": ['ndmi', 'savi', 'nli', 'I1', 'I3', 'sin_doy', 'cos_doy'], 
             "MODEL": None, 
             "MODEL_PARAMS": None},
}


def check_config():
    assert isinstance(DEV_YEARS, int) and DEV_YEARS > 0, "DEV_YEARS must be a positive int"
    for site, decisions in DECISIONS.items():
        assert site in SITES, f"DECISIONS has unknown site {site}"
        for name in ("LEADS", "WINDOWS"):
            values = decisions[name]
            if values is None:
                continue
            ok = all(isinstance(v, int) and v > 0 for v in values)
            assert ok, f"{site}.{name} must be positive integers, got {values}"
            assert len(set(values)) == len(values), f"{site}.{name} has repeated values"
        for name in ("SPECTRAL", "FINAL_COLUMNS"):
            values = decisions[name]
            assert values is None or len(set(values)) == len(values), f"{site}.{name} has repeated values"


check_config()
