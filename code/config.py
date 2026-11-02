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
RAIN_WINDOWS = [3, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 30]   # candidate accumulation windows (days) for exploration
RAIN_MOISTURE_INDEX = "str1"            # index whose anomaly responds to rain
REGIME_THRESHOLD_CM = 0     # water level at/below this = "below surface" regime

# Validation folds (counted in days). The development must have exactly MIN_TRAIN + GAP + N_SPLITS * TEST_SIZE rows.
N_SPLITS = 5                # number of forward folds (and as many backward) of the joint scheme
TEST_SIZE = 70              # days per test fold
MIN_TRAIN = 365             # days in the first training set
GAP = 15                    # days dropped between train and test (joint, halves) and between development and holdout
INNER_SPLITS = 3            # temporal folds inside ElasticNetCV (no gap)
ENET_L1_RATIOS = [0.1, 0.5, 0.9, 1.0]
ENET_MAX_ITER = 200000
SEED = 0                    # random state of every stochastic step

N_BOOTSTRAP = 1000          # block-bootstrap resamples of the out-of-fold predictions
BOOTSTRAP_BLOCK_DAYS = 30
BOOTSTRAP_LEVEL = 0.95      # confidence level of the intervals

REDUNDANCY_THRESHOLD = 0.6  # |Spearman| above which two bands/indices count as redundant (exploration only)

# VIIRS NBAR band behind each generic name used by the spectral indices.
BANDS = {"green": "M4", "red": "I1", "nir": "I2", "swir1": "I3", "swir2": "M11"}

# ---------------------------------------------------------------- SITES
SITES = {
    "sdh1": {
        "piezo_path": RAW_DIR / "groundwater" / "SDH1_daily-insitu-data.csv",
        "piezo_column": "SDH1PS01_gw-depth_m",
        "nbar_path": RAW_DIR / "reflectance" / "SDH1_VIIRS_2012_01_2026_10.csv",
        "chirps_path": RAW_DIR / "precipitation" / "SDH1_CHIRPS_2012_01_01_2026_09_30.csv",
    },
    "sdh2": {
        "piezo_path": RAW_DIR / "groundwater" / "SDH2_daily-insitu-data.csv",
        "piezo_column": "SDH2PP01_gw-depth_m",
        # No SDH2 VIIRS file in data/raw/reflectance yet.
        "nbar_path": RAW_DIR / "reflectance" / "SDH2_VIIRS_2012_01_2026_10.csv",
        "chirps_path": RAW_DIR / "precipitation" / "SDH2_CHIRPS_2012_01_01_2026_09_30.csv",
    },
}

# Shared by both sites for now; move into SITES[site] if they diverge.
for _site in SITES.values():
    _site.update({
        "piezo_unit": "m",                  # unit of the raw piezometer column
        "piezo_zero_cm": 0.0,               # offset subtracted after converting to cm
        "physical_range_cm": (-100, 30),    # plausible level range
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
# chosen by the user. WINDOWS [14] (windows 12-14 are equivalent in 02). MODEL and MODEL_PARAMS are filled from 06.
# 2026-11-01: SPECTRAL = the 8 indices + bands I1, I2, I3 (user's choice); FINAL_COLUMNS = SPECTRAL + rain + calendar.
DECISIONS = {
    "sdh1": {"LEADS": [], "WINDOWS": [14],
             "SPECTRAL": ["ndvi", "gndvi", "ndwi", "mndwi", "ndmi", "nbr", "str1", "str2", "I1", "I2", "I3"],
             "FINAL_COLUMNS": ["ndvi", "gndvi", "ndwi", "mndwi", "ndmi", "nbr", "str1", "str2", "I1", "I2", "I3",
                               "pr_14d", "sin_doy", "cos_doy"], "MODEL": None, "MODEL_PARAMS": None},
    "sdh2": {"LEADS": [], "WINDOWS": [14],
             "SPECTRAL": None,
             "FINAL_COLUMNS": ["str2", "ndmi", "ndvi", "nbr", "pr_14d", "sin_doy", "cos_doy"], "MODEL": None, "MODEL_PARAMS": None},
}


def check_config():
    assert isinstance(DEV_YEARS, int) and DEV_YEARS > 0, "DEV_YEARS must be a positive int"
    for site, decisions in DECISIONS.items():
        assert site in SITES, f"DECISIONS has unknown site {site}"
        for name in ("LEADS", "WINDOWS"):
            values = decisions[name]
            if values is None:
                continue
            ok = (name == "LEADS" or len(values) > 0) and all(isinstance(v, int) and v > 0 for v in values)
            assert ok, f"{site}.{name} must be positive integers, got {values}"
            assert len(set(values)) == len(values), f"{site}.{name} has repeated values"
        for name in ("SPECTRAL", "FINAL_COLUMNS"):
            values = decisions[name]
            assert values is None or len(set(values)) == len(values), f"{site}.{name} has repeated values"


check_config()
