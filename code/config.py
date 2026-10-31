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

# Validation folds (counted in days; the gap between train and test comes from the leads).
N_SPLITS = 4                # number of forward/backward folds
TEST_SIZE = 85              # days per test fold
MIN_TRAIN = 365             # minimum days in the first training set
INNER_SPLITS = 3            # temporal folds inside ElasticNetCV
ENET_L1_RATIOS = [0.1, 0.5, 0.9, 1.0]
ENET_MAX_ITER = 200000
SEED = 0                    # random state of every stochastic step

REDUNDANCY_THRESHOLD = 0.6  # |Spearman| above which two bands/indices count as redundant (exploration only)

# Variable selection (script 05)
N_PERMUTATIONS = 10         # shuffles per group and fold
STABLE_MIN_FRAC = 0.8       # share of folds in which the Elastic Net must keep a group
BLOCKS = ("spectral", "lead", "rain", "calendar")
ABLATION_MODELS = ["elasticnet", "boosting"]   # models of the ablation (default hyperparameters); the first decides the proposal
ABLATION_VERSIONS = {       # nested column sets compared in the ablation, simplest first
    "V1": ["spectral"], "V2": ["spectral", "rain"], "V3": ["spectral", "rain", "calendar"]}

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
    })

# ---------------------------------------------------------------- DECISIONS
# Filled after reading the outputs of 02_exploration.py (record date and reason).
# 2026-10-27: rain windows chosen after step 2 (rain reference index: str1). Leads
# chosen based on knowledge of the study area
# SPECTRAL: bands and indices that become features (your choice from the clusters of 02; leads are made only for them).
# 2026-10-29: LEADS empty (CCF of leads reflects autocorrelation), WINDOWS [12] (2026-10-30; windows 12-14 are equivalent in 02), KEPT_BLOCKS spectral/rain/calendar.
# 2026-10-29: SPECTRAL chosen by the user from the clusters of 02 (REDUNDANCY_THRESHOLD 0.6). FINAL_COLUMNS = the 7 columns of V3 chosen by the user (2026-10-29; 05 alone kept only sin_doy, which loses to the climatology). 2026-10-28 (provisional, to be reviewed): KEPT_BLOCKS = all blocks because V1-V3 (no calendar) are far worse than V4 in 05 and the
# "simplest within noise" rule picked V1, which leaves no columns. FINAL_COLUMNS = output of 05 with
# those blocks (re-set 2026-10-29 after SPECTRAL changed; provisional). MODEL and MODEL_PARAMS are filled from 06.
DECISIONS = {
    "sdh1": {"LEADS": [], "WINDOWS": [12],
             "SPECTRAL": ["str2", "ndmi", "ndvi", "nbr"],
             "KEPT_BLOCKS": ["spectral", "rain", "calendar"], "FINAL_COLUMNS": ["str2", "ndmi", "ndvi", "nbr", "pr_12d", "sin_doy", "cos_doy"], "MODEL": None, "MODEL_PARAMS": None},
    "sdh2": {"LEADS": [], "WINDOWS": [12],
             "SPECTRAL": None,
             "KEPT_BLOCKS": None, "FINAL_COLUMNS": ["str2", "ndmi", "ndvi", "nbr", "pr_12d", "sin_doy", "cos_doy"], "MODEL": None, "MODEL_PARAMS": None},
}


def check_config():
    for version, names in ABLATION_VERSIONS.items():
        assert set(names) <= set(BLOCKS), f"ABLATION_VERSIONS {version} has unknown blocks"
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
        assert decisions["KEPT_BLOCKS"] is None or set(decisions["KEPT_BLOCKS"]) <= set(BLOCKS), f"{site}.KEPT_BLOCKS has unknown blocks"
        for name in ("SPECTRAL", "FINAL_COLUMNS"):
            values = decisions[name]
            assert values is None or len(set(values)) == len(values), f"{site}.{name} has repeated values"


check_config()
