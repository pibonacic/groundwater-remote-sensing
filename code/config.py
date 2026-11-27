"""Project parameters and decisions. Scripts import this module and never modify it."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root; code/ lives inside it
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"

# ---------------------------------------------------------------- GENERAL
# Development / holdout cut. Each scheme has exactly one key: "dev_years" (development = the first dev_years years
# from the first valid day, the rest is holdout) or "holdout_days" (holdout = the last holdout_days valid days, the rest
# is development). Valid days: water level + the NBAR bands of BANDS (+ rain only when the site has rain WINDOWS).
# Outputs and processed tables of each scheme go to their own paths (out_dir, processed_path).
PARTITION = "hold_1y"      # "dev_2y" or "hold_1y"
PARTITIONS = {"dev_2y": {"dev_years": 2}, "hold_1y": {"holdout_days": 365}}
MIN_PAIRS = 30              # minimum overlapping days to compute a correlation
CCF_KS = range(-30, 31)     # k > 0: spectral value k days AFTER the water level
ANOMALY_HARMONICS_K = 2     # 02: anomaly = series minus its harmonic fit of order K (constant + sin/cos of k*theta, k = 1..K,
                            # theta = 2 pi (day of year - 1) / days in the year), fitted on the piezometric period
RAIN_WINDOWS = [3, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 30]   # candidate accumulation windows (days) for exploration
RAIN_MOISTURE_INDEX = "str1"            # index whose anomaly responds to rain
REGIME_THRESHOLD_CM = 0     # depth above this = "below surface" regime (depth is positive below the surface)

# Exploration (02) only: diagnostic, on the whole piezometric record (development + holdout, both schemes); outputs
# in outputs/<site>/02_exploration/. The decisions of the model are still taken on the development of each scheme.
REGIME_BINS_CM = (0, 20)    # three regimes by raw depth: ponded < 0, shallow 0-20, deep > 20 cm
SLOW_WINDOW_DAYS = 31       # slow part = centered rolling median of this many days; fast part = the rest
PHASE_SMOOTH_DAYS = 31      # rise / fall of the level: sign of the derivative of the centered rolling mean of the depth
PHASE_MIN_RATE_CM_DAY = 0.1  # days with |derivative| below this are neither rise nor fall
PHASE_SLOPE_N_BOOTSTRAP = 200  # block-bootstrap draws of the Theil-Sen slopes (slow: all pairs of days)
TRAJECTORY_DAYS = 7         # rolling mean of the trajectory drawn on the scatters colored by date
JUMP_RECOVERY_DAYS = 10     # a jump is recovered if the level returns more than JUMP_RECOVERY_FRACTION of it
JUMP_RECOVERY_FRACTION = 0.5  # within JUMP_RECOVERY_DAYS days (jumps: |day-to-day change| > max_jump_cm of SITES)
EVENT_MIN_RISE_CM = 2.0     # rise events (consecutive days of decreasing depth) with at least this rise ...
EVENT_MIN_RAIN_MM = 2.0     # ... and at least this rain from EVENT_RAIN_LEAD_DAYS before the start to the end
EVENT_RAIN_LEAD_DAYS = 2    # give the apparent specific yield Sy = rain (mm) / rise (mm)
SEPTEMBER_VARIABLES = ["I1", "I2", "I3", "mlswi26", "str1", "bai_burn"]
SEPTEMBER_MIN_PAIRS = 20    # September of one year has at most 30 days (MIN_PAIRS would drop most years)
ANNUAL_ANOMALY_VARIABLES = ["I3", "mlswi26"]   # yearly / seasonal anomalies vs the 2012-2026 climatology, detrended
HYDRO_YEAR_START_MONTH = 4  # hydrological year = April to March, labeled by the year it starts (2025 = Apr 2025 - Mar 2026)

# Validation folds (hv_blocks, counted in days): consecutive validation blocks of hv_v days from the first development
# day (the last block absorbs the remainder), with hv_h days dropped from the training set on each side of the block
# (no gap before the first block nor after the last one).
# (hv_v and hv_h are set per site in SITES).
GAP = 0                     # days dropped between development and holdout
ACF_MAX_LAG = 120           # lags (days) of the ACF / PACF of the target in 02 (to explore hv_h)
ENET_MAX_ITER = 200000

# Hyperparameter grids of 05 (tuned on the hv_blocks folds by the mean RMSE across folds). The hybrids use ENET_GRID
# (their ElasticNet) and RF_GRID (their trees): hybrid_joint tunes both together by the RMSE of the hybrid,
# hybrid_staged first the ElasticNet alone and then the trees on top of it (see MODEL_SPECS in functions/models.py).
ENET_GRID = {"alpha": [0.01, 0.1, 1, 10, 100],   # elastic_net and the ElasticNet of the hybrid
             "l1_ratio": [0.01, 0.5, 1.0]}       # 0.01 is almost a Ridge regression
RF_GRID = {"max_depth": [4, 8, None], "min_samples_leaf": [5, 10, 20, 40], "max_features": [0.5, 1.0]}   # random_forest and the trees of the hybrid
CATBOOST_GRID = {"learning_rate": [0.02, 0.05], "depth": [2, 3, 4], "l2_leaf_reg": [3, 10, 30]}
SEED = 0                    # random state of every stochastic step

N_BOOTSTRAP = 1000          # block-bootstrap resamples of the out-of-fold predictions
BOOTSTRAP_BLOCK_DAYS = 30           # block length of the bootstrap on the out-of-fold predictions (05)
BOOTSTRAP_BLOCK_DAYS_HOLDOUT = 30   # block length of the bootstrap on the holdout predictions (07); 2026-11-18: 15 -> 30 (user)
BOOTSTRAP_LEVEL = 0.95      # confidence level of the intervals

# Predictions of 08 (whole satellite series): the final model is refit on development + holdout; its uncertainty
# combines an ensemble refit on block-bootstrap resamples of the training days (blocks of BOOTSTRAP_BLOCK_DAYS) with
# out-of-fold hv_blocks residuals of the same season (see functions/prediction.py).
N_ENSEMBLE = 200            # members of the block-bootstrap ensemble
PREDICTION_LEVELS = (0.8, 0.95)   # central prediction intervals

REDUNDANCY_THRESHOLD = 0.9  # |Spearman| above which two bands/indices count as redundant (exploration only)

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
        "hv_v": 66,                         # days per hv_blocks validation block
        "hv_h": 50,                         # days dropped on each side of the validation block
    },
    "sdh2": {
        "piezo_path": RAW_DIR / "groundwater" / "SDH2_daily-insitu-data.csv",
        "piezo_column": "SDH2PP01_gw-depth_m",
        "nbar_path": RAW_DIR / "reflectance" / "SDH2_VIIRS_2012_01_2026_11.csv",
        "chirps_path": RAW_DIR / "precipitation" / "SDH2_CHIRPS_2012_01_01_2026_09_30.csv",
        "piezo_start": "2024-05-24",        # 2026-11-13: earlier records are wrong (user's review of 01)
        "hv_v": 66,
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
# 2026-11-13: the development / holdout cut is chosen with PARTITION ("dev_2y", the scheme up to commit 292178c, or
# "hold_1y", last 365 valid days as holdout) to compare both without switching branches. FINAL_COLUMNS, MODEL and
# MODEL_PARAMS can differ per scheme through BY_PARTITION (overrides of the site base); for now both schemes use the
# same FINAL_COLUMNS. Scripts read the combined decisions with decisions(site).
# 2026-11-14: scheme holdout_1y renamed hold_1y (scheme names stay short); processed tables are named
# development_<site>_<scheme>.csv and holdout_<site>_<scheme>.csv. The ElasticNet of the hybrid is no longer an
# ElasticNetCV with inner folds: its alpha and l1_ratio are tuned on hv_blocks with the trees (grid of 05), like the
# elastic_net candidate.
# 2026-11-16: all hyperparameter grids in config.py (ENET_GRID, RF_GRID, CATBOOST_GRID); ENET grid reduced to
# alpha [0.01, 0.1, 1, 10, 100] x l1_ratio [0.01, 0.5, 1.0] (hybrid: 360 combinations instead of 1080).
# FINAL_COLUMNS per scheme (BY_PARTITION) = a step of the backward elimination of 06 with hybrid on the 17 columns
# (user's choice; 05 on the 17 columns kept in 05_models_all_vars): sdh1 dev_2y step 14 (3 columns), sdh1 hold_1y
# step 10 (7), sdh2 dev_2y step 11 (6), sdh2 hold_1y step 13 (4). The base FINAL_COLUMNS stay the 17 columns.
# 2026-11-16: sdh1 dev_2y changed to step 10 (7 columns; user's choice after 08 with step 14).
# 2026-11-16: two versions of the hybrid kept to compare them: hybrid_joint (the previous hybrid, ElasticNet and
# trees tuned together) and hybrid_staged (ElasticNet first, then the trees with it fixed). 06 runs the backward
# elimination for both. Simplified the same day: hybrid_staged reuses the alpha / l1_ratio chosen for elastic_net
# in 05 and only tunes its trees, because its own ElasticNet stage was the same model as elastic_net and chose the
# same alpha / l1_ratio with the same rmse_mean in the 4 runs.
# 2026-11-16: FINAL_COLUMNS per scheme = the set with n columns on the backward elimination path of 06 with
# hybrid_staged on the 17 columns, v=66 (user's choice; always the path of hybrid_staged): sdh1 dev_2y 3 columns
# (step 14), sdh1 hold_1y 4 (step 13), sdh2 dev_2y 6 (step 11), sdh2 hold_1y 4 (step 13). 05 on the 17 columns kept
# in 05_models_all_vars. The previous choices are left commented out.
# 2026-11-16: MODEL = hybrid_staged for both sites and schemes (user's choice after 07, the old 08); 08_predictions
# refits it on development + holdout with the parameters chosen in 05 (MODEL_PARAMS None = the chosen row of tuning.csv).
# Script 08_holdout renamed 07_holdout (and its output folders).
DECISIONS = {
    "sdh1": {"LEADS": [], "WINDOWS": [],
             "SPECTRAL":      ["ndvi", "ndmi", "ndsaii", "str1", "dvi", "mlswi26", "savi", "mi", "ndsinw", "nli", "bai_burn", "snirvswir", "I1", "I2", "I3"],
             "FINAL_COLUMNS": ["ndvi", "ndmi", "ndsaii", "str1", "dvi", "mlswi26", "savi", "mi", "ndsinw", "nli", "bai_burn", "snirvswir", "I1", "I2", "I3", "sin_doy", "cos_doy"], 
             "MODEL": "hybrid_staged",
             "MODEL_PARAMS": None,
             "BY_PARTITION": {   # scheme -> overrides of FINAL_COLUMNS / MODEL / MODEL_PARAMS
                 #"dev_2y": {"FINAL_COLUMNS": ["dvi", "mi", "ndsinw", "bai_burn", "I1", "sin_doy", "cos_doy"]},
                 #"hold_1y": {"FINAL_COLUMNS": ["str1", "dvi", "mlswi26", "I2", "I3", "sin_doy", "cos_doy"]},
                 "dev_2y": {"FINAL_COLUMNS": ["I3", "sin_doy", "cos_doy"]},
                 "hold_1y": {"FINAL_COLUMNS": ["mlswi26", "I2", "sin_doy", "cos_doy"]},
             },
             },
    "sdh2": {"LEADS": [], "WINDOWS": [],
             "SPECTRAL":      ["ndvi", "ndmi", "ndsaii", "str1", "dvi", "mlswi26", "savi", "mi", "ndsinw", "nli", "bai_burn", "snirvswir", "I1", "I2", "I3"],
             "FINAL_COLUMNS": ["ndvi", "ndmi", "ndsaii", "str1", "dvi", "mlswi26", "savi", "mi", "ndsinw", "nli", "bai_burn", "snirvswir", "I1", "I2", "I3", "sin_doy", "cos_doy"], 
             "MODEL": "hybrid_staged",
             "MODEL_PARAMS": None,
             "BY_PARTITION": {   # scheme -> overrides of FINAL_COLUMNS / MODEL / MODEL_PARAMS
                 #"dev_2y": {"FINAL_COLUMNS": ["mlswi26", "nli", "I2", "I3", "sin_doy", "cos_doy"]},
                 #"hold_1y": {"FINAL_COLUMNS": ["savi", "bai_burn", "I3", "sin_doy"]},
                 "dev_2y": {"FINAL_COLUMNS": ["mlswi26", "nli", "I2", "I3", "sin_doy", "cos_doy"]},
                 "hold_1y": {"FINAL_COLUMNS": ["mlswi26", "mi", "bai_burn", "sin_doy"]},
             },
             },
}
# All vars: "ndvi", "ndmi", "ndsaii", "str1", "dvi", "mlswi26", "savi", "mi", "ndsinw", "nli", "bai_burn", "snirvswir", "I1", "I2", "I3", "sin_doy", "cos_doy"

PARTITION_OVERRIDES = ("FINAL_COLUMNS", "MODEL", "MODEL_PARAMS")


def decisions(site):
    """Decisions of a site for the current PARTITION: the site base with the overrides of BY_PARTITION[PARTITION]."""
    base = {k: v for k, v in DECISIONS[site].items() if k != "BY_PARTITION"}
    return base | DECISIONS[site].get("BY_PARTITION", {}).get(PARTITION, {})


def out_dir(site, step):
    """Output folder of a step for the current PARTITION (created if missing)."""
    path = OUTPUTS_DIR / site / PARTITION / step
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_path(kind, site):
    """Development ("development") or holdout ("holdout") table of a site for the current PARTITION."""
    assert kind in ("development", "holdout"), f"kind must be development or holdout, got {kind}"
    return PROCESSED_DIR / f"{kind}_{site}_{PARTITION}.csv"


def check_config():
    assert PARTITION in PARTITIONS, f"PARTITION must be one of {list(PARTITIONS)}, got {PARTITION}"
    for name, scheme in PARTITIONS.items():
        keys = set(scheme) & {"dev_years", "holdout_days"}
        assert len(keys) == 1 and len(scheme) == 1, f"PARTITIONS[{name}] needs exactly one of dev_years / holdout_days"
        value = scheme[keys.pop()]
        assert isinstance(value, int) and value > 0, f"PARTITIONS[{name}] must be a positive int, got {value}"
    for site in DECISIONS:
        assert site in SITES, f"DECISIONS has unknown site {site}"
        for scheme, overrides in DECISIONS[site].get("BY_PARTITION", {}).items():
            assert scheme in PARTITIONS, f"{site}.BY_PARTITION has unknown scheme {scheme}"
            extra = set(overrides) - set(PARTITION_OVERRIDES)
            assert not extra, f"{site}.BY_PARTITION[{scheme}] can only override {PARTITION_OVERRIDES}, got {extra}"
        for scheme in PARTITIONS:
            _check_decisions(site, {k: v for k, v in DECISIONS[site].items() if k != "BY_PARTITION"}
                             | DECISIONS[site].get("BY_PARTITION", {}).get(scheme, {}))


def _check_decisions(site, site_decisions):
    for name in ("LEADS", "WINDOWS"):
        values = site_decisions[name]
        if values is None:
            continue
        ok = all(isinstance(v, int) and v > 0 for v in values)
        assert ok, f"{site}.{name} must be positive integers, got {values}"
        assert len(set(values)) == len(values), f"{site}.{name} has repeated values"
    model = site_decisions["MODEL"]
    assert model is None or isinstance(model, str), f"{site}.MODEL must be None or a name of MODEL_SPECS, got {model}"
    params = site_decisions["MODEL_PARAMS"]
    assert params is None or isinstance(params, dict), f"{site}.MODEL_PARAMS must be None or a dict, got {params}"
    for name in ("SPECTRAL", "FINAL_COLUMNS"):
        values = site_decisions[name]
        assert values is None or len(set(values)) == len(values), f"{site}.{name} has repeated values"


check_config()
