"""Step 4: build the feature table, split development/holdout and save them."""
import config as cfg
from functions.data import load_table, save_table
from functions.features import (
    TARGET, build_feature_table, compute_partition, modelling_days, split_dev_holdout)

SITE = "sdh1"

decisions = cfg.decisions(SITE)
leads, windows, spectral = decisions["LEADS"], decisions["WINDOWS"], decisions["SPECTRAL"]
if leads is None or windows is None or spectral is None:
    raise SystemExit(f"Set LEADS, WINDOWS and SPECTRAL for {SITE} in config.py DECISIONS first.")

# 1. Load the clean daily table from step 1.
daily = load_table(cfg.INTERIM_DIR / f"{SITE}_daily.csv")
y = daily[TARGET]
bands = daily.drop(columns=[TARGET, "qa_low", "precip_mm"])

# 2. Build the features on the WHOLE series (chosen bands and indices, leads, rain windows, calendar).
#    Shifting or accumulating before trimming would lose rows and mix periods.
table = build_feature_table(bands, daily["precip_mm"], leads, windows, cfg.BANDS, spectral)

# 3. Cut development / holdout with the scheme PARTITION of config.py, on the days with a complete row (water level,
#    the NBAR bands of BANDS and rain only if there are rain WINDOWS): "dev_2y" = the first 2 years are development,
#    "hold_1y" = the last 365 days are holdout. GAP days between the two are dropped so the holdout does not touch
#    the development.
gap = cfg.GAP
scheme = cfg.PARTITIONS[cfg.PARTITION]
partition = compute_partition(modelling_days(daily, cfg.BANDS.values(), bool(windows)), scheme, gap)
(X_dev, y_dev), (X_hold, y_hold) = split_dev_holdout(table, y, partition)
if "holdout_days" in scheme and len(y_hold) != scheme["holdout_days"]:
    raise SystemExit(f"holdout has {len(y_hold)} rows, not {scheme['holdout_days']}: some holdout days lack a feature")

# 4. Save: development and holdout in separate files of this scheme (only script 08 reads the holdout).
save_table(y_dev.to_frame().join(X_dev), cfg.processed_path("development", SITE))
save_table(y_hold.to_frame().join(X_hold), cfg.processed_path("holdout", SITE))

# 5. Summary: check the periods, the gap and the number of features per group.
print(f"{SITE}: partition {cfg.PARTITION} {scheme}")
print(f"{SITE}: SPECTRAL={spectral} LEADS={leads} WINDOWS={windows} gap={gap} days")
print(f"development: {y_dev.index.min().date()} to {y_dev.index.max().date()}, {len(y_dev)} rows")
print(f"holdout:     {y_hold.index.min().date()} to {y_hold.index.max().date()}, {len(y_hold)} rows")
print(f"features: {table.shape[1]} columns")
