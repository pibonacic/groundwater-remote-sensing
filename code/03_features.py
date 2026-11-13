"""Step 4: build the feature table, split development/holdout and save them."""
import config as cfg
from functions.data import load_table, save_table
from functions.features import (
    TARGET, build_feature_table, compute_partition, split_dev_holdout)

SITE = "sdh2"

leads = cfg.DECISIONS[SITE]["LEADS"]
windows = cfg.DECISIONS[SITE]["WINDOWS"]
spectral = cfg.DECISIONS[SITE]["SPECTRAL"]
if leads is None or windows is None or spectral is None:
    raise SystemExit(f"Set LEADS, WINDOWS and SPECTRAL for {SITE} in config.py DECISIONS first.")

# 1. Load the clean daily table from step 1.
daily = load_table(cfg.INTERIM_DIR / f"{SITE}_daily.csv")
y = daily[TARGET]
bands = daily.drop(columns=[TARGET, "qa_low", "precip_mm"])

# 2. Build the features on the WHOLE series (chosen bands and indices, leads, rain windows, calendar).
#    Shifting or accumulating before trimming would lose rows and mix periods.
table = build_feature_table(bands, daily["precip_mm"], leads, windows, cfg.BANDS, spectral)

# 3. Cut development / holdout. The development is the first DEV_YEARS years of the
#    piezometer record and the holdout is everything after it. GAP days between the two
#    are dropped so the holdout does not touch the development.
gap = cfg.GAP
partition = compute_partition(y.dropna().index, cfg.DEV_YEARS, gap)
(X_dev, y_dev), (X_hold, y_hold) = split_dev_holdout(table, y, partition)

# 4. Save: development and holdout in separate files (only script 08 reads the holdout).
save_table(y_dev.to_frame().join(X_dev), cfg.PROCESSED_DIR / f"dev_{SITE}.csv")
save_table(y_hold.to_frame().join(X_hold), cfg.PROCESSED_DIR / f"holdout_{SITE}.csv")

# 5. Summary: check the periods, the gap and the number of features per group.
print(f"{SITE}: SPECTRAL={spectral} LEADS={leads} WINDOWS={windows} gap={gap} days")
print(f"development: {y_dev.index.min().date()} to {y_dev.index.max().date()}, {len(y_dev)} rows")
print(f"holdout:     {y_hold.index.min().date()} to {y_hold.index.max().date()}, {len(y_hold)} rows")
print(f"features: {table.shape[1]} columns")
