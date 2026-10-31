"""Steps 5-6: reference models (climatology, linear) evaluated with temporal folds."""
import pandas as pd

import config as cfg
from functions.data import save_table
from functions.features import load_development, season_of
from functions.models import SeasonalClimatology, make_linear_baseline
from functions.plots import oof_plot
from functions.validation import compare, halves_splits, make_split_sets, oof_report

SITE = "sdh1"

out_dir = cfg.OUTPUTS_DIR / SITE / "04_baselines"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the development period only (the holdout is never loaded here).
X, y = load_development(cfg.PROCESSED_DIR / f"dev_{SITE}.csv")
gap = max(cfg.DECISIONS[SITE]["LEADS"], default=0)
low, high = cfg.SITES[SITE]["physical_range_cm"]

# 2. Fold schemes: forward (past -> future), backward (future -> past, the hindcast
#    direction), joint (both together) and halves_fwd / halves_bwd (train on one half, test on the other; diagnostic only).
split_sets = make_split_sets(X.index, gap, cfg.N_SPLITS, cfg.TEST_SIZE, cfg.MIN_TRAIN)
split_sets.update(halves_splits(X.index, gap))

# 3. Reference models: monthly climatology (and a smooth sin/cos version) and an
#    Elastic Net on all features. Any later model has to beat these.
models = {
    "climatology": SeasonalClimatology(),
    "climatology_harmonic": SeasonalClimatology(harmonics=True),
    "linear": make_linear_baseline(gap, cfg.INNER_SPLITS, cfg.ENET_L1_RATIOS, cfg.ENET_MAX_ITER),
}

# 4. Evaluate each model on each scheme; skill is measured against the climatology.
table, oof = compare(models, X, y, split_sets, baseline="climatology")

# 5. Pooled out-of-fold error overall and per season (always report by season).
by_season = oof_report(oof, y, high - low, {"season": season_of(y.index)}, baseline="climatology")

# 6. Save the tables, the out-of-fold predictions and a figure.
save_table(table.set_index(["scheme", "model"]), out_dir / "baselines.csv")
save_table(by_season, out_dir / "baselines_by_season.csv")
oof = {scheme: by_model for scheme, by_model in oof.items() if scheme != "joint"}   # joint = fwd + bwd, nothing new to draw
for scheme, by_model in oof.items():
    save_table(pd.DataFrame(by_model), out_dir / f"oof_{scheme}.csv")
oof_plot(y, oof, f"{SITE}: out-of-fold predictions").savefig(out_dir / "oof.png", dpi=150)

# 7. Summary. skill > 0 means better than the monthly climatology.
print(f"{SITE}: {len(X)} development rows, {X.shape[1]} features, gap={gap} days")
print(table.round(3).to_string(index=False))
print(by_season[by_season["stratum"] == "season"][["scheme", "model", "group", "n", "rmse", "skill_vs_climatology"]]
      .round(3).to_string(index=False))
