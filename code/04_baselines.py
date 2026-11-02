"""Steps 5-6: reference models (climatology, linear) evaluated with temporal folds."""
import pandas as pd

import config as cfg
from functions.data import save_table
from functions.features import load_development, season_of
from functions.models import SeasonalClimatology, make_linear_baseline
from functions.plots import oof_joint_plot
from functions.validation import block_bootstrap, compare, make_split_sets, oof_report, summarize_folds

SITE = "sdh1"

out_dir = cfg.OUTPUTS_DIR / SITE / "04_baselines"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the development period only (the holdout is never loaded here).
X, y = load_development(cfg.PROCESSED_DIR / f"dev_{SITE}.csv")
low, high = cfg.SITES[SITE]["physical_range_cm"]

# 2. Fold schemes: joint (expanding training window, forward folds and their backward mirror) to evaluate
#    the out-of-fold predictions, and the diagnostic halves_fwd / halves_bwd (one cut each) for the bootstrap.
split_sets = make_split_sets(X.index, cfg.GAP, cfg.N_SPLITS, cfg.TEST_SIZE, cfg.MIN_TRAIN)

# 3. Reference models: monthly climatology (and a smooth sin/cos version) and an
#    Elastic Net on all features (inner CV without gap). Any later model has to beat these.
models = {
    "climatology": SeasonalClimatology(),
    "climatology_harmonic": SeasonalClimatology(harmonics=True),
    "linear": make_linear_baseline(0, cfg.INNER_SPLITS, cfg.ENET_L1_RATIOS, cfg.ENET_MAX_ITER),
}

# 4. Evaluate each model on each scheme, fold by fold; skill is measured against the climatology.
by_fold, oof = compare(models, X, y, split_sets, "climatology", high - low)
summary = summarize_folds(by_fold)

# 5. Pooled out-of-fold error overall and per season (always report by season).
by_season = oof_report(oof, y, high - low, {"season": season_of(y.index)}, baseline="climatology")

# 6. Block-bootstrap intervals of RMSE and skill on the halves predictions only, then save the tables,
#    the joint out-of-fold predictions and a figure.
halves = {scheme: oof[scheme] for scheme in ("halves_fwd", "halves_bwd")}
boot = block_bootstrap(y, halves, "climatology", cfg.BOOTSTRAP_BLOCK_DAYS, cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_LEVEL, cfg.SEED)
save_table(boot.set_index(["scheme", "model"]), out_dir / "bootstrap.csv")
save_table(summary, out_dir / "metrics_summary.csv")
save_table(by_fold, out_dir / "metrics_by_fold.csv")
save_table(by_season, out_dir / "baselines_by_season.csv")
save_table(pd.DataFrame(oof["joint"]), out_dir / "oof_joint.csv")
oof_joint_plot(y, oof["joint"], f"{SITE}: joint out-of-fold predictions").savefig(out_dir / "oof_joint.png", dpi=150)

# 7. Summary. skill > 0 means better than the monthly climatology.
print(f"{SITE}: {len(X)} development rows, {X.shape[1]} features, gap={cfg.GAP} days")
print(summary.round(3).to_string())
print(boot.round(3).to_string(index=False))
print(by_season[by_season["stratum"] == "season"][["scheme", "model", "group", "n", "rmse", "skill_vs_climatology"]]
      .round(3).to_string(index=False))
