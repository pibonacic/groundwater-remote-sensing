"""Step 10: tune and compare the candidate models on the final columns with the hv_blocks folds."""
import pandas as pd

import config as cfg
from functions.data import save_table
from functions.features import load_development, season_of
from functions.models import MODEL_SPECS, SeasonalClimatology, build_pipeline, enet_params, pipeline_grid, tune
from functions.plots import bootstrap_ci_plot, by_fold_boxplot, oof_predictions_plot, season_metrics_plot, taylor_plot
from functions.validation import block_bootstrap, compare, hv_block_splits, metrics_table, summarize_folds

SITE = "sdh2"
hv_v, hv_h = cfg.SITES[SITE]["hv_v"], cfg.SITES[SITE]["hv_h"]   # hv_blocks folds of this site
BASELINE = "climatology"

decisions = cfg.DECISIONS[SITE]
if decisions["FINAL_COLUMNS"] is None:
    raise SystemExit(f"Set FINAL_COLUMNS for {SITE} in config.py DECISIONS first (your choice of features).")
out_dir = cfg.OUTPUTS_DIR / SITE / "05_models"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the development period only, the hv_blocks folds and the clip limits (the physical range of the depth).
#    The inner ElasticNetCV of the hybrid has no gap.
X, y = load_development(cfg.PROCESSED_DIR / f"dev_{SITE}.csv")
enet = enet_params(0, cfg.INNER_SPLITS, cfg.ENET_L1_RATIOS, cfg.ENET_MAX_ITER)
splits = hv_block_splits(X.index, hv_v, hv_h)
limits = cfg.SITES[SITE]["physical_range_cm"]

# 2. Tune every candidate on the hv_blocks folds: the configuration with the lowest mean RMSE across folds.
models = {BASELINE: SeasonalClimatology()}
tuning, best_params = [], {}
for name in MODEL_SPECS:
    pipe = build_pipeline(name, enet, cfg, decisions["FINAL_COLUMNS"], limits=limits, seed=cfg.SEED)
    best_params[name], summary = tune(pipe, pipeline_grid(name, cfg), X, y, splits)
    models[name] = pipe.set_params(**best_params[name])
    tuning.append(summary.assign(model=name))

# 3. Evaluate the tuned models and the climatology fold by fold (skill against the climatology on the same days),
#    then on the pooled out-of-fold predictions, overall and by season.
by_fold, oof = compare(models, X, y, splits, BASELINE)
summary = summarize_folds(by_fold)
preds = oof.drop(columns="observed")
oof_metrics = metrics_table(y, preds, BASELINE, {"season": season_of(y.index)})

# 4. Block-bootstrap intervals of the pooled RMSE, ME and skill (paired between models).
boot = block_bootstrap(y, preds, BASELINE, cfg.BOOTSTRAP_BLOCK_DAYS, cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_LEVEL, cfg.SEED)

# 5. Save the tables and the figures.
save_table(pd.concat(tuning, ignore_index=True).set_index("model"), out_dir / "tuning.csv")
save_table(oof, out_dir / "oof_predictions.csv")
save_table(by_fold, out_dir / "by_fold_metrics.csv")
save_table(summary, out_dir / "by_fold_metrics_summary.csv")
save_table(oof_metrics.set_index("model"), out_dir / "oof_predictions_metrics.csv")
save_table(boot, out_dir / "bootstrap_metrics.csv")
oof_predictions_plot(oof, f"{SITE}: hv_blocks out-of-fold predictions").savefig(out_dir / "oof_predictions.png", dpi=150)
by_fold_boxplot(by_fold, BASELINE, f"{SITE}: metrics by hv_blocks fold ({len(splits)} folds)").savefig(
    out_dir / "by_fold_boxplot.png", dpi=150)
season_metrics_plot(oof_metrics, BASELINE, f"{SITE}: pooled out-of-fold metrics by season (hv_blocks)").savefig(
    out_dir / "oof_predictions_metrics.png", dpi=150)
bootstrap_ci_plot(boot, BASELINE, f"{SITE}: hv_blocks out-of-fold (block bootstrap, {cfg.BOOTSTRAP_BLOCK_DAYS}-day blocks, "
                  f"{cfg.BOOTSTRAP_LEVEL:.0%} interval)").savefig(out_dir / "bootstrap_metrics.png", dpi=150)
taylor_plot(oof_metrics[oof_metrics["stratum"] == "all"], y.std(ddof=0),
            f"{SITE}: Taylor diagram, hv_blocks out-of-fold").savefig(out_dir / "taylor.png", dpi=150)

# 6. Summary. Then edit MODEL and MODEL_PARAMS in config.py with date and reason.
print(f"{SITE}: {len(X)} development rows ({X.index.min().date()} to {X.index.max().date()}), "
      f"columns {decisions['FINAL_COLUMNS']}, {len(splits)} hv_blocks folds (v={hv_v}, h={hv_h})")
print("best parameters:", {k: {p.removeprefix('model__regressor__'): v for p, v in b.items()} for k, b in best_params.items()})
print("by fold (mean / std):\n", summary.round(3).to_string())
print("rmse by fold:\n", by_fold["rmse"].unstack("fold").round(2).to_string())
print(boot.round(3).to_string())
print("skill by season:\n", oof_metrics.pivot_table(index="model", columns="group", values="skill", sort=False).round(2).to_string())
