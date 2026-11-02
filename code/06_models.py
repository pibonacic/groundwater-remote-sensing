"""Step 10: tune and compare the candidate models on the final columns."""
import pandas as pd

import config as cfg
from functions.data import save_table
from functions.features import load_development, season_of
from functions.models import (
    MODEL_SPECS, SeasonalClimatology, build_pipeline, enet_params, pipeline_grid, pipeline_simplest, tune)
from functions.plots import bootstrap_ci_plot, folds_bars, folds_boxplot, halves_plot, oof_joint_plot, splits_plot
from functions.validation import block_bootstrap, compare, make_split_sets, oof_report, split_roles, summarize_folds

SITE = "sdh1"

decisions = cfg.DECISIONS[SITE]
if decisions["FINAL_COLUMNS"] is None:
    raise SystemExit(f"Set FINAL_COLUMNS for {SITE} in config.py DECISIONS first (your choice of features).")
out_dir = cfg.OUTPUTS_DIR / SITE / "06_models"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the development period only, the folds and the clip limits (the physical range of the level).
#    joint (forward folds and their backward mirror) is the validation scheme; halves_fwd / halves_bwd
#    (one cut each) are diagnostics of the tuned models. The inner ElasticNetCV of the hybrid has no gap.
X, y = load_development(cfg.PROCESSED_DIR / f"dev_{SITE}.csv")
enet = enet_params(0, cfg.INNER_SPLITS, cfg.ENET_L1_RATIOS, cfg.ENET_MAX_ITER)
split_sets = make_split_sets(X.index, cfg.GAP, cfg.N_SPLITS, cfg.TEST_SIZE, cfg.MIN_TRAIN)
roles, blocks = split_roles(X.index, split_sets["joint"])
limits = cfg.SITES[SITE]["physical_range_cm"]
low, high = limits

# 2. Tune every candidate on the folds of joint: the most regularized configuration within one
#    paired standard error (fold-by-fold differences) of the lowest mean RMSE.
models = {"climatology": SeasonalClimatology()}
tuning, best_params = [], {}
for name in MODEL_SPECS:
    pipe = build_pipeline(name, enet, decisions["FINAL_COLUMNS"], limits=limits, seed=cfg.SEED)
    best_params[name], summary = tune(pipe, pipeline_grid(name), pipeline_simplest(name), X, y, split_sets)
    models[name] = pipe.set_params(**best_params[name])
    tuning.append(summary.assign(model=name))

# 3. Evaluate the tuned models and the climatology on joint and on halves_fwd / halves_bwd, fold by fold;
#    skill against the monthly climatology on the same test rows.
by_fold, oof = compare(models, X, y, split_sets, "climatology", high - low)
summary = summarize_folds(by_fold)
by_season = oof_report(oof, y, high - low, {"season": season_of(y.index)}, baseline="climatology")

# 4. Block-bootstrap intervals of RMSE and skill on the halves predictions only (paired between models).
halves = {scheme: oof[scheme] for scheme in ("halves_fwd", "halves_bwd")}
boot = block_bootstrap(y, halves, "climatology", cfg.BOOTSTRAP_BLOCK_DAYS, cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_LEVEL, cfg.SEED)

# 5. Save the tables, the joint out-of-fold predictions and the figures.
save_table(pd.concat(tuning, ignore_index=True), out_dir / "tuning.csv")
save_table(pd.DataFrame(oof["joint"]), out_dir / "oof_joint.csv")
save_table(summary, out_dir / "metrics_summary.csv")
save_table(by_fold, out_dir / "metrics_by_fold.csv")
save_table(by_season, out_dir / "models_by_season.csv")
save_table(boot.set_index(["scheme", "model"]), out_dir / "bootstrap.csv")
save_table(roles.rename_axis("date"), out_dir / "splits.csv")
joint_folds = by_fold[by_fold["scheme"] == "joint"]
oof_joint_plot(y, oof["joint"], f"{SITE}: joint out-of-fold predictions").savefig(out_dir / "oof_joint.png", dpi=150)
folds_boxplot(joint_folds, f"{SITE}: joint folds").savefig(out_dir / "folds_boxplot.png", dpi=150)
folds_bars(joint_folds, f"{SITE}: joint folds").savefig(out_dir / "folds_bars.png", dpi=150)
splits_plot(roles, blocks, y, f"{SITE}: joint folds (gap {cfg.GAP} days)").savefig(out_dir / "splits.png", dpi=150)
halves_plot(y, halves, X.index[len(X) // 2], f"{SITE}: halves predictions").savefig(out_dir / "halves.png", dpi=150)
bootstrap_ci_plot(boot, "climatology", f"{SITE}: halves (block bootstrap, {cfg.BOOTSTRAP_LEVEL:.0%} interval)").savefig(
    out_dir / "bootstrap_ci.png", dpi=150)

# 6. Summary. Then edit MODEL and MODEL_PARAMS in config.py with date and reason.
print(f"{SITE}: {len(X)} development rows ({X.index.min().date()} to {X.index.max().date()}), "
      f"columns {decisions['FINAL_COLUMNS']}, gap={cfg.GAP} days")
print("joint blocks:\n", blocks.to_string(index=False))
print("best parameters:", {k: {p.removeprefix('model__'): v for p, v in b.items()} for k, b in best_params.items()})
print(summary.round(3).to_string())
print("joint folds:\n", joint_folds.pivot_table(index=["metric", "model"], columns=["direction", "fold"], values="value", sort=False)
      .loc[["rmse", "skill"]].round(2).to_string())
print(boot.round(3).to_string(index=False))
print(by_season[by_season["stratum"] == "season"][["scheme", "model", "group", "n", "rmse", "skill_vs_climatology"]]
      .round(2).pivot_table(index=["scheme", "model"], columns="group", values="skill_vs_climatology").to_string())
