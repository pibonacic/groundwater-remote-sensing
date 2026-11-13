"""Step 11: SHAP importance of the features of every model and backward elimination by retraining for one model."""
import ast
import time

import pandas as pd

import config as cfg
from functions.data import save_table
from functions.features import load_development
from functions.importance import backward_elimination, feature_units, readd_check, shap_by_unit, shap_oof
from functions.models import MODEL_SPECS, build_pipeline, enet_params
from functions.plots import elimination_path_plot, readd_plot, shap_beeswarm_plot, shap_importance_plot
from functions.validation import hv_block_splits, rmse_diff_bootstrap

SITE = "sdh1"
MODEL = "hybrid"     # model of the backward elimination (retraining is the costly part; one model per run)

decisions = cfg.DECISIONS[SITE]
out_dir = cfg.OUTPUTS_DIR / SITE / "06_features"
out_dir.mkdir(parents=True, exist_ok=True)
start = time.time()

# 1. Development period, hv_blocks folds and the hyperparameters chosen in 05 (kept fixed here). Every column is a
#    unit of its own (sin_doy and cos_doy included); groups go in GROUPED_UNITS of functions/importance.py.
X, y = load_development(cfg.PROCESSED_DIR / f"dev_{SITE}.csv")
splits = hv_block_splits(X.index, cfg.HV_V, cfg.HV_H)
tuning = pd.read_csv(cfg.OUTPUTS_DIR / SITE / "05_models" / "tuning.csv")
enet = enet_params(0, cfg.INNER_SPLITS, cfg.ENET_L1_RATIOS, cfg.ENET_MAX_ITER)
limits = cfg.SITES[SITE]["physical_range_cm"]
units = feature_units(decisions["FINAL_COLUMNS"])
pipes = {name: build_pipeline(name, enet, cfg, decisions["FINAL_COLUMNS"], limits=limits, seed=cfg.SEED,
                              params=ast.literal_eval(tuning[(tuning["model"] == name) & tuning["chosen"]]["params"].iloc[0]))
         for name in MODEL_SPECS}

# 2. Out-of-fold SHAP of every model: each validation block is explained by the model fitted on its training days
#    (before the physical clip). Importance = mean |SHAP| of each unit (cm of depth).
unit_values = X
importance, shap_units = {}, {}
for name, pipe in pipes.items():
    values, _, _ = shap_oof(pipe, X, y, splits)
    shap_units[name] = shap_by_unit(values, units)
    importance[name] = shap_units[name].abs().mean()
importance = pd.DataFrame(importance).sort_values(MODEL, ascending=False)

# 3. Backward elimination for MODEL: at each step the unit whose removal leaves the lowest mean RMSE across folds is
#    removed (every candidate retrained on the 12 folds). Bootstrap (blocks of BOOTSTRAP_BLOCK_DAYS, paired) of the
#    pooled RMSE of each step and of its difference with all units and with the best step.
path, oofs = backward_elimination(pipes[MODEL], units, X, y, splits)
best = int(path["rmse_mean"].idxmin())
boot = dict(block_days=cfg.BOOTSTRAP_BLOCK_DAYS, n_boot=cfg.N_BOOTSTRAP, level=cfg.BOOTSTRAP_LEVEL, seed=cfg.SEED)
vs_full = rmse_diff_bootstrap(y, oofs, "step_0", **boot)
vs_best = rmse_diff_bootstrap(y, oofs, f"step_{best}", **boot)
path = path.join(vs_full.set_axis(path.index).rename(columns=lambda c: c.replace("d_rmse", "d_rmse_full").replace(
    "significant", "significant_vs_full")))
path = path.join(vs_best.set_axis(path.index)[["d_rmse", "d_rmse_lo", "d_rmse_hi", "significant"]].rename(
    columns=lambda c: c.replace("d_rmse", "d_rmse_best").replace("significant", "significant_vs_best")))

# 4. Proposed set: the smallest set along the path whose pooled RMSE is significantly lower than that of all units
#    (upper end of the interval of ΔRMSE below 0); all units if no step is significantly better.
better = path.index[path["d_rmse_full_hi"] < 0]
proposed_step = int(better.max()) if len(better) else 0
proposed = path.loc[proposed_step, "units"].split(", ")

# 5. Add back each removed unit to the proposed set, one at a time (significant if the interval leaves out 0).
readd_rmse, readd_oofs = readd_check(pipes[MODEL], units, proposed, X, y, splits)
readd = rmse_diff_bootstrap(y, readd_oofs, "chosen", **boot).drop(index="chosen").join(readd_rmse)

# 6. Save the tables and the figures.
save_table(importance.rename_axis("unit"), out_dir / "shap_importance.csv")
save_table(path, out_dir / f"elimination_path_{MODEL}.csv")
save_table(readd.rename_axis("added"), out_dir / f"readd_{MODEL}.csv")
shap_importance_plot(importance, f"{SITE}: out-of-fold SHAP importance (hv_blocks)").savefig(out_dir / "shap_importance.png", dpi=150)
for name, s in shap_units.items():
    shap_beeswarm_plot(s, unit_values, f"{SITE}: {name}, out-of-fold SHAP").savefig(out_dir / f"shap_beeswarm_{name}.png", dpi=150)
elimination_path_plot(path, proposed_step, f"{SITE}: {MODEL}, backward elimination (block bootstrap {cfg.BOOTSTRAP_BLOCK_DAYS} days, "
                      f"{cfg.BOOTSTRAP_LEVEL:.0%})").savefig(out_dir / f"elimination_path_{MODEL}.png", dpi=150)
readd_plot(readd, f"{SITE}: {MODEL}, adding back to the proposed set").savefig(out_dir / f"readd_{MODEL}.png", dpi=150)

# 7. Summary. Then edit FINAL_COLUMNS in config.py with date and reason and run 05 and 08 again.
print(f"{SITE}: {len(units)} units, {len(splits)} hv_blocks folds; elimination for {MODEL} ({time.time() - start:.0f} s)")
print("mean |SHAP| (cm):\n", importance.round(2).to_string())
print("elimination path:\n", path[["removed", "n_units", "rmse_mean", "rmse", "d_rmse_full", "d_rmse_full_lo",
                                   "d_rmse_full_hi", "significant_vs_full", "significant_vs_best"]].round(3).to_string())
print(f"best step: {best} ({path.loc[best, 'n_units']} units); proposed step: {proposed_step} ({len(proposed)} units)")
print("adding back:\n", readd[["rmse_mean", "rmse", "d_rmse", "d_rmse_lo", "d_rmse_hi", "significant"]].round(3).to_string())
better = readd.index[readd["d_rmse_hi"] < 0].tolist()
print("units that improve the proposed set significantly when added back:", better or "none")
print("proposed FINAL_COLUMNS:", [c for u in proposed for c in units[u]])
