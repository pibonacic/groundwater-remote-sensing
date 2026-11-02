"""Step 12: evaluate the tuned models on the holdout (the only script that reads it)."""
import ast

import pandas as pd

import config as cfg
from functions.data import save_table
from functions.features import load_development, season_of
from functions.models import MODEL_SPECS, SeasonalClimatology, build_pipeline, enet_params
from functions.plots import bootstrap_ci_plot, oof_joint_plot
from functions.validation import block_bootstrap, oof_report

SITE = "sdh1"

decisions = cfg.DECISIONS[SITE]
out_dir = cfg.OUTPUTS_DIR / SITE / "08_holdout"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the development period (training) and the holdout (evaluation), and the clip limits.
X_dev, y_dev = load_development(cfg.PROCESSED_DIR / f"dev_{SITE}.csv")
X_hold, y_hold = load_development(cfg.PROCESSED_DIR / f"holdout_{SITE}.csv")
limits = cfg.SITES[SITE]["physical_range_cm"]
low, high = limits

# 2. Hyperparameters chosen in 06 (the "chosen" row of each model in tuning.csv); nothing is tuned here.
tuning = pd.read_csv(cfg.OUTPUTS_DIR / SITE / "06_models" / "tuning.csv")
params = {name: ast.literal_eval(tuning[(tuning["model"] == name) & tuning["chosen"]]["params"].iloc[0])
          for name in MODEL_SPECS}

# 3. Fit every tuned model and the climatology on the whole development period and predict the holdout.
enet = enet_params(0, cfg.INNER_SPLITS, cfg.ENET_L1_RATIOS, cfg.ENET_MAX_ITER)
models = {"climatology": SeasonalClimatology()}
models |= {name: build_pipeline(name, enet, decisions["FINAL_COLUMNS"], limits=limits, seed=cfg.SEED, params=params[name])
           for name in MODEL_SPECS}
preds = {"holdout": {name: pd.Series(model.fit(X_dev, y_dev).predict(X_hold), index=y_hold.index)
                     for name, model in models.items()}}

# 4. Metrics overall, by season and by month; skill against the climatology.
strata = {"season": season_of(y_hold.index), "month": pd.Series(y_hold.index.month, index=y_hold.index)}
metrics = oof_report(preds, y_hold, high - low, strata, baseline="climatology")

# 5. Block-bootstrap intervals of RMSE and skill (paired between models).
boot = block_bootstrap(y_hold, preds, "climatology", cfg.BOOTSTRAP_BLOCK_DAYS, cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_LEVEL, cfg.SEED)

# 6. Save the tables and the figures.
save_table(metrics, out_dir / "holdout_metrics.csv")
save_table(boot.set_index(["scheme", "model"]), out_dir / "bootstrap.csv")
save_table(y_hold.to_frame("observed").join(pd.DataFrame(preds["holdout"])), out_dir / "predictions.csv")
save_table(pd.DataFrame([{"model": name, "param": key.removeprefix("model__"), "value": value}
                         for name, p in params.items() for key, value in p.items()]).set_index("model"), out_dir / "params.csv")
oof_joint_plot(y_hold, preds["holdout"], f"{SITE}: holdout predictions (trained on the development period)").savefig(
    out_dir / "predictions.png", dpi=150)
bootstrap_ci_plot(boot, "climatology", f"{SITE}: holdout (block bootstrap, {cfg.BOOTSTRAP_LEVEL:.0%} interval)").savefig(
    out_dir / "bootstrap_ci.png", dpi=150)

# 7. Summary.
print(f"{SITE}: trained on {len(y_dev)} days ({y_dev.index.min().date()} to {y_dev.index.max().date()}), "
      f"holdout {len(y_hold)} days ({y_hold.index.min().date()} to {y_hold.index.max().date()})")
print("parameters:", {k: {p.removeprefix("model__"): v for p, v in b.items()} for k, b in params.items()})
cols = ["model", "group", "n", "rmse", "mae", "bias", "nrmse", "skill_vs_climatology"]
print(metrics[metrics["stratum"] == "all"][cols].round(3).to_string(index=False))
print("skill by month:\n", metrics[metrics["stratum"] == "month"].pivot_table(
    index="model", columns="group", values="skill_vs_climatology", sort=False).round(2).to_string())
print("rmse by month:\n", metrics[metrics["stratum"] == "month"].pivot_table(
    index="model", columns="group", values="rmse", sort=False).round(2).to_string())
print(f"block bootstrap ({boot['n_blocks'].iloc[0]} blocks of {cfg.BOOTSTRAP_BLOCK_DAYS} days; few blocks = rough interval):")
print(boot.round(3).to_string(index=False))
