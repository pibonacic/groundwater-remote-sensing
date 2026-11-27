"""Step 7: evaluate the tuned models on the holdout (the only script that reads it)."""
import ast

import pandas as pd

import config as cfg
from functions.data import save_table
from functions.features import load_development, season_of
from functions.models import MODEL_SPECS, SeasonalClimatology, build_pipeline
from functions.plots import bootstrap_ci_plot, oof_predictions_plot, season_metrics_plot
from functions.evaluation import block_bootstrap, metrics_table

SITE = "sdh1"
BASELINE = "climatology"

decisions = cfg.decisions(SITE)
out_dir = cfg.out_dir(SITE, "07_holdout")

# 1. Load the development period (training) and the holdout (evaluation), and the clip limits.
X_dev, y_dev = load_development(cfg.processed_path("development", SITE))
X_hold, y_hold = load_development(cfg.processed_path("holdout", SITE))
limits = cfg.SITES[SITE]["physical_range_cm"]

# 2. Hyperparameters chosen in 05 (the "chosen" row of each model in tuning.csv); nothing is tuned here.
tuning = pd.read_csv(cfg.out_dir(SITE, "05_models") / "tuning.csv")
params = {name: ast.literal_eval(tuning[(tuning["model"] == name) & tuning["chosen"]]["params"].iloc[0])
          for name in MODEL_SPECS}

# 3. Fit every tuned model and the climatology on the whole development period and predict the holdout.
models = {BASELINE: SeasonalClimatology()}
models |= {name: build_pipeline(name, cfg, decisions["FINAL_COLUMNS"], limits=limits, seed=cfg.SEED, params=params[name])
           for name in MODEL_SPECS}
preds = pd.DataFrame({name: model.fit(X_dev, y_dev).predict(X_hold) for name, model in models.items()}, index=y_hold.index)

# 4. Metrics overall and by season; skill against the climatology.
metrics = metrics_table(y_hold, preds, BASELINE, {"season": season_of(y_hold.index)})

# 5. Block-bootstrap intervals of RMSE, ME and skill (paired between models).
boot = block_bootstrap(y_hold, preds, BASELINE, cfg.BOOTSTRAP_BLOCK_DAYS_HOLDOUT, cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_LEVEL, cfg.SEED)

# 6. Save the tables and the figures.
observed = y_hold.rename("observed").to_frame().join(preds)
save_table(metrics.set_index("model"), out_dir / "holdout_metrics_by_season.csv")
save_table(boot, out_dir / "holdout_bootstrap_metrics.csv")
save_table(observed, out_dir / "predictions.csv")
save_table(pd.DataFrame([{"model": name, "param": key.removeprefix("model__regressor__"), "value": value}
                         for name, p in params.items() for key, value in p.items()]).set_index("model"), out_dir / "params.csv")
oof_predictions_plot(observed, f"{SITE}: holdout predictions (trained on the development period)").savefig(
    out_dir / "predictions.png", dpi=150)
season_metrics_plot(metrics, BASELINE, f"{SITE}: holdout metrics by season").savefig(
    out_dir / "holdout_metrics_by_season.png", dpi=150)
bootstrap_ci_plot(boot, BASELINE, f"{SITE}: holdout (block bootstrap, {cfg.BOOTSTRAP_BLOCK_DAYS_HOLDOUT}-day blocks, "
                  f"{cfg.BOOTSTRAP_LEVEL:.0%} interval)").savefig(out_dir / "holdout_bootstrap_metrics.png", dpi=150)

# 7. Summary.
print(f"{SITE}: trained on {len(y_dev)} days ({y_dev.index.min().date()} to {y_dev.index.max().date()}), "
      f"holdout {len(y_hold)} days ({y_hold.index.min().date()} to {y_hold.index.max().date()})")
print("parameters:", {k: {p.removeprefix("model__regressor__"): v for p, v in b.items()} for k, b in params.items()})
print(metrics[metrics["stratum"] == "all"].drop(columns=["stratum", "group"]).round(3).to_string(index=False))
print("skill by season:\n", metrics.pivot_table(index="model", columns="group", values="skill", sort=False).round(2).to_string())
print(f"block bootstrap ({boot['n_blocks'].iloc[0]} blocks of {cfg.BOOTSTRAP_BLOCK_DAYS_HOLDOUT} days; few blocks = rough interval):")
print(boot.round(3).to_string())
