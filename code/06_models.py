"""Step 10: tune and compare the candidate models on the final columns."""
import pandas as pd

import config as cfg
from functions.data import save_table
from functions.features import load_development, season_of
from functions.models import (
    MODEL_SPECS, SeasonalClimatology, build_pipeline, enet_params, make_linear_baseline, pipeline_grid, tune)
from functions.plots import oof_plot
from functions.validation import compare, halves_splits, make_split_sets, oof_report

SITE = "sdh1"

decisions = cfg.DECISIONS[SITE]
if decisions["FINAL_COLUMNS"] is None:
    raise SystemExit(f"Set FINAL_COLUMNS for {SITE} in config.py DECISIONS first (output of 05_selection.py).")
out_dir = cfg.OUTPUTS_DIR / SITE / "06_models"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the development period only, the folds and the clip limits (the physical range of the level).
X, y = load_development(cfg.PROCESSED_DIR / f"dev_{SITE}.csv")
gap = max(decisions["LEADS"], default=0)
enet = enet_params(gap, cfg.INNER_SPLITS, cfg.ENET_L1_RATIOS, cfg.ENET_MAX_ITER)
split_sets = make_split_sets(X.index, gap, cfg.N_SPLITS, cfg.TEST_SIZE, cfg.MIN_TRAIN)
limits = cfg.SITES[SITE]["physical_range_cm"]
low, high = limits

# 2. Tune every candidate on the joint folds (forward + backward); tuning.csv also has each scheme's RMSE.
models = {"climatology": SeasonalClimatology(),
          "linear_all": make_linear_baseline(gap, cfg.INNER_SPLITS, cfg.ENET_L1_RATIOS, cfg.ENET_MAX_ITER)}
tuning, best_params = [], {}
for name in MODEL_SPECS:
    pipe = build_pipeline(name, enet, decisions["FINAL_COLUMNS"], limits=limits, seed=cfg.SEED)
    best_params[name], summary = tune(pipe, pipeline_grid(name), X, y, split_sets)
    models[name] = pipe.set_params(**best_params[name])
    tuning.append(summary.assign(model=name))

# 3. Compare all models with the baselines on the same folds, plus the halves schemes (diagnostic only,
#    not used to tune); skill against the monthly climatology.
split_sets.update(halves_splits(X.index, gap))
table, oof = compare(models, X, y, split_sets, baseline="climatology")
by_season = oof_report(oof, y, high - low, {"season": season_of(y.index)}, baseline="climatology")

# 4. Save the tables, the out-of-fold predictions and the figure.
save_table(pd.concat(tuning, ignore_index=True), out_dir / "tuning.csv")
save_table(table.set_index(["scheme", "model"]), out_dir / "models.csv")
save_table(by_season, out_dir / "models_by_season.csv")
oof = {scheme: by_model for scheme, by_model in oof.items() if scheme != "joint"}   # joint = fwd + bwd, nothing new to draw
for scheme, by_model in oof.items():
    save_table(pd.DataFrame(by_model), out_dir / f"oof_{scheme}.csv")
oof_plot(y, oof, f"{SITE}: out-of-fold predictions").savefig(out_dir / "oof.png", dpi=150)

# 5. Summary. Then edit MODEL and MODEL_PARAMS in config.py with date and reason.
print(f"{SITE}: {len(X)} development rows, columns {decisions['FINAL_COLUMNS']}, gap={gap} days")
print("best parameters:", {k: {p.removeprefix('model__regressor__'): v for p, v in b.items()} for k, b in best_params.items()})
print(table.round(3).to_string(index=False))
print(by_season[by_season["stratum"] == "season"][["scheme", "model", "group", "n", "rmse", "skill_vs_climatology"]]
      .round(2).pivot_table(index=["scheme", "model"], columns="group", values="skill_vs_climatology").to_string())
