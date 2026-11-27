"""Step 8: predict the water level over the whole satellite series with the chosen model refit on every observed day
(development + holdout), with prediction intervals (functions/prediction.py)."""
import ast
import time

import numpy as np
import pandas as pd

import config as cfg
from functions.data import load_table, save_table
from functions.features import feature_table, load_development
from functions.models import CLIP_PREFIX, MODEL_SPECS, build_pipeline
from functions.plots import (annual_cycle_plot, coverage_plot, member_examples_plot, prediction_series_plot,
                             year_doy_heatmap)
from functions.prediction import (annual_summary, bootstrap_ensemble, coverage_table, daily_ranges, draw_residuals,
                                  oof_residuals, outside_training_range, predictive_quantiles, residual_ranges,
                                  simulate)
from functions.evaluation import hv_block_splits

SITE = "sdh1"

t0 = time.time()
site = cfg.SITES[SITE]
decisions = cfg.decisions(SITE)
model_name, columns = decisions["MODEL"], decisions["FINAL_COLUMNS"]
if model_name not in MODEL_SPECS:
    raise SystemExit(f"Set MODEL for {SITE} in config.py DECISIONS to one of {list(MODEL_SPECS)} (got {model_name}).")
out_dir = cfg.out_dir(SITE, "08_predictions")
limits = site["physical_range_cm"]
levels = cfg.PREDICTION_LEVELS

# 1. Features over the whole series (the same table as 03, from the clean daily table of step 1); the days to
#    predict are those with every final column.
daily = load_table(cfg.INTERIM_DIR / f"{SITE}_daily.csv")
X_all = feature_table(daily, decisions, cfg.BANDS)
X_all = X_all[X_all[columns].notna().all(axis=1)]

# 2. Training data: development + holdout of the scheme PARTITION (every observed day).
X_dev, y_dev = load_development(cfg.processed_path("development", SITE))
X_hold, y_hold = load_development(cfg.processed_path("holdout", SITE))
X, y = pd.concat([X_dev, X_hold]), pd.concat([y_dev, y_hold])

# 3. The chosen model with the parameters chosen in 05 (MODEL_PARAMS, if set, take precedence); nothing is tuned.
if decisions["MODEL_PARAMS"] is not None:
    params = {CLIP_PREFIX + k: v for k, v in decisions["MODEL_PARAMS"].items()}
else:
    tuning = pd.read_csv(cfg.out_dir(SITE, "05_models") / "tuning.csv")
    params = ast.literal_eval(tuning[(tuning["model"] == model_name) & tuning["chosen"]]["params"].iloc[0])
pipeline = build_pipeline(model_name, cfg, columns, limits=limits, seed=cfg.SEED, params=params)


def predict_with_intervals(X_train, y_train, X_new):
    """Point prediction of the model refit on (X_train, y_train), the members of the block-bootstrap ensemble, the
    out-of-fold hv_blocks residuals, the residual drawn for each day and member, the simulated members (member -
    residual) and the predictive quantiles on X_new."""
    pred = pd.Series(pipeline.fit(X_train, y_train).predict(X_new), index=X_new.index, name="pred")
    members = bootstrap_ensemble(pipeline, X_train, y_train, X_new, cfg.N_ENSEMBLE, cfg.BOOTSTRAP_BLOCK_DAYS, cfg.SEED)
    members.columns = [f"member_{b}" for b in range(members.shape[1])]
    residuals = oof_residuals(pipeline, X_train, y_train, hv_block_splits(X_train.index, site["hv_v"], site["hv_h"]))
    draws = draw_residuals(members, residuals, cfg.SEED)
    sim = simulate(members, draws, limits)
    return pred, members, residuals, draws, sim, predictive_quantiles(sim, levels)


# 4. Validation of the intervals: the same procedure trained on the development period only, scored on the holdout
#    (coverage, width and interval score; overall and by whether the inputs fall outside the training range).
val_pred, _, _, _, _, val_q = predict_with_intervals(X_dev, y_dev, X_hold)
val_outside = outside_training_range(X_dev, X_hold, columns)
coverage = coverage_table(y_hold, val_q, levels, {"all": y_hold.notna(), "extrapolates": val_outside > 0,
                                                  "inside": val_outside == 0})

# 5. Final predictions: refit on every observed day and predict the whole series.
pred, members, residuals, draws, sim, quantiles = predict_with_intervals(X, y, X_all)
n_outside = outside_training_range(X, X_all, columns)
table = pred.to_frame().join(quantiles).join(n_outside)
table["extrapolates"] = table["n_outside"] > 0
table.insert(0, "observed", y.reindex(table.index))
annual = annual_summary(pred, members, n_outside, cfg.REGIME_THRESHOLD_CM, cfg.BOOTSTRAP_LEVEL)

# 6. Save the tables and the figures.
save_table(table, out_dir / "predictions.csv")
save_table(annual, out_dir / "annual_summary.csv")
save_table(coverage.set_index("stratum"), out_dir / "coverage.csv")
save_table(residuals, out_dir / "oof_residuals.csv")
save_table(residual_ranges(residuals), out_dir / "oof_residual_ranges_by_season.csv")
save_table(members, out_dir / "ensemble_members.csv")
save_table(daily_ranges(members, draws), out_dir / "ensemble_daily_ranges.csv")
save_table(pd.DataFrame([{"item": "model", "value": model_name}, {"item": "scheme", "value": cfg.PARTITION},
                         {"item": "columns", "value": columns}, {"item": "n_ensemble", "value": cfg.N_ENSEMBLE},
                         {"item": "block_days", "value": cfg.BOOTSTRAP_BLOCK_DAYS},
                         {"item": "training", "value": f"{y.index.min().date()} to {y.index.max().date()} ({len(y)} days)"}]
                        + [{"item": k.removeprefix(CLIP_PREFIX), "value": v} for k, v in params.items()]).set_index("item"),
           out_dir / "params.csv")
label = f"{SITE} ({cfg.PARTITION}, {model_name})"
annual_cycle_plot(pred, y, f"{label}: modeled and observed depth by day of the year").savefig(
    out_dir / "annual_cycle.png", dpi=150)
year_doy_heatmap(pred, y.index, f"{label}: modeled depth").savefig(out_dir / "heatmap.png", dpi=150)
cycle = pred.groupby([pred.index.month, pred.index.day]).transform("mean")
year_doy_heatmap(pred - cycle, y.index, f"{label}: modeled depth minus its mean for the day of the year "
                 "(positive = deeper)", label="anomaly (cm)", cmap="RdBu_r", center=0).savefig(
    out_dir / "heatmap_anomaly.png", dpi=150)
prediction_series_plot(table, levels, f"{label}: predictions over the whole series (refit on every observed day)").savefig(
    out_dir / "series.png", dpi=150)
examples = np.random.default_rng(cfg.SEED).choice(members.columns, size=4, replace=False)
member_examples_plot(members[examples], sim[examples], f"{label}: random members of the ensemble, before and after "
                     "subtracting their drawn out-of-fold residual").savefig(out_dir / "member_examples.png", dpi=150)
coverage_plot(y_hold, val_pred, val_q, levels, coverage, f"{label}: interval check, trained on the development "
              "period and predicted on the holdout").savefig(out_dir / "coverage.png", dpi=150)

# 7. Summary.
print(f"{label}: trained on {len(y)} days ({y.index.min().date()} to {y.index.max().date()}), predicted {len(table)} "
      f"days ({table.index.min().date()} to {table.index.max().date()}), columns {columns}")
print("parameters:", {k.removeprefix(CLIP_PREFIX): v for k, v in params.items()})
print("interval check on the holdout (trained on the development period):\n", coverage.round(3).to_string(index=False))
print("annual summary:\n", annual.round(1).to_string())
print(f"done in {time.time() - t0:.0f} s")
