"""Steps 5-6: hv_blocks folds and the reference model (monthly climatology). Metrics are computed in 05."""
import config as cfg
from functions.data import save_table
from functions.features import load_development
from functions.models import SeasonalClimatology
from functions.plots import oof_predictions_plot, splits_plot
from functions.validation import compare, hv_block_splits, hv_blocks_table

SITE = "sdh2"
hv_v, hv_h = cfg.SITES[SITE]["hv_v"], cfg.SITES[SITE]["hv_h"]   # hv_blocks folds of this site

out_dir = cfg.OUTPUTS_DIR / SITE / "04_baselines"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the development period only (the holdout is never loaded here).
X, y = load_development(cfg.PROCESSED_DIR / f"dev_{SITE}.csv")

# 2. hv_blocks folds: consecutive validation blocks of hv_v days from the first day (the last one absorbs the
#    remainder), hv_h days dropped from the training set on each side of the block.
splits = hv_block_splits(X.index, hv_v, hv_h)
blocks = hv_blocks_table(X.index, splits)

# 3. Reference model: mean depth per calendar month of the training days. Any later model has to beat it.
_, oof = compare({"climatology": SeasonalClimatology()}, X, y, splits, "climatology")

# 4. Save the folds and the out-of-fold predictions of the climatology, with their figures.
save_table(blocks, out_dir / "splits.csv")
save_table(oof, out_dir / "oof_predictions.csv")
splits_plot(blocks, y, f"{SITE}: hv_blocks folds (v = {hv_v}, h = {hv_h} days)").savefig(out_dir / "splits.png", dpi=150)
oof_predictions_plot(oof, f"{SITE}: hv_blocks out-of-fold predictions of the climatology").savefig(
    out_dir / "oof_predictions.png", dpi=150)

# 5. Summary.
print(f"{SITE}: {len(X)} development rows, {len(splits)} hv_blocks folds (v={hv_v}, h={hv_h})")
print(blocks.to_string())
