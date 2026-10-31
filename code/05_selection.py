"""Steps 8-9: ablation by blocks and leave-one-variable-out ablation.

The permutation / Elastic Net selection frequency analysis (stable groups) and the automatic final columns are
commented out for now; FINAL_COLUMNS in config.py is set by hand.
"""
import pandas as pd

import config as cfg
from functions.data import save_table
from functions.features import feature_blocks, load_development
from functions.models import build_pipeline, enet_params
from functions.selection import run_ablation, simplest_within_noise, variable_ablation
# from functions.selection import candidate_columns, enet_selected, finalize_features, group_permutation, stable_groups
from functions.validation import halves_splits, make_split_sets

SITE = "sdh1"

decisions = cfg.DECISIONS[SITE]
out_dir = cfg.OUTPUTS_DIR / SITE / "05_selection"
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Load the development period only, the column groups and the folds. Every result below is
#    reported over all folds (joint) and for the forward and backward folds separately.
X, y = load_development(cfg.PROCESSED_DIR / f"dev_{SITE}.csv")
blocks = feature_blocks(X.columns)   # groups = build_groups(X.columns) is only used by the commented analyses
gap = max(decisions["LEADS"], default=0)
enet = enet_params(gap, cfg.INNER_SPLITS, cfg.ENET_L1_RATIOS, cfg.ENET_MAX_ITER)
split_sets = make_split_sets(X.index, gap, cfg.N_SPLITS, cfg.TEST_SIZE, cfg.MIN_TRAIN)


def pipeline(model="elasticnet", columns=None):
    return build_pipeline(model, enet, columns, seed=cfg.SEED)


# # 2. Step 8: for each group, how often the Elastic Net keeps it and how much the RMSE grows when
# #    its columns are shuffled together. Stable = both, in most folds.
# schemes = ("fwd", "bwd")
# selected = pd.concat({s: enet_selected(pipeline(), X, y, split_sets[s]) for s in schemes})
# permutation = pd.concat({s: group_permutation(pipeline(), X, y, groups, split_sets[s], cfg.N_PERMUTATIONS, cfg.SEED)
#                          for s in schemes})
# stable = stable_groups(selected, permutation, groups, cfg.STABLE_MIN_FRAC)


# 3. Step 9: ablation by blocks (cfg.ABLATION_VERSIONS), same folds for all, against the monthly
#    climatology, with each model of cfg.ABLATION_MODELS. Proposes the version with fewest columns within the noise.
ablation = run_ablation({m: pipeline(m) for m in cfg.ABLATION_MODELS}, X, y, split_sets, blocks, cfg.ABLATION_VERSIONS)
ablation, alerts = simplest_within_noise(
    ablation, [v for v, names in cfg.ABLATION_VERSIONS.items() if "calendar" in names])
proposal = ablation[(ablation["model"] == cfg.ABLATION_MODELS[0]) & (ablation["scheme"] == "joint") & ablation["simplest"]]

# 4. Leave-one-variable-out ablation: delta = RMSE without the variable - RMSE with all the columns, per fold.
#    Schemes: fwd, bwd, joint (fwd + bwd folds) and halves (fold 1 = halves_fwd, fold 2 = halves_bwd).
halves = halves_splits(X.index, gap)
base_splits = {"fwd": split_sets["fwd"], "bwd": split_sets["bwd"], "halves": halves["halves_fwd"] + halves["halves_bwd"]}
by_variable = variable_ablation(lambda model, columns: pipeline(model, columns), cfg.ABLATION_MODELS, X, y, base_splits)

# (kept for reference, commented out)
# kept_blocks = decisions["KEPT_BLOCKS"] or cfg.ABLATION_VERSIONS[proposal["version"].iloc[0]]
# columns = candidate_columns(groups, stable.index[stable["joint_stable"]], blocks, kept_blocks)
# final_columns = finalize_features(pipeline(columns=columns), X[columns], y) if columns else []

# 5. Save the tables that support the decisions. The old outputs of the commented analyses stay in the folder.
save_table(ablation, out_dir / "ablation.csv")
save_table(by_variable.set_index(["model", "scheme", "variable"]), out_dir / "variable_ablation.csv")
# for name, table in {"enet_selected": selected, "group_permutation": permutation, "stable_groups": stable}.items():
#     save_table(table, out_dir / f"{name}.csv")

# 6. Summary. Then edit FINAL_COLUMNS in config.py with date and reason.
print(f"{SITE}: {X.shape[1]} columns, {len(split_sets['fwd'])} forward + {len(split_sets['bwd'])} backward folds")
print("ablation by blocks (simplest = fewest columns within noise):\n", ablation.set_index(["model", "scheme", "version"])[
    ["n_columns", "rmse_mean", "rmse_std", "skill", "simplest"]].round(3).to_string())
print(f"proposed version ({cfg.ABLATION_MODELS[0]}, joint): {proposal['version'].iloc[0]}")
for alert in alerts:
    print("ALERT:", alert)
print("ablation by variable (delta in cm; passes_rule = mean > std):\n", by_variable.set_index(["model", "scheme", "variable"])[
    ["delta_mean", "delta_std", "positive_frac", "passes_rule"]].round(2).to_string())
