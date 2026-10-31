"""Variable selection: group permutation, Elastic Net selection and block ablation."""
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from functions.models import SeasonalClimatology
from functions.validation import compare, evaluate


def _model_columns(fitted, X):
    """Columns that reach the model of a fitted pipeline."""
    return list(fitted[:-1].transform(X.iloc[:1]).columns)


def _nonzero(fitted, X):
    """Boolean Series over X's columns: True where the Elastic Net coefficient is not zero."""
    coef = fitted.named_steps["model"].regressor_.coef_
    return pd.Series(coef != 0, index=_model_columns(fitted, X)).reindex(X.columns, fill_value=False)


def enet_selected(pipeline, X, y, splits):
    """Folds x columns table: does the column have a non-zero Elastic Net coefficient."""
    inspected = evaluate(pipeline, X, y, splits, inspect=lambda fitted, X_te, y_te: _nonzero(fitted, X_te))["inspected"]
    return pd.DataFrame(inspected)


def group_permutation(pipeline, X, y, groups, splits, n_repeats, seed):
    """Folds x groups table: RMSE increase when the columns of a group are shuffled together in the test rows."""
    def inspect(fitted, X_te, y_te):
        rng = np.random.default_rng(seed)
        base = root_mean_squared_error(y_te, fitted.predict(X_te))
        increase = {}
        for name, cols in groups.items():
            gains = []
            for _ in range(n_repeats):
                shuffled = X_te.copy()
                shuffled[cols] = X_te[cols].iloc[rng.permutation(len(X_te))].to_numpy()
                gains.append(root_mean_squared_error(y_te, fitted.predict(shuffled)) - base)
            increase[name] = np.mean(gains)
        return pd.Series(increase)
    return pd.DataFrame(evaluate(pipeline, X, y, splits, inspect=inspect)["inspected"])


def stable_groups(selected, permutation, groups, min_frac):
    """Stability of each group over all folds ("joint") and within each scheme.

    selected and permutation have the scheme as the first level of the row index. For each of
    joint, fwd and bwd the table has the share of folds in which the Elastic Net keeps the group
    (enet_frac), the mean RMSE increase when it is shuffled (cm) and the share of folds where
    that increase is positive. <name>_stable = enet_frac >= min_frac and a positive increase in
    most folds; stable_both = stable in every scheme.
    """
    schemes = list(selected.index.get_level_values(0).unique())
    table = pd.DataFrame(index=list(groups))
    for name in ["joint"] + schemes:
        sel, perm = (selected, permutation) if name == "joint" else (selected.loc[name], permutation.loc[name])
        enet_frac = pd.Series({g: sel[cols].any(axis=1).mean() for g, cols in groups.items()})
        positive = (perm > 0).mean()
        table[f"{name}_enet_frac"] = enet_frac
        table[f"{name}_perm_gain_mean"] = perm.mean()
        table[f"{name}_perm_positive_frac"] = positive
        table[f"{name}_stable"] = (enet_frac >= min_frac) & (positive > 0.5)
    table["stable_both"] = table[[f"{s}_stable" for s in schemes]].all(axis=1)
    return table


def candidate_columns(groups, kept_groups, blocks, kept_blocks):
    """Columns of the kept groups that also belong to one of the kept blocks."""
    allowed = {c for b in kept_blocks for c in blocks[b]}
    return [c for g in kept_groups for c in groups[g] if c in allowed]


def finalize_features(pipeline, X, y):
    """Fit the pipeline once on all the development data and return the columns the Elastic Net uses."""
    fitted = pipeline.fit(X, y)
    return list(X.columns[_nonzero(fitted, X)])


def run_ablation(pipelines, X, y, split_sets, blocks, versions):
    """Evaluate each pipeline on the nested column sets of versions (block names) with the same folds.

    Returns one row per model x version x scheme (rmse mean/std across folds, skill vs monthly climatology).
    """
    frames = []
    for model_name, pipeline in pipelines.items():
        for version, names in versions.items():
            columns = [c for b in names for c in blocks[b]]
            if not columns:
                continue
            table, _ = compare({"climatology": SeasonalClimatology(), "model": pipeline}, X[columns], y,
                               split_sets, baseline="climatology")
            frames.append(table[table["model"] == "model"].assign(
                model=model_name, version=version, n_columns=len(columns)))
    return pd.concat(frames, ignore_index=True)


def simplest_within_noise(results, calendar_versions):
    """For each model and scheme, the version with fewest columns whose mean RMSE is within one fold std of the best.

    Returns (table, alerts): the results with `within_noise` and `simplest` flags, and a message for
    each model, scheme and version of calendar_versions that does not beat the monthly climatology.
    """
    table = results.copy()
    keys = ["model", "scheme"]
    best = table.loc[table.groupby(keys)["rmse_mean"].idxmin()].set_index(keys)
    limit = (best["rmse_mean"] + best["rmse_std"]).reindex(pd.MultiIndex.from_frame(table[keys])).to_numpy()
    table["within_noise"] = table["rmse_mean"].to_numpy() <= limit
    fewest = table[table["within_noise"]].groupby(keys)["n_columns"].idxmin()
    table["simplest"] = table.index.isin(fewest)
    alerts = [f"{r.model} / {r.scheme}: {r.version} (with calendar) does not beat the monthly climatology (skill {r.skill:.2f})"
              for r in table.itertuples() if r.version in calendar_versions and r.skill <= 0]
    return table, alerts


def variable_deltas(make_pipeline, X, y, splits):
    """Variables x folds table: RMSE of the fold without the variable minus RMSE with all the columns."""
    columns = list(X.columns)
    full = evaluate(make_pipeline(columns), X, y, splits)["folds"]["rmse"].to_numpy()
    return pd.DataFrame({c: evaluate(make_pipeline([k for k in columns if k != c]), X, y, splits)["folds"]["rmse"].to_numpy() - full
                         for c in columns}).T


def variable_ablation(make_pipeline, model_names, X, y, base_splits, joint_of=("fwd", "bwd")):
    """Leave-one-variable-out ablation for each model and scheme (the base schemes plus "joint").

    make_pipeline(model_name, columns) builds a fresh pipeline. "joint" stacks the folds of the schemes in
    joint_of (fold 1..n in that order), so it costs no extra fits. One row per model x scheme x variable with
    the delta of each fold (fold_1, fold_2, ...), their mean, std across folds and share of positive deltas;
    passes_rule = mean delta > std (the variable helps by more than the spread between folds).
    """
    frames = []
    for model in model_names:
        deltas = {scheme: variable_deltas(lambda cols: make_pipeline(model, cols), X, y, splits)
                  for scheme, splits in base_splits.items()}
        deltas["joint"] = pd.concat([deltas[s] for s in joint_of], axis=1, ignore_index=True)
        for scheme, d in deltas.items():
            d.columns = [f"fold_{i + 1}" for i in range(d.shape[1])]
            stats = pd.DataFrame({"delta_mean": d.mean(axis=1), "delta_std": d.std(axis=1, ddof=1),
                                  "positive_frac": (d > 0).mean(axis=1)})
            stats["passes_rule"] = stats["delta_mean"] > stats["delta_std"]
            frames.append(pd.concat([d, stats], axis=1).rename_axis("variable").reset_index().assign(model=model, scheme=scheme))
    table = pd.concat(frames, ignore_index=True)
    fold_cols = sorted((c for c in table if c.startswith("fold_")), key=lambda c: int(c[5:]))
    return table[["model", "scheme", "variable"] + fold_cols + ["delta_mean", "delta_std", "positive_frac", "passes_rule"]]
