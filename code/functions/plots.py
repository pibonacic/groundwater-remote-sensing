"""One function per figure. Each returns a matplotlib Figure and saves nothing."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram

from functions.features import season_of

SEASON_COLORS = {"DJF": "tab:blue", "MAM": "tab:orange", "JJA": "tab:gray", "SON": "tab:green"}


def quality_overview_plot(df, gw_col, band, title):
    """Water level, one NBAR band (masked days marked) and 30-day rain."""
    fig = Figure(figsize=(11, 7), layout="constrained")
    ax_gw, ax_band, ax_rain = fig.subplots(3, 1, sharex=True)
    ax_gw.plot(df[gw_col], lw=0.8)
    ax_gw.set_ylabel("water level (cm)")
    ax_band.plot(df[band], lw=0.5)
    masked = df.index[df["qa_low"] == 1]
    ax_band.scatter(masked, [df[band].min()] * len(masked), s=2, c="tab:red", label="qa_low day")
    ax_band.set_ylabel(f"NBAR {band}")
    ax_band.legend(loc="upper right")
    ax_rain.plot(df["precip_mm"].rolling(30, min_periods=30).sum(), lw=0.8)
    ax_rain.set_ylabel("CHIRPS 30-day (mm)")
    fig.suptitle(title)
    return fig


def scatter_by_season(X, y, title):
    """Each column of X against y, colored by season (DJF = austral summer)."""
    fig = Figure(figsize=(3.2 * len(X.columns), 3.4), layout="constrained")
    axes = fig.subplots(1, len(X.columns), squeeze=False)[0]
    season = season_of(y.index)
    for ax, col in zip(axes, X):
        for name, color in SEASON_COLORS.items():
            m = season == name
            ax.scatter(X[col][m], y[m], s=6, c=color, label=name)
        ax.set_xlabel(col)
    axes[0].set_ylabel(y.name)
    axes[-1].legend(markerscale=2)
    fig.suptitle(title)
    return fig


def zscore_series_plot(X, y, title):
    """One panel per column of X: z-scored y (left axis) and z-scored column (right axis) over time.
    Both axes of a panel share the same range."""
    zscore = lambda s: (s - s.mean()) / s.std()
    fig = Figure(figsize=(11, 2.2 * len(X.columns)), layout="constrained")
    axes = fig.subplots(len(X.columns), 1, sharex=True, squeeze=False)[:, 0]
    zy = zscore(y)
    for ax, col in zip(axes, X):
        zx = zscore(X[col])
        ax.plot(zy, lw=0.8, c="tab:blue")
        ax.set_ylabel(f"{y.name} (z)", color="tab:blue")
        right = ax.twinx()
        right.plot(zx, lw=0.6, c="tab:orange")
        right.set_ylabel(f"{col} (z)", color="tab:orange")
        ax.axhline(0, c="k", lw=0.4)
        low, high = min(zy.min(), zx.min()), max(zy.max(), zx.max())
        pad = 0.05 * (high - low)
        ax.set_ylim(low - pad, high + pad)
        right.set_ylim(low - pad, high + pad)
    axes[-1].set_xlim(y.index.min(), y.index.max())
    fig.suptitle(title)
    return fig


def ccf_plot(ccf_raw, ccf_anom, title):
    """CCF per index, raw and monthly anomalies; k = 0 marked."""
    fig = Figure(figsize=(11, 4), layout="constrained")
    for ax, ccf, name in zip(fig.subplots(1, 2, sharey=True), (ccf_raw, ccf_anom), ("raw", "monthly anomalies")):
        for col in ccf:
            ax.plot(ccf.index, ccf[col], label=col)
        ax.axvline(0, c="k", lw=0.5)
        ax.axhline(0, c="k", lw=0.5)
        ax.set_title(name)
        ax.set_xlabel("k (days; k > 0: spectral value after the water level)")
    ax.legend()
    fig.suptitle(title)
    return fig


def rain_window_plot(table, title):
    """Spearman rho against accumulation window."""
    fig = Figure(figsize=(5, 3.5), layout="constrained")
    ax = fig.subplots()
    ax.plot(table.index, table["rho"], marker="o")
    ax.set_xlabel("window (days)")
    ax.set_ylabel("Spearman rho")
    ax.set_title(title)
    return fig


def dendrogram_plot(linkage_matrix, labels, kept, threshold, title):
    """Clusters of bands and indices (distance 1 - |Spearman|); the cut at 1 - threshold is dashed and the kept column of each cluster is marked *."""
    fig = Figure(figsize=(7, 0.3 * len(labels) + 1.5), layout="constrained")
    ax = fig.subplots()
    dendrogram(linkage_matrix, labels=[f"{c} *" if c in kept else c for c in labels], orientation="right",
               color_threshold=1 - threshold, ax=ax)
    ax.axvline(1 - threshold, c="k", ls="--", lw=0.8)
    ax.set_xlabel("distance (1 - |Spearman|)")
    ax.set_title(title)
    return fig


SCHEME_COLORS = {"halves_fwd": "#eb6834", "halves_bwd": "#1baf7a", "holdout": "#2a78d6"}


def bootstrap_ci_plot(boot, baseline, title):
    """Pooled out-of-fold RMSE and skill of each model with its bootstrap interval, one color per scheme."""
    order = boot[boot["scheme"] == boot["scheme"].iloc[0]].sort_values("skill", ascending=False)["model"].tolist()
    schemes = list(boot["scheme"].unique())
    fig = Figure(figsize=(11, 0.55 * len(order) + 1.8), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (value, label) in zip(axes, [("rmse", "RMSE (cm)"), ("skill", f"skill vs {baseline}")]):
        for k, scheme in enumerate(schemes):
            part = boot[boot["scheme"] == scheme].set_index("model").reindex(order)
            if value == "skill":
                part = part.drop(index=baseline, errors="ignore")
            y = [order.index(m) + (k - (len(schemes) - 1) / 2) * 0.25 for m in part.index]
            ax.errorbar(part[value], y, xerr=[part[value] - part[f"{value}_lo"], part[f"{value}_hi"] - part[value]],
                        fmt="o", ms=6, lw=1.5, color=SCHEME_COLORS.get(scheme), label=scheme, mec="white", mew=1)
        ax.set_xlabel(label)
        ax.grid(axis="x", lw=0.4, alpha=0.4)
    axes[1].axvline(0, c="k", lw=0.8)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    axes[0].legend(title="scheme", loc="best")
    fig.suptitle(title)
    return fig


def _model_colors(models):
    return {model: f"C{i}" for i, model in enumerate(models)}


def oof_joint_plot(y, oof, title):
    """Observed water level and the joint out-of-fold predictions, one panel per model."""
    colors = _model_colors(oof)
    fig = Figure(figsize=(11, 2.4 * len(oof)), layout="constrained")
    axes = fig.subplots(len(oof), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (model, pred) in zip(axes, oof.items()):
        ax.plot(y, c="k", lw=1, label="observed")
        ax.plot(pred, c=colors[model], lw=1, label=model)
        ax.set_ylabel("water level (cm)")
        ax.legend(loc="upper right")
    fig.suptitle(title)
    return fig


def folds_boxplot(by_fold, title):
    """Spread across folds of RMSE and skill, one box per model (the folds drawn as points)."""
    models = list(by_fold["model"].unique())
    fig = Figure(figsize=(4 + 1.2 * len(models), 4), layout="constrained")
    for ax, metric, label in zip(fig.subplots(1, 2), ("rmse", "skill"), ("RMSE (cm)", "skill vs climatology")):
        values = [by_fold[(by_fold["model"] == m) & (by_fold["metric"] == metric)]["value"].to_numpy() for m in models]
        ax.boxplot(values, tick_labels=models, widths=0.5, medianprops={"color": "k"}, showfliers=False)
        for i, (model, v) in enumerate(zip(models, values), start=1):
            ax.scatter(np.full(len(v), i), v, s=12, c=_model_colors(models)[model], zorder=3)
        if metric == "skill":
            ax.axhline(0, c="k", lw=0.8)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    return fig


def folds_bars(by_fold, title):
    """RMSE and skill of each fold (grouped by direction and fold), one color per model."""
    models = list(by_fold["model"].unique())
    folds = by_fold[["direction", "fold"]].drop_duplicates().apply(lambda r: f"{r['direction']}{r['fold']}", axis=1).tolist()
    colors, width = _model_colors(models), 0.8 / len(models)
    fig = Figure(figsize=(11, 6), layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    for ax, metric, label in zip(axes, ("rmse", "skill"), ("RMSE (cm)", "skill vs climatology")):
        for k, model in enumerate(models):
            v = by_fold[(by_fold["model"] == model) & (by_fold["metric"] == metric)]["value"].to_numpy()
            ax.bar(np.arange(len(folds)) + (k - (len(models) - 1) / 2) * width, v, width, color=colors[model], label=model)
        ax.axhline(0, c="k", lw=0.8)
        ax.set_ylabel(label)
    axes[-1].set_xticks(range(len(folds)), folds)
    axes[0].legend(ncols=len(models), loc="upper left")
    fig.suptitle(title)
    return fig


ROLE_COLORS = {"train_initial": "#c9d6e8", "gap_1": "#e34948"}
FOLD_COLORS = ["#1baf7a", "#eda100", "#4a3aa7", "#e87ac0", "#2a78d6"]


def splits_plot(roles, blocks, y, title):
    """Role of each development day in the forward and backward folds over the water level.

    The gaps of folds 2.. (inside the previous test block) are hatched.
    """
    fig = Figure(figsize=(11, 5), layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    day = pd.Timedelta(days=1)
    for ax, direction in zip(axes, ("fwd", "bwd")):
        role = roles[direction]
        runs = (role != role.shift()).cumsum()
        for _, run in role.groupby(runs):
            name, start, end = run.iloc[0], run.index.min(), run.index.max() + day
            color = ROLE_COLORS.get(name) or FOLD_COLORS[(int(name.split("_")[1]) - 1) % len(FOLD_COLORS)]
            ax.axvspan(start, end, color=color, alpha=0.5, lw=0)
            label = name.replace("fold_", "fold ").replace("train_initial", "initial train").replace("gap_1", "gap")
            ax.text(start + (end - start) / 2, 1.02, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        for _, b in blocks[(blocks["direction"] == direction) & (blocks["fold"] > 1)].iterrows():
            ax.axvspan(b["gap_start"], b["gap_end"] + day, fill=False, hatch="///", lw=0, ec="k", alpha=0.6)
        ax.plot(y, c="k", lw=0.8)
        ax.set_ylabel(f"{direction}: water level (cm)")
    axes[-1].set_xlim(roles.index.min(), roles.index.max() + day)
    fig.suptitle(title)
    return fig


def halves_plot(y, oof, cut, title):
    """Observed water level and the predictions of halves_fwd and halves_bwd, one panel per model, with the cut."""
    models = list(oof["halves_fwd"])
    fig = Figure(figsize=(11, 2.4 * len(models)), layout="constrained")
    axes = fig.subplots(len(models), 1, sharex=True, squeeze=False)[:, 0]
    for ax, model in zip(axes, models):
        ax.plot(y, c="k", lw=1, label="observed")
        for scheme in ("halves_fwd", "halves_bwd"):
            ax.plot(oof[scheme][model], c=SCHEME_COLORS[scheme], lw=1, label=scheme)
        ax.axvline(cut, c="k", ls="--", lw=0.8)
        ax.set_ylabel(f"{model}\nwater level (cm)")
    axes[0].legend(ncols=3, loc="upper right")
    fig.suptitle(title)
    return fig
