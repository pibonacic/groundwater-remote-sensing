"""One function per figure. Each returns a matplotlib Figure and saves nothing."""
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

from functions.features import season_of
from functions.validation import METRIC_LABELS, METRIC_REFERENCE, METRICS

SEASON_COLORS = {"DJF": "tab:blue", "MAM": "tab:orange", "JJA": "tab:gray", "SON": "tab:green"}
DEPTH_LABEL = "depth to water table (cm)"


def _depth_axis(ax, label=DEPTH_LABEL, **kwargs):
    """Depth is positive below the surface: invert the axis so deeper is down."""
    ax.set_ylabel(label, **kwargs)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


def quality_overview_plot(df, gw_col, band, title):
    """Water level, one NBAR band (masked days marked) and 30-day rain."""
    fig = Figure(figsize=(11, 7), layout="constrained")
    ax_gw, ax_band, ax_rain = fig.subplots(3, 1, sharex=True)
    ax_gw.plot(df[gw_col], lw=0.8)
    _depth_axis(ax_gw)
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
    _depth_axis(axes[0])
    axes[-1].legend(markerscale=2)
    fig.suptitle(title)
    return fig


def zscore_series_plot(X, y, title, suffix=" (z)"):
    """One panel per column of X: z-scored y (left axis, inverted: y is a depth) and z-scored column (right axis)
    over time. Both axes of a panel share the same range; suffix goes after the names in the axis labels."""
    zscore = lambda s: (s - s.mean()) / s.std()
    fig = Figure(figsize=(11, 2.2 * len(X.columns)), layout="constrained")
    axes = fig.subplots(len(X.columns), 1, sharex=True, squeeze=False)[:, 0]
    zy = zscore(y)
    for ax, col in zip(axes, X):
        zx = zscore(X[col])
        ax.plot(zy, lw=0.8, c="tab:blue")
        ax.set_ylabel(f"{y.name}{suffix}, deeper down", color="tab:blue")
        right = ax.twinx()
        right.plot(zx, lw=0.6, c="tab:orange")
        right.set_ylabel(f"{col}{suffix}", color="tab:orange")
        ax.axhline(0, c="k", lw=0.4)
        low, high = min(zy.min(), zx.min()), max(zy.max(), zx.max())
        pad = 0.05 * (high - low)
        ax.set_ylim(high + pad, low - pad)
        right.set_ylim(low - pad, high + pad)
    axes[-1].set_xlim(y.index.min(), y.index.max())
    fig.suptitle(title)
    return fig


def ccf_plot(ccf_raw, ccf_anom, title):
    """CCF per index, raw and seasonal anomalies; k = 0 marked."""
    fig = Figure(figsize=(11, 4), layout="constrained")
    for ax, ccf, name in zip(fig.subplots(1, 2, sharey=True), (ccf_raw, ccf_anom), ("raw", "seasonal anomalies")):
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


def acf_pacf_plot(memory, h, title):
    """ACF and PACF of the target (columns) for each series in memory (rows: raw, anomaly), with the 95% band of no
    correlation and the gap h of the hv_blocks folds."""
    fig = Figure(figsize=(11, 3.2 * len(memory)), layout="constrained")
    axes = fig.subplots(len(memory), 2, sharex=True, squeeze=False)
    for row, (kind, table) in zip(axes, memory.items()):
        for ax, col in zip(row, ("acf", "pacf")):
            ax.vlines(table.index, 0, table[col], lw=1)
            ax.fill_between(table.index, -table["conf"], table["conf"], color="0.85", lw=0)
            ax.axhline(0, c="k", lw=0.6)
            ax.axvline(h, c="tab:red", ls="--", lw=0.8, label=f"h = {h}")
            ax.set_title(f"{col.upper()}, {kind}")
            ax.set_ylabel(col.upper())
        row[0].legend(loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("lag (days)")
    fig.suptitle(title)
    return fig


def _model_colors(models):
    return {model: f"C{i}" for i, model in enumerate(models)}


def oof_predictions_plot(preds, title):
    """Observed depth and the predictions of each model, one panel per model (deeper down).

    preds has the column observed and one column per model."""
    models = [c for c in preds if c != "observed"]
    colors = _model_colors(models)
    fig = Figure(figsize=(11, 2.4 * len(models)), layout="constrained")
    axes = fig.subplots(len(models), 1, sharex=True, squeeze=False)[:, 0]
    for ax, model in zip(axes, models):
        ax.plot(preds["observed"], c="k", lw=1, label="observed")
        ax.plot(preds[model], c=colors[model], lw=1, label=model)
        _depth_axis(ax, "depth (cm)")
        ax.legend(loc="lower right")
    fig.suptitle(title)
    return fig


def splits_plot(blocks, y, title):
    """The observed depth (top) and, one row per hv_blocks fold, its training days, gaps and validation block."""
    day = pd.Timedelta(days=1)
    fig = Figure(figsize=(11, 2.5 + 0.3 * len(blocks)), layout="constrained")
    ax_y, ax = fig.subplots(2, 1, sharex=True, height_ratios=[1, max(1.5, 0.12 * len(blocks))])
    ax_y.plot(y, c="k", lw=0.8)
    _depth_axis(ax_y, "depth (cm)")
    first, last = y.index.min(), y.index.max() + day
    for k, b in blocks.iterrows():
        start, end = b["test_start"], b["test_end"] + day
        ax.barh(k, (last - first).days, left=first, color="#c9d6e8", height=0.8)
        ax.barh(k, b["gap_before"], left=start - b["gap_before"] * day, color="#e34948", height=0.8)
        ax.barh(k, b["gap_after"], left=end, color="#e34948", height=0.8)
        ax.barh(k, (end - start).days, left=start, color="#1baf7a", height=0.8)
    ax.set_yticks(blocks.index, [f"fold {k}" for k in blocks.index])
    ax.invert_yaxis()
    ax.set_xlim(first, last)
    fig.legend(handles=[Patch(color="#c9d6e8", label="train"), Patch(color="#e34948", label="gap h"),
                       Patch(color="#1baf7a", label="validation v")], ncols=3, loc="outside lower center")
    fig.suptitle(title)
    return fig


def by_fold_boxplot(by_fold, baseline, title):
    """Spread across folds of every metric (one panel each), one box per model with the folds drawn as points; the
    baseline is left out of the skill panel (its skill is 0 by definition)."""
    models = list(by_fold.index.get_level_values("model").unique())
    colors = _model_colors(models)
    fig = Figure(figsize=(15, 8), layout="constrained")
    for ax, metric in zip(fig.subplots(2, 4).ravel(), METRICS):
        shown = [m for m in models if not (metric == "skill" and m == baseline)]
        values = [by_fold.loc[m, metric].dropna().to_numpy() for m in shown]
        ax.boxplot(values, tick_labels=shown, widths=0.5, medianprops={"color": "k"}, showfliers=False)
        for i, (model, v) in enumerate(zip(shown, values), start=1):
            ax.scatter(np.full(len(v), i), v, s=12, c=colors[model], zorder=3)
        if metric in METRIC_REFERENCE:
            ax.axhline(METRIC_REFERENCE[metric], c="k", lw=0.8, ls="--")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle(title)
    return fig


def season_metrics_plot(table, baseline, title):
    """Dot plot of every metric (one panel each) by group (all, then the seasons), one point per model.

    table is the output of metrics_table (columns model, group and the metrics); the baseline is left out of the skill panel."""
    models = list(dict.fromkeys(table["model"]))
    groups = list(dict.fromkeys(table["group"]))
    colors = _model_colors(models)
    width = 0.7 / max(len(models) - 1, 1)
    x = np.arange(len(groups))
    fig = Figure(figsize=(15, 8), layout="constrained")
    for ax, metric in zip(fig.subplots(2, 4).ravel(), METRICS):
        for k, model in enumerate(models):
            if metric == "skill" and model == baseline:
                continue
            values = table[table["model"] == model].set_index("group").reindex(groups)[metric]
            ax.plot(x + (k - (len(models) - 1) / 2) * width, values, "o", ms=6, mec="white", color=colors[model])
        if metric in METRIC_REFERENCE:
            ax.axhline(METRIC_REFERENCE[metric], c="k", lw=0.8, ls="--")
        for i in x[:-1]:
            ax.axvline(i + 0.5, c="0.85", lw=0.6)
        ax.set_xticks(x, groups)
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(axis="y", lw=0.4, alpha=0.4)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[m], mec="white", ms=7, label=m) for m in models],
               ncols=len(models), loc="outside lower center")
    fig.suptitle(title)
    return fig


def bootstrap_ci_plot(boot, baseline, title):
    """Pooled RMSE, ME and skill of each model with its bootstrap interval (boot indexed by model)."""
    order = boot.sort_values("skill", ascending=False).index.tolist()
    colors = _model_colors(boot.index)
    fig = Figure(figsize=(13, 0.55 * len(order) + 1.8), layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, value in zip(axes, ("rmse", "me", "skill")):
        part = boot.reindex(order)
        if value == "skill":
            part = part.drop(index=baseline, errors="ignore")
        for model, row in part.iterrows():
            ax.errorbar(row[value], order.index(model), xerr=[[row[value] - row[f"{value}_lo"]], [row[f"{value}_hi"] - row[value]]],
                        fmt="o", ms=7, lw=1.5, color=colors[model], mec="white", mew=1)
        if value in METRIC_REFERENCE:
            ax.axvline(METRIC_REFERENCE[value], c="k", lw=0.8)
        ax.set_xlabel(METRIC_LABELS[value])
        ax.grid(axis="x", lw=0.4, alpha=0.4)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].invert_yaxis()
    fig.suptitle(title)
    return fig


def taylor_plot(metrics, sd_obs, title):
    """Taylor diagram: each model at the radius alpha (standard deviation of the predictions over that of the
    observations) and the angle arccos(r); the star is the observations. The dotted circles around the star are the
    centered RMSE, labeled in cm (sd_obs is the standard deviation of the observations).

    metrics has one row per model with the columns model, r and alpha (pooled predictions).
    """
    models = list(dict.fromkeys(metrics["model"]))
    colors = _model_colors(models)
    span = np.pi if (metrics["r"] < 0).any() else np.pi / 2
    rmax = max(1.25, 1.1 * metrics["alpha"].max())
    corr = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    corr = np.r_[-corr[:0:-1], corr] if span == np.pi else corr
    ticks = np.arange(0, rmax, 0.2)
    fig = Figure(figsize=(6.2, 6.2), layout="constrained")
    phi = np.linspace(0, np.pi, 300)
    step = next(v for v in (1, 2, 2.5, 5, 10, 20, 50) if v >= 1.4 * sd_obs / 5)
    ax = fig.add_subplot(projection="polar")
    ax.set_thetamin(0)
    ax.set_thetamax(np.degrees(span))
    ax.set_rlim(0, rmax)
    ax.set_rticks(ticks)
    ax.set_thetagrids(np.degrees(np.arccos(corr)), [f"{c:g}" for c in corr])
    ax.plot(np.linspace(0, span, 100), np.ones(100), c="k", ls="--", lw=0.8)
    for level in np.arange(step, 1.45 * sd_obs, step):
        d = level / sd_obs
        cx, cy = 1 + d * np.cos(phi), d * np.sin(phi)
        theta, rho = np.arctan2(cy, cx), np.hypot(cx, cy)
        keep = (theta <= span + 1e-9) & (rho <= rmax)
        ax.plot(theta[keep], rho[keep], c="0.55", ls=":", lw=0.8)
        lx, ly = 1 + d * np.cos(np.radians(125)), d * np.sin(np.radians(125))
        if np.arctan2(ly, lx) <= span and np.hypot(lx, ly) <= rmax:
            ax.text(np.arctan2(ly, lx), np.hypot(lx, ly), f"{level:g}", fontsize=8, color="0.3", ha="center",
                    va="center", bbox={"fc": "white", "ec": "none", "pad": 0.6, "alpha": 0.85})
    ax.plot(0, 1, marker="*", c="k", ms=13, ls="")
    for _, row in metrics.iterrows():
        ax.plot(np.arccos(np.clip(row["r"], -1, 1)), row["alpha"], marker="o", ms=9, mec="white", ls="",
                color=colors[row["model"]])
    ax.set_xlabel("α = σpred / σobs", labelpad=16)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if span == np.pi / 2:
        for v in ticks[1:]:
            ax.annotate(f"{v:.1f}", (span, v), xytext=(-5, 0), textcoords="offset points", ha="right", va="center", fontsize=9)
        ax.set_ylabel("α = σpred / σobs", labelpad=24)
    ax.text(span / 2, rmax * 1.2, "Correlation r", rotation=np.degrees(span / 2) - 90, ha="center", va="center")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[m], mec="white", ms=9, label=m) for m in models]
               + [Line2D([], [], marker="*", ls="", color="k", ms=13, label="observed"),
                  Line2D([], [], ls="--", color="k", lw=0.8, label="σpred = σobs"),
                  Line2D([], [], ls=":", color="0.55", label="centered RMSE (cm)")],
               ncols=3, loc="outside lower center")
    fig.suptitle(title)
    return fig


def shap_importance_plot(table, title):
    """Mean |SHAP| (cm) of each unit, one panel per model (columns of table), units sorted within each panel."""
    fig = Figure(figsize=(3.6 * table.shape[1], 0.32 * len(table) + 1.6), layout="constrained")
    axes = fig.subplots(1, table.shape[1], squeeze=False)[0]
    colors = _model_colors(["climatology", *table.columns])
    for ax, model in zip(axes, table):
        values = table[model].sort_values()
        ax.barh(range(len(values)), values, color=colors[model])
        ax.set_yticks(range(len(values)), values.index)
        ax.set_xlabel("mean |SHAP| (cm)")
        ax.set_title(model)
        ax.grid(axis="x", lw=0.4, alpha=0.4)
    fig.suptitle(title)
    return fig


def shap_beeswarm_plot(shap_units, values, title):
    """One row per unit (most important on top): the out-of-fold SHAP of each day (positive = deeper), colored by
    the value of the unit on that day (rank within the unit, low = blue, high = red)."""
    order = shap_units.abs().mean().sort_values().index
    rng = np.random.default_rng(0)
    fig = Figure(figsize=(9, 0.38 * len(order) + 1.6), layout="constrained")
    ax = fig.subplots()
    for i, unit in enumerate(order):
        s = shap_units[unit]
        jitter = rng.uniform(-0.3, 0.3, len(s))
        points = ax.scatter(s, i + jitter, c=values[unit].rank(pct=True).reindex(s.index), cmap="coolwarm", s=5,
                            vmin=0, vmax=1, lw=0)
    ax.axvline(0, c="k", lw=0.6)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("SHAP (cm; positive = predicted deeper)")
    fig.colorbar(points, ax=ax, label="value of the unit (percentile)", ticks=[0, 1], shrink=0.6).ax.set_yticklabels(["low", "high"])
    fig.suptitle(title)
    return fig


def elimination_path_plot(path, proposed_step, title):
    """Backward elimination: difference of the pooled out-of-fold RMSE with the full set against the number of units
    left, with its bootstrap interval; each step is labeled with the unit removed, the steps whose difference is
    significant are red; the step with the lowest ΔRMSE and the proposed step are marked with their value."""
    x = path["n_units"].to_numpy()
    fig = Figure(figsize=(11, 5.5), layout="constrained")
    ax = fig.subplots()
    value = "d_rmse_full"
    ax.errorbar(x, path[value], yerr=[path[value] - path[f"{value}_lo"], path[f"{value}_hi"] - path[value]],
                fmt="none", ecolor="0.6", lw=1)
    ax.plot(x, path[value], c="0.4", lw=0.8)
    ax.scatter(x, path[value], c=np.where(path["significant_vs_full"], "tab:red", "tab:blue"), s=30, zorder=3)
    lowest = path[value].idxmin()
    for step, color, name in ((lowest, "tab:purple", "lowest"), (proposed_step, "tab:green", "proposed")):
        row = path.loc[step]
        ax.axvline(row["n_units"], c=color, ls="--", lw=1)
        ax.annotate(f"{name}: ΔRMSE = {row[value]:.2f} cm\n[{row[f'{value}_lo']:.2f}, {row[f'{value}_hi']:.2f}], "
                    f"{row['n_units']} units", (row["n_units"], row[value]), xytext=(8, -14), textcoords="offset points",
                    ha="left", va="top", fontsize=8.5, color=color,
                    bbox={"fc": "white", "ec": color, "lw": 0.6, "pad": 2, "alpha": 0.9})
    ax.axhline(0, c="k", lw=0.8)
    for _, row in path.iterrows():
        if row["removed"]:
            ax.annotate(f"−{row['removed']}", (row["n_units"], row[f"{value}_hi"]), xytext=(0, 4),
                        textcoords="offset points", rotation=90, ha="center", va="bottom", fontsize=8)
    low, high = path[f"{value}_lo"].min(), path[f"{value}_hi"].max()
    ax.set_ylim(low - 0.2 * (high - low), high + 0.3 * (high - low))
    ax.set_ylabel("ΔRMSE vs all units (cm; negative = better)")
    ax.set_xlabel("units left")
    ax.set_xticks(x)
    ax.invert_xaxis()
    ax.grid(lw=0.4, alpha=0.4)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="tab:blue", label="not significant vs all units"),
                        Line2D([], [], marker="o", ls="", color="tab:red", label="significant vs all units"),
                        Line2D([], [], ls="--", color="tab:purple", label="lowest ΔRMSE"),
                        Line2D([], [], ls="--", color="tab:green", label="proposed set")], ncols=4, loc="outside lower center")
    fig.suptitle(title)
    return fig


def readd_plot(readd, title):
    """ΔRMSE (with bootstrap interval) of adding back each removed unit to the proposed set; negative = better."""
    readd = readd.sort_values("d_rmse")
    fig = Figure(figsize=(7, 0.35 * len(readd) + 1.6), layout="constrained")
    ax = fig.subplots()
    y = np.arange(len(readd))
    colors = np.where(readd["d_rmse_hi"] < 0, "tab:green", np.where(readd["d_rmse_lo"] > 0, "tab:red", "tab:blue"))
    ax.errorbar(readd["d_rmse"], y, xerr=[readd["d_rmse"] - readd["d_rmse_lo"], readd["d_rmse_hi"] - readd["d_rmse"]],
                fmt="none", ecolor="0.6", lw=1)
    ax.scatter(readd["d_rmse"], y, c=colors, s=30, zorder=3)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(y, readd.index)
    ax.set_xlabel("ΔRMSE vs proposed set (cm; negative = better)")
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    fig.suptitle(title)
    return fig
