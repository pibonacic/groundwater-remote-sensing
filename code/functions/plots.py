"""One function per figure. Each returns a matplotlib Figure and saves nothing."""
import matplotlib
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import dendrogram

from functions.features import season_of
from functions.evaluation import METRIC_LABELS, METRIC_REFERENCE, METRICS
from functions.validation import GENERATORS, SCENARIOS, corr_grid, r_critical, slope_sign

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


def scatter_by_season(X, y, title, ncols=3):
    """Each column of X against y, colored by season (DJF = austral summer); ncols panels per row."""
    nrows = -(-len(X.columns) // ncols)
    fig = Figure(figsize=(3.4 * ncols, 3.2 * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, sharey=True, squeeze=False)
    season = season_of(y.index)
    for ax, col in zip(axes.flat, X):
        for name, color in SEASON_COLORS.items():
            m = season == name
            ax.scatter(X[col][m], y[m], s=6, c=color, label=name)
        ax.set_xlabel(col)
    for ax in axes.flat[len(X.columns):]:
        ax.set_visible(False)
    _depth_axis(axes[0, 0])
    for row in axes:
        row[0].set_ylabel(DEPTH_LABEL)
    axes[0, -1].legend(markerscale=2)
    fig.suptitle(title)
    return fig


Z_LIMIT = 4   # y range of zscore_series_plot: +-4 standard deviations


def zscore_series_plot(X, y, title, suffix=" (z)"):
    """One panel per column of X: z-scored y (left axis, inverted: y is a depth) and z-scored column (right axis)
    over time. Both axes of every panel share the same range, +-Z_LIMIT (the same in all panels); suffix goes after the names in the axis labels."""
    zscore = lambda s: (s - s.mean()) / s.std()
    fig = Figure(figsize=(11, 2.2 * len(X.columns)), layout="constrained")
    axes = fig.subplots(len(X.columns), 1, sharex=True, squeeze=False)[:, 0]
    zy = zscore(y)
    zX = X.apply(zscore)
    low, high = -Z_LIMIT, Z_LIMIT   # the same range in every panel; values beyond it are cut
    pad = 0
    for ax, col in zip(axes, X):
        zx = zX[col]
        ax.plot(zy, lw=0.8, c="tab:blue")
        ax.set_ylabel(f"{y.name}{suffix}, deeper down", color="tab:blue")
        right = ax.twinx()
        right.plot(zx, lw=0.6, c="tab:orange")
        right.set_ylabel(f"{col}{suffix}", color="tab:orange")
        ax.axhline(0, c="k", lw=0.4)
        ax.set_ylim(high + pad, low - pad)
        right.set_ylim(low - pad, high + pad)
    axes[-1].set_xlim(y.index.min(), y.index.max())
    fig.suptitle(title)
    return fig


def _grid(n, ncols, width=3.2, height=2.6, **kwargs):
    """Figure with n panels in rows of ncols (the unused ones hidden)."""
    nrows = -(-n // ncols)
    fig = Figure(figsize=(width * ncols, height * nrows + 0.6), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, **kwargs)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes


def ccf_plot(ccf, summary, title, ncols=5):
    """CCF of each variable (one panel each) against k, with k = 0 and the k of the largest |rho| marked (red when
    it lies on the edge of the k range: the maximum may be outside)."""
    fig, axes = _grid(len(ccf.columns), ncols, sharex=True, sharey=True)
    for ax, col in zip(axes.flat, ccf):
        ax.plot(ccf.index, ccf[col], c="tab:blue", lw=1.2)
        ax.axvline(0, c="k", lw=0.5)
        ax.axhline(0, c="k", lw=0.5)
        if col in summary.index:
            row = summary.loc[col]
            color = "tab:red" if row["at_edge"] else "k"
            ax.plot(row["k_max"], row["rho_max"], "o", c=color, ms=5)
            ax.set_title(f"{col}: k = {row['k_max']}, rho = {row['rho_max']:.2f}" + (" (edge)" if row["at_edge"] else ""),
                         fontsize=8.5, color=color)
        ax.grid(lw=0.4, alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("k (days; k > 0: variable after the level)")
    for row in axes:
        row[0].set_ylabel("Spearman rho")
    fig.suptitle(title)
    return fig


def lagged_scatter_plot(X, y, summary, title, ncols=5):
    """Each variable shifted to the k of its largest |CCF| (x(t + k) against the depth at t), colored by season;
    k and rho in the title of each panel, red when k lies on the edge of the k range."""
    cols = [c for c in X if c in summary.index]
    fig, axes = _grid(len(cols), ncols, sharey=True)
    season = season_of(y.index)
    for ax, col in zip(axes.flat, cols):
        row = summary.loc[col]
        x = X[col].shift(-int(row["k_max"]))
        for name, color in SEASON_COLORS.items():
            m = season == name
            ax.scatter(x[m], y[m], s=4, c=color, label=name, lw=0)
        edge = bool(row["at_edge"])
        ax.set_title(f"{col}(t{int(row['k_max']):+d}): rho = {row['rho_max']:.2f}" + (" EDGE" if edge else ""),
                     fontsize=8.5, color="tab:red" if edge else "k")
        ax.set_xlabel(col, fontsize=8)
    _depth_axis(axes[0, 0])
    for r in axes:
        r[0].set_ylabel(DEPTH_LABEL, fontsize=8)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=n) for n, c in SEASON_COLORS.items()],
               ncols=4, loc="outside lower center")
    fig.suptitle(title)
    return fig


def corr_heatmap(panels, title, label="Spearman rho", dot_label="interval excludes 0", hide=None):
    """Heatmaps of correlations: panels maps a panel title to (rho, significant), two tables with the variables as
    rows and the strata as columns. Diverging scale centered on 0; a dot marks the cells whose interval leaves
    out 0; empty cells (too few days) are gray. hide: optional boolean table (same shape as rho, one panel) of cells
    left blank (white), e.g. the repeated half of a symmetric matrix."""
    n_rows = max(len(rho) for rho, _ in panels.values())
    widths = [max(len(rho.columns), 2) for rho, _ in panels.values()]
    fig = Figure(figsize=(1.6 + 0.42 * sum(widths) + 1.2 * len(panels), 0.28 * n_rows + 2), layout="constrained")
    axes = fig.subplots(1, len(panels), squeeze=False, width_ratios=widths)[0]
    for ax, (name, (rho, sig)) in zip(axes, panels.items()):
        values = np.ma.masked_invalid(rho.to_numpy(float))
        ax.set_facecolor("0.85")
        mesh = ax.pcolormesh(values, cmap="RdBu_r", vmin=-1, vmax=1)
        if hide is not None:
            blank = np.ma.masked_where(~hide.reindex(index=rho.index, columns=rho.columns).to_numpy(bool),
                                       np.zeros(rho.shape))
            ax.pcolormesh(blank, cmap=matplotlib.colors.ListedColormap(["white"]))
        yy, xx = np.nonzero(sig.reindex(index=rho.index, columns=rho.columns).fillna(False).to_numpy(bool))
        ax.scatter(xx + 0.5, yy + 0.5, s=9, c="k", marker="o", lw=0)
        ax.set_xticks(np.arange(len(rho.columns)) + 0.5, [str(c) for c in rho.columns], rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(rho.index)) + 0.5, rho.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(name, fontsize=9)
    fig.colorbar(mesh, ax=axes, label=f"{label} (dot: {dot_label})", shrink=0.8)
    fig.suptitle(title)
    return fig


def lag_heatmap(table, stats, alpha, title):
    """corr_heatmap of a cross_correlation table: one panel per modeled statistic, wells as rows and lags as columns
    (dot: p < alpha)."""
    wells_list = sorted(table.dropna(subset=["rho"])["well"].unique())
    panels = {}
    for stat in stats:
        sub = table[table["prediction"] == stat]
        rho_l = sub.pivot(index="well", columns="lag_months", values="rho").reindex(wells_list)
        p_l = sub.pivot(index="well", columns="lag_months", values="p").reindex(wells_list)
        panels[f"modeled {stat}"] = (rho_l, p_l < alpha)
    return corr_heatmap(panels, title, dot_label=f"p < {alpha}")


def date_scatter_plot(X, y, window, title, ncols=5):
    """Each variable against the depth, points colored by date, with the trajectory of the rolling mean of window
    days of both (drawn only where the window is complete), so the order in time is visible."""
    fig, axes = _grid(len(X.columns), ncols, sharey=True)
    dates = mdates.date2num(y.index)
    smooth_y = y.rolling(window, center=True, min_periods=window).mean()
    for ax, col in zip(axes.flat, X):
        points = ax.scatter(X[col], y, c=dates, cmap="viridis", s=4, lw=0)
        ax.plot(X[col].rolling(window, center=True, min_periods=window).mean(), smooth_y, c="k", lw=0.6, alpha=0.7)
        ax.set_xlabel(col, fontsize=8)
    _depth_axis(axes[0, 0])
    for r in axes:
        r[0].set_ylabel(DEPTH_LABEL, fontsize=8)
    bar = fig.colorbar(points, ax=axes, shrink=0.6, label="date")
    locator = mdates.AutoDateLocator()
    bar.ax.yaxis.set_major_locator(locator)
    bar.ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle(title)
    return fig


def phase_relation_plot(table, title):
    """Spearman rho of each variable with the depth in the rise and in the fall of the level, with intervals."""
    variables = list(dict.fromkeys(table["variable"]))
    fig = Figure(figsize=(7, 0.32 * len(variables) + 1.8), layout="constrained")
    ax = fig.subplots()
    y = np.arange(len(variables))
    for offset, (phase, color) in zip((-0.15, 0.15), (("rise", "tab:blue"), ("fall", "tab:orange"))):
        part = table[table["phase"] == phase].set_index("variable").reindex(variables)
        ax.errorbar(part["rho"], y + offset, xerr=[part["rho"] - part["rho_lo"], part["rho_hi"] - part["rho"]],
                    fmt="o", ms=5, lw=1, color=color, label=f"{phase} of the level")
    ax.axvline(0, c="k", lw=0.8)
    ax.set_yticks(y, variables)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho with the depth")
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    return fig


def harmonic_stability_plot(curves, summary, comparison, labels, title, ncols=5):
    """Harmonic curves of each variable fitted on two periods (labels: names of A and B) and, shaded around the
    curve of A, the interval of the difference B - A; the shift and the share of significant days in the title."""
    rows = summary[summary["comparison"] == comparison].set_index("variable")
    fig, axes = _grid(len(rows), ncols)
    for ax, col in zip(axes.flat, rows.index):
        c = curves[(comparison, col)]
        ax.fill_between(c.index, c["curve_a"] + c["diff_lo"], c["curve_a"] + c["diff_hi"], color="tab:red", alpha=0.15, lw=0)
        ax.plot(c.index, c["curve_a"], c="tab:blue", lw=1.2)
        ax.plot(c.index, c["curve_b"], c="tab:red", lw=1.2)
        if col == "gw_cm":
            ax.invert_yaxis()
        r = rows.loc[col]
        ax.set_title(f"{col}: shift {r['shift']:.3g} [{r['shift_lo']:.3g}, {r['shift_hi']:.3g}]\n"
                     f"significant days {r['frac_days_significant']:.0%}", fontsize=8)
        ax.set_xticks([1, 91, 182, 274], ["Jan", "Apr", "Jul", "Oct"])
    fig.legend(handles=[Line2D([], [], c="tab:blue", label=labels[0]), Line2D([], [], c="tab:red", label=labels[1]),
                        Patch(color="tab:red", alpha=0.15, label="interval of the difference (around the first curve)")],
               ncols=3, loc="outside lower center")
    fig.suptitle(title)
    return fig


def slow_series_plot(frame, slow, title):
    """One panel per column: the daily series and its slow part (rolling median) on top."""
    fig = Figure(figsize=(11, 1.6 * len(frame.columns) + 0.6), layout="constrained")
    axes = fig.subplots(len(frame.columns), 1, sharex=True, squeeze=False)[:, 0]
    for ax, col in zip(axes, frame):
        ax.plot(frame[col], c="0.6", lw=0.6)
        ax.plot(slow[col], c="tab:blue", lw=1.2)
        ax.set_ylabel(col, fontsize=8)
        if col == "gw_cm":
            ax.invert_yaxis()
    axes[-1].set_xlim(frame.index.min(), frame.index.max())
    fig.legend(handles=[Line2D([], [], c="0.6", label="daily"), Line2D([], [], c="tab:blue", label="slow part")],
               ncols=2, loc="outside lower center")
    fig.suptitle(title)
    return fig


def jumps_plot(y, precip, jumps, title):
    """Depth with the jumps marked (filled: recovered, open: not recovered; blue: rise of the level, red: fall) and
    the daily rain below."""
    fig = Figure(figsize=(12, 5.5), layout="constrained")
    ax, ax_rain = fig.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    ax.plot(y, c="k", lw=0.8)
    for direction, color in (("rise", "tab:blue"), ("fall", "tab:red")):
        for recovered in (True, False):
            j = jumps[(jumps["direction"] == direction) & (jumps["recovered"] == recovered)] if len(jumps) else jumps
            if len(j):
                ax.scatter(j["date"], j["depth_after_cm"], s=30, edgecolors=color, lw=1.2,
                           facecolors=color if recovered else "none", zorder=3,
                           label=f"{direction} of the level, {'recovered' if recovered else 'not recovered'}")
    _depth_axis(ax, "depth (cm)")
    ax.grid(lw=0.4, alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, ncols=2, frameon=False)
    ax_rain.bar(precip.index, precip.to_numpy(), width=1, color="tab:blue")
    ax_rain.set_ylabel("CHIRPS (mm/day)")
    ax_rain.set_xlim(y.index.min(), y.index.max())
    fig.suptitle(title)
    return fig


def september_plot(frame, years, rho, title):
    """Top row: each column of frame in September, one line per year (x: day of September). Bottom row: each
    variable (not the depth) against the depth in September, colored by year, with the Spearman of each year."""
    variables = [c for c in frame if c != "gw_cm"]
    year_list = sorted(pd.unique(years))
    colors = {yr: f"C{i}" for i, yr in enumerate(year_list)}
    fig = Figure(figsize=(3 * len(frame.columns), 6), layout="constrained")
    axes = fig.subplots(2, len(frame.columns), squeeze=False)
    for ax, col in zip(axes[0], frame):
        for yr in year_list:
            s = frame[col][years == yr]
            ax.plot(s.index.day, s.to_numpy(), c=colors[yr], lw=1.2)
        if col == "gw_cm":
            ax.invert_yaxis()
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("day of September", fontsize=8)
    for ax, col in zip(axes[1], variables):
        for yr in year_list:
            m = years == yr
            r = rho[(rho["variable"] == col) & (rho["year"] == yr)]["rho"]
            ax.scatter(frame[col][m], frame["gw_cm"][m], s=8, c=colors[yr], lw=0,
                       label=f"{yr}: rho {r.iloc[0]:.2f}" if len(r) and pd.notna(r.iloc[0]) else f"{yr}")
        ax.set_xlabel(col, fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        _depth_axis(ax, "depth (cm)", fontsize=8)
    axes[1, -1].set_visible(False)
    fig.legend(handles=[Line2D([], [], c=colors[yr], label=str(yr)) for yr in year_list], ncols=len(year_list),
               loc="outside lower center")
    fig.suptitle(title)
    return fig


def annual_anomaly_plot(table, title):
    """Mean anomaly per period (rows of table: variable, period, mean, mean_lo, mean_hi), one panel per variable,
    with its interval; the depth panel has the axis inverted (deeper down)."""
    variables = list(dict.fromkeys(table["variable"]))
    periods = list(dict.fromkeys(table["period"]))   # in the order of table
    fig = Figure(figsize=(max(8, 0.3 * len(periods) + 3), 2.3 * len(variables) + 0.6), layout="constrained")
    axes = fig.subplots(len(variables), 1, sharex=True, squeeze=False)[:, 0]
    x = {p: i for i, p in enumerate(periods)}
    for ax, var in zip(axes, variables):
        part = table[table["variable"] == var]
        xs = part["period"].map(x)
        ax.errorbar(xs, part["mean"], yerr=[part["mean"] - part["mean_lo"], part["mean_hi"] - part["mean"]],
                    fmt="o", ms=4, lw=1, color="tab:blue")
        ax.axhline(0, c="k", lw=0.6)
        ax.set_ylabel(var, fontsize=8)
        ax.grid(lw=0.4, alpha=0.4)
        if var.startswith("gw_cm"):
            ax.invert_yaxis()
    axes[-1].set_xticks(range(len(periods)), periods, rotation=90, fontsize=7)
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


MONTH_TICKS = pd.date_range("2000-01-01", periods=12, freq="MS")   # x axis of the annual plots: a leap year


def _calendar_date(index):
    """Same month and day in the leap year 2000 (the common x axis of the annual plots)."""
    return pd.to_datetime({"year": 2000, "month": index.month, "day": index.day})


def _month_axis(ax):
    ax.set_xticks(MONTH_TICKS, MONTH_TICKS.strftime("%b"))
    ax.set_xlim(MONTH_TICKS[0], pd.Timestamp("2000-12-31"))


def annual_cycle_plot(pred, observed, title):
    """January to December: the predicted depth of every year as semi-transparent lines and the observed depth of
    each observed year as an opaque line (deeper down)."""
    fig = Figure(figsize=(11, 4), layout="constrained")
    ax = fig.subplots()
    pred_years = sorted(set(pred.index.year))
    for year in pred_years:
        p = pred[pred.index.year == year]
        ax.plot(_calendar_date(p.index), p.to_numpy(), c="tab:blue", lw=0.8, alpha=0.3)
    handles = [Line2D([], [], c="tab:blue", lw=0.8, alpha=0.5, label=f"modeled ({pred_years[0]}–{pred_years[-1]})")]
    obs_years = sorted(set(observed.index.year))
    shades = np.linspace(0.15, 0.55, len(obs_years))
    for year, shade in zip(obs_years, shades):
        o = observed[observed.index.year == year]
        color = (shade * 0.6, shade * 0.75, shade + 0.25)
        ax.plot(_calendar_date(o.index), o.to_numpy(), c=color, lw=1.8)
        handles.append(Line2D([], [], c=color, lw=1.8, label=f"observed {year}"))
    _depth_axis(ax, "depth (cm)")
    _month_axis(ax)
    ax.grid(lw=0.4, alpha=0.4)
    ax.legend(handles=handles, loc="upper left", frameon=False)
    fig.suptitle(title)
    return fig


def year_doy_heatmap(values, observed_days, title, label="depth (cm)", cmap="magma_r", center=None):
    """One row per year and one column per day of the year (leap-year calendar, so the same date is in the same
    column every year), colored by values; days without a value are blank. Dashed rectangles mark the observed
    days (one per run of consecutive observed days in each year). center: a diverging scale symmetric around it."""
    cal = _calendar_date(values.index)
    years = np.arange(values.index.year.min(), values.index.year.max() + 1)
    grid = pd.DataFrame(np.nan, index=years, columns=pd.date_range("2000-01-01", "2000-12-31"))
    for (year, day), value in zip(zip(values.index.year, cal), values.to_numpy()):
        grid.loc[year, day] = value
    no_leap = ~pd.DatetimeIndex([f"{y}-01-01" for y in years]).is_leap_year
    feb28, feb29 = pd.Timestamp("2000-02-28"), pd.Timestamp("2000-02-29")
    grid.loc[no_leap, feb29] = grid.loc[no_leap, feb28]   # display only: no blank Feb 29 column in common years
    fig = Figure(figsize=(11, 0.28 * len(years) + 1.6), layout="constrained")
    ax = fig.subplots()
    x_edges = np.append(grid.columns, grid.columns[-1] + pd.Timedelta(days=1))
    y_edges = np.append(years, years[-1] + 1) - 0.5
    kwargs = {}
    if center is not None:
        span = np.nanmax(np.abs(grid.to_numpy() - center))
        kwargs = {"vmin": center - span, "vmax": center + span}
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(grid.to_numpy()), cmap=cmap, shading="flat", **kwargs)
    observed = pd.DatetimeIndex(observed_days)
    for year in sorted(set(observed.year)):
        days = observed[observed.year == year]
        run = (days.to_series().diff() != pd.Timedelta(days=1)).cumsum()
        for _, d in days.to_series().groupby(run.to_numpy()):
            start, end = _calendar_date(d.index[[0, -1]])
            ax.add_patch(Rectangle((start, year - 0.5), end + pd.Timedelta(days=1) - start, 1, fill=False, ls="--",
                                   lw=1.2, ec="white" if center is None else "k"))
    ax.set_yticks(years)
    ax.invert_yaxis()
    _month_axis(ax)
    ax.set_xlim(x_edges[0], x_edges[-1])
    bar = fig.colorbar(mesh, ax=ax, label=label, pad=0.01)
    if center is None:
        bar.ax.invert_yaxis()   # shallow on top, as the depth axes
    ax.set_title("dashed: observed days", loc="right", fontsize=9)
    fig.suptitle(title)
    return fig


def _interval_bands(ax, quantiles, levels, color="tab:blue"):
    for level, alpha in zip(sorted(levels, reverse=True), (0.18, 0.35)):
        pct = round(level * 100)
        ax.fill_between(quantiles.index, quantiles[f"lo{pct}"], quantiles[f"hi{pct}"], color=color, alpha=alpha,
                        lw=0, label=f"{pct}% interval")


def prediction_series_plot(table, levels, title):
    """Whole predicted series: point prediction, prediction intervals, observed depth and, as ticks at the bottom,
    the days whose inputs fall outside the training range.

    table has pred, lo{pct} / hi{pct}, observed (NaN when not observed) and n_outside."""
    fig = Figure(figsize=(13, 4.2), layout="constrained")
    ax = fig.subplots()
    _interval_bands(ax, table, levels)
    ax.plot(table["pred"], c="tab:blue", lw=0.7, label="prediction")
    ax.plot(table["observed"], c="k", lw=0.9, label="observed")
    extrap = table.index[table["n_outside"] > 0]
    ax.plot(extrap, np.full(len(extrap), 0.02), "|", c="tab:red", ms=6, transform=ax.get_xaxis_transform(),
            label="inputs outside training range")
    _depth_axis(ax, "depth (cm)")
    ax.set_xlim(table.index.min(), table.index.max())
    ax.grid(lw=0.4, alpha=0.4)
    ax.legend(loc="upper left", ncols=5, frameon=False, fontsize=8)
    fig.suptitle(title)
    return fig


def coverage_plot(y, pred, quantiles, levels, coverage, title):
    """Validation of the intervals on the holdout: observed depth, prediction and intervals; the empirical coverage
    of each level (coverage table, stratum "all") goes in the title."""
    fig = Figure(figsize=(11, 3.8), layout="constrained")
    ax = fig.subplots()
    _interval_bands(ax, quantiles, levels)
    ax.plot(pred, c="tab:blue", lw=0.9, label="prediction")
    ax.plot(y, c="k", lw=1, label="observed")
    _depth_axis(ax, "depth (cm)")
    ax.grid(lw=0.4, alpha=0.4)
    ax.legend(loc="upper left", ncols=4, frameon=False, fontsize=8)
    cov = coverage[coverage["stratum"] == "all"]
    fig.suptitle(f"{title}\n" + ", ".join(f"{round(l * 100)}% interval covers {c:.0%}"
                                          for l, c in zip(cov["level"], cov["coverage"])))
    return fig


def member_examples_plot(members, sim, title):
    """One panel per column of members: the prediction of that ensemble member and the same member minus its drawn
    out-of-fold residual (the simulated depth, sim)."""
    fig = Figure(figsize=(13, 2.4 * len(members.columns)), layout="constrained")
    axes = fig.subplots(len(members.columns), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    for ax, member in zip(axes, members):
        ax.plot(sim[member], c="tab:orange", lw=0.5, alpha=0.8, label="member − r")
        ax.plot(members[member], c="tab:blue", lw=0.8, label="member")
        ax.set_title(member, loc="left", fontsize=9)
        _depth_axis(ax, "depth (cm)")
        ax.grid(lw=0.4, alpha=0.4)
    axes[0].legend(loc="upper left", ncols=2, frameon=False, fontsize=8)
    axes[-1].set_xlim(members.index.min(), members.index.max())
    fig.suptitle(title)
    return fig


PRED_COLOR = "0.25"   # modeled median (dark gray); the min-max range is a translucent gray band
ALTITUDE_LABEL = "water table altitude (m a.s.l.)"
MONTH_NAMES = pd.date_range("2000-01-01", periods=12, freq="MS").strftime("%b")


def _pred_lines(ax, pred, median_label, range_label, band=True):
    """Modeled median as a dark gray dashed line and, with band, the min-max range as a gray band
    (pred: min, median, max)."""
    if band:
        ax.fill_between(pred.index, pred["min"].to_numpy(), pred["max"].to_numpy(), color="0.5", alpha=0.3, lw=0,
                        label=range_label)
    ax.plot(pred.index, pred["median"].to_numpy(), ls="--", lw=1.6, c=PRED_COLOR, label=median_label)


PANEL_LABELS = ("depth (z-score)", "depth (cm)", ALTITUDE_LABEL)


def _three_panels(figsize, labels=PANEL_LABELS, zero_lines=False):
    """z-score and depth (deeper down) on top, altitude (higher up) below, sharing the x axis. labels: y labels of
    the three panels; zero_lines: a line at 0 in every panel (anomalies), not only in the z-score panel."""
    fig = Figure(figsize=figsize, layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    _depth_axis(axes[0], labels[0])
    _depth_axis(axes[1], labels[1])
    axes[2].set_ylabel(labels[2])
    for ax in axes if zero_lines else axes[:1]:
        ax.axhline(0, c="k", lw=0.6)
    for ax in axes:
        ax.grid(lw=0.4, alpha=0.4)
    return fig, axes


def _month_ticks(axes):
    for ax in axes:
        ax.set_xticks(range(1, 13), MONTH_NAMES)
        ax.set_xlim(0.5, 12.5)


def well_validation_plot(pred_z, wells_z, pred_raw, wells_raw, pred_alt, wells_alt, title,
                         pred_label="modeled monthly {}", well_label="{} (monthly mean)", labels=PANEL_LABELS,
                         zero_lines=False, band=True):
    """Monthly modeled water table (median: dark gray dashed; min-max: gray band, not in the z-score panel) against
    the monthly mean of independent wells (solid with markers): z-scores of depth, depth (cm) and altitude (m a.s.l.).

    pred_* have the columns min, median, max; wells_* one column per well, on the same index with every month of the
    period (months without a visit are NaN, so the line of a well is cut where it skips a month). pred_label and
    well_label are formatted with the statistic and the well; labels and zero_lines as in _three_panels; band=False
    leaves out the min-max band in every panel (then pred_* need only the median)."""
    fig, axes = _three_panels((12, 10), labels, zero_lines)
    for ax, pred, wells, with_band in ((axes[0], pred_z, wells_z, False), (axes[1], pred_raw, wells_raw, band),
                                       (axes[2], pred_alt, wells_alt, band)):
        _pred_lines(ax, pred, pred_label.format("median"), pred_label.format("min-max"), with_band)
        for i, well in enumerate(wells):   # NaN months break the line (wells has every month of the period)
            ax.plot(wells.index, wells[well].to_numpy(), ls="-", marker="o", ms=4, lw=1.2, c=f"C{i}",
                    label=well_label.format(well))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=6, fontsize=8, frameon=False, loc="outside lower center")
    fig.suptitle(title)
    return fig


def seasonal_cycle_plot(pred_z, wells_z, pred_raw, wells_raw, pred_alt, wells_alt, title):
    """Seasonal cycle (calendar months 1-12) of the modeled median (dashed) and min-max (band, not in the z-score
    panel) and of the wells (solid with markers). Same inputs as well_validation_plot with the month number as index;
    a NaN month cuts the line."""
    fig = well_validation_plot(pred_z, wells_z, pred_raw, wells_raw, pred_alt, wells_alt, title,
                               pred_label="modeled {} (climatology)", well_label="{} (climatology)")
    _month_ticks(fig.axes[:3])
    return fig


def wells_cycle_plot(wells_z, wells_raw, wells_alt, title):
    """Seasonal cycle (calendar months 1-12) of every well, solid with markers (a NaN month cuts the line): z-scores
    of depth, depth (cm) and altitude (m a.s.l.). wells_* have one column per well, month number as index."""
    colors = list(matplotlib.colormaps["tab20"].colors) + list(matplotlib.colormaps["tab20b"].colors)
    fig, axes = _three_panels((12, 11))
    for ax, wells in zip(axes, (wells_z, wells_raw, wells_alt)):
        for i, well in enumerate(wells_raw):
            if well in wells:
                ax.plot(wells.index, wells[well].to_numpy(), ls="-", marker="o", ms=4, lw=1.2,
                        c=colors[i % len(colors)], label=well)
    _month_ticks(axes)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=8, fontsize=8, frameon=False, loc="outside lower center")
    fig.suptitle(title)
    return fig


def wells_series_plot(wells_z, wells_raw, wells_alt, pred_z, pred_raw, pred_alt, title, z_limit=None,
                      raw_limit=None, alt_limit=None):
    """Every well (monthly mean, solid with markers, cut where a month is missing), the modeled monthly median
    (dark gray dashed) and min-max (gray band, not in the z-score panel): z-scores of depth, depth (cm) and altitude
    (m a.s.l.). wells_* have one column per well and every month of the period; pred_* have the columns min, median,
    max on the same index. z_limit: the z-score panel shows -z_limit to z_limit; raw_limit: the largest depth shown;
    alt_limit: the lowest altitude shown. Wells beyond a limit are drawn as triangles on that edge and counted above
    the panel."""
    colors = list(matplotlib.colormaps["tab20"].colors) + list(matplotlib.colormaps["tab20b"].colors)
    well_color = {well: colors[i % len(colors)] for i, well in enumerate(wells_raw.columns)}
    fig, axes = _three_panels((13, 12))
    for ax, wells, pred, band in ((axes[0], wells_z, pred_z, False), (axes[1], wells_raw, pred_raw, True),
                                  (axes[2], wells_alt, pred_alt, True)):
        for well in wells:
            ax.plot(wells.index, wells[well].to_numpy(), ls="-", marker="o", ms=3, lw=1, c=well_color[well], label=well)
        _pred_lines(ax, pred, "modeled monthly median", "modeled monthly min-max", band)

    def mark_beyond(ax, wells, low, high, unit, deeper_is_high):
        """Triangles on the edge for the months of wells outside low..high (pointing deeper or shallower), counted
        above the panel."""
        stacked = wells.stack().dropna()
        outside = stacked[(stacked < low) | (stacked > high)]
        for (month, well), value in outside.items():
            edge = high if value > high else low
            deeper = (value > high) == deeper_is_high
            ax.plot(month, edge, marker="v" if deeper else "^", ms=7, ls="", c=well_color[well])
        ax.set_title(f"{len(outside)} monthly well values outside {low:g} to {high:g} {unit}, drawn as triangles on "
                     f"the edge", fontsize=8, loc="right")

    if z_limit is not None:
        axes[0].set_ylim(z_limit, -z_limit)
        mark_beyond(axes[0], wells_z, -z_limit, z_limit, "sd", True)
    if raw_limit is not None:
        low = min(wells_raw.min().min(), pred_raw.min().min(), 0)
        axes[1].set_ylim(raw_limit, low - 0.03 * (raw_limit - low))
        mark_beyond(axes[1], wells_raw, low, raw_limit, "cm", True)
    if alt_limit is not None:
        high = max(wells_alt.max().max(), pred_alt.max().max())
        axes[2].set_ylim(alt_limit, high + 0.03 * (high - alt_limit))
        mark_beyond(axes[2], wells_alt, alt_limit, high, "m a.s.l.", False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=8, fontsize=8, frameon=False, loc="outside lower center")
    fig.suptitle(title)
    return fig


def manual_measurements_plot(observed, manual, observed_monthly, manual_monthly, xlim, title, observed_label,
                             observed_alt, manual_alt):
    """Top: the observed daily depth of the site (continuous line) and the manual measurements of the wells on their
    day (markers, one color per well). Middle: the same at monthly resolution (monthly medians placed on day 15;
    lines join consecutive months only). Bottom: the daily values as water table altitude (m a.s.l., higher up):
    observed_alt, the daily altitude of the site, and manual_alt (columns date, well, altitude_m; only the wells
    with surface altitude). manual: columns date, well, depth_cm; manual_monthly: one column per well on every
    month of the window (day 15); depth in cm, deeper down."""
    wells = list(manual_monthly.columns)
    colors = list(matplotlib.colormaps["tab10" if len(wells) <= 10 else "tab20"].colors)
    color = {well: colors[i % len(colors)] for i, well in enumerate(wells)}
    fig = Figure(figsize=(12, 11), layout="constrained")
    ax_day, ax_month, ax_alt = fig.subplots(3, 1, sharex=True)
    ax_day.plot(observed.index, observed.to_numpy(), c="k", lw=1.4, label=observed_label)
    for well, rows in manual.groupby("well", sort=False):
        ax_day.plot(rows["date"], rows["depth_cm"], ls="", marker="o", ms=7, mec="k", mew=0.5, c=color[well],
                    label=well)
    ax_month.plot(observed_monthly.index, observed_monthly.to_numpy(), c="k", lw=1.4, marker="s", ms=6,
                  label=f"{observed_label}, monthly median")
    for well in wells:
        ax_month.plot(manual_monthly.index, manual_monthly[well].to_numpy(), ls="-", lw=1, marker="o", ms=7,
                      mec="k", mew=0.5, c=color[well], label=well)
    ax_alt.plot(observed_alt.index, observed_alt.to_numpy(), c="k", lw=1.4)
    for well, rows in manual_alt.groupby("well", sort=False):
        ax_alt.plot(rows["date"], rows["altitude_m"], ls="", marker="o", ms=7, mec="k", mew=0.5, c=color[well])
    _depth_axis(ax_day, "depth (cm), daily")
    _depth_axis(ax_month, "depth (cm), monthly (day 15)")
    ax_alt.set_ylabel(ALTITUDE_LABEL)
    for ax in (ax_day, ax_month):
        ax.axhline(0, c="0.5", lw=0.6)
    for ax in (ax_day, ax_month, ax_alt):
        ax.grid(lw=0.4, alpha=0.4)
    ax_alt.set_xlim(*xlim)
    ax_alt.xaxis.set_major_locator(mdates.MonthLocator())
    ax_alt.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    handles, labels = ax_day.get_legend_handles_labels()
    fig.legend(handles, labels, ncols=6, fontsize=8, frameon=False, loc="outside lower center")
    fig.suptitle(title)
    return fig


def precip_accumulation_plot(accumulated, depth, windows, title, depth_label="modeled depth"):
    """One panel per accumulation window (rows, shared x; windows: label -> days, column precip_<label>_mm):
    accumulated rainfall (mm, left axis, light blue area) and, on a right twin axis, the daily depth (cm, deeper
    down): depth["pred"] as a line and, when depth has lo80 and hi80, the 80% interval as a gray band."""
    fig = Figure(figsize=(13, 2.6 * len(windows)), layout="constrained")
    axes = np.atleast_1d(fig.subplots(len(windows), 1, sharex=True, sharey=True))
    band = {"lo80", "hi80"} <= set(depth.columns)
    twins = []
    for ax, (w, d) in zip(axes, windows.items()):
        rain = accumulated[f"precip_{w}_mm"]
        ax.fill_between(rain.index, 0, rain, color="tab:blue", alpha=0.35, lw=0)
        ax.plot(rain, c="tab:blue", lw=0.8)
        ax.set_ylabel(f"{w} rain (mm)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_title(f"{w} = {d} days", fontsize=9, loc="left")
        twin = ax.twinx()
        if twins:   # same depth limits in every panel (shared axis is inverted once, by the first twin)
            twin.sharey(twins[0])
        twins.append(twin)
        if band:
            twin.fill_between(depth.index, depth["lo80"], depth["hi80"], color="0.5", alpha=0.25, lw=0)
        twin.plot(depth["pred"], c=PRED_COLOR, lw=0.9)
        twin.set_ylabel(f"{depth_label} (cm)")
        if len(twins) == 1:
            twin.invert_yaxis()
    span = (accumulated.index.max() - accumulated.index.min()).days
    axes[-1].set_xlim(accumulated.index.min(), max(accumulated.index.max(), depth.index.max()))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator() if span > 4 * 365 else mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y" if span > 4 * 365 else "%Y-%m"))
    handles, labels = [Patch(color="tab:blue", alpha=0.35)], ["accumulated rain"]
    handles.append(Line2D([], [], c=PRED_COLOR, lw=0.9))
    labels.append(depth_label)
    if band:
        handles.append(Patch(color="0.5", alpha=0.25))
        labels.append(f"{depth_label} 80% interval")
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels), frameon=False, fontsize=8)
    fig.suptitle(title)
    return fig


# ---------------------------------------------------------------- SPI vs monthly level z-score (step 10)
SPI_COLOR, Z_COLOR = "#1f77b4", "#d95f02"
DATE_FORMAT = mdates.DateFormatter("%b\n%Y")   # month and year, for the short (months-long) x axes
YEAR_FORMAT = mdates.DateFormatter("%Y")


def _fmt(x, nd=3, sign=False):
    """Number with nd decimals, optionally always signed."""
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _lagged(q, k):
    """Label of the SPI-q of k months earlier."""
    return f"SPI-{q}(t−{k})"


def _year_grid(axes, x_lo, x_hi):
    """Yearly ticks and vertical lines: 1 January (solid, major) and 1 July (dashed, minor)."""
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(1, month=1, day=1))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1, month=7, day=1))
        ax.xaxis.set_major_formatter(YEAR_FORMAT)
        ax.grid(axis="x", which="both", visible=False)
        ax.tick_params(axis="x", which="major", length=6)
        ax.tick_params(axis="x", which="minor", length=3)
        ax.set_xlim(x_lo, x_hi)
        for year in range(x_lo.year, x_hi.year + 1):
            jan, jul = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 7, 1)
            if x_lo <= jan <= x_hi:
                ax.axvline(jan, color="0.35", lw=0.9, zorder=0)
            if x_lo <= jul <= x_hi:
                ax.axvline(jul, color="0.7", lw=0.7, ls="--", zorder=0)


def level_anomaly_figure(an, site, modeled=False):
    """Daily level with its seasonal harmonic curve ± seasonal sigma, daily z and monthly mean z (analyze_anomalies)."""
    table, z, zm, model = an["table"], an["z"], an["zm"], an["model"]
    sigma_label = f"|r|, Kσ={model['k_sigma']}"
    name = "modeled level (pred)" if modeled else "level"
    fig = Figure(figsize=(13 if modeled else 11, 10.2), layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    ax = axes[0]
    ax.plot(table.index, table["level_m"], color="0.55", lw=0.6 if modeled else 0.8, label="Modeled (pred), daily" if modeled else "Daily")
    ax.plot(table.index, table["seasonal_curve_m"], color="C0", lw=1.6, label=f"Seasonal curve (K={model['k']}) ± seasonal σ ({sigma_label})")
    ax.fill_between(table.index, table["seasonal_curve_m"] - table["seasonal_sigma_m"],
                    table["seasonal_curve_m"] + table["seasonal_sigma_m"], color="C0", alpha=0.2, lw=0)
    ax.set_ylabel("Water table level relative to the ground (m)")
    ax.set_title(f"{site}: daily {name} and harmonic seasonal curve")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))   # room for the legend
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    ax = axes[1]
    ax.plot(z.index, z.values, color="0.3", lw=0.5 if modeled else 0.7)
    ax.fill_between(z.index, z.values, 0, where=(z.values < 0), color="C3", alpha=0.4, interpolate=True, label="z < 0")
    ax.fill_between(z.index, z.values, 0, where=(z.values > 0), color="C0", alpha=0.3, interpolate=True, label="z > 0")
    ax.axhline(0, color="k", lw=0.8)
    for v in (-2, -1, 1, 2):
        ax.axhline(v, color="0.4", lw=0.8, ls="--")
    ax.set_ylabel("z (–)")
    ax.set_title(f"Standardized anomaly of the {name} (harmonics K={model['k']}, seasonal σ from {sigma_label})")
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    ax.bar(zm.index, zm.values, width=26 if modeled else 25, color=np.where(zm.values >= 0, "C0", "C3"), alpha=0.7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("z (–)")
    ax.set_title(f"Monthly mean z of the {name} (months with ≥ 80% of the days)")
    ax.set_xlim(table.index.min() - pd.Timedelta(days=10), table.index.max() + pd.Timedelta(days=10))
    if modeled:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(YEAR_FORMAT)
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(DATE_FORMAT)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.supxlabel(f"Anomalies relative to the record {an['first']:%Y-%m-%d} to {an['last']:%Y-%m-%d}; curve and σ fitted on the same "
                  f"data; monthly SD of z between {_fmt(an['sd_z'].min(), 2)} and {_fmt(an['sd_z'].max(), 2)}; "
                  f"effective n ≈ {_fmt(an['n_eff'], 1)}.", fontsize=8, color="0.3")
    return fig


def spi_lag0_figure(spi, zm, qs, lags):
    """Bars of SPI-q(t) next to the monthly z (t), same month (lag 0, descriptive only), one panel per q, with their r."""
    idx = zm.index
    fig = Figure(figsize=(10, 10), layout="constrained")
    axes = fig.subplots(len(qs), 1, sharex=True)
    for ax, q in zip(axes, qs):
        x = spi[f"SPI_{q}"].reindex(idx)
        ok = x.notna() & zm.notna()
        r = float(np.corrcoef(x[ok], zm[ok])[0, 1])
        xn = mdates.date2num(idx.to_pydatetime())
        ax.bar(xn - 6, x.values, width=12, color=SPI_COLOR, alpha=0.8, label=f"SPI-{q}(t)")
        ax.bar(xn + 6, zm.values, width=12, color=Z_COLOR, alpha=0.8, label="monthly z(t)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Standardized value (–)")
        ax.set_title(f"SPI-{q} and z, lag 0: r = {_fmt(r, 3, True)}, n = {int(ok.sum())}")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.2 * (hi - lo))
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlim(idx.min() - pd.Timedelta(days=20), idx.max() + pd.Timedelta(days=20))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(DATE_FORMAT)
    fig.supxlabel(f"Note: lag 0 is descriptive only and is not part of the hypothesis test (which uses k = {', '.join(map(str, lags))} months).",
                  fontsize=9, color="0.3")
    return fig


def correlation_matrix_figure(cor, fam, qs, ks, suptitle=None):
    """Pearson r of the cells SPI-q(t−k) vs z(t) as one heatmap per scenario, with the Gao p of every cell and the family p of
    both surrogate generators (cor: list of correlations() dicts, fam: family_test results)."""
    fig = Figure(figsize=(12, 5.6), layout="constrained")
    axes = fig.subplots(1, len(SCENARIOS))
    im = None
    for ax, scenario, c in zip(axes, SCENARIOS, cor):
        r = c["r"]
        im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for iq in range(len(qs)):
            for ik in range(len(ks)):
                color = "white" if abs(r[iq, ik]) > 0.6 else "black"
                ax.text(ik, iq - 0.08, _fmt(r[iq, ik], 3, True), ha="center", va="center", fontsize=13, fontweight="bold", color=color)
                ax.text(ik, iq + 0.22, f"Gao p = {_fmt(c['p_gao'][iq, ik], 4)}", ha="center", va="center", fontsize=8, color=color)
        iq, ik = np.unravel_index(int(np.argmax(r)), r.shape)
        ax.add_patch(Rectangle((ik - 0.5, iq - 0.5), 1, 1, fill=False, ec="k", lw=3))   # best cell
        ax.set_xticks(range(len(ks)), [f"k = {k}" for k in ks])
        ax.set_yticks(range(len(qs)), [f"SPI-{q}" for q in qs])
        family = "\n".join(f"family p {gen} = {_fmt(fam[(scenario, gen)]['p_family'], 4)} "
                           f"(critical r = {_fmt(fam[(scenario, gen)]['r_crit95'], 3)})" for gen in GENERATORS)
        ax.set_title(f"{scenario} (n = {int(c['n'].max())})\nbest: SPI-{qs[iq]}, k = {ks[ik]}, r = {_fmt(r[iq, ik], 3, True)}\n{family}",
                     fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    return fig


def series_cross_correlation_figure(spi, zm, cor, fam, qs, ks, q, k, excluded_months):
    """Top: monthly z(t) and SPI-q(t−k) in time (with the k months before the window as context). Bottom: r of every cell per
    scenario against the critical r of the family test (both generators) and of a single cell (Gao)."""
    iq, ik = qs.index(q), ks.index(k)
    idx = zm.index
    x_full = spi[f"SPI_{q}"].shift(k)
    x = x_full.reindex(idx)
    context = x_full.reindex(pd.date_range(idx[0] - pd.DateOffset(months=k), idx[0], freq="ME"))
    fig = Figure(figsize=(12, 9.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(context.index, context.values, "--o", color=SPI_COLOR, alpha=0.4, lw=1.1, ms=3.5, mfc="none",
            label=f"{_lagged(q, k)}, context before the window")
    ax.plot(idx, x.values, "-o", color=SPI_COLOR, lw=1.5, ms=4.5, label=_lagged(q, k))
    excluded = np.isin(idx.month, excluded_months)
    ax.plot(idx, zm.values, "-", color=Z_COLOR, lw=1.5)
    ax.plot(idx[~excluded], zm.values[~excluded], "s", color=Z_COLOR, ms=5.5, label="monthly z(t)")
    ax.plot(idx[excluded], zm.values[excluded], "s", mfc="white", mec=Z_COLOR, mew=1.6, ms=7, ls="none",
            label=f"Jan–Mar (dropped in the {SCENARIOS[1]} scenario)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Standardized value (–)")
    (r_a, r_b), (n_a, n_b) = [c["r"][iq, ik] for c in cor], [int(c["n"][iq, ik]) for c in cor]
    ax.set_title(f"z(t) and {_lagged(q, k)}: r = {_fmt(r_a, 3, True)} ({SCENARIOS[0]}, n = {n_a}); "
                 f"r = {_fmt(r_b, 3, True)} ({SCENARIOS[1]}, n = {n_b})", fontsize=11)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.28 * (hi - lo))
    ax.legend(loc="upper center", fontsize=8.5, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xlim(context.index.min() - pd.Timedelta(days=20), idx.max() + pd.Timedelta(days=20))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(DATE_FORMAT)
    q_colors = ["#4c72b0", "#dd8452", "#55a868"]
    styles = {"AR(1)": ("#8c1c13", "--"), "FFT phase": ("#6a3d9a", "-."), "gao": ("0.15", ":")}
    all_r = [c["r"] for c in cor]
    y_max = max(float(np.nanmax(a)) for a in all_r) + 0.1
    y_min = min(0.0, min(float(np.nanmin(a)) for a in all_r) - 0.05)
    axs = []
    for j, (scenario, c) in enumerate(zip(SCENARIOS, cor)):
        ax = fig.add_subplot(gs[1, j], sharey=axs[0] if axs else None)
        axs.append(ax)
        r, w = c["r"], 0.25
        for i, qq in enumerate(qs):
            ax.bar(np.arange(len(ks)) + (i - 1) * w, r[i], width=w, color=q_colors[i], label=f"SPI-{qq}")
        ib, kb = np.unravel_index(int(np.argmax(r)), r.shape)
        ax.add_patch(Rectangle((kb + (ib - 1) * w - w / 2, 0), w, r[ib, kb], fill=False, ec="k", lw=2.4, zorder=5))   # best cell
        lines = [(fam[(scenario, gen)]["r_crit95"], gen, f"critical r, family {gen}") for gen in GENERATORS]
        lines.append((r_critical(c["nu"][ib, kb]), "gao", f"critical r, single cell Gao (SPI-{qs[ib]}, k = {ks[kb]})"))
        for value, key, label in lines:
            ax.axhline(value, color=styles[key][0], ls=styles[key][1], lw=1.6, label=f"{label} = {_fmt(value, 3)}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(ks)), [f"k = {kk}" for kk in ks])
        ax.set_title(f"{scenario} (n = {int(c['n'].max())})", fontsize=11)
        ax.set_ylim(y_min, y_max + 0.3)
        ax.grid(axis="y", alpha=0.3)
        if j == 0:
            ax.set_ylabel("Pearson r, SPI-q(t−k) vs z(t)")
        else:
            ax.tick_params(labelleft=False)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    fig.supxlabel("Black outline: best cell (maximum r). Lines are one-sided 5% thresholds: family = 95th percentile of the maximum r "
                  "over the 9 cells in the surrogates; single cell = Student t with the Gao ν.", fontsize=8, color="0.3")
    return fig


def surrogate_process_figure(sx, yw, dates, keep, cor_obs, fam, surrogates, scenario, qs, ks, n_examples=15, n_mini=4):
    """Process of the family test in one scenario (AR(1) left, FFT phase right): the real z and some surrogates, the 9-cell
    correlation map of the data and of the first surrogates, and the histogram of the maxima with p and critical r."""
    n_surr = len(next(iter(surrogates.values())))
    r1 = float(np.clip(pd.Series(yw).autocorr(1), -0.99, 0.99))
    amplitude = np.abs(np.fft.rfft(yw - yw.mean()))
    period = len(yw) / (int(np.argmax(amplitude[1:])) + 1)   # period (months) of the dominant oscillation
    titles = {"AR(1)": f"AR(1): persistence of 1 month (r1 = {_fmt(r1, 2, True)})",
              "FFT phase": f"FFT phase: same periodogram as z (oscillation of ~{_fmt(period, 1)} months)"}
    names = {"AR(1)": "AR(1) generator", "FFT phase": "Random-phase generator (FFT)"}
    fig = Figure(figsize=(14, 15.5))   # manual margins: the section headings below are placed on the grid rows
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.55, 1.2], hspace=0.42, wspace=0.18, left=0.06, right=0.97, top=0.9, bottom=0.07)
    obs = cor_obs["r"]
    ib, kb = np.unravel_index(int(np.argmax(obs)), obs.shape)

    def mini(ax, r, title):   # 3x3 map of r with the best cell outlined
        ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        iq, ik = np.unravel_index(int(np.argmax(r)), r.shape)
        ax.add_patch(Rectangle((ik - 0.5, iq - 0.5), 1, 1, fill=False, ec="k", lw=2.2))
        ax.text(ik, iq, _fmt(r[iq, ik], 2, True), ha="center", va="center", fontsize=6.5, fontweight="bold",
                color="white" if abs(r[iq, ik]) > 0.6 else "black")
        ax.set_xticks(range(len(ks)), [str(k) for k in ks], fontsize=7)
        ax.set_yticks(range(len(qs)), [str(q) for q in qs], fontsize=7)
        ax.set_title(title, fontsize=8.5)

    for j, gen in enumerate(GENERATORS):
        sur, f = surrogates[gen], fam[(scenario, gen)]
        ax = fig.add_subplot(gs[0, j])   # row 1: real z and surrogates
        for m in range(len(dates)):
            if not keep[m]:
                ax.axvspan(dates[m] - pd.Timedelta(days=15), dates[m] + pd.Timedelta(days=15), color="0.85", alpha=0.7, lw=0, zorder=0)
        for i in range(n_examples):
            ax.plot(dates, sur[i], color="0.45", lw=0.8, alpha=0.4, zorder=1)
        ax.plot(dates, yw, "-o", color="k", lw=2.6, ms=4.5, zorder=3, label="real monthly z")
        ax.plot([], [], color="0.45", lw=0.8, label=f"{n_examples} synthetic series (iterations 1 to {n_examples})")
        ax.add_patch(Rectangle((0, 0), 0, 0, color="0.85", label="Jan–Mar (dropped in this scenario)"))
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_title(titles[gen], fontsize=11)
        ax.set_ylabel("z (–)")
        ax.xaxis.set_major_formatter(DATE_FORMAT)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))
        ax.legend(loc="upper right", fontsize=7.5)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.annotate("Synthetic series", xy=(0, 1.12), xycoords="axes fraction", fontsize=12, fontweight="bold", annotation_clip=False)
        sub = gs[1, j].subgridspec(1, n_mini + 1, wspace=0.12)   # row 2: what is stored of each iteration
        ax0 = fig.add_subplot(sub[0, 0])
        mini(ax0, obs, f"observed\nmax r = {_fmt(obs[ib, kb], 3, True)}")
        ax0.set_ylabel("SPI-q", fontsize=8)
        ax0.set_xlabel("k", fontsize=8)
        for i in range(n_mini):
            r_i, _ = corr_grid(sx, np.where(keep, sur[i], np.nan))
            a = fig.add_subplot(sub[0, i + 1])
            mini(a, r_i, f"iteration {i + 1}\nmax r = {_fmt(float(np.nanmax(r_i)), 3, True)}")
            a.set_yticks([])
            a.set_xlabel("k", fontsize=8)
        if j == 0:
            fig.text(0.06, gs[1, 0].get_position(fig).y1 + 0.005, "What is stored from each iteration: Pearson r (SPI-q × k) and its maximum",
                     fontsize=12, fontweight="bold")
        ax = fig.add_subplot(gs[2, j])   # row 3: distribution of the maxima
        mx, r_obs = f["max_s"], f["r_max"]
        ax.hist(mx, bins=60, color="0.6", edgecolor="white", lw=0.3)
        ax.axvline(r_obs, color="C3", lw=2, label=f"observed = {_fmt(r_obs, 3, True)}")
        ax.axvline(f["r_crit95"], color="C0", ls="--", lw=1.8, label=f"critical r 95% = {_fmt(f['r_crit95'], 3)}")
        ax.axvspan(r_obs, max(mx.max(), r_obs) + 0.02, color="C3", alpha=0.25, label=f"family p = {_fmt(f['p_family'], 4)}")
        ax.set_xlabel("maximum r over the 9 combinations, per iteration")
        ax.set_ylabel("number of iterations")
        ax.set_title(f"{names[gen]}: result of {n_surr:,} iterations", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        inset = ax.inset_axes([0.07, 0.64, 0.28, 0.31])   # running p against the number of iterations
        n_it = np.unique(np.round(np.logspace(0, np.log10(n_surr), 200)).astype(int))
        inset.plot(n_it, (1 + np.cumsum(mx >= r_obs - 1e-12)[n_it - 1]) / (1 + n_it), color="C3", lw=1.2)
        inset.axhline(f["p_family"], color="k", ls="--", lw=1)
        inset.set_xscale("log")
        inset.set_ylim(0, 0.5)
        inset.set_title("running p vs iterations", fontsize=7.5)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=0.3)
        if j == 0:
            fig.text(0.06, gs[2, 0].get_position(fig).y1 + 0.032, f"Result of {n_surr:,} iterations", fontsize=12, fontweight="bold")
    fig.suptitle(f"Family test ({scenario}): the whole search is repeated on synthetic series of z and the observed maximum is compared "
                 "with the distribution of the maxima", fontsize=12, y=0.965)
    fig.text(0.5, 0.012, "Each iteration: synthetic z → drop Jan–Mar → Pearson r in the 9 combinations → keep the maximum. "
             "p = fraction of maxima ≥ observed.", ha="center", fontsize=10, color="0.25")
    return fig


def slope_scatter_figure(slopes, q, k, excluded_months, title=None, small_markers=False):
    """Top: z(t) vs SPI-q(t−k) with the OLS and RMA lines per scenario (labels: 3 months with the largest Cook's distance;
    no band for the OLS interval, the numbers are in the text box). Bottom: OLS and RMA slopes after leaving each month out."""
    both = np.concatenate([np.r_[p["x"], p["y"]] for p in slopes.values()])
    lim = (float(both.min()) - 0.3, float(both.max()) + 0.3)
    ms_point, ms_loo = (3.5, 3.0) if small_markers else (6, 6)
    fig = Figure(figsize=(13, 12.4), layout="constrained")
    axes = fig.subplots(2, 2, gridspec_kw={"height_ratios": [1.45, 1.0]})
    c_ols, c_rma = "#1f77b4", "#d95f02"
    loo_all = np.concatenate([np.r_[p["loo"][:, 0], p["loo"][:, 1], p["b_ols"], p["b_rma"]] for p in slopes.values()])
    pad = 0.08 * float(np.ptp(loo_all))
    ylim_loo = (float(loo_all.min()) - pad, float(loo_all.max()) + pad)   # shared by the bottom panels
    for j, (scenario, p) in enumerate(slopes.items()):
        ax = axes[0, j]
        x, y, dates = p["x"], p["y"], p["dates"]
        xs = np.linspace(lim[0], lim[1], 100)
        ax.plot(xs, p["a_ols"] + p["b_ols"] * xs, color=c_ols, lw=2, label="OLS")
        ax.plot(xs, p["a_rma"] + p["b_rma"] * xs, color=c_rma, lw=2, ls="--", label="RMA")
        ax.plot(xs, xs, color="0.5", lw=1, ls=":", label="1:1")
        excluded = np.isin(dates.month, excluded_months)
        if j == 0:   # all months: Jan–Mar as open circles
            ax.plot(x[~excluded], y[~excluded], "o", color="k", ms=ms_point, label="other months", zorder=4)
            ax.plot(x[excluded], y[excluded], "o", mfc="white", mec="k", mew=1.0 if small_markers else 1.5, ms=ms_point + 1, label="Jan–Mar", zorder=4)
        else:
            ax.plot(x, y, "o", color="k", ms=ms_point, label="data", zorder=4)
        threshold = 4 / p["n"]
        for i in np.argsort(p["cook"])[::-1][:3]:
            right = x[i] > lim[1] - 1.0
            star = "*" if p["cook"][i] > threshold else ""
            ax.annotate(f"{dates[i]:%Y-%m} (D = {_fmt(p['cook'][i], 2)}){star}", (x[i], y[i]), xytext=(-6 if right else 5, 6),
                        textcoords="offset points", fontsize=8, ha="right" if right else "left")
        ax.plot([], [], " ", label="labels: 3 months with the largest Cook's distance (D)")
        ax.plot([], [], " ", label=f"* D > 4/n = {_fmt(threshold, 3)}")
        ax.margins(0)   # the lines span lim in both axes, so the data limits are lim
        ax.set_aspect("equal", adjustable="datalim")   # same scale on both axes (the limits may grow, the box does not shrink)
        ax.set_xlabel(_lagged(q, k))
        ax.set_ylabel("z(t)")
        ax.set_title(f"{scenario} (n = {p['n']})", fontsize=11)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.98, f"r = {_fmt(p['r'], 3, True)}; Gao ν = {_fmt(p['nu'], 1)}\n"
                f"OLS slope = {_fmt(p['b_ols'], 3, True)}\n  [95% CI: {_fmt(p['ci_ols'][0], 3, True)} ; {_fmt(p['ci_ols'][1], 3, True)}]\n"
                f"RMA slope = {_fmt(p['b_rma'], 3, True)}\n  [95% CI: {_fmt(p['ci_rma'][0], 3, True)} ; {_fmt(p['ci_rma'][1], 3, True)}]\n"
                f"leaving one point out: OLS {_fmt(p['loo'][:, 0].min(), 3, True)} to {_fmt(p['loo'][:, 0].max(), 3, True)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.92), zorder=6)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.5, ncol=3, framealpha=0.92)
        ax = axes[1, j]   # slope after leaving each month out
        ax.axhline(p["b_ols"], color=c_ols, lw=1.5, label=f"full OLS = {_fmt(p['b_ols'], 3, True)}")
        ax.axhline(p["b_rma"], color=c_rma, lw=1.5, ls="--", label=f"full RMA = {_fmt(p['b_rma'], 3, True)}")
        ax.plot(dates, p["loo"][:, 0], "o", color=c_ols, ms=ms_loo, label="OLS without that month")
        ax.plot(dates, p["loo"][:, 1], "D", color=c_rma, ms=ms_loo - 1 if small_markers else 5, label="RMA without that month")
        ax.set_title("Slope when one month is left out", fontsize=11)
        ax.set_ylabel("slope (z per unit of SPI)")
        if small_markers:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(YEAR_FORMAT)
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(DATE_FORMAT)
        ax.grid(alpha=0.3)
        ax.set_ylim(ylim_loo)
        if j == 1:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=7.5, ncol=3, framealpha=0.92)
        ax.set_xlabel("month (t) left out")
    fig.supxlabel("Slope = change of z per unit of SPI. 1:1 = proportional amplitudes; < 1 = the level damps the signal. "
                  "RMA treats both series symmetrically.", fontsize=9, color="0.3")
    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def rain_level_figure(accumulated, level, x_range, lag_months, kind, level_note):
    """CHIRPS rain accumulated backwards over 15 and 90 days (left axis, mm, from 0) and the raw water table level (right
    axis, m). Top: no lag; bottom: rain lagged by lag_months (value at t = rain at t − lag_months). accumulated: precip_15d_mm
    and precip_90d_mm; level: daily series (m, negative below the ground); kind: "modeled" or "observed"."""
    x_lo, x_hi = x_range

    def clip(s):
        return s[(s.index >= x_lo) & (s.index <= x_hi)]

    def lagged(s):   # new index = index + lag_months (whole record)
        shifted = s.copy()
        shifted.index = shifted.index + pd.DateOffset(months=lag_months)
        return shifted[~shifted.index.duplicated(keep="first")]   # 28–31 Aug -> 28 Feb: the 28 Aug is kept

    acc = {name: accumulated[f"precip_{name}_mm"] for name in ("15d", "90d")}
    now = {name: clip(s) for name, s in acc.items()}
    late = {name: clip(lagged(s)) for name, s in acc.items()}
    level = clip(level)
    top = float(max(s.max() for s in (*now.values(), *late.values())))
    margin = 0.22   # headroom for the legend, in both axes
    lo, hi = float(level.min()), float(level.max())
    pad = 0.05 * (hi - lo)
    level_lim = (lo - pad, lo - pad + ((hi + pad) - (lo - pad)) * (1 + margin))
    c15, c90, c_level = "#1f77b4", "#1b9e77", "#5b2c06"
    fig = Figure(figsize=(30, 12), layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    for ax, series, lagged_rain in ((axes[0], now, False), (axes[1], late, True)):
        twin = ax.twinx()
        ax.set_zorder(twin.get_zorder() + 2)
        ax.patch.set_visible(False)
        a90, a15 = (1.0, 0.15) if lagged_rain else (0.15, 0.8)   # the series that is not the focus of the panel is faint
        ax.plot(series["90d"].index, series["90d"].values, "-", color=c90, lw=1.6, alpha=a90, zorder=3)
        ax.plot(series["15d"].index, series["15d"].values, "-", color=c15, lw=0.8, alpha=a15, zorder=4)
        twin.plot(level.index, level.values, "-", color=c_level, lw=1.0, zorder=5)
        ax.set_ylim(0, top * (1 + margin))
        twin.set_ylim(*level_lim)
        ax.set_ylabel("Accumulated rain (mm)")
        twin.set_ylabel(f"{kind.capitalize()} water table level (m, relative to the ground)", color=c_level)
        twin.tick_params(axis="y", colors=c_level)
        ax.grid(axis="y", alpha=0.3)
        _year_grid([ax], x_lo, x_hi)
        ax.tick_params(axis="x", labelbottom=True)
        suffix = f" lagged {lag_months} months" if lagged_rain else ""
        ax.legend(handles=[Line2D([], [], color=c15, lw=1.0, alpha=a15, label=f"15-day rolling accumulation{suffix} (mm)"),
                           Line2D([], [], color=c90, lw=2.0, alpha=a90, label=f"90-day (≈ 3 months) rolling accumulation{suffix} (mm)"),
                           Line2D([], [], color=c_level, lw=1.0, label=f"{kind.capitalize()} water table level (m, right axis)")],
                  loc="upper center", fontsize=10, ncol=3)
        ax.set_title(f"Rain lagged {lag_months} months (value at t = rain at t − {lag_months} months) and {kind} water table level"
                     if lagged_rain else f"Rain without lag and {kind} water table level", fontsize=13)
    fig.suptitle(f"CHIRPS rain (15- and 90-day accumulations) and {kind} water table level, {x_lo.year}–{x_hi.year}", fontsize=15)
    fig.supxlabel("Backward rolling accumulations (sum of the days ending at t). Bottom panel: rain lagged "
                  f"{lag_months} months (value at t = rain at t − {lag_months} months).\n{level_note} no lag. "
                  "Same rain limits in both panels.", fontsize=10, color="0.2", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    return fig


def monthly_series_figure(spi_q, zm, x, yw, dates, cor, fam, lag0, q, k, excluded_months):
    """Modeled monthly z with SPI-q(t) (top, lag 0, descriptive, r and n only) and SPI-q(t−k) (bottom, with r, n and the p of
    Gao and of the single-cell surrogate tests, per scenario). x, yw, dates: the lagged SPI and z of the contiguous window;
    lag0: lag0_correlations()."""
    z, spi_now = pd.Series(yw, index=dates), spi_q.reindex(zm.index)
    ok = spi_now.notna() & zm.notna()
    spi_now, z_now = spi_now[ok], zm[ok]
    fig = Figure(figsize=(30, 10), layout="constrained")
    ax, ax2 = fig.subplots(2, 1, sharex=True, sharey=True)

    def series(a, xs, zs, lag_text):
        excluded = np.isin(zs.index.month, excluded_months)
        a.plot(xs.index, xs.values, "-o", color=SPI_COLOR, lw=0.9, ms=2.5, label=f"SPI-{q}({lag_text})")
        a.plot(zs.index, zs.values, "-", color=Z_COLOR, lw=0.9)
        a.plot(zs.index[~excluded], zs.values[~excluded], "s", color=Z_COLOR, ms=3.2, label="modeled z(t)")
        a.plot(zs.index[excluded], zs.values[excluded], "s", mfc="white", mec=Z_COLOR, mew=0.9, ms=4.2, ls="none",
               label=f"modeled z(t) in Jan–Mar (dropped in the {SCENARIOS[1]} scenario)")
        a.axhline(0, color="k", lw=0.7)
        a.set_ylabel("Standardized value (–)")
        a.grid(axis="y", alpha=0.3)

    series(ax, spi_now, z_now, "t")
    series(ax2, pd.Series(x, index=dates), z, f"t−{k}")
    lo, hi = min(ax.get_ylim()[0], ax2.get_ylim()[0]), max(ax.get_ylim()[1], ax2.get_ylim()[1])
    ax.set_ylim(lo, hi + 0.2 * (hi - lo))
    for a in (ax, ax2):
        a.legend(loc="upper center", fontsize=9, ncol=3)
    _year_grid([ax, ax2], min(dates.min(), z_now.index.min()) - pd.Timedelta(days=30), max(dates.max(), z_now.index.max()) + pd.Timedelta(days=30))
    ax.set_title(f"modeled z(t) and SPI-{q}(t), lag 0 (descriptive): " + ";  ".join(
        f"r = {_fmt(l['r'], 3, True)}, n = {l['n']} ({s})" for l, s in zip(lag0, SCENARIOS)), fontsize=12)
    ax2.set_title(f"modeled z(t) and {_lagged(q, k)}", fontsize=12)
    fig.suptitle(f"Bottom panel, {_lagged(q, k)}: " + ";  ".join(
        f"r = {_fmt(c['r'], 3, True)}, n = {c['n']}, Gao p = {_fmt(c['p_gao'], 4)} ({s})" for c, s in zip(cor, SCENARIOS)), fontsize=13)
    fig.supxlabel(_single_cell_note(cor, fam, q, k) + "\nOne-sided test (r > 0), only for the bottom panel. Gao: ν = n·δ − 2; "
                  f"AR(1) and FFT: {len(fam[(SCENARIOS[0], 'AR(1)')]['r_s']):,} surrogates, statistic = r of a single cell. "
                  "Lag 0 is descriptive and has no p-value.", fontsize=10, color="0.2", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    return fig


def _single_cell_note(cor, fam, q, k):
    """One line per scenario with r, n and the one-sided p of the Gao and surrogate tests of the single cell."""
    return "\n".join(f"Bottom panel, {_lagged(q, k)}, {scenario}: r = {_fmt(c['r'], 3, True)}, n = {c['n']}; one-sided p: "
                     f"Gao = {_fmt(c['p_gao'], 4)}, AR(1) = {_fmt(fam[(scenario, 'AR(1)')]['p'], 4)}, "
                     f"FFT = {_fmt(fam[(scenario, 'FFT phase')]['p'], 4)}" for scenario, c in zip(SCENARIOS, cor))


def daily_series_figure(spi_q, daily_z, cor, fam, lag0, q, k):
    """Like monthly_series_figure but with the daily z as a line and the monthly SPI-q as bars (top: SPI(t); bottom: SPI of t−k
    placed on month t); Jan–Mar shaded. r, n and p are those of the monthly z."""
    z = daily_z.dropna()
    spi_now = spi_q.dropna()
    spi_now.index = spi_now.index.to_period("M").to_timestamp()   # first day of each month
    spi_lag = spi_q.shift(k).dropna()
    spi_lag.index = spi_lag.index.to_period("M").to_timestamp()
    x_lo, x_hi = pd.Timestamp(z.index.min().year, 1, 1), pd.Timestamp(z.index.max().year, 11, 30)
    spi_now, spi_lag = spi_now[(spi_now.index >= x_lo) & (spi_now.index <= x_hi)], spi_lag[(spi_lag.index >= x_lo) & (spi_lag.index <= x_hi)]
    fig = Figure(figsize=(30, 10), layout="constrained")
    ax, ax2 = fig.subplots(2, 1, sharex=True, sharey=True)
    for a, bars in ((ax, spi_now), (ax2, spi_lag)):
        a.bar(bars.index, bars.values, width=bars.index.days_in_month.to_numpy(), align="edge", color=SPI_COLOR, alpha=0.5, lw=0, zorder=1)
        a.plot(z.index, z.values, "-", color=Z_COLOR, lw=0.5, zorder=3)
        a.axhline(0, color="k", lw=0.7, zorder=2)
        for year in range(x_lo.year, x_hi.year + 1):
            a.axvspan(pd.Timestamp(year, 1, 1), pd.Timestamp(year, 3, 31) + pd.Timedelta(days=1), color="0.5", alpha=0.08, lw=0, zorder=0)
        a.set_ylabel("Standardized value (–)")
        a.grid(axis="y", alpha=0.3)
    lo, hi = min(ax.get_ylim()[0], ax2.get_ylim()[0]), max(ax.get_ylim()[1], ax2.get_ylim()[1])
    ax.set_ylim(lo, hi + 0.2 * (hi - lo))
    for a, lag_text in ((ax, "t"), (ax2, f"t−{k}")):
        a.legend(handles=[Patch(fc=SPI_COLOR, alpha=0.5, label=f"SPI-{q}({lag_text}), monthly bars"),
                          Line2D([], [], color=Z_COLOR, lw=1.0, label="daily modeled z"),
                          Patch(fc="0.5", alpha=0.15, label=f"Jan–Mar (dropped in the {SCENARIOS[1]} scenario)")],
                 loc="upper center", fontsize=9, ncol=3)
    _year_grid([ax, ax2], x_lo, x_hi)
    ax.set_title(f"daily modeled z and SPI-{q}(t), lag 0 (descriptive): " + ";  ".join(
        f"r = {_fmt(l['r'], 3, True)}, n = {l['n']} ({s})" for l, s in zip(lag0, SCENARIOS)), fontsize=12)
    ax2.set_title(f"daily modeled z and {_lagged(q, k)}", fontsize=12)
    fig.suptitle(f"Bottom panel, {_lagged(q, k)}: " + ";  ".join(
        f"r = {_fmt(c['r'], 3, True)}, n = {c['n']}, Gao p = {_fmt(c['p_gao'], 4)} ({s})" for c, s in zip(cor, SCENARIOS)), fontsize=13)
    fig.supxlabel(_single_cell_note(cor, fam, q, k) + "\nOne-sided test (r > 0), only for the bottom panel. Gao: ν = n·δ − 2; "
                  f"AR(1) and FFT: {len(fam[(SCENARIOS[0], 'AR(1)')]['r_s']):,} surrogates, statistic = r of a single cell. "
                  "Lag 0 is descriptive and has no p-value.\nCorrelations use the monthly z; the daily line is only visual.",
                  fontsize=10, color="0.2", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    return fig


def single_cell_surrogate_figure(yw, dates, cor, fam, surrogates, q, k, n_examples=10, n_last=60):
    """Single-cell surrogate test: real z and some surrogates (last n_last months) and the histogram of the surrogate r with the
    observed r, the critical r and the p, per generator (columns) and scenario (rows)."""
    n_surr = len(next(iter(surrogates.values())))
    names = {"AR(1)": "AR(1)", "FFT phase": "Random phase (FFT)"}
    all_r = np.concatenate([fam[key]["r_s"] for key in fam] + [[fam[key]["r_obs"] for key in fam]])
    xlim = (float(np.nanmin(all_r)) - 0.03, float(np.nanmax(all_r)) + 0.05)
    fig = Figure(figsize=(13, 13.2), layout="constrained")
    axes = fig.subplots(3, 2)
    for j, gen in enumerate(GENERATORS):
        ax = axes[0, j]
        for i in range(n_examples):
            ax.plot(dates[-n_last:], surrogates[gen][i, -n_last:], color="0.55", lw=0.8, alpha=0.6, zorder=1)
        ax.plot([], [], color="0.55", lw=0.8, label=f"{n_examples} synthetic series (iterations 1 to {n_examples})")
        ax.plot(dates[-n_last:], yw[-n_last:], "-o", color="k", lw=1.8, ms=3, zorder=3, label="real modeled z")
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_title(f"{names[gen]}: real and synthetic z, last {n_last} of {len(yw)} months", fontsize=10.5)
        ax.set_ylabel("z (–)")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.28 * (hi - lo))
        ax.legend(loc="upper center", fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(DATE_FORMAT)
        for i, scenario in enumerate(SCENARIOS):
            f = fam[(scenario, gen)]
            ax = axes[1 + i, j]
            ax.hist(f["r_s"][np.isfinite(f["r_s"])], bins=60, range=xlim, color="0.65", edgecolor="white", lw=0.3)
            ax.set_xlim(xlim)
            ax.axvspan(f["r_obs"], xlim[1], color="C3", alpha=0.22, label=f"tail: p = {_fmt(f['p'], 4)}")
            ax.axvline(f["r_obs"], color="C3", lw=2, label=f"observed r = {_fmt(f['r_obs'], 3, True)}")
            ax.axvline(f["r_crit95"], color="C0", ls="--", lw=1.8, label=f"critical r 95% = {_fmt(f['r_crit95'], 3)}")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
            ax.set_xlabel(f"Pearson r, {_lagged(q, k)} vs synthetic z")
            ax.set_ylabel("number of iterations")
            ax.set_title(f"{names[gen]}, {scenario} (n = {cor[i]['n']}): {n_surr:,} iterations", fontsize=10.5)
            ax.legend(loc="upper right", fontsize=8.5)
            ax.grid(alpha=0.3)
    fig.suptitle(f"Surrogate test, single cell SPI-{q}, k = {k} (modeled level)", fontsize=12.5)
    fig.supxlabel(f"Each iteration: synthetic z → (drop Jan–Mar) → Pearson r with {_lagged(q, k)}. p = fraction of synthetic r ≥ observed.",
                  fontsize=10, color="0.25")
    return fig


def scatter_grid_figure(cells, r2, qs, ks, excluded_months, title):
    """z(t) vs SPI-q(t−k) for every cell (rows q, columns k) with the OLS R² of both scenarios and the sign of the slope (no
    lines); open points: months t in the excluded months. cells: scatter_pairs(); r2: scatter_r2()."""
    biggest = max(float(np.max(np.abs(np.concatenate([v[0], v[1]])))) for v in cells.values())
    lim = float(np.ceil(biggest * 1.06 * 4) / 4)
    fig = Figure(figsize=(22, 15), layout="constrained")
    axes = fig.subplots(len(qs), len(ks), sharex=True, sharey=True)
    n_typical = [r2[(qs[0], ks[0], s)][0] for s in SCENARIOS]
    for i, q in enumerate(qs):
        for j, k in enumerate(ks):
            ax = axes[i, j]
            x, y, months = cells[(q, k)]
            hole = np.isin(months.month, excluded_months)
            ax.axhline(0, color="0.4", lw=0.6, zorder=0)
            ax.axvline(0, color="0.4", lw=0.6, zorder=0)
            ax.scatter(x[~hole], y[~hole], s=11, color=SPI_COLOR, alpha=0.6, lw=0, zorder=2)
            ax.scatter(x[hole], y[hole], s=11, facecolors="none", edgecolors=SPI_COLOR, alpha=0.6, linewidths=0.8, zorder=2)
            lines = [f"R² {name} = {_fmt(r2[(q, k, s)][1], 3)} ({slope_sign(r2[(q, k, s)][2])})"
                     for s, name in zip(SCENARIOS, ("all", "no Jan–Mar"))]
            if [r2[(q, k, s)][0] for s in SCENARIOS] != n_typical:
                lines.append("n = " + " and ".join(str(r2[(q, k, s)][0]) for s in SCENARIOS))
            ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=10, color="0.15", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
            ax.grid(alpha=0.25)
            if i == 0:
                ax.set_title("k = 0" if k == 0 else f"k = {k} months", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"SPI-{q}\nmodeled z(t)", fontsize=11, fontweight="bold")
            if i == len(qs) - 1:
                ax.set_xlabel("SPI-q(t−k)", fontsize=11)
    axes[0, 0].set_xlim(-lim, lim)
    axes[0, 0].set_ylim(-lim, lim)
    fig.suptitle(title, fontsize=16)
    fig.supxlabel(f"n = {n_typical[0]} (all) and {n_typical[1]} (no Jan–Mar) in every panel. OLS R² of z(t) on SPI-q(t−k), with intercept; "
                  "(+)/(−) = sign of the slope.\nOpen points = level months in Jan–Mar, filled points = other months. SPI clipped to ±3.", fontsize=11, color="0.3")
    return fig
