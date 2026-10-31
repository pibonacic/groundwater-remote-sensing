"""One function per figure. Each returns a matplotlib Figure and saves nothing."""
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


def oof_plot(y, oof, title):
    """Observed water level and out-of-fold predictions of each model, one panel per scheme."""
    fig = Figure(figsize=(11, 3 * len(oof)), layout="constrained")
    axes = fig.subplots(len(oof), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (scheme, by_model) in zip(axes, oof.items()):
        ax.plot(y, c="k", lw=1, label="observed")
        for model, pred in by_model.items():
            ax.plot(pred, lw=1, label=model)
        ax.set_ylabel(f"{scheme}: water level (cm)")
    axes[0].legend(ncols=4, loc="upper right")
    fig.suptitle(title)
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
