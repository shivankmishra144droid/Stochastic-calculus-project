"""Shared figure style (PROJECT_PLAN.md §7) and a save helper."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

GREEN, SLATE, GREY = config.GREEN, config.SLATE, config.GREY
PALETTE = [GREEN, SLATE, "#B45309", "#2563EB", GREY]

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight",
})


def savefig(fig, name: str) -> str:
    """Save to results/figures/<name>.pdf and close the figure; returns the path."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def save_table(df, name: str, index: bool = False) -> str:
    """Save a DataFrame to results/tables/<name>.csv; returns the path."""
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    return str(path)
