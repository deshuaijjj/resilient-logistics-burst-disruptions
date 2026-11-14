"""Draw Fig. 1 of the article (which maximum-duration outage contains a given
outage, in the two-stage and in the multi-stage problem). Schematic, no data.

Run from anywhere:  python3 checks/optimization_letters/make_figure1.py
Writes Fig1.pdf and Fig1.png next to this file.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _repo_root(start):
    """The repository root: the first directory above this file that holds src/ and results/."""
    d = start
    while True:
        if os.path.isdir(os.path.join(d, "src")) and os.path.isdir(os.path.join(d, "results")):
            return d
        up = os.path.dirname(d)
        if up == d:
            raise RuntimeError("repository root (with src/ and results/) not found above " + start)
        d = up


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _repo_root(HERE)

MM = 1 / 25.4
WIDTH = 119 * MM
BLUE, ORANGE = "#2a78d6", "#eb6834"      # validated pair (dataviz palette)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def fig_cover(name="Fig1"):
    """DRAFT for skim path version 3 (2026-11-11, redrawn after the cold read):
    which longer outage stands in for a given outage, in the two-stage and in
    the multi-stage problem. Schematic, no data. One line with a maximum-duration
    outage of six months and a horizon of 24 months, as lines 1 and 2 of the
    lunar example."""
    T = 24
    GREY = "#8a8985"
    # (main label, second label, months out, colour, hatch, cells after the horizon, height of the row)
    rows = [("An outage of months 5 to 7 (maximum duration: 6 months)", None, range(5, 8), GREY, "....", 0, 2.35),
            ("is contained in the maximum-duration outage with the same first month",
             "in both problems the operator can plan for it from month 5 on", range(5, 11), BLUE, "////", 0, 3.2),
            (None, None, None, None, None, 0, 0.9),
            ("An outage of months 22 to 24", None, range(22, 25), GREY, "....", 0, 2.35),
            ("Two-stage: in the maximum-duration outage that ends in month 24",
             "planning for it means acting from month 19 on", range(19, 25), BLUE, "////", 0, 3.2),
            ("Multi-stage: in the maximum-duration outage with the same first month",
             "in month 19 the operator has not yet seen which line will fail", range(22, 25), ORANGE,
             "\\\\\\\\", 3, 3.2)]
    top = sum(r[-1] for r in rows)
    fig, ax = plt.subplots(figsize=(WIDTH, 72 * MM))
    y = top
    for label, second, out, color, hatch, ghost, height in rows:
        y -= height
        if label is None:
            continue
        ax.text(0.55, y + (2.45 if second else 1.6), label, ha="left", va="center")
        if second:
            ax.text(0.55, y + 1.62, second, ha="left", va="center", color=MUTED)
        for m in range(1, T + 1):
            on = m in out
            ax.add_patch(Rectangle((m - 0.5 + 0.05, y + 0.05), 0.9, 0.9, facecolor=color if on else "white",
                                   edgecolor="white" if on else GRID, hatch=hatch if on else None, lw=0.5, zorder=2))
            if on:
                ax.add_patch(Rectangle((m - 0.5 + 0.05, y + 0.05), 0.9, 0.9, facecolor="none", edgecolor=color,
                                       lw=0.6, zorder=3))
        for g in range(1, ghost + 1):                      # the part after the horizon, outside the model
            ax.add_patch(Rectangle((T + g - 0.5 + 0.05, y + 0.05), 0.9, 0.9, facecolor="none", edgecolor=color,
                                   lw=0.7, ls=(0, (2, 2)), zorder=2))
    for m in (1, 5, 10, 15, 19, 22, 24):
        ax.text(m, -0.35, str(m), ha="center", va="top", color=MUTED)
    ax.text(0.3, -0.35, "Month", ha="right", va="top", color=MUTED)
    ax.plot([T + 0.5, T + 0.5], [-0.25, top - 1.0], color=INK, lw=0.8, zorder=4)
    ax.text(T + 0.75, -0.35, "end of the\nhorizon", ha="left", va="top", color=MUTED, linespacing=1.05)
    ax.set_xlim(-3.2, T + 4.6)
    ax.set_ylim(-2.3, top + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_cover()
    print("wrote Fig1.pdf and Fig1.png to", HERE)
