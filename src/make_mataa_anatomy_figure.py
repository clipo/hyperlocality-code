"""Schematic of a mata'a labeled with the dimensions of stylistic variability scored.

Plots a real provenanced 200-point mata'a outline (oriented stem-down/blade-up by
morphometrics.canonical) and annotates the dimensions the paradigmatic analysis
scores: stem length and stem width (the heritable, load-bearing stem proportions),
stem shape and shoulder shape, against the idiosyncratic whole-blade outline that
serves as the scale-free control. Mirrors src/make_moai_anatomy_figure.py.

Illustrative. DEM-free, system python3.
Run:  python3 src/make_mataa_anatomy_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import mataa_io
import morphometrics as mm

WS = "data/xyfinaldatawithids-smallassemblagesremoved.txt"
EXAMPLE_ID = 1616          # provenanced Ahu Tautira piece, classic broad blade / narrow stem
OBSIDIAN = "#34323a"
EDGE = "#111114"
ACCENT = "#7a2c2c"


def _callout(ax, name, detail, xy, xytext, ha="left"):
    ax.annotate(f"{name}\n{detail}", xy=xy, xytext=xytext, ha=ha, va="center",
                fontsize=8.4, zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=0.8),
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.3,
                                connectionstyle="arc3,rad=0.12"))


def _dim_arrow(ax, p0, p1, label, lab_xy, ha="center"):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color="#1f3b6e", lw=1.6))
    ax.annotate(label, lab_xy, ha=ha, va="center", fontsize=8.6, color="#1f3b6e",
                fontweight="bold")


def main(path="figures/mataa_dimensions.png"):
    meta, outlines = mataa_io.load_ws(WS)
    idx = meta.index[meta["ID"].astype(str) == str(EXAMPLE_ID)][0]
    c, _ = mm.canonical(outlines[idx])

    base_y, top_y = c[:, 1].min(), c[:, 1].max()
    cen, w = mm.width_profile(c)
    sy = mm.shoulder_y(cen, w)                       # stem<->blade boundary
    # stem width: outline width a third of the way up the stem
    ys = base_y + 0.35 * (sy - base_y)
    band = c[np.abs(c[:, 1] - ys) < 0.06 * (top_y - base_y)]
    sx0, sx1 = band[:, 0].min(), band[:, 0].max()
    # blade max width + its height
    bi = cen >= sy
    bw_y = cen[bi][np.argmax(w[bi])]
    blade = c[np.abs(c[:, 1] - bw_y) < 0.06 * (top_y - base_y)]
    bx0, bx1 = blade[:, 0].min(), blade[:, 0].max()
    xL, xR = c[:, 0].min(), c[:, 0].max()

    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.add_patch(Polygon(c, closed=True, facecolor=OBSIDIAN, edgecolor=EDGE,
                         linewidth=1.6, zorder=2))
    # shoulder reference line
    ax.plot([xL - 30, xR + 30], [sy, sy], "--", color="#888888", lw=1.0, zorder=3)

    # dimension arrows: stem length (vertical) + stem width (horizontal)
    axL = sx0 - 140
    _dim_arrow(ax, (axL, base_y), (axL, sy), "stem\nlength",
               (axL - 60, (base_y + sy) / 2), ha="right")
    _dim_arrow(ax, (sx0, ys), (sx1, ys), "stem width", (((sx0 + sx1) / 2), ys - 95))

    # callouts
    span = top_y - base_y
    _callout(ax, "Shoulder shape", "angle where blade meets stem",
             xy=(bx1 * 0.7, sy), xytext=(xR + 230, sy + 0.10 * span))
    _callout(ax, "Stem shape", "stem outline form",
             xy=(sx1, (base_y + sy) / 2), xytext=(xR + 230, base_y + 0.18 * span))
    _callout(ax, "Blade outline", "idiosyncratic / functional —\nthe scale-free control "
                                   "(no\nspatial structure)",
             xy=(bx0 * 0.55, bw_y), xytext=(xL - 540, bw_y))
    _callout(ax, "Stem (hafting element)", "standardized, heritable —\ncarries the "
                                           "hyperlocality signal",
             xy=(0, base_y + 0.18 * span), xytext=(xR + 230, base_y - 0.02 * span))

    ax.set_xlim(xL - 700, xR + 720)
    ax.set_ylim(base_y - 0.12 * span, top_y + 0.07 * span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Dimensions of mata'a stem-form variability scored in the analysis",
                 fontsize=13)
    fig.text(0.5, 0.06,
             "Paradigmatic classes = stem length × width (14 classes) and stem shape × "
             "shoulder shape (6 classes). The narrow stem is the standardized, heritable\n"
             "hafting element that carries the signal; the whole-blade outline is the "
             f"idiosyncratic scale-free control (no spatial structure). Outline: piece "
             f"{EXAMPLE_ID}, Ahu Tautira.",
             ha="center", fontsize=8.4, color="#444444")
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
