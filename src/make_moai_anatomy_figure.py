"""Schematic of a moai labeled with the dimensions of stylistic variability scored.

A stylized front and profile view of a moai with callouts to the seven categorical
style attributes used in the analysis (from moai.CANDIDATE_LOCI), each annotated
with the states actually observed in the public moai database. The two head/face
attributes that carry the strongest between-community signal are flagged with their
per-locus posterior F_ST.

Illustrative/schematic (the moai outline is drawn, not measured). DEM-free, system
python3.  Run:  python3 src/make_moai_anatomy_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import moai as moai_mod

STONE = "#b6a892"
EDGE = "#4a4036"
ACCENT = "#7a2c2c"


def _states(md, attr, n=4):
    vc = moai_mod._clean(md[attr]).dropna().value_counts()
    out = [str(s).replace("_", "-") for s in vc.index[:n]]
    return " · ".join(out)


def _mirror_close(right):
    """Full symmetric polygon from a right-half (x>=5) point list, top->bottom."""
    left = [(10 - x, y) for (x, y) in reversed(right)][1:-1]
    return right + left


def _front(ax):
    right = [(5.0, 19.2), (6.7, 19.0), (7.0, 15.5), (6.9, 12.6), (8.4, 11.4),
             (8.1, 5.6), (7.7, 4.3), (8.4, 4.0), (8.4, 1.0), (5.0, 1.0)]
    ax.add_patch(Polygon(_mirror_close(right), closed=True, facecolor=STONE,
                         edgecolor=EDGE, linewidth=1.6, zorder=2))
    # base line
    ax.plot([1.7, 8.3], [4.0, 4.0], color=EDGE, lw=1.3, zorder=3)
    # brow
    ax.plot([3.4, 6.6], [15.4, 15.4], color=EDGE, lw=2.2, zorder=3)
    # nose
    ax.add_patch(Polygon([(4.6, 15.2), (5.4, 15.2), (5.65, 12.7), (4.35, 12.7)],
                         closed=True, facecolor="none", edgecolor=EDGE, lw=1.3, zorder=3))
    # mouth
    ax.plot([4.4, 5.6], [12.1, 12.1], color=EDGE, lw=1.6, zorder=3)
    # long ears
    for ex in (3.15, 6.85):
        ax.plot([ex, ex], [15.6, 12.7], color=EDGE, lw=1.3, zorder=3)
    # arms angling to the hands at the navel
    ax.plot([3.2, 4.2], [11.2, 5.6], color=EDGE, lw=1.2, zorder=3)
    ax.plot([6.8, 5.8], [11.2, 5.6], color=EDGE, lw=1.2, zorder=3)
    # long fingers meeting under the navel
    for fy in (5.5, 5.15, 4.8):
        ax.plot([3.6, 5.0], [fy + 0.25, fy], color=EDGE, lw=1.0, zorder=3)
        ax.plot([6.4, 5.0], [fy + 0.25, fy], color=EDGE, lw=1.0, zorder=3)
    ax.set_title("Front (plan) view", fontsize=11, loc="center")


def _profile(ax):
    prof = [(7.2, 19.0), (4.7, 18.7), (3.6, 16.2), (3.0, 15.2), (2.6, 14.1),
            (2.45, 13.1), (3.35, 12.75), (3.2, 12.1), (3.75, 11.4), (3.95, 9.5),
            (3.75, 6.0), (3.65, 4.3), (3.1, 4.05), (3.1, 1.0), (8.6, 1.0),
            (8.6, 4.05), (8.0, 4.3), (7.9, 6.0), (7.85, 11.6), (7.7, 15.2)]
    ax.add_patch(Polygon(prof, closed=True, facecolor=STONE, edgecolor=EDGE,
                         linewidth=1.6, zorder=2))
    ax.plot([3.1, 8.6], [4.05, 4.05], color=EDGE, lw=1.3, zorder=3)   # base line
    # brow ridge + eye socket
    ax.plot([3.0, 4.6], [15.0, 14.7], color=EDGE, lw=2.0, zorder=3)
    ax.plot([3.9, 4.9], [14.2, 14.0], color=EDGE, lw=1.1, zorder=3)
    # ear on the side of the head
    ax.plot([6.4, 6.4], [15.5, 12.7], color=EDGE, lw=1.3, zorder=3)
    # arm down the front
    ax.plot([4.2, 4.1], [11.2, 6.2], color=EDGE, lw=1.1, zorder=3)
    ax.set_title("Profile view", fontsize=11, loc="center")


def _callout(ax, name, states, xy, xytext, fst=None, ha="left"):
    body = f"{name}\n{states}"
    if fst:
        body += f"\n{fst}"
    ax.annotate(
        body, xy=xy, xytext=xytext, ha=ha, va="center", fontsize=8.2, zorder=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", lw=0.8),
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.3,
                        connectionstyle="arc3,rad=0.12"))


def main(path="figures/moai_dimensions.png"):
    md = moai_mod.load()
    fig, (axF, axP) = plt.subplots(1, 2, figsize=(14, 9))
    for ax in (axF, axP):
        ax.set_xlim(-5.2, 11.5)
        ax.set_ylim(0, 20.5)
        ax.set_aspect("equal")
        ax.axis("off")
    _front(axF)
    _profile(axP)

    # front-view callouts (text to the left, arrows to the feature)
    _callout(axF, "Head plan shape", _states(md, "HEAD_PLAN_SHAPE"),
             xy=(4.9, 17.2), xytext=(-5.0, 18.0),
             fst="strongest signal: posterior F_ST ≈ 0.14")
    _callout(axF, "Body plan shape", _states(md, "BODY_PLAN_SHAPE"),
             xy=(4.6, 8.5), xytext=(-5.0, 10.5))
    _callout(axF, "Hand placement on base", _states(md, "HANDS_PLACEMENT_ON_BASE"),
             xy=(5.0, 5.1), xytext=(-5.0, 4.6))
    _callout(axF, "Base shape", _states(md, "BASE_SHAPE"),
             xy=(6.8, 2.4), xytext=(9.2, 3.0), ha="left")
    _callout(axF, "Long ears", "elongated lobes",
             xy=(6.85, 14.0), xytext=(9.2, 16.5), ha="left")

    # profile-view callouts
    _callout(axP, "Head profile shape", _states(md, "HEAD_PROFILE_SHAPE"),
             xy=(7.6, 16.5), xytext=(9.0, 18.5), ha="left",
             fst="posterior F_ST ≈ 0.07")
    _callout(axP, "Nose profile", _states(md, "NOSE_PROFIE"),
             xy=(2.5, 13.4), xytext=(-5.2, 14.5))
    _callout(axP, "Body profile shape", _states(md, "BODY_PRO_SHAPE"),
             xy=(7.85, 8.5), xytext=(9.0, 10.5), ha="left")
    _callout(axP, "Base shape", _states(md, "BASE_SHAPE"),
             xy=(4.5, 2.4), xytext=(-5.2, 3.0))

    fig.suptitle("Dimensions of moai stylistic variability scored in the analysis",
                 fontsize=14, y=0.97)
    fig.text(0.5, 0.045,
             "Seven categorical style attributes (states observed in the public moai "
             "database). Absolute size is excluded because it is time-transgressive; "
             "only scale-free style states are used.",
             ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
