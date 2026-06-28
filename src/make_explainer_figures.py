"""Illustrative figures for talks/teaching and the manuscripts' supporting set:

  moai_style_variability.png   where moai style states sit across the island
  mataa_style_variability.png  stem-class composition across the SW mata'a cluster
  intervisibility_explainer.png  what the ahu intervisibility analysis is doing

These are descriptive/illustrative; the headline statistics live in the Bayesian
figures (make_figures.py) and Figure 6 (make_viewshed_figure.py). All three render
from committed data over the cached ESRI relief basemap -- no licensed DEM needed.

Run:  python3 src/make_explainer_figures.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

import figbase
import moai as moai_mod
import tables_io

# Map helpers (kept in sync with make_figures.py; duplicated here so this script
# runs under system python3 without pulling in the PyMC/GDAL figure stack).


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2, foreground="white")]


def _pie_on_map(ax, x, y, fracs, colors, r, edge="#222222"):
    start = 90.0
    for f, c in zip(fracs, colors):
        if f <= 0:
            continue
        ext = 360.0 * f
        ax.add_patch(Wedge((x, y), r, start - ext, start, facecolor=c,
                           edgecolor=edge, linewidth=0.5, zorder=6))
        start -= ext


def _relief_panel(ax, lon0, lon1, lat0, lat1, z, title):
    img, origin, zz = figbase.basemap(lon0, lon1, lat0, lat1, z=z, source="relief")
    H, W = img.shape[:2]
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, loc="left")
    return origin, zz, W, H


def _km_bar(ax, origin, z, lat, lon, W, H, km, fx=0.06, fy=0.94):
    x0, _ = figbase.lonlat_to_px(lon, lat, origin, z)
    x1, _ = figbase.lonlat_to_px(lon + km / figbase.km_per_deg_lon(lat), lat, origin, z)
    px = abs(x1 - x0)
    bx, by = fx * W, fy * H
    ax.plot([bx, bx + px], [by, by], "k-", lw=2.5, zorder=10, path_effects=_halo())
    ax.annotate(f"{km:g} km", (bx + px / 2, by - 0.018 * H), ha="center",
                fontsize=7, zorder=10, path_effects=_halo())


def _pie_callout(ax, x, y, dx, dy, fracs, colors, r, label, ha="center"):
    px, py = x + dx, y + dy
    ax.plot([x, px], [y, py], color="#333333", lw=0.6, zorder=5,
            path_effects=_halo())
    ax.scatter([x], [y], s=16, marker="o", facecolor="white", edgecolor="#222222",
               linewidths=0.8, zorder=7)
    _pie_on_map(ax, px, py, fracs, colors, r)
    ax.annotate(label, (px, py + r * 1.35), ha=ha, va="top", fontsize=7.5,
                zorder=8, path_effects=_halo())


# whole-island extent + zoom (matches fig4 panel C)
ISLAND = (-109.48, -109.21, -27.215, -27.040)
Z = 13
PALETTE = ["#d7191c", "#2c7bb6", "#fdae61", "#1a9641", "#7b3294", "#e7298a"]
GREY = "#d9d9d9"


def _scatter_by_state(ax, md, attr, origin, z, max_states=5):
    """Plot every moai colored by its cleaned state for one style attribute."""
    P = lambda lo, la: figbase.lonlat_to_px(lo, la, origin, z)
    vals = moai_mod._clean(md[attr])
    states = list(vals.dropna().value_counts().index[:max_states])
    uns = md[~vals.isin(states)]
    if len(uns):
        xs, ys = zip(*[P(lo, la) for lo, la in zip(uns.longitude, uns.latitude)])
        ax.scatter(xs, ys, s=8, c=GREY, edgecolors="none", zorder=4)
    for st, col in zip(states, PALETTE):
        g = md[vals == st]
        xs, ys = zip(*[P(lo, la) for lo, la in zip(g.longitude, g.latitude)])
        ax.scatter(xs, ys, s=30, c=col, edgecolors="white", linewidths=0.4,
                   zorder=5, label=f"{str(st).replace('_', ' ')} ({len(g)})")
    ax.legend(loc="lower left", fontsize=6.5, framealpha=0.9)


def fig_moai_style(path="figures/moai_style_variability.png"):
    md = moai_mod.load()
    attrs = [("HEAD_PLAN_SHAPE", "Head plan shape"),
             ("HEAD_PROFILE_SHAPE", "Head profile shape"),
             ("NOSE_PROFIE", "Nose profile"),
             ("BASE_SHAPE", "Base shape")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (attr, label) in zip(axes.ravel(), attrs):
        o, z, W, H = _relief_panel(ax, *ISLAND, Z, label)
        _scatter_by_state(ax, md, attr, o, z)
    _km_bar(axes[1, 1], o, z, -27.205, -109.30, W, H, 2, fx=0.74)
    fig.suptitle("Moai stylistic variability across Rapa Nui (one point per ahu-placed statue)",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def fig_mataa_style(path="figures/mataa_style_variability.png"):
    lw, _ = tables_io.stem_length_width()
    length_class = {}
    for cls in lw.columns:
        length_class.setdefault(cls[0], []).append(cls)
    Lkeys = sorted(length_class)                       # A, B, C, D
    Lcol = {"A": "#1b9e77", "B": "#d95f02", "C": "#7570b3", "D": "#e7298a"}
    sites = ["Ahu Tautira", "Orongo", "Rano Kau", "Vinapu", "Orito"]
    place = {"Orito": "Maunga Orito"}

    def fracs(s):
        cnt = lw.loc[s]
        return [cnt[length_class[k]].sum() / cnt.sum() for k in Lkeys]

    fig, (axM, axB) = plt.subplots(1, 2, figsize=(13, 5.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # left: SW cluster map with stem-length-class pies
    aoff = {"Ahu Tautira": (-0.17, -0.05), "Orongo": (-0.10, 0.16),
            "Rano Kau": (-0.20, 0.02), "Vinapu": (0.04, 0.20),
            "Orito": (0.05, -0.16)}
    o, z, W, H = _relief_panel(axM, -109.462, -109.355, -27.205, -27.130, Z,
                               "A. Stem-class composition, SW cluster (sites within 5.5 km)")
    for s in sites:
        x, y = figbase.lonlat_to_px(*figbase.PLACES[place.get(s, s)], o, z)
        dx, dy = aoff[s]
        _pie_callout(axM, x, y, dx * W, dy * H, fracs(s),
                     [Lcol[k] for k in Lkeys], r=0.034 * W, label=s)
    _km_bar(axM, o, z, -27.20, -109.458, W, H, 1, fx=0.72, fy=0.94)
    for k in Lkeys:
        axM.scatter([], [], s=55, marker="o", c=Lcol[k], label=f"length {k}")
    axM.legend(loc="upper right", fontsize=7.5, framealpha=0.9, title="stem length")

    # right: stacked bars make the between-site differences explicit
    y = np.arange(len(sites))[::-1]
    left = np.zeros(len(sites))
    F = np.array([fracs(s) for s in sites])
    for ki, k in enumerate(Lkeys):
        axB.barh(y, F[:, ki], left=left, color=Lcol[k], edgecolor="white",
                 height=0.7, label=f"length {k}")
        left = left + F[:, ki]
    axB.set_yticks(y)
    axB.set_yticklabels(sites, fontsize=9)
    axB.set_xlim(0, 1)
    axB.set_xlabel("proportion of stem-length classes", fontsize=9)
    axB.set_title("B. The same composition, by site", fontsize=9.5, loc="left")
    axB.annotate("Each site draws its stem forms from a different mix\n"
                 "(posterior cultural F$_{ST}$ ≈ 0.03 [0.01, 0.05], stem length×width)",
                 (0.5, -0.24), xycoords="axes fraction", ha="center", fontsize=8.5)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    fig.suptitle("Mata'a stem-form variability across the southwest assemblages",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def _mutual_edges(vs):
    """List of (i, j) ahu pairs that can each see the other's moai."""
    ahu = vs["ahu"]
    mh = vs["min_height"]
    N = len(ahu)
    h = lambda j: ahu[j].get("max_h") or 4.0
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (mh[i][j] is not None and mh[i][j] <= h(j)
                    and mh[j][i] is not None and mh[j][i] <= h(i)):
                edges.append((i, j))
    return edges


def fig_intervisibility(path="figures/intervisibility_explainer.png"):
    vs = json.load(open("data/viewshed/ahu_viewshed.json"))
    comm = json.load(open("data/viewshed/ahu_communities.json"))["ahu"]
    ahu = vs["ahu"]
    edges = _mutual_edges(vs)
    lon = [a["lon"] for a in ahu]
    lat = [a["lat"] for a in ahu]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.2))

    # Panel A: lines of sight from one example platform
    oA, zA, WA, HA = _relief_panel(axA, *ISLAND, Z,
                                   "A. What it measures: lines of sight from one platform")
    PA = lambda i: figbase.lonlat_to_px(lon[i], lat[i], oA, zA)
    px = [PA(i)[0] for i in range(len(ahu))]
    py = [PA(i)[1] for i in range(len(ahu))]
    ex = next((i for i, a in enumerate(ahu) if a["site"] == "AHU VINAPU"), None)
    if ex is None:
        deg = {i: 0 for i in range(len(ahu))}
        for i, j in edges:
            deg[i] += 1; deg[j] += 1
        ex = max(deg, key=deg.get)
    neigh = [j for (i, j) in edges if i == ex] + [i for (i, j) in edges if j == ex]
    axA.scatter(px, py, s=14, c=GREY, edgecolors="none", zorder=4)
    for j in neigh:
        axA.plot([px[ex], px[j]], [py[ex], py[j]], "-", color="#d7191c",
                 lw=1.3, alpha=0.9, zorder=5, path_effects=_halo())
    axA.scatter([px[j] for j in neigh], [py[j] for j in neigh], s=46, c="#d7191c",
                edgecolors="white", linewidths=0.5, zorder=6)
    axA.scatter([px[ex]], [py[ex]], s=200, marker="*", c="#fee08b",
                edgecolors="#222222", linewidths=0.8, zorder=7)
    axA.annotate(f"{ahu[ex]['site'].title()}\nsees {len(neigh)} other platforms",
                 (px[ex], py[ex]), textcoords="offset points", xytext=(8, 8),
                 fontsize=8.5, fontweight="bold", zorder=8, path_effects=_halo())
    _km_bar(axA, oA, zA, -27.205, -109.47, WA, HA, 2)

    # Panel B: the whole network resolves into bounded visual communities
    oB, zB, WB, HB = _relief_panel(axB, *ISLAND, Z,
                                   "B. What it finds: eight bounded visual communities")
    PB = lambda i: figbase.lonlat_to_px(lon[i], lat[i], oB, zB)
    bx = [PB(i)[0] for i in range(len(ahu))]
    by = [PB(i)[1] for i in range(len(ahu))]
    for i, j in edges:
        axB.plot([bx[i], bx[j]], [by[i], by[j]], "-", color="#555555",
                 lw=0.6, alpha=0.5, zorder=4)
    labels = [c["community"] for c in comm]
    ncomm = max(labels) + 1
    cols = (PALETTE + ["#a6761d", "#66a61e", "#1b9e77", "#666666"])
    for i in range(len(ahu)):
        lab = labels[i]
        c = "#bbbbbb" if lab < 0 else cols[lab % len(cols)]
        axB.scatter([bx[i]], [by[i]], s=42, c=c, edgecolors="white",
                    linewidths=0.5, zorder=6)
    axB.annotate(f"{ncomm} visual communities ~5 km across\n"
                 "network modularity 0.73 (vs 0.49 random)",
                 (0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=8.5,
                 zorder=9, path_effects=_halo())
    _km_bar(axB, oB, zB, -27.205, -109.47, WB, HB, 2)

    fig.suptitle("Ahu intervisibility: from per-monument sightlines to bounded communities",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    fig_moai_style()
    fig_mataa_style()
    fig_intervisibility()


if __name__ == "__main__":
    main()
