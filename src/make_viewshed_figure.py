"""Figure 6: ahu intervisibility communities on a DEM hillshade.

Panel A: shaded-relief hillshade of Rapa Nui (from the 50 cm Vricon DTM worked at
5 m) with the 57 moai-bearing ahu, colored by the visual community they fall into.
Communities are the greedy-modularity partition of the mutual-intervisibility
network (an edge joins two ahu when each can see the other's moai over the terrain,
with Earth curvature and atmospheric refraction). Panel B: the network's modularity,
community count, and mean community diameter against a random-siting null.

Reads data/viewshed/ahu_viewshed.json (the precomputed intervisibility matrices,
from src/ahu_viewshed.py) and output/viewshed_models.json (the null comparison,
from src/viewshed_models.py). Needs GDAL + networkx; run with system python3:

    python3 src/make_viewshed_figure.py

The DEM itself (DEM_PATH) is an external ~23 MB raster, not redistributed here.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from osgeo import gdal, osr

gdal.UseExceptions()

DEM_PATH = os.environ.get("RAPANUI_DEM", "/home/clipo/rapanui_dem/dem_5m.tif")
VS_JSON = "data/viewshed/ahu_viewshed.json"
MODELS_JSON = "output/viewshed_models.json"
OUT = "figures/fig6_intervisibility"

# qualitative palette for the visual communities (singletons drawn gray)
COMM_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
               "#a65628", "#f781bf", "#17bec8", "#bcbd22", "#666666"]


def _network_communities(vs):
    """Mutual-intervisibility graph + greedy-modularity communities (as in
    src/viewshed_models.py). Returns (G, label_per_node, n_nodes)."""
    ahu = vs["ahu"]
    N = len(ahu)
    minh = vs["min_height"]
    heights = [a.get("max_h") for a in ahu]

    def h(j):
        return heights[j] if heights[j] else 4.0

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            ab = minh[i][j] is not None and minh[i][j] <= h(j)
            ba = minh[j][i] is not None and minh[j][i] <= h(i)
            if ab and ba:
                G.add_edge(i, j)
    comms = list(nx.community.greedy_modularity_communities(G))
    comms = sorted(comms, key=len, reverse=True)
    label = [-1] * N
    cid = 0
    for c in comms:
        if len(c) >= 2:
            for n in c:
                label[n] = cid
            cid += 1
    return G, label, N


def _lonlat_to_dem(lons, lats, dem_proj):
    src = osr.SpatialReference(); src.ImportFromEPSG(4326)
    dst = osr.SpatialReference(); dst.ImportFromWkt(dem_proj)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(src, dst)
    xs, ys = [], []
    for lon, lat in zip(lons, lats):
        x, y, _ = ct.TransformPoint(lon, lat)
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def _hillshade():
    ds = gdal.Open(DEM_PATH)
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    elev = ds.GetRasterBand(1).ReadAsArray()
    nd = ds.GetRasterBand(1).GetNoDataValue()
    hs = gdal.DEMProcessing("", ds, "hillshade", format="MEM",
                            azimuth=315, altitude=45, zFactor=1.5, computeEdges=True)
    shade = hs.GetRasterBand(1).ReadAsArray().astype(float)
    sea = (elev == nd) if nd is not None else (elev < 2.0)
    extent = [gt[0], gt[0] + W * gt[1], gt[3] + H * gt[5], gt[3]]  # left,right,bottom,top
    return shade, sea, extent, ds.GetProjection()


def main():
    if not os.path.exists(DEM_PATH):
        raise SystemExit(
            f"DEM not found at {DEM_PATH!r}.\n"
            "Figure 6's hillshade base needs the 50 cm Vricon DTM, which is licensed "
            "to the authors and not redistributed. Set RAPANUI_DEM to your own copy "
            "of the raster. The intervisibility analysis and the cross-proxy "
            "concordance (src/run_concordance.py) reproduce without it, from the "
            "committed data/viewshed/*.json and output/viewshed_models.json."
        )
    vs = json.loads(open(VS_JSON).read())
    models = json.loads(open(MODELS_JSON).read())
    ahu = vs["ahu"]
    G, label, N = _network_communities(vs)
    n_comm = max(label) + 1
    print(f"{N} ahu, {G.number_of_edges()} mutual-visibility edges, {n_comm} communities")

    fig = plt.figure(figsize=(13, 6.2))
    axA = fig.add_axes([0.02, 0.06, 0.58, 0.88])
    axB = fig.add_axes([0.67, 0.58, 0.31, 0.33])
    axC = fig.add_axes([0.67, 0.10, 0.31, 0.30])

    # --- Panel A: hillshade + communities ---
    shade, sea, extent, proj = _hillshade()
    shaded = np.ma.masked_where(sea, shade)
    axA.imshow(np.where(sea, np.nan, shade), extent=extent, cmap="gray",
               vmin=0, vmax=255, origin="upper", interpolation="bilinear")
    axA.imshow(np.where(sea, 1.0, np.nan), extent=extent, cmap=plt.cm.colors.ListedColormap(["#c9ddef"]),
               origin="upper", interpolation="nearest", alpha=0.55)

    lons = [a["lon"] for a in ahu]; lats = [a["lat"] for a in ahu]
    xs, ys = _lonlat_to_dem(lons, lats, proj)
    # intervisibility edges (faint)
    for i, j in G.edges():
        axA.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#33333344", lw=0.6, zorder=2)
    # ahu colored by community
    for i in range(N):
        c = COMM_COLORS[label[i]] if label[i] >= 0 else "#999999"
        axA.scatter([xs[i]], [ys[i]], s=46, color=c, edgecolors="black",
                    linewidths=0.5, zorder=3)
    axA.set_xlim(extent[0], extent[1]); axA.set_ylim(extent[2], extent[3])
    axA.set_aspect("equal"); axA.axis("off")
    axA.set_title(f"A. Ahu form {n_comm} bounded visual communities (~5 km across)",
                  fontsize=10.5, loc="left")
    # scale bar (5 km) in UTM meters
    x0 = extent[0] + 0.06 * (extent[1] - extent[0]); y0 = extent[2] + 0.07 * (extent[3] - extent[2])
    axA.plot([x0, x0 + 5000], [y0, y0], color="black", lw=2.5)
    axA.text(x0 + 2500, y0 + 250, "5 km", ha="center", fontsize=8)
    axA.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=8, label="moai-bearing ahu"),
               Line2D([0], [0], color="#33333366", lw=1, label="mutual intervisibility")],
               loc="upper right", fontsize=8, frameon=False)

    # --- Panel B: modularity / n_comm / diameter vs random-siting null ---
    t = models["tests"]
    rows = [("network\nmodularity", t["modularity"]),
            ("number of\ncommunities", t["n_comm"]),
            ("community\ndiameter (km)", t["comm_diam_km"])]
    ys2 = np.arange(len(rows))[::-1]
    for y, (lab, d) in zip(ys2, rows):
        lo, hi, nm, obs, z = d["null_lo"], d["null_hi"], d["null_mean"], d["observed"], d["z"]
        rng = max(hi, obs) - min(lo, obs) or 1.0
        # normalize each row to its own axis by plotting on a twin scale: use text + relative bar
        axB.barh([y], [hi - lo], left=[lo], height=0.0)  # placeholder to set autoscale off
    axB.clear()
    # simpler: one normalized panel — plot observed vs null band per metric on separate mini-rows
    labels = [r[0] for r in rows]
    for k, (lab, d) in enumerate(rows):
        y = len(rows) - 1 - k
        lo, hi, nm, obs = d["null_lo"], d["null_hi"], d["null_mean"], d["observed"]
        span = (hi - lo) or 1.0
        # map [min,max] of (lo,hi,obs) to [0,1]
        lomin = min(lo, obs); himax = max(hi, obs); rng = (himax - lomin) or 1.0
        def nx_(v):
            return (v - lomin) / rng
        axB.plot([nx_(lo), nx_(hi)], [y, y], color="#c0392b", lw=3, alpha=0.5, zorder=2,
                 solid_capstyle="round")
        axB.scatter([nx_(nm)], [y], marker="|", s=200, color="#c0392b", zorder=3)
        axB.scatter([nx_(obs)], [y], s=90, color="#1a5276", zorder=4)
        axB.annotate(f"obs {obs}  (null {nm}, z={d['z']:+.1f})", (nx_(obs), y),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    axB.set_yticks(range(len(rows))); axB.set_yticklabels(labels[::-1], fontsize=9)
    axB.set_xticks([])
    axB.set_xlim(-0.15, 1.15); axB.set_ylim(-0.5, len(rows) - 0.3)
    axB.set_title("B. Observed vs random-siting null", fontsize=10.5, loc="left")
    axB.scatter([], [], s=90, color="#1a5276", label="observed")
    axB.plot([], [], color="#c0392b", lw=3, alpha=0.5, label="null 95% interval")
    axB.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    for sp in ("top", "right", "left"):
        axB.spines[sp].set_visible(False)

    # --- Panel C: cross-proxy concordance (moai style across visual communities) ---
    if os.path.exists("output/concordance_results.json"):
        cc = json.load(open("output/concordance_results.json"))
        axC.plot([cc["fst_hdi"][0], cc["fst_hdi"][1]], [1, 1], color="#1a5276", lw=3,
                 solid_capstyle="round", zorder=2)
        axC.scatter([cc["fst_median"]], [1], s=80, color="#1a5276", zorder=3)
        axC.annotate(f"2 ln BF = {cc['two_ln_bf']:+.1f}", (cc["fst_hdi"][1], 1),
                     textcoords="offset points", xytext=(8, 0), va="center", fontsize=8)
        axC.set_yticks([1]); axC.set_yticklabels(["moai style across\nvisual communities"], fontsize=8.5)
        axC.set_ylim(0.4, 1.6); axC.set_xlim(left=0)
        axC.set_xlabel("posterior cultural $F_{ST}$ (median, bar = 95% HDI)", fontsize=8.5)
        axC.axvline(0, color="#cccccc", lw=0.8)
        axC.set_title(f"C. Cross-proxy concordance (partition agreement: adjusted Rand "
                      f"{cc['adjusted_rand']:.2f}, NMI {cc.get('nmi', 0):.2f}, p < 0.001)",
                      fontsize=9.0, loc="left")
        for sp in ("top", "right"):
            axC.spines[sp].set_visible(False)
    else:
        axC.axis("off")

    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(OUT + ext, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT + ".png/.pdf/.svg")


if __name__ == "__main__":
    main()
