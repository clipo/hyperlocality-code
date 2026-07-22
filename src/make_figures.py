"""Build the publication figures into figures/:

  fig1_concept.png  panmixia vs low-connectivity hyperlocality schematic
  fig2_map.png      shaded-relief island map with each proxy's sample locations
  fig3_results.png  cultural F_ST vs null (4 proxies) + umu gradient + moai
                    feature F_ST + mata'a isolation-by-distance
  fig4_variability.png  geography of the variability: per-place class composition
                    (mata'a SW pies, umu quadrangle pies, moai head-plan by statue)
  fig5_convergence.png  genetic vs cultural F_ST; regional-contrast distance slope

The Bayesian statistics (posterior F_ST, 95% HDIs, Bayes factors) are read from
output/bayes_results.json, the single source of truth written by run_bayes.py.
The fast isolation-by-distance regressions are recomputed live from bayes.py.
"""
import json
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

import figbase
import popgen
import tables_io
import umu as umu_mod
import moai as moai_mod
import pukao as pukao_mod
import spatial
import bayes

RNG = np.random.default_rng(20260623)

BAYES_JSON = "output/bayes_results.json"

# East-coast parcel surveys (vs the SW coast) for the region-contrast IBD control.
PARCELS_FIG = ["Parcel 6", "Parcel 7", "Parcel 8",
               "Parcel 9", "Parcel 10", "Parcel 11"]


def _bayes():
    """Load the Bayesian results JSON (run src/run_bayes.py first)."""
    if not os.path.exists(BAYES_JSON):
        raise FileNotFoundError(
            f"{BAYES_JSON} not found -- run `PYTHONPATH=src .venv/bin/python "
            "src/run_bayes.py` before building figures.")
    with open(BAYES_JSON) as f:
        return json.load(f)["proxies"]


def _bf_label(two_ln_bf):
    """Short annotation for a 2 ln BF value on the Kass-Raftery scale."""
    a = abs(two_ln_bf)
    band = "very strong" if a >= 10 else "strong" if a >= 6 else \
           "positive" if a >= 2 else "weak"
    side = "structure" if two_ln_bf > 0 else "panmixia"
    return f"2 ln BF = {two_ln_bf:+.0f} ({band} for {side})"


def _savefig(fig, path, dpi=160):
    """Save a figure in three formats: .png (raster, for the manuscript),
    .pdf and .svg (vector). `path` is the .png target; siblings get .pdf/.svg."""
    base = path[:-4] if path.lower().endswith(".png") else path
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(base + ext, dpi=dpi)
    return base
CLAN_COLORS = {
    "Gnatimo": "#e41a1c", "Gnaure": "#377eb8", "Haumoana": "#4daf4a",
    "Hituira Tupahotu Koroarongo": "#984ea3", "Marama": "#ff7f00",
    "Miru Hama Miru": "#a65628",
}


# ----------------------------------------------------------------------------
# Manual label offsets (display points) so the crowded south-coast names do not
# collide. Each entry: (dx, dy, ha). Tuned against the rendered relief base.
_SITE_LABELS = {
    "Orongo":       (-6, -10, "right"),
    "Rano Kau":     (-8,  10, "right"),
    "Vinapu":       (-2, -13, "center"),
    "Maunga Orito": (-6,  12, "right"),
    "Ahu Tautira":  (-8,   4, "right"),
    "Hanga Poukura": (10, -10, "left"),
    "Vaihu":        (8,   -4, "left"),
}
_FEATURE_LABELS = {
    "Maunga Terevaka": (0,  12, "center"),
    "Poike":           (10,  0, "left"),
    "Rano Raraku":     (8,   6, "left"),
    "Motu Nui":        (-8, -2, "right"),
}


def fig_map(path="figures/fig2_map.png"):
    # Island extent: SW point (Orongo, Motu Nui) to the Poike peninsula east.
    lon0, lon1, lat0, lat1 = -109.48, -109.21, -27.215, -27.040
    img, origin, z = figbase.basemap(lon0, lon1, lat0, lat1, z=13, source="relief")
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(9.2, 6.3))
    ax.imshow(img)

    def P(lon, lat):
        return figbase.lonlat_to_px(lon, lat, origin, z)

    # --- sample locations of the four proxies (the subject of the figure) ---
    # moai (ahu-placed), one color: the analysis groups them by geography, not clan
    m = moai_mod.load()
    xs, ys = zip(*[P(lo, la) for lo, la in zip(m.longitude, m.latitude)])
    ax.scatter(xs, ys, s=11, c="#4d7ea8", edgecolors="none", alpha=0.85,
               zorder=4, label="moai (ahu-placed)")

    # pukao (UTM 12S -> lon/lat)
    pk = pukao_mod.load()
    pll = [figbase.utm12s_to_lonlat(e, n) for e, n in zip(pk["x"], pk["y"])]
    xs, ys = zip(*[P(lo, la) for lo, la in pll])
    ax.scatter(xs, ys, s=24, marker="^", facecolors="none",
               edgecolors="black", linewidths=0.9, zorder=5, label="pukao")

    # mata'a SW assemblage sites and umu quadrangles (south-coast band)
    for s in ["Orongo", "Rano Kau", "Vinapu", "Maunga Orito", "Ahu Tautira"]:
        x, y = P(*figbase.PLACES[s])
        ax.scatter([x], [y], s=150, marker="*", c="#111111", zorder=7)
    ax.scatter([], [], s=150, marker="*", c="#111111", label="mata'a site")
    for s in ["Rano Kau", "Vinapu", "Maunga Orito", "Hanga Poukura", "Vaihu"]:
        x, y = P(*figbase.PLACES[s])
        ax.scatter([x], [y], s=95, marker="s", facecolors="none",
                   edgecolors="#c0392b", linewidths=1.6, zorder=6)
    ax.scatter([], [], s=95, marker="s", facecolors="none",
               edgecolors="#c0392b", linewidths=1.6, label="umu quadrangle")

    # --- named locations (text place-names) ---
    for s, (dx, dy, ha) in _SITE_LABELS.items():
        x, y = P(*figbase.PLACES[s])
        ax.annotate(s, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=8, zorder=9, path_effects=_halo())

    # --- geographic features (landforms, for orientation) ---
    for s, (dx, dy, ha) in _FEATURE_LABELS.items():
        lo, la = figbase.FEATURES[s]
        x, y = P(lo, la)
        mk = "." if s == "Motu Nui" else "^"
        ax.scatter([x], [y], s=34, marker=mk, c="#5b3a1a", zorder=6)
        ax.annotate(s, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=8, style="italic", color="#5b3a1a",
                    zorder=9, path_effects=_halo())

    _scale_bar(ax, origin, z, lat0, lon0, W, H)
    _north_arrow(ax, W, H)
    _pacific_inset(ax)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Rapa Nui (Easter Island): geography and proxy sample locations",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92, ncol=1)
    fig.text(0.012, 0.012, "Relief: ESRI World Shaded Relief", fontsize=6,
             color="#777777")
    fig.tight_layout()
    _savefig(fig, path, 170)
    plt.close(fig)
    print("wrote", path)


def _north_arrow(ax, W, H):
    x, y = 0.965 * W, 0.16 * H
    ax.annotate("", xy=(x, y - 0.085 * H), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.6),
                zorder=10)
    ax.annotate("N", (x, y - 0.095 * H), ha="center", va="bottom",
                fontsize=10, fontweight="bold", zorder=10,
                path_effects=_halo())


def _pacific_inset(ax):
    """Locator: Rapa Nui in the SE Pacific relative to South America."""
    lon0, lon1, lat0, lat1 = -145, -60, -52, 8
    img, origin, z = figbase.basemap(lon0, lon1, lat0, lat1, z=3, source="relief")
    h, w = img.shape[:2]
    iax = ax.inset_axes([0.008, 0.62, 0.30, 0.37])
    iax.imshow(img)
    rx, ry = figbase.lonlat_to_px(*figbase.RAPA_NUI, origin, z)
    iax.scatter([rx], [ry], s=55, marker="*", c="#d62728",
                edgecolors="white", linewidths=0.6, zorder=5)
    iax.annotate("Rapa Nui", (rx, ry), textcoords="offset points",
                 xytext=(-6, -9), ha="right", fontsize=7, fontweight="bold",
                 color="#b3121f", path_effects=_halo())
    sx, sy = figbase.lonlat_to_px(-65, -28, origin, z)
    iax.annotate("SOUTH\nAMERICA", (sx, sy), ha="center", fontsize=6.5,
                 color="#555555", path_effects=_halo())
    iax.set_xlim(0, w)
    iax.set_ylim(h, 0)
    iax.set_xticks([])
    iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_edgecolor("#333333")
        sp.set_linewidth(0.8)


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2, foreground="white")]


def _scale_bar(ax, origin, z, lat, lon, W, H):
    x0, y0 = figbase.lonlat_to_px(lon, lat, origin, z)
    dlon = 2.0 / figbase.km_per_deg_lon(lat)          # 2 km in degrees lon
    x1, _ = figbase.lonlat_to_px(lon + dlon, lat, origin, z)
    px = abs(x1 - x0)
    bx, by = 0.06 * W, 0.93 * H
    ax.plot([bx, bx + px], [by, by], "k-", lw=3, zorder=9,
            path_effects=_halo())
    ax.annotate("2 km", (bx + px / 2, by - 8), ha="center", fontsize=8,
                zorder=10, path_effects=_halo())


# ----------------------------------------------------------------------------
def fig_concept(path="figures/fig1_concept.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    variants = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]
    rng = np.random.default_rng(7)

    # community node positions (shared layout)
    pos = np.array([[0.25, 0.72], [0.62, 0.80], [0.80, 0.50],
                    [0.60, 0.22], [0.28, 0.30], [0.45, 0.52]])

    def draw_community(ax, c, freqs, r=0.075):
        # node as a ring of colored tokens reflecting variant frequencies
        ax.add_patch(Circle(c, r, facecolor="white", edgecolor="#333333",
                            lw=1.2, zorder=3))
        counts = np.round(freqs * 12).astype(int)
        toks = [v for v, n in zip(variants, counts) for _ in range(n)]
        k = len(toks)
        for i, col in enumerate(toks):
            ang = 2 * np.pi * i / max(k, 1)
            ax.add_patch(Circle((c[0] + 0.45 * r * np.cos(ang),
                                 c[1] + 0.45 * r * np.sin(ang)),
                                0.16 * r, facecolor=col, edgecolor="none",
                                zorder=4))

    # LEFT: panmixia / high connectivity -> one shared pool, all alike
    axL = axes[0]
    shared = np.array([0.45, 0.25, 0.20, 0.10])
    for a in range(len(pos)):
        for b in range(a + 1, len(pos)):
            axL.plot(*zip(pos[a], pos[b]), color="#bbbbbb", lw=0.8, zorder=1)
    for c in pos:
        draw_community(axL, c, shared)
    axL.set_title("High connectivity (panmixia)\none island-wide pool",
                  fontsize=11)
    axL.text(0.5, 0.045, r"between-community $F_{ST}\approx 0$",
             ha="center", fontsize=11, transform=axL.transAxes)

    # RIGHT: low connectivity -> drift to divergent local repertoires
    axR = axes[1]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]  # sparse ring
    for a, b in edges:
        axR.plot(*zip(pos[a], pos[b]), color="#bbbbbb", lw=0.8, zorder=1)
    for c in pos:
        f = rng.dirichlet([0.5, 0.5, 0.5, 0.5])
        draw_community(axR, c, f)
    axR.set_title("Low connectivity (hyperlocality)\nbounded, divergent pools",
                  fontsize=11)
    axR.text(0.5, 0.045, r"between-community $F_{ST}>0$",
             ha="center", fontsize=11, transform=axR.transAxes)

    for ax in axes:
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.0, 0.95)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("What drives hyperlocal variability: connectivity governs how "
                 "drift partitions cultural variants", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, path, 160)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------
def fig_results(path="figures/fig3_results.png"):
    B = _bayes()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    axA, axB, axC, axD = axes.ravel()

    # --- Panel A: posterior cultural F_ST (median + 95% HDI) across proxies ---
    rows = [
        ("mata'a\n(stem L×W)", B["mataa"]["headline"]["lengthwidth"]),
        ("umu\n(oven style)", B["umu"]["headline"]),
        ("moai\n(style, multilocus)", B["moai"]["headline"]),
        ("pukao\n(style, multilocus)", B["pukao"]["headline"]),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, rec) in zip(ys, rows):
        strong = rec["two_ln_bf"] >= 6
        col = "#1a5276" if strong else "#888888"
        axA.plot([rec["hdi_lo"], rec["hdi_hi"]], [y, y], color=col, lw=2.6, zorder=2)
        axA.scatter([rec["median"]], [y], s=70, color=col, zorder=3)
        axA.annotate(_bf_label(rec["two_ln_bf"]),
                     (rec["hdi_hi"], y), textcoords="offset points",
                     xytext=(8, 0), va="center", fontsize=8)
    axA.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    axA.set_yticks(ys)
    axA.set_yticklabels([r[0] for r in rows], fontsize=9)
    axA.set_xlabel("posterior cultural $F_{ST}$  (point = median, bar = 95% HDI)")
    axA.set_title("A. Posterior between-community structure exceeds zero in 3 of 4 proxies",
                  fontsize=9.5, loc="left")
    axA.set_xlim(left=0)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # --- Panel B: umu west-to-east style gradient ---
    udf = umu_mod.load()
    order = ["Rano Kau", "Vinapu", "Maunga Orito", "Hanga Poukura", "Vaihu"]
    udf = udf.set_index("quadrangle").loc[order]
    C = udf[umu_mod.STYLES].values.astype(float)
    P = C / C.sum(1, keepdims=True)
    bottom = np.zeros(len(order))
    scol = {"Rectangular": "#8c510a", "Pentagonal": "#01665e",
            "Circular": "#dfc27d", "Irregular": "#bbbbbb"}
    for k, s in enumerate(umu_mod.STYLES):
        axB.bar(range(len(order)), P[:, k], bottom=bottom, label=s,
                color=scol[s], edgecolor="white", lw=0.5)
        bottom += P[:, k]
    axB.set_xticks(range(len(order)))
    axB.set_xticklabels([o.replace(" ", "\n") for o in order], fontsize=8)
    axB.set_ylabel("style frequency")
    axB.set_title("B. Umu oven-style gradient, west → east", fontsize=9.5, loc="left")
    # headroom above the stacked bars (which fill 0-1) so the legend never overlaps them
    axB.set_ylim(0, 1.26)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.legend(fontsize=7.5, loc="upper center", ncol=4, frameon=False,
               columnspacing=1.0, handletextpad=0.4)

    # --- Panel C: moai per-locus posterior F_ST (median + 95% HDI) ---
    marg = B["moai"]["marginal"]
    feats, med, lo, hi = [], [], [], []
    for a, rec in marg.items():
        feats.append(a.replace("_", " ").title().replace("Plan", "plan"))
        med.append(rec["median"]); lo.append(rec["hdi_lo"]); hi.append(rec["hdi_hi"])
    ordr = np.argsort(med)
    feats = [feats[i] for i in ordr]
    med = np.array(med)[ordr]; lo = np.array(lo)[ordr]; hi = np.array(hi)[ordr]
    yy = range(len(feats))
    # a locus "carries" the signal if its 95% HDI clears the 0.01 structure floor
    cols = ["#1a5276" if v > 0.01 else "#aaaaaa" for v in lo]
    axC.barh(list(yy), med, color=cols, zorder=2)
    axC.hlines(list(yy), lo, hi, color="#222222", lw=1.2, zorder=3)
    axC.set_yticks(list(yy))
    axC.set_yticklabels(feats, fontsize=8)
    axC.set_xlabel("posterior marginal $F_{ST}$ (median, bar = 95% HDI)")
    axC.set_title("C. Moai: which style features carry the signal", fontsize=9.5, loc="left")
    for sp in ("top", "right"):
        axC.spines[sp].set_visible(False)

    # --- Panel D: mata'a isolation by distance with posterior credible band ---
    lw, _ = tables_io.stem_length_width()
    names = list(lw.index)
    keep = [i for i, nm_ in enumerate(names) if spatial.has_coords(nm_)]
    ng = [names[i] for i in keep]
    Dgeo = spatial.distance_matrix(ng)
    Dd2 = popgen.neiman_d2(lw.values.astype(float)[keep])
    iu = np.triu_indices_from(Dgeo, 1)
    x, y = Dgeo[iu], Dd2[iu]
    ibd = B["mataa"]["ibd"]["plain"]
    # the model standardizes both axes; map the standardized slope back to data units
    mx, sx, my, sy = x.mean(), x.std(), y.mean(), y.std()
    xs = np.linspace(x.min(), x.max(), 50)
    def _line(b_std):
        return my + b_std * (sy / sx) * (xs - mx)
    axD.scatter(x, y, s=26, color="#3a6ea5", alpha=0.8, zorder=3)
    axD.fill_between(xs, _line(ibd["slope_hdi_lo"]), _line(ibd["slope_hdi_hi"]),
                     color="#c0392b", alpha=0.18, zorder=1, label="95% credible band")
    axD.plot(xs, _line(ibd["slope_median"]), color="#c0392b", lw=1.6, zorder=2,
             label=f"slope {ibd['slope_median']:+.2f}, P(slope>0)={ibd['p_positive']:.2f}")
    axD.set_xlabel("geographic distance (km)")
    axD.set_ylabel(r"compositional distance (Neiman $d^2$)")
    axD.set_title("D. Mata'a isolation by distance (island-wide)",
                  fontsize=9.5, loc="left")
    axD.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        axD.spines[sp].set_visible(False)

    fig.tight_layout()
    _savefig(fig, path, 160)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------
# Literature genetic F_ST values for the convergence figure (cited in caption,
# not recomputed here): Dudgeon 2008 between-site aDNA F_ST; Basque reference
# (Perez-Miranda et al. 2005, via Dudgeon 2008) as a typical structured human pop.
GEN_RAPANUI = 0.131
GEN_BASQUE = 0.0053


def fig_convergence(path="figures/fig5_convergence.png"):
    """Genes and artifacts converge on the same hyperlocal structure."""
    B = _bayes()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.3))

    # --- Panel A: posterior cultural F_ST vs genetic F_ST ---
    m = B["mataa"]["headline"]["lengthwidth"]
    u = B["umu"]["headline"]
    o = B["moai"]["headline"]
    rows = [  # (label, median, lo, hi, color)
        ("Rapa Nui genes\n(aDNA; Dudgeon 2008)", GEN_RAPANUI, np.nan, np.nan, "#6a3d9a"),
        ("typical human pop.\n(Basque, reference)", GEN_BASQUE, np.nan, np.nan, "#b39ddb"),
        ("umu oven style", u["median"], u["hdi_lo"], u["hdi_hi"], "#1a5276"),
        ("moai style (multilocus)", o["median"], o["hdi_lo"], o["hdi_hi"], "#1a5276"),
        ("mata'a stem L×W", m["median"], m["hdi_lo"], m["hdi_hi"], "#1a5276"),
    ]
    ys = np.arange(len(rows))[::-1]
    top = ys.max()
    for y, (lab, med, lo, hi, col) in zip(ys, rows):
        if np.isfinite(lo):
            axA.plot([lo, hi], [y, y], color=col, lw=2.6, zorder=2)
        axA.scatter([med], [y], s=95, color=col, zorder=3)
        # value label to the right of the point (or its HDI bar), clear of the axis
        anchor = hi if np.isfinite(lo) else med
        axA.annotate(f"{med:.3f}", (anchor, y), textcoords="offset points",
                     xytext=(9, 0), va="center", ha="left", fontsize=8.5)
    axA.axvline(0.01, color="#888888", ls="--", lw=1, zorder=1)
    axA.text(0.01, top + 0.55, "0.01 (reference)", rotation=0, ha="left",
             va="bottom", fontsize=7.4, color="#777777")
    axA.scatter([], [], s=95, color="#6a3d9a", label="genetic $F_{ST}$")
    axA.scatter([], [], s=95, color="#1a5276", label="cultural $F_{ST}$ (this study)")
    axA.set_yticks(ys)
    axA.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    axA.set_xlabel("$F_{ST}$  (cultural: posterior median, bar = 95% HDI)")
    axA.set_title("A. Genes and artifacts both record intra-island structure",
                  fontsize=9.5, loc="left")
    axA.set_xlim(-0.005, 0.18)
    axA.set_ylim(-0.6, top + 1.1)
    axA.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.04))
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # --- Panel B: island-wide distance slope is a regional contrast ---
    def slopes(mat):
        names = [n for n in mat.index if spatial.has_coords(n)]
        counts = mat.loc[names].values.astype(float)
        Dgeo = spatial.distance_matrix(names)
        Dcomp = popgen.neiman_d2(counts)
        region = np.array([0 if n in PARCELS_FIG else 1 for n in names])
        Dregion = (region[:, None] != region[None, :]).astype(float)
        plain = bayes.ibd_regression(Dcomp, Dgeo)
        ctrl = bayes.ibd_regression(Dcomp, Dgeo, region=Dregion)
        return plain, ctrl

    lw, _ = tables_io.stem_length_width()
    ss, _ = tables_io.stem_shoulder_shape()
    p_lw, c_lw = slopes(lw)
    p_ss, c_ss = slopes(ss)
    groups = ["mata'a\nstem L×W", "mata'a\nshape×shoulder"]
    x = np.arange(len(groups))
    w = 0.36
    pl = [p_lw, p_ss]
    ct = [c_lw, c_ss]

    def _bars(offset, recs, color, label):
        meds = [r["slope_median"] for r in recs]
        los = [r["slope_median"] - r["slope_hdi_lo"] for r in recs]
        his = [r["slope_hdi_hi"] - r["slope_median"] for r in recs]
        axB.bar(x + offset, meds, w, color=color, label=label, zorder=2)
        axB.errorbar(x + offset, meds, yerr=[los, his], fmt="none",
                     ecolor="#222222", elinewidth=1.1, capsize=3, zorder=3)

    _bars(-w/2, pl, "#3a6ea5", "distance only")
    _bars(+w/2, ct, "#c0392b", "controlling for region")
    axB.axhline(0, color="#333333", lw=0.8, zorder=1)
    axB.set_xticks(x)
    axB.set_xticklabels(groups, fontsize=8.5)
    axB.set_ylabel("posterior distance slope (median, bar = 95% HDI)")
    axB.set_ylim(-0.9, 1.15)
    axB.set_title("B. The island-wide distance signal is a regional contrast",
                  fontsize=9.5, loc="left")
    # one short takeaway, in clear space at the bottom; full reading is in the caption
    axB.text(0.5, -0.78, "controlling for region collapses the slope to zero:\n"
             "a step between regional pools, not a cline", ha="center", va="center",
             fontsize=7.8, color="#555555", style="italic")
    axB.legend(fontsize=8, frameon=False, loc="upper right", ncol=1)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle("Convergent evidence for hyperlocal community structure on Rapa Nui",
                 fontsize=12, y=0.99)
    _savefig(fig, path, 160)
    plt.close(fig)
    print("wrote", path)


def _pie_on_map(ax, x, y, fracs, colors, r, edge="#222222"):
    """Draw a small pie (wedges) centered at pixel (x, y) on a map axis."""
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
    """Small scale bar of length `km` at axis-fraction (fx, fy)."""
    x0, _ = figbase.lonlat_to_px(lon, lat, origin, z)
    x1, _ = figbase.lonlat_to_px(lon + km / figbase.km_per_deg_lon(lat), lat, origin, z)
    px = abs(x1 - x0)
    bx, by = fx * W, fy * H
    ax.plot([bx, bx + px], [by, by], "k-", lw=2.5, zorder=10, path_effects=_halo())
    ax.annotate(f"{km:g} km", (bx + px / 2, by - 0.018 * H), ha="center",
                fontsize=7, zorder=10, path_effects=_halo())


def _pie_callout(ax, x, y, dx, dy, fracs, colors, r, label, ha="center"):
    """Dot at the true location (x, y), pie pulled to (x+dx, y+dy) with a leader."""
    px, py = x + dx, y + dy
    ax.plot([x, px], [y, py], color="#333333", lw=0.6, zorder=5,
            path_effects=_halo())
    ax.scatter([x], [y], s=16, marker="o", facecolor="white", edgecolor="#222222",
               linewidths=0.8, zorder=7)
    _pie_on_map(ax, px, py, fracs, colors, r)
    ax.annotate(label, (px, py + r * 1.35), ha=ha, va="top", fontsize=7.5,
                zorder=8, path_effects=_halo())


def fig_variability(path="figures/fig4_variability.png"):
    """Geography of the variability: where each proxy's classes differ in space."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    axA, axB, axC = axes

    # --- Panel A: mata'a stem class composition across the SW cluster ---
    # collapse the 14 L×W classes to the 4 stem-length classes (first letter A-D)
    lw, _ = tables_io.stem_length_width()
    length_class = {}
    for cls in lw.columns:
        length_class.setdefault(cls[0], []).append(cls)
    Lkeys = sorted(length_class)                       # A, B, C, D
    Lcol = {"A": "#1b9e77", "B": "#d95f02", "C": "#7570b3", "D": "#e7298a"}
    place = {"Orito": "Maunga Orito"}
    # exploded-callout offsets (fractions of panel W,H) so the tight cluster reads
    aoff = {"Ahu Tautira": (-0.17, -0.05), "Orongo": (-0.10, 0.16),
            "Rano Kau": (-0.20, 0.02), "Vinapu": (0.04, 0.20),
            "Orito": (0.05, -0.16)}
    oA, zA, WA, HA = _relief_panel(axA, -109.462, -109.355, -27.205, -27.130, 13,
                                   "A. Mata'a stem class, SW cluster (<5.5 km)")
    for s in ["Ahu Tautira", "Orongo", "Rano Kau", "Vinapu", "Orito"]:
        cnt = lw.loc[s]
        fracs = [cnt[length_class[k]].sum() / cnt.sum() for k in Lkeys]
        x, y = figbase.lonlat_to_px(*figbase.PLACES[place.get(s, s)], oA, zA)
        dx, dy = aoff[s]
        _pie_callout(axA, x, y, dx * WA, dy * HA, fracs,
                     [Lcol[k] for k in Lkeys], r=0.034 * WA, label=s)
    _km_bar(axA, oA, zA, -27.20, -109.458, WA, HA, 1, fx=0.72, fy=0.94)
    for k in Lkeys:
        axA.scatter([], [], s=55, marker="o", c=Lcol[k], label=f"length {k}")
    axA.legend(loc="upper right", fontsize=7, framealpha=0.9, title="stem length")

    # --- Panel B: umu oven-style composition along the south coast ---
    udf = umu_mod.load().set_index("quadrangle")
    scol = {"Rectangular": "#8c510a", "Pentagonal": "#01665e",
            "Circular": "#dfc27d", "Irregular": "#bbbbbb"}
    boff = {"Rano Kau": (-12, 0, "right"), "Vinapu": (0, 13, "center"),
            "Maunga Orito": (0, -15, "center"), "Hanga Poukura": (0, 14, "center"),
            "Vaihu": (0, 14, "center")}
    oB, zB, WB, HB = _relief_panel(axB, -109.452, -109.318, -27.205, -27.130, 13,
                                   "B. Umu oven style, west to east")
    for q in ["Rano Kau", "Vinapu", "Maunga Orito", "Hanga Poukura", "Vaihu"]:
        c = udf.loc[q, umu_mod.STYLES].astype(float)
        fr = (c / c.sum()).values
        x, y = figbase.lonlat_to_px(*figbase.PLACES[q], oB, zB)
        _pie_on_map(axB, x, y, fr, [scol[s] for s in umu_mod.STYLES], r=0.038 * WB)
        dx, dy, ha = boff[q]
        axB.annotate(q.replace(" ", "\n"), (x, y), textcoords="offset points",
                     xytext=(dx, dy), ha=ha, fontsize=7, zorder=8,
                     path_effects=_halo())
    _km_bar(axB, oB, zB, -27.20, -109.448, WB, HB, 2)
    for s in umu_mod.STYLES:
        axB.scatter([], [], s=55, marker="o", c=scol[s], label=s)
    axB.legend(loc="upper left", fontsize=7, framealpha=0.9, title="oven rim style")

    # --- Panel C: moai head-plan shape across the island (per statue) ---
    md = moai_mod.load()
    hp = moai_mod._clean(md["HEAD_PLAN_SHAPE"])
    hcol = {"Inv_Trapezoid": "#d7191c", "Rectangular": "#2c7bb6",
            "Trapezoid": "#fdae61", "Round": "#1a9641"}
    oC, zC, WC, HC = _relief_panel(axC, -109.48, -109.21, -27.215, -27.040, 13,
                                   "C. Moai head-plan shape (per statue)")
    def P(lo, la): return figbase.lonlat_to_px(lo, la, oC, zC)
    uns = md[hp.isna()]
    xs, ys = zip(*[P(lo, la) for lo, la in zip(uns.longitude, uns.latitude)])
    axC.scatter(xs, ys, s=9, c="#d9d9d9", edgecolors="none", zorder=4)
    for st, col in hcol.items():
        g = md[hp == st]
        if len(g) == 0:
            continue
        xs, ys = zip(*[P(lo, la) for lo, la in zip(g.longitude, g.latitude)])
        axC.scatter(xs, ys, s=34, c=col, edgecolors="white", linewidths=0.5,
                    zorder=5, label=f"{st.replace('_', ' ')} ({len(g)})")
    axC.scatter([], [], s=9, c="#d9d9d9", label="not scored")
    _km_bar(axC, oC, zC, -27.205, -109.47, WC, HC, 5)
    axC.legend(loc="lower right", fontsize=6.8, framealpha=0.92, title="head plan")

    fig.suptitle("Geography of the variability: class composition differs between "
                 "nearby places in each proxy", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(fig, path, 160)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    warnings.filterwarnings("ignore")
    for fn in (fig_concept, fig_results, fig_convergence, fig_map, fig_variability):
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"[{fn.__name__} FAILED] {e}")
            traceback.print_exc()
