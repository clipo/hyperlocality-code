#!/usr/bin/env python
"""Bayesian tests of the costly-signaling and hyperlocality models against the ahu viewshed.

Three analyses, written to output/viewshed_models.json:

1. SITED-FOR-VISIBILITY (signaling): compare the observed ahu to a random-siting null -- many
   replicates of 57 monuments dropped on the same land surface -- for mean visible area, sea
   fraction, intervisibility degree, mutual edges, and network modularity. If the observed
   monuments are sited to be seen, they exceed the null; the posterior P(observed > null) and a
   standardized effect quantify the support.

2. SIZE <-> AUDIENCE (signaling escalation): Bayesian regression of moai height on the size of
   its audience (number of ahu that can see it, and visible land area). A positive slope means
   bigger signals sit where more can see them.

3. VISUAL COMMUNITY STRUCTURE (hyperlocality): is the mutual-visibility network more modular,
   and at a finer spatial scale, than the null -- i.e. do ahu form locally-bounded visual
   communities consistent with small interacting groups?

Pure numpy/scipy/networkx + GDAL over the local 5 m DEM; GPU-free. One-time precompute.
The DEM is the licensed 50 cm Vricon DTM (worked at 5 m), not redistributed; point
RAPANUI_DEM at your own copy. The committed output (output/viewshed_models.json)
lets the figure reproduce without the DEM.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
from pathlib import Path

import numpy as np
import networkx as nx
from osgeo import gdal, osr

gdal.UseExceptions()

DEM = os.environ.get("RAPANUI_DEM", "/home/clipo/rapanui_dem/dem_5m.tif")
_REPO = Path(__file__).resolve().parent.parent
VS = _REPO / "data" / "viewshed" / "ahu_viewshed.json"
OUT = _REPO / "output" / "viewshed_models.json"
OBS_H, CURV, MAXDIST, SEA_LEVEL = 1.7, 0.85714, 15000, 2.0
OOR, NEVER = 1.0e6, 60.0
K = 60                      # null replicates
HMODE = gdal.GVOT_MIN_TARGET_HEIGHT_FROM_GROUND


def _net_stats(minh, dist, heights, N):
    """Intervisibility graph stats at each node's own moai height."""
    def h(j):
        return heights[j] if heights[j] else 4.0
    deg = [sum(1 for j in range(N) if i != j and minh[i][j] is not None and minh[i][j] <= h(j))
           for i in range(N)]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    diam = []
    for i in range(N):
        for j in range(i + 1, N):
            ab = minh[i][j] is not None and minh[i][j] <= h(j)
            ba = minh[j][i] is not None and minh[j][i] <= h(i)
            if ab and ba:
                G.add_edge(i, j)
    mutual = G.number_of_edges()
    try:
        comms = list(nx.community.greedy_modularity_communities(G))
        Q = nx.community.modularity(G, comms) if mutual else 0.0
        for c in comms:
            if len(c) >= 2:
                ds = [dist[i][j] for i in c for j in c if i < j]
                diam.append(max(ds) / 1000 if ds else 0)
    except Exception:  # noqa: BLE001
        Q, comms = 0.0, []
    return {"mean_degree": float(np.mean(deg)), "mutual": mutual, "modularity": round(Q, 3),
            "n_comm": sum(1 for c in comms if len(c) >= 2),
            "comm_diam_km": round(float(np.mean(diam)), 2) if diam else 0.0}


def _viewshed_set(band, gt, W, Hh, sea, nd, pts):
    """For a set of (x,y) points: per-point visible land/sea km², and the NxN min-height matrix."""
    n = len(pts)
    minh = [[None] * n for _ in range(n)]
    vis_area, vis_sea = [], []
    cell = abs(gt[1] * gt[5])
    for i, (x, y) in enumerate(pts):
        vds = gdal.ViewshedGenerate(band, "MEM", "", [], x, y, OBS_H, 0.0, 1.0, 0.0, OOR,
                                    (nd if nd is not None else -9999.0), CURV, gdal.GVM_Edge,
                                    MAXDIST, heightMode=HMODE)
        vb = vds.GetRasterBand(1).ReadAsArray(); vgt = vds.GetGeoTransform()
        vW, vH = vds.RasterXSize, vds.RasterYSize
        coff = int(round((vgt[0] - gt[0]) / gt[1])); roff = int(round((vgt[3] - gt[3]) / gt[5]))
        sw = sea[roff:roff + vH, coff:coff + vW]
        ground = (vb <= 0.0) & (vb > -1e5)
        if ground.shape == sw.shape:
            vis_sea.append(float((ground & sw).sum()) * cell / 1e6)
            vis_area.append(float(ground.sum()) * cell / 1e6)
        else:
            vis_sea.append(0.0); vis_area.append(float(ground.sum()) * cell / 1e6)
        for j, (bx, by) in enumerate(pts):
            if i == j:
                continue
            px = int((bx - vgt[0]) / vgt[1]); py = int((by - vgt[3]) / vgt[5])
            if 0 <= px < vW and 0 <= py < vH:
                v = float(vb[py, px])
                if v < NEVER:
                    minh[i][j] = max(v, 0.0)
        vds = None
    return minh, vis_area, vis_sea


def main():
    vs = json.loads(VS.read_text()); ahu = vs["ahu"]; N = len(ahu)
    heights = [a.get("max_h") for a in ahu]
    obs_net = _net_stats(vs["min_height"], vs["dist"], heights, N)
    obs = {**obs_net,
           "mean_visible_km2": round(st.mean(a["visible_km2"] for a in ahu), 1),
           "mean_sea_frac": round(st.mean((a.get("visible_sea_km2", 0) / a["visible_km2"])
                                          if a["visible_km2"] else 0 for a in ahu), 3)}

    ds = gdal.Open(DEM); band0 = ds.GetRasterBand(1); gt = ds.GetGeoTransform()
    W, Hh = ds.RasterXSize, ds.RasterYSize
    nd = band0.GetNoDataValue(); elev = band0.ReadAsArray()
    sea = (elev == nd) if nd is not None else (elev < SEA_LEVEL)
    land_idx = np.argwhere((elev >= SEA_LEVEL) & (elev != nd))   # (row,col) land pixels
    filled = np.where(sea, 0.0, elev).astype(np.float32)
    mem = gdal.GetDriverByName("MEM").Create("", W, Hh, 1, gdal.GDT_Float32)
    mem.SetGeoTransform(gt); mem.SetProjection(ds.GetProjection())
    mem.GetRasterBand(1).WriteArray(filled); band = mem.GetRasterBand(1)

    rng = np.random.default_rng(20260626)
    null = {k: [] for k in obs}
    for r in range(K):
        sel = land_idx[rng.choice(len(land_idx), N, replace=False)]
        pts = [(gt[0] + (c + 0.5) * gt[1], gt[3] + (rw + 0.5) * gt[5]) for rw, c in sel]
        minh, va, vsea = _viewshed_set(band, gt, W, Hh, sea, nd, pts)
        dist = [[round(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
                 for j in range(N)] for i in range(N)]
        hh = list(heights); rng.shuffle(hh)             # observed height distribution, randomized
        ns = _net_stats(minh, dist, hh, N)
        null["mean_visible_km2"].append(round(float(np.mean(va)), 1))
        null["mean_sea_frac"].append(round(float(np.mean([s / a if a else 0 for s, a in zip(vsea, va)])), 3))
        for k in obs_net:
            null[k].append(ns[k])
        if (r + 1) % 10 == 0:
            print(f"  null replicate {r+1}/{K}")

    tests = {}
    for k, ov in obs.items():
        arr = np.array(null[k], float)
        p_gt = float((arr < ov).sum() + 0.5) / (len(arr) + 1)   # posterior P(observed > null)
        z = (ov - arr.mean()) / (arr.std() + 1e-9)
        tests[k] = {"observed": ov, "null_mean": round(float(arr.mean()), 2),
                    "null_lo": round(float(np.percentile(arr, 2.5)), 2),
                    "null_hi": round(float(np.percentile(arr, 97.5)), 2),
                    "p_gt": round(p_gt, 3), "z": round(float(z), 2)}

    # size <-> audience: Bayesian regression height ~ degree, height ~ visible land
    def bayes_slope(x, y, n_iter=4000, seed=1):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if len(x) < 8:
            return None
        X = np.column_stack([np.ones(len(x)), (x - x.mean()) / (x.std() + 1e-9)])
        XtXinv = np.linalg.inv(X.T @ X); rng2 = np.random.default_rng(seed)
        beta = XtXinv @ X.T @ y; sig2 = float(np.var(y - X @ beta)) or 1.0
        bs = []
        for _ in range(n_iter):
            L = np.linalg.cholesky(sig2 * XtXinv)
            beta = (XtXinv @ X.T @ y) + L @ rng2.standard_normal(2)
            resid = y - X @ beta
            sig2 = 1.0 / rng2.gamma(2 + len(x) / 2, 1 / (1 + (resid @ resid) / 2))
            bs.append(beta[1])
        b = np.array(bs)
        return {"beta": round(float(np.median(b)), 3), "lo": round(float(np.percentile(b, 2.5)), 3),
                "hi": round(float(np.percentile(b, 97.5)), 3),
                "credible": bool(np.percentile(b, 2.5) > 0 or np.percentile(b, 97.5) < 0), "n": len(x)}
    hpairs = [(a, a.get("max_h")) for a in ahu if a.get("max_h")]
    H = [p[1] for p in hpairs]
    deg = {i: 0 for i in range(N)}
    for i, a in enumerate(ahu):
        a_deg = sum(1 for j in range(N) if i != j and vs["min_height"][i][j] is not None
                    and vs["min_height"][i][j] <= (heights[j] or 4.0))
        deg[i] = a_deg
    size = {
        "vs_degree": bayes_slope([deg[i] for i, a in enumerate(ahu) if a.get("max_h")], H),
        "vs_visible_land": bayes_slope([a.get("visible_land_km2", 0) for a in ahu if a.get("max_h")], H),
    }

    OUT.write_text(json.dumps({"obs": obs, "tests": tests, "size": size, "K": K,
                               "params": {"obs_h": OBS_H, "maxdist_m": MAXDIST, "res_m": 5}}))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
