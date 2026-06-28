"""S-3: coastal, siting-matched null for the ahu intervisibility network.

The committed intervisibility test (viewshed_models.py) compares the observed
monument network to random sitings drawn from ALL land pixels. But real ahu are
coastal monuments; a ring of coastal points around a central massif is forced into
a sparse, modular visibility graph by terrain alone. That null therefore confounds
deliberate bounding with coastal geometry. Here we restrict the random sitings to a
coastal band matched to where the ahu actually sit (land within the 95th-percentile
distance-to-coast of the 57 real ahu) and raise the replicate count (K>=1000). If
the observed modularity / community structure still exceeds this siting-matched
null, the bounded visual communities are a real placement signal; if not, they are
explained by coastal geometry.

Needs the licensed DEM: RAPANUI_DEM=/home/clipo/rapanui_dem/dem_5m.tif
Run:  PYTHONPATH=src RAPANUI_DEM=... python3 src/run_viewshed_coastal_null.py [K]
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from scipy import ndimage

from viewshed_models import (_net_stats, _viewshed_set, SEA_LEVEL, DEM)

gdal.UseExceptions()
_REPO = Path(__file__).resolve().parent.parent
VS = _REPO / "data" / "viewshed" / "ahu_viewshed.json"
OUT = _REPO / "output" / "viewshed_coastal_null.json"
K = int(sys.argv[1]) if len(sys.argv) > 1 else 800


def _ct(a, b):
    s = osr.SpatialReference(); s.ImportFromEPSG(a); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d = osr.SpatialReference(); d.ImportFromEPSG(b); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(s, d)


def main():
    vs = json.loads(VS.read_text()); ahu = vs["ahu"]; N = len(ahu)
    heights = [a.get("max_h") for a in ahu]
    obs_net = _net_stats(vs["min_height"], vs["dist"], heights, N)
    print(f"observed: modularity {obs_net['modularity']}, n_comm {obs_net['n_comm']}, "
          f"diam {obs_net['comm_diam_km']} km, mean_degree {obs_net['mean_degree']:.2f}, "
          f"mutual {obs_net['mutual']}")

    ds = gdal.Open(DEM); band0 = ds.GetRasterBand(1); gt = ds.GetGeoTransform()
    W, Hh = ds.RasterXSize, ds.RasterYSize
    nd = band0.GetNoDataValue(); elev = band0.ReadAsArray()
    sea = (elev == nd) if nd is not None else (elev < SEA_LEVEL)
    land = (elev >= SEA_LEVEL) & (elev != nd)
    filled = np.where(sea, 0.0, elev).astype(np.float32)
    mem = gdal.GetDriverByName("MEM").Create("", W, Hh, 1, gdal.GDT_Float32)
    mem.SetGeoTransform(gt); mem.SetProjection(ds.GetProjection())
    mem.GetRasterBand(1).WriteArray(filled); band = mem.GetRasterBand(1)

    # distance (m) of every land pixel to the nearest sea pixel
    px_m = abs(gt[1])
    dist_coast = ndimage.distance_transform_edt(~sea) * px_m   # 0 on sea, grows inland

    # real ahu pixel positions -> their distance-to-coast (the siting profile to match)
    ct = _ct(4326, 32712)
    ahu_dc = []
    for a in ahu:
        x, y, _ = ct.TransformPoint(a["lon"], a["lat"])
        c = int((x - gt[0]) / gt[1]); r = int((y - gt[3]) / gt[5])
        if 0 <= r < Hh and 0 <= c < W:
            ahu_dc.append(float(dist_coast[r, c]))
    ahu_dc = np.array(ahu_dc)
    print(f"real ahu distance-to-coast: median {np.median(ahu_dc):.0f} m, "
          f"mean {ahu_dc.mean():.0f} m, 95th {np.percentile(ahu_dc,95):.0f} m, "
          f"max {ahu_dc.max():.0f} m")

    # Distance-to-coast-MATCHED siting: each random monument is placed at a land pixel
    # whose distance-to-coast matches a randomly chosen ahu's (within +-TOL). This holds
    # the coastal-ring geometry fixed, so any modularity excess cannot be a coastal
    # artifact. (A simple band threshold is too loose on so small an island.)
    land_rc = np.argwhere(land & (dist_coast > 0))
    land_d = dist_coast[land_rc[:, 0], land_rc[:, 1]]
    order = np.argsort(land_d); land_rc = land_rc[order]; land_d = land_d[order]
    TOL = 150.0
    print(f"distance-to-coast-matched siting (tol +-{TOL:.0f} m); K={K}")

    rng = np.random.default_rng(20260628)
    keys = ["modularity", "n_comm", "comm_diam_km", "mean_degree", "mutual"]
    null = {k: [] for k in keys}
    for rep in range(K):
        targets = rng.choice(ahu_dc, N, replace=True)
        rows = []
        for t in targets:
            lo = int(np.searchsorted(land_d, t - TOL)); hi = int(np.searchsorted(land_d, t + TOL))
            idx = int(rng.integers(lo, hi)) if hi > lo else int(np.argmin(np.abs(land_d - t)))
            rows.append(land_rc[idx])
        sel = np.array(rows)
        pts = [(gt[0] + (c + 0.5) * gt[1], gt[3] + (rw + 0.5) * gt[5]) for rw, c in sel]
        minh, _, _ = _viewshed_set(band, gt, W, Hh, sea, nd, pts)
        dist = [[round(math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
                 for j in range(N)] for i in range(N)]
        hh = list(heights); rng.shuffle(hh)
        ns = _net_stats(minh, dist, hh, N)
        for k in keys:
            null[k].append(ns[k])
        if (rep + 1) % 25 == 0:
            print(f"  matched-siting null replicate {rep+1}/{K}", flush=True)
        if (rep + 1) % 50 == 0:   # checkpoint so a disconnect/kill keeps partial progress
            (OUT.parent / "viewshed_coastal_null_partial.json").write_text(
                json.dumps({"obs": obs_net, "K_done": rep + 1, "null_partial": null}))

    tests = {}
    for k in keys:
        arr = np.array(null[k], float); ov = obs_net[k]
        # P(observed > null) for modularity/n_comm; report both tails via z
        p_gt = float((arr < ov).sum() + 0.5) / (len(arr) + 1)
        z = (ov - arr.mean()) / (arr.std() + 1e-9)
        tests[k] = {"observed": ov, "null_mean": round(float(arr.mean()), 3),
                    "null_lo": round(float(np.percentile(arr, 2.5)), 3),
                    "null_hi": round(float(np.percentile(arr, 97.5)), 3),
                    "p_obs_gt_null": round(p_gt, 4), "z": round(float(z), 2)}
        print(f"{k:14s} obs {ov:8.3f}  null {tests[k]['null_mean']:8.3f} "
              f"[{tests[k]['null_lo']}, {tests[k]['null_hi']}]  "
              f"P(obs>null)={p_gt:.3f}  z={tests[k]['z']}")

    OUT.write_text(json.dumps({"obs": obs_net, "tests": tests, "K": K,
                               "null_model": "distance-to-coast-matched siting",
                               "tol_m": TOL,
                               "ahu_dc_median_m": float(np.median(ahu_dc)),
                               "ahu_dc_mean_m": float(ahu_dc.mean())}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
