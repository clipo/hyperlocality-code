#!/usr/bin/env python
"""Ahu viewshed + intervisibility analysis over the 50 cm Vricon DTM (worked at 5 m).

For every ahu that carries moai (location + estimated moai heights from the inventory) this
computes, on the high-resolution terrain with Earth curvature + atmospheric refraction:

  - the viewshed from a viewer standing on the ahu, split into visible LAND and visible SEA;
  - a min-target-height raster (GDAL GVOT_MIN_TARGET_HEIGHT_FROM_GROUND): per cell, the minimum
    object height needed to be seen from the ahu. Sampling it at every other ahu gives, in ONE
    viewshed per observer, the minimum moai height for that ahu to be visible from this one --
    so the intervisibility network can be re-thresholded by any target height (each ahu's own
    moai height, or an interactive slider) without recomputing; and
  - inter-ahu distances.

Writes a self-contained JSON (ahu, min-height matrix, distance matrix) + per-ahu viewshed PNG
overlays. Local DEM only; needs GDAL.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

# Licensed 50 cm Vricon DTM (worked at 5 m); not redistributed. Point RAPANUI_DEM
# at your own copy. The moai location/height source is the committed in-repo CSV
# (data/moai/moai_locations_heights.csv, from the public moai database via
# src/make_moai_csv.py), so this generator needs only the DEM + that CSV -- no
# external inventory. Its output (data/viewshed/ahu_viewshed.json) is committed so
# the downstream concordance + figure reproduce without the DEM.
DEM = os.environ.get("RAPANUI_DEM", "/home/clipo/rapanui_dem/dem_5m.tif")
_REPO = Path(__file__).resolve().parent.parent
MOAI_CSV = _REPO / "data" / "moai" / "moai_locations_heights.csv"
OUT = _REPO / "data" / "viewshed" / "ahu_viewshed.json"
PNG_DIR = _REPO / "output" / "viewsheds"
OBS_H = 1.7          # viewer eye height (m)
CURV = 0.85714       # Earth curvature + standard atmospheric refraction
MAXDIST = 15000      # m line-of-sight cap
SEA_LEVEL = 2.0      # m; cells at/below this are treated as sea/coast
OOR = 1.0e6          # out-of-range / never-visible sentinel
NEVER = 60.0         # min-height above this (m) => treat as not realistically visible


def _ct(a, b):
    s = osr.SpatialReference(); s.ImportFromEPSG(a); s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    d = osr.SpatialReference(); d.ImportFromEPSG(b); d.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(s, d)


def load_ahu():
    """Group the committed ahu-placed-moai CSV by ahu; mean position + moai heights.

    Reads data/moai/moai_locations_heights.csv (from the public moai database via
    src/make_moai_csv.py). No external inventory required.
    """
    import csv
    ct = _ct(4326, 32712)
    groups = defaultdict(list)
    with open(MOAI_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["latitude"] and r["longitude"] and r["ahu"]:
                groups[r["ahu"].strip()].append(
                    {"latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
                     "total_length_cm": float(r["total_length_cm"]) if r["total_length_cm"] else None})
    out = []
    for site, ms in groups.items():
        lat = st.mean(m["latitude"] for m in ms); lon = st.mean(m["longitude"] for m in ms)
        H = [m["total_length_cm"] / 100 for m in ms if m.get("total_length_cm")]
        x, y, _ = ct.TransformPoint(lon, lat)
        out.append({"site": site, "lat": lat, "lon": lon, "x": x, "y": y, "n_moai": len(ms),
                    "max_h": (round(max(H), 1) if H else None),
                    "med_h": (round(st.median(H), 1) if H else None)})
    return out


def _dedup(ahu, thresh_m=120):
    kept = []
    for a in sorted(ahu, key=lambda r: -r["n_moai"]):
        near = next((k for k in kept if math.hypot(a["x"] - k["x"], a["y"] - k["y"]) < thresh_m), None)
        if near:
            near["n_moai"] += a["n_moai"]
            hs = [h for h in (near["max_h"], a["max_h"]) if h]
            near["max_h"] = max(hs) if hs else None
        else:
            kept.append(a)
    return kept


def main(write_png=True):
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    ds = gdal.Open(DEM); band = ds.GetRasterBand(1); gt = ds.GetGeoTransform()
    W, Hh = ds.RasterXSize, ds.RasterYSize
    nodata = band.GetNoDataValue()
    cell_area = abs(gt[1] * gt[5])
    elev = band.ReadAsArray()
    # the DTM is land-only (ocean = nodata); fill the ocean with sea level (0 m) in an in-memory
    # DEM so the sea surface is part of the viewshed, and keep the original nodata as a sea mask.
    sea = (elev == nodata) if nodata is not None else (elev < SEA_LEVEL)
    filled = np.where(sea, 0.0, elev).astype(np.float32)
    mem = gdal.GetDriverByName("MEM").Create("", W, Hh, 1, gdal.GDT_Float32)
    mem.SetGeoTransform(gt); mem.SetProjection(ds.GetProjection())
    mem.GetRasterBand(1).WriteArray(filled)
    band = mem.GetRasterBand(1)
    hmode = gdal.GVOT_MIN_TARGET_HEIGHT_FROM_GROUND

    def w2p(x, y):
        return int((x - gt[0]) / gt[1]), int((y - gt[3]) / gt[5])

    ahu = _dedup([a for a in load_ahu() if 0 <= w2p(a["x"], a["y"])[0] < W
                  and 0 <= w2p(a["x"], a["y"])[1] < Hh])
    for a in ahu:
        px, py = w2p(a["x"], a["y"]); a["elev"] = round(float(elev[py, px]), 1)
    N = len(ahu)
    print(f"{N} ahu within the DEM")

    minH = [[None] * N for _ in range(N)]          # min moai height for i to see j
    dist = [[0] * N for _ in range(N)]
    for i, a in enumerate(ahu):
        vds = gdal.ViewshedGenerate(band, "MEM", "", [], a["x"], a["y"], OBS_H, 0.0,
                                    1.0, 0.0, OOR, (nodata if nodata is not None else -9999.0),
                                    CURV, gdal.GVM_Edge, MAXDIST, heightMode=hmode)
        vb = vds.GetRasterBand(1).ReadAsArray()
        vgt = vds.GetGeoTransform(); vW, vH = vds.RasterXSize, vds.RasterYSize
        # align the viewshed window to the DEM grid for the land/sea mask
        coff = int(round((vgt[0] - gt[0]) / gt[1])); roff = int(round((vgt[3] - gt[3]) / gt[5]))
        sea_w = sea[roff:roff + vH, coff:coff + vW]
        ground = (vb <= 0.0) & (vb > -1e5)          # cells whose ground is visible
        if ground.shape == sea_w.shape:
            a["visible_sea_km2"] = round(float((ground & sea_w).sum()) * cell_area / 1e6, 2)
            a["visible_land_km2"] = round(float((ground & ~sea_w).sum()) * cell_area / 1e6, 2)
        a["visible_km2"] = round(float(ground.sum()) * cell_area / 1e6, 2)
        for j, b in enumerate(ahu):
            if i == j:
                continue
            dist[i][j] = round(math.hypot(a["x"] - b["x"], a["y"] - b["y"]))
            px = int((b["x"] - vgt[0]) / vgt[1]); py = int((b["y"] - vgt[3]) / vgt[5])
            if 0 <= px < vW and 0 <= py < vH:
                v = float(vb[py, px])
                if v < NEVER:
                    minH[i][j] = round(max(v, 0.0), 1)
        if write_png:
            _save_png(ground, vgt, a, PNG_DIR)
        vds = None
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N} viewsheds")

    for a in ahu:
        a.pop("x"); a.pop("y")
    OUT.write_text(json.dumps({
        "ahu": ahu, "min_height": minH, "dist": dist,
        "params": {"obs_h": OBS_H, "maxdist_m": MAXDIST, "res_m": 5, "curv": CURV,
                   "sea_level_m": SEA_LEVEL,
                   "dtm": "Vricon 50cm V3D DTM (worked at 5m), EGM2008"}}))
    print(f"wrote {OUT}: {N} ahu")


def _save_png(ground, vgt, a, png_dir):
    ys, xs = np.where(ground)
    if len(xs) == 0:
        a["png"] = None
        return
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    sub = ground[y0:y1 + 1, x0:x1 + 1]
    rgba = np.zeros((sub.shape[0], sub.shape[1], 4), np.uint8)
    rgba[sub] = (44, 127, 184, 105)
    ulx = vgt[0] + x0 * vgt[1]; uly = vgt[3] + y0 * vgt[5]
    lrx = vgt[0] + (x1 + 1) * vgt[1]; lry = vgt[3] + (y1 + 1) * vgt[5]
    ct = _ct(32712, 4326)
    lon0, lat0, _ = ct.TransformPoint(ulx, uly)
    lon1, lat1, _ = ct.TransformPoint(lrx, lry)
    from PIL import Image
    fn = "".join(c if c.isalnum() else "_" for c in a["site"])[:40] + ".png"
    Image.fromarray(rgba, "RGBA").save(png_dir / fn)
    a["png"] = fn
    a["png_bounds"] = [[min(lat0, lat1), min(lon0, lon1)], [max(lat0, lat1), max(lon0, lon1)]]


if __name__ == "__main__":
    main()
