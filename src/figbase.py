"""Figure infrastructure: OpenTopoMap basemap + coordinate transforms.

No geospatial libraries are available, so we fetch and stitch web-mercator tiles
with urllib + PIL directly, cache them under figures/_tiles, and provide the
lon/lat -> pixel mapping needed to overlay points. Includes a manual UTM zone-12S
inverse for the pukao coordinates and a table of known Rapa Nui place coordinates
(several pulled from the moai database) to anchor the mata'a and umu proxies.
"""
import io
import math
import os
import time
import urllib.request

import numpy as np
from PIL import Image

TILE = 256
UA = {"User-Agent": "mataa-research-figure/1.0 (clipo@binghamton.edu)"}
CACHE = "figures/_tiles"


def _lonlat_to_global_px(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n * TILE
    lr = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lr)) / math.pi) / 2.0 * n * TILE
    return x, y


# Tile sources. OpenTopoMap carries roads and town labels; the publication map
# uses ESRI World Shaded Relief instead -- a label-free terrain raster, so the
# island reads as landform (craters, the Poike peninsula, the Terevaka massif)
# rather than as modern infrastructure.
SOURCES = {
    "otm": ("https://a.tile.opentopomap.org/{z}/{x}/{y}.png", "otm"),
    "relief": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}", "esr"),
}


def _fetch_tile(z, x, y, source="otm"):
    url_tmpl, tag = SOURCES[source]
    os.makedirs(CACHE, exist_ok=True)
    path = f"{CACHE}/{tag}_{z}_{x}_{y}.png"
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    url = url_tmpl.format(z=z, x=x, y=y)
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=20).read()
            im = Image.open(io.BytesIO(data)).convert("RGB")
            im.save(path)
            time.sleep(0.3)            # be polite to the tile server
            return im
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"tile {z}/{x}/{y} failed")


def basemap(lon_min, lon_max, lat_min, lat_max, z=13, source="otm"):
    """Return (image_array, extent_px_origin, z) for a stitched web-mercator basemap.

    extent gives the global-pixel coordinates of the stitched image's top-left,
    so lon/lat can be converted to local pixels via lonlat_to_px below. `source`
    selects a tile set from SOURCES ("relief" = label-free ESRI shaded relief).
    """
    n = 2 ** z
    def xt(lon): return int((lon + 180.0) / 360.0 * n)
    def yt(lat):
        lr = math.radians(lat)
        return int((1.0 - math.asinh(math.tan(lr)) / math.pi) / 2.0 * n)
    x0, x1 = xt(lon_min), xt(lon_max)
    y0, y1 = yt(lat_max), yt(lat_min)          # note: lat_max -> smaller y
    W, H = (x1 - x0 + 1) * TILE, (y1 - y0 + 1) * TILE
    canvas = Image.new("RGB", (W, H))
    for i, xx in enumerate(range(x0, x1 + 1)):
        for j, yy in enumerate(range(y0, y1 + 1)):
            canvas.paste(_fetch_tile(z, xx, yy, source), (i * TILE, j * TILE))
    origin = (x0 * TILE, y0 * TILE)
    return np.asarray(canvas), origin, z


def lonlat_to_px(lon, lat, origin, z):
    gx, gy = _lonlat_to_global_px(lon, lat, z)
    return gx - origin[0], gy - origin[1]


def km_per_deg_lon(lat):
    return 111.320 * math.cos(math.radians(lat))


def utm12s_to_lonlat(E, N, south=True):
    """Inverse UTM (WGS84) for zone 12S -- used for the pukao coordinates."""
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    k0 = 0.9996
    E = E - 500000.0
    if south:
        N = N - 10000000.0
    M = N / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu))
    C1 = ep2 * math.cos(phi1) ** 2
    T1 = math.tan(phi1) ** 2
    N1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    R1 = a * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    D = E / (N1 * k0)
    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D ** 2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * ep2) * D ** 4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * ep2 - 3 * C1 ** 2) * D ** 6 / 720)
    lon0 = math.radians(12 * 6 - 183)
    lon = lon0 + (D - (1 + 2 * T1 + C1) * D ** 3 / 6
                  + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * ep2 + 24 * T1 ** 2)
                  * D ** 5 / 120) / math.cos(phi1)
    return math.degrees(lon), math.degrees(lat)


# Known Rapa Nui place coordinates (lon, lat). Vinapu, Ahu Tautira, Tongariki,
# Akahanga read from the moai database; crater/hill positions are standard.
PLACES = {
    "Orongo":        (-109.4439, -27.1886),
    "Rano Kau":      (-109.4392, -27.1844),
    "Vinapu":        (-109.4066, -27.1768),
    "Maunga Orito":  (-109.3790, -27.1640),
    "Ahu Tautira":   (-109.4306, -27.1473),
    "Hanga Poukura": (-109.3650, -27.1620),
    "Vaihu":         (-109.3368, -27.1492),
}

# Natural landforms used to orient the reader (lon, lat). Summit and crater
# positions are standard published values; these are geography, not sample sites.
FEATURES = {
    "Maunga Terevaka": (-109.3766, -27.0852),   # highest point (507 m), NW massif
    "Poike":           (-109.2445, -27.0915),   # eastern volcano / peninsula
    "Rano Raraku":     (-109.2890, -27.1220),   # statue-quarry crater
    "Rano Kau":        (-109.4392, -27.1844),   # SW crater (also a sample area)
    "Motu Nui":        (-109.4530, -27.2050),   # islets off the SW point
}

# Rapa Nui's own position, for the Pacific locator inset.
RAPA_NUI = (-109.35, -27.12)
