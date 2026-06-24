"""Umu pae (stone-lined earth oven) hyperlocality from McCoy 1978 Table 1.

The cleanest of the proxies for the class-frequency method: McCoy's Table 1 is
already a deme x class contingency table -- oven STYLE (the "allele": Rectangular,
Pentagonal, Circular, Irregular) tallied across five south-coast survey
QUADRANGLES (the "deme": Rano Kau, Vinapu, Maunga Orito, Hanga Poukura, Vaihu).
n = 344 classifiable ovens; "Incomplete" rims are unclassifiable and dropped.

One multi-allelic locus, so popgen.gst_permutation applies directly (no multilocus
pooling needed). Coordinates are map-read centroids from McCoy's Fig. 4 survey
grid: a regular ~4.5 km lattice (scale bar 2 km ~ 116 px; column spacing ~261 px),
two rows (I,II south; IV,V,VI north), so absolute km are approximate and the IBD
Mantel (only 5 demes / 10 pairs) is EXPLORATORY -- lean on F_ST and the monotone
W->E compositional gradient, not the distance slope (cf. the mata'a caveats).
"""
import numpy as np
import pandas as pd


STYLES = ["Rectangular", "Pentagonal", "Circular", "Irregular"]  # Incomplete dropped

# Map-read centroids from Fig. 4 (km; x east, y north). ~4.5 km square cells,
# bottom row = I, II; top row = IV, V, VI (VI easternmost).
COORDS = {
    "Rano Kau":      (0.0, 0.0),   # I   (SW, Rano Kau crater)
    "Vinapu":        (4.5, 0.0),   # II  (S coast, E of Rano Kau)
    "Maunga Orito":  (0.0, 4.5),   # IV  (inland, obsidian source)
    "Hanga Poukura": (4.5, 4.5),   # V
    "Vaihu":         (9.0, 4.5),   # VI  (easternmost)
}


def load(path="data/umu/umu_pae_table1.csv"):
    df = pd.read_csv(path, comment="#")
    return df


def counts(df, styles=STYLES):
    """Deme x style counts matrix of CLASSIFIABLE ovens (Incomplete excluded)."""
    return df[styles].values.astype(float)


def coords_km(df):
    return np.array([COORDS[q] for q in df["quadrangle"]], float)


def distance_km(df):
    xy = coords_km(df)
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))
