"""Geographic positions of the mata'a assemblages, read from the provenance map.

Coordinates were extracted from the vector source `data/Figure3.svg`: each label's
leader-line arrowhead pins it to its find location, and the scale bar's evenly
spaced minor ticks fix the unit->km conversion (4 km divided into 4 ticks =>
~18.21 SVG units per km). Coordinates are in SVG user units (x increases east,
y increases north); only relative configuration matters for the spatial tests.

The two parcel surveys appear as a single arrow each, so the three Te Kahurea
parcels (6-8) share one position and the three Te Miro Oone parcels (9-11) share
another. This creates within-survey distance ties -- a real limitation, flagged
where it matters. Mantel / isolation-by-distance tests are rank-based and so are
robust to moderate coordinate error and to the scale-bar's absolute calibration
(the IBD *pattern* is scale-invariant; only the per-km slope depends on it).
"""
import numpy as np

UNITS_PER_KM = 18.21  # from Figure3.svg scale-bar minor ticks (1 km per tick)

# Raw leader-arrow target coordinates (SVG units), assigned to names by their
# relative arrangement and cross-checked against real Rapa Nui geography
# (Orongo SW-most, Ahu Tautira north, Orito north of Vinapu, Te Kahurea east).
COORDS_UNITS = {
    "Ahu Tautira": (162.97, 280.53),
    "Orongo":      (120.61, 203.36),
    "Orito":       (216.38, 233.12),
    "Rano Kau":    (185.89, 200.84),
    "Vinapu":      (217.22, 224.60),
    # parcel surveys (shared positions within each survey)
    "Te Miro Oone": (337.59, 306.81),   # parcels 9, 10, 11
    "Te Kahurea":   (409.59, 352.54),   # parcels 6, 7, 8
}

# Map every assemblage label that appears in the count tables to a position.
PARCEL_SURVEY = {
    "Parcel 6": "Te Kahurea", "Parcel 7": "Te Kahurea", "Parcel 8": "Te Kahurea",
    "Parcel 9": "Te Miro Oone", "Parcel 10": "Te Miro Oone", "Parcel 11": "Te Miro Oone",
}


def _km(name):
    """(x, y) in km for an assemblage name (resolving parcels to their survey)."""
    key = PARCEL_SURVEY.get(name, name)
    if key not in COORDS_UNITS:
        # tolerate alternate spellings used elsewhere in the repo
        alt = name.replace("_", " ")
        key = PARCEL_SURVEY.get(alt, alt)
    x, y = COORDS_UNITS[key]
    return np.array([x, y]) / UNITS_PER_KM


def distance_matrix(names):
    """Symmetric pairwise great-island (planar) distance matrix in km."""
    pts = np.array([_km(n) for n in names])
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    return d


def has_coords(name):
    key = PARCEL_SURVEY.get(name, name)
    if key in COORDS_UNITS:
        return True
    alt = PARCEL_SURVEY.get(name.replace("_", " "), name.replace("_", " "))
    return alt in COORDS_UNITS


if __name__ == "__main__":
    names = ["Ahu Tautira", "Orongo", "Orito", "Rano Kau", "Vinapu",
             "Te Kahurea", "Te Miro Oone"]
    D = distance_matrix(names)
    print("pairwise distances (km):")
    print("            " + "".join(f"{n[:6]:>8}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>12}" + "".join(f"{D[i, j]:8.1f}" for j in range(len(names))))
    print(f"\nisland span (max pairwise): {D.max():.1f} km")
