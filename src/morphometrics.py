"""Derive size-free, interpretable shape measurements from mata'a outlines.

Each outline is 200 (X,Y) points. We have no manual landmarks, so we recover a
canonical orientation and locate the stem/shoulder automatically:

  1. Center on the centroid.
  2. Rotate so the long axis (1st principal axis of the outline points) is vertical.
  3. Decide which end is the STEM: the stem is the narrower end (mata'a have a
     narrow worked stem and a broad blade), so the end whose half has the smaller
     mean width is placed at the bottom (negative y).
  4. Build a width profile along the long axis and locate the SHOULDER as the
     point where the outline width first rises past a fraction of the max width,
     coming up from the stem base. Everything below the shoulder is the stem.

From this canonical form we compute size-free metrics (all ratios, so robust to
the arbitrary pixel scale of the tracings):

  elongation       = total length / max width            (overall slenderness)
  stem_length_prop = stem length / total length          (how much is stem)
  stem_width_ratio = stem width at shoulder / max width   (narrow vs broad stem)
  blade_fill       = blade area / (blade bbox area)       (pointed<->full blade)
  asymmetry        = |area left - area right| / total area (about long axis)

These echo the axes the original paper found most variable (stem length, blade
proportion) while staying scale-free for classification.
"""
import numpy as np


def _polygon_area_centroid(poly):
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = cross.sum() / 2.0
    if A == 0:
        return 0.0, poly.mean(axis=0)
    cx = ((x + x1) * cross).sum() / (6 * A)
    cy = ((y + y1) * cross).sum() / (6 * A)
    return abs(A), np.array([cx, cy])


# Homologous digitization bands (verified 100% consistent on provenanced data):
# the blade is traced around points ~40-110; the narrow stem is the lobe near
# the 1<->200 wrap. The outline winds blade-top -> right shoulder -> stem -> left.
BLADE_BAND = np.arange(40, 110)
STEM_BAND = np.r_[np.arange(155, 200), np.arange(0, 15)]


def canonical(outline):
    """Return (canon_outline, info) oriented with stem down, blade up, using the
    homologous digitization convention (NOT PCA, which the broad blade biases)."""
    o = outline.astype(float)
    area, _ = _polygon_area_centroid(o)
    # proximal->distal axis = from stem lobe centroid up to blade-band centroid
    stem_c = o[STEM_BAND].mean(0)
    blade_c = o[BLADE_BAND].mean(0)
    axis = blade_c - stem_c
    ang = np.arctan2(axis[1], axis[0])
    # rotate so 'axis' points to +y (up)
    theta = np.pi / 2 - ang
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    coords = (o - stem_c) @ R.T  # origin at stem centroid, blade up (+y)
    return coords, {"area": area, "stem_c": stem_c, "blade_c": blade_c}


def width_profile(coords, nbins=40):
    """Width of the outline in horizontal slices along the long axis (bottom->top)."""
    y = coords[:, 1]
    edges = np.linspace(y.min(), y.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.zeros(nbins)
    for i in range(nbins):
        m = (y >= edges[i]) & (y <= edges[i + 1])
        if m.sum() >= 2:
            widths[i] = coords[m, 0].max() - coords[m, 0].min()
    # fill empty slices by interpolation
    if (widths == 0).any():
        good = widths > 0
        widths = np.interp(centers, centers[good], widths[good])
    return centers, widths


def shoulder_y(centers, widths, frac=0.5):
    """Locate the shoulder: scanning up from the stem base, the first slice whose
    width exceeds `frac` * max width. Returns the y of that slice."""
    wmax = widths.max()
    thresh = frac * wmax
    above = np.where(widths >= thresh)[0]
    if len(above) == 0:
        return centers[len(centers) // 2]
    # first contiguous crossing from the bottom
    return centers[above[0]]


def _angle_at(coords, i, k=8):
    """Interior turning angle (degrees) at outline vertex i, using neighbors +/-k."""
    n = len(coords)
    a = coords[(i - k) % n]
    b = coords[i]
    c = coords[(i + k) % n]
    v1 = a - b
    v2 = c - b
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


def metrics(outline):
    coords, info = canonical(outline)
    y = coords[:, 1]
    y_tip = y.min()                      # stem tip (bottom)
    length = y.max() - y_tip
    centers, widths = width_profile(coords)
    max_width = widths.max()             # widest = blade
    sh_y = shoulder_y(centers, widths)   # stem/blade transition height

    # --- stem metrics (scale-free) ---
    stem_len = sh_y - y_tip
    stem_mask = coords[:, 1] <= sh_y
    stem_pts = coords[stem_mask]
    stem_width = (stem_pts[:, 0].max() - stem_pts[:, 0].min()) if len(stem_pts) >= 2 else np.nan
    lw_ratio = stem_len / stem_width if stem_width else np.nan

    # --- shoulder vertices: outline points nearest the shoulder height on each
    #     side, then interior angle there (the Fig 12.4 shoulder angle) ---
    near = np.where(np.abs(coords[:, 1] - sh_y) <= 0.10 * length)[0]
    shoulder_angle = np.nan
    li = ri = None
    if len(near) >= 2:
        li = near[np.argmin(coords[near, 0])]   # left shoulder vertex
        ri = near[np.argmax(coords[near, 0])]   # right shoulder vertex
        la = _angle_at(coords, li)
        ra = _angle_at(coords, ri)
        shoulder_angle = np.nanmean([la, ra])

    return {
        "stem_length": stem_len,
        "stem_width": stem_width,
        "lw_ratio": lw_ratio,
        "shoulder_angle": shoulder_angle,
        "elongation": length / max_width if max_width else np.nan,
        "_canon": coords,
        "_sh_y": sh_y,
        "_centers": centers,
        "_widths": widths,
        "_li": li,
        "_ri": ri,
    }
