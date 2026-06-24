"""Moai hyperlocality from the public moai database (n=481 ahu-placed).

Same paradigmatic-class method as the other proxies (popgen.py): discrete
morphological STYLE states are the "alleles", and the DEME is a GEOGRAPHIC unit --
a coordinate-based spatial cluster of ahu-placed moai. We deliberately do NOT use
the ethnohistoric clan / lineage territories (CLAN_BOUNDARY) as the analytical
unit: those territories are an ethnohistoric reconstruction whose relation to the
pre-contact organization is uncertain, and the hyperlocality question -- is moai
style structured by location at a fine spatial grain? -- does not need a clan
model to answer it. All 481 ahu-placed moai carry coordinates, so the geographic
analysis uses the full sample (no clan filter).

We use CATEGORICAL style attributes rather than the continuous measurements as the
primary analysis for two reasons: (1) it is the common method shared with mata'a,
umu and pukao; (2) moai SIZE is strongly time-transgressive (statues grew through
time), so absolute dimensions confound style with chronology. Scale-free shape
states are far less time-confounded. A continuous scale-free-ratio complement is
in run_moai.py with that caveat stated.

Missing / indeterminate readings are dropped PER LOCUS (Nei multilocus G_ST pools
H_T/H_S over loci and tolerates unequal n). Candidate loci are filtered to those
that are well-covered AND polymorphic so monomorphic columns (e.g. EAR_LOBE) and
conflated ones (EYE_SOCKETS mixes carving with size) drop out automatically.
"""
import numpy as np
import pandas as pd

import popgen

# Primary spatial-deme resolution: complete-linkage clustering of ahu coordinates
# into this many demes. Reported across 6/8/10 in run_moai.py to show the result
# is stable across resolution; the signal strengthens at finer grain.
PRIMARY_K = 6

MISSING = {"Missing", "missing", "MISSING", "Indeterminate", "Ind.", "nan",
           "NaN", "None", "NONE", "N/A", "", "Not Visible", "Not Recorded",
           "Not Applicable", "Not Carved"}

# distinct facial/cranial/body/base features (avoid pseudo-replicating one region)
CANDIDATE_LOCI = ["HEAD_PLAN_SHAPE", "HEAD_PROFILE_SHAPE", "BODY_PLAN_SHAPE",
                  "BODY_PRO_SHAPE", "BASE_SHAPE", "NOSE_PROFIE",
                  "HANDS_PLACEMENT_ON_BASE"]

MIN_N = 60       # minimum observed readings to use a locus
MIN_MINOR = 5    # the 2nd-commonest state must reach this (true polymorphism)


def load(path="data/moai/MOAI_DATABASE_PUBLIC.xlsx"):
    """All ahu-placed moai with valid coordinates (n=481). No clan filter: the
    analysis is geographic, so we keep the full coordinated sample."""
    df = pd.read_excel(path)
    ahu = df[df["LOCATION_TYPE"] == "AHU"].copy()
    ahu["longitude"] = pd.to_numeric(ahu["longitude"], errors="coerce")
    ahu["latitude"] = pd.to_numeric(ahu["latitude"], errors="coerce")
    ahu = ahu[ahu[["longitude", "latitude"]].notna().all(axis=1)]
    return ahu.reset_index(drop=True)


def primary_demes(df, k=PRIMARY_K):
    """Geographic demes for the primary analysis: coordinate clusters at PRIMARY_K."""
    return spatial_demes(df, k=k, method="complete")


def _clean(series):
    s = series.astype(str).str.strip()
    return s.where(~s.isin(MISSING))


def select_loci(df, candidates=CANDIDATE_LOCI):
    """Keep candidate attributes that are well-covered and genuinely polymorphic."""
    keep = []
    for c in candidates:
        vc = _clean(df[c]).dropna().value_counts()
        if vc.sum() >= MIN_N and len(vc) >= 2 and vc.iloc[1] >= MIN_MINOR:
            keep.append(c)
    return keep


def locus_counts(df, demes, attr):
    vals = _clean(df[attr])
    obs = vals.notna()
    states = sorted(vals[obs].unique())
    deme_ids = sorted(pd.unique(demes))
    M = np.zeros((len(deme_ids), len(states)), float)
    si = {s: k for k, s in enumerate(states)}
    di = {d: i for i, d in enumerate(deme_ids)}
    for v, d, k in zip(vals, demes, obs):
        if k:
            M[di[d], si[v]] += 1
    return M, states, deme_ids


def all_locus_counts(df, demes, loci):
    return {a: locus_counts(df, demes, a)[0] for a in loci}


def multilocus_gst(count_dict):
    return popgen.gst_multilocus(list(count_dict.values()))


def multilocus_permutation(df, demes, loci, n_perm=9999, rng=None):
    """Panmixia null: permute the deme label across all moai, rebuild every locus."""
    if rng is None:
        rng = np.random.default_rng(20260623)
    demes = np.asarray(demes)
    obs = multilocus_gst(all_locus_counts(df, demes, loci))
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = multilocus_gst(all_locus_counts(df, rng.permutation(demes), loci))
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, p, float(null.mean()), null


# ---- geography: lon/lat -> local km (equirectangular about the island) ----
def _lonlat_to_km(lon, lat):
    lat0 = np.nanmean(lat)
    x = (lon - np.nanmean(lon)) * np.cos(np.radians(lat0)) * 111.320
    y = (lat - np.nanmean(lat)) * 110.574
    return np.column_stack([x, y])


def deme_centroids_km(df, demes):
    xy = _lonlat_to_km(df["longitude"].values, df["latitude"].values)
    deme_ids = sorted(pd.unique(demes))
    demes = np.asarray(demes)
    cent = np.array([xy[demes == d].mean(0) for d in deme_ids])
    return cent, deme_ids


def deme_distance_km(df, demes):
    cent, _ = deme_centroids_km(df, demes)
    diff = cent[:, None, :] - cent[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def spatial_demes(df, k=8, method="complete"):
    """Coordinate-based demes: cluster ahu lon/lat (local km) into k groups.

    This is the PRIMARY deme definition for moai: objective, geographic, and
    independent of the ethnohistoric clan model. 'single' linkage is deliberately
    avoided -- it chains the coastal ahu into one cluster; 'complete'/'ward' give
    compact spatial demes. Returns a string label per row, aligned with df."""
    from scipy.cluster.hierarchy import linkage, fcluster
    xy = _lonlat_to_km(df["longitude"].values, df["latitude"].values)
    Z = linkage(xy, method=method)
    return fcluster(Z, t=k, criterion="maxclust").astype(str)


def compositional_distance(count_dict):
    mats = list(count_dict.values())
    D = np.zeros((mats[0].shape[0],) * 2)
    for m in mats:
        D += popgen.neiman_d2(m)
    return D
