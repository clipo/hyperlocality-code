"""Pukao (topknot) hyperlocality: class-frequency cultural F_ST + IBD.

Same population-thinking move as the mata'a anchor case (see popgen.py), applied
to the categorical pukao style attributes in data/pukao/Pukao.csv (n=40, each with
UTM coordinates). Here the "alleles" are the discrete style states of each
attribute (one multi-allelic LOCUS per attribute), and the "demes" are spatial
clusters of pukao grouped by proximity (most pukao sit on or beside an ahu, so a
~1.5 km single-linkage cut groups neighbouring monuments into one locality).

Two signatures, mirroring run_hyperlocality.py:
  (3) multilocus cultural F_ST (Nei G_ST, pooled over attributes) vs a panmixia
      null that re-deals every pukao's attribute states across demes;
  (4) isolation-by-distance: Mantel of deme compositional distance vs km between
      deme centroids.

Missing / indeterminate readings ("Ind. - ...", "Indeterminate") are dropped
PER LOCUS, so a pukao can contribute to some attributes and not others -- Nei's
multilocus G_ST pools H_T/H_S numerators across loci and tolerates unequal n.

NB the two coordinate columns are mislabelled in the source (NORTHING holds the
~657k easting, EASTING the ~6 993k northing) but planar distance is invariant to
the swap, so we treat them as (x, y) metres directly.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

import popgen

# Style attributes used as loci. Profile_S2 (a second profile-shape reading) is
# excluded as a near-duplicate of Profile_Sh to avoid pseudo-replicating one trait.
LOCI = ["Profile_Sh", "Plan_Shape", "Profile_Sy", "Knob", "KnobProfil"]

# State strings that mean "not observed", dropped per locus.
_MISSING = {"Indeterminate", "Ind. - Not Visible", "Ind. - Missing",
            "Ind. - Eroded", "Ind. - Not Recorded"}


def load(path="data/pukao/Pukao.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"NORTHING": "x", "EASTING": "y"})
    return df


def deme_labels(df, threshold_m=1500.0):
    """Single-linkage spatial clusters of pukao -> deme id per row.

    threshold_m is the linkage cut: pukao within this distance chain into one
    locality. 1500 m gives ~6 balanced demes on these data; vary for robustness.
    """
    xy = df[["x", "y"]].values.astype(float)
    Z = linkage(xy, method="single")
    return fcluster(Z, t=threshold_m, criterion="distance")


def locus_counts(df, demes, attr):
    """Counts matrix (deme x state) for one attribute, dropping missing readings.

    Rows are demes in sorted id order; columns are the observed real states.
    Demes that contribute zero observed readings are kept as all-zero rows
    (popgen.gst weights by row n, so they drop out naturally).
    """
    vals = df[attr].astype(str)
    obs = ~vals.isin(_MISSING)
    states = sorted(vals[obs].unique())
    deme_ids = sorted(np.unique(demes))
    M = np.zeros((len(deme_ids), len(states)), float)
    s_index = {s: k for k, s in enumerate(states)}
    d_index = {d: i for i, d in enumerate(deme_ids)}
    for v, d, keep in zip(vals, demes, obs):
        if keep:
            M[d_index[d], s_index[v]] += 1
    return M, states, deme_ids


def all_locus_counts(df, demes, loci=LOCI):
    return {a: locus_counts(df, demes, a)[0] for a in loci}


def multilocus_gst(count_dict):
    """Nei multilocus G_ST pooled over the per-attribute counts matrices."""
    return popgen.gst_multilocus(list(count_dict.values()))


def multilocus_permutation(df, demes, loci=LOCI, n_perm=9999,
                           rng=None):
    """Panmixia null: permute the deme label across ALL pukao, rebuild every
    locus's counts, recompute multilocus G_ST. One shared permutation per
    replicate keeps the loci on the same shuffled individuals (a pukao's whole
    attribute set moves together), which is the right exchangeability unit.

    Returns (observed, p, null_mean, null_distribution).
    """
    if rng is None:
        rng = np.random.default_rng(20260623)
    obs = multilocus_gst(all_locus_counts(df, demes, loci))
    demes = np.asarray(demes)
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(demes)
        null[b] = multilocus_gst(all_locus_counts(df, perm, loci))
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, p, float(null.mean()), null


def deme_centroids(df, demes):
    """Mean (x, y) metres per deme, in sorted deme-id order."""
    deme_ids = sorted(np.unique(demes))
    cent = np.array([df.loc[demes == d, ["x", "y"]].mean().values
                     for d in deme_ids], float)
    return cent, deme_ids


def deme_distance_km(df, demes):
    cent, _ = deme_centroids(df, demes)
    diff = cent[:, None, :] - cent[None, :, :]
    return np.sqrt((diff ** 2).sum(-1)) / 1000.0


def compositional_distance(count_dict):
    """Summed Neiman d^2 across loci between demes (a multilocus analogue)."""
    mats = list(count_dict.values())
    n = mats[0].shape[0]
    D = np.zeros((n, n))
    for m in mats:
        D += popgen.neiman_d2(m)
    return D
