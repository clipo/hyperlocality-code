"""Sensitivity of the cross-proxy concordance to the number of geographic moai
clusters (k).

The headline concordance (src/run_concordance.py) compares the ahu intervisibility
communities to the geographic complete-linkage moai clusters at k=6 (primary) and
k=8 (matched to the visual-community count). A reader may ask whether the agreement
is an artifact of choosing k=8. This script sweeps k and reports the adjusted Rand
index and normalized mutual information against a label-permutation null at each k,
to show the agreement is robust across resolutions rather than tuned to one cut.

Pure numpy + the (PyMC-free) moai/popgen modules; runs under system python3.

Run:  PYTHONPATH=src python3 src/run_concordance_ksweep.py
"""
import json
from math import comb

import numpy as np

import moai as moai_mod

COMM = "data/viewshed/ahu_communities.json"
MATCH_KM = 0.3


def adjusted_rand(a, b):
    ua = {v: i for i, v in enumerate(np.unique(a))}
    ub = {v: i for i, v in enumerate(np.unique(b))}
    C = np.zeros((len(ua), len(ub)), dtype=int)
    for x, y in zip(a, b):
        C[ua[x], ub[y]] += 1
    sum_c = sum(comb(int(n), 2) for n in C.flatten())
    sa = sum(comb(int(n), 2) for n in C.sum(1))
    sb = sum(comb(int(n), 2) for n in C.sum(0))
    n = len(a)
    exp = sa * sb / comb(n, 2)
    denom = 0.5 * (sa + sb) - exp
    return (sum_c - exp) / denom if denom else 0.0


def nmi(a, b):
    ua = {v: i for i, v in enumerate(np.unique(a))}
    ub = {v: i for i, v in enumerate(np.unique(b))}
    C = np.zeros((len(ua), len(ub)), dtype=float)
    for x, y in zip(a, b):
        C[ua[x], ub[y]] += 1
    n = len(a)
    pa = C.sum(1) / n
    pb = C.sum(0) / n
    P = C / n
    mi = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if P[i, j] > 0:
                mi += P[i, j] * np.log(P[i, j] / (pa[i] * pb[j]))
    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    denom = 0.5 * (ha + hb)
    return mi / denom if denom else 0.0


def _perm_p(metric, a, b, rng, n=2000):
    obs = metric(a, b)
    null = np.array([metric(a, rng.permutation(b)) for _ in range(n)])
    return obs, float(null.mean()), (np.sum(null >= obs) + 1) / (n + 1)


def main():
    comm = json.load(open(COMM))["ahu"]
    a_lon = np.array([a["lon"] for a in comm])
    a_lat = np.array([a["lat"] for a in comm])
    a_comm = np.array([a["community"] for a in comm])

    df = moai_mod.load()
    assigned, keep = [], []
    for lon, lat in zip(df["longitude"].values, df["latitude"].values):
        dists = np.hypot((a_lon - lon) * np.cos(np.radians(lat)) * 111.320,
                         (a_lat - lat) * 110.574)
        j = int(np.argmin(dists))
        if dists[j] <= MATCH_KM and a_comm[j] >= 0:
            assigned.append(a_comm[j]); keep.append(True)
        else:
            assigned.append(-1); keep.append(False)
    keep = np.array(keep)
    demes = np.array([assigned[i] for i in range(len(df)) if keep[i]]).astype(str)
    n_vis = len(set(demes))
    print(f"matched {keep.sum()} of {len(df)} moai to {n_vis} visual communities")

    rng = np.random.default_rng(20260626)
    rows = {}
    print("\nk   adjusted Rand (null)      p        NMI (null)         p")
    for k in range(4, 13):
        geo = moai_mod.spatial_demes(df, k=k, method="complete")[keep]
        ari, ari_nm, ari_p = _perm_p(adjusted_rand, demes, geo, rng)
        nmi_o, nmi_nm, nmi_p = _perm_p(nmi, demes, geo, rng)
        rows[k] = {"ari": ari, "ari_null": ari_nm, "ari_p": ari_p,
                   "nmi": nmi_o, "nmi_null": nmi_nm, "nmi_p": nmi_p}
        print(f"{k:<3} {ari:.3f} ({ari_nm:.3f})   {ari_p:.4g}   "
              f"{nmi_o:.3f} ({nmi_nm:.3f})   {nmi_p:.4g}")

    out = {"n_visual_communities": n_vis, "matched": int(keep.sum()),
           "n_total": int(len(df)), "by_k": rows}
    json.dump(out, open("output/concordance_ksweep.json", "w"), indent=2)
    print("\nwrote output/concordance_ksweep.json")


if __name__ == "__main__":
    main()
