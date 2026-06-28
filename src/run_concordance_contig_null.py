"""S-1: contiguity-preserving null for the cross-proxy concordance.

The manuscript reports that the intervisibility partition and the style-based
geographic moai partition agree above chance (adjusted Rand 0.36, NMI 0.62,
p < 0.001) against a LABEL-PERMUTATION null. That null destroys all spatial
contiguity, so it is beaten by any spatially contiguous partition and does not
test the claim that the intervisibility boundaries specifically coincide with the
moai partition. Here we replace it with a CONTIGUITY-PRESERVING null: random
Voronoi-seed partitions of the same ahu into the same number of groups. We ask
whether the observed intervisibility-vs-geographic agreement exceeds what an
arbitrary contiguous partition of the same grain achieves against the same
geographic partition.

Pure numpy + (PyMC-free) moai. Run:
  PYTHONPATH=src python3 src/run_concordance_contig_null.py
"""
import json

import numpy as np

import moai as moai_mod
from run_concordance_ksweep import adjusted_rand, nmi

COMM = "data/viewshed/ahu_communities.json"
MATCH_KM = 0.3
K_NULL = 2000
SEED = 20260628


def main():
    comm = json.load(open(COMM))["ahu"]
    a_lon = np.array([a["lon"] for a in comm])
    a_lat = np.array([a["lat"] for a in comm])
    a_comm = np.array([a["community"] for a in comm])

    df = moai_mod.load()
    ahu_idx, keep = [], []
    for lon, lat in zip(df["longitude"].values, df["latitude"].values):
        dd = np.hypot((a_lon - lon) * np.cos(np.radians(lat)) * 111.320,
                      (a_lat - lat) * 110.574)
        j = int(np.argmin(dd))
        if dd[j] <= MATCH_KM and a_comm[j] >= 0:
            ahu_idx.append(j); keep.append(True)
        else:
            ahu_idx.append(-1); keep.append(False)
    keep = np.array(keep)
    ahu_of_moai = np.array([ahu_idx[i] for i in range(len(df)) if keep[i]])

    vis = a_comm[ahu_of_moai].astype(str)                       # intervisibility partition
    k = len(set(vis.tolist()))
    geo = moai_mod.spatial_demes(df, k=k, method="complete")[keep]   # style-based geographic partition (k=8)

    obs_ari = adjusted_rand(vis, geo)
    obs_nmi = nmi(vis, geo)
    print(f"matched {keep.sum()} moai; {k} communities")
    print(f"observed  ARI(intervis, geographic) = {obs_ari:.3f}   NMI = {obs_nmi:.3f}")

    used = np.array(sorted(set(ahu_of_moai.tolist())))
    ux, uy = a_lon[used], a_lat[used]
    rng = np.random.default_rng(SEED)
    ari_null = np.empty(K_NULL); nmi_null = np.empty(K_NULL)
    for b in range(K_NULL):
        seeds = rng.choice(len(used), size=k, replace=False)
        sx, sy = ux[seeds], uy[seeds]
        grp = np.empty(len(used), dtype=int)
        for i in range(len(used)):
            dd = np.hypot((sx - ux[i]) * np.cos(np.radians(uy[i])) * 111.320,
                          (sy - uy[i]) * 110.574)
            grp[i] = int(np.argmin(dd))
        lab = {int(used[i]): int(grp[i]) for i in range(len(used))}
        rand_part = np.array([lab[int(a)] for a in ahu_of_moai]).astype(str)
        ari_null[b] = adjusted_rand(rand_part, geo)
        nmi_null[b] = nmi(rand_part, geo)

    p_ari = (np.sum(ari_null >= obs_ari) + 1) / (K_NULL + 1)
    p_nmi = (np.sum(nmi_null >= obs_nmi) + 1) / (K_NULL + 1)
    print(f"\ncontiguity-preserving null ({K_NULL} Voronoi-seed partitions):")
    print(f"  ARI null median {np.median(ari_null):.3f} "
          f"[{np.percentile(ari_null,2.5):.3f}, {np.percentile(ari_null,97.5):.3f}]  "
          f"P(null>=obs)={p_ari:.4g}")
    print(f"  NMI null median {np.median(nmi_null):.3f} "
          f"[{np.percentile(nmi_null,2.5):.3f}, {np.percentile(nmi_null,97.5):.3f}]  "
          f"P(null>=obs)={p_nmi:.4g}")
    verdict = ("intervisibility agrees with the geographic partition MORE than contiguous chance"
               if (p_ari < 0.05 or p_nmi < 0.05) else
               "intervisibility agreement is NO better than a contiguous random partition")
    print(f"\n-> {verdict}")

    out = {"obs_ari": float(obs_ari), "obs_nmi": float(obs_nmi), "k": int(k),
           "k_null": K_NULL, "ari_null_median": float(np.median(ari_null)),
           "nmi_null_median": float(np.median(nmi_null)),
           "p_ari": float(p_ari), "p_nmi": float(p_nmi), "seed": SEED}
    json.dump(out, open("output/concordance_contig_null.json", "w"), indent=2)
    print("\nwrote output/concordance_contig_null.json")


if __name__ == "__main__":
    main()
