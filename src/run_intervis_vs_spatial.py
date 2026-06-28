"""S-2 (decisive): does the intervisibility partition explain moai style better
than an arbitrary contiguity-matched spatial partition of the same grain?

The convergence headline claims that the ahu visual (intervisibility) communities
and moai style pick out the same boundaries. A skeptic replies that moai style is
spatially autocorrelated, so *any* partition of the coastal ahu into a handful of
contiguous regions would "predict" style — the agreement would then be an artifact
of spatial autocorrelation, not evidence that the intervisibility boundaries are
socially real.

Test. Statistic = multilocus Nei G_ST of moai style across a partition of the ahu.
  - Observed: G_ST across the 8 intervisibility communities.
  - Null: G_ST across many random *contiguous* partitions of the same analyzed ahu
    into the same number of groups (Voronoi-seed partitions: pick k ahu at random as
    seeds, assign every ahu to its nearest seed in local km — spatially contiguous,
    same grain). Each moai inherits its ahu's group; the moai-to-ahu assignment is
    held fixed, so only the boundary geometry changes.
If the intervisibility partition's G_ST exceeds the bulk of this contiguity-matched
null, the intervisibility boundaries capture style structure beyond generic spatial
autocorrelation, and the convergence is earned. If it sits inside the null, the
convergence collapses to spatial autocorrelation.

Pure numpy + (PyMC-free) moai/popgen. Run:
  PYTHONPATH=src python3 src/run_intervis_vs_spatial.py
"""
import json

import numpy as np

import moai as moai_mod

COMM = "data/viewshed/ahu_communities.json"
MATCH_KM = 0.3
K_NULL = 2000
SEED = 20260628


def km(lon, lat, lon0, lat0):
    x = (lon - lon0) * np.cos(np.radians(lat0)) * 111.320
    y = (lat - lat0) * 110.574
    return np.hypot(x, y)


def main():
    comm = json.load(open(COMM))["ahu"]
    a_lon = np.array([a["lon"] for a in comm])
    a_lat = np.array([a["lat"] for a in comm])
    a_comm = np.array([a["community"] for a in comm])
    n_ahu = len(comm)

    df = moai_mod.load()
    loci = moai_mod.select_loci(df)

    # assign each moai to its nearest analyzed ahu (within MATCH_KM); keep those
    ahu_idx, keep = [], []
    for lon, lat in zip(df["longitude"].values, df["latitude"].values):
        d = np.hypot((a_lon - lon) * np.cos(np.radians(lat)) * 111.320,
                     (a_lat - lat) * 110.574)
        j = int(np.argmin(d))
        if d[j] <= MATCH_KM and a_comm[j] >= 0:
            ahu_idx.append(j); keep.append(True)
        else:
            ahu_idx.append(-1); keep.append(False)
    keep = np.array(keep)
    sub = df[keep].reset_index(drop=True)
    ahu_of_moai = np.array([ahu_idx[i] for i in range(len(df)) if keep[i]])

    # observed: intervisibility communities
    obs_demes = a_comm[ahu_of_moai].astype(str)
    k = len(set(obs_demes.tolist()))
    obs_gst = moai_mod.multilocus_gst(moai_mod.all_locus_counts(sub, obs_demes, loci))
    print(f"matched {keep.sum()} of {len(df)} moai to {k} intervisibility communities")
    print(f"observed multilocus G_ST (intervisibility partition) = {obs_gst:.4f}")

    # universe of ahu eligible to be partitioned: those with a real community and
    # at least one matched moai (so null partitions are over the same points)
    used_ahu = np.array(sorted(set(ahu_of_moai.tolist())))
    ux, uy = a_lon[used_ahu], a_lat[used_ahu]
    lon0, lat0 = float(ux.mean()), float(uy.mean())

    rng = np.random.default_rng(SEED)
    null = np.empty(K_NULL)
    sizes_n = []
    for b in range(K_NULL):
        seeds = rng.choice(len(used_ahu), size=k, replace=False)
        sx, sy = ux[seeds], uy[seeds]
        # assign each used ahu to nearest seed in local km -> contiguous Voronoi groups
        grp_for_used = np.empty(len(used_ahu), dtype=int)
        for i in range(len(used_ahu)):
            dd = np.hypot((sx - ux[i]) * np.cos(np.radians(uy[i])) * 111.320,
                          (sy - uy[i]) * 110.574)
            grp_for_used[i] = int(np.argmin(dd))
        # map ahu index -> group label
        lab = {int(used_ahu[i]): int(grp_for_used[i]) for i in range(len(used_ahu))}
        null_demes = np.array([lab[int(a)] for a in ahu_of_moai]).astype(str)
        sizes_n.append(len(set(null_demes.tolist())))
        null[b] = moai_mod.multilocus_gst(moai_mod.all_locus_counts(sub, null_demes, loci))

    p = (np.sum(null >= obs_gst) + 1) / (K_NULL + 1)
    lo, med, hi = np.percentile(null, [2.5, 50, 97.5])
    eff_k = float(np.mean(sizes_n))
    print(f"\ncontiguity-matched null ({K_NULL} Voronoi-seed partitions into {k} groups,"
          f" mean realized groups {eff_k:.1f}):")
    print(f"  null G_ST  median {med:.4f}  95% [{lo:.4f}, {hi:.4f}]  max {null.max():.4f}")
    print(f"  P(null >= observed) = {p:.4g}")
    verdict = ("intervisibility BEATS arbitrary contiguous partitions"
               if p < 0.05 else
               "intervisibility does NOT beat contiguity-matched partitions")
    print(f"  -> {verdict}")

    out = {"observed_gst": float(obs_gst), "k_communities": int(k),
           "n_moai": int(keep.sum()), "n_ahu_partitioned": int(len(used_ahu)),
           "k_null": K_NULL, "null_median": float(med),
           "null_ci": [float(lo), float(hi)], "null_max": float(null.max()),
           "mean_realized_groups": eff_k, "p_value": float(p), "seed": SEED}
    json.dump(out, open("output/intervis_vs_spatial.json", "w"), indent=2)
    print("\nwrote output/intervis_vs_spatial.json")


if __name__ == "__main__":
    main()
