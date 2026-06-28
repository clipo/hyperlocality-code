"""S-2, bias-corrected. Raw Nei G_ST is biased upward for partitions with more or
smaller or more unequal demes, so comparing G_ST across partitions of different
balance is confounded. For each partition we therefore also compute the
bias-corrected excess

    excess(P) = G_ST(P) - mean_perm G_ST(P),

where the permutation holds P's deme sizes fixed and re-deals the artifacts (the
partition's own panmixia expectation). excess removes the size/number bias, so the
intervisibility partition and the contiguity-matched Voronoi nulls are compared on
an even footing.

Reports both the raw-G_ST test (as in run_intervis_vs_spatial.py) and the
bias-corrected-excess test. Fast pure-numpy multilocus G_ST; sanity-checked against
moai.multilocus_gst. Run:  PYTHONPATH=src python3 src/run_intervis_bc.py
"""
import json

import numpy as np

import moai as moai_mod

COMM = "data/viewshed/ahu_communities.json"
MATCH_KM = 0.3
K_NULL = 1000
N_PERM = 200
SEED = 20260628


def fast_mlgst(d, codes, Ks, k):
    HT = HS = 0.0
    for code, K in zip(codes, Ks):
        m = code >= 0
        dd, cc = d[m], code[m]
        M = np.zeros((k, K))
        np.add.at(M, (dd, cc), 1.0)
        n_i = M.sum(1)
        tot = n_i.sum()
        if tot == 0:
            continue
        P = M / np.where(n_i[:, None] > 0, n_i[:, None], 1.0)
        h_i = 1.0 - (P ** 2).sum(1)
        HS += float((n_i / tot * h_i).sum())
        pbar = M.sum(0) / tot
        HT += float(1.0 - (pbar ** 2).sum())
    return (HT - HS) / HT if HT > 0 else 0.0


def excess(d, codes, Ks, k, rng, n_perm):
    g = fast_mlgst(d, codes, Ks, k)
    pm = np.mean([fast_mlgst(rng.permutation(d), codes, Ks, k) for _ in range(n_perm)])
    return g - pm, g


def main():
    comm = json.load(open(COMM))["ahu"]
    a_lon = np.array([a["lon"] for a in comm])
    a_lat = np.array([a["lat"] for a in comm])
    a_comm = np.array([a["community"] for a in comm])

    df = moai_mod.load()
    loci = moai_mod.select_loci(df)

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
    sub = df[keep].reset_index(drop=True)
    ahu_of_moai = np.array([ahu_idx[i] for i in range(len(df)) if keep[i]])

    # precompute per-locus integer codes over kept moai (-1 = missing)
    codes, Ks = [], []
    for a in loci:
        vals = moai_mod._clean(sub[a])
        obs = vals.notna()
        states = sorted(vals[obs].unique())
        si = {s: i for i, s in enumerate(states)}
        code = np.array([si[v] if k else -1 for v, k in zip(vals, obs)])
        codes.append(code); Ks.append(len(states))

    # observed intervisibility partition (relabel communities to 0..k-1)
    obs_lab = a_comm[ahu_of_moai]
    uniq = sorted(set(obs_lab.tolist()))
    relab = {c: i for i, c in enumerate(uniq)}
    d_obs = np.array([relab[c] for c in obs_lab])
    k = len(uniq)

    # sanity check fast vs reference
    ref = moai_mod.multilocus_gst(moai_mod.all_locus_counts(sub, obs_lab.astype(str), loci))
    fast = fast_mlgst(d_obs, codes, Ks, k)
    assert abs(ref - fast) < 1e-9, f"fast G_ST mismatch: {ref} vs {fast}"
    print(f"matched {keep.sum()} of {len(df)} moai to {k} intervisibility communities")
    print(f"observed multilocus G_ST = {fast:.4f}  (matches reference {ref:.4f})")

    rng = np.random.default_rng(SEED)
    exc_obs, _ = excess(d_obs, codes, Ks, k, rng, N_PERM)

    used = np.array(sorted(set(ahu_of_moai.tolist())))
    ux, uy = a_lon[used], a_lat[used]

    raw = np.empty(K_NULL); exc = np.empty(K_NULL)
    for b in range(K_NULL):
        seeds = rng.choice(len(used), size=k, replace=False)
        sx, sy = ux[seeds], uy[seeds]
        grp = np.empty(len(used), dtype=int)
        for i in range(len(used)):
            dd = np.hypot((sx - ux[i]) * np.cos(np.radians(uy[i])) * 111.320,
                          (sy - uy[i]) * 110.574)
            grp[i] = int(np.argmin(dd))
        lab = {int(used[i]): int(grp[i]) for i in range(len(used))}
        d_null = np.array([lab[int(a)] for a in ahu_of_moai])
        present = sorted(set(d_null.tolist()))
        re2 = {c: i for i, c in enumerate(present)}
        d_null = np.array([re2[c] for c in d_null]); kk = len(present)
        e, g = excess(d_null, codes, Ks, kk, rng, N_PERM)
        raw[b] = g; exc[b] = e

    p_raw = (np.sum(raw >= fast) + 1) / (K_NULL + 1)
    p_exc = (np.sum(exc >= exc_obs) + 1) / (K_NULL + 1)
    print(f"\nraw G_ST:   observed {fast:.4f}  |  null median {np.median(raw):.4f} "
          f"[{np.percentile(raw,2.5):.4f}, {np.percentile(raw,97.5):.4f}]  P(null>=obs)={p_raw:.4g}")
    print(f"excess:     observed {exc_obs:.4f}  |  null median {np.median(exc):.4f} "
          f"[{np.percentile(exc,2.5):.4f}, {np.percentile(exc,97.5):.4f}]  P(null>=obs)={p_exc:.4g}")
    verdict = ("intervisibility BEATS contiguity-matched partitions (earns convergence)"
               if p_exc < 0.05 else
               "intervisibility does NOT beat contiguity-matched partitions "
               "(convergence largely spatial autocorrelation)")
    print(f"\n-> {verdict}")

    out = {"n_moai": int(keep.sum()), "k": int(k), "k_null": K_NULL, "n_perm": N_PERM,
           "raw_obs": float(fast), "raw_null_median": float(np.median(raw)), "p_raw": float(p_raw),
           "excess_obs": float(exc_obs), "excess_null_median": float(np.median(exc)),
           "p_excess": float(p_exc), "seed": SEED}
    json.dump(out, open("output/intervis_bc.json", "w"), indent=2)
    print("\nwrote output/intervis_bc.json")


if __name__ == "__main__":
    main()
