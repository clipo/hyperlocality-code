"""Cross-proxy concordance: do moai style demes coincide with the ahu visual
(intervisibility) communities?

Two independent proxies define community structure on the same landscape:
  - moai STYLE (this paper's categorical style attributes), and
  - ahu INTERVISIBILITY (the visual communities from the DEM viewshed network,
    src/viewshed_models.py -> data/viewshed/ahu_communities.json).

If they record the same social boundaries, moai style should differ between the
visual communities. We assign every ahu-placed moai to the nearest analyzed ahu
(hence its visual community) and estimate the Bayesian multilocus cultural F_ST
across those communities. We also measure partition agreement between the
intervisibility communities and the paper's geographic complete-linkage moai
clusters with the adjusted Rand index against a label-permutation null.

Run (PyMC venv):  PYTHONPATH=src .venv/bin/python src/run_concordance.py
"""
import json

import numpy as np

import bayes
import moai as moai_mod

COMM = "data/viewshed/ahu_communities.json"
MATCH_KM = 0.3   # a moai is "on" an analyzed ahu if within this distance


def _km(lon, lat, lon0, lat0):
    x = (lon - lon0) * np.cos(np.radians(lat0)) * 111.320
    y = (lat - lat0) * 110.574
    return np.hypot(x, y)


def adjusted_rand(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua = {v: i for i, v in enumerate(np.unique(a))}
    ub = {v: i for i, v in enumerate(np.unique(b))}
    C = np.zeros((len(ua), len(ub)))
    for x, y in zip(a, b):
        C[ua[x], ub[y]] += 1
    from math import comb
    sum_c = sum(comb(int(n), 2) for n in C.flatten())
    sa = sum(comb(int(n), 2) for n in C.sum(1))
    sb = sum(comb(int(n), 2) for n in C.sum(0))
    n = comb(int(C.sum()), 2)
    exp = sa * sb / n if n else 0
    return (sum_c - exp) / (0.5 * (sa + sb) - exp) if (0.5 * (sa + sb) - exp) else 0.0


def nmi(a, b):
    """Normalized mutual information (0 = independent, 1 = identical partitions).

    A second, information-theoretic agreement metric, complementary to the
    pair-counting adjusted Rand index. Normalized by the arithmetic mean of the
    two entropies (the sklearn default)."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    ua = {v: i for i, v in enumerate(np.unique(a))}
    ub = {v: i for i, v in enumerate(np.unique(b))}
    C = np.zeros((len(ua), len(ub)))
    for x, y in zip(a, b):
        C[ua[x], ub[y]] += 1
    pa = C.sum(1) / n
    pb = C.sum(0) / n

    def H(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    Ha, Hb = H(pa), H(pb)
    mi = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] > 0:
                pij = C[i, j] / n
                mi += pij * np.log(pij / (pa[i] * pb[j]))
    denom = 0.5 * (Ha + Hb)
    return mi / denom if denom > 0 else 0.0


def _perm_p(metric, a, b, rng, n=2000):
    obs = metric(a, b)
    null = np.array([metric(a, rng.permutation(b)) for _ in range(n)])
    return obs, float(null.mean()), (np.sum(null >= obs) + 1) / (n + 1)


def main():
    comm = json.load(open(COMM))["ahu"]
    a_lon = np.array([a["lon"] for a in comm])
    a_lat = np.array([a["lat"] for a in comm])
    a_comm = np.array([a["community"] for a in comm])
    lat0, lon0 = float(a_lat.mean()), float(a_lon.mean())

    df = moai_mod.load()
    loci = moai_mod.select_loci(df)
    # assign each moai to nearest analyzed ahu within MATCH_KM
    assigned, keep = [], []
    for lon, lat in zip(df["longitude"].values, df["latitude"].values):
        dists = np.hypot((a_lon - lon) * np.cos(np.radians(lat)) * 111.320,
                         (a_lat - lat) * 110.574)
        k = int(np.argmin(dists))
        if dists[k] <= MATCH_KM and a_comm[k] >= 0:
            assigned.append(a_comm[k]); keep.append(True)
        else:
            assigned.append(-1); keep.append(False)
    keep = np.array(keep)
    sub = df[keep].reset_index(drop=True)
    demes = np.array([assigned[i] for i in range(len(df)) if keep[i]]).astype(str)
    n_comm = len(set(demes))
    print(f"matched {keep.sum()} of {len(df)} ahu-placed moai to {n_comm} visual communities "
          f"(within {MATCH_KM} km of an analyzed ahu)")
    sizes = {c: int((demes == c).sum()) for c in sorted(set(demes))}
    print(f"moai per visual community: {sizes}")

    counts = list(moai_mod.all_locus_counts(sub, demes, loci).values())
    idata = bayes.fst_posterior_multilocus(counts)
    s = bayes.fst_summary(idata)
    bf = bayes.bayes_factor_structure(counts)
    print("\n[moai style F_ST across the intervisibility communities]")
    print(f"  posterior F_ST = {s['median']:.3f} [{s['hdi_lo']:.3f}, {s['hdi_hi']:.3f}]  "
          f"R-hat={s['rhat']:.3f}")
    print(f"  2 ln BF (structure vs panmixia) = {bf['two_ln_bf']:+.1f}  ({bf['evidence']})")

    # partition agreement: intervisibility communities vs geographic moai clusters,
    # by two complementary metrics (pair-counting ARI + information-theoretic NMI),
    # against the paper's primary (k=6) and the matched (k=8) geographic partitions.
    rng = np.random.default_rng(20260626)
    agree = {}
    print("\n[partition agreement: intervisibility communities vs geographic moai clusters]")
    for k in (6, 8):
        geo = moai_mod.spatial_demes(df, k=k, method="complete")[keep]
        ari, ari_nm, ari_p = _perm_p(adjusted_rand, demes, geo, rng)
        nmi_o, nmi_nm, nmi_p = _perm_p(nmi, demes, geo, rng)
        agree[f"k{k}"] = {"ari": ari, "ari_null": ari_nm, "ari_p": ari_p,
                          "nmi": nmi_o, "nmi_null": nmi_nm, "nmi_p": nmi_p}
        print(f"  vs geographic k={k}:  adjusted Rand = {ari:.3f} (null {ari_nm:.3f}, p={ari_p:.4g})"
              f"   |   NMI = {nmi_o:.3f} (null {nmi_nm:.3f}, p={nmi_p:.4g})")

    payload = {"matched": int(keep.sum()), "n_total": int(len(df)), "n_comm": n_comm,
               "moai_per_community": sizes,
               "fst_median": s["median"], "fst_hdi": [s["hdi_lo"], s["hdi_hi"]],
               "two_ln_bf": bf["two_ln_bf"], "evidence": bf["evidence"],
               "agreement": agree,
               # headline fields (k=8 partition) for the figure/manuscript
               "adjusted_rand": agree["k8"]["ari"], "ari_p": agree["k8"]["ari_p"],
               "nmi": agree["k8"]["nmi"], "nmi_p": agree["k8"]["nmi_p"]}
    json.dump(payload, open("output/concordance_results.json", "w"), indent=2)
    print("\nwrote output/concordance_results.json")


if __name__ == "__main__":
    main()
