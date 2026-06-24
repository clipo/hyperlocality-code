"""Robustness battery for the mata'a hyperlocality result (run_hyperlocality.py).

Stress-tests the two claims -- significant cultural F_ST, and isolation-by-distance
-- against their weak points:

  1. estimator dependence   : Nei G_ST vs Bell/Richerson variance-ratio F_ST
  2. sampling uncertainty   : bootstrap 95% CI on F_ST (resample artifacts)
  3. which attribute         : marginal F_ST for each underlying measurement
  4. one-assemblage leverage: leave-one-out on global F_ST and Mantel r
  5. the parcel-tie / east-west objection: partial Mantel | survey and | region,
     and Mantel with the within-survey tied pairs removed
  6. coordinate error       : jitter the map-read positions and re-test IBD

A claim that survives all six is robust; the point of the exercise is to find
where it does NOT.
"""
import numpy as np
import pandas as pd

import spatial
import popgen
import tables_io

RNG = np.random.default_rng(7)
PARCELS = ["Parcel 6", "Parcel 7", "Parcel 8", "Parcel 9", "Parcel 10", "Parcel 11"]


def marginalize(mat, keyfunc):
    """Collapse cross-classified columns into one attribute's marginal counts."""
    groups = {}
    for c in mat.columns:
        groups.setdefault(keyfunc(c), []).append(c)
    out = pd.DataFrame({k: mat[cols].sum(axis=1) for k, cols in groups.items()},
                       index=mat.index)
    return out


def estimator_and_ci(mat, label):
    counts = mat.values.astype(float)
    g = popgen.gst(counts)[0]
    b = popgen.fst_bell(counts)
    ci, _ = popgen.gst_bootstrap_ci(counts, n_boot=2000, rng=RNG)
    print(f"\n[1-2] {label}")
    print(f"   Nei G_ST      = {g:.4f}   Bell variance-ratio F_ST = {b:.4f}")
    print(f"   bootstrap 95% CI (G_ST) = [{ci[0]:.4f}, {ci[2]:.4f}]  median {ci[1]:.4f}")
    print(f"   -> {'CI excludes 0' if ci[0] > 0 else 'CI includes 0'}; "
          f"estimators {'agree' if abs(g - b) < 0.02 else 'differ'}")


def marginal_fst(mat, splits, label):
    print(f"\n[3] marginal F_ST by attribute -- {label}")
    cmats = []
    for attr, keyfunc in splits:
        m = marginalize(mat, keyfunc)
        obs, p, nm, _ = popgen.gst_permutation(m.values.astype(float), n_perm=4999)
        cmats.append(m.values.astype(float))
        print(f"   {attr:16s}: states={list(m.columns)}  "
              f"G_ST={obs:.4f}  null={nm:.4f}  p={p:.4g}")
    ml = popgen.gst_multilocus(cmats)
    print(f"   multilocus (both attributes as loci): G_ST = {ml:.4f}")


def leave_one_out(mat, label):
    names = list(mat.index)
    counts = mat.values.astype(float)
    geo_names = [n for n in names if spatial.has_coords(n)]
    print(f"\n[4] leave-one-assemblage-out -- {label}")
    g_full = popgen.gst(counts)[0]
    Dgeo = spatial.distance_matrix(geo_names)
    r_full, _ = popgen.mantel(popgen.neiman_d2(counts), Dgeo, n_perm=1, method="spearman")
    g_vals, r_vals, worst = [], [], None
    for drop in names:
        keep = [i for i, n in enumerate(names) if n != drop]
        sub = counts[keep]
        g = popgen.gst(sub)[0]
        gn = [names[i] for i in keep if spatial.has_coords(names[i])]
        D = spatial.distance_matrix(gn)
        r, p = popgen.mantel(popgen.neiman_d2(counts[[names.index(n) for n in gn]]),
                             D, n_perm=4999, method="spearman")
        g_vals.append(g)
        r_vals.append((r, p))
        if worst is None or p > worst[1]:
            worst = (drop, p, r)
    print(f"   global G_ST {g_full:.4f}; LOO range [{min(g_vals):.4f}, {max(g_vals):.4f}]")
    rs = [r for r, _ in r_vals]
    ps = [p for _, p in r_vals]
    print(f"   IBD Spearman r {r_full:+.3f}; LOO r range [{min(rs):+.3f}, {max(rs):+.3f}]; "
          f"max p={max(ps):.4g}")
    print(f"   weakest when dropping '{worst[0]}': p={worst[1]:.4g}, r={worst[2]:+.3f}")


def tie_and_region_controls(mat, label):
    names = [n for n in mat.index if spatial.has_coords(n)]
    counts = mat.loc[names].values.astype(float)
    Dgeo = spatial.distance_matrix(names)
    Dcomp = popgen.neiman_d2(counts)
    region = np.array([0 if n in PARCELS else 1 for n in names])  # east vs SW
    survey = np.array([spatial.PARCEL_SURVEY.get(n, n) for n in names])
    Dregion = (region[:, None] != region[None, :]).astype(float)
    Dsurvey = (survey[:, None] != survey[None, :]).astype(float)  # 0 = same survey point
    print(f"\n[5] parcel-tie / east-west controls -- {label}")
    r, p = popgen.mantel(Dcomp, Dgeo, n_perm=9999, method="pearson")
    print(f"   plain Mantel                         r={r:+.3f}  p={p:.4g}")
    rp, pp = popgen.partial_mantel(Dcomp, Dgeo, Dregion, n_perm=9999)
    print(f"   partial Mantel | east/west region    r={rp:+.3f}  p={pp:.4g}")
    rs, psv = popgen.partial_mantel(Dcomp, Dgeo, Dsurvey, n_perm=9999)
    print(f"   partial Mantel | same-survey ties    r={rs:+.3f}  p={psv:.4g}")
    # Mantel excluding the within-survey tied (d_geo == 0) pairs
    iu = np.triu_indices_from(Dgeo, 1)
    keep = Dgeo[iu] > 0
    x, y = Dgeo[iu][keep], Dcomp[iu][keep]
    r0 = np.corrcoef(x, y)[0, 1]
    n = len(names)
    cnt = 0
    for _ in range(9999):
        perm = RNG.permutation(n)
        Yp = popgen.neiman_d2(counts[perm])[iu][keep]
        if np.corrcoef(x, Yp)[0, 1] >= r0:
            cnt += 1
    print(f"   Mantel, tied pairs removed ({keep.sum()} of {keep.size} pairs)  "
          f"r={r0:+.3f}  p={(cnt + 1) / 10000:.4g}")


def coord_jitter(mat, label, sigma_km=1.0, n_rep=300):
    names = [n for n in mat.index if spatial.has_coords(n)]
    counts = mat.loc[names].values.astype(float)
    Dcomp = popgen.neiman_d2(counts)
    base = np.array([spatial._km(n) for n in names])
    iu = np.triu_indices(len(names), 1)
    y = Dcomp[iu]
    rs, sig = [], 0
    for _ in range(n_rep):
        pts = base + RNG.normal(0, sigma_km, base.shape)
        D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        x = D[iu]
        r = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
        rs.append(r)
        # quick 499-perm significance
        cnt = 0
        for _ in range(499):
            p = RNG.permutation(len(names))
            xp = D[np.ix_(p, p)][iu]
            if np.corrcoef(np.argsort(np.argsort(xp)),
                           np.argsort(np.argsort(y)))[0, 1] >= r:
                cnt += 1
        sig += (cnt + 1) / 500 < 0.05
    rs = np.array(rs)
    print(f"\n[6] coordinate jitter (sigma={sigma_km} km, {n_rep} reps) -- {label}")
    print(f"   Spearman r: mean {rs.mean():+.3f}  5th pct {np.percentile(rs, 5):+.3f}")
    print(f"   fraction with Mantel p<0.05: {sig / n_rep:.0%}")


SW = ["Ahu Tautira", "Orongo", "Orito", "Rano Kau", "Vinapu"]


def within_cluster_harden(mat, label, cluster=SW):
    """Harden the load-bearing claim: differentiation WITHIN the tight SW cluster
    (sites <=5.5 km apart). Estimators + bootstrap CI + leave-one-site-out."""
    sub = mat.loc[[c for c in cluster if c in mat.index]]
    counts = sub.values.astype(float)
    g = popgen.gst(counts)[0]
    b = popgen.fst_bell(counts)
    _, p, nm, _ = popgen.gst_permutation(counts, n_perm=9999)
    ci, _ = popgen.gst_bootstrap_ci(counts, n_boot=2000, rng=RNG)
    print(f"\n[7] within-SW-cluster F_ST ({len(sub)} sites <=5.5 km) -- {label}")
    print(f"   Nei G_ST={g:.4f}  Bell={b:.4f}  panmixia null={nm:.4f}  p={p:.4g}")
    print(f"   bootstrap 95% CI = [{ci[0]:.4f}, {ci[2]:.4f}]")
    print("   leave-one-site-out (drop -> G_ST, p):")
    for drop in sub.index:
        keep = [n for n in sub.index if n != drop]
        c = sub.loc[keep].values.astype(float)
        gg, pp, _, _ = popgen.gst_permutation(c, n_perm=4999)
        print(f"      -{drop:12s} G_ST={gg:.4f}  p={pp:.4g}")


if __name__ == "__main__":
    lw, _ = tables_io.stem_length_width()
    ss, _ = tables_io.stem_shoulder_shape()

    print("############# Table 3: stem length x width #############")
    estimator_and_ci(lw, "length x width")
    marginal_fst(lw, [("stem length", lambda c: c[0]),
                      ("stem width", lambda c: c[1:])], "length x width")
    leave_one_out(lw, "length x width")
    tie_and_region_controls(lw, "length x width")
    coord_jitter(lw, "length x width")
    within_cluster_harden(lw, "length x width")

    print("\n\n############# Table 2/4: stem shape x shoulder shape #############")
    estimator_and_ci(ss, "shape x shoulder")
    marginal_fst(ss, [("stem shape", lambda c: c[:-1]),
                      ("shoulder shape", lambda c: c[-1])], "shape x shoulder")
    leave_one_out(ss, "shape x shoulder")
    tie_and_region_controls(ss, "shape x shoulder")
    coord_jitter(ss, "shape x shoulder")
    within_cluster_harden(ss, "shape x shoulder")
