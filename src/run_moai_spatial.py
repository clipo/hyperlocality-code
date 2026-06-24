"""Moai LOCAL spatial structure without data-driven clusters.

Addresses the reviewer concern that complete-linkage clustering on coordinates can
manufacture between-group variance. Instead of clustering, we work at the level of
individual statues:

  - pairwise style distance = fraction of jointly observed style loci on which two
    statues differ (Gower distance over the 7 categorical loci; pairs must share
    >= MIN_SHARED observed loci);
  - pairwise geographic distance in km;
  - SAME-ahu pairs are excluded (LOCATION_NAME), so the signal is strictly
    between-place, not statues carved together on one platform;
  - a distance-binned correlogram shows whether nearby statues are more alike, and
    at what scale (the "local spatial variability");
  - significance by a genotype-permutation null: shuffle the style genotypes across
    statues (geography fixed), 9,999 times. Under the null, style is independent of
    location, so any distance->similarity relationship vanishes. This does not
    manufacture clusters and does not assume a deme partition.

It cannot separate community boundaries from other spatial autocorrelation in style
(carving sequence, local quarrying); that identification limit is the equifinality
point made in the Discussion. What it can show is whether moai style is locally
spatially structured at all, and at what grain.

Run: PYTHONPATH=src python src/run_moai_spatial.py > output/moai_spatial.txt 2>&1
"""
import numpy as np

import moai

RNG = np.random.default_rng(20260623)
MIN_LOCI = 3        # keep statues with at least this many observed style loci
MIN_SHARED = 2      # a pair must share at least this many observed loci
BINS = [0, 2, 4, 6, 10, 100]   # geographic distance classes (km)
N_PERM = 9999


def _code_matrix(df, loci):
    """n x L integer codes; -1 = missing."""
    cols = []
    for a in loci:
        s = moai._clean(df[a])
        cats = {v: i for i, v in enumerate(sorted(s.dropna().unique()))}
        cols.append(s.map(cats).fillna(-1).astype(int).values)
    return np.column_stack(cols)


def _pair_style_dist(codes):
    """Pairwise Gower distance over shared observed loci; NaN if < MIN_SHARED shared.
    Returns (D_style, shared_count) as n x n arrays."""
    n, L = codes.shape
    diff = np.zeros((n, n), float)
    shared = np.zeros((n, n), float)
    for k in range(L):
        c = codes[:, k]
        obs = (c >= 0)
        both = np.outer(obs, obs)
        ne = (c[:, None] != c[None, :]) & both
        shared += both
        diff += ne
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(shared >= MIN_SHARED, diff / shared, np.nan)
    return D, shared


def main():
    df = moai.load()
    loci = moai.select_loci(df)
    df = df.copy()
    df["loc"] = df["LOCATION_NAME"].astype(str)
    codes = _code_matrix(df, loci)
    keep = (codes >= 0).sum(1) >= MIN_LOCI
    df = df[keep].reset_index(drop=True)
    codes = codes[keep]
    n = len(df)
    print("=" * 68)
    print("MOAI LOCAL SPATIAL STRUCTURE (individual statues, no clustering)")
    print("=" * 68)
    print(f"statues with >= {MIN_LOCI} observed style loci: {n}")
    print(f"distinct ahu (LOCATION_NAME): {df['loc'].nunique()}")

    xy = moai._lonlat_to_km(df["longitude"].values, df["latitude"].values)
    Dgeo = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    loc = np.asarray(df["loc"], dtype=object)
    same_ahu = (loc[:, None] == loc[None, :])

    iu = np.triu_indices(n, 1)
    geo = Dgeo[iu]
    same = same_ahu[iu]

    def corr_and_bins(style_full):
        s = style_full[iu]
        valid = np.isfinite(s) & (~same)        # between-ahu pairs with shared loci
        gv, sv = geo[valid], s[valid]
        r = np.corrcoef(gv, sv)[0, 1] if len(sv) > 2 else np.nan
        means = []
        for b in range(len(BINS) - 1):
            m = (gv >= BINS[b]) & (gv < BINS[b + 1])
            means.append(sv[m].mean() if m.sum() else np.nan)
        return r, np.array(means), valid.sum()

    Dstyle, _ = _pair_style_dist(codes)
    r_obs, bin_obs, npair = corr_and_bins(Dstyle)
    print(f"between-ahu pairs with >= {MIN_SHARED} shared loci: {npair}")
    print(f"\nobserved Mantel-type r (style distance vs geographic distance): {r_obs:+.3f}")
    print("  (positive r = nearer statues more alike = local spatial structure)")

    # genotype-permutation null: shuffle rows of the code matrix (geography fixed)
    null_r = np.empty(N_PERM)
    null_bins = np.empty((N_PERM, len(BINS) - 1))
    for p in range(N_PERM):
        perm = RNG.permutation(n)
        Dp, _ = _pair_style_dist(codes[perm])
        null_r[p], null_bins[p], _ = corr_and_bins(Dp)
    p_r = (np.sum(null_r >= r_obs) + 1) / (N_PERM + 1)
    print(f"  null mean r = {np.nanmean(null_r):+.3f};  p(>= obs) = {p_r:.4g}")

    print("\n[distance-binned mean style distance: observed vs null]")
    print(f"  {'distance (km)':<16}{'pairs':>7}{'obs':>9}{'null':>9}{'p(obs<null)':>13}")
    s_all = Dstyle[iu]
    valid = np.isfinite(s_all) & ~same          # between-ahu pairs with shared loci
    gv = geo[valid]                             # their geographic distances (km)
    for b in range(len(BINS) - 1):
        lbl = f"{BINS[b]}-{BINS[b+1] if BINS[b+1]<100 else ''}".rstrip("-")
        inb = (gv >= BINS[b]) & (gv < BINS[b + 1])   # pairs falling in this bin
        nb = int(inb.sum())
        nullcol = null_bins[:, b]
        # p that observed bin mean is LOWER than null (more similar than chance)
        pb = (np.sum(nullcol <= bin_obs[b]) + 1) / (N_PERM + 1)
        print(f"  {lbl+' km':<16}{nb:>7}{bin_obs[b]:>9.3f}"
              f"{np.nanmean(nullcol):>9.3f}{pb:>13.4g}")
    print("\nInterpretation: a significantly LOW mean style distance in the short bins")
    print("(p small) indicates fine-scale spatial structure in moai style; it does not")
    print("by itself attribute that structure to community boundaries vs other local")
    print("autocorrelation (carving sequence, quarrying).")


if __name__ == "__main__":
    main()
