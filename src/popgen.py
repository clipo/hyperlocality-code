"""Cultural population-genetics statistics for the mata'a count matrices.

Treats each assemblage as a "deme" and each paradigmatic class as an "allele" at
one multi-allelic locus (the Riede/Shennan population-thinking move: classes are
variants, assemblages are populations). Two quantities, following the framework
used in ../mls-emergence (signatures 3 & 4):

  * Cultural F_ST  -- Nei's G_ST: the share of total class diversity that lies
    BETWEEN assemblages. Significance by a panmixia (random-assignment) null:
    pool every artifact and re-deal them into assemblages of the observed sizes.
    If everyone on this small island drew from one island-wide interaction pool,
    observed F_ST should sit inside that null. (Bell, Richerson & McElreath 2009;
    Richerson et al. 2016 give the cultural-F_ST threshold and magnitudes.)

  * Between-assemblage distance -- Neiman's (1995, Eq. 10) squared Euclidean
    distance d^2_ij = sum_k (p_ik - p_jk)^2, and pairwise G_ST, for the
    isolation-by-distance / Mantel test in run_hyperlocality.py.

Small-sample bias in G_ST is real but cancels in the significance call: the
panmixia null is computed with the identical estimator, so observed-vs-null is a
clean test even though the point estimate is biased upward.
"""
import numpy as np

RNG = np.random.default_rng(20260623)


def _freqs(counts):
    """Row-normalize a counts matrix (assemblages x classes) to frequencies."""
    n = counts.sum(1, keepdims=True)
    return np.divide(counts, n, out=np.zeros_like(counts, float), where=n > 0)


def gst(counts):
    """Nei's G_ST for one multi-allelic locus (single counts matrix).

    H_S = sample-size-weighted mean within-assemblage heterozygosity (1 - sum p^2)
    H_T = total heterozygosity from the pooled allele frequencies
    G_ST = (H_T - H_S) / H_T
    """
    counts = np.asarray(counts, float)
    n_i = counts.sum(1)
    P = _freqs(counts)
    h_i = 1.0 - (P ** 2).sum(1)                 # within-assemblage diversity
    w = n_i / n_i.sum()
    H_S = float((w * h_i).sum())
    pbar = counts.sum(0) / counts.sum()         # pooled (size-weighted) freqs
    H_T = float(1.0 - (pbar ** 2).sum())
    return (H_T - H_S) / H_T if H_T > 0 else 0.0, H_S, H_T


def gst_permutation(counts, n_perm=9999, rng=RNG):
    """Panmixia null for G_ST: re-deal pooled artifacts into observed-size groups.

    Returns (observed_gst, p_value, null_mean, null_distribution).
    p = P(null G_ST >= observed) with the conventional +1 / +1 correction.
    """
    counts = np.asarray(counts, float)
    obs = gst(counts)[0]
    n_i = counts.sum(1).astype(int)
    # build the pooled "urn": one entry per artifact, labelled by its class
    classes = np.repeat(np.arange(counts.shape[1]),
                        counts.sum(0).astype(int))
    K = counts.shape[1]
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(classes)
        # cut into assemblages of the observed sizes and recount
        cuts = np.cumsum(n_i)[:-1]
        groups = np.split(perm, cuts)
        m = np.array([np.bincount(g, minlength=K) for g in groups], float)
        null[b] = gst(m)[0]
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, p, float(null.mean()), null


def gst_multilocus(count_list):
    """Pool several loci (e.g. two attribute marginals): sum H_T and H_S numerators.

    G_ST = (sum_l H_T_l - sum_l H_S_l) / sum_l H_T_l  (Nei's multilocus G_ST).
    """
    HT = HS = 0.0
    for c in count_list:
        _, hs, ht = gst(c)
        HS += hs
        HT += ht
    return (HT - HS) / HT if HT > 0 else 0.0


def fst_bell(counts):
    """Cultural F_ST via the Bell, Richerson & McElreath (2009) / Cavalli-Sforza
    variance-ratio estimator -- an independent cross-check on Nei's G_ST.

    Per allele k:  F_ST,k = var_k / [pbar_k (1 - pbar_k)], with pbar size-weighted
    and var_k = sum_i (p_ik - pbar_k)^2 / (s - 1) across the s assemblages.
    Overall (ratio of averages):  sum_k var_k / sum_k pbar_k (1 - pbar_k).
    """
    counts = np.asarray(counts, float)
    s = counts.shape[0]
    P = _freqs(counts)
    pbar = counts.sum(0) / counts.sum()
    var_k = ((P - pbar[None, :]) ** 2).sum(0) / (s - 1)
    den = (pbar * (1 - pbar)).sum()
    return float(var_k.sum() / den) if den > 0 else 0.0


def _artifact_labels(counts):
    """Expand a counts matrix to per-assemblage arrays of class indices."""
    counts = np.asarray(counts, int)
    K = counts.shape[1]
    return [np.repeat(np.arange(K), row) for row in counts]


def gst_bootstrap_ci(counts, n_boot=2000, rng=RNG, estimator=gst):
    """Percentile CI for an F_ST estimator by resampling artifacts WITHIN each
    assemblage (preserves the observed assemblage sizes)."""
    labels = _artifact_labels(counts)
    K = np.asarray(counts).shape[1]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        m = np.array([np.bincount(rng.choice(lab, size=len(lab), replace=True),
                                  minlength=K) if len(lab) else np.zeros(K)
                      for lab in labels], float)
        v = estimator(m)
        vals[b] = v[0] if isinstance(v, tuple) else v
    return np.percentile(vals, [2.5, 50, 97.5]), vals


def partial_mantel(Dy, Dx, Dz, n_perm=9999, rng=RNG):
    """Partial Mantel: correlation of Dy and Dx controlling for Dz, on the upper
    triangle, with residuals from OLS regression on Dz. Permutes Dy's objects.

    Returns (partial_r, p). Use Dz = a binary design matrix (e.g. same-survey or
    same-region) to ask whether distance predicts composition BEYOND that split.
    """
    iu = np.triu_indices_from(Dy, k=1)
    y, x, z = Dy[iu], Dx[iu], Dz[iu]

    def resid(v):
        A = np.column_stack([np.ones_like(z), z])
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta

    yr, xr = resid(y), resid(x)
    r_obs = float(np.corrcoef(yr, xr)[0, 1])
    n = Dy.shape[0]
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        Yp = Dy[np.ix_(p, p)][iu]
        if np.corrcoef(resid(Yp), xr)[0, 1] >= r_obs:
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)


def neiman_d2(counts):
    """Pairwise Neiman (1995) squared Euclidean distance between assemblages."""
    P = _freqs(np.asarray(counts, float))
    diff = P[:, None, :] - P[None, :, :]
    return (diff ** 2).sum(-1)


def pairwise_gst(counts):
    """Pairwise (2-assemblage) G_ST matrix -- a genetic-distance analogue."""
    counts = np.asarray(counts, float)
    S = counts.shape[0]
    D = np.zeros((S, S))
    for i in range(S):
        for j in range(i + 1, S):
            D[i, j] = D[j, i] = gst(counts[[i, j]])[0]
    return D


def mantel(D1, D2, n_perm=9999, rng=RNG, method="pearson"):
    """Mantel test between two distance matrices (upper triangle).

    Returns (r, p_value). p is one-sided P(perm r >= observed) for positive
    association (isolation-by-distance predicts a positive correlation).
    """
    D1 = np.asarray(D1, float)
    D2 = np.asarray(D2, float)
    iu = np.triu_indices_from(D1, k=1)
    a, b = D1[iu], D2[iu]
    if method == "spearman":
        a = _rankdata(a)
        b = _rankdata(b)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    r_obs = float((a * b).mean())
    n = D1.shape[0]
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        Bp = D2[np.ix_(p, p)][iu]
        if method == "spearman":
            Bp = _rankdata(Bp)
        Bp = (Bp - Bp.mean()) / Bp.std()
        if (a * Bp).mean() >= r_obs:
            count += 1
    return r_obs, (count + 1) / (n_perm + 1)


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(len(x))
    # average ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    start = csum - cnt
    avg = (start + csum - 1) / 2.0
    return avg[inv]
