"""Pukao hyperlocality analysis -- mirror of run_hyperlocality.py for topknots.

Outputs a text report to stdout (redirect to output/pukao_results.txt) and an
IBD scatter to output/hyperlocality_pukao.png. All numbers here are what the
manuscript Pukao paragraph should cite.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import popgen
import pukao

RNG = np.random.default_rng(20260623)


def multilocus_bootstrap_ci(df, demes, loci=pukao.LOCI, n_boot=2000, rng=RNG):
    """Percentile CI for multilocus G_ST by resampling pukao within each deme."""
    demes = np.asarray(demes)
    deme_ids = sorted(np.unique(demes))
    idx_by_deme = {d: np.where(demes == d)[0] for d in deme_ids}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_by_deme.values()])
        sub = df.iloc[pick].reset_index(drop=True)
        sub_demes = demes[pick]
        vals[b] = pukao.multilocus_gst(
            pukao.all_locus_counts(sub, sub_demes, loci))
    return np.percentile(vals, [2.5, 50, 97.5]), vals


def report(threshold_m=1500.0):
    df = pukao.load()
    demes = pukao.deme_labels(df, threshold_m=threshold_m)
    deme_ids = sorted(np.unique(demes))
    sizes = [int((demes == d).sum()) for d in deme_ids]

    print("=" * 70)
    print(f"PUKAO HYPERLOCALITY  (single-linkage cut {threshold_m:.0f} m)")
    print("=" * 70)
    print(f"{len(df)} pukao -> {len(deme_ids)} demes, sizes {sizes}")
    print(f"loci: {', '.join(pukao.LOCI)}")

    # --- per-locus F_ST + panmixia null ---
    counts = pukao.all_locus_counts(df, demes)
    print("\n[per-locus G_ST vs panmixia null]")
    for attr, m in counts.items():
        n_obs = int(m.sum())
        obs, p, nm, _ = popgen.gst_permutation(m, n_perm=9999, rng=RNG)
        print(f"  {attr:<12} n={n_obs:<3} K={m.shape[1]}  "
              f"G_ST={obs:.4f}  null={nm:.4f}  p={p:.4g}  ({obs/nm:.1f}x)")

    # --- multilocus F_ST ---
    obs, p, nm, null = pukao.multilocus_permutation(df, demes, n_perm=9999, rng=RNG)
    print("\n[multilocus cultural F_ST]")
    print(f"  Nei multilocus G_ST = {obs:.4f}")
    print(f"  panmixia null mean  = {nm:.4f}  (95th pct {np.percentile(null,95):.4f})")
    print(f"  p(>=obs | panmixia) = {p:.4g}  -> "
          f"{'STRUCTURED' if p < 0.05 else 'no detectable structure'}")
    print(f"  excess over null    = {obs-nm:+.4f}  ({obs/nm:.1f}x)")

    # Bell variance-ratio cross-check, averaged across loci
    bell_vals = [popgen.fst_bell(m) for m in counts.values()]
    print(f"  Bell F_ST (mean over loci) = {np.mean(bell_vals):.4f}")

    (lo, med, hi), _ = multilocus_bootstrap_ci(df, demes)
    print(f"  bootstrap 95% CI    = [{lo:.4f}, {hi:.4f}]  median {med:.4f}  "
          f"-> {'excludes 0' if lo > 0 else 'includes 0'}")

    # --- IBD Mantel ---
    if len(deme_ids) >= 4:
        Dgeo = pukao.deme_distance_km(df, demes)
        Dcomp = pukao.compositional_distance(counts)
        span = Dgeo[np.triu_indices_from(Dgeo, 1)].max()
        print(f"\n[isolation by distance]  {len(deme_ids)} demes, "
              f"max centroid sep {span:.1f} km")
        for meth in ("pearson", "spearman"):
            r, pr = popgen.mantel(Dcomp, Dgeo, n_perm=9999, method=meth, rng=RNG)
            print(f"  Mantel {meth:<8} r={r:+.3f}  p={pr:.4g}")
        _ibd_plot(Dgeo, Dcomp, threshold_m)
    else:
        print("\n[too few demes for IBD]")
    return obs, p


def _ibd_plot(Dgeo, Dcomp, threshold_m, path="output/hyperlocality_pukao.png"):
    iu = np.triu_indices_from(Dgeo, 1)
    x, y = Dgeo[iu], Dcomp[iu]
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(x, y, s=34, color="#7d3c98", alpha=0.85, zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, a + b * xs, color="#c0392b", lw=1.6,
            label=f"slope {b:.4f}/km   r={r:.2f}")
    ax.set_xlabel("geographic distance between demes (km)")
    ax.set_ylabel(r"compositional distance ($\sum$ Neiman $d^2$)")
    ax.set_title(f"Pukao isolation by distance (cut {threshold_m:.0f} m)")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    # primary result, then robustness across cut thresholds
    report(threshold_m=1500.0)
    for thr in (1000.0, 2000.0):
        print()
        report(threshold_m=thr)
