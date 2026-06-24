"""Moai hyperlocality analysis (categorical style, GEOGRAPHIC spatial-cluster demes).

The deme is a coordinate-based spatial cluster of ahu-placed moai (moai.PRIMARY_K),
not the ethnohistoric clan territory. All 481 ahu-placed moai carry coordinates, so
the analysis uses the full sample. Results are reported across k = 6, 8, 10 to show
they are stable across resolution; a single clan cross-check is reported only to
confirm the geographic result does not depend on dropping the clan model.

Mirrors run_pukao.py. Redirect stdout to output/moai_results.txt; writes
output/hyperlocality_moai.png. Continuous scale-free-ratio complement at the end
(reported with the time-transgression caveat; underpowered by sparse measurement).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import popgen
import moai

RNG = np.random.default_rng(20260623)


def multilocus_bootstrap_ci(df, demes, loci, n_boot=1000, rng=RNG):
    demes = np.asarray(demes)
    idx_by = {d: np.where(demes == d)[0] for d in sorted(pd.unique(demes))}
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([rng.choice(ix, len(ix), replace=True)
                               for ix in idx_by.values()])
        vals[b] = moai.multilocus_gst(
            moai.all_locus_counts(df.iloc[pick].reset_index(drop=True),
                                  demes[pick], loci))
    return np.percentile(vals, [2.5, 50, 97.5])


def report():
    df = moai.load()
    demes = moai.primary_demes(df)
    loci = moai.select_loci(df)
    deme_ids = sorted(pd.unique(demes))
    sizes = {d: int((demes == d).sum()) for d in deme_ids}

    print("=" * 70)
    print("MOAI HYPERLOCALITY  (ahu-placed, GEOGRAPHIC spatial-cluster demes, "
          "categorical style)")
    print("=" * 70)
    print(f"{len(df)} ahu-placed moai in {len(deme_ids)} spatial clusters "
          f"(complete linkage, k={moai.PRIMARY_K})")
    print(f"cluster sizes: {sizes}")
    print(f"loci used ({len(loci)}): {', '.join(loci)}")

    # per-locus
    counts = moai.all_locus_counts(df, demes, loci)
    print("\n[per-locus G_ST vs panmixia null]")
    for a, m in counts.items():
        obs, p, nm, _ = popgen.gst_permutation(m, n_perm=9999, rng=RNG)
        print(f"  {a:<24} n={int(m.sum()):<4} K={m.shape[1]}  "
              f"G_ST={obs:.4f}  null={nm:.4f}  p={p:.4g}  ({obs/nm:.1f}x)")

    # multilocus
    obs, p, nm, null = moai.multilocus_permutation(df, demes, loci, n_perm=9999, rng=RNG)
    print("\n[multilocus cultural F_ST]")
    print(f"  Nei multilocus G_ST = {obs:.4f}")
    print(f"  panmixia null mean  = {nm:.4f}  (95th pct {np.percentile(null,95):.4f})")
    print(f"  p(>=obs|panmixia)   = {p:.4g}  -> "
          f"{'STRUCTURED' if p < 0.05 else 'no detectable structure'}")
    print(f"  excess over null    = {obs-nm:+.4f}  ({obs/nm:.1f}x)")
    print(f"  Bell F_ST (mean over loci) = "
          f"{np.mean([popgen.fst_bell(m) for m in counts.values()]):.4f}")
    lo, med, hi = multilocus_bootstrap_ci(df, demes, loci)
    print(f"  bootstrap 95% CI    = [{lo:.4f}, {hi:.4f}]  median {med:.4f}")

    # leave-one-deme-out
    print("\n[leave-one-cluster-out multilocus F_ST]")
    for d in deme_ids:
        mask = demes != d
        o, pp, _, _ = moai.multilocus_permutation(
            df[mask].reset_index(drop=True), demes[mask], loci, n_perm=4999, rng=RNG)
        print(f"  drop cluster {d:<6} F_ST={o:.4f}  p={pp:.4g}")

    # resolution robustness: the result is stable across the number of spatial
    # clusters and strengthens at finer (few-km) grain -- the moai analogue of the
    # mata'a within-cluster test. 'single' linkage is excluded (chains coastal ahu).
    print("\n[resolution robustness — spatial clusters at k = 6, 8, 10]")
    for k in (6, 8, 10):
        sd = moai.spatial_demes(df, k=k, method="complete")
        keep = [d for d in pd.unique(sd) if (sd == d).sum() >= 5]
        mask = np.isin(sd, keep)
        o, pp, nmk, _ = moai.multilocus_permutation(
            df[mask].reset_index(drop=True), sd[mask], loci, n_perm=4999, rng=RNG)
        print(f"  k={k:<2}  demes={len(keep):<2} (n>=5)  n={int(mask.sum()):<3}  "
              f"G_ST={o:.4f}  null={nmk:.4f}  ({o/nmk:.1f}x)  p={pp:.4g}")

    # clan cross-check ONLY: confirm the geographic result does not depend on
    # dropping the ethnohistoric clan model. Not the analytical unit.
    cl = df[df["CLAN_BOUNDARY"] != "NONE"].reset_index(drop=True)
    cdem = cl["CLAN_BOUNDARY"].values
    oc, pc, nmc, _ = moai.multilocus_permutation(cl, cdem, loci, n_perm=4999, rng=RNG)
    print("\n[clan cross-check — NOT the analytical unit, reported for comparison only]")
    print(f"  ethnohistoric territories (n={len(cl)}, {len(set(cdem))} clans): "
          f"G_ST={oc:.4f}  null={nmc:.4f}  ({oc/nmc:.1f}x)  p={pc:.4g}")

    # IBD Mantel on spatial-cluster centroids
    Dgeo = moai.deme_distance_km(df, demes)
    Dcomp = moai.compositional_distance(counts)
    span = Dgeo[np.triu_indices_from(Dgeo, 1)].max()
    print(f"\n[IBD Mantel]  {len(deme_ids)} cluster centroids, max sep {span:.1f} km, "
          f"{len(deme_ids)*(len(deme_ids)-1)//2} pairs")
    for meth in ("pearson", "spearman"):
        r, pr = popgen.mantel(Dcomp, Dgeo, n_perm=9999, method=meth, rng=RNG)
        print(f"  {meth:<8} r={r:+.3f}  p={pr:.4g}")
    _ibd_plot(Dgeo, Dcomp)

    _continuous_complement(df, demes, deme_ids)
    return obs, p


def _continuous_complement(df, demes, deme_ids):
    """Scale-free shape ratios: among-cluster eta^2 with a label-permutation test.
    Underpowered (sparse measurement) and reported as such; size itself is omitted
    because moai size is time-transgressive."""
    print("\n[continuous scale-free ratios — COMPLEMENT, time-transgression caveat]")
    def num(c):
        return pd.to_numeric(df[c], errors="coerce").values
    ratios = {
        "FACE_WIDTH/FACE_LENGTH": num("FACE_WIDTHcm") / num("FACE_LENGTHcm"),
        "HEAD_WIDTH/HEAD_DEPTH":  num("HEAD_WIDTHcm") / num("HEAD_DEPTHcm"),
        "BASE_WIDTH/BASE_DEPTH":  num("BASE_WIDTHcm") / num("BASE_DEPTHcm"),
    }
    d = np.asarray(demes)
    for name, r in ratios.items():
        ok = np.isfinite(r)
        if ok.sum() < 20:
            print(f"  {name:<24} n={ok.sum()} too sparse — skipped")
            continue
        rv, dv = r[ok], d[ok]
        eta2 = _eta2(rv, dv)
        rng = np.random.default_rng(20260623)
        null = np.array([_eta2(rv, rng.permutation(dv)) for _ in range(4999)])
        p = (np.sum(null >= eta2) + 1) / 5000
        present = [c for c in deme_ids if (dv == c).sum() >= 3]
        print(f"  {name:<24} n={ok.sum():<3} clusters>=3:{len(present)}  "
              f"eta^2={eta2:.3f}  p={p:.4g}")


def _eta2(values, groups):
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    ss_between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        ss_between += len(v) * (v.mean() - grand) ** 2
    return ss_between / ss_tot if ss_tot > 0 else 0.0


def _ibd_plot(Dgeo, Dcomp, path="output/hyperlocality_moai.png"):
    iu = np.triu_indices_from(Dgeo, 1)
    x, y = Dgeo[iu], Dcomp[iu]
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(x, y, s=34, color="#1e8449", alpha=0.85, zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, a + b * xs, color="#c0392b", lw=1.6,
            label=f"slope {b:.4f}/km   r={r:.2f}")
    ax.set_xlabel("geographic distance between cluster centroids (km)")
    ax.set_ylabel(r"compositional distance ($\sum$ Neiman $d^2$)")
    ax.set_title("Moai isolation by distance (spatial clusters)")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    report()
