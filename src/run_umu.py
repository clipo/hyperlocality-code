"""Umu pae hyperlocality analysis (McCoy 1978 Table 1). Mirrors run_pukao.py.

Redirect stdout to output/umu_results.txt; writes output/hyperlocality_umu.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import popgen
import umu

RNG = np.random.default_rng(20260623)


def report():
    df = umu.load()
    C = umu.counts(df)
    demes = list(df["quadrangle"])
    n_i = C.sum(1).astype(int)

    print("=" * 70)
    print("UMU PAE HYPERLOCALITY  (McCoy 1978 Table 1)")
    print("=" * 70)
    print(f"{C.shape[0]} demes (survey quadrangles), {C.shape[1]} styles "
          f"({', '.join(umu.STYLES)})")
    print(f"n = {int(C.sum())} classifiable ovens; per-deme n = "
          f"{dict(zip(demes, n_i))}")
    # within-deme clustering caveat (source: data CSV header / McCoy 1976):
    # 33 of Rano Kau's 54 circular ovens are at one site (r-21, near Orongo),
    # which is why dropping Rano Kau roughly halves F_ST below.
    print("  NB 33 of Rano Kau's 54 circular ovens are at one site (r-21) "
          "-> within-deme clustering caveat")

    # style frequencies per deme -- shows the W->E gradient
    P = C / C.sum(1, keepdims=True)
    print("\n[style frequencies per deme]")
    print("  deme            " + "".join(f"{s[:5]:>8}" for s in umu.STYLES))
    for q, row in zip(demes, P):
        print(f"  {q:<15}" + "".join(f"{v:8.2f}" for v in row))

    # --- cultural F_ST vs panmixia ---
    obs, p, nm, null = popgen.gst_permutation(C, n_perm=9999, rng=RNG)
    print("\n[cultural F_ST vs panmixia null]")
    print(f"  Nei G_ST           = {obs:.4f}")
    print(f"  panmixia null mean = {nm:.4f}  (95th pct {np.percentile(null,95):.4f})")
    print(f"  p(>=obs|panmixia)  = {p:.4g}  -> "
          f"{'STRUCTURED' if p < 0.05 else 'no detectable structure'}")
    print(f"  excess over null   = {obs-nm:+.4f}  ({obs/nm:.1f}x)")
    print(f"  Bell variance-ratio F_ST = {popgen.fst_bell(C):.4f}")
    (lo, med, hi), _ = popgen.gst_bootstrap_ci(C, n_boot=2000, rng=RNG)
    print(f"  bootstrap 95% CI   = [{lo:.4f}, {hi:.4f}]  median {med:.4f}")

    # --- leave-one-deme-out ---
    print("\n[leave-one-deme-out F_ST]")
    for i, q in enumerate(demes):
        sub = np.delete(C, i, axis=0)
        o, pp, _, _ = popgen.gst_permutation(sub, n_perm=4999, rng=RNG)
        print(f"  drop {q:<15} F_ST={o:.4f}  p={pp:.4g}")

    # --- per-style marginal F_ST (which class drives it) ---
    print("\n[per-style marginal F_ST (style k vs rest)]")
    tot = C.sum(1, keepdims=True)
    for k, s in enumerate(umu.STYLES):
        binary = np.column_stack([C[:, k], tot[:, 0] - C[:, k]])
        o, pp, _, _ = popgen.gst_permutation(binary, n_perm=4999, rng=RNG)
        print(f"  {s:<12} F_ST={o:.4f}  p={pp:.4g}")

    # --- pairwise G_ST ---
    Dfst = popgen.pairwise_gst(C)
    print("\n[pairwise G_ST]")
    print("  " + " " * 15 + "".join(f"{q[:6]:>8}" for q in demes))
    for i, q in enumerate(demes):
        print(f"  {q:<15}" + "".join(f"{Dfst[i,j]:8.3f}" for j in range(len(demes))))

    # --- IBD Mantel (EXPLORATORY: 5 demes) ---
    Dgeo = umu.distance_km(df)
    Dd2 = popgen.neiman_d2(C)
    print(f"\n[IBD Mantel — EXPLORATORY, {len(demes)} demes / "
          f"{len(demes)*(len(demes)-1)//2} pairs, map-read coords]")
    for meth in ("pearson", "spearman"):
        r, pr = popgen.mantel(Dd2, Dgeo, n_perm=9999, method=meth, rng=RNG)
        r2, pr2 = popgen.mantel(Dfst, Dgeo, n_perm=9999, method=meth, rng=RNG)
        print(f"  {meth:<8} Neiman d^2 r={r:+.3f} p={pr:.4g} | "
              f"pairwise G_ST r={r2:+.3f} p={pr2:.4g}")
    _ibd_plot(Dgeo, Dd2, demes)
    return obs, p


def _ibd_plot(Dgeo, Dcomp, demes, path="output/hyperlocality_umu.png"):
    iu = np.triu_indices_from(Dgeo, 1)
    x, y = Dgeo[iu], Dcomp[iu]
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(x, y, s=34, color="#b9770e", alpha=0.85, zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, a + b * xs, color="#c0392b", lw=1.6,
            label=f"slope {b:.4f}/km   r={r:.2f}")
    ax.set_xlabel("geographic distance between quadrangles (km, map-read)")
    ax.set_ylabel(r"compositional distance (Neiman $d^2$)")
    ax.set_title("Umu pae isolation by distance (exploratory, n=5 demes)")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    report()
