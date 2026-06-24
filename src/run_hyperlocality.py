"""Test mata'a hyperlocality: is assemblage similarity structured by space, far
more than a person-can-cross-it-in-a-day island would predict?

Two signatures, following ../mls-emergence:
  (3) cultural F_ST (Nei G_ST) vs a panmixia null  -> is there ANY between-place
      structure beyond sampling noise?
  (4) isolation-by-distance (Mantel: compositional distance ~ geographic km)
      -> is that structure SPATIAL, decaying with distance over a few km?

Run on the published Canvas count tables (Approach B, where real structure lives)
and, for contrast, on the scale-free outline classes (Approach A).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spatial
import popgen
import tables_io


def analyze(mat, label, plot_path=None):
    counts = mat.values.astype(float)
    names = list(mat.index)
    keep = [i for i, n in enumerate(names) if spatial.has_coords(n)]
    names_g = [names[i] for i in keep]
    counts_g = counts[keep]

    print(f"\n================ {label} ================")
    print(f"{len(names)} assemblages, {mat.shape[1]} classes, "
          f"n = {int(counts.sum())} artifacts")

    # --- Signature 3: cultural F_ST vs panmixia ---
    obs, p, null_mean, null = popgen.gst_permutation(counts, n_perm=9999)
    print(f"\n[3] cultural F_ST (Nei G_ST) = {obs:.4f}")
    print(f"    panmixia null mean       = {null_mean:.4f}  "
          f"(95th pct {np.percentile(null, 95):.4f})")
    print(f"    p(F_ST >= obs | panmixia) = {p:.4g}   "
          f"-> {'structured' if p < 0.05 else 'no detectable structure'}")
    print(f"    excess over null          = {obs - null_mean:+.4f} "
          f"({obs / null_mean:.1f}x)")

    # --- Signature 4: isolation by distance ---
    if len(names_g) >= 4:
        Dgeo = spatial.distance_matrix(names_g)
        Dd2 = popgen.neiman_d2(counts_g)
        Dfst = popgen.pairwise_gst(counts_g)
        for dname, Dcomp in [("Neiman d^2", Dd2), ("pairwise G_ST", Dfst)]:
            r, pr = popgen.mantel(Dcomp, Dgeo, n_perm=9999, method="pearson")
            rs, prs = popgen.mantel(Dcomp, Dgeo, n_perm=9999, method="spearman")
            print(f"\n[4] IBD Mantel ({dname} vs km), {len(names_g)} placed assemblages:")
            print(f"    Pearson  r = {r:+.3f}  p = {pr:.4g}")
            print(f"    Spearman r = {rs:+.3f}  p = {prs:.4g}  "
                  f"-> {'isolation-by-distance' if pr < 0.05 else 'no IBD'}")
        if plot_path:
            _ibd_plot(Dgeo, Dd2, names_g, label, plot_path)
    else:
        print("\n[4] too few geolocated assemblages for IBD")
    return obs, p


def _ibd_plot(Dgeo, Dcomp, names, label, path):
    iu = np.triu_indices_from(Dgeo, k=1)
    x, y = Dgeo[iu], Dcomp[iu]
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.scatter(x, y, s=28, color="#3a6ea5", alpha=0.8, zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, a + b * xs, color="#c0392b", lw=1.6,
            label=f"slope {b:.4f}/km   r={r:.2f}")
    ax.set_xlabel("geographic distance (km)")
    ax.set_ylabel("compositional distance  (Neiman $d^2$)")
    ax.set_title(f"Isolation by distance — {label}")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"    wrote {path}")


def within_cluster(mat, label, cluster):
    """Sharper hyperlocality test: is there differentiation WITHIN a tight
    spatial cluster (here the SW sites, all within ~5 km), where an island-wide
    interaction model predicts none? Tests F_ST-vs-panmixia and IBD on the subset.
    """
    names = list(mat.index)
    idx = [names.index(c) for c in cluster if c in names]
    sub = mat.iloc[idx]
    counts = sub.values.astype(float)
    D = spatial.distance_matrix(list(sub.index))
    span = D[np.triu_indices_from(D, 1)].max()
    print(f"\n---------------- {label}: within-cluster ({len(idx)} sites, "
          f"max sep {span:.1f} km) ----------------")
    obs, p, nm, _ = popgen.gst_permutation(counts, n_perm=9999)
    print(f"    F_ST = {obs:.4f}  null {nm:.4f}  p = {p:.4g}  "
          f"({obs / nm:.1f}x null)")
    r, pr = popgen.mantel(popgen.neiman_d2(counts), D, n_perm=9999, method="spearman")
    print(f"    IBD Spearman r = {r:+.3f}  p = {pr:.4g}")


SW = ["Ahu Tautira", "Orongo", "Orito", "Rano Kau", "Vinapu"]

if __name__ == "__main__":
    # Approach B: published Canvas counts (11 assemblages, real measurements)
    lw, _ = tables_io.stem_length_width()
    analyze(lw, "B: stem length x width (Table 3)",
            "output/hyperlocality_lengthwidth.png")
    within_cluster(lw, "B length x width", SW)
    ss, _ = tables_io.stem_shoulder_shape()
    analyze(ss, "B: stem shape x shoulder shape (Table 2/4)",
            "output/hyperlocality_shapeshoulder.png")
    within_cluster(ss, "B shape x shoulder", SW)

    # Approach A: scale-free outline classes (5 assemblages) -- contrast
    try:
        from classify import build
        _, matA, _ = build(n_states=3)
        analyze(matA, "A: outline metric classes (3x3)",
                "output/hyperlocality_outline.png")
    except Exception as e:
        print(f"\n[A outline classes skipped: {e}]")
