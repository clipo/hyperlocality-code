"""Cross-linkage robustness of the moai geographic-cluster differentiation (k=6).

The primary moai analysis uses complete-linkage spatial clustering. To show k=6 is
not a complete-linkage artifact, we recompute the multilocus Nei G_ST and its
panmixia-permutation null under complete, average, ward, and single linkage. Reports
the ratio G_ST / null-mean and the permutation p for each (committed so the
manuscript's cross-linkage numbers trace to code).

Pure numpy + (PyMC-free) moai/popgen. Run:
  PYTHONPATH=src python3 src/run_moai_linkage_sweep.py
"""
import json
import numpy as np

import moai as moai_mod
import popgen

SEED = 20260628
N_PERM = 9999


def main():
    df = moai_mod.load()
    loci = moai_mod.select_loci(df)
    rng = np.random.default_rng(SEED)
    out = {}
    print(f"moai multilocus G_ST by linkage method (k=6, {N_PERM} permutations)")
    for method in ["complete", "average", "ward", "single"]:
        demes = moai_mod.spatial_demes(df, k=6, method=method).astype(str)
        counts = moai_mod.all_locus_counts(df, demes, loci)
        g = moai_mod.multilocus_gst(counts)
        null = np.array([moai_mod.multilocus_gst(
            moai_mod.all_locus_counts(df, rng.permutation(demes), loci))
            for _ in range(N_PERM)])
        nm = float(null.mean())
        p = (np.sum(null >= g) + 1) / (N_PERM + 1)
        out[method] = {"gst": float(g), "null_mean": nm, "ratio": float(g / nm),
                       "p": float(p), "n_clusters": int(len(set(demes)))}
        print(f"  {method:9s} G_ST={g:.4f}  null={nm:.4f}  ratio={g/nm:.2f}x  p={p:.4f}")
    json.dump(out, open("output/moai_linkage_sweep.json", "w"), indent=2)
    print("\nwrote output/moai_linkage_sweep.json")


if __name__ == "__main__":
    main()
