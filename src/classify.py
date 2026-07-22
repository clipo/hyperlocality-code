"""Paradigmatic classification + assemblage x class frequency matrix.

Two scale-free dimensions (the dimensions of the published *first* seriation,
Lipo-Hunt-Hundtoft 2014):
  Dimension 1  Stem length/width ratio   -> S (low) | M | L (high)
  Dimension 2  Shoulder angle            -> a (acute) | b | c (open)

Each split into 3 states at the 33rd/67th percentiles of the provenanced sample
(the paper split stem ratio into 3 and shoulder angle into 3 around the mean).
A paradigmatic class is the intersection, e.g. 'S-a', giving up to 9 classes.
"""
import numpy as np
import pandas as pd
from mataa_io import load_ws, ASSEMBLAGES
import morphometrics as mm

ORDER = ASSEMBLAGES  # rows


def quantile_labeler(values, labels):
    n = len(labels)
    cuts = np.nanpercentile(values, [100 * k / n for k in range(1, n)])

    def lab(v):
        if np.isnan(v):
            return None
        return labels[int(np.searchsorted(cuts, v, side="right"))]
    return lab, cuts


def build(n_states=3):
    """n_states states per dimension (2 -> 4 classes, 3 -> 9 classes)."""
    meta, outlines = load_ws("data/xyfinaldatawithids-smallassemblagesremoved.txt")
    rec = []
    for i in range(len(outlines)):
        m = mm.metrics(outlines[i])
        rec.append((meta["ID"].iloc[i], meta["Site"].iloc[i],
                    m["lw_ratio"], m["shoulder_angle"]))
    df = pd.DataFrame(rec, columns=["ID", "Site", "lw_ratio", "shoulder_angle"])
    prov = df[df["Site"].isin(ASSEMBLAGES)].copy()

    lw_labels = ["S", "M", "L"][:n_states] if n_states == 3 else ["S", "L"]
    an_labels = ["a", "b", "c"][:n_states] if n_states == 3 else ["a", "c"]
    lw_lab, lw_cut = quantile_labeler(prov["lw_ratio"].values, lw_labels)
    an_lab, an_cut = quantile_labeler(prov["shoulder_angle"].values, an_labels)
    prov = prov.copy()
    prov["lw_c"] = prov["lw_ratio"].map(lw_lab)
    prov["an_c"] = prov["shoulder_angle"].map(an_lab)
    prov["class"] = prov["lw_c"] + prov["an_c"]

    classes = [f"{x}{y}" for x in lw_labels for y in an_labels]
    mat = (prov.groupby(["Site", "class"]).size().unstack(fill_value=0)
           .reindex(index=ORDER, columns=classes, fill_value=0))
    return prov, mat, {"lw_cut": lw_cut, "an_cut": an_cut}


if __name__ == "__main__":
    prov, mat, cuts = build()
    print("lw_ratio tertile cuts:", np.round(cuts["lw_cut"], 3),
          " shoulder-angle cuts:", np.round(cuts["an_cut"], 1))
    print("\n=== counts: assemblage x class ===")
    print(mat)
    print("\nrow totals:\n", mat.sum(1))
    print("\ncolumn totals:\n", mat.sum(0))
    mat.to_csv("output/frequency_matrix_counts.csv")
    pct = mat.div(mat.sum(1), axis=0) * 100
    pct.round(1).to_csv("output/frequency_matrix_pct.csv")
    print("\n=== row percentages ===")
    print(pct.round(1))
