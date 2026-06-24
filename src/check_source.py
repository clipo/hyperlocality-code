"""Obsidian-source homogeneity check for the mata'a proxy.

Addresses the reviewer concern that geochemical source (raw-material economy)
could drive the between-assemblage stem-form signal. Two facts come out of the
outline dataset `data/xyfinaldatawithids.csv`:

  1. Of the pieces with a determined source, the overwhelming majority are Orito
     (a single island source), so source variation is too small to generate the
     between-place stem differences we report.
  2. Source and assemblage provenance are recorded on disjoint subsets here: the
     sourced pieces are not provenanced to a named assemblage, and the
     provenanced pieces are not geochemically sourced. So a direct per-assemblage
     source-vs-form test is not possible with these data (flagged as future work).

Run: PYTHONPATH=src python src/check_source.py
"""
import pandas as pd

KNOWN = ["Orito", "Rano Kau 1", "Motu Iti"]


def main(path="data/xyfinaldatawithids.csv"):
    df = pd.read_csv(path)
    df["Site"] = df["Site"].astype(str)
    known = df[df["Source"].isin(KNOWN)]
    counts = known["Source"].value_counts()
    orito_pct = 100.0 * (known["Source"] == "Orito").mean()
    print(f"total pieces: {len(df)}")
    print(f"pieces with a determined source: {len(known)}")
    print("  by source:", dict(counts))
    print(f"  Orito share of determined-source pieces: {orito_pct:.1f}%")

    # are source and assemblage provenance recorded on the same pieces?
    prov_sites = sorted(s for s in df["Site"].unique()
                        if s not in ("Unknown", "nan"))
    sourced_with_site = known[~known["Site"].isin(["Unknown", "nan"])]
    print(f"\nprovenanced assemblages present: {prov_sites}")
    print("determined-source pieces that are also provenanced to a named "
          f"assemblage: {len(sourced_with_site)}")
    print("=> source and assemblage provenance are on disjoint subsets; no "
          "per-assemblage source test is possible with these data.")


if __name__ == "__main__":
    main()
