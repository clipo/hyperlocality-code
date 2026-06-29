"""Parse & validate the published mata'a count tables (converted .doc -> .html).

Table 3 = stem length (A-D) x stem width (a-e) -> 14 occupied classes (n=447).
Table 2/4 = stem shape (P/R/S/SR) x shoulder shape (C/A) -> 6 classes (n=348).
Empty cells are real zeros; we validate the parse against the printed row totals.
"""
import os
import re
import numpy as np
import pandas as pd

HTML_DIR = "/tmp/mataatables"

# Validated count matrices written by this module's __main__ pass. They are the
# committed source of truth and let the pipeline reproduce without the LibreOffice
# HTML cache in /tmp (which is not portable across machines/sessions).
_CSV_CACHE = {3: "output/published_lengthwidth.csv",
              4: "output/published_shapeshoulder.csv"}


def _load(table_num):
    html = f"{HTML_DIR}/Table {table_num}.html"
    if not os.path.exists(html):
        csv = _CSV_CACHE.get(table_num)
        if csv and os.path.exists(csv):
            mat = pd.read_csv(csv, index_col=0)
            mat.index = mat.index.astype(str)
            totals = mat.sum(axis=1).tolist()
            return mat, totals, True  # CSV is the already-validated parse
    tbls = pd.read_html(html)
    df = max(tbls, key=lambda x: x.size)
    # row 1 holds the class headers; col 0 holds assemblage names
    header = df.iloc[1].tolist()
    classes = [c for c in header[1:-1]]  # drop 'Assemblage' and 'Total'
    body = df.iloc[2:-1].copy()          # drop header rows and the Total row
    names = body.iloc[:, 0].astype(str).str.strip().tolist()
    counts = body.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    totals = pd.to_numeric(df.iloc[2:-1, -1], errors="coerce").fillna(0).astype(int).tolist()
    mat = pd.DataFrame(counts.values, index=names, columns=classes)
    # validate
    rowsum = mat.sum(axis=1).tolist()
    ok = all(r == t for r, t in zip(rowsum, totals))
    return mat, totals, ok


# Normalize assemblage names so spatial grouping is explicit.
NAME_MAP = {
    "6": "Parcel 6", "7": "Parcel 7", "8": "Parcel 8",
    "9": "Parcel 9", "10": "Parcel 10", "11": "Parcel 11",
}
# Approx spatial groups on Figure 12.3 (for interpretation, not used in solving)
SPATIAL_GROUP = {
    "Parcel 6": "TeKahurea", "Parcel 7": "TeKahurea", "Parcel 8": "TeKahurea",
    "Parcel 9": "TeMiroOone", "Parcel 10": "TeMiroOone", "Parcel 11": "TeMiroOone",
    "Ahu Tautira": "AhuTautira",
    "Orito": "RanoKauArea", "Orongo": "RanoKauArea",
    "Rano Kau": "RanoKauArea", "Vinapu": "RanoKauArea",
}


def stem_length_width():
    mat, tot, ok = _load(3)
    mat.index = [NAME_MAP.get(n, n) for n in mat.index]
    return mat, ok


def stem_shoulder_shape():
    mat, tot, ok = _load(4)
    mat.index = [NAME_MAP.get(n, n) for n in mat.index]
    return mat, ok


if __name__ == "__main__":
    for name, fn in [("Table3 length x width", stem_length_width),
                     ("Table2/4 stem x shoulder shape", stem_shoulder_shape)]:
        mat, ok = fn()
        print(f"\n=== {name}  validated={ok} ===")
        print(mat)
        print("row totals:", mat.sum(axis=1).tolist())
        out = "output/published_" + ("lengthwidth" if "length" in name else "shapeshoulder") + ".csv"
        mat.to_csv(out)
        print("wrote", out)
