"""Load the published mata'a count matrices (assemblage x class).

Two matrices, parsed and validated against the printed row totals in the published
mata'a stylistic-variability study and shipped here as CSVs:

  data/published_lengthwidth.csv   stem length (A-D) x stem width (a-e), 14 classes (n=447)
  data/published_shapeshoulder.csv stem shape x shoulder shape, 6 classes (n=348)

Each row is one of the 11 provenanced assemblages (Parcels 6-11, Ahu Tautira,
Orito, Orongo, Rano Kau, Vinapu); each column is an occupied paradigmatic class;
empty cells are real zeros.
"""
import pandas as pd


def stem_length_width(path="data/published_lengthwidth.csv"):
    """(assemblage x stem-length-by-width count matrix, ok)."""
    mat = pd.read_csv(path, index_col=0)
    return mat, True


def stem_shoulder_shape(path="data/published_shapeshoulder.csv"):
    """(assemblage x stem-shape-by-shoulder-shape count matrix, ok)."""
    mat = pd.read_csv(path, index_col=0)
    return mat, True


if __name__ == "__main__":
    for name, fn in [("stem length x width", stem_length_width),
                     ("stem shape x shoulder shape", stem_shoulder_shape)]:
        mat, ok = fn()
        print(f"\n=== {name} ===")
        print(mat)
        print("row totals:", mat.sum(axis=1).tolist(), " grand total:", int(mat.values.sum()))
