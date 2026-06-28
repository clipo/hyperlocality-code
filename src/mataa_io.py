"""Shared loader for the Rapa Nui mata'a outline data.

Each biface is recorded as a closed outline of 200 (X,Y) points, preceded by
metadata columns: ID, Island, Collection, Site, Source.
"""
import numpy as np
import pandas as pd

META_COLS = ["ID", "Island", "Collection", "Site", "Source"]
N_POINTS = 200

# Sites that can be placed on the Rapa Nui map (Figure 3). "Unknown" is the
# large unprovenanced Bishop Museum collection.
PROVENANCED = ["Ahu Tautira", "Orongo", "Orito", "Rano Kau"]

# Assemblages used for the location-based seriation. These are the provenanced
# Rapa Nui sites with digitized outlines, plus the pooled Parcela survey pieces.
# (The published chapter's Vinapu and the Te Kahurea / Te Miro O'one parcel
# split came from separate Canvas measurements and are not resolvable here.)
ASSEMBLAGES = ["Ahu_Tautira", "Orongo", "Orito", "Rano_Kau", "Parcela"]


def load_ws(path):
    """Load the whitespace-delimited variant (has the 6 Parcela pieces and
    underscore-joined single-token site names). CR line terminators."""
    with open(path, "r", errors="replace") as f:
        text = f.read().replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    rows = [ln.split() for ln in lines[1:]]
    meta_rows, coord_rows = [], []
    for r in rows:
        meta_rows.append(r[:5])
        nums = np.array(r[5:5 + 2 * N_POINTS], dtype=float)
        coord_rows.append(nums)
    meta = pd.DataFrame(meta_rows, columns=META_COLS)
    arr = np.vstack(coord_rows)
    xs = arr[:, 0::2]
    ys = arr[:, 1::2]
    outlines = np.stack([xs, ys], axis=-1)
    return meta, outlines


def load(path):
    """Return (meta_df, outlines) where outlines is (n, 200, 2) float array.

    Coordinates are image pixel coordinates; Y is flipped so shapes plot
    'tip up' the way they are drawn.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Clean stray whitespace in the metadata
    for c in META_COLS:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    coord_cols = [c for c in df.columns if c.startswith("X") or c.startswith("Y")]
    xs = df[[f"X{i}" for i in range(1, N_POINTS + 1)]].to_numpy(float)
    ys = df[[f"Y{i}" for i in range(1, N_POINTS + 1)]].to_numpy(float)
    outlines = np.stack([xs, ys], axis=-1)  # (n, 200, 2)
    meta = df[META_COLS].copy()
    return meta, outlines


if __name__ == "__main__":
    meta, outlines = load("data/xyfinaldatawithids.csv")
    print("rows:", len(meta), "outline shape:", outlines.shape)
    print(meta["Site"].value_counts())
    # sanity: any NaNs in coordinates?
    print("NaN coords in any row:", np.isnan(outlines).any(axis=(1, 2)).sum())
