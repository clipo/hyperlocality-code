#!/usr/bin/env python
"""Extract ahu-placed moai locations + heights from the public moai database to CSV.

Writes data/moai/moai_locations_heights.csv: one row per ahu-placed moai that
carries coordinates (n=481), from the public moai database
(data/moai/MOAI_DATABASE_PUBLIC.xlsx; Schumacher 2013). This is the location/height
source the intervisibility precompute needs (src/ahu_viewshed.py groups these by
ahu), so committing the CSV makes that step self-contained -- no external inventory.

Columns:
  moai_id        OBJECTID from the database (stable per record)
  ahu            grouping key: LOCATION_NAME uppercased, punctuation stripped,
                 whitespace collapsed (merges spelling/punctuation variants of the
                 same platform, e.g. "Ahu Hanga Te'e" / "AHU HANGA TE E")
  location_name  raw LOCATION_NAME from the database (provenance)
  longitude      decimal degrees (WGS84)
  latitude       decimal degrees (WGS84)
  total_length_cm  moai total length in cm (blank where the database has no numeric
                 measurement); divide by 100 for height in meters

Run:  python3 src/make_moai_csv.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

XLSX = Path("data/moai/MOAI_DATABASE_PUBLIC.xlsx")
OUT = Path("data/moai/moai_locations_heights.csv")


def norm_ahu(s):
    s = str(s).upper().strip()
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    df = pd.read_excel(XLSX)
    for c in ("longitude", "latitude", "TOTAL_LENGTH_cm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    a = df[(df["LOCATION_TYPE"] == "AHU")
           & df["longitude"].notna() & df["latitude"].notna()].copy()
    a["ahu"] = a["LOCATION_NAME"].map(norm_ahu)
    a = a.sort_values(["ahu", "OBJECTID"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["moai_id", "ahu", "location_name",
                    "longitude", "latitude", "total_length_cm"])
        for _, r in a.iterrows():
            h = r["TOTAL_LENGTH_cm"]
            w.writerow([int(r["OBJECTID"]), r["ahu"], r["LOCATION_NAME"],
                        f"{r['longitude']:.8f}", f"{r['latitude']:.8f}",
                        "" if pd.isna(h) else f"{h:g}"])
    n_h = int(a["TOTAL_LENGTH_cm"].notna().sum())
    print(f"wrote {OUT}: {len(a)} ahu-placed moai, {a['ahu'].nunique()} distinct ahu, "
          f"{n_h} with a numeric height")


if __name__ == "__main__":
    main()
