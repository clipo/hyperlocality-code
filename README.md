# hyperlocality-code

Reproduction code and data for **"Measuring hyperlocality: cultural F~ST~ and
fine-grained spatial differentiation in Rapa Nui material culture"**
(Lipo, DiNapoli & Hunt).

This repository contains everything needed to reproduce the quantitative results
and figures in the paper from the source data, using only open Python scientific
libraries. It is a self-contained subset of the analysis: the manuscript text and
copyrighted source PDFs are not included.

## What the paper tests

Rapa Nui (Easter Island, Chile) is 164 km² and can be crossed on foot in a day, so
an island-wide pool of social interaction was available. Most models of cultural
transmission predict that such a pool erases differences between places. The
artifact record shows the opposite: material culture is patterned at a fine spatial
grain. We test the prediction of Lipo, DiNapoli, Madsen & Hunt (2021, *PLOS ONE*
e0250690) that variation in four artifact classes is structured by location, by
treating classes as alleles and spatial groups of artifacts as demes, measuring
between-place structure as **cultural F~ST~** against a **panmixia permutation null**,
and adding **within-cluster** and **isolation-by-distance** tests to ask whether the
structure is local.

The four proxies are *mata'a* (stemmed obsidian tools), *umu* (stone-lined earth
ovens), *pukao* (carved red scoria hats), and *moai* (the statues).

## Repository layout

```
src/                  analysis code (run from the repo root with PYTHONPATH=src)
  popgen.py           cultural F_ST (Nei G_ST), panmixia null, Bell estimator,
                      bootstrap CIs, Mantel / partial Mantel
  spatial.py          mata'a assemblage coordinates (from the provenance map)
  tables_io.py        loads the published mata'a count matrices (CSV)
  umu.py  moai.py  pukao.py    per-proxy data loaders + deme construction
  figbase.py          basemap tiles for the figures
  run_hyperlocality.py    mata'a: F_ST, within-cluster, isolation-by-distance
  harden.py               mata'a robustness battery (jitter, bootstrap, LOO)
  run_umu.py  run_moai.py  run_pukao.py    the other proxies
  run_moai_spatial.py     moai individual-statue spatial structure (no clustering)
  check_source.py         obsidian-source provenance counts
  make_figures.py         all figures, written as .png, .pdf and .svg
data/                 source data (see "Data" below)
output/               script results are written here
figures/              figures are written here
reproduce.sh          runs the whole pipeline
requirements.txt      Python dependencies
```

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

(numpy, scipy, pandas, matplotlib, openpyxl, Pillow.)

## Reproduce everything

```
bash reproduce.sh
```

This runs every analysis script, writes the numeric results to `output/`, and writes
the figures to `figures/` in three formats (`.png`, `.pdf`, `.svg`). The figures step
downloads a shaded-relief basemap and so needs an internet connection; the numeric
results do not.

To run a single step:

```
PYTHONPATH=src python3 src/run_hyperlocality.py
```

All randomized tests use fixed seeds (20260623 for the main analyses, 7 for the
robustness battery), so results are bit-for-bit reproducible. Headline G~ST~ and
island-wide Mantel tests use 9,999 permutations.

## Expected results

| Proxy | Cultural F~ST~ | Panmixia null | Ratio | p |
|---|---|---|---|---|
| *mata'a* stem length × width | 0.053 | 0.022 | 2.4× | 0.0001 |
| *umu* oven style | 0.082 | 0.012 | 7.1× | 0.0001 |
| *moai* style (multilocus) | 0.087 | 0.042 | 2.1× | 0.0001 |
| *pukao* | 0.13–0.22 | ~0.15 | 1.1–1.3× | n.s. |

Locality checks: *mata'a* differentiation holds **within a 5.5 km cluster**
(F~ST~ = 0.037, null 0.016, p = 0.0008); the island-wide Mantel signal is a regional
(east vs southwest) contrast, not a smooth gradient (partial Mantel controlling
region is not significant). *moai* style is more alike at short range between
different *ahu* (Mantel r = 0.055, p = 0.04; significant only below ~4 km). *pukao*
is uninformative (smallest sample; wide null).

## Data

All data here are either public databases or counts/coordinates read from published
figures and tables. Copyrighted source PDFs are **not** redistributed.

- `data/published_lengthwidth.csv`, `data/published_shapeshoulder.csv` — the
  *mata'a* paradigmatic-class count matrices (11 provenanced assemblages × occupied
  classes), parsed from the published stylistic-variability study and validated
  against the printed row totals.
- `data/xyfinaldatawithids.csv` — *mata'a* outline dataset with the obsidian
  `Source` field used by `check_source.py`.
- `data/Figure3.svg` — the *mata'a* provenance map, included as the source from
  which assemblage coordinates were read (the coordinates themselves are encoded in
  `src/spatial.py`).
- `data/umu/umu_pae_table1.csv` — McCoy's earth-oven rim-style tallies by survey
  quadrangle.
- `data/moai/MOAI_DATABASE_PUBLIC.xlsx` — the public *moai* database (coordinates +
  categorical style attributes).
- `data/pukao/Pukao.csv` — recorded *pukao* with coordinates and categorical
  attributes.

### Known data limitations (stated honestly)

- *mata'a* and *umu* coordinates are **read from published maps**, not field GPS, so
  the spatial claims rest on rank-based, jitter-robust, within-cluster tests rather
  than absolute distances.
- The *moai* and *pukao* style attributes come from existing databases and carry
  whatever inter-observer variation the original scoring introduced.
- Assemblages are time-averaged; the analysis describes between-community structure
  accumulated over each assemblage's deposition interval.

## Method in one paragraph

For each proxy we form a frequency matrix of classes by deme and compute Nei's
multilocus G~ST~ (the cultural F~ST~). Significance is assessed against a panmixia
null that re-deals every artifact into demes of the observed sizes and recomputes
G~ST~ on each of 9,999 random partitions; the reported p is the share of null values
at least as large as the observed one. We cross-check with the Bell variance-ratio
estimator and report within-deme bootstrap intervals. Locality is tested by
restricting to a tight spatial cluster and by Mantel / partial Mantel correlation of
compositional distance with geographic distance. See `src/popgen.py`.

## Citation

If you use this code or data, please cite the paper (Lipo, DiNapoli & Hunt) and the
model it tests:

> Lipo, C.P., DiNapoli, R.J., Madsen, M.E., Hunt, T.L. 2021. Population structure
> drives cultural diversity in finite populations: A hypothesis for localized
> community patterns on Rapa Nui (Easter Island). *PLOS ONE* 16(3): e0250690.

## License

MIT (see `LICENSE`).
