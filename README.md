# hyperlocality-code

Reproduction code and data for **"Measuring cultural hyperlocality: a multi-proxy
cultural F~ST~ test of bounded transmission on Rapa Nui (Easter Island)"**
(Lipo, DiNapoli & Hunt).

This repository reproduces the quantitative results and figures of the paper from the
source data, using open Python scientific libraries. The manuscript text and
copyrighted source PDFs are not included. The licensed 50 cm terrain model used for
the monument-intervisibility analysis is **not redistributable**; the committed
intermediates let the rest reproduce without it (see *Licensed DEM* below).

## What the paper tests

Rapa Nui is 164 km² and can be crossed on foot in a day, so an island-wide pool of
social interaction was available; most models of cultural transmission predict such a
pool erases between-place differences. The artifact record shows the opposite. We test
the prediction of Lipo, DiNapoli, Madsen & Hunt (2021, *PLOS ONE* e0250690) that
variation in four artifact classes is patterned by location, by treating classes as
alleles and spatial groups of artifacts as demes and estimating between-community
**cultural F~ST~**. The four proxies are *mata'a* (stemmed obsidian tools), *umu*
(earth ovens), *pukao* (topknots), and *moai* (the statues).

The analysis has three parts:

1. **Cultural F~ST~ (primary).** A hierarchical Bayesian (Balding–Nichols
   Dirichlet-multinomial) model estimates F~ST~ as a posterior and weighs structure
   against panmixia with a **Bayes factor**. The standard frequentist statistics
   (Nei G~ST~, the Bell estimator, permutation tests) are computed alongside **for
   comparison**.
2. **Spatial locality.** Within-cluster fits, isolation-by-distance regressions, and
   an individual-statue (clustering-free) test of *moai* style.
3. **Monument intervisibility and matched nulls.** Visual communities from the *ahu*
   intervisibility network, the cross-proxy concordance, and three spatially matched
   nulls (S-1: contiguity-preserving concordance null; S-2: intervisibility vs.
   contiguity-matched partition; S-3: distance-to-coast-matched viewshed null).

## Repository layout

```
src/                  analysis code (run from the repo root with PYTHONPATH=src)
  bayes.py            hierarchical Bayesian cultural F_ST + SMC Bayes factor (PyMC)
  run_bayes.py        primary Bayesian analysis -> output/bayes_results.json
  popgen.py           frequentist cultural F_ST (Nei G_ST), permutation null, Bell,
                      bootstrap CIs, Mantel / partial Mantel
  tables_io.py spatial.py umu.py moai.py pukao.py   data loaders + deme construction
  figbase.py          basemap tiles for the figures
  run_hyperlocality.py harden.py run_umu.py run_moai.py run_pukao.py
                      frequentist comparison + robustness battery
  run_moai_spatial.py        moai individual-statue spatial structure (no clustering)
  run_moai_linkage_sweep.py  moai cross-linkage robustness
  run_moai_territory_bayes.py Bayesian named-territory cross-check
  check_source.py            obsidian-source provenance counts
  ahu_viewshed.py viewshed_models.py   viewsheds + random-siting null (need the DEM)
  make_communities.py        visual communities from the viewshed matrix (DEM-free)
  run_concordance.py         moai style across the visual communities
  run_concordance_ksweep.py  concordance sensitivity to cluster count
  run_concordance_contig_null.py   S-1: contiguity-preserving concordance null
  run_intervis_vs_spatial.py run_intervis_bc.py   S-2: intervisibility vs matched null
  run_viewshed_coastal_null.py     S-3: distance-to-coast-matched viewshed null (DEM)
  make_figures.py make_viewshed_figure.py make_demic_structure_figure.py
  make_explainer_figures.py make_mataa_anatomy_figure.py make_moai_anatomy_figure.py
  make_moai_csv.py           extract ahu-placed moai coords/heights from the database
data/                 source data (see "Data" below), incl. committed viewshed/ matrix
output/               numeric results + committed intermediates the figures read
figures/              figures are written here
reproduce.sh          runs the whole pipeline (DEM-free by default)
requirements.txt / requirements-lock.txt   dependencies (loose / pinned)
Dockerfile  Makefile  pyproject.toml
```

## Reproduce in a container (recommended)

```
docker build -t hyperlocality-code .
docker run --rm \
  -v "$PWD/output:/work/output" -v "$PWD/figures:/work/figures" \
  hyperlocality-code
```

The image pins the exact dependency versions in `requirements-lock.txt` (Python 3.12,
PyMC 5.25, ArviZ 0.23, NumPy 1.26, SciPy 1.17). The numeric pipeline runs offline; the
main-figure step downloads a shaded-relief basemap, so allow network access for the
maps.

## Reproduce with a local Python environment

Python 3.12 (tested on 3.12.3). PyMC requires NumPy < 2.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt     # exact pinned versions used for the paper
# or: pip install -r requirements.txt     # looser version bounds
bash reproduce.sh
```

`reproduce.sh` runs the Bayesian analysis, the frequentist comparison, the
intervisibility and matched-null tests, and the figures. To run a single step:

```
PYTHONPATH=src python3 src/run_bayes.py
```

`make install | reproduce | lint | docker | clean` wraps the common tasks.

Each analysis uses a fixed seed, so results reproduce run to run: the Bayesian
analysis uses 20260623; the spatial and null analyses use 20260626–20260628; the
mata'a robustness battery uses 7. Bayes factors use sequential Monte Carlo (2,000
particles per chain); permutation tests use up to 9,999 permutations.

## Licensed DEM (not redistributable)

The monument-intervisibility analysis runs over a 50 cm Vricon V3D digital terrain
model (worked at 5 m), licensed to the authors and **not shared**. The scripts read it
through the `RAPANUI_DEM` environment variable. Because it is not redistributed, the
committed intermediates — `data/viewshed/ahu_viewshed.json` (the mutual-visibility
matrix), `output/viewshed_models.json` (the random-siting null), and
`output/viewshed_coastal_null.json` (the S-3 distance-matched null) — let everything
reproduce **without the DEM**. `reproduce.sh` recomputes the viewsheds only if
`RAPANUI_DEM` is set (and GDAL/`osgeo` is installed); otherwise it uses the committed
intermediates, and Figure 6's hillshade panel falls back to the committed PNG.

## Expected results

**Cultural F~ST~ (Bayesian posterior median [95% HDI]; Bayes factor on the
Kass–Raftery 2 ln BF scale), with the frequentist Nei G~ST~ shown for comparison:**

| Proxy | Posterior F~ST~ [95% HDI] | 2 ln BF | Nei G~ST~ |
|---|---|---|---|
| *mata'a* stem length × width | 0.028 [0.013, 0.047] | +19.4 | 0.053 |
| *mata'a* shape × shoulder | 0.058 [0.019, 0.113] | +12.6 | 0.060 |
| *mata'a* outline (scale-free control) | 0.003 [0.000, 0.018] | −10.2 | — |
| *umu* oven style | 0.073 [0.022, 0.160] | +29.0 | 0.082 |
| *moai* style (multilocus, k = 6) | 0.049 [0.020, 0.086] | +13.9 | 0.087 |
| *pukao* style | 0.047 [0.000, 0.147] | −5.3 | 0.185 |

Three proxies (*mata'a*, *umu*, *moai*) exclude zero with Bayes factors favoring
structure; the scale-free outline control and the small *pukao* sample favor panmixia.
*Mata'a* differentiation is **hyperlocal**, holding within a 5.5 km cluster
(stem-width F~ST~ = 0.068 [0.015, 0.152], 2 ln BF = +10.8).

**Intervisibility and matched nulls.** The *moai* are mostly intervisible only locally
(mean mutual-visibility degree 2.9 of 56). The visual/style concordance does **not**
exceed a spatially matched null (S-1: ARI 0.36 vs 0.54, P = 0.93; S-2: G~ST~ 0.099 vs
0.120, P = 0.83), so it reflects spatial autocorrelation, not a shared community map.
Against a distance-to-coast-matched null (S-3) only the network modularity remains
marginally elevated (0.73 vs 0.56, P ≈ 0.01); community count and size fall within the
null. *Moai* cross-linkage robustness: G~ST~ is 2.1×/2.0×/1.6× its null under
complete/average/ward linkage (single-linkage chains the coastal *ahu*). Named-territory
cross-check: posterior F~ST~ = 0.043 [0.016, 0.078], 2 ln BF = +9.3.

## Data

All data are public databases or counts/coordinates read from published figures and
tables. Copyrighted source PDFs are not redistributed.

- `output/published_lengthwidth.csv`, `output/published_shapeshoulder.csv` — the
  *mata'a* paradigmatic-class count matrices (validated against printed row totals;
  the committed source of truth for `tables_io.py`).
- `data/xyfinaldatawithids.csv`, `data/xyfinaldatawithids-smallassemblagesremoved.txt`
  — *mata'a* outline data (the `Source` field used by `check_source.py`; the outline
  used by the anatomy figure).
- `data/Figure3.svg` — the *mata'a* provenance map; coordinates are encoded in
  `src/spatial.py`.
- `data/umu/umu_pae_table1.csv` — McCoy's earth-oven rim-style tallies.
- `data/moai/MOAI_DATABASE_PUBLIC.xlsx`, `data/moai/moai_locations_heights.csv` — the
  public *moai* database and the extracted coordinates/heights.
- `data/pukao/Pukao.csv` — recorded *pukao*.
- `data/viewshed/ahu_viewshed.json`, `data/viewshed/ahu_communities.json` — the
  committed mutual-visibility matrix and the derived visual communities (so the
  intervisibility analysis reproduces without the licensed DEM).

### Known data limitations (stated honestly)

- *mata'a* and *umu* coordinates are read from published maps, not field GPS, so the
  spatial claims rest on jitter-robust, within-cluster tests rather than absolute
  distances.
- The *moai* and *pukao* style attributes come from existing databases and carry
  whatever inter-observer variation the original scoring introduced.
- Assemblages are time-averaged; the analysis describes between-community structure
  accumulated over each assemblage's deposition interval.

## Method in one paragraph

For each proxy we form a frequency matrix of classes by deme and estimate cultural
F~ST~ as a parameter in a hierarchical Balding–Nichols Dirichlet-multinomial model,
reporting the posterior median, its 95% HDI, and a structure-versus-panmixia Bayes
factor (marginal likelihoods by sequential Monte Carlo). The standard frequentist
statistics (Nei's multilocus G~ST~ against a panmixia permutation null, the Bell
variance-ratio estimator, bootstrap intervals) are computed alongside for comparison.
Locality is tested by restricting to a tight spatial cluster, by Mantel / partial
Mantel correlation, and by an individual-statue randomization test. Monument
intervisibility is read from a high-resolution terrain model; the visual communities
and the cross-proxy concordance are evaluated against spatially matched nulls. See
`src/bayes.py`, `src/popgen.py`, and `src/viewshed_models.py`.

## Citation

If you use this code or data, please cite the paper (Lipo, DiNapoli & Hunt) and the
model it tests:

> Lipo, C.P., DiNapoli, R.J., Madsen, M.E., Hunt, T.L. 2021. Population structure
> drives cultural diversity in finite populations: A hypothesis for localized
> community patterns on Rapa Nui (Easter Island). *PLOS ONE* 16(3): e0250690.

## License

MIT (see `LICENSE`).
