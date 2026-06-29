#!/usr/bin/env bash
# Reproduce every quantitative result and figure in the hyperlocality paper.
# Numeric results go to output/; figures (png, pdf, svg) go to figures/.
#
# Default run is DEM-free: the licensed 50 cm terrain model is not redistributed,
# so the intervisibility steps reproduce from the committed intermediates in
# data/viewshed/ and output/. To recompute the viewsheds from your own copy of
# the DEM, set RAPANUI_DEM=/path/to/dem.tif (also needs GDAL/osgeo).
#
# Usage:  bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
mkdir -p output figures

echo "== Primary analysis: Bayesian cultural F_ST + structure-vs-panmixia Bayes factors =="
python3 src/run_bayes.py

echo "== Frequentist comparison (Nei G_ST, Bell estimator, permutation tests) =="
python3 src/run_hyperlocality.py | tee output/mataa_results.txt
python3 src/harden.py            | tee output/mataa_harden.txt
python3 src/run_umu.py           | tee output/umu_results.txt
python3 src/run_moai.py          | tee output/moai_results.txt
python3 src/run_moai_spatial.py  | tee output/moai_spatial.txt
python3 src/run_pukao.py         | tee output/pukao_results.txt
python3 src/check_source.py      | tee output/source_check.txt

echo "== Moai robustness: cross-linkage sweep + Bayesian named-territory cross-check =="
python3 src/run_moai_linkage_sweep.py
python3 src/run_moai_territory_bayes.py

echo "== Intervisibility (DEM-free: from committed data/viewshed intermediates) =="
python3 src/make_communities.py            # visual communities from the viewshed matrix
python3 src/run_concordance.py             # moai style across the visual communities
python3 src/run_concordance_ksweep.py      # concordance sensitivity to cluster count
python3 src/run_concordance_contig_null.py # S-1: contiguity-preserving concordance null
python3 src/run_intervis_vs_spatial.py     # S-2: intervisibility vs contiguity-matched null
python3 src/run_intervis_bc.py             # S-2: bias-corrected variant

echo "== DEM-dependent steps (viewshed recompute) =="
if [ -n "${RAPANUI_DEM:-}" ]; then
  python3 src/ahu_viewshed.py
  python3 src/viewshed_models.py
  python3 src/run_viewshed_coastal_null.py # S-3: distance-to-coast-matched viewshed null
else
  echo "  RAPANUI_DEM not set; using committed data/viewshed/ + output/viewshed_models.json"
  echo "  + output/viewshed_coastal_null.json (S-3)."
fi

echo "== Figures =="
python3 src/make_figures.py || echo "  (main figures need internet for the relief basemap tiles)"
python3 src/make_demic_structure_figure.py   # Fig S3
python3 src/make_explainer_figures.py        # style-variability + intervisibility explainers
python3 src/make_mataa_anatomy_figure.py     # Fig S4
python3 src/make_moai_anatomy_figure.py      # Fig S6
if [ -n "${RAPANUI_DEM:-}" ]; then
  python3 src/make_viewshed_figure.py        # Fig 6 hillshade (needs the DEM)
else
  echo "  Fig 6 hillshade needs RAPANUI_DEM; the committed figures/fig6_intervisibility.png is used."
fi

echo
echo "Done. Numeric results are in output/ and figures in figures/."
