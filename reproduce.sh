#!/usr/bin/env bash
# Reproduce every quantitative result in the hyperlocality paper.
# Outputs go to output/; figures (png, pdf, svg) go to figures/.
#
# Usage:  bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

mkdir -p output figures
echo "== mata'a: cultural F_ST, within-cluster, isolation-by-distance =="
python3 src/run_hyperlocality.py | tee output/mataa_results.txt
echo "== mata'a: robustness battery (jitter, bootstrap, leave-one-out) =="
python3 src/harden.py            | tee output/mataa_harden.txt
echo "== umu =="
python3 src/run_umu.py           | tee output/umu_results.txt
echo "== moai: spatial-cluster F_ST =="
python3 src/run_moai.py          | tee output/moai_results.txt
echo "== moai: individual-statue spatial structure (clustering-free) =="
python3 src/run_moai_spatial.py  | tee output/moai_spatial.txt
echo "== pukao =="
python3 src/run_pukao.py         | tee output/pukao_results.txt
echo "== obsidian source counts (provenance check) =="
python3 src/check_source.py      | tee output/source_check.txt

echo "== figures (needs internet for the relief basemap tiles) =="
python3 src/make_figures.py || echo "  (figures skipped or partial; basemap tiles need internet)"

echo
echo "Done. Numeric results are in output/ and figures in figures/."
