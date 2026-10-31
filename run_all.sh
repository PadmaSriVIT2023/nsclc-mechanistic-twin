#!/usr/bin/env bash
# Reproduces every number and figure from the public data files in data/.
set -euo pipefail
cd "$(dirname "$0")"
for s in 01_harmonise 02_mechanistic 03_models 04_figures 05_twin_sensitivity 06_table1 07_fig1_design 08_export_tool 09_supp_figs 10_extended_analyses 11_numerics; do
  echo "=== $s $(date +%T)"; python3 code/$s.py
done
echo "=== ALL DONE $(date +%T)"
