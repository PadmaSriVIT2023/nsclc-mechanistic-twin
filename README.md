# NSCLC mechanistic metastatic-seeding twin + survival ML

Code, derived data and research tool for:

> Sridharan P, Ghosh M. *A mechanistic metastatic-seeding digital twin combined with survival machine learning for resected non-small cell lung cancer: development in TCGA and external validation in two independent cohorts.* (submitted)

## What this repository does
| Step | Script | Output |
|---|---|---|
| Download public clinical files (cBioPortal DataHub) | `fetch_data.sh` | `data/` (+ SHA256SUMS) |
| Harmonise cohorts | `code/01_harmonise.py` | `derived/*.csv`, `derived/cohort_flow.json` |
| Mechanistic twin: calibration on TRACERx + features | `code/02_mechanistic.py` | `derived/mechanistic_summary.json`, twin features |
| Nested CV, final models, external validation, paired bootstrap | `code/03_models.py` | `results/*.csv` |
| Calibration, decision curves, risk groups, learning curves, SHAP | `code/04_figures.py` | `figures/`, `results/figure_numbers.json` |
| Sobol sensitivity, twin trajectories | `code/05_twin_sensitivity.py` | `results/sobol_indices.csv`, figures |
| Table 1, design figure, tool export, supplementary figure | `code/06`–`09` | `results/`, `figures/`, `tool/model_export.json` |
| Hazard ratios, PH diagnostics, calibration slope, subgroups, sensitivity, twin vs TCGA DFS, sample size | `code/10_extended_analyses.py` | `results/extended_analyses.json`, `figures/figS4_*` |
| Twin quadrature / Monte Carlo convergence | `code/11_numerics.py` | `results/twin_numerics.json` |
| Manuscript and supplement (numbers filled from outputs) | `manuscript/build.py` | `manuscript/out/*.pdf/.docx/.tex` |

```bash
./fetch_data.sh      # ~3 MB of public clinical files
./run_all.sh         # ~20 min on 4 cores; fixed seed 20261020
cd manuscript && python3 build.py && python3 make_docs.py
```
or `docker build -t nsclc-twin . && docker run --rm nsclc-twin`.

## Cohorts
| Role | Study (cBioPortal id) | Analysed n (deaths) |
|---|---|---|
| Development | TCGA LUAD + LUSC (`luad_tcga`, `lusc_tcga`) | 945 (337) |
| External 1 | OncoSG (`luad_oncosg_2020`) | 296 (88) |
| External 2 | NCI Sherlock-Lung (`lung_nci_2022`) | 169 (42) |
| Twin calibration only | TRACERx (`nsclc_tracerx_2017`) | 95 (24 relapses) |

## Key results
External Harrell's C of the clinical + twin Cox model: OncoSG 0.707 (0.649–0.763), Sherlock-Lung 0.663 (0.588–0.749). Twin features added no discrimination over clinical variables (paired ΔC -0.002 (-0.007 to +0.002) and +0.000 (-0.002 to +0.003)). Absolute risks require local baseline recalibration.

## Leakage safeguards
* Standardisation inside sklearn pipelines; hyper-parameters tuned by inner CV nested in outer 5×3 CV.
* External cohorts evaluated once, by models frozen after TCGA development.
* Twin parameters come from literature and TRACERx only; no TCGA/OncoSG/Sherlock-Lung outcome enters the twin.

## Research tool
`tool/nsclc_twin_tool.html` is a self-contained page reproducing the frozen primary model exactly (checked against scikit-survival in `code/08_export_tool.py`). **Research use only; not a medical device.** See `MODEL_CARD.md`.

## Licence and data terms
Code: MIT. Source data remain under the terms of the original studies and cBioPortal; this repository redistributes only derived, de-identified analysis tables.
