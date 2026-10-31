#!/usr/bin/env bash
# Downloads the public clinical files used in this study from the cBioPortal DataHub (GitHub LFS media mirror).
set -euo pipefail
cd "$(dirname "$0")"
B=https://media.githubusercontent.com/media/cBioPortal/datahub/master/public
for s in luad_tcga lusc_tcga luad_oncosg_2020 lung_nci_2022 nsclc_tracerx_2017; do
  mkdir -p data/$s
  for f in data_clinical_patient.txt data_clinical_sample.txt; do curl -fsSL -o data/$s/$f $B/$s/$f; done
done
sha256sum data/*/*.txt > data/SHA256SUMS.txt
echo "done"
