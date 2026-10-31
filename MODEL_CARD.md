# Model card: NSCLC clinical + mechanistic-twin Cox model

**Model.** Ridge-penalised Cox proportional hazards (scikit-survival), 9 standardised inputs: age, female, stage II/III/IV indicators, ever-smoker, squamous histology, twin P60, twin log(1+median T*). Coefficients and scaling in `tool/model_export.json`.

**Intended use.** Research and education on hybrid mechanistic–ML survival modelling in resected NSCLC. **Not** for individual clinical decisions; not prospectively validated; not a regulated medical device.

**Training data.** TCGA LUAD + LUSC, 945 complete cases, 337 deaths; overall survival censored at 60 months; 3-year OS 59.5%.

**Evaluation.** Nested CV (TCGA) Harrell's C 0.619 ± 0.032. External: OncoSG C 0.707, IBS 0.208; NCI Sherlock-Lung C 0.663, IBS 0.217.

**Calibration.** As developed, 36-month mortality was overestimated externally (44.5% vs 25.7% observed in OncoSG; 40.9% vs 19.5% in Sherlock-Lung). Re-estimating the baseline hazard locally corrected mean risk (27.0%, 19.0%). The tool offers these re-estimated baselines.

**Mechanistic twin.** Gompertzian growth + Poisson metastatic seeding (Iwata/Benzekry). VDT from a CT meta-analysis; stage volumes from TRACERx; seeding (m=-11.1, σ=2.0) calibrated on TRACERx recurrence only. Outputs are in-silico model quantities, not observations.

**Limitations.** Squamous histology not externally validated; small external event counts; twin cannot separate stage II from III; treatment not modelled; OS includes non-cancer deaths; populations differ in ancestry and smoking.

**Fairness considerations.** Performance reported separately for a Singaporean LUAD cohort and a never-smoker cohort; no cohort with African or South Asian ancestry was available for validation.
