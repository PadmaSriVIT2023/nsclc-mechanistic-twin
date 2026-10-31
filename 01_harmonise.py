"""01_harmonise.py - Build harmonised NSCLC cohorts from public cBioPortal datahub files.

Development: TCGA LUAD + LUSC (cBioPortal legacy studies luad_tcga, lusc_tcga).
External 1 : OncoSG LUAD (luad_oncosg_2020; Singapore, Chen et al. Nat Genet 2020).
External 2 : NCI Sherlock-Lung never-smokers (lung_nci_2022; Zhang et al. Nat Genet 2021/2022), NSCLC only.
Calibration: TRACERx (nsclc_tracerx_2017; Jamal-Hanjani et al. NEJM 2017) - measured tumour volume by T stage.

Harmonised predictors (available in every cohort): age, sex, AJCC stage group (I-IV), ever-smoker, squamous histology.
Endpoint: overall survival, administratively censored at 60 months.
"""
import numpy as np, pandas as pd, json, pathlib

D = pathlib.Path(__file__).resolve().parents[1] / "data"
O = pathlib.Path(__file__).resolve().parents[1] / "derived"; O.mkdir(exist_ok=True)
HORIZON = 60.0
flow = {}

def rd(study, f="data_clinical_patient.txt"):
    return pd.read_csv(D / study / f, sep="\t", comment="#", na_values=["[Not Available]", "[Discrepancy]", "[Unknown]", "[Not Applicable]"])

def stage_group(s):
    if pd.isna(s): return np.nan
    s = str(s).upper().replace("STAGE", "").strip()
    for g, v in (("IV", 4), ("III", 3), ("II", 2), ("I", 1)):
        if s.startswith(g): return v
    for g, v in (("4", 4), ("3", 3), ("2", 2), ("1", 1)):
        if s.startswith(g): return v
    return np.nan

def os_cols(df, m="OS_MONTHS", s="OS_STATUS"):
    t = pd.to_numeric(df[m], errors="coerce")
    e = df[s].astype(str).str.startswith("1").astype(float).where(df[s].notna())
    ev = (e == 1) & (t <= HORIZON)
    return np.minimum(t, HORIZON), ev.astype(float).where(e.notna())

def finish(df, name):
    n0 = len(df)
    df = df[df.time > 0]
    df = df.dropna(subset=["age", "female", "stage", "ever_smoker", "squamous", "time", "event"])
    flow[name]["after_complete_case"] = len(df)
    flow[name]["excluded_missing_or_zero_followup"] = n0 - len(df)
    flow[name]["events"] = int(df.event.sum())
    df["cohort"] = name
    return df

# ---------------- TCGA development ----------------
rows = []
for study, sq in (("luad_tcga", 0), ("lusc_tcga", 1)):
    p = rd(study); p = p[p.PATIENT_ID.str.startswith("TCGA")]
    flow_key = "TCGA"; flow.setdefault(flow_key, {}); flow[flow_key][f"{study}_patients"] = len(p)
    neo = p.HISTORY_NEOADJUVANT_TRTYN.astype(str).str.startswith("Yes")
    flow[flow_key][f"{study}_excluded_neoadjuvant"] = int(neo.sum()); p = p[~neo]
    t, e = os_cols(p)
    smk = pd.to_numeric(p.TOBACCO_SMOKING_HISTORY_INDICATOR, errors="coerce")  # 1 never; 2 current; 3,4,5 former
    rows.append(pd.DataFrame({"patient_id": p.PATIENT_ID, "age": pd.to_numeric(p.AGE, errors="coerce"),
        "female": (p.SEX == "Female").astype(float).where(p.SEX.notna()), "stage": p.AJCC_PATHOLOGIC_TUMOR_STAGE.map(stage_group),
        "t_stage": p.AJCC_TUMOR_PATHOLOGIC_PT.astype(str).str.extract(r"T(\d)")[0].astype(float),
        "n_stage": p.AJCC_NODES_PATHOLOGIC_PN.astype(str).str.extract(r"N(\d)")[0].astype(float),
        "ever_smoker": (smk > 1).astype(float).where(smk.notna()), "squamous": float(sq), "time": t, "event": e}))
tcga = pd.concat(rows, ignore_index=True)
dup = tcga.patient_id.duplicated(keep=False)
flow["TCGA"]["duplicate_patient_ids_removed"] = int(dup.sum()); tcga = tcga[~dup]
flow["TCGA"]["n_before_complete_case"] = len(tcga)
tcga = finish(tcga, "TCGA")

# ---------------- OncoSG external ----------------
o = rd("luad_oncosg_2020"); flow["OncoSG"] = {"patients": len(o)}
t, e = os_cols(o)
oncosg = finish(pd.DataFrame({"patient_id": o.PATIENT_ID, "age": o.AGE, "female": (o.SEX == "Female").astype(float),
    "stage": o.STAGE.map(stage_group), "t_stage": np.nan, "n_stage": np.nan,
    "ever_smoker": o.SMOKING_STATUS.map({"Yes": 1.0, "No": 0.0}), "squamous": 0.0, "time": t, "event": e}), "OncoSG")

# ---------------- NCI Sherlock-Lung external ----------------
n = rd("lung_nci_2022"); flow["NCI_SherlockLung"] = {"patients": len(n)}
carc = n.HISTOLOGY.isin(["Carcinoids"]); flow["NCI_SherlockLung"]["excluded_carcinoid"] = int(carc.sum()); n = n[~carc]
t, e = os_cols(n)
nci = finish(pd.DataFrame({"patient_id": n.PATIENT_ID, "age": n.AGE, "female": (n.SEX == "Female").astype(float),
    "stage": n.TUMOR_STAGE.map(stage_group), "t_stage": np.nan, "n_stage": np.nan, "ever_smoker": 0.0,
    "squamous": 0.0, "time": t, "event": e}), "NCI_SherlockLung")

# ---------------- overlap check (patient ID namespaces) ----------------
ids = {k: set(v.patient_id) for k, v in (("TCGA", tcga), ("OncoSG", oncosg), ("NCI", nci))}
flow["overlap"] = {f"{a}&{b}": len(ids[a] & ids[b]) for a in ids for b in ids if a < b}
flow["overlap_note"] = "No shared patient identifiers; Sherlock-Lung tissue source sites (Laval, Nice, EAGLE-NCI, Yale, Moffitt) and OncoSG (Singapore) are not TCGA sites."

# ---------------- TRACERx calibration data (stage -> measured volume) ----------------
tx = rd("nsclc_tracerx_2017")
tx = pd.DataFrame({"t_stage_full": tx.TUMOR_STAGE, "stage": tx.TUMOR_STAGE.map(stage_group), "volume_cm3": tx.TUMOR_VOLUME,
                   "squamous": tx.HISTOLOGY.str.contains("Squamous").astype(float)}).dropna()
tx.to_csv(O / "tracerx_volume.csv", index=False)

for df, nm in ((tcga, "tcga"), (oncosg, "oncosg"), (nci, "nci")): df.to_csv(O / f"{nm}.csv", index=False)
json.dump(flow, open(O / "cohort_flow.json", "w"), indent=2)
print(json.dumps(flow, indent=2))
for df in (tcga, oncosg, nci):
    print(df.cohort.iloc[0], len(df), "events", int(df.event.sum()), "stage", df.stage.value_counts().sort_index().to_dict(),
          "age", round(df.age.median(), 1), "female", round(df.female.mean(), 3), "ever", round(df.ever_smoker.mean(), 3), "sq", round(df.squamous.mean(), 3),
          "median FU", round(df.time.median(), 1))
