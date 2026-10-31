"""07_fig1_design.py - Study design and cohort flow (numbers read from derived/cohort_flow.json)."""
import json, pathlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R = pathlib.Path(__file__).resolve().parents[1]; F = json.load(open(R / "derived/cohort_flow.json")); FIG = R / "figures"
mech = json.load(open(R / "derived/mechanistic_summary.json"))
t = F["TCGA"]; o = F["OncoSG"]; n = F["NCI_SherlockLung"]
fig, ax = plt.subplots(figsize=(10, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
def box(x, y, w, h, txt, fc="#eef2ff", ec="#3730a3", fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)
def arr(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1))
# column 1: data
box(1, 47, 22, 10, f"TCGA LUAD + LUSC\n{t['luad_tcga_patients']} + {t['lusc_tcga_patients']} patients\n−{t['luad_tcga_excluded_neoadjuvant'] + t['lusc_tcga_excluded_neoadjuvant']} neoadjuvant, −{t['excluded_missing_or_zero_followup']} incomplete\nDevelopment n = {t['after_complete_case']} ({t['events']} deaths)", fc="#dbeafe")
box(1, 33, 22, 10, f"OncoSG (Singapore, LUAD)\n{o['patients']} patients, −{o['excluded_missing_or_zero_followup']} incomplete\nExternal 1 n = {o['after_complete_case']} ({o['events']} deaths)", fc="#fef3c7", ec="#92400e")
box(1, 19, 22, 10, f"NCI Sherlock-Lung (never-smokers)\n{n['patients']} patients, −{n['excluded_carcinoid']} carcinoid, −{n['excluded_missing_or_zero_followup']} incomplete\nExternal 2 n = {n['after_complete_case']} ({n['events']} deaths)", fc="#fef3c7", ec="#92400e")
box(1, 5, 22, 10, f"TRACERx (measured CT volumes)\nn = {mech['tracerx_check']['n']} ({mech['tracerx_check']['events']} relapses)\nUsed ONLY for twin calibration", fc="#dcfce7", ec="#166534")
# column 2: harmonisation + twin
box(29, 40, 24, 14, "Harmonised predictors\nage, sex, AJCC stage group,\never-smoking, squamous histology\nEndpoint: overall survival\n(censored at 60 months)", fc="#f3f4f6", ec="#374151")
box(29, 8, 24, 24, "Mechanistic digital twin\nGompertz primary growth\n+ Poisson metastatic seeding\n(Iwata/Benzekry)\nVDT: CT meta-analysis\nV0: TRACERx volumes\n(m, σ) from TRACERx + literature\n→ twin features P60, log T*\n(no TCGA/external outcomes used)", fc="#dcfce7", ec="#166534")
# column 3: modelling
box(59, 36, 18, 18, "Nested CV on TCGA\n5-fold × 3 repeats\ninner 3-fold tuning\nscaling inside folds\nCoxPH · RSF · GBS", fc="#ede9fe", ec="#5b21b6")
box(59, 12, 18, 18, "Ablation\nTNM stage only\nTwin only\nClinical\nClinical + twin", fc="#ede9fe", ec="#5b21b6")
# column 4: evaluation
box(82, 36, 17, 18, "External validation\n(models frozen)\nHarrell / Uno C, td-AUC,\nIBS, calibration,\ndecision curves", fc="#fee2e2", ec="#991b1b")
box(82, 12, 17, 18, "Open research tool\nmodel card,\nTRIPOD+AI,\nreproducible code\n(research use only)", fc="#fee2e2", ec="#991b1b")
for y in (52, 38, 24): arr(23.5, y, 28.5, 47)
arr(23.5, 10, 28.5, 18); arr(41, 39.5, 41, 32.5); arr(53.5, 47, 58.5, 45); arr(53.5, 20, 58.5, 21); arr(68, 35.5, 68, 30.5)
arr(77.5, 45, 81.5, 45); arr(77.5, 21, 81.5, 21)
fig.tight_layout(); fig.savefig(FIG / "fig1_study_design.png", dpi=300); print("ok")
