"""10_extended_analyses.py - Additional analyses for the extended manuscript.

A. Unpenalised clinical Cox model: hazard ratios with 95% CIs (statsmodels PHReg) and proportional-hazards
   diagnostics (correlation of scaled Schoenfeld residuals with event-time rank; Grambsch-Therneau style).
B. External calibration slope (Cox regression on the frozen linear predictor) and observed/expected ratio at 36 mo.
C. Subgroup discrimination in the external cohorts (stage, sex, smoking) with bootstrap CIs.
D. Sensitivity analyses: homogeneous seeding (sigma = 0, re-calibrated m), VDT priors x0.7 / x1.3, 36-month horizon,
   exclusion of stage IV.
E. Independent check of the twin against TCGA disease-free survival by stage and histology (never used to fit the twin).
F. Sample-size adequacy (Riley et al. 2019 criterion for shrinkage >= 0.9) and events per parameter.
"""
import numpy as np, pandas as pd, json, pathlib, importlib.util, joblib, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
warnings.filterwarnings("ignore")
R = pathlib.Path(__file__).resolve().parents[1]; O = R / "derived"; RES = R / "results"; FIG = R / "figures"
def load(name, path):
    s = importlib.util.spec_from_file_location(name, R / path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m2 = load("m2", "code/02_mechanistic.py"); m3 = load("m3", "code/03_models.py")
SEED = 20261020; rng = np.random.default_rng(SEED); OUT = {}
data = {k: pd.read_csv(O / f"{f}.csv") for k, f in (("TCGA", "tcga"), ("OncoSG", "oncosg"), ("NCI_SherlockLung", "nci"))}
final = joblib.load(RES / "final_models.joblib")
prim, pcols, pparams = final["Clinical + mechanistic|CoxPH"]; ALPHA = pparams["coxphsurvivalanalysis__alpha"]
CLIN = m3.FEATS["Clinical"]; CT = m3.FEATS["Clinical + mechanistic"]; TW = m3.FEATS["Mechanistic only"]

# ---------------- A. unpenalised clinical Cox: HRs and PH diagnostics ----------------
X, y = m3.prep(data["TCGA"])
mod = PHReg(y["time"], X[CLIN].values, status=y["event"].astype(int), ties="breslow").fit()
ci = np.exp(mod.conf_int())
hr = pd.DataFrame({"feature": CLIN, "HR": np.exp(mod.params), "lo": ci[:, 0], "hi": ci[:, 1], "p": mod.pvalues})
sch = mod.schoenfeld_residuals                                  # rows = all obs; NaN for censored
ev = y["event"]; t_ev = y["time"][ev]; rk = stats.rankdata(t_ev)
ph = []
for j, f in enumerate(CLIN):
    r = sch[ev, j]; ok = ~np.isnan(r)
    rho, p = stats.pearsonr(rk[ok], r[ok]) if r[ok].std() > 0 else (np.nan, np.nan); ph.append((f, rho, p))
hr["ph_rho"] = [p[1] for p in ph]; hr["ph_p"] = [p[2] for p in ph]
hr.round(4).to_csv(RES / "clinical_cox_hr_ph.csv", index=False); OUT["A_hr"] = hr.round(4).to_dict("records")

# likelihood-ratio statistic of the clinical model (null log partial likelihood computed directly)
def null_llf(t, e):
    order = np.argsort(t); t, e = t[order], e[order]; ll = 0.0
    for u in np.unique(t[e]):
        d = ((t == u) & e).sum(); ll -= d * np.log((t >= u).sum())
    return ll
LR = 2 * (mod.llf - null_llf(y["time"], y["event"])); n = len(y); R2cs = 1 - np.exp(-LR / n)
OUT["F_samplesize"] = {"LR_chi2_clinical": float(LR), "R2_CS_clinical": float(R2cs)}
for p_par, lbl in ((len(CLIN), "clinical_7"), (len(CT), "clinical_twin_9")):
    S = 0.9; n_req = p_par / ((S - 1) * np.log(1 - R2cs / S))
    OUT["F_samplesize"][lbl] = {"parameters": p_par, "n_required_shrinkage_0.9": float(n_req), "n_available": n,
                                "events": int(y["event"].sum()), "EPP": float(y["event"].sum() / p_par),
                                "expected_shrinkage_vanHouwelingen": float(1 - p_par / LR)}

# ---------------- B. external calibration slope and O/E ----------------
cal = []
for c in ("OncoSG", "NCI_SherlockLung"):
    Xe, ye = m3.prep(data[c]); lp = prim.predict(Xe[pcols])
    f = PHReg(ye["time"], lp[:, None], status=ye["event"].astype(int), ties="breslow").fit()
    s36 = np.array([fn(36.0) for fn in prim.predict_survival_function(Xe[pcols])])
    tt, ss = kaplan_meier_estimator(ye["event"], ye["time"]); obs = 1 - ss[tt <= 36][-1]; exp_ = np.mean(1 - s36)
    cal.append({"cohort": c, "slope": float(f.params[0]), "slope_lo": float(f.conf_int()[0, 0]), "slope_hi": float(f.conf_int()[0, 1]),
                "observed_36": float(obs), "expected_36": float(exp_), "OE": float(obs / exp_)})
pd.DataFrame(cal).round(4).to_csv(RES / "external_calibration_slope.csv", index=False); OUT["B_cal"] = cal

# ---------------- C. subgroups ----------------
def cboot(e, t, r, B=500):
    c = concordance_index_censored(e, t, r)[0]; bs = []
    for _ in range(B):
        i = rng.integers(0, len(t), len(t))
        if e[i].sum() > 1 and len(np.unique(r[i])) > 1: bs.append(concordance_index_censored(e[i], t[i], r[i])[0])
    return c, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
sub = []
for c in ("OncoSG", "NCI_SherlockLung"):
    d = data[c]; Xe, ye = m3.prep(d); lp = prim.predict(Xe[pcols])
    groups = {"All": np.ones(len(d), bool), "Stage I": d.stage.values == 1, "Stage II-IV": d.stage.values >= 2,
              "Female": d.female.values == 1, "Male": d.female.values == 0}
    if c == "OncoSG": groups.update({"Ever-smoker": d.ever_smoker.values == 1, "Never-smoker": d.ever_smoker.values == 0})
    for g, msk in groups.items():
        e, t = ye["event"][msk], ye["time"][msk]
        if e.sum() < 5: continue
        cc, lo, hi = cboot(e, t, lp[msk]); sub.append({"cohort": c, "subgroup": g, "n": int(msk.sum()), "events": int(e.sum()), "C": cc, "lo": lo, "hi": hi})
pd.DataFrame(sub).round(4).to_csv(RES / "subgroup_cindex.csv", index=False); OUT["C_sub"] = sub

# ---------------- D. sensitivity analyses ----------------
tp = json.load(open(O / "twin_params.json")); vols = {float(k): v for k, v in tp["vols"].items()}
def twin_features(m, sg, vdt_mult=1.0):
    p = m2.draw(400, np.random.default_rng(tp["seed"])); p["VDT_a"] = p["VDT_a"] * vdt_mult; p["VDT_s"] = p["VDT_s"] * vdt_mult; cache = {}
    for st in (1.0, 2.0, 3.0, 4.0):
        for sq in (0.0, 1.0):
            Mb, E = m2.curves(vols[st], sq, p); cache[(st, sq)] = np.minimum(m2.tstar(Mb, E, st, p, m, sg), m2.CAP)
    out = {}
    for k, d in data.items():
        dd = d.copy(); dd["mech_P60"] = [np.mean(cache[(r.stage, r.squamous)] <= 60) for r in dd.itertuples()]
        dd["mech_logTstar"] = [np.log(np.median(cache[(r.stage, r.squamous)]) + 1.0) for r in dd.itertuples()]; out[k] = dd
    return out
def evaluate(dsets, label, horizon=60.0, drop_iv=False):
    rows = []
    for fs, cols in (("Twin only", TW), ("Clinical + twin", CT), ("Clinical", CLIN)):
        def prep(d):
            d = d.copy()
            if drop_iv: d = d[d.stage < 4]
            d["event"] = ((d.event == 1) & (d.time <= horizon)).astype(float); d["time"] = np.minimum(d.time, horizon); return m3.prep(d)
        Xt, yt = prep(dsets["TCGA"]); mdl = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=ALPHA)).fit(Xt[cols], yt)
        for c in ("OncoSG", "NCI_SherlockLung"):
            Xe, ye = prep(dsets[c]); mt = m3.metrics(mdl, Xt[cols], yt, Xe[cols], ye)
            rows.append({"analysis": label, "features": fs, "cohort": c, "C": mt["harrell_c"], "IBS": mt["ibs"]})
    return rows
sens = evaluate(data, "Primary analysis")
# sigma = 0: recalibrate m only, on TRACERx (same objective)
tx = m2.load_tracerx(); target = m2.km_at(tx.event.values.astype(bool), tx.time.values, [12, 24])
pc = m2.draw(m2.N_CAL_DRAWS, np.random.default_rng(m2.SEED)); Cc = [m2.curves(r.volume_cm3, r.squamous, pc) for r in tx.itertuples()]
best = (None, np.inf)
for mm in np.arange(-16, -6.0, 0.1):
    T = np.array([m2.tstar(c[0], c[1], st, pc, mm, 0.0) for c, st in zip(Cc, tx.stage.values)])
    S12, S24, R60 = (T > 12).mean(), (T > 24).mean(), (T <= 60).mean()
    L = (S12 - target[0]) ** 2 + (S24 - target[1]) ** 2 + max(0, .30 - R60) ** 2 + max(0, R60 - .55) ** 2
    if L < best[1]: best = (mm, L, S12, S24, R60)
OUT["D_sigma0_calibration"] = {"m": float(best[0]), "loss": float(best[1]), "S12": float(best[2]), "S24": float(best[3]), "R60": float(best[4]),
                               "primary_loss": float(json.load(open(O / "mechanistic_summary.json"))["calibration_loss"])}
sens += evaluate(twin_features(best[0], 0.0), "Homogeneous seeding (sigma = 0)")
sens += evaluate(twin_features(tp["m"], tp["sigma"], 0.7), "VDT prior x0.7 (no recalibration)")
sens += evaluate(twin_features(tp["m"], tp["sigma"], 1.3), "VDT prior x1.3 (no recalibration)")
sens += evaluate(data, "36-month horizon", horizon=36.0)
sens += evaluate(data, "Stage IV excluded", drop_iv=True)
sd = pd.DataFrame(sens); sd.round(4).to_csv(RES / "sensitivity_analyses.csv", index=False); OUT["D_sens"] = sd.round(4).to_dict("records")

# ---------------- E. twin vs TCGA disease-free survival ----------------
raw = []
for s, sq in (("luad_tcga", 0.0), ("lusc_tcga", 1.0)):
    p = pd.read_csv(R / f"data/{s}/data_clinical_patient.txt", sep="\t", comment="#", na_values=["[Not Available]", "[Discrepancy]"])
    raw.append(pd.DataFrame({"patient_id": p.PATIENT_ID, "dfs_t": pd.to_numeric(p.DFS_MONTHS, errors="coerce"),
                             "dfs_e": p.DFS_STATUS.astype(str).str.startswith("1")}).assign(squamous=sq))
dfs = data["TCGA"][["patient_id", "stage", "squamous"]].merge(pd.concat(raw), on=["patient_id", "squamous"]).dropna()
dfs = dfs[(dfs.dfs_t > 0) & (dfs.stage < 4)]
cache = np.load(O / "tstar_cache.npy", allow_pickle=True).item()
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2)); comp = []
for ax, st in zip(axs, (1.0, 2.0, 3.0)):
    for sq, col, lab in ((0.0, "#2f6db3", "Adenocarcinoma"), (1.0, "#b91c1c", "Squamous")):
        d = dfs[(dfs.stage == st) & (dfs.squamous == sq)]
        tt, ss = kaplan_meier_estimator(d.dfs_e.values, d.dfs_t.values)
        ax.step(tt, ss, where="post", c=col, lw=1.2, label=f"TCGA DFS, {lab} (n={len(d)})")
        tw = cache[f"{st}_{sq}"]; g = np.arange(0, 121); ax.plot(g, [np.mean(tw > x) for x in g], ls="--", c=col, lw=1.4, label=f"Twin, {lab}")
        km = lambda q: float(ss[tt <= q][-1]) if (tt <= q).any() else 1.0
        comp.append({"stage": int(st), "histology": lab, "n": len(d), "events": int(d.dfs_e.sum()),
                     "KM_DFS_24": km(24), "twin_RFS_24": float(np.mean(tw > 24)), "KM_DFS_60": km(60), "twin_RFS_60": float(np.mean(tw > 60))})
    ax.set_xlim(0, 120); ax.set_ylim(0, 1.02); ax.set_title(f"Stage {'I' * int(st)}"); ax.set_xlabel("Months"); ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Relapse-free probability"); axs[0].legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(FIG / "figS4_twin_vs_tcga_dfs.png", dpi=300); plt.close()
cd = pd.DataFrame(comp); cd.round(4).to_csv(RES / "twin_vs_tcga_dfs.csv", index=False); OUT["E_dfs"] = cd.round(4).to_dict("records")
OUT["E_dfs_mean_abs_diff_24"] = float(np.mean(np.abs(cd.KM_DFS_24 - cd.twin_RFS_24))); OUT["E_dfs_mean_abs_diff_60"] = float(np.mean(np.abs(cd.KM_DFS_60 - cd.twin_RFS_60)))
# discrimination of twin P60 for TCGA DFS (independent outcome)
r = dfs.merge(data["TCGA"][["patient_id", "mech_P60"]], on="patient_id")
OUT["E_dfs_cindex_twin"] = float(concordance_index_censored(r.dfs_e.values, r.dfs_t.values, r.mech_P60.values)[0])
OUT["E_dfs_cindex_stage"] = float(concordance_index_censored(r.dfs_e.values, r.dfs_t.values, r.stage.values)[0])
OUT["E_dfs_n"] = int(len(r)); OUT["E_dfs_events"] = int(r.dfs_e.sum())

# ---------------- H. analytic cure probability check ----------------
# P(no seeding) = E_mu[exp(-mu * Lambda)] estimated two ways for stage I adenocarcinoma
p = m2.draw(400, np.random.default_rng(tp["seed"])); Mb, E = m2.curves(vols[1.0], 0.0, p); mu = 10 ** (tp["m"] + tp["sigma"] * p["z_mu"])
OUT["H_cure_stageI_adeno"] = {"analytic_mean_exp": float(np.mean(np.exp(-mu * E))), "simulated_n0": float(np.mean(m2.poisson.ppf(p["u"], mu * E) == 0)),
                              "median_Lambda_per_mu": float(np.median(E))}
json.dump(OUT, open(RES / "extended_analyses.json", "w"), indent=1, default=float)
print(hr.round(3).to_string()); print(pd.DataFrame(cal).round(3).to_string()); print(pd.DataFrame(sub).round(3).to_string())
print(sd.pivot_table(index="analysis", columns=["cohort", "features"], values="C").round(3).to_string()); print(cd.round(3).to_string())
print({k: v for k, v in OUT.items() if k in ("F_samplesize", "D_sigma0_calibration", "E_dfs_mean_abs_diff_24", "E_dfs_mean_abs_diff_60", "E_dfs_cindex_twin", "E_dfs_cindex_stage", "E_dfs_n", "E_dfs_events", "H_cure_stageI_adeno")})
