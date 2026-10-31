"""03_models.py - Leakage-safe survival modelling, ablation, external validation, calibration, DCA, learning curves.

Every preprocessing step (standardisation) lives inside an sklearn Pipeline, so it is re-fitted on the training
portion of every fold. Hyper-parameters are tuned by an inner CV nested inside each outer fold. External cohorts
(OncoSG, NCI Sherlock-Lung) are touched only once, by the final models refitted on all TCGA patients.
"""
import numpy as np, pandas as pd, json, pathlib, warnings, time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
warnings.filterwarnings("ignore")

R = pathlib.Path(__file__).resolve().parents[1]; O = R / "derived"; RES = R / "results"; RES.mkdir(exist_ok=True)
SEED = 20261020; T_EVAL = 36.0; TAU = 54.0; IBS_GRID = np.arange(6, 49, 3.0)
N_BOOT = 500

def prep(df):
    X = pd.DataFrame({"age": df.age, "female": df.female, "stage_II": (df.stage == 2).astype(float),
                      "stage_III": (df.stage == 3).astype(float), "stage_IV": (df.stage == 4).astype(float),
                      "ever_smoker": df.ever_smoker, "squamous": df.squamous,
                      "mech_P60": df.mech_P60, "mech_logTstar": df.mech_logTstar})
    return X, Surv.from_arrays(df.event.astype(bool), df.time)

FEATS = {
    "Stage only (TNM)": ["stage_II", "stage_III", "stage_IV"],
    "Mechanistic only": ["mech_P60", "mech_logTstar"],
    "Clinical": ["age", "female", "stage_II", "stage_III", "stage_IV", "ever_smoker", "squamous"],
    "Clinical + mechanistic": ["age", "female", "stage_II", "stage_III", "stage_IV", "ever_smoker", "squamous", "mech_P60", "mech_logTstar"],
}
def learners():
    return {
        "CoxPH": (make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis()), {"coxphsurvivalanalysis__alpha": [0.01, 0.1, 1.0, 10.0]}),
        "RSF": (make_pipeline(StandardScaler(), RandomSurvivalForest(n_estimators=300, random_state=SEED, n_jobs=-1)),
                {"randomsurvivalforest__min_samples_leaf": [10, 25, 50], "randomsurvivalforest__max_features": ["sqrt", None]}),
        "GBS": (make_pipeline(StandardScaler(), GradientBoostingSurvivalAnalysis(random_state=SEED)),
                {"gradientboostingsurvivalanalysis__n_estimators": [100, 300], "gradientboostingsurvivalanalysis__learning_rate": [0.03, 0.1],
                 "gradientboostingsurvivalanalysis__max_depth": [1, 2]}),
    }

def metrics(model, Xtr, ytr, Xte, yte):
    risk = model.predict(Xte)
    ev, tm = yte["event"], yte["time"]
    # IPCW estimators need test times strictly inside the training follow-up range (both are censored at 60 mo)
    yte = Surv.from_arrays(ev, np.minimum(tm, ytr["time"].max() - 0.01))
    out = {"harrell_c": concordance_index_censored(ev, tm, risk)[0]}
    tau = min(TAU, tm[ev].max() - 1e-6, ytr["time"].max() - 1e-6)
    try: out["uno_c"] = concordance_index_ipcw(ytr, yte, risk, tau=tau)[0]
    except Exception: out["uno_c"] = np.nan
    try: out["td_auc_36"] = cumulative_dynamic_auc(ytr, yte, risk, [T_EVAL])[0][0]
    except Exception: out["td_auc_36"] = np.nan
    try:
        g = IBS_GRID[(IBS_GRID < tm.max()) & (IBS_GRID > tm.min())]
        sf = model.predict_survival_function(Xte); S = np.vstack([f(g) for f in sf])
        out["ibs"] = integrated_brier_score(ytr, yte, S, g)
    except Exception: out["ibs"] = np.nan
    return out

def tune(est, grid, X, y, seed):
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    gs = GridSearchCV(est, grid, cv=list(cv.split(X, y["event"])), n_jobs=-1, error_score=np.nan)
    gs.fit(X, y); return gs.best_estimator_, gs.best_params_

if __name__ == "__main__":
    t0 = time.time()
    tc, on, nc = (pd.read_csv(O / f"{n}.csv") for n in ("tcga", "oncosg", "nci"))
    Xt, yt = prep(tc); Xo, yo = prep(on); Xn, yn = prep(nc)

    # ---------------- nested CV on TCGA ----------------
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows = []
    for k, (tr, te) in enumerate(outer.split(Xt, yt["event"])):
        for fs, cols in FEATS.items():
            for ln, (est, grid) in learners().items():
                if fs in ("Stage only (TNM)", "Mechanistic only") and ln != "CoxPH": continue
                m, bp = tune(est, grid, Xt.iloc[tr][cols], yt[tr], SEED + k)
                r = metrics(m, Xt.iloc[tr][cols], yt[tr], Xt.iloc[te][cols], yt[te]); r.update(fold=k, features=fs, learner=ln)
                rows.append(r)
        print(f"fold {k} done {time.time()-t0:.0f}s", flush=True)
    cvres = pd.DataFrame(rows); cvres.to_csv(RES / "nested_cv_folds.csv", index=False)
    cvsum = cvres.groupby(["features", "learner"])[["harrell_c", "uno_c", "td_auc_36", "ibs"]].agg(["mean", "std"]).round(3)
    cvsum.to_csv(RES / "nested_cv_summary.csv"); print(cvsum)

    # ---------------- final models on all TCGA -> external ----------------
    rng = np.random.default_rng(SEED); final = {}; ext_rows = []; boot = {}
    for fs, cols in FEATS.items():
        for ln, (est, grid) in learners().items():
            if fs in ("Stage only (TNM)", "Mechanistic only") and ln != "CoxPH": continue
            m, bp = tune(est, grid, Xt[cols], yt, SEED); final[(fs, ln)] = (m, cols, bp)
            for cname, X, y in (("OncoSG", Xo, yo), ("NCI_SherlockLung", Xn, yn)):
                r = metrics(m, Xt[cols], yt, X[cols], y); r.update(cohort=cname, features=fs, learner=ln, params=json.dumps(bp))
                risk = m.predict(X[cols]); ev, tm = y["event"], y["time"]; bs = []
                idx = [rng.integers(0, len(X), len(X)) for _ in range(N_BOOT)]
                for i in idx: bs.append(concordance_index_censored(ev[i], tm[i], risk[i])[0] if ev[i].sum() > 1 else np.nan)
                boot[(cname, fs, ln)] = np.array(bs)
                r["harrell_c_lo"], r["harrell_c_hi"] = np.nanpercentile(bs, [2.5, 97.5])
                ext_rows.append(r)
    ext = pd.DataFrame(ext_rows); ext.to_csv(RES / "external_validation.csv", index=False)
    print(ext[["cohort", "features", "learner", "harrell_c", "harrell_c_lo", "harrell_c_hi", "uno_c", "td_auc_36", "ibs"]].round(3).to_string())

    # paired bootstrap: Clinical+mech vs Clinical and vs Stage-only (same resamples via re-computation)
    paired = []
    for cname, X, y in (("OncoSG", Xo, yo), ("NCI_SherlockLung", Xn, yn)):
        ev, tm = y["event"], y["time"]; rr = np.random.default_rng(SEED + 1)
        idx = [rr.integers(0, len(X), len(X)) for _ in range(N_BOOT)]
        for ln in ("CoxPH", "RSF", "GBS"):
            for base_fs, base_ln in (("Clinical", ln), ("Stage only (TNM)", "CoxPH")):
                ma, ca, _ = final[("Clinical + mechanistic", ln)]; mb, cb, _ = final[(base_fs, base_ln)]
                ra, rb = ma.predict(X[ca]), mb.predict(X[cb])
                d = [concordance_index_censored(ev[i], tm[i], ra[i])[0] - concordance_index_censored(ev[i], tm[i], rb[i])[0] for i in idx if ev[i].sum() > 1]
                paired.append({"cohort": cname, "model": f"Clinical + mechanistic ({ln})", "comparator": f"{base_fs} ({base_ln})",
                               "delta_c": np.mean(d), "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5)})
    pd.DataFrame(paired).round(4).to_csv(RES / "paired_delta_c.csv", index=False); print(pd.DataFrame(paired).round(3).to_string())

    # save final models + data for figures/app
    import joblib; joblib.dump({f"{k[0]}|{k[1]}": v for k, v in final.items()}, RES / "final_models.joblib")
    print(f"total {time.time()-t0:.0f}s")
