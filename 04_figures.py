"""04_figures.py - Calibration, decision-curve analysis, risk-group KM, learning curves, SHAP, Twin sensitivity.
Uses the final models saved by 03_models.py; every number printed here is written to results/figure_numbers.json."""
import numpy as np, pandas as pd, json, pathlib, joblib, warnings
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import importlib.util, sys
warnings.filterwarnings("ignore")
R = pathlib.Path(__file__).resolve().parents[1]; O = R / "derived"; RES = R / "results"; FIG = R / "figures"; FIG.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("m3", R / "code/03_models.py"); m3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m3)
SEED = 20261020; T = 36.0
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
COL = {"Stage only (TNM)": "#8c8c8c", "Clinical": "#2f6db3", "Clinical + mechanistic": "#c2410c", "Mechanistic only": "#6b8e23"}
nums = {}

final = joblib.load(RES / "final_models.joblib")
data = {n: pd.read_csv(O / f"{f}.csv") for n, f in (("TCGA", "tcga"), ("OncoSG", "oncosg"), ("NCI_SherlockLung", "nci"))}
XY = {k: m3.prep(v) for k, v in data.items()}
ext = pd.read_csv(RES / "external_validation.csv")
best_ln = ext[ext.features == "Clinical + mechanistic"].groupby("learner").harrell_c.mean().idxmax()
nums["primary_learner"] = best_ln

def surv_at(model, X, t):
    return np.array([f(t) for f in model.predict_survival_function(X)])

def recal_surv_at(model, X, y, t):
    """Recalibration-in-the-large: keep the TCGA-derived linear predictor, re-estimate the Breslow baseline hazard
    in the external cohort. Discrimination is unchanged by construction; only absolute risk is updated."""
    lp = model.predict(X); ev, tm = y["event"], y["time"]; H = 0.0
    for u in np.unique(tm[ev & (tm <= t)]):
        H += ((tm == u) & ev).sum() / np.exp(lp[tm >= u]).sum()
    return np.exp(-H * np.exp(lp))

# ---------- Fig 2: calibration at 36 months (quintiles) ----------
fig, axs = plt.subplots(1, 2, figsize=(7, 3.3))
for ax, c in zip(axs, ("OncoSG", "NCI_SherlockLung")):
    X, y = XY[c]
    for fs, recal in (("Clinical", False), ("Clinical + mechanistic", False), ("Clinical + mechanistic", True)):
        m, cols, _ = final[f"{fs}|{best_ln}"]
        pr = 1 - (recal_surv_at(m, X[cols], y, T) if recal else surv_at(m, X[cols], T)); q = pd.qcut(pr, 5, labels=False, duplicates="drop")
        obs, pred = [], []
        for g in np.unique(q):
            i = q == g; tt, ss = kaplan_meier_estimator(y["event"][i], y["time"][i])
            obs.append(1 - (ss[tt <= T][-1] if (tt <= T).any() else 1)); pred.append(pr[i].mean())
        lab = fs + (" (baseline re-estimated)" if recal else " (as developed)")
        ax.plot(pred, obs, "o--" if recal else "o-", color="#7c3aed" if recal else COL[fs], label=lab, ms=4)
        nums[f"calib_{c}_{lab}_mean_abs_err"] = float(np.mean(np.abs(np.array(obs) - np.array(pred))))
        nums[f"calib_{c}_{lab}_mean_pred_vs_obs"] = [float(np.mean(pr)), float(1 - kaplan_meier_estimator(y["event"], y["time"])[1][kaplan_meier_estimator(y["event"], y["time"])[0] <= T][-1])]
    ax.plot([0, 1], [0, 1], ":", c="k", lw=.8); ax.set_xlim(0, .8); ax.set_ylim(0, .8)
    ax.set_title(c.replace("_", " ")); ax.set_xlabel("Predicted 36-month mortality"); ax.set_ylabel("Observed (Kaplan-Meier)")
axs[0].legend(frameon=False, fontsize=6, loc="upper left"); fig.tight_layout(); fig.savefig(FIG / "fig2_calibration.png"); plt.close()

# ---------- Fig 3: decision curve at 36 months ----------
def net_benefit(pr, y, thr):
    n = len(pr); out = []
    for pt in thr:
        sel = pr >= pt
        if sel.sum() == 0: out.append(0.0); continue
        tt, ss = kaplan_meier_estimator(y["event"][sel], y["time"][sel])
        risk = 1 - (ss[tt <= T][-1] if (tt <= T).any() else 1)
        tp = risk * sel.mean(); fp = (1 - risk) * sel.mean()
        out.append(tp - fp * pt / (1 - pt))
    return np.array(out)
thr = np.arange(0.05, 0.61, 0.01)
fig, axs = plt.subplots(1, 2, figsize=(7, 3.3))
for ax, c in zip(axs, ("OncoSG", "NCI_SherlockLung")):
    X, y = XY[c]; tt, ss = kaplan_meier_estimator(y["event"], y["time"]); prev = 1 - ss[tt <= T][-1]
    ax.plot(thr, prev - (1 - prev) * thr / (1 - thr), c="#bbbbbb", lw=1, label="Treat all"); ax.axhline(0, c="k", lw=.8, label="Treat none")
    for fs in ("Stage only (TNM)", "Clinical", "Clinical + mechanistic"):
        ln = "CoxPH"; m, cols, _ = final[f"{fs}|{ln}"]
        nb = net_benefit(1 - recal_surv_at(m, X[cols], y, T), y, thr); ax.plot(thr, nb, c=COL[fs], label=fs + " (recalibrated)")
        nums[f"dca_{c}_{fs}_NB"] = {f"{t:.2f}": float(nb[np.argmin(abs(thr - t))]) for t in (0.10, 0.20, 0.30, 0.40)}
    nums[f"dca_{c}_treat_all_NB"] = {f"{t:.2f}": float(prev - (1 - prev) * t / (1 - t)) for t in (0.10, 0.20, 0.30, 0.40)}
    ax.set_ylim(-0.05, max(prev, .05) + .05); ax.set_title(c.replace("_", " ")); ax.set_xlabel("Threshold probability (36-month death)"); ax.set_ylabel("Net benefit")
axs[0].legend(frameon=False, fontsize=6.5); fig.tight_layout(); fig.savefig(FIG / "fig3_decision_curve.png"); plt.close()

# ---------- Fig 4: risk groups (TCGA tertile cut-points applied unchanged to externals) ----------
m, cols, _ = final[f"Clinical + mechanistic|{best_ln}"]
cuts = np.quantile(m.predict(XY["TCGA"][0][cols]), [1 / 3, 2 / 3]); nums["risk_cutpoints_TCGA_tertiles"] = cuts.tolist()
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for ax, c in zip(axs, ("TCGA", "OncoSG", "NCI_SherlockLung")):
    X, y = XY[c]; g = np.digitize(m.predict(X[cols]), cuts)
    for k, lab, colr in ((0, "Low", "#2f6db3"), (1, "Intermediate", "#d4a017"), (2, "High", "#b91c1c")):
        if (g == k).sum() < 2: continue
        tt, ss = kaplan_meier_estimator(y["event"][g == k], y["time"][g == k]); ax.step(tt, ss, where="post", c=colr, label=f"{lab} (n={(g==k).sum()})")
    chi, p = compare_survival(y, g); nums[f"logrank_{c}"] = {"chi2": float(chi), "p": float(p), "n_per_group": np.bincount(g, minlength=3).tolist()}
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 60); ax.set_title(f"{c.replace('_',' ')}  (log-rank p={p:.1e})", fontsize=8); ax.set_xlabel("Months"); ax.set_ylabel("Overall survival")
    ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(FIG / "fig4_risk_groups_km.png"); plt.close()

# ---------- Fig 5: learning curves (data efficiency of mechanistic prior), CoxPH ----------
sizes = [50, 100, 200, 400, 945]; reps = 30; rng = np.random.default_rng(SEED); lc = []
Xt, yt = XY["TCGA"]
for n in sizes:
    for r in range(reps if n < 945 else 1):
        i = rng.choice(len(Xt), n, replace=False) if n < 945 else np.arange(len(Xt))
        if yt["event"][i].sum() < 5: continue
        for fs in ("Clinical", "Clinical + mechanistic", "Mechanistic only"):
            cols = m3.FEATS[fs]; mdl = make_pipeline(StandardScaler(), CoxPHSurvivalAnalysis(alpha=1.0)).fit(Xt.iloc[i][cols], yt[i])
            for c in ("OncoSG", "NCI_SherlockLung"):
                X, y = XY[c]; lc.append({"n": n, "rep": r, "features": fs, "cohort": c,
                                         "c": concordance_index_censored(y["event"], y["time"], mdl.predict(X[cols]))[0]})
lc = pd.DataFrame(lc); lc.to_csv(RES / "learning_curves.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
for ax, c in zip(axs, ("OncoSG", "NCI_SherlockLung")):
    for fs in ("Mechanistic only", "Clinical", "Clinical + mechanistic"):
        s = lc[(lc.cohort == c) & (lc.features == fs)].groupby("n").c.agg(["mean", lambda x: x.quantile(.1), lambda x: x.quantile(.9)])
        s.columns = ["mean", "lo", "hi"]; ax.plot(s.index, s["mean"], "o-", c=COL[fs], label=fs, ms=3); ax.fill_between(s.index, s.lo, s.hi, color=COL[fs], alpha=.15)
        nums[f"lc_{c}_{fs}"] = s["mean"].round(4).to_dict()
    ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels(sizes); ax.set_title(c.replace("_", " ")); ax.set_xlabel("TCGA training size (patients)"); ax.set_ylabel("External Harrell's C")
axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(FIG / "fig5_learning_curves.png"); plt.close()

# ---------- Fig 6: SHAP (permutation explainer on primary model, TCGA background) ----------
import shap
m, cols, _ = final[f"Clinical + mechanistic|{best_ln}"]
bg = shap.sample(Xt[cols], 100, random_state=SEED); Xs = shap.sample(Xt[cols], 300, random_state=SEED + 1)
expl = shap.PermutationExplainer(m.predict, bg, seed=SEED); sv = expl(Xs, max_evals=2 * len(cols) + 1 + 200)
imp = pd.Series(np.abs(sv.values).mean(0), index=cols).sort_values(); nums["shap_mean_abs"] = imp.round(4).to_dict()
# stability: rank agreement across 5 bootstrap explanation subsets
ranks = []
for b in range(5):
    Xb = shap.sample(Xt[cols], 200, random_state=SEED + 10 + b); s2 = expl(Xb, max_evals=2 * len(cols) + 1 + 200)
    ranks.append(pd.Series(np.abs(s2.values).mean(0), index=cols).rank())
from scipy.stats import kendalltau
taus = [kendalltau(ranks[i], ranks[j])[0] for i in range(5) for j in range(i + 1, 5)]; nums["shap_rank_stability_kendall_tau_mean"] = float(np.mean(taus))
plt.figure(figsize=(4.5, 3.2)); plt.barh(imp.index, imp.values, color="#2f6db3"); plt.xlabel("mean |SHAP| (risk score)"); plt.tight_layout(); plt.savefig(FIG / "fig6_shap.png"); plt.close()
plt.figure(); shap.plots.beeswarm(sv, show=False, max_display=len(cols)); plt.tight_layout(); plt.savefig(FIG / "fig6b_shap_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close()

json.dump(nums, open(RES / "figure_numbers.json", "w"), indent=2, default=float); print(json.dumps(nums, indent=2, default=float))
