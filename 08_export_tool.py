"""08_export_tool.py - Export the frozen primary model (Clinical + twin CoxPH) and twin summaries to JSON for the
open research tool. The tool reproduces model predictions exactly (checked below against sksurv)."""
import json, pathlib, joblib, numpy as np, pandas as pd, importlib.util
R = pathlib.Path(__file__).resolve().parents[1]; RES = R / "results"; O = R / "derived"; OUT = R / "tool"; OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("m3", R / "code/03_models.py"); m3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m3)
final = joblib.load(RES / "final_models.joblib")
model, cols, params = final["Clinical + mechanistic|CoxPH"]
sc, cox = model.steps[0][1], model.steps[1][1]
tc = pd.read_csv(O / "tcga.csv"); X, y = m3.prep(tc)
grid = np.arange(0, 61, 1.0)
bs = cox.baseline_survival_  # step function of baseline survival at lp=0 (scaled features)
base = [float(bs(min(max(t, bs.x[0]), bs.x[-1]))) for t in grid]
cache = np.load(O / "tstar_cache.npy", allow_pickle=True).item()
twin = {}
for k, v in cache.items():
    st, sq = k.split("_"); twin[f"{int(float(st))}_{int(float(sq))}"] = {
        "P60": float(np.mean(v <= 60)), "logTstar": float(np.log(np.median(v) + 1.0)),
        "relapse_free_curve": [float(np.mean(v > t)) for t in range(0, 121, 2)]}
lp_tcga = model.predict(X[cols])
export = {"features": cols, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": cox.coef_.tolist(),
          "baseline_survival_months": grid.tolist(), "baseline_survival": base, "twin": twin,
          "risk_cutpoints": np.quantile(lp_tcga, [1 / 3, 2 / 3]).tolist(), "alpha": params,
          "training": {"cohort": "TCGA LUAD+LUSC", "n": int(len(tc)), "events": int(tc.event.sum())}}
# exactness check
row = X[cols].iloc[0].values; lp = float(((row - sc.mean_) / sc.scale_) @ cox.coef_)
assert abs(lp - lp_tcga[0]) < 1e-8
s36 = base[36] ** np.exp(lp); s36_sk = float(model.predict_survival_function(X[cols].iloc[[0]])[0](36))
export["check"] = {"lp": lp, "S36_tool": s36, "S36_sksurv": s36_sk}
assert abs(s36 - s36_sk) < 1e-6, (s36, s36_sk)
# baseline survival re-estimated in each external cohort (recalibration-in-the-large; same linear predictor)
for nm, f in (("OncoSG", "oncosg"), ("NCI Sherlock-Lung", "nci")):
    d = pd.read_csv(O / f"{f}.csv"); Xe, ye = m3.prep(d); lp_e = model.predict(Xe[cols]); ev, tm = ye["event"], ye["time"]; H = []
    for t in grid:
        h = sum(((tm == u) & ev).sum() / np.exp(lp_e[tm >= u]).sum() for u in np.unique(tm[ev & (tm <= t)]))
        H.append(float(np.exp(-h)))
    export.setdefault("baselines_recalibrated", {})[nm] = H
json.dump(export, open(OUT / "model_export.json", "w"), indent=1)
hr = pd.DataFrame({"feature": cols, "coef_per_SD": cox.coef_, "HR_per_SD": np.exp(cox.coef_), "SD": sc.scale_, "HR_per_unit": np.exp(cox.coef_ / sc.scale_)})
hr.round(4).to_csv(RES / "primary_model_coefficients.csv", index=False); print(hr.round(3)); print(export["check"])
