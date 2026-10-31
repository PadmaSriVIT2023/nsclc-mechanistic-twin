"""05_twin_sensitivity.py - Global (Sobol) sensitivity of the mechanistic relapse time and uncertainty-banded
in-silico trajectories (model scenarios; not clinical observations)."""
import numpy as np, pandas as pd, json, pathlib, importlib.util
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import poisson
from SALib.sample import sobol as ss
from SALib.analyze import sobol as sa
R = pathlib.Path(__file__).resolve().parents[1]; O = R / "derived"; RES = R / "results"; FIG = R / "figures"
RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("m2", R / "code/02_mechanistic.py"); m2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m2)
tp = json.load(open(O / "twin_params.json")); m, sg = tp["m"], tp["sigma"]; vols = {float(k): v for k, v in tp["vols"].items()}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})

# ---- Sobol: output = log(1 + expected-burden relapse time) for a single patient, inputs over plausible ranges ----
problem = {"num_vars": 6, "names": ["VDT (log days)", "log10 K", "log10 D", "log10 mu", "V0 (log cm3)", "seed quantile u"],
           "bounds": [[np.log(100), np.log(400)], [12.5, 13.5], [8.5, 9.5], [m - 2, m + 2], [np.log(1), np.log(80)], [0.0, 1.0]]}
X = ss.sample(problem, 512, calc_second_order=False, seed=20261020); Y = []
for v in X:
    Mb, E = m2.burden_curve(np.exp(v[4]), np.exp(v[0]), 10 ** v[1])
    n = poisson.ppf(min(v[5], 0.999999), 10 ** v[3] * E)
    hit = np.nonzero(n * Mb >= 10 ** v[2])[0]
    t = m2.TAU[hit[0]] / 30.4375 if (n > 0 and len(hit)) else m2.CAP
    Y.append(np.log1p(min(t, m2.CAP)))
Si = sa.analyze(problem, np.array(Y), calc_second_order=False, seed=20261020)
sob = pd.DataFrame({"S1": Si["S1"], "S1_conf": Si["S1_conf"], "ST": Si["ST"], "ST_conf": Si["ST_conf"]}, index=problem["names"]).round(3)
sob.to_csv(RES / "sobol_indices.csv"); print(sob)

# ---- trajectories and mechanistic relapse-free curves, adenocarcinoma, stages I-III ----
rng = np.random.default_rng(20261020); p = m2.draw(400, rng)
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2)); tm = m2.TAU / 30.4375; keep = tm <= 120; out = {}
for st, c in ((1.0, "#2f6db3"), (2.0, "#d4a017"), (3.0, "#b91c1c")):
    Mb, E = m2.curves(vols[st], 0.0, p)
    mu = 10 ** (m + sg * p["z_mu"]); n = poisson.ppf(p["u"], mu * E)
    B = n[:, None] * Mb; seeded = n > 0
    lg = np.log10(np.maximum(B[seeded], 1)); q = np.percentile(lg, [10, 50, 90], axis=0)
    axs[0].plot(tm[keep], q[1][keep], c=c, label=f"Stage {'I' * int(st)} (seeded draws)"); axs[0].fill_between(tm[keep], q[0][keep], q[2][keep], color=c, alpha=.15)
    T = np.minimum(m2.tstar(Mb, E, st, p, m, sg), m2.CAP); tt = np.arange(0, 121)
    axs[1].plot(tt, [np.mean(T > x) for x in tt], c=c, label=f"Stage {'I' * int(st)}")
    out[f"stage{int(st)}_adeno"] = {"P_relapse_free_24": float(np.mean(T > 24)), "P_relapse_free_60": float(np.mean(T > 60)),
                                     "P_no_seeding": float(np.mean(~seeded))}
axs[0].axhline(9, ls=":", c="k", lw=.8); axs[0].text(2, 9.2, "detection limit D", fontsize=7)
axs[0].set_xlabel("Months after resection"); axs[0].set_ylabel("log10 metastatic cells"); axs[0].legend(frameon=False, fontsize=7)
axs[1].set_xlabel("Months after resection"); axs[1].set_ylabel("Twin relapse-free probability"); axs[1].set_ylim(0, 1.02); axs[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig7_twin_trajectories.png"); plt.close()

# ---- Sobol bar chart ----
plt.figure(figsize=(4.6, 2.8)); yy = np.arange(len(sob))
plt.barh(yy - .18, sob.S1, .35, xerr=sob.S1_conf, color="#93c5fd", label="First order S1")
plt.barh(yy + .18, sob.ST, .35, xerr=sob.ST_conf, color="#1d4ed8", label="Total ST")
plt.yticks(yy, sob.index); plt.xlabel("Sobol index"); plt.legend(frameon=False, fontsize=7); plt.tight_layout(); plt.savefig(FIG / "figS2_sobol.png"); plt.close()
json.dump({"sobol": sob.to_dict(), **out}, open(RES / "twin_sensitivity.json", "w"), indent=2); print(out)
