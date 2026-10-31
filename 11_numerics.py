"""11_numerics.py - Numerical convergence of the twin: quadrature nodes and Monte Carlo draws."""
import numpy as np, json, pathlib, importlib.util
R = pathlib.Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("m2", R / "code/02_mechanistic.py"); m2 = importlib.util.module_from_spec(s); s.loader.exec_module(m2)
tp = json.load(open(R / "derived/twin_params.json")); vols = {float(k): v for k, v in tp["vols"].items()}
out = {"quadrature": [], "montecarlo": []}
# quadrature: same draws, vary N_EMIT
p = m2.draw(400, np.random.default_rng(tp["seed"]))
for ne in (75, 150, 300, 600, 1200):
    m2.N_EMIT = ne; row = {"N_EMIT": ne}
    for st in (1.0, 3.0):
        Mb, E = m2.curves(vols[st], 0.0, p); T = np.minimum(m2.tstar(Mb, E, st, p, tp["m"], tp["sigma"]), m2.CAP)
        row[f"P60_stage{int(st)}"] = float(np.mean(T <= 60)); row[f"medianT_stage{int(st)}"] = float(np.median(T))
    out["quadrature"].append(row)
m2.N_EMIT = 300
# Monte Carlo: vary number of draws, 10 independent seeds each -> SD of P60
for nd in (100, 200, 400, 800, 1600):
    vals = []
    for k in range(10):
        q = m2.draw(nd, np.random.default_rng(1000 + k)); Mb, E = m2.curves(vols[1.0], 0.0, q)
        vals.append(np.mean(np.minimum(m2.tstar(Mb, E, 1.0, q, tp["m"], tp["sigma"]), m2.CAP) <= 60))
    out["montecarlo"].append({"draws": nd, "P60_mean": float(np.mean(vals)), "P60_sd": float(np.std(vals, ddof=1)),
                              "binomial_se": float(np.sqrt(np.mean(vals) * (1 - np.mean(vals)) / nd))})
json.dump(out, open(R / "results/twin_numerics.json", "w"), indent=1); print(json.dumps(out, indent=1))
