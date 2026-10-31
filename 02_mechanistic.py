"""02_mechanistic.py - Mechanistic metastatic-seeding digital twin (Iwata/Benzekry framework) and features.

Model
-----
The primary tumour (cells) grows Gompertzianly from one cell:
    Vp(t) = K ** (1 - exp(-alpha t)),     alpha = ln 2 / (VDT * ln(K / N_ref)),   N_ref = 1e9 cells (~1 cm^3)
Metastases are seeded at rate mu * Vp(t) (Iwata et al., J Theor Biol 2000; Benzekry et al., Cancer Res 2016) and
grow by the same law.  Diagnosis/resection occurs at T_d, when Vp(T_d) = N0 (stage-specific volume measured in
TRACERx).  The number of seeded metastases is n ~ Poisson(Lambda), Lambda = mu * integral_0^{T_d} Vp(s) ds, so a
patient with n = 0 is cured.  Given n >= 1, the post-resection metastatic burden is
    M(T_d + tau) = n * [ integral_0^{T_d} Vp(s) Vm(T_d + tau - s) ds ] / [ integral_0^{T_d} Vp(s) ds ]
and the mechanistic relapse time T* is the first tau with M >= D (clinical detection limit, ~1e9 cells).
Stage IV disease is overt at diagnosis (T* = 0).  Metastatic propensity is heterogeneous between patients:
log10 mu ~ Normal(m, sigma).

Parameters
----------
VDT (histology-specific) : Jiang et al., Eur J Cancer 2024;212:114339 (pooled CT VDT: adenocarcinoma 223 d, squamous 140 d)
V0 by stage              : TRACERx measured tumour volumes (Jamal-Hanjani et al., NEJM 2017; cBioPortal nsclc_tracerx_2017)
K, D                     : tumour-burden conventions (1e13, 1e9 cells), varied in the Sobol sensitivity analysis
(m, sigma)               : the only fitted parameters, calibrated to (i) TRACERx Kaplan-Meier recurrence-free probability
                           at 12 and 24 months and (ii) the published envelope that 30-55% of resected NSCLC patients
                           relapse (Uramoto & Tanaka, Transl Lung Cancer Res 2014;3:242-249), applied at 60 months.
No TCGA, OncoSG or NCI outcome is used anywhere in this module, so the mechanistic features carry no outcome leakage.
"""
import numpy as np, pandas as pd, json, pathlib, itertools
from scipy.stats import poisson
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored

R = pathlib.Path(__file__).resolve().parents[1]; O = R / "derived"
SEED = 20261020
N_DRAWS = 400                     # Monte Carlo draws per (stage, histology) group
N_CAL_DRAWS = 200                 # draws per TRACERx patient during calibration
N_EMIT = 300                      # quadrature nodes for the seeding integral
CELLS = 1e9
TAU = np.arange(0, 20 * 365.25, 30.4375)          # monthly grid, 20 years after resection
CAP = 240.0                                       # months; T* beyond the grid is treated as "no relapse within 20 y"
RELAPSE_ENVELOPE = (0.30, 0.55)                   # Uramoto & Tanaka 2014

PRIORS = {
    "VDT_adeno_days": [223.0, 0.30],   # lognormal (median, sdlog)
    "VDT_squam_days": [140.0, 0.30],
    "log10_K": [13.0, 0.25],           # normal on log10 scale
    "log10_D": [9.0, 0.25],
    "V0_sdlog": 0.50,                  # between-patient spread of V0 around the TRACERx stage median
}

def draw(n, rng):
    p = {"VDT_a": PRIORS["VDT_adeno_days"][0] * np.exp(rng.normal(0, PRIORS["VDT_adeno_days"][1], n)),
         "VDT_s": PRIORS["VDT_squam_days"][0] * np.exp(rng.normal(0, PRIORS["VDT_squam_days"][1], n)),
         "K": 10 ** rng.normal(*PRIORS["log10_K"], n), "D": 10 ** rng.normal(*PRIORS["log10_D"], n),
         "v0eps": rng.normal(0, PRIORS["V0_sdlog"], n), "z_mu": rng.normal(0, 1, n), "u": rng.uniform(0, 1, n)}
    return p

def gomp(t, a, K):
    return K ** (1 - np.exp(-a * t))

def burden_curve(V0_cm3, VDT, K):
    """Returns (mean burden per metastasis on TAU, integral of Vp)."""
    a = np.log(2) / (VDT * np.log(K / CELLS))
    Td = -np.log(1 - np.log(V0_cm3 * CELLS) / np.log(K)) / a
    s = np.linspace(0, Td, N_EMIT); w = gomp(s, a, K) * (s[1] - s[0])
    M = (gomp((Td - s)[None, :] + TAU[:, None], a, K) * w[None, :]).sum(1)
    return M / w.sum(), w.sum()

def curves(V0, squamous, p):
    vdt = p["VDT_s"] if squamous > 0.5 else p["VDT_a"]
    res = [burden_curve(V0 * np.exp(p["v0eps"][i]), vdt[i], p["K"][i]) for i in range(len(vdt))]
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])

def tstar(Mbar, E, stage, p, m, sigma):
    """Relapse time (months) per draw; np.inf = cured / no relapse within 20 years."""
    if stage >= 4: return np.zeros(len(E))
    mu = 10 ** (m + sigma * p["z_mu"][:len(E)])
    n = poisson.ppf(p["u"][:len(E)], mu * E)
    hit = (n[:, None] * Mbar) >= p["D"][:len(E), None]
    out = np.where(hit.any(1), TAU[hit.argmax(1)] / 30.4375, np.inf)
    out[n == 0] = np.inf
    return out

def stage_volumes():
    tx = pd.read_csv(O / "tracerx_volume.csv")
    med = tx.groupby("stage").volume_cm3.median().to_dict(); med[4.0] = med[3.0]
    return {float(k): float(v) for k, v in med.items()}

def load_tracerx():
    t = pd.read_csv(R / "data/nsclc_tracerx_2017/data_clinical_patient.txt", sep="\t", comment="#")
    t = t.assign(stage=t.TUMOR_STAGE.astype(str).str[0].astype(float), volume_cm3=t.TUMOR_VOLUME,
                 squamous=t.HISTOLOGY.str.contains("Squamous").astype(float), time=t.RFS_MONTHS,
                 event=t.RFS_STATUS.str.startswith("1"))
    return t.dropna(subset=["volume_cm3", "time", "stage"]).reset_index(drop=True)

def km_at(ev, tm, pts):
    x, y = kaplan_meier_estimator(ev, tm)
    return np.array([y[x <= q][-1] if (x <= q).any() else 1.0 for q in pts])

def calibrate(tx):
    """Grid search over (m, sigma); all TRACERx patients x draws are vectorised in one array."""
    target = km_at(tx.event.values.astype(bool), tx.time.values, [12, 24])
    rng = np.random.default_rng(SEED); p = draw(N_CAL_DRAWS, rng)
    C = [curves(r.volume_cm3, r.squamous, p) for r in tx.itertuples()]
    Mb = np.vstack([c[0] for c in C]); E = np.concatenate([c[1] for c in C])          # (P*draws, T), (P*draws,)
    P = len(C); z = np.tile(p["z_mu"], P); u = np.tile(p["u"], P); D = np.tile(p["D"], P)
    stage4 = np.repeat(tx.stage.values >= 4, N_CAL_DRAWS)
    lo, hi = RELAPSE_ENVELOPE; grid = []
    for m, sg in itertools.product(np.arange(-16, -6.0, 0.1), [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
        n = poisson.ppf(u, 10 ** (m + sg * z) * E)
        hit = (n[:, None] * Mb) >= D[:, None]
        T = np.where(hit.any(1) & (n > 0), TAU[hit.argmax(1)] / 30.4375, np.inf); T[stage4] = 0
        S12, S24, R60 = (T > 12).mean(), (T > 24).mean(), (T <= 60).mean()
        pen = max(0, lo - R60) ** 2 + max(0, R60 - hi) ** 2
        grid.append((m, sg, (S12 - target[0]) ** 2 + (S24 - target[1]) ** 2 + pen, S12, S24, R60))
    g = pd.DataFrame(grid, columns=["m", "sigma", "loss", "S12", "S24", "R60"]); g.to_csv(O / "calibration_grid.csv", index=False)
    return g.loc[g.loss.idxmin()], target, C

if __name__ == "__main__":
    tx = load_tracerx()
    b, target, Ccal = calibrate(tx)
    m, sg = float(b.m), float(b.sigma)
    print("calibrated", b.to_dict(), "target", target)
    vols = stage_volumes()
    rng = np.random.default_rng(SEED + 7); p = draw(N_DRAWS, rng)
    out = {"calibrated_m_log10_mu": m, "calibrated_sigma_log10_mu": sg, "calibration_loss": float(b.loss),
           "tracerx_km_relapse_free_12_24": target.tolist(), "model_relapse_free_12_24": [float(b.S12), float(b.S24)],
           "model_relapse_by_60_tracerx": float(b.R60), "relapse_envelope_60": RELAPSE_ENVELOPE,
           "stage_median_volume_cm3_TRACERx": vols, "priors": PRIORS, "n_draws": N_DRAWS}
    cache = {}
    for st in (1.0, 2.0, 3.0, 4.0):
        for sq in (0.0, 1.0):
            Mb, E = curves(vols[st], sq, p); cache[(st, sq)] = np.minimum(tstar(Mb, E, st, p, m, sg), CAP)
    key = lambda k: f"stage{int(k[0])}_{'squam' if k[1] else 'adeno'}"
    out["P_relapse_by_24mo"] = {key(k): round(float(np.mean(v <= 24)), 3) for k, v in cache.items()}
    out["P_relapse_by_60mo"] = {key(k): round(float(np.mean(v <= 60)), 3) for k, v in cache.items()}
    out["P_no_relapse_20y"] = {key(k): round(float(np.mean(v >= CAP)), 3) for k, v in cache.items()}
    np.save(O / "tstar_cache.npy", {f"{k[0]}_{k[1]}": v for k, v in cache.items()}, allow_pickle=True)
    json.dump({"m": m, "sigma": sg, "vols": vols, "priors": PRIORS, "seed": SEED + 7, "n_draws": N_DRAWS}, open(O / "twin_params.json", "w"), indent=2)
    for nm in ("tcga", "oncosg", "nci"):
        df = pd.read_csv(O / f"{nm}.csv")
        df = df.drop(columns=[c for c in df.columns if c.startswith("mech_")])
        df["mech_P60"] = [np.mean(cache[(r.stage, r.squamous)] <= 60) for r in df.itertuples()]
        df["mech_logTstar"] = [np.log(np.median(cache[(r.stage, r.squamous)]) + 1.0) for r in df.itertuples()]
        df.to_csv(O / f"{nm}.csv", index=False)
        out[f"{nm}_mech_only_cindex"] = round(concordance_index_censored(df.event.astype(bool), df.time, df.mech_P60)[0], 4)

    # TRACERx: per-patient measured volume -> P(relapse by 24 mo).  (m, sigma) were fitted to TRACERx marginal
    # survival, so this is an internal consistency check, not an independent validation.
    pc = draw(N_CAL_DRAWS, np.random.default_rng(SEED))
    risk = np.array([np.mean(tstar(c[0], c[1], st, pc, m, sg) <= 24) for c, st in zip(Ccal, tx.stage.values)])
    ev, tm = tx.event.values.astype(bool), tx.time.values; rr = np.random.default_rng(SEED)
    bs = [concordance_index_censored(ev[i], tm[i], risk[i])[0] for i in (rr.integers(0, len(tx), len(tx)) for _ in range(1000)) if ev[i].sum() > 1]
    out["tracerx_check"] = {"n": int(len(tx)), "events": int(ev.sum()),
                            "cindex_twin_measured_volume": round(concordance_index_censored(ev, tm, risk)[0], 4),
                            "cindex_twin_95CI": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                            "cindex_stage_alone": round(concordance_index_censored(ev, tm, tx.stage.values)[0], 4),
                            "cindex_volume_alone": round(concordance_index_censored(ev, tm, tx.volume_cm3.values)[0], 4)}
    json.dump(out, open(O / "mechanistic_summary.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "priors"}, indent=1, default=float))
