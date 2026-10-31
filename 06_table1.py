"""06_table1.py - Cohort characteristics (Table 1)."""
import pandas as pd, numpy as np, pathlib
from sksurv.nonparametric import kaplan_meier_estimator
R = pathlib.Path(__file__).resolve().parents[1]; O = R / "derived"; RES = R / "results"
rows = {}
for nm, f in (("TCGA (development)", "tcga"), ("OncoSG (external 1)", "oncosg"), ("NCI Sherlock-Lung (external 2)", "nci")):
    d = pd.read_csv(O / f"{f}.csv")
    # reverse KM median follow-up
    t, s = kaplan_meier_estimator(~d.event.astype(bool), d.time); fu = t[np.argmax(s <= 0.5)] if (s <= 0.5).any() else np.nan
    t2, s2 = kaplan_meier_estimator(d.event.astype(bool), d.time)
    pc = lambda x: f"{int(x.sum())} ({100*x.mean():.1f})"
    rows[nm] = {"Patients, n": len(d), "Age, median (IQR)": f"{d.age.median():.0f} ({d.age.quantile(.25):.0f}-{d.age.quantile(.75):.0f})",
                "Female, n (%)": pc(d.female), "Ever-smoker, n (%)": pc(d.ever_smoker), "Squamous histology, n (%)": pc(d.squamous),
                **{f"Stage {r}, n (%)": pc(d.stage == k) for k, r in ((1, "I"), (2, "II"), (3, "III"), (4, "IV"))},
                "Deaths within 60 mo, n": int(d.event.sum()), "Median follow-up, mo (reverse KM)": f"{fu:.1f}",
                "3-year OS, % (KM)": f"{100*s2[t2 <= 36][-1]:.1f}"}
tab = pd.DataFrame(rows); tab.to_csv(RES / "table1_cohorts.csv"); print(tab.to_string())
