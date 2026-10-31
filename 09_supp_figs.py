"""09_supp_figs.py - Supplementary Fig. S1: twin calibration loss surface over (m, sigma)."""
import pandas as pd, numpy as np, pathlib, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parents[1]
g = pd.read_csv(R / "derived/calibration_grid.csv"); best = g.loc[g.loss.idxmin()]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
for sg in sorted(g.sigma.unique()):
    d = g[g.sigma == sg]; axs[0].plot(d.m, np.log10(d.loss + 1e-6), label=f"σ={sg:g}", lw=1)
axs[0].axvline(best.m, ls=":", c="k"); axs[0].set_xlabel("m (mean log10 μ)"); axs[0].set_ylabel("log10 calibration loss"); axs[0].legend(fontsize=6, frameon=False, ncol=2)
d = g[g.sigma == best.sigma]
axs[1].plot(d.m, d.S12, label="Twin relapse-free 12 mo"); axs[1].plot(d.m, d.S24, label="Twin relapse-free 24 mo"); axs[1].plot(d.m, d.R60, label="Twin relapse by 60 mo")
axs[1].axhspan(0.30, 0.55, color="#d1d5db", alpha=.5, label="Literature envelope (60 mo)")
axs[1].axvline(best.m, ls=":", c="k"); axs[1].set_xlabel(f"m (σ = {best.sigma:g})"); axs[1].set_ylabel("Probability"); axs[1].legend(fontsize=6, frameon=False)
for a in axs: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(R / "figures/figS1_calibration_surface.png", dpi=300); print(best.to_dict())
