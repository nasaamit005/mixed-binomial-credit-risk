"""Publication-quality static charts (PNG): python scripts/make_charts.py
Requires matplotlib:  pip install -e ".[figures]"   Output: docs/figures/"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from mbcr.analysis import error_grid
from mbcr.capital import capital_gap
from mbcr.exact import mixed_binomial_pmf, mixed_binomial_pmf_t, var_es_continuous
from mbcr.vasicek import density_vasicek, var_limit_t, var_vasicek

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
PD, RHO, LGD, ALPHA = 0.02, 0.15, 0.45, 0.999
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})
FOOT = f"Baseline PD = {PD:.0%}, rho = {RHO}, Gaussian one-factor model. Exact curves use deterministic quadrature."


def save(fig, name, foot=FOOT):
    fig.text(0.01, 0.005, foot, fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def chart1():
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.linspace(1e-4, 0.36, 1500)
    ax.plot(xs, density_vasicek(xs, PD, RHO), "k--", lw=2.5, label="Vasicek limit (n = infinity)", zorder=5)
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, 5))
    for n, c in zip((25, 50, 100, 250, 1000), cols):
        pmf = mixed_binomial_pmf(n, PD, RHO)
        k = np.arange(n + 1) / n
        ax.plot(k, pmf * n, "-o", ms=3, lw=1.5, color=c, label=f"exact, n = {n}")
        ax.axvline(var_es_continuous(pmf, ALPHA)[0], color=c, ls=":", lw=1)
    t = mixed_binomial_pmf_t(100, PD, RHO, 5, n_w=100)
    ax.plot(np.arange(101) / 100, t * 100, color="crimson", lw=2, label="t-copula (nu = 5), n = 100")
    ax.axvline(var_vasicek(PD, RHO, ALPHA), color="k", lw=1.5)
    ax.annotate("99.9% VaR of the limit", (var_vasicek(PD, RHO, ALPHA), 40), xytext=(8, 0),
                textcoords="offset points", fontsize=9)
    ax.set(yscale="log", xlim=(0, 0.36), ylim=(1e-4, 1e2), xlabel="loss fraction",
           ylabel="probability density (log scale)",
           title="1. The exact loss distribution: small portfolios carry more tail mass")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "01_exact_loss_distribution.png")


def chart2():
    ns = [25, 50, 100, 250, 500, 1000, 2500, 5000]
    rhos = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    df = error_grid(ns, [PD], rhos, [ALPHA])
    z = df.pivot(index="rho", columns="n", values="var_rel_err_limit").loc[rhos, ns].values * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(z, cmap="Reds", norm=LogNorm(vmin=max(z.min(), 0.05), vmax=z.max()), aspect="auto",
                   origin="lower")
    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_yticks(range(len(rhos)), [f"{r:.2f}" for r in rhos])
    ax.grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.1f}%" if z[i, j] < 10 else f"{z[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8.5, color="white" if z[i, j] > z.max() ** 0.6 else "black")
    fig.colorbar(im, ax=ax, label="relative VaR error of the limit (%, log colour scale)")
    ax.set(xlabel="portfolio size n", ylabel="asset correlation rho",
           title="2. Where the large-portfolio approximation fails (99.9% VaR, limit understates risk)")
    save(fig, "02_error_heatmap.png", f"PD = {PD:.0%}, alpha = 99.9%, Gaussian one-factor model; rho varies by row. Exact finite-n VaR vs Vasicek limit.")


def chart3():
    ns = [25, 50, 100, 200, 400, 800, 1600, 3200]
    df = error_grid(ns, [PD], [RHO], [0.99])
    e1 = (df.var_exact - df.var_inf).abs().values
    e2 = (df.var_exact - df.var_corr).abs().values
    n = np.array(ns, float)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(n, e1, "o-", lw=2, label="|exact - Vasicek limit|  (slope -1)")
    ax.loglog(n, e2, "s-", lw=2, label="|exact - (limit + c/n)|  (slope -2)")
    ax.loglog(n, e1[0] * n[0] / n, ":", color="grey", label="reference n^-1")
    ax.loglog(n, e2[0] * (n[0] / n) ** 2, ":", color="silver", label="reference n^-2")
    ax.set(xlabel="portfolio size n", ylabel="absolute VaR error (loss fraction)",
           title="3. Convergence order: the analytic 1/n correction gains one full order (VaR 99%)")
    ax.legend(fontsize=9)
    save(fig, "03_convergence_order.png", f"PD = {PD:.0%}, rho = {RHO}, alpha = 99%. Corrected-residual wiggles reflect the 1/n lattice of the exact loss.")


def chart4():
    alphas = (0.99, 0.995, 0.999)
    labels = ["Gaussian"] + [f"t-copula nu = {v}" for v in (30, 10, 5, 3)]
    vals = [[var_vasicek(PD, RHO, a) for a in alphas]] + \
           [[var_limit_t(PD, RHO, v, a) for a in alphas] for v in (30, 10, 5, 3)]
    fig, ax = plt.subplots(figsize=(10, 6))
    w = 0.16
    for i, (lab, v) in enumerate(zip(labels, vals)):
        bars = ax.bar(np.arange(3) + (i - 2) * w, v, w, label=lab,
                      color=["#555", "#6baed6", "#3182bd", "#de2d26", "#a50f15"][i])
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x + 0.008, f"{x:.1%}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3), [f"{a:.1%}" for a in alphas])
    ax.set(xlabel="confidence level", ylabel="infinite-portfolio VaR (loss fraction)",
           title="4. Tail dependence: t-copula versus Gaussian (same PD and correlation)")
    ax.legend(fontsize=9)
    save(fig, "04_tail_dependence.png", f"PD = {PD:.0%}, rho = {RHO}. Infinite-portfolio VaR; t-copula has the same marginal PD and correlation parameter.")


def chart5():
    ns = (50, 100, 250, 500, 1000, 5000)
    rows = [capital_gap(n, PD, RHO, LGD, ALPHA) for n in ns]
    k = [r["K_n"] * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar([str(n) for n in ns], k, color="#3182bd")
    ax.axhline(rows[0]["K_inf"] * 100, color="k", ls="--", lw=1.8,
               label=f"large-portfolio (Vasicek) capital = {rows[0]['K_inf'] * 100:.2f}%")
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{r['rel_gap']:+.1%}", ha="center",
                fontsize=10, fontweight="bold")
    ax.set(xlabel="portfolio size n", ylabel="unexpected-loss capital (% of exposure)",
           ylim=(0, max(k) * 1.18),
           title=f"5. Capital impact: finite portfolios need more than the limit implies (99.9%, LGD {LGD:.0%})")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "05_capital_impact.png", f"PD = {PD:.0%}, rho = {RHO}, LGD = {LGD:.0%}. Illustrative: uses the Vasicek rho directly, not the Basel IRB rho(PD).")


if __name__ == "__main__":
    for f in (chart1, chart2, chart3, chart4, chart5):
        f()
