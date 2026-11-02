"""Build a single self-contained interactive HTML report: python scripts/make_report.py"""
from pathlib import Path

from mbcr import viz

OUT = Path("reports/interactive_report.html")
PD, RHO, LGD, ALPHA = 0.02, 0.15, 0.45, 0.999

figs = [
    ("Exact loss distribution vs the Vasicek limit",
     ("Each curve is the exact mixed-binomial density (no simulation noise). Small portfolios put "
     "more mass in the right tail than the limit predicts. Dotted lines mark each portfolio's VaR."),
     viz.fig_loss_distribution(PD, RHO, ALPHA, nu=5)),
    ("Where does the large-portfolio approximation fail?",
     "Heat-map of relative VaR error over portfolio size and asset correlation.",
     viz.fig_error_heatmap(PD, ALPHA, [25, 50, 100, 250, 500, 1000, 2500, 5000], [0.05, 0.10, 0.15, 0.20, 0.30, 0.40])),
    ("Convergence order",
     "Error of the plain limit decays like 1/n; adding the analytic 1/n term gains one order (1/n^2).",
     viz.fig_convergence(PD, RHO, 0.99)),
    ("Tail dependence (t-copula)",
     "Same PD and correlation, heavier joint tails: the limit VaR rises as degrees of freedom fall.",
     viz.fig_t_vs_gaussian(PD, RHO)),
    ("Capital impact",
     "Unexpected-loss capital for finite n vs the Vasicek limit used by the IRB formula.",
     viz.fig_capital(PD, RHO, LGD)),
]

parts = [("<html><head><meta charset='utf-8'><title>mbcr report</title>"
         "<style>body{font-family:system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}"
         "p{color:#444}</style></head><body><h1>Mixed-binomial credit risk: interactive report</h1>"
         f"<p>Baseline: PD={PD:.0%}, rho={RHO}, LGD={LGD:.0%}. All curves exact or seeded.</p>")]
for i, (title, text, fig) in enumerate(figs):
    parts.append(f"<h2>{title}</h2><p>{text}</p>")
    parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
parts.append("</body></html>")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print("wrote", OUT)
