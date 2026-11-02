"""Build the self-contained interactive page:  python scripts/make_report.py
Writes docs/index.html (publish it with GitHub Pages) and reports/interactive_report.html."""
from pathlib import Path

from mbcr import viz

PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"
PD, RHO, LGD, ALPHA = 0.02, 0.15, 0.45, 0.999

SECTIONS = [
    ("1. The exact loss distribution",
     ("Each curve is the exact mixed-binomial density (no simulation noise) for a portfolio of n loans. "
     "The dashed black curve is the Vasicek large-portfolio limit. Small portfolios put more probability "
     "in the right tail than the limit predicts. Vertical dotted lines mark each portfolio’s 99.9% VaR. "
     "The red curve is a t-copula portfolio (ν = 5, n = 100) with heavier joint tails."),
     viz.fig_loss_distribution(PD, RHO, ALPHA, nu=5)),
    ("2. Where the large-portfolio approximation fails",
     ("Relative error of the Vasicek VaR over portfolio size (columns) and asset correlation (rows). "
     "Hover a cell for the exact value. Red means the limit understates risk."),
     viz.fig_error_heatmap(PD, ALPHA, [25, 50, 100, 250, 500, 1000, 2500, 5000],
                           [0.05, 0.10, 0.15, 0.20, 0.30, 0.40])),
    ("3. Convergence order",
     ("Both axes are logarithmic. The plain limit’s error falls like 1/n. Adding the closed-form 1/n "
     "granularity term gains a full order, so the residual falls like 1/n²."),
     viz.fig_convergence(PD, RHO, 0.99)),
    ("4. Tail dependence (t-copula)",
     ("Same PD and correlation, but heavier joint tails: the infinite-portfolio VaR rises sharply as the "
     "degrees of freedom ν fall. Copula choice is a bigger source of model risk than finite size once n ≥ 250."),
     viz.fig_t_vs_gaussian(PD, RHO)),
    ("5. Capital impact",
     ("Unexpected-loss capital for a finite portfolio against the Vasicek-limit capital that large-portfolio "
     "formulas such as Basel IRB assume. Hover a bar for the gap."),
     viz.fig_capital(PD, RHO, LGD)),
]

# make the first chart readable regardless of the installed viz.py version
SECTIONS[0][2].update_yaxes(range=[-4, 2])

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--muted:#5a6572;--line:#dde1e6;--accent:#c0392b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14171a;--card:#1d2126;--ink:#e8eaed;--muted:#9aa4af;--line:#2d333a;--accent:#ff6b5b}}
:root[data-theme="dark"]{--bg:#14171a;--card:#1d2126;--ink:#e8eaed;--muted:#9aa4af;--line:#2d333a;--accent:#ff6b5b}
:root{box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
*,*::before,*::after{box-sizing:inherit}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:1.5rem 1rem 3rem}
h1{font-size:1.9rem;line-height:1.2;margin:.2rem 0 .4rem}
h2{font-size:1.2rem;margin:0 0 .3rem}
p.lead{color:var(--muted);margin:.2rem 0 1.2rem}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1rem 1rem .6rem;margin:0 0 1.2rem}
section p{color:var(--muted);margin:.2rem 0 .8rem}
.plot{background:#fff;border-radius:8px;overflow:hidden}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.8rem;margin:1rem 0 1.4rem}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem}
.kpi b{display:block;font-size:1.5rem;color:var(--accent)}
.kpi span{color:var(--muted);font-size:.9rem}
.warn{border-left:4px solid var(--accent);background:var(--card);padding:.7rem 1rem;border-radius:8px;color:var(--muted);margin-bottom:1.2rem}
footer{color:var(--muted);font-size:.9rem}
"""

KPIS = [
    ("+11.1%", "99.9% VaR above the Vasicek limit at n = 100 (PD 2%, ρ 0.15)"),
    ("~n⁻²", "residual after the closed-form 1/n correction (plain limit: ~n⁻¹)"),
    ("43.3% vs 17.6%", "infinite-n 99.9% VaR, t-copula (ν = 5) vs Gaussian"),
    ("+12.5%", "capital gap at n = 100 versus the large-portfolio limit"),
]


def build() -> str:
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        "<title>Mixed-binomial credit risk: interactive results</title>",
        ("<link rel=\"icon\" href=\"data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' "
        "viewBox='0 0 100 100'><text y='.9em' font-size='90'>📉</text></svg>\">"),
        f"<style>{CSS}</style>",
        f'<script src="{PLOTLY_JS}"></script></head><body><main>',
        "<h1>Mixed-binomial credit risk</h1>",
        ("<p class='lead'>How accurate is the Vasicek large-portfolio approximation for real, finite loan "
        f"portfolios? Baseline: PD = {PD:.0%}, asset correlation ρ = {RHO}, LGD = {LGD:.0%}. "
        "Curves are exact (deterministic quadrature) unless stated. Hover, zoom and click legend entries.</p>"),
        "<div class='kpis'>" + "".join(f"<div class='kpi'><b>{v}</b><span>{t}</span></div>" for v, t in KPIS)
        + "</div>",
    ]
    for title, text, fig in SECTIONS:
        fig.update_layout(margin={"l": 60, "r": 20, "t": 60, "b": 60}, paper_bgcolor="white", plot_bgcolor="white")
        parts.append(f"<section><h2>{title}</h2><p>{text}</p><div class='plot'>"
                     + fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
                     + "</div></section>")
    parts.append(
        "<div class='warn'><b>Read with care.</b> The 1/n expansion is the established granularity-adjustment "
        "framework, not a new theorem. It is asymptotic and unreliable for very small, low-PD portfolios "
        "(worst case: a 41% residual at n = 25, PD 0.1%). The capital figures use the Vasicek ρ directly, "
        "not the Basel IRB-prescribed ρ(PD), and the IRB constants must be verified against the BIS text "
        "before any publication use.</div>"
        "<footer>Source code, tests and derivations: the <b>mixed-binomial-credit-risk</b> repository "
        "(docs/DERIVATION.md, docs/RESULTS.md).</footer></main></body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    html = build()
    for out in (Path("docs/index.html"), Path("reports/interactive_report.html")):
        out.parent.mkdir(exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print("wrote", out)
