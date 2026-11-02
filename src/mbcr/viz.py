"""Interactive Plotly figures. Shared by scripts/make_report.py and app/streamlit_app.py."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

from .analysis import error_grid
from .capital import capital_gap
from .exact import mixed_binomial_pmf, mixed_binomial_pmf_t, var_es_continuous
from .vasicek import density_vasicek, es_vasicek, var_limit_t, var_vasicek

TEMPLATE = "plotly_white"


def fig_loss_distribution(pd_, rho, alpha, n_list=(25, 50, 100, 250, 1000), nu=None):
    """Exact loss density (pmf*n) for each n vs the Vasicek limit, with VaR markers."""
    fig = go.Figure()
    xmax = min(1.0, float(var_vasicek(pd_, rho, 0.9999)) * 1.3)
    xs = np.linspace(1e-4, xmax, 1500)
    fig.add_trace(go.Scatter(x=xs, y=density_vasicek(xs, pd_, rho), name="Vasicek limit (n=inf)",
                             line={"color": "black", "width": 3, "dash": "dash"}))
    for n in n_list:
        pmf = mixed_binomial_pmf(n, pd_, rho)
        k = np.arange(n + 1) / n
        v, _ = var_es_continuous(pmf, alpha)
        fig.add_trace(go.Scatter(x=k, y=pmf * n, mode="lines+markers", name=f"exact n={n}",
                                 marker={"size": 4}, hovertemplate="loss=%{x:.3f}<br>density=%{y:.3f}",
                                 customdata=[v] * len(k)))
        fig.add_vline(x=v, line={"width": 1, "dash": "dot"}, opacity=0.35)
    if nu is not None:
        n0 = min(n_list[-1], 100)
        pmf_t = mixed_binomial_pmf_t(n0, pd_, rho, nu, n_w=100)
        fig.add_trace(go.Scatter(x=np.arange(n0 + 1) / n0, y=pmf_t * n0, name=f"t-copula nu={nu}, n={n0}",
                                 line={"color": "crimson"}))
    fig.add_vline(x=var_vasicek(pd_, rho, alpha), line={"color": "black"},
                  annotation_text=f"VaR {alpha:.1%} limit")
    fig.update_layout(template=TEMPLATE, title=f"Loss density: finite n vs Vasicek (PD={pd_:.2%}, rho={rho:.2f})",
                      xaxis={"title": "loss fraction", "range": [0, xmax]},
                      yaxis={"title": "density (log scale)", "type": "log", "range": [-4, 2]},
                      hovermode="x unified", legend={"orientation": "h", "y": -0.2})
    return fig


def fig_error_heatmap(pd_, alpha, n_list, rho_list, metric="var"):
    df = error_grid(n_list, [pd_], rho_list, [alpha])
    col = f"{metric}_rel_err_limit"
    z = df.pivot(index="rho", columns="n", values=col).loc[rho_list, n_list] * 100
    fig = go.Figure(go.Heatmap(z=z.values, x=[str(n) for n in n_list], y=[f"{r:.2f}" for r in rho_list],
                               colorscale="Reds", colorbar={"title": "% error"},
                               hovertemplate="n=%{x}<br>rho=%{y}<br>error=%{z:.2f}%<extra></extra>"))
    fig.update_layout(template=TEMPLATE, xaxis_title="portfolio size n", yaxis_title="asset correlation rho",
                      title=f"Relative {'VaR' if metric == 'var' else 'ES'} error of the Vasicek limit"
                            f" (alpha={alpha:.1%}, PD={pd_:.2%})")
    return fig


def fig_convergence(pd_, rho, alpha, n_list=(25, 50, 100, 200, 400, 800, 1600, 3200)):
    df = error_grid(n_list, [pd_], [rho], [alpha])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.n, y=(df.var_exact - df.var_inf).abs(), mode="lines+markers",
                             name="|exact - limit|  (slope -1)"))
    fig.add_trace(go.Scatter(x=df.n, y=(df.var_exact - df.var_corr).abs(), mode="lines+markers",
                             name="|exact - (limit + c/n)|  (slope -2)"))
    n0 = np.array(n_list, dtype=float)
    y1 = (df.var_exact - df.var_inf).abs().iloc[0] * n0[0] / n0
    y2 = (df.var_exact - df.var_corr).abs().iloc[0] * (n0[0] / n0) ** 2
    fig.add_trace(go.Scatter(x=n0, y=y1, line={"dash": "dot", "color": "grey"}, name="ref n^-1"))
    fig.add_trace(go.Scatter(x=n0, y=y2, line={"dash": "dot", "color": "silver"}, name="ref n^-2"))
    fig.update_layout(template=TEMPLATE, xaxis={"type": "log", "title": "n"}, yaxis={"type": "log", "title": "absolute VaR error"},
                      title=f"Convergence order (VaR {alpha:.1%}): analytic 1/n correction gains one order")
    return fig


def fig_capital(pd_, rho, lgd, n_list=(50, 100, 250, 500, 1000, 5000), alpha=0.999):
    rows = [capital_gap(n, pd_, rho, lgd, alpha) for n in n_list]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(r["n"]) for r in rows], y=[r["K_n"] * 100 for r in rows], name="finite-n capital",
                         customdata=[r["rel_gap"] * 100 for r in rows],
                         hovertemplate="n=%{x}<br>K=%{y:.2f}%<br>gap vs limit=%{customdata:+.1f}%"))
    fig.add_hline(y=rows[0]["K_inf"] * 100, line={"color": "black", "dash": "dash"},
                  annotation_text="Vasicek-limit capital")
    fig.update_layout(template=TEMPLATE, xaxis_title="portfolio size n", yaxis_title="unexpected-loss capital (% of exposure)",
                      title=f"Capital under-statement by the large-portfolio formula (alpha={alpha:.1%}, LGD={lgd:.0%})")
    return fig


def fig_t_vs_gaussian(pd_, rho, nu_list=(3, 5, 10, 30), alphas=(0.99, 0.995, 0.999)):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{a:.1%}" for a in alphas], y=[var_vasicek(pd_, rho, a) for a in alphas], name="Gaussian"))
    for nu in nu_list:
        fig.add_trace(go.Bar(x=[f"{a:.1%}" for a in alphas], y=[var_limit_t(pd_, rho, nu, a) for a in alphas],
                             name=f"t nu={nu}"))
    fig.update_layout(template=TEMPLATE, barmode="group", yaxis_title="limit VaR (loss fraction)",
                      title=f"Tail dependence: infinite-n VaR, Gaussian vs t-copula (PD={pd_:.2%}, rho={rho:.2f})")
    return fig


def fig_es_curve(pd_, rho, alphas=None):
    alphas = np.linspace(0.9, 0.9995, 40) if alphas is None else alphas
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=alphas, y=[var_vasicek(pd_, rho, a) for a in alphas], name="VaR (limit)"))
    fig.add_trace(go.Scatter(x=alphas, y=[es_vasicek(pd_, rho, a, m=20001) for a in alphas], name="ES (limit)"))
    fig.update_layout(template=TEMPLATE, xaxis_title="confidence level", yaxis_title="loss fraction",
                      title="Risk measures of the Vasicek limit")
    return fig


def z_for_alpha(alpha):
    return float(norm.ppf(1 - alpha))
