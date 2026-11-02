"""Interactive dashboard:  pip install -e ".[app]" && streamlit run app/streamlit_app.py"""
import streamlit as st

from mbcr import viz
from mbcr.analysis import error_grid
from mbcr.correction import granularity_coefficients

st.set_page_config(page_title="Mixed-binomial credit risk", layout="wide")
st.title("Finite-sample accuracy of the Vasicek approximation")

with st.sidebar:
    pd_ = st.slider("PD", 0.001, 0.15, 0.02, 0.001, format="%.3f")
    rho = st.slider("Asset correlation rho", 0.02, 0.45, 0.15, 0.01)
    alpha = st.select_slider("Confidence level", [0.95, 0.99, 0.995, 0.999], value=0.999)
    lgd = st.slider("LGD", 0.1, 1.0, 0.45, 0.05)
    n = st.select_slider("Portfolio size n", [25, 50, 100, 250, 500, 1000, 2500, 5000], value=100)
    nu = st.slider("t-copula degrees of freedom (0 = off)", 0, 30, 0)

g = granularity_coefficients(pd_, rho, alpha)
row = error_grid([n], [pd_], [rho], [alpha]).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Exact VaR", f"{row.var_exact:.3%}", f"{row.var_rel_err_limit:+.1%} vs limit", delta_color="inverse")
c2.metric("Vasicek-limit VaR", f"{row.var_inf:.3%}")
c3.metric("Exact ES", f"{row.es_exact:.3%}", f"{row.es_rel_err_limit:+.1%} vs limit", delta_color="inverse")
c4.metric("1/n-corrected VaR", f"{row.var_corr:.3%}", f"{row.var_rel_err_corr:+.2%} residual")
st.caption(f"First-order coefficient: VaR_n ~ {g.var_inf:.4f} + {g.var_coeff:.3f}/n")

t1, t2, t3, t4, t5 = st.tabs(["Distribution", "Error map", "Convergence", "Tail dependence", "Capital"])
with t1:
    st.plotly_chart(viz.fig_loss_distribution(pd_, rho, alpha, nu=nu or None), width="stretch")
with t2:
    metric = st.radio("Measure", ["var", "es"], horizontal=True)
    st.plotly_chart(viz.fig_error_heatmap(pd_, alpha, [25, 50, 100, 250, 500, 1000, 2500],
                                          [0.05, 0.10, 0.15, 0.20, 0.30, 0.40], metric), width="stretch")
with t3:
    st.plotly_chart(viz.fig_convergence(pd_, rho, alpha), width="stretch")
with t4:
    st.plotly_chart(viz.fig_t_vs_gaussian(pd_, rho), width="stretch")
with t5:
    st.plotly_chart(viz.fig_capital(pd_, rho, lgd, alpha=alpha), width="stretch")
    st.warning("IRB constants must be verified against the current BIS text (see src/mbcr/capital.py).")
