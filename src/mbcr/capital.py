"""Basel IRB corporate risk-weight function and finite-n capital gap.

!! The coefficients below follow the Basel II/III corporate IRB formula as commonly
!! published (correlation 0.12-0.24, b(PD) smoothing, 99.9% confidence). They are kept in
!! IRB_PARAMS so you can audit/override them. VERIFY every constant against the current BIS
!! text (bis.org, CRE31) and your jurisdiction's implementation before using results in a paper.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

IRB_PARAMS = {"r_min": 0.12, "r_max": 0.24, "k": 50.0, "b0": 0.11852, "b1": 0.05478, "conf": 0.999,
                  "rw_mult": 12.5}


def irb_correlation(pd, p=IRB_PARAMS):
    w = (1 - np.exp(-p["k"] * pd)) / (1 - np.exp(-p["k"]))
    return p["r_min"] * w + p["r_max"] * (1 - w)


def irb_capital_requirement(pd, lgd, maturity=2.5, p=IRB_PARAMS):
    """K per unit exposure (unexpected loss at 99.9%, maturity-adjusted)."""
    r = irb_correlation(pd, p)
    b = (p["b0"] - p["b1"] * np.log(pd)) ** 2
    cond = norm.cdf((norm.ppf(pd) + np.sqrt(r) * norm.ppf(p["conf"])) / np.sqrt(1 - r))
    return (lgd * cond - pd * lgd) * (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)


def irb_risk_weight(pd, lgd, maturity=2.5, p=IRB_PARAMS):
    return p["rw_mult"] * irb_capital_requirement(pd, lgd, maturity, p)


def capital_gap(n, pd, rho, lgd=1.0, alpha=0.999, use_correction=False):
    """Unexpected-loss capital (VaR - EL, per unit exposure) for a finite portfolio vs its
    infinite-granularity limit, same (PD, rho, LGD). Returns dict with relative gap."""
    from .correction import granularity_coefficients
    from .exact import mixed_binomial_pmf, var_es_continuous
    from .vasicek import var_vasicek
    k_inf = lgd * (var_vasicek(pd, rho, alpha) - pd)
    if use_correction:
        g = granularity_coefficients(pd, rho, alpha)
        var_n = g.var_inf + g.var_coeff / n
    else:
        var_n, _ = var_es_continuous(mixed_binomial_pmf(n, pd, rho), alpha)
    k_n = lgd * (var_n - pd)
    return {"n": n, "K_inf": float(k_inf), "K_n": float(k_n), "gap": float(k_n - k_inf),
            "rel_gap": float((k_n - k_inf) / k_inf)}
