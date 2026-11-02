"""Experiment grids: exact error of the Vasicek limit and of the 1/n correction."""
from __future__ import annotations

import itertools

import pandas as pd

from .correction import granularity_coefficients
from .exact import mixed_binomial_pmf, var_es_continuous, var_es_discrete
from .vasicek import es_vasicek


def error_grid(n_list, pd_list, rho_list, alphas=(0.99, 0.999)) -> pd.DataFrame:
    rows = []
    for n, pd_, rho in itertools.product(n_list, pd_list, rho_list):
        pmf = mixed_binomial_pmf(n, pd_, rho)
        for a in alphas:
            vc, ec = var_es_continuous(pmf, a)
            vd, ed = var_es_discrete(pmf, a)
            g = granularity_coefficients(pd_, rho, a)
            es_inf = es_vasicek(pd_, rho, a)
            rows.append({
                "n": n, "pd": pd_, "rho": rho, "alpha": a,
                "var_exact": vc, "es_exact": ec, "var_lattice": vd, "es_lattice": ed,
                "var_inf": g.var_inf, "es_inf": es_inf,
                "var_corr": g.var_inf + g.var_coeff / n, "es_corr": es_inf + g.es_coeff / n})
    df = pd.DataFrame(rows)
    df["var_rel_err_limit"] = df.var_exact / df.var_inf - 1
    df["es_rel_err_limit"] = df.es_exact / df.es_inf - 1
    df["var_rel_err_corr"] = df.var_exact / df.var_corr - 1
    df["es_rel_err_corr"] = df.es_exact / df.es_corr - 1
    return df
