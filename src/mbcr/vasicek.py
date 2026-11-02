"""Infinite-granularity (Vasicek) limit and its t-copula analogue."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, norm
from scipy.stats import t as student_t

from .models import cond_pd_gaussian


def var_vasicek(pd, rho, alpha):
    return norm.cdf((norm.ppf(pd) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1.0 - rho))


def cdf_vasicek(x, pd, rho):
    return norm.cdf((np.sqrt(1.0 - rho) * norm.ppf(x) - norm.ppf(pd)) / np.sqrt(rho))


def density_vasicek(x, pd, rho):
    s = np.sqrt((1.0 - rho) / rho)
    q = norm.ppf(x)
    return s * np.exp(-0.5 * ((np.sqrt(1.0 - rho) * q - norm.ppf(pd)) ** 2) / rho + 0.5 * q ** 2)


def es_vasicek(pd, rho, alpha, m=200001):
    """ES = average of p(Z) over the worst (1-alpha) factor states."""
    u = np.linspace(alpha, 1.0 - 1e-10, m)
    return float(np.trapezoid(cond_pd_gaussian(-norm.ppf(u), pd, rho), u) / (1.0 - alpha))


def cdf_limit_t(y, pd, rho, nu, n_w=2000):
    """CDF of the infinite-n loss fraction under the t-copula (integrates out the mixing w)."""
    u = (np.arange(n_w) + 0.5) / n_w
    w = chi2.ppf(u, nu) / nu
    tq = student_t.ppf(pd, nu)
    arg = (np.sqrt(1.0 - rho) * norm.ppf(y) - np.sqrt(w) * tq) / np.sqrt(rho)
    return float(norm.cdf(arg).mean())


def var_limit_t(pd, rho, nu, alpha):
    return brentq(lambda y: cdf_limit_t(y, pd, rho, nu) - alpha, 1e-12, 1 - 1e-12, xtol=1e-12)
