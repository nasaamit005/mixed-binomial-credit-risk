"""First-order (1/n) granularity correction for the Gaussian one-factor model.

Write the loss fraction as X = Y + e with Y = p(Z) (systematic part) and Var(e | Y) =
sigma^2(Y)/n, sigma^2(y) = y(1-y). The Gordy / Martin-Wilde expansion gives

    q_alpha(X) = q_alpha(Y) - 1/(2 n f(q)) d/dq [ f(q) sigma^2(q) ] + O(n^-2)
    ES_alpha(X) = ES_alpha(Y) + f(q) sigma^2(q) / (2 n (1-alpha)) + O(n^-2)

with f the Vasicek density and q = q_alpha(Y). Using the factor parametrisation
(z* = Phi^-1(1-alpha), c = sqrt(rho/(1-rho)), a = (Phi^-1(PD) - sqrt(rho) z*)/sqrt(1-rho)) all
terms are closed-form -- see docs/DERIVATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class GranularityCoefficients:
    var_inf: float
    var_coeff: float     # VaR_n ~= var_inf + var_coeff / n
    es_coeff: float      # ES_n  ~= ES_inf  + es_coeff / n


def granularity_coefficients(pd, rho, alpha) -> GranularityCoefficients:
    z = norm.ppf(1.0 - alpha)
    c = np.sqrt(rho / (1.0 - rho))
    a = (norm.ppf(pd) - np.sqrt(rho) * z) / np.sqrt(1.0 - rho)
    p = norm.cdf(a)
    phia = norm.pdf(a)
    s2 = p * (1.0 - p)
    dlogh = -z - a * c - c * phia * (1.0 - 2.0 * p) / s2
    var_coeff = s2 * dlogh / (2.0 * c * phia)
    h = norm.pdf(z) * s2 / (c * phia)          # f(q) * sigma^2(q)
    es_coeff = h / (2.0 * (1.0 - alpha))
    return GranularityCoefficients(float(p), float(var_coeff), float(es_coeff))


def granularity_correction(n, pd, rho, alpha):
    """Return (VaR_n, ES_n) first-order approximations; ES needs es_vasicek for the base."""
    from .vasicek import es_vasicek
    g = granularity_coefficients(pd, rho, alpha)
    return g.var_inf + g.var_coeff / n, es_vasicek(pd, rho, alpha) + g.es_coeff / n
