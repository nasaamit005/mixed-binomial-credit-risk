"""Exact (noise-free) finite-n loss distribution of the homogeneous mixed-binomial model.

P(L=k) = int Bin(k; n, p(z)) phi(z) dz  is evaluated by deterministic quadrature, so tail
comparisons against the Vasicek limit carry NO Monte Carlo error.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom, chi2, norm

from .models import cond_pd_gaussian, cond_pd_t


def _integrate_pmf(n, p, weights):
    k = np.arange(n + 1)[:, None]
    return binom.pmf(k, n, p[None, :]) @ weights


def mixed_binomial_pmf(n, pd, rho, zmax=8.0, m=2001):
    """Gaussian one-factor: P(L=k), k=0..n."""
    z = np.linspace(-zmax, zmax, m)
    w = norm.pdf(z) * (z[1] - z[0])
    pmf = _integrate_pmf(n, cond_pd_gaussian(z, pd, rho), w)
    return pmf / pmf.sum()


def mixed_binomial_pmf_t(n, pd, rho, nu, zmax=8.0, m=801, n_w=200):
    """t-copula: integrates Z on a grid and the chi2 mixing variable on a midpoint-quantile grid."""
    z = np.linspace(-zmax, zmax, m)
    wz = norm.pdf(z) * (z[1] - z[0])
    u = (np.arange(n_w) + 0.5) / n_w
    wgrid = chi2.ppf(u, nu) / nu
    pmf = np.zeros(n + 1)
    for w in wgrid:
        pmf += _integrate_pmf(n, cond_pd_t(z, w, pd, rho, nu), wz)
    pmf /= n_w
    return pmf / pmf.sum()


def var_es_discrete(pmf, alpha):
    """Lattice VaR and Rockafellar-Uryasev ES of L/n (VaR = smallest k/n with CDF >= alpha)."""
    n = len(pmf) - 1
    cdf = np.cumsum(pmf)
    ks = int(np.searchsorted(cdf, alpha))
    x = np.arange(n + 1) / n
    var = x[ks]
    tail = pmf[ks + 1:] @ x[ks + 1:]
    es = (tail + var * (cdf[ks] - alpha)) / (1.0 - alpha)
    return float(var), float(es)


def var_es_continuous(pmf, alpha):
    """VaR/ES of the continuity-corrected loss (L + U - 1/2)/n, U~Uniform(0,1).

    Removes the O(1/n) lattice artefact so the result is comparable with the smooth
    granularity-adjustment expansion (both effects are O(1/n) and otherwise confounded).
    """
    n = len(pmf) - 1
    cdf = np.cumsum(pmf)
    ks = int(np.searchsorted(cdf, alpha))
    cdf_prev = cdf[ks] - pmf[ks]
    var = (ks - 0.5 + (alpha - cdf_prev) / pmf[ks]) / n
    m = cdf[ks] - alpha
    frac = m / pmf[ks]
    partial_mean = (ks + 0.5 - frac / 2.0) / n
    tail = pmf[ks + 1:] @ (np.arange(ks + 1, n + 1) / n)
    es = (m * partial_mean + tail) / (1.0 - alpha)
    return float(var), float(es)


def loss_moments(pmf):
    n = len(pmf) - 1
    x = np.arange(n + 1) / n
    mean = float(pmf @ x)
    return mean, float(pmf @ (x - mean) ** 2)
