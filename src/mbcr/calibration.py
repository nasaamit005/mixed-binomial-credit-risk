"""Calibration of (PD, rho) from default-count time series, plus data access helpers.

Likelihood: for years t with d_t defaults among n_t obligors,
    L(PD, rho) = prod_t  int Bin(d_t; n_t, p(z)) phi(z) dz     (Gauss-Hermite quadrature)
Estimators: maximum likelihood (L-BFGS) and a random-walk Metropolis posterior sampler.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp
from scipy.stats import binom, norm

from .models import SEED, cond_pd_gaussian

_NODES, _WTS = np.polynomial.hermite_e.hermegauss(80)
_LOGW = np.log(_WTS / np.sqrt(2 * np.pi))


def simulate_default_counts(n_t, pd, rho, seed=SEED):
    """Synthetic 'years' of default counts with a known truth (for recovery tests)."""
    rng = np.random.default_rng(seed)
    n_t = np.asarray(n_t)
    z = rng.standard_normal(len(n_t))
    return rng.binomial(n_t, cond_pd_gaussian(z, pd, rho))


def log_likelihood(pd, rho, d, n):
    d, n = np.asarray(d), np.asarray(n)
    p = cond_pd_gaussian(_NODES, pd, rho)                       # (K,)
    ll = binom.logpmf(d[:, None], n[:, None], p[None, :])       # (T,K)
    return float(logsumexp(ll + _LOGW[None, :], axis=1).sum())


def fit_mle(d, n, start=(0.02, 0.15)):
    def nll(theta):
        pd_, rho = expit(theta[0]), expit(theta[1])
        return -log_likelihood(pd_, rho, d, n)
    res = minimize(nll, [logit(start[0]), logit(start[1])], method="L-BFGS-B")
    pd_hat, rho_hat = expit(res.x)
    return {"pd": float(pd_hat), "rho": float(rho_hat), "loglik": -float(res.fun),
            "success": bool(res.success)}


def fit_bayes(d, n, n_iter=6000, burn=1500, step=0.15, seed=SEED,
              prior_pd=(-3.5, 2.0), prior_rho=(-1.8, 1.0)):
    """Random-walk Metropolis on (logit PD, logit rho) with Normal priors on the logit scale."""
    rng = np.random.default_rng(seed)

    def logpost(th):
        pd_, rho = expit(th)
        lp = norm.logpdf(th[0], *prior_pd) + norm.logpdf(th[1], *prior_rho)
        return lp + log_likelihood(pd_, rho, d, n)

    th = np.array([prior_pd[0], prior_rho[0]], dtype=float)
    cur = logpost(th)
    chain, acc = np.empty((n_iter, 2)), 0
    for i in range(n_iter):
        prop = th + step * rng.standard_normal(2)
        lp = logpost(prop)
        if np.log(rng.random()) < lp - cur:
            th, cur, acc = prop, lp, acc + 1
        chain[i] = th
    s = expit(chain[burn:])
    return {"pd": s[:, 0], "rho": s[:, 1], "acceptance": acc / n_iter}


def fetch_fred_csv(series_id: str, cache_dir: str | Path = "data/raw", force=False) -> pd.DataFrame:
    """Download a FRED series via the public CSV endpoint (no API key); cache under data/raw.

    NOTE: Fed charge-off / delinquency rates are LOSS-rate proxies, not default counts.
    Use rates_to_pseudo_counts() only for illustrative calibration and say so in any paper.
    """
    cache = Path(cache_dir) / f"{series_id}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=[0], index_col=0)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(raw)
    return pd.read_csv(io.StringIO(raw), parse_dates=[0], index_col=0)


def rates_to_pseudo_counts(rate_pct: pd.Series, n_eff: int = 5000, annualise=True):
    """Convert an annualised percentage rate series to (defaults, n) arrays for the likelihood.

    Quarterly annualised rates are averaged within each calendar year. n_eff is an ASSUMED
    effective portfolio size; results are sensitive to it (document as a sensitivity)."""
    s = rate_pct.dropna() / 100.0
    yearly = s.groupby(s.index.year).mean() if annualise else s
    n = np.full(len(yearly), n_eff)
    return np.round(yearly.values * n_eff).astype(int), n, yearly.index.values
