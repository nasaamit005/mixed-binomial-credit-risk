"""Monte Carlo engine with importance sampling on the global factor.

Supports: Gaussian / Student-t copula, optional per-group sector factors (two-factor model),
heterogeneous groups (PD, rho, EAD, LGD), fixed or Beta-distributed stochastic LGD.

Importance sampling: draw Z ~ N(shift, 1) with shift < 0 (bad states) and reweight by
dN(0,1)/dN(shift,1) = exp(-shift*Z + shift^2/2). Estimators are self-normalised.
Standard errors come from batch means so they are valid for weighted estimators too.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from .models import SEED, Portfolio


def _beta_params(mean, sd):
    mean = float(np.clip(mean, 1e-6, 1 - 1e-6))
    var = min(sd ** 2, mean * (1 - mean) * 0.999)
    k = mean * (1 - mean) / var - 1.0
    return mean * k, (1 - mean) * k


def simulate_losses(pf: Portfolio, N: int, seed: int = SEED, model: str = "gaussian",
                    nu: float = 5.0, lgd_model: str = "fixed", lgd_sd: float = 0.25,
                    is_shift: float = 0.0):
    """Return (loss_fraction[N], weights[N])."""
    if model not in ("gaussian", "t"):
        raise ValueError("model must be 'gaussian' or 't'")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(N) + is_shift
    weights = np.exp(-is_shift * z + 0.5 * is_shift ** 2)
    mix = np.sqrt(rng.chisquare(nu, N) / nu) if model == "t" else np.ones(N)
    loss = np.zeros(N)
    for g in range(len(pf.n)):
        sr = 0.0 if pf.sector_rho is None else pf.sector_rho[g]
        thr = norm.ppf(pf.pd[g]) * mix if model == "gaussian" else student_t.ppf(pf.pd[g], nu) * mix
        num = thr - np.sqrt(pf.rho[g]) * z
        if sr > 0:
            num = num - np.sqrt(sr) * rng.standard_normal(N)
        p = norm.cdf(num / np.sqrt(1.0 - pf.rho[g] - sr))
        d = rng.binomial(pf.n[g], p)
        if lgd_model == "fixed":
            loss += pf.ead[g] * pf.lgd[g] * d
        elif lgd_model == "beta":
            a, b = _beta_params(pf.lgd[g], lgd_sd)
            tot = int(d.sum())
            if tot:
                draws = rng.beta(a, b, tot)
                loss += pf.ead[g] * np.bincount(np.repeat(np.arange(N), d), weights=draws,
                                               minlength=N)
        else:
            raise ValueError("lgd_model must be 'fixed' or 'beta'")
    return loss / pf.total_exposure, weights


def weighted_var_es(losses, weights, alpha):
    order = np.argsort(losses)
    x, w = losses[order], weights[order]
    cw = np.cumsum(w) / w.sum()
    k = int(np.searchsorted(cw, alpha))
    var = x[min(k, len(x) - 1)]
    es = var + (w * np.maximum(x - var, 0.0)).sum() / w.sum() / (1.0 - alpha)
    return float(var), float(es)


def estimate_risk(losses, weights, alpha, n_batches=20):
    """VaR/ES with batch-means standard errors and the Kish effective sample size."""
    var, es = weighted_var_es(losses, weights, alpha)
    idx = np.array_split(np.arange(len(losses)), n_batches)
    bv, be = zip(*(weighted_var_es(losses[i], weights[i], alpha) for i in idx))
    ess = float(weights.sum() ** 2 / (weights ** 2).sum())
    return {"var": var, "es": es,
            "var_se": float(np.std(bv, ddof=1) / np.sqrt(n_batches)),
            "es_se": float(np.std(be, ddof=1) / np.sqrt(n_batches)),
            "ess": ess}


def default_is_shift(alpha):
    """Half the alpha-tail factor quantile (negative Z = bad state).

    Empirical finding (see docs/RESULTS.md): centring the proposal exactly on the tail
    quantile is too aggressive -- weight variance explodes (ESS ~140 of 200k at alpha=0.999)
    and the standard error is WORSE than plain MC at alpha=0.99. Half the quantile gave the
    smallest ES standard error in a sweep (about 8x lower than plain MC at alpha=0.999)."""
    return float(0.5 * norm.ppf(1.0 - alpha))
