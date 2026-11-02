"""Factor-model building blocks and the Portfolio container.

Latent-variable (Merton/Vasicek) set-up. Obligor i defaults when a latent variable falls
below a threshold calibrated so that the marginal default probability is exactly PD.

Gaussian one-factor:  X_i = sqrt(rho) Z + sqrt(1-rho) eps_i
    => P(default | Z) = Phi( (Phi^-1(PD) - sqrt(rho) Z) / sqrt(1-rho) )

Student-t (common mixing variable w ~ chi2(nu)/nu, tail dependence):
    X_i = (sqrt(rho) Z + sqrt(1-rho) eps_i) / sqrt(w),   default iff X_i <= t_nu^-1(PD)
    => P(default | Z, w) = Phi( (sqrt(w) t_nu^-1(PD) - sqrt(rho) Z) / sqrt(1-rho) )

Two-factor: a global factor Z (loading sqrt(rho_g)) plus an independent sector factor S_g
(loading sqrt(s_g)) per group:  denominator sqrt(1 - rho_g - s_g).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

SEED = 20261101  # single project-wide default seed => every published number is reproducible


def cond_pd_gaussian(z, pd, rho):
    """P(default | Z=z) under the Gaussian one-factor model."""
    return norm.cdf((norm.ppf(pd) - np.sqrt(rho) * z) / np.sqrt(1.0 - rho))


def cond_pd_t(z, w, pd, rho, nu):
    """P(default | Z=z, W=w) under the t-copula one-factor model."""
    tq = student_t.ppf(pd, nu)
    return norm.cdf((np.sqrt(w) * tq - np.sqrt(rho) * z) / np.sqrt(1.0 - rho))


@dataclass
class Portfolio:
    """Groups of homogeneous obligors. All arrays have length G (number of groups).

    n:   obligors per group            pd:  one-year PD per group
    rho: global-factor correlation     ead: exposure per obligor
    lgd: mean loss given default       sector_rho: optional sector-factor weight s_g
    """
    n: np.ndarray
    pd: np.ndarray
    rho: np.ndarray
    ead: np.ndarray
    lgd: np.ndarray
    sector_rho: np.ndarray | None = None

    def __post_init__(self):
        self.n = np.atleast_1d(np.asarray(self.n, dtype=int))
        for name in ("pd", "rho", "ead", "lgd"):
            setattr(self, name, np.atleast_1d(np.asarray(getattr(self, name), dtype=float)))
        if self.sector_rho is not None:
            self.sector_rho = np.atleast_1d(np.asarray(self.sector_rho, dtype=float))
            if np.any(self.rho + self.sector_rho >= 1.0):
                raise ValueError("rho + sector_rho must be < 1")
        if np.any(self.rho >= 1.0) or np.any(self.rho <= 0.0):
            raise ValueError("rho must be in (0, 1)")

    @classmethod
    def homogeneous(cls, n, pd, rho, ead=1.0, lgd=1.0, sector_rho=None):
        return cls([n], [pd], [rho], [ead], [lgd], None if sector_rho is None else [sector_rho])

    @property
    def total_exposure(self) -> float:
        return float(np.sum(self.n * self.ead))

    def expected_loss_fraction(self) -> float:
        return float(np.sum(self.n * self.ead * self.pd * self.lgd) / self.total_exposure)
