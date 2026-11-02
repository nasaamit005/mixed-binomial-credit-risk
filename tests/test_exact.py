import numpy as np
import pytest
from scipy.stats import norm

from mbcr import (
    density_vasicek,
    es_vasicek,
    mixed_binomial_pmf,
    var_es_continuous,
    var_es_discrete,
    var_vasicek,
)
from mbcr.exact import loss_moments, mixed_binomial_pmf_t
from mbcr.vasicek import cdf_vasicek


def test_pmf_normalised_and_mean_equals_pd():
    pmf = mixed_binomial_pmf(200, 0.03, 0.2)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    assert loss_moments(pmf)[0] == pytest.approx(0.03, abs=1e-6)


def test_variance_decomposition():
    """Var(L/n) = Var(p(Z)) + E[p(1-p)]/n  (systematic + idiosyncratic)."""
    n, pd_, rho = 150, 0.02, 0.15
    _mean, var = loss_moments(mixed_binomial_pmf(n, pd_, rho))
    # Var(p(Z)) = P(X1<t,X2<t) - PD^2 with correlation rho: closed form via bivariate normal
    from scipy.stats import multivariate_normal as mvn
    t = norm.ppf(pd_)
    joint = mvn(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([t, t])
    var_sys = joint - pd_ ** 2
    var_idio = (pd_ - joint) / n          # E[p(1-p)] = PD - E[p^2] = PD - joint
    assert var == pytest.approx(var_sys + var_idio, rel=2e-4)


def test_converges_to_vasicek_limit():
    pd_, rho, a = 0.02, 0.15, 0.999
    errs = [abs(var_es_continuous(mixed_binomial_pmf(n, pd_, rho), a)[0] / var_vasicek(pd_, rho, a) - 1)
            for n in (200, 800, 3200)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.01


def test_error_decays_like_one_over_n():
    pd_, rho, a = 0.02, 0.15, 0.99
    v = lambda n: var_es_continuous(mixed_binomial_pmf(n, pd_, rho), a)[0] - var_vasicek(pd_, rho, a)
    ratio = v(500) / v(1000)
    assert ratio == pytest.approx(2.0, rel=0.1)


def test_vasicek_limit_closed_forms_consistent():
    pd_, rho = 0.02, 0.15
    q = var_vasicek(pd_, rho, 0.99)
    assert cdf_vasicek(q, pd_, rho) == pytest.approx(0.99, abs=1e-12)
    x = np.linspace(1e-4, 0.5, 200001)
    assert np.trapezoid(density_vasicek(x, pd_, rho), x) == pytest.approx(cdf_vasicek(0.5, pd_, rho), abs=2e-3)
    assert es_vasicek(pd_, rho, 0.99) > q


def test_lattice_var_dominates_continuous_on_average():
    pmf = mixed_binomial_pmf(100, 0.02, 0.15)
    vd, ed = var_es_discrete(pmf, 0.99)
    vc, ec = var_es_continuous(pmf, 0.99)
    assert abs(vd - vc) <= 1.0 / 100           # differ by at most one lattice step
    assert ec >= vc and ed >= vd


def test_t_copula_fatter_tail_than_gaussian():
    n, pd_, rho = 100, 0.02, 0.15
    g = var_es_continuous(mixed_binomial_pmf(n, pd_, rho), 0.999)[0]
    t = var_es_continuous(mixed_binomial_pmf_t(n, pd_, rho, nu=4), 0.999)[0]
    assert t > g
    big_nu = var_es_continuous(mixed_binomial_pmf_t(n, pd_, rho, nu=400), 0.99)[0]
    gauss = var_es_continuous(mixed_binomial_pmf(n, pd_, rho), 0.99)[0]
    assert big_nu == pytest.approx(gauss, rel=0.05)
