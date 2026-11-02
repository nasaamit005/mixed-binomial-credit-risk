import pytest

from mbcr import es_vasicek, mixed_binomial_pmf, var_es_continuous
from mbcr.correction import granularity_coefficients

CASES = [(0.02, 0.15), (0.01, 0.12), (0.05, 0.20), (0.005, 0.24)]


@pytest.mark.parametrize("pd_,rho", CASES)
@pytest.mark.parametrize("alpha", [0.99, 0.999])
def test_correction_beats_limit_by_order_of_magnitude(pd_, rho, alpha):
    n = 500
    v, e = var_es_continuous(mixed_binomial_pmf(n, pd_, rho), alpha)
    g = granularity_coefficients(pd_, rho, alpha)
    err_lim = abs(v - g.var_inf)
    err_cor = abs(v - (g.var_inf + g.var_coeff / n))
    assert err_cor < err_lim / 10
    e_inf = es_vasicek(pd_, rho, alpha)
    assert abs(e - (e_inf + g.es_coeff / n)) < abs(e - e_inf) / 10


def test_residual_is_second_order():
    """Corrected residual should shrink ~ n^-2 : doubling n cuts it by ~4x (loose: 3x)."""
    pd_, rho, a = 0.02, 0.15, 0.99
    g = granularity_coefficients(pd_, rho, a)
    res = lambda n: abs(var_es_continuous(mixed_binomial_pmf(n, pd_, rho), a)[0]
                        - (g.var_inf + g.var_coeff / n))
    assert res(200) / res(400) > 3.0


def test_positive_coefficients_in_tail():
    g = granularity_coefficients(0.02, 0.15, 0.999)
    assert g.var_coeff > 0 and g.es_coeff > 0
