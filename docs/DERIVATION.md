# Derivation of the 1/n granularity correction (Gaussian one-factor)

Every symbol below maps to a function in `src/mbcr/correction.py`, and every claim is
checked numerically in `tests/test_correction.py`.

## 1. Model
Conditional on the systematic factor Z ~ N(0,1), defaults are independent Bernoulli(p(Z)):

    p(z) = Phi( a(z) ),   a(z) = (Phi^-1(PD) - sqrt(rho) z) / sqrt(1 - rho)

Loss fraction X = L/n, L | Z ~ Bin(n, p(Z)). Write X = Y + e with Y = p(Z) and
E[e | Z] = 0, Var(e | Z) = p(Z)(1-p(Z))/n =: sigma^2(Y)/n.

## 2. Variance decomposition (tested: `test_variance_decomposition`)
    Var(X) = Var(p(Z))  +  E[p(1-p)]/n
             systematic     idiosyncratic ("granularity")
The Vasicek limit keeps only the first term. The error is therefore O(1/n).

## 3. Quantile expansion
For a smooth conditional perturbation with variance sigma^2(y)/n (Gordy 2003; Martin & Wilde 2002):

    q_alpha(X) = q_alpha(Y) - 1/(2 n f(q)) d/dq [ f(q) sigma^2(q) ] + O(n^-2),   q = q_alpha(Y)

f is the Vasicek density of Y. **Sign check:** in the right tail f' < 0, so the correction
is positive -- finite portfolios have *more* tail risk than the limit.

## 4. Closed form via the factor parametrisation
Let z* = Phi^-1(1-alpha), c = sqrt(rho/(1-rho)), a* = a(z*), p* = p(z*), s2 = p*(1-p*).
Since Y = p(Z) is decreasing in z with |p'(z)| = c phi(a), and f(p(z)) = phi(z) / (c phi(a)):

    h(z) := f sigma^2 = phi(z) s2(z) / (c phi(a(z)))
    d ln h / dz = -z - c a - c phi(a) (1 - 2p) / (p (1-p))          [uses a' = -c]
    dh/dq = (dh/dz)/p'(z) = -h'(z) / (c phi(a))

Substituting in section 3:

    VaR_n = p* + (1/n) * s2 * (d ln h/dz) / (2 c phi(a*))

## 5. Expected shortfall
Integrating the quantile correction over levels u in [alpha, 1], with du = f dq:

    ES_n = ES_inf + (1/n) * f(q) sigma^2(q) / (2 (1 - alpha)) + O(n^-2)

The boundary term at q -> 1 vanishes. This is the closed form `es_coeff`.

## 6. Why validation uses the continuity-corrected exact quantile
L is integer-valued, so the lattice VaR jumps in steps of 1/n -- an O(1/n) artefact of the
SAME order as the correction, which would confound the comparison. `var_es_continuous` uses
(L + U - 1/2)/n with U ~ Uniform(0,1) (same mean, variance larger by only 1/(12 n^2)).
Both versions are reported in `error_grid` (`var_lattice` vs `var_exact`).

## 7. What is and is not claimed
* The expansion itself is the known granularity-adjustment framework (Gordy; Martin-Wilde;
  Gordy & Luetkebohmert for the regulatory version). It is **not** a new theorem.
* Contribution here: closed form specialised to the homogeneous Vasicek case, exhaustive
  validation against the *exact* finite-n law (no simulation noise), the lattice/continuous
  distinction, and extensions (t-copula, heterogeneity, stochastic LGD) by simulation.
* Validity: asymptotic in n. It degrades for tiny n and very small PD (see RESULTS.md).
