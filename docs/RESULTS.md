# Results log (all numbers reproducible: `pytest`, `python scripts/run_grid.py`)

Baseline PD = 2%, rho = 0.15, Gaussian one-factor. "Exact" = deterministic quadrature of the mixed
binomial, continuity-corrected unless stated.

## R1. The Vasicek limit understates tail risk for finite portfolios
VaR99.9%: limit = 17.63%. Exact: n=100 -> 19.58 (+11.1%), n=250 -> 18.43, n=500 -> 18.03, n=1000 -> 17.83, n=2500 -> 17.71.
Error decays like 1/n (test: `test_error_decays_like_one_over_n`).

## R2. The analytic 1/n term removes the first-order error
Residual at VaR99.9%: n=100 limit +1.95e-2 -> corrected -5.1e-4; n=500: +4.0e-3 -> -1.0e-5.
Doubling n cuts the corrected residual by > 3x (second order). Holds for ES (closed form) too.

## R3. Correction to an earlier result
The first-pass table in the chat used the **lattice** VaR (jumps of 1/n), which inflated the apparent
error (e.g. n=100, alpha=99%: lattice VaR 12.0% vs continuity-corrected 11.9%). Lattice and smooth
effects are both O(1/n); report both (columns `var_lattice`, `var_exact`).

## R4. Importance sampling: do NOT centre on the tail quantile
Sweep of the factor shift (n=100, 200k scenarios, ES SE averaged over 3 seeds, alpha=99.9%):
shift 0 -> 3.3e-3; -1.0 -> 6.7e-4; **-1.5 -> 4.1e-4**; -2 -> 6.4e-4; -3.09 (= z_alpha) -> 2.8e-3 (ESS 137/200k).
Default is therefore `0.5 * Phi^-1(1-alpha)`: roughly 8x lower SE than plain MC at 99.9%.

## R5. Tail dependence dominates granularity
Infinite-n VaR99.9%: Gaussian 17.6%; t-copula nu=30 -> 23.1%; nu=10 -> 32.4%; nu=5 -> 43.3%; nu=3 -> 53.2%.
Model risk from the copula choice is far larger than finite-n error for n >= 250.

## R6. Capital impact (LGD 45%, unexpected loss at 99.9%)
K_inf = 7.03% of exposure. n=100 -> 7.91% (+12.5%); n=1000 -> 7.12% (+1.3%).
(Illustrative, uses the Vasicek rho directly, not the IRB-prescribed rho(PD); IRB constants must be verified.)

## R7. Stochastic LGD
Beta LGD (mean 45%, sd 30%) leaves EL unchanged but raises the 99.9% VaR/ES (n=200, PD 3%): ES 12.5% -> 12.8%.

## Known limitations (state these in any paper)
* Grid (`data/derived/error_grid.csv`, 576 cases): for PD <= 0.5% and n <= 50, half of the cases have
  a relative limit error above 100% (worst ~950%). The 1/n correction is also least reliable there:
  worst relative residual 41% (n=25, PD 0.1%, rho 0.05, alpha 99.9%). For n >= 500 the corrected
  residual is at most 7.6% over the whole grid. The expansion is asymptotic; do not apply it blindly
  to tiny or very low-PD portfolios.
* Exact t-copula pmf uses a midpoint-quantile grid for the chi-square mixing variable (200 points):
  accurate for moderate nu, approximate for extreme tail of w. Cross-checked against simulation in tests.
* Two-factor and heterogeneous models are simulation-only (no exact law, no closed-form correction yet).
* Calibration to FRED uses loss-rate proxies; Freddie Mac integration is a documented workflow, not yet executed.
* IRB constants are from memory/public literature and must be verified against BIS before publication.
