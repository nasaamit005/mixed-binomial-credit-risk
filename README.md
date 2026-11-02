# mixed-binomial-credit-risk

**[Live interactive results](https://nasaamit005.github.io/mixed-binomial-credit-risk/)**

**How accurate is the Vasicek large-portfolio approximation for real, finite portfolios, and can the error be corrected analytically?**

`mbcr` computes the *exact* finite-n loss distribution of the one-factor credit model (a mixed
binomial), compares it with the infinite-granularity Vasicek limit, validates a closed-form 1/n
correction, and extends the analysis to t-copulas, two-factor and heterogeneous portfolios, stochastic
LGD, importance sampling, calibration and Basel IRB capital.

| Module (`src/mbcr/`) | What it does |
|---|---|
| `exact.py` | noise-free mixed-binomial pmf (Gaussian and t-copula); lattice and continuity-corrected VaR/ES |
| `vasicek.py` | infinite-n limit: VaR, ES, density, t-copula limit |
| `correction.py` | closed-form 1/n VaR and ES correction (derivation in `docs/DERIVATION.md`) |
| `simulation.py` | Monte Carlo + importance sampling; heterogeneous groups, sector factors, Beta LGD |
| `calibration.py` | MLE and Bayesian (Metropolis) for (PD, rho); FRED loader |
| `capital.py` | Basel IRB risk-weight function and finite-n capital gap |
| `viz.py`, `app/` | interactive Plotly figures and Streamlit dashboard |

## Quick start
```bash
pip install -e ".[dev,app]"
pytest                                # 32 tests, fixed seeds
python scripts/run_grid.py            # exact error grid -> data/derived/error_grid.csv
python scripts/make_report.py         # interactive HTML report -> reports/
streamlit run app/streamlit_app.py    # live dashboard with sliders
```

## Headline findings (details and caveats in `docs/RESULTS.md`)
* The Vasicek limit **understates** tail risk; at PD 2%, rho 0.15, n=100, 99.9% VaR is ~11% higher.
* Adding the analytic 1/n term cuts the error by 1-2 orders of magnitude (residual ~ n^-2).
* Importance sampling centred on the tail quantile *backfires*; half-quantile shift is ~8x better.
* Copula choice (t vs Gaussian) matters far more than finite-n error once n >= 250.

## Reproducibility and rigor
Global seed `SEED = 20261101`; exact results use deterministic quadrature; every claim in
`docs/RESULTS.md` is backed by a test or a script; CI runs lint + tests on Python 3.10-3.12.

## Data and licensing
See `docs/DATA.md`. Raw data is never committed. Freddie Mac data may not be redistributed.

## Honest scope
The 1/n expansion is the established granularity-adjustment framework (Gordy; Martin-Wilde), not a new
theorem. IRB constants must be verified against the BIS text. Run a literature check before claiming novelty.

