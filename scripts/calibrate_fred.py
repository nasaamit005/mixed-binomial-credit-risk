"""Illustrative calibration of (PD, rho) from Fed business-loan charge-off rates (PROXY data).

Requires data/raw/CORBLACBS.csv (run scripts/fetch_fred.py). n_eff is an ASSUMED effective
portfolio size, so the output is a sensitivity exercise, NOT an estimate of true default correlation.
"""
import numpy as np

from mbcr.calibration import fetch_fred_csv, fit_bayes, fit_mle, rates_to_pseudo_counts

series = fetch_fred_csv("CORBLACBS").iloc[:, 0]
for n_eff in (1000, 5000, 20000):
    d, n, years = rates_to_pseudo_counts(series, n_eff=n_eff)
    mle = fit_mle(d, n, start=(max(d.mean() / n_eff, 1e-3), 0.1))
    post = fit_bayes(d, n, n_iter=4000, burn=1000)
    print(f"n_eff={n_eff:>6}: years={len(years)}  MLE PD={mle['pd']:.4f} rho={mle['rho']:.3f} | "
          f"posterior rho 90% CI=({np.percentile(post['rho'], 5):.3f}, {np.percentile(post['rho'], 95):.3f})")
