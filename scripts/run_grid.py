"""Full exact-error grid -> data/derived/error_grid.csv (safe to publish; synthetic)."""
from pathlib import Path

from mbcr.analysis import error_grid

df = error_grid(n_list=[25, 50, 100, 250, 500, 1000, 2500, 5000],
                pd_list=[0.001, 0.005, 0.01, 0.02, 0.05, 0.10],
                rho_list=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
                alphas=(0.99, 0.999))
out = Path("data/derived/error_grid.csv")
df.to_csv(out, index=False)
print(f"wrote {out} ({len(df)} rows)")
print(df.groupby("n")[["var_rel_err_limit", "var_rel_err_corr"]].agg(["mean", "max"]).round(4))
