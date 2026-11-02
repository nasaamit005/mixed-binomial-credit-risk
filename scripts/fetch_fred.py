"""Download Federal Reserve charge-off / delinquency series from FRED into data/raw/ (gitignored).

Usage: python scripts/fetch_fred.py [SERIES_ID ...]
Default series (verified to exist on FRED, Nov 2026): all loans, business, CRE, credit card.
"""
import sys

from mbcr.calibration import fetch_fred_csv

DEFAULT = ["CORALACBS", "CORBLACBS", "CORCREXFACBS", "CORCCACBS", "DRBLACBN"]

for sid in (sys.argv[1:] or DEFAULT):
    df = fetch_fred_csv(sid, force=True)
    print(f"{sid}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
