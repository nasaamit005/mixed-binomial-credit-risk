.PHONY: install test lint grid report app
install: ; pip install -e ".[dev,app]"
test: ; pytest
lint: ; ruff check src tests scripts
grid: ; python scripts/run_grid.py
report: ; python scripts/make_report.py
app: ; streamlit run app/streamlit_app.py
