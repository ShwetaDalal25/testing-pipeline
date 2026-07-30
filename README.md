# Timber Forecast Uncertainty Dashboard

Live app: https://timber-forecast-uncertainty-dashboard-shwetadalal.streamlit.app

This project is an interactive Streamlit dashboard showing how uncertainty in forest inventory inputs, such as top height and DBH, can propagate into timber-volume forecasts using Monte Carlo simulation.

The dashboard uses synthetic public-safe data for a Grand Est-style forest region in France. It demonstrates a simplified deterministic timber forecast, then adds input uncertainty to produce forecast bands for Norway spruce, oak, and total standing timber volume.

## What the dashboard demonstrates

- Synthetic forest input table with forest units, species, area, planting year, yield class, top height, DBH and starting volume.
- Deterministic timber-volume forecast.
- Monte Carlo propagation of uncertainty in top height and DBH.
- Comparison of deterministic forecast curves and uncertainty bands for Norway spruce, oak and total regional volume.
- Reproducible app packaging with Streamlit and GitHub Actions

## Project structure

```text
.
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── .streamlit/config.toml
├── .github/workflows/ci.yml
├── tests/test_app_static.py
└── README.md
```

## Run locally

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
streamlit run app.py
```

The app should open at `http://localhost:8501`.


## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Select:
   - Repository: this project repository
   - Branch: `main`
   - Main file path: `app.py`
5. Deploy.

Streamlit Community Cloud uses the GitHub repository and `requirements.txt`.

## Continuous integration

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on pushes and pull requests to `main`.

It performs:

1. Python dependency installation.
2. Python syntax compilation.
3. Static tests with `pytest`.

This gives the repository visible evidence of basic CI and containerisation.

## Portfolio/CV wording

> Built and deployed a Streamlit dashboard for timber-forecast uncertainty propagation using synthetic forest inventory inputs, Monte Carlo simulation and GitHub Actions CI checks.

## Notes

All values, regions and tree groups are synthetic and designed for demonstration only.
