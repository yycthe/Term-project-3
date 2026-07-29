# NBA Game Predictor

A data-science pipeline and Streamlit application that estimates the home team's win probability for NBA games. The project combines historical game data, leakage-aware feature engineering, Elo ratings, rolling team statistics, Optuna model search, probability calibration, and prediction tracking.

## Features

- Trains Logistic Regression, Random Forest, and XGBoost candidates
- Optimizes primarily for log loss with an AUC floor
- Searches rolling windows, Elo settings, history windows, and feature counts
- Builds ensemble and preprocessing artifacts
- Displays upcoming games and calibrated win probabilities in Streamlit
- Stores prediction history and settles completed games
- Supports optional Firestore persistence for cloud deployments
- Generates metrics, diagnostics, policy, and experiment reports

## Project layout

| File | Purpose |
| --- | --- |
| `agent.py` | Multi-phase Optuna training and search policy |
| `features.py` | Leakage-aware feature engineering and Elo/rolling statistics |
| `models.py` | Model construction, calibration, and evaluation |
| `evaluate.py` | Holdout evaluation |
| `nba_fetch.py` | Teams, schedules, results, and settlement data |
| `app.py` | Streamlit prediction interface |
| `storage.py` | Local or Firestore prediction persistence |
| `report.py` | Experiment report generation |
| `config.py` | Paths, thresholds, search space, and runtime settings |

Pretrained artifacts are stored in `models/`; metrics and diagnostics are stored in `outputs/`.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Retrain and evaluate

```bash
python agent.py
python evaluate.py
python report.py
```

Training can be computationally expensive: the configured two-phase search allows up to 500 trials.

## Optional Firebase persistence

Set `USE_FIREBASE=1` and provide:

- `FIREBASE_PROJECT_ID`
- either `FIREBASE_SERVICE_ACCOUNT_JSON`
- or `FIREBASE_SERVICE_ACCOUNT_B64`

Without Firebase, predictions and agent state use local files under `outputs/`.

## Methodology notes

The pipeline removes post-game leakage signals, uses time-aware historical windows, and evaluates probability quality rather than accuracy alone. Reported performance is historical and does not guarantee future results.

## Disclaimer

For educational and research use only. Predictions are not financial or betting advice.
