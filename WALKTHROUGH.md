# Agentic NBA Predictor — Project Walkthrough (Current Version)

## 1) Goal

This project predicts NBA game outcomes **before tip-off** and outputs calibrated win probabilities.  
It is built as a leakage-aware, agent-driven pipeline instead of a single training script.

## 2) Core Pipeline

The orchestrator in `agent.py` runs a closed loop:

1. **OBSERVE**: load data, detect columns, run leakage checks.
2. **PLAN**: define objective and search space (LogLoss target + AUC floor).
3. **EXPLORE (Phase 1)**: broad search over model/config space.
4. **REFLECT + ADAPT**: analyze top trials and write narrowed policy to `outputs/policy.json`.
5. **EXPLOIT (Phase 2)**: search in narrowed space.
6. **DIAGNOSE**: run ablations if progress stalls, write `outputs/diagnostics.md`.
7. **FINALIZE**: save model artifacts and outputs for the UI.

## 3) Data + Leakage Controls

- Raw source: `nba_game_logs_combined.csv`
- Leakage defense:
  - keyword/correlation-based leakage detection (`data.py`)
  - rolling features are computed with `shift(1)` (`features.py`)
  - season-aware evaluation split (`evaluate.py`): latest season as test set

## 4) Models and Optimization

- Model families:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Optimizer:
  - Optuna with TPE sampler (`agent.py`)
- Search dimensions include:
  - `window_years`, `elo_k`, `model_type`, `strategy`, `top_k`
  - feature-type toggles for Auto_Select
  - model-specific hyperparameters

## 5) Metrics and Calibration

Primary metric is **LogLoss** with an **AUC floor constraint**.  
Also reported: AUC, Brier Score, Accuracy.  
Tree-based models use isotonic calibration via `CalibratedClassifierCV`.

## 6) Current Saved Result Snapshot

From `outputs/metrics.json`:

- Best model: `XGBoost (W=3y, K=40, S=Auto_Select)`
- AUC: `0.6738`
- LogLoss: `0.6410`
- BrierScore: `0.2249`
- Accuracy: `0.6305`
- Selected feature(s): `ELO_DIFF`
- Ensemble:
  - ensemble_auc: `0.6773`
  - ensemble_logloss: `0.6392`

## 7) How to Run

Use Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate nba-predictor
python agent.py
streamlit run app.py
```

## 8) Key Output Files

| File | Purpose |
|---|---|
| `models/best_model.pkl` | final single best model |
| `models/preprocessor.pkl` | fitted preprocessor |
| `models/top_models_ensemble.pkl` | top models used for ensemble |
| `outputs/metrics.json` | best model metrics snapshot |
| `outputs/best_config.json` | best hyperparameter/config snapshot |
| `outputs/policy.json` | EXPLORE -> EXPLOIT narrowing policy |
| `outputs/diagnostics.md` | ablation/diagnostic report |
| `outputs/report.md` | run report table + summary |
