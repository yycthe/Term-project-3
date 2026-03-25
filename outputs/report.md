# Agentic Sports Win/Loss Predictor Report

## Data Overview
- Dataset Path: nba_game_logs_combined.csv
- Total Samples: 14288
- Target Column: WIN
- Date Column: GAME_DATE

## Leakage Checks
Suspicious columns excluded:
WL, W, L, W_PCT, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PTS

## Feature Engineering Upgrades
### Why Difference Features?
To capture matchup strength directly, we compute the difference between Home and Away rolling statistics (e.g., `DIFF_ROLL_WIN_RATIO`). This allows the model to learn from the relative strength of opponents in a single feature, rather than needing to combine multiple independent team features.

### Elo Rating System
We implemented a chronological Elo rating system (`ELO_PRE`). This system calculates team strength based on game outcomes BEFORE each game, ensuring no data leakage. `ELO_DIFF` provides a powerful indicator of the expected performance gap between teams.

## Leakage & Validation
- **Season-Based Split**: Training is performed on all historical seasons, and evaluation is done on the **latest season**. This simulates a real-world forecasting scenario. (Current season: 2025-26)
- **Rolling Stats Validation**: All rolling features are computed per-team and explicitly shifted by 1 game. We verified that no current-game information is present in our features.

## Model Comparison
| Model | Iteration | Features | AUC | LogLoss | Brier Score | Accuracy |
|-------|-----------|----------|-----|---------|-------------|----------|
| XGBoost (W=8y, K=50, S=Auto_Select) | 0 | optimized | 0.6926 | 0.6293 | 0.2199 | 0.6450 |

## Final Best Model Metrics
- **Model**: XGBoost (W=8y, K=50, S=Auto_Select)
- **AUC**: 0.6926
- **LogLoss**: 0.6293

## Top Feature Importances
N/A

## Calibration Notes
A calibration curve was generated for the best model. Probability calibration was performed using `CalibratedClassifierCV`.
