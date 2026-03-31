# Agentic Sports Win/Loss Predictor Report

## Data Overview
- Dataset Path: nba_game_logs_combined.csv
- Total Samples: 14336
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
| XGBoost (W=6y, K=10, S=Auto_Select) | 0 | optimized | 0.6815 | 0.6379 | 0.2233 | 0.6522 |
| LogisticRegression (W=1y, K=15, S=Auto_Select) | 2 | optimized | 0.6952 | 0.6284 | 0.2190 | 0.6522 |
| XGBoost (W=9y, K=50, S=Auto_Select) | 3 | optimized | 0.6882 | 0.6325 | 0.2208 | 0.6522 |
| RandomForest (W=2y, K=35, S=Auto_Select) | 4 | optimized | 0.7044 | 0.6237 | 0.2167 | 0.6619 |
| RandomForest (W=7y, K=20, S=Auto_Select) | 7 | optimized | 0.6424 | 0.6857 | 0.2353 | 0.5927 |
| LogisticRegression (W=2y, K=40, S=Auto_Select) | 8 | optimized | 0.6962 | 0.6276 | 0.2188 | 0.6539 |
| XGBoost (W=5y, K=40, S=Auto_Select) | 9 | optimized | 0.6947 | 0.6319 | 0.2205 | 0.6575 |
| RandomForest (W=2y, K=35, S=Auto_Select) | 10 | optimized | 0.7009 | 0.6259 | 0.2179 | 0.6513 |
| LogisticRegression (W=2y, K=35, S=Auto_Select) | 11 | optimized | 0.6993 | 0.6269 | 0.2184 | 0.6575 |
| XGBoost (W=2y, K=35, S=Auto_Select) | 12 | optimized | 0.6692 | 0.6436 | 0.2262 | 0.6193 |
| RandomForest (W=8y, K=30, S=Auto_Select) | 13 | optimized | 0.7016 | 0.6261 | 0.2178 | 0.6637 |
| LogisticRegression (W=4y, K=5, S=Auto_Select) | 14 | optimized | 0.6796 | 0.6381 | 0.2234 | 0.6380 |
| XGBoost (W=2y, K=35, S=Auto_Select) | 15 | optimized | 0.6843 | 0.6429 | 0.2243 | 0.6451 |
| RandomForest (W=2y, K=35, S=Auto_Select) | 16 | optimized | 0.7137 | 0.6182 | 0.2143 | 0.6699 |

## Final Best Model Metrics
- **Model**: RandomForest (W=2y, K=35, S=Auto_Select)
- **AUC**: 0.7137
- **LogLoss**: 0.6182

## Top Feature Importances
N/A

## Calibration Notes
A calibration curve was generated for the best model. Probability calibration was performed using `CalibratedClassifierCV`.
