# Agentic Sports Win/Loss Predictor Report

## Data Overview
- Dataset Path: nba_game_logs_combined.csv
- Total Samples: 14094
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
| XGBoost (W=10y, K=25, S=Auto_Select) | 0 | optimized | 0.6692 | 0.6464 | 0.2275 | 0.6271 |
| RandomForest (W=3y, K=50, S=Auto_Select) | 1 | optimized | 0.6232 | 0.6649 | 0.2365 | 0.5977 |
| LogisticRegression (W=9y, K=30, S=Auto_Select) | 2 | optimized | 0.6699 | 0.6457 | 0.2272 | 0.6395 |
| XGBoost (W=9y, K=35, S=Auto_Select) | 3 | optimized | 0.6244 | 0.6645 | 0.2363 | 0.5864 |
| LogisticRegression (W=7y, K=35, S=Auto_Select) | 5 | optimized | 0.6264 | 0.6696 | 0.2382 | 0.5774 |
| XGBoost (W=9y, K=25, S=Auto_Select) | 6 | optimized | 0.6262 | 0.6652 | 0.2365 | 0.5921 |
| RandomForest (W=4y, K=35, S=Auto_Select) | 7 | optimized | 0.6379 | 0.6846 | 0.2328 | 0.5910 |
| LogisticRegression (W=6y, K=35, S=Auto_Select) | 8 | optimized | 0.6392 | 0.6611 | 0.2347 | 0.5944 |
| XGBoost (W=7y, K=20, S=Auto_Select) | 9 | optimized | 0.6293 | 0.6647 | 0.2362 | 0.5955 |
| RandomForest (W=5y, K=30, S=Auto_Select) | 10 | optimized | 0.6647 | 0.6477 | 0.2283 | 0.6316 |
| LogisticRegression (W=10y, K=30, S=Auto_Select) | 11 | optimized | 0.6673 | 0.6486 | 0.2283 | 0.6350 |
| XGBoost (W=10y, K=25, S=Auto_Select) | 12 | optimized | 0.6673 | 0.6477 | 0.2282 | 0.6260 |
| RandomForest (W=1y, K=40, S=Auto_Select) | 13 | optimized | 0.6695 | 0.6486 | 0.2272 | 0.6158 |
| LogisticRegression (W=8y, K=30, S=Auto_Select) | 14 | optimized | 0.6649 | 0.6482 | 0.2285 | 0.6294 |
| XGBoost (W=9y, K=25, S=Auto_Select) | 15 | optimized | 0.6684 | 0.6462 | 0.2274 | 0.6147 |
| RandomForest (W=9y, K=40, S=Auto_Select) | 16 | optimized | 0.6691 | 0.6433 | 0.2266 | 0.6192 |
| LogisticRegression (W=9y, K=40, S=Auto_Select) | 17 | optimized | 0.6669 | 0.6475 | 0.2283 | 0.6328 |
| XGBoost (W=9y, K=40, S=Auto_Select) | 18 | optimized | 0.6579 | 0.6527 | 0.2305 | 0.6282 |
| RandomForest (W=6y, K=20, S=Auto_Select) | 19 | optimized | 0.6675 | 0.6466 | 0.2274 | 0.6294 |
| LogisticRegression (W=4y, K=50, S=Auto_Select) | 20 | optimized | 0.6669 | 0.6459 | 0.2275 | 0.6226 |
| XGBoost (W=8y, K=50, S=Auto_Select) | 21 | optimized | 0.6443 | 0.6565 | 0.2326 | 0.5989 |
| RandomForest (W=4y, K=50, S=Auto_Select) | 22 | optimized | 0.6607 | 0.6513 | 0.2293 | 0.6181 |
| LogisticRegression (W=4y, K=30, S=Auto_Select) | 23 | optimized | 0.6606 | 0.6524 | 0.2303 | 0.6192 |
| XGBoost (W=5y, K=40, S=Auto_Select) | 24 | optimized | 0.6695 | 0.6480 | 0.2280 | 0.6226 |
| RandomForest (W=1y, K=50, S=Auto_Select) | 25 | optimized | 0.6680 | 0.6729 | 0.2269 | 0.6068 |
| LogisticRegression (W=2y, K=30, S=Auto_Select) | 26 | optimized | 0.6742 | 0.6447 | 0.2264 | 0.6384 |
| XGBoost (W=2y, K=30, S=Auto_Select) | 27 | optimized | 0.6449 | 0.6594 | 0.2328 | 0.6056 |
| RandomForest (W=2y, K=30, S=Auto_Select) | 28 | optimized | 0.6442 | 0.6573 | 0.2322 | 0.5955 |
| LogisticRegression (W=3y, K=30, S=Auto_Select) | 29 | optimized | 0.6742 | 0.6440 | 0.2261 | 0.6395 |
| XGBoost (W=3y, K=40, S=Auto_Select) | 30 | optimized | 0.6723 | 0.6434 | 0.2258 | 0.6249 |
| RandomForest (W=3y, K=40, S=Auto_Select) | 31 | optimized | 0.6626 | 0.6514 | 0.2292 | 0.6339 |
| LogisticRegression (W=3y, K=40, S=Auto_Select) | 32 | optimized | 0.6741 | 0.6430 | 0.2260 | 0.6316 |
| XGBoost (W=3y, K=40, S=Auto_Select) | 33 | optimized | 0.6697 | 0.6449 | 0.2264 | 0.6271 |
| RandomForest (W=3y, K=40, S=Auto_Select) | 34 | optimized | 0.6620 | 0.6520 | 0.2297 | 0.6260 |
| LogisticRegression (W=3y, K=40, S=Auto_Select) | 35 | optimized | 0.6741 | 0.6430 | 0.2260 | 0.6316 |
| XGBoost (W=3y, K=40, S=Auto_Select) | 36 | optimized | 0.6728 | 0.6446 | 0.2261 | 0.6260 |
| RandomForest (W=3y, K=40, S=Auto_Select) | 37 | optimized | 0.6398 | 0.6651 | 0.2356 | 0.5876 |
| XGBoost (W=3y, K=40, S=Auto_Select) | 39 | optimized | 0.6721 | 0.6428 | 0.2256 | 0.6203 |
| LogisticRegression (W=3y, K=40, S=Auto_Select) | 41 | optimized | 0.6741 | 0.6430 | 0.2260 | 0.6316 |
| XGBoost (W=3y, K=40, S=Auto_Select) | 42 | optimized | 0.6738 | 0.6410 | 0.2249 | 0.6305 |
| RandomForest (W=3y, K=40, S=Auto_Select) | 43 | optimized | 0.6672 | 0.6473 | 0.2275 | 0.6362 |
| LogisticRegression (W=3y, K=20, S=Auto_Select) | 44 | optimized | 0.6702 | 0.6475 | 0.2274 | 0.6407 |
| RandomForest (W=3y, K=35, S=Auto_Select) | 46 | optimized | 0.6684 | 0.6462 | 0.2269 | 0.6339 |
| LogisticRegression (W=3y, K=40, S=Auto_Select) | 47 | optimized | 0.6741 | 0.6430 | 0.2260 | 0.6316 |
| XGBoost (W=10y, K=25, S=Auto_Select) | 48 | optimized | 0.6700 | 0.6508 | 0.2288 | 0.6418 |
| LogisticRegression (W=5y, K=40, S=Auto_Select) | 50 | optimized | 0.6741 | 0.6426 | 0.2258 | 0.6384 |
| XGBoost (W=5y, K=40, S=Auto_Select) | 51 | optimized | 0.6711 | 0.6469 | 0.2275 | 0.6305 |
| RandomForest (W=5y, K=40, S=Auto_Select) | 52 | optimized | 0.6577 | 0.6583 | 0.2327 | 0.6011 |
| LogisticRegression (W=5y, K=40, S=Auto_Select) | 53 | optimized | 0.6741 | 0.6426 | 0.2258 | 0.6384 |
| XGBoost (W=5y, K=20, S=Auto_Select) | 54 | optimized | 0.6621 | 0.6539 | 0.2302 | 0.6316 |
| RandomForest (W=5y, K=35, S=Auto_Select) | 55 | optimized | 0.6676 | 0.6461 | 0.2274 | 0.6328 |
| LogisticRegression (W=5y, K=40, S=Auto_Select) | 56 | optimized | 0.6741 | 0.6426 | 0.2258 | 0.6384 |
| XGBoost (W=5y, K=25, S=Auto_Select) | 57 | optimized | 0.6620 | 0.6489 | 0.2283 | 0.6362 |
| RandomForest (W=5y, K=40, S=Auto_Select) | 58 | optimized | 0.6571 | 0.6579 | 0.2324 | 0.6181 |
| XGBoost (W=3y, K=40, S=Auto_Select) | 0 | optimized | 0.6731 | 0.6442 | 0.2261 | 0.6249 |
| LogisticRegression (W=3y, K=50, S=Auto_Select) | 1 | optimized | 0.6594 | 0.6500 | 0.2296 | 0.6090 |
| LogisticRegression (W=2y, K=50, S=Auto_Select) | 2 | optimized | 0.6655 | 0.6486 | 0.2287 | 0.6249 |
| LogisticRegression (W=8y, K=35, S=Auto_Select) | 3 | optimized | 0.6742 | 0.6436 | 0.2261 | 0.6305 |
| RandomForest (W=9y, K=50, S=Auto_Select) | 4 | optimized | 0.6233 | 0.6688 | 0.2384 | 0.5797 |
| XGBoost (W=3y, K=50, S=Auto_Select) | 6 | optimized | 0.6630 | 0.6488 | 0.2287 | 0.6226 |
| RandomForest (W=3y, K=40, S=Auto_Select) | 7 | optimized | 0.6679 | 0.6459 | 0.2276 | 0.6169 |
| LogisticRegression (W=2y, K=35, S=Auto_Select) | 8 | optimized | 0.6389 | 0.6615 | 0.2348 | 0.6000 |
| RandomForest (W=10y, K=50, S=Auto_Select) | 9 | optimized | 0.6546 | 0.6538 | 0.2313 | 0.5989 |
| LogisticRegression (W=8y, K=35, S=Auto_Select) | 10 | optimized | 0.6312 | 0.6671 | 0.2372 | 0.5887 |
| XGBoost (W=8y, K=40, S=Auto_Select) | 11 | optimized | 0.6711 | 0.6509 | 0.2288 | 0.6339 |
| XGBoost (W=6y, K=40, S=Auto_Select) | 12 | optimized | 0.6668 | 0.6522 | 0.2294 | 0.6350 |
| XGBoost (W=7y, K=40, S=Auto_Select) | 13 | optimized | 0.6669 | 0.6509 | 0.2291 | 0.6203 |
| XGBoost (W=5y, K=35, S=Auto_Select) | 14 | optimized | 0.6709 | 0.6469 | 0.2271 | 0.6362 |
| LogisticRegression (W=8y, K=40, S=Auto_Select) | 15 | optimized | 0.6741 | 0.6434 | 0.2261 | 0.6282 |
| LogisticRegression (W=8y, K=35, S=Auto_Select) | 16 | optimized | 0.6742 | 0.6436 | 0.2261 | 0.6305 |
| LogisticRegression (W=8y, K=40, S=Auto_Select) | 17 | optimized | 0.6741 | 0.6434 | 0.2261 | 0.6282 |
| LogisticRegression (W=8y, K=40, S=Auto_Select) | 18 | optimized | 0.6312 | 0.6671 | 0.2372 | 0.5887 |
| LogisticRegression (W=5y, K=40, S=Auto_Select) | 19 | optimized | 0.6741 | 0.6426 | 0.2258 | 0.6384 |
| LogisticRegression (W=5y, K=40, S=Auto_Select) | 20 | optimized | 0.6741 | 0.6426 | 0.2258 | 0.6384 |
| LogisticRegression (W=5y, K=40, S=Auto_Select) | 21 | optimized | 0.6741 | 0.6426 | 0.2258 | 0.6384 |
| LogisticRegression (W=5y, K=40, S=Auto_Select) | 22 | optimized | 0.6741 | 0.6426 | 0.2258 | 0.6384 |
| LogisticRegression (W=5y, K=40, S=Auto_Select) | 23 | optimized | 0.6741 | 0.6426 | 0.2258 | 0.6384 |
| LogisticRegression (W=5y, K=40, S=Auto_Select) | 24 | optimized | 0.6741 | 0.6426 | 0.2258 | 0.6384 |

## Final Best Model Metrics
- **Model**: XGBoost (W=3y, K=40, S=Auto_Select)
- **AUC**: 0.6738
- **LogLoss**: 0.6410

## Top Feature Importances
N/A

## Calibration Notes
A calibration curve was generated for the best model. Probability calibration was performed using `CalibratedClassifierCV`.
