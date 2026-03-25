# Agent Diagnostic Report
Generated: 2026-03-11T20:16:43.134299
Best LogLoss at diagnosis: 0.6410
Best AUC at diagnosis: 0.6738
Goal: LogLoss <= 0.63  (AUC floor >= 0.6)

## Data Sanity
- Train size : 3,690 matchups
- Test size  : 885 matchups
- Test home-win rate : 53.79%
- Available features : 55
- Avg NaN ratio      : 0.0000

## Ablation Studies
| Configuration | AUC | LogLoss | Brier |
|---|---|---|---|
| Diff_Only + LogReg (baseline) | 0.6595 | 0.6521 | 0.2302 |
| Auto_Select + LogReg | 0.6634 | 0.6491 | 0.2289 |
| No_Elo + RandomForest | 0.6297 | 0.6653 | 0.2366 |
| Four_Factors + RandomForest | 0.5863 | 0.6825 | 0.2445 |
| Auto_Select (Roll+Diff only) + XGBoost | 0.5944 | 0.6774 | 0.2422 |
| Auto_Select (No Roll) + XGBoost | 0.6353 | 0.6604 | 0.2344 |
| Auto_Select (All types) + RandomForest | 0.6498 | 0.6566 | 0.2321 |
| **Best Model (Agent)** | **0.6738** | **0.6410** | **0.2249** |

## Agent Recommendations
- Current best LogLoss=0.6410  AUC=0.6738
- Goal: LogLoss <= 0.63  AUC >= 0.6
- Consider: wider window, more Phase 2 budget, or threshold adjustment.
- Best config uses feature types: roll, elo
- Consider testing different feature type combinations in ablation studies.
