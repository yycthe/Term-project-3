import os
import json
import config

def generate_markdown_report(data_info, leakage_info, results_table, best_model_metrics, top_features):
    report = f"""# Agentic Sports Win/Loss Predictor Report

## Data Overview
- Dataset Path: {config.DATA_PATH}
- Total Samples: {data_info['shape'][0]}
- Target Column: {data_info['target']}
- Date Column: {data_info['date']}

## Leakage Checks
Suspicious columns excluded:
{", ".join(leakage_info) if leakage_info else "None detected"}

## Feature Engineering Upgrades
### Why Difference Features?
To capture matchup strength directly, we compute the difference between Home and Away rolling statistics (e.g., `DIFF_ROLL_WIN_RATIO`). This allows the model to learn from the relative strength of opponents in a single feature, rather than needing to combine multiple independent team features.

### Elo Rating System
We implemented a chronological Elo rating system (`ELO_PRE`). This system calculates team strength based on game outcomes BEFORE each game, ensuring no data leakage. `ELO_DIFF` provides a powerful indicator of the expected performance gap between teams.

## Leakage & Validation
- **Season-Based Split**: Training is performed on all historical seasons, and evaluation is done on the **latest season**. This simulates a real-world forecasting scenario. (Current season: {config.NBA_CURRENT_SEASON})
- **Rolling Stats Validation**: All rolling features are computed per-team and explicitly shifted by 1 game. We verified that no current-game information is present in our features.

## Model Comparison
| Model | Iteration | Features | AUC | LogLoss | Brier Score | Accuracy |
|-------|-----------|----------|-----|---------|-------------|----------|
"""
    for row in results_table:
        feat_type = row.get('features_type', 'N/A')
        it = row.get('trial', row.get('iteration', 'N/A'))
        report += f"| {row['model']} | {it} | {feat_type} | {row['AUC']:.4f} | {row['LogLoss']:.4f} | {row['BrierScore']:.4f} | {row['Accuracy']:.4f} |\n"

    report += f"""
## Final Best Model Metrics
- **Model**: {best_model_metrics['model']}
- **AUC**: {best_model_metrics['AUC']:.4f}
- **LogLoss**: {best_model_metrics['LogLoss']:.4f}

## Top Feature Importances
{top_features.head(10).to_markdown() if top_features is not None else "N/A"}

## Calibration Notes
A calibration curve was generated for the best model. Probability calibration was performed using `CalibratedClassifierCV`.
"""
    
    report_path = os.path.join(config.OUTPUTS_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    return report_path
