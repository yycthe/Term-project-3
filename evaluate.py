import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve
import config
import os
import logging

logger = logging.getLogger(__name__)


def _sanitize_matrix(X, clip_value=1e6):
    """Stabilize matrices before model inference."""
    try:
        from scipy import sparse
        if sparse.issparse(X):
            X = X.copy()
            X.data = np.nan_to_num(
                X.data, nan=0.0, posinf=clip_value, neginf=-clip_value
            )
            np.clip(X.data, -clip_value, clip_value, out=X.data)
            return X
    except Exception:
        pass

    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=clip_value, neginf=-clip_value)
    np.clip(X, -clip_value, clip_value, out=X)
    return X


def time_aware_split(df, date_col, train_ratio=0.8, window_years=None):
    """
    Splits data by season: latest season = test, previous seasons = train.
    The window_years parameter limits how many years of training data to use.
    The TEST set is always the latest season, regardless of window_years.
    """
    df = df.sort_values(date_col)
    
    if config.SEASON_COL in df.columns:
        seasons = sorted(df[config.SEASON_COL].unique())
        test_season = seasons[-1]
        
        train_df = df[df[config.SEASON_COL] < test_season]
        
        if window_years is not None:
            min_season = test_season - window_years
            train_df = train_df[train_df[config.SEASON_COL] >= min_season]
            logger.debug(f"Moving window: seasons {min_season}–{test_season - 1} ({window_years}y)")

        test_df = df[df[config.SEASON_COL] == test_season]
        
        logger.debug(f"Season split — Test: {test_season}, Train seasons: {sorted(train_df[config.SEASON_COL].unique())}")
    else:
        logger.debug(f"No season column. Falling back to {train_ratio:.0%} time split.")
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
    return train_df, test_df


def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
        
    metrics = {
        "AUC": auc,
        "LogLoss": log_loss(y_true, y_prob),
        "BrierScore": brier_score_loss(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred)
    }
    return metrics


def plot_calibration_curve(y_true, y_prob, model_name):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(config.OUTPUTS_DIR, f"calibration_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        return feat_imp
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        return feat_imp
    return None


def get_ensemble_probs(top_trials, df_raw, target, date_col):
    """
    Combines probabilities from the top-N trials on the TEST SET only.
    Each trial may use different features, preprocessors, and Elo K values.
    
    Returns:
        tuple: (ensemble_probs, y_test) — averaged probabilities and aligned ground truth.
               Returns (None, None) if evaluation fails.
    """
    import data
    import features
    import models
    
    all_probs = []
    y_test = None
    
    for trial_info in top_trials:
        try:
            trial_k = trial_info['config'].get('elo_k', 20)
            trial_roll = trial_info['config'].get('roll_window', 10)
            
            # Re-engineer features with the trial's specific Elo K and roll window
            df_feat_raw = features.engineer_features(
                df_raw, target, date_col, include_elo=True, elo_k=trial_k, roll_window=trial_roll
            )
            df_feat = data.create_matchup_df(df_feat_raw, target, date_col)
            df_feat = features.add_difference_features(df_feat)
            
            # FIX: Split to get ONLY test data (latest season)
            _, test_df = time_aware_split(df_feat, date_col)
            
            test_input = test_df[trial_info['features']].replace([np.inf, -np.inf], np.nan)
            X_test = trial_info['preprocessor'].transform(test_input)
            X_test = _sanitize_matrix(X_test)
            p = models.predict_proba_stable(trial_info['model'], X_test)[:, 1]
            all_probs.append(p)
            
            if y_test is None:
                y_test = test_df['TARGET'].values
                
        except Exception as e:
            logger.warning(f"Ensemble member failed during evaluation: {e}")
            continue
    
    if not all_probs:
        return None, None
    
    ensemble_p = np.mean(all_probs, axis=0)
    return ensemble_p, y_test
