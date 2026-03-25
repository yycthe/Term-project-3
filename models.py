from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
import numpy as np
import warnings
import config

def get_baseline_model():
    # Stronger regularization reduces extreme coefficients on noisy sports features.
    return LogisticRegression(
        max_iter=2000,
        solver='liblinear',
        C=0.1,
        random_state=config.RANDOM_STATE,
    )

def get_random_forest_model():
    return RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)

def get_xgboost_model():
    """Create XGBoost model - always uses CPU to avoid device mismatch warnings."""
    return XGBClassifier(
        eval_metric='logloss', 
        tree_method='hist',     # Use histogram-based algorithm (CPU)
        device='cpu',           # Always use CPU
        random_state=config.RANDOM_STATE
    )

def get_xgboost_device():
    """Always return 'cpu' - force CPU usage everywhere."""
    return 'cpu'

def get_strong_models():
    models = {
        "RandomForest": get_random_forest_model(),
        "XGBoost": get_xgboost_model()
    }
    return models

def run_hyperparameter_search(model, X_train, y_train):
    """Expanded hyperparameter search."""
    if isinstance(model, XGBClassifier):
        param_dist = {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
    elif isinstance(model, RandomForestClassifier):
        param_dist = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    else:
        return model

    search = RandomizedSearchCV(
        model, param_distributions=param_dist, 
        n_iter=10, cv=3, scoring='roc_auc', 
        random_state=config.RANDOM_STATE, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def _sanitize_matrix(X, clip_value=1e3):
    """Ensure matrix has finite values with bounded magnitude."""
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

def _has_matmul_runtime_warning(caught_warnings):
    terms = (
        "divide by zero encountered in matmul",
        "overflow encountered in matmul",
        "invalid value encountered in matmul",
    )
    for w in caught_warnings:
        if issubclass(w.category, RuntimeWarning):
            msg = str(w.message).lower()
            if any(t in msg for t in terms):
                return True
    return False

def _is_finite_array(arr):
    try:
        return np.isfinite(arr).all()
    except Exception:
        return False

def _is_valid_probability_array(arr):
    try:
        a = np.asarray(arr, dtype=float)
        if a.ndim != 2 or a.shape[1] < 2:
            return False
        if not np.isfinite(a).all():
            return False
        return (a >= -1e-9).all() and (a <= 1.0 + 1e-9).all()
    except Exception:
        return False

def predict_proba_stable(model, X, clip_value=1e3):
    """
    Predict probabilities with runtime-warning shielding.
    Raises FloatingPointError if prediction remains numerically unstable.
    """
    X_clean = _sanitize_matrix(X, clip_value=clip_value)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        probs = model.predict_proba(X_clean)
    had_warning = _has_matmul_runtime_warning(caught)
    bad = not _is_valid_probability_array(probs)

    # Retry on warnings or bad outputs, but only fail if outputs remain invalid.
    if had_warning or bad:
        X_retry = _sanitize_matrix(X_clean, clip_value=100.0)
        with warnings.catch_warnings(record=True) as caught_retry:
            warnings.simplefilter("always", RuntimeWarning)
            probs = model.predict_proba(X_retry)
        bad_retry = not _is_valid_probability_array(probs)
        if bad_retry:
            raise FloatingPointError("Numerically unstable predict_proba (matmul warning or non-finite output)")

    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    row_sum = probs.sum(axis=1, keepdims=True)
    probs = probs / np.where(row_sum == 0, 1.0, row_sum)
    return probs

def calibrate_model(model, X_train, y_train):
    X_clean = _sanitize_matrix(X_train, clip_value=1e3)
    y_clean = np.asarray(y_train, dtype=float)
    mask = np.isfinite(y_clean)
    X_clean = X_clean[mask]
    y_clean = y_clean[mask]

    # LogisticRegression is already probabilistic; extra sigmoid calibration often
    # triggers numeric instability on near-separable folds.
    if isinstance(model, LogisticRegression):
        attempts = [
            {"solver": "liblinear", "C": 0.1},
            {"solver": "liblinear", "C": 0.03},
            {"solver": "liblinear", "C": 0.01},
            {"solver": "lbfgs", "C": 0.01},
        ]
        for a in attempts:
            candidate = LogisticRegression(
                max_iter=2000,
                solver=a["solver"],
                C=a["C"],
                random_state=config.RANDOM_STATE,
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always", RuntimeWarning)
                candidate.fit(X_clean, y_clean)
            coef_ok = (
                hasattr(candidate, "coef_")
                and np.isfinite(candidate.coef_).all()
                and np.abs(candidate.coef_).max() < 1e6
            )
            if not coef_ok:
                continue
            try:
                probe_n = min(512, X_clean.shape[0]) if hasattr(X_clean, "shape") else 0
                if probe_n > 0:
                    _ = predict_proba_stable(candidate, X_clean[:probe_n], clip_value=100.0)
                return candidate
            except FloatingPointError:
                continue

        # Last-resort safe fallback to avoid trial crashes.
        fallback = DummyClassifier(strategy="prior")
        fallback.fit(X_clean, y_clean)
        return fallback

    # Prefer isotonic to avoid extra logistic optimization instability inside sigmoid calibration.
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        calibrated.fit(X_clean, y_clean)

    if _has_matmul_runtime_warning(caught):
        # Retry with stricter clipping.
        X_retry = _sanitize_matrix(X_clean, clip_value=100.0)
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            calibrated.fit(X_retry, y_clean)
        X_clean = X_retry

    try:
        probe_n = min(512, X_clean.shape[0]) if hasattr(X_clean, "shape") else 0
        if probe_n > 0:
            _ = predict_proba_stable(calibrated, X_clean[:probe_n], clip_value=100.0)
    except FloatingPointError:
        fallback = DummyClassifier(strategy="prior")
        fallback.fit(X_clean, y_clean)
        return fallback

    return calibrated
