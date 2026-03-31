"""
Agentic NBA Game Predictor
==========================
A truly autonomous ML agent that predicts NBA game outcomes.

Unlike a simple optimisation script, this agent operates in a closed loop
where *reflection drives strategy updates* and *diagnostics are first-class
actions* — not just parameter tuning.

Agent Loop
----------
    OBSERVE  →  PLAN  →  PHASE 1 (Explore)
                              ↓
                         REFLECT + ADAPT  →  generate policy.json
                              ↓
                         PHASE 2 (Exploit, policy-driven)
                              ↓
                         DIAGNOSE (if stuck)
                              ↓
                         FINALIZE  →  save artefacts + memory

Agentic evidence (auditable):
  1. Reflect → Adapt: Phase 2 search space is *provably narrowed* from Phase 1
     analysis.  policy.json records the before/after with reasoning.
  2. Multiple action types: the agent can TRAIN, REFLECT, DIAGNOSE, and ADAPT
     — not just call Optuna in a loop.
  3. Cross-run memory: agent_memory.json stores policy + best config; the next
     run warm-starts and uses the prior policy to seed Phase 1.
  4. Error resilience: per-trial exception handling.
  5. Transparent reasoning: every decision is logged.
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import optuna
import logging
from datetime import datetime

import config
import data
import features
import models
import evaluate
import report
import storage

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Agent")


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """Autonomous ML agent for NBA game-outcome prediction."""

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self):
        # ML state
        self.best_model = None
        self.best_metrics = {"AUC": 0, "LogLoss": 999}
        self.best_feature_names = []
        self.best_config = {}
        self.best_preprocessor = None
        self.top_trials = []
        self.experiment_results = []

        # Agent state
        self.total_trials = 0
        self.start_time = None
        self.improvement_log = []
        self.memory = {}
        self.policy = None                 # narrowed search space for Phase 2
        self._improved_this_trial = False  # per-trial flag

    # ── Utility ───────────────────────────────────────────────────────────────

    def _say(self, phase, msg):
        elapsed = ""
        if self.start_time:
            secs = time.time() - self.start_time
            m, s = divmod(int(secs), 60)
            elapsed = f" [{m:02d}:{s:02d}]"
        print(f"  [{phase:<8}]{elapsed}  {msg}")

    # ── Memory (cross-run persistence) ────────────────────────────────────────

    def _load_memory(self):
        try:
            self.memory = storage.load_agent_memory()
            return self.memory if isinstance(self.memory, dict) else {}
        except Exception:
            return {}

    def _save_memory(self):
        mem = {
            "last_run": datetime.now().isoformat(),
            "total_trials_this_run": self.total_trials,
            "total_trials_ever": self.memory.get("total_trials_ever", 0) + self.total_trials,
            "best_auc_ever": max(
                self.best_metrics.get("AUC", 0),
                self.memory.get("best_auc_ever", 0),
            ),
            "best_logloss_ever": min(
                self.best_metrics.get("LogLoss", 999),
                self.memory.get("best_logloss_ever", 999),
            ),
            "best_config": self.best_config,
            "improvement_log": self.improvement_log,
            "top_3_aucs": [t["auc"] for t in self.top_trials],
            "top_3_models": [t.get("model_type", "?") for t in self.top_trials],
            "policy": self.policy,  # save exploited policy for next run
        }
        storage.save_agent_memory(mem)

    # ── Search-space helpers ──────────────────────────────────────────────────

    @staticmethod
    def _default_search_space():
        """Full, wide search space (Phase 1 defaults)."""
        return {
            "window_years": list(config.SEARCH_WINDOW_YEARS),
            "elo_k": list(config.SEARCH_ELO_K),
            "model_types": list(config.SEARCH_MODELS),
            "strategies": list(config.SEARCH_FEATURE_STRATEGIES),
            "top_k": list(config.SEARCH_TOP_K),
        }

    @staticmethod
    def _narrow_range(original_sorted, top_values, buffer=1):
        """Narrow a sorted list to the range observed in *top_values* +/- buffer steps."""
        if not top_values or not original_sorted:
            return list(original_sorted)
        orig = sorted(original_sorted)
        v_min, v_max = min(top_values), max(top_values)
        indices = [i for i, v in enumerate(orig) if v_min <= v <= v_max]
        if not indices:
            return list(orig)
        lo = max(0, min(indices) - buffer)
        hi = min(len(orig) - 1, max(indices) + buffer)
        return orig[lo : hi + 1]

    @staticmethod
    def _space_summary(space):
        return (
            f"W:{len(space['window_years'])} | "
            f"K:{len(space['elo_k'])} | "
            f"M:{len(space['model_types'])} | "
            f"Top:{len(space['top_k'])}"
        )

    # ── ML helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_matrix(X, clip_value=1e6):
        """
        Make transformed matrices numerically stable for downstream linear algebra:
        - replace NaN/Inf
        - clip extreme magnitudes
        """
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

    @staticmethod
    def _filter_features_by_type(all_features, params):
        """
        Filter features by type based on Optuna parameters.
        Allows free combination of feature types:
        - ROLL_* features (rolling statistics)
        - DIFF_* features (difference features)
        - RATIO_* features (ratio features)
        - ELO_* features (Elo ratings)
        - HOME_*/AWAY_* features (individual team features)
        """
        filtered = []
        
        for feat in all_features:
            # Check each feature type
            is_diff = feat.startswith("DIFF_")
            is_ratio = feat.startswith("RATIO_")
            # Keep roll detection strict so DIFF_ROLL_* is governed by DIFF toggle.
            is_roll = (
                feat.startswith("ROLL_")
                or feat.startswith("HOME_ROLL_")
                or feat.startswith("AWAY_ROLL_")
            )
            is_elo = "ELO" in feat.upper()
            is_home_away = feat.startswith("HOME_") or feat.startswith("AWAY_")
            
            # Apply filters
            use_feat = True
            
            if params.get("use_roll_features", True) is False and is_roll:
                use_feat = False
            if params.get("use_diff_features", True) is False and is_diff:
                use_feat = False
            if params.get("use_ratio_features", True) is False and is_ratio:
                use_feat = False
            if params.get("use_elo_features", True) is False and is_elo:
                use_feat = False
            if params.get("use_home_away_features", True) is False and is_home_away and not (is_diff or is_ratio):
                use_feat = False
            
            # Always keep essential features
            if feat in ("HOME_IS_HOME_CALC", "TARGET", "GAME_ID"):
                use_feat = True
            
            if use_feat:
                filtered.append(feat)
        
        return filtered
    
    def _auto_select_features(self, X_train, y_train, top_k):
        """
        Advanced automatic feature selection using multiple methods:
        1. RandomForest feature importance
        2. XGBoost feature importance
        3. Correlation with target
        4. Aggregate scores and select top_k features
        """
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        
        # Fill NaN values for model training
        X_filled = X_train.fillna(0)
        
        # Remove constant features (zero variance) to avoid correlation warnings
        variances = X_filled.var()
        non_constant_cols = [col for col in X_train.columns if variances.get(col, 0) > 1e-8]
        
        if not non_constant_cols:
            # If all features are constant, return first top_k features
            return X_train.columns[:top_k].tolist() if len(X_train.columns) >= top_k else X_train.columns.tolist()
        
        X_non_constant = X_filled[non_constant_cols]
        
        # Method 1: RandomForest importance
        rf_sel = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            random_state=config.RANDOM_STATE, n_jobs=-1,
        )
        rf_sel.fit(X_non_constant, y_train)
        rf_imp = pd.Series(rf_sel.feature_importances_, index=non_constant_cols, dtype=float)
        # Add zero importance for constant features
        rf_imp_full = pd.Series(0.0, index=X_train.columns, dtype=float)
        rf_imp_full[non_constant_cols] = rf_imp
        
        # Method 2: XGBoost importance
        try:
            xgb_sel = XGBClassifier(
                n_estimators=50, max_depth=5,
                eval_metric='logloss', tree_method='hist', device='cpu',
                random_state=config.RANDOM_STATE, n_jobs=-1,
            )
            xgb_sel.fit(X_non_constant, y_train)
            xgb_imp = pd.Series(xgb_sel.feature_importances_, index=non_constant_cols, dtype=float)
            # Add zero importance for constant features
            xgb_imp_full = pd.Series(0.0, index=X_train.columns, dtype=float)
            xgb_imp_full[non_constant_cols] = xgb_imp
        except Exception:
            # Fallback if XGBoost fails
            xgb_imp_full = pd.Series(0.0, index=X_train.columns, dtype=float)
        
        # Method 3: Correlation with target (absolute value) - only for non-constant features
        try:
            y_series = pd.Series(y_train, index=X_non_constant.index, dtype=float)
            corr_scores = X_non_constant.corrwith(y_series).abs()
            corr_scores = corr_scores.fillna(0.0).astype(float)
            # Add zero correlation for constant features
            corr_scores_full = pd.Series(0.0, index=X_train.columns, dtype=float)
            corr_scores_full[non_constant_cols] = corr_scores
        except Exception:
            # Fallback if correlation calculation fails
            corr_scores_full = pd.Series(0.0, index=X_train.columns, dtype=float)
        
        # Normalize all scores to [0, 1] for fair aggregation
        rf_max = rf_imp_full.max()
        rf_min = rf_imp_full.min()
        rf_norm = (rf_imp_full - rf_min) / (rf_max - rf_min + 1e-8) if rf_max > rf_min else pd.Series(0.0, index=X_train.columns, dtype=float)
        
        xgb_max = xgb_imp_full.max()
        xgb_min = xgb_imp_full.min()
        xgb_norm = (xgb_imp_full - xgb_min) / (xgb_max - xgb_min + 1e-8) if xgb_max > xgb_min else pd.Series(0.0, index=X_train.columns, dtype=float)
        
        corr_max = corr_scores_full.max()
        corr_min = corr_scores_full.min()
        corr_norm = (corr_scores_full - corr_min) / (corr_max - corr_min + 1e-8) if corr_max > corr_min else pd.Series(0.0, index=X_train.columns, dtype=float)
        
        # Aggregate: weighted average (RF: 40%, XGB: 40%, Corr: 20%)
        combined_score = 0.4 * rf_norm + 0.4 * xgb_norm + 0.2 * corr_norm
        
        # Select top_k features
        selected = combined_score.sort_values(ascending=False).head(top_k).index.tolist()
        
        return selected

    @staticmethod
    def _get_feature_columns(df, matchup_target, original_target):
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {
            matchup_target, "GAME_ID", "TARGET",
            f"HOME_{original_target}", f"AWAY_{original_target}",
            "EFG_PCT", "TOV_PCT", "FT_RATE",
            "HOME_EFG_PCT", "HOME_TOV_PCT", "HOME_FT_RATE",
            "AWAY_EFG_PCT", "AWAY_TOV_PCT", "AWAY_FT_RATE",
            "SEASON_START_YEAR", "MIN", "HOME_MIN", "AWAY_MIN",
        }
        return [c for c in numeric if c not in exclude and not c.endswith("_ID")]

    @staticmethod
    def _diverse_top_trials(trials, n=3):
        """
        Select the best trial per model type first, then fill remaining
        slots by overall LogLoss.  Guarantees model diversity in ensemble.
        """
        sorted_all = sorted(trials, key=lambda x: x["logloss"])
        selected = []
        seen_types = set()

        # Round 1: best of each model type
        for t in sorted_all:
            mt = t.get("model_type", "?")
            if mt not in seen_types:
                selected.append(t)
                seen_types.add(mt)
            if len(selected) >= n:
                break

        # Round 2: fill remaining by overall LogLoss
        if len(selected) < n:
            for t in sorted_all:
                if t not in selected:
                    selected.append(t)
                if len(selected) >= n:
                    break

        return selected[:n]

    @staticmethod
    def _build_model(params):
        mt = params["model_type"]
        mp = params.get("model_params", {})
        if mt == "XGBoost":
            m = models.get_xgboost_model()  # Already uses device='cpu'
            m.set_params(**mp)
            # Ensure device is always CPU (in case mp accidentally contains device)
            m.set_params(device='cpu')
        elif mt == "RandomForest":
            m = models.get_random_forest_model()
            m.set_params(**mp)
        else:
            m = models.get_baseline_model()
        return m

    def _quick_eval(self, df_clean, target, date_col,
                    strategy, model_type, window_years=5, elo_k=20, top_k=20,
                    feat_type_params=None):
        """Run one fast evaluation with fixed params (used by diagnostics)."""
        df_f = features.engineer_features(
            df_clean, target, date_col, include_elo=True, elo_k=elo_k,
        )
        df_f = data.create_matchup_df(df_f, target, date_col)
        df_f = features.add_difference_features(df_f)
        mt = "TARGET"
        feat = self._get_feature_columns(df_f, mt, target)

        train_df, test_df = evaluate.time_aware_split(df_f, date_col, window_years=window_years)

        if strategy == "Auto_Select":
            # Step 1: Filter features by type if feat_type_params provided
            if feat_type_params:
                feat = self._filter_features_by_type(feat, feat_type_params)
            # Step 2: Apply Auto_Select
            feat = self._auto_select_features(train_df[feat].fillna(0), train_df[mt], top_k)
        elif strategy == "Diff_Only":
            feat = [c for c in feat
                    if c.startswith("DIFF_") or c.startswith("RATIO_")
                    or c in ("ELO_DIFF", "HOME_IS_HOME_CALC")]
        elif strategy == "Four_Factors_Only":
            ff = config.FOUR_FACTORS_LIST
            feat = [c for c in feat if any(f in c for f in ff) or c == "HOME_IS_HOME_CALC"]
        elif strategy == "No_Elo":
            feat = [c for c in feat if "ELO" not in c.upper()]

        if not feat:
            return {"AUC": 0.5, "LogLoss": 9.0, "BrierScore": 0.5, "Accuracy": 0.5}

        # Remove constant features (zero variance) before scaling to avoid warnings
        train_feat_df = train_df[feat].fillna(0)
        variances = train_feat_df.var()
        non_constant_feat = [f for f in feat if variances.get(f, 0) > 1e-8]
        
        if not non_constant_feat:
            return {"AUC": 0.5, "LogLoss": 9.0, "BrierScore": 0.5, "Accuracy": 0.5}
        
        feat = non_constant_feat

        pre = features.get_preprocessing_pipeline(feat, [])
        train_input = train_df[feat].replace([np.inf, -np.inf], np.nan)
        test_input = test_df[feat].replace([np.inf, -np.inf], np.nan)
        X_tr = pre.fit_transform(train_input)
        X_te = pre.transform(test_input)
        X_tr = self._sanitize_matrix(X_tr)
        X_te = self._sanitize_matrix(X_te)

        mdl = (models.get_xgboost_model() if model_type == "XGBoost"
               else models.get_random_forest_model() if model_type == "RandomForest"
               else models.get_baseline_model())
        mdl = models.calibrate_model(mdl, X_tr, train_df[mt])
        try:
            probs = models.predict_proba_stable(mdl, X_te)[:, 1]
        except FloatingPointError:
            return {"AUC": 0.5, "LogLoss": 9.0, "BrierScore": 0.5, "Accuracy": 0.5}
        return evaluate.calculate_metrics(test_df[mt], probs)

    # ══════════════════════════════════════════════════════════════════════════
    #  OBSERVE
    # ══════════════════════════════════════════════════════════════════════════

    def _observe(self):
        print()
        print("=" * 62)
        print("   AGENTIC NBA PREDICTOR")
        print("=" * 62)

        # ── Auto-fetch latest games from NBA API (date-aware) ─────────────────
        from datetime import date
        today = date.today().isoformat()
        self._say("OBSERVE", f"Today: {today} — checking for new games ...")
        try:
            import nba_fetch
            result = nba_fetch.fetch_latest_games(config.DATA_PATH)
            if isinstance(result, tuple):
                n_new, latest_date = result[0], result[1]
            else:
                n_new, latest_date = result, None
            if latest_date is not None:
                try:
                    latest_str = latest_date.date().isoformat() if hasattr(latest_date, "date") else str(latest_date)
                except Exception:
                    latest_str = str(latest_date)
                if n_new > 0:
                    self._say("OBSERVE", f"Added {n_new} new game rows — data now through {latest_str}")
                else:
                    self._say("OBSERVE", f"No new games. Data is current through {latest_str}")
            else:
                if n_new > 0:
                    self._say("OBSERVE", f"Added {n_new} new game rows from NBA API")
                else:
                    self._say("OBSERVE", "NBA data check skipped or no new data (see logs)")
        except Exception as e:
            self._say("OBSERVE", f"NBA fetch skipped ({type(e).__name__}: {e})")

        self._say("OBSERVE", "Loading data ...")
        df = data.load_data()
        if df is None:
            self._say("OBSERVE", "FATAL — data not found.")
            return None
        df, cols = data.detect_columns(df)
        if df is None:
            self._say("OBSERVE", "FATAL — column detection failed.")
            return None

        self._say("OBSERVE", f"Shape : {df.shape[0]:,} rows x {df.shape[1]} cols")
        self._say("OBSERVE", f"Target: {cols['target']}  |  Date: {cols['date']}")

        df = features.add_game_level_metrics(df)
        leakage = data.leakage_check(df, cols["target"], cols["date"])
        self._say("OBSERVE", f"Leakage columns removed: {len(leakage)}")

        df_clean = data.get_clean_df(
            df, cols["target"], cols["date"], drop_leakage=True, matchup=False,
        )
        self._say("OBSERVE", f"Clean : {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")
        return df_clean, cols, leakage

    # ══════════════════════════════════════════════════════════════════════════
    #  PLAN
    # ══════════════════════════════════════════════════════════════════════════

    def _plan(self):
        """Build the initial search space, informed by cross-run memory."""
        self._say("PLAN", f"Goal: LogLoss <= {config.LOGLOSS_THRESHOLD}  (AUC floor >= {config.AUC_FLOOR})")

        search_space = self._default_search_space()

        mem = self._load_memory()
        if mem:
            self._say("PLAN",
                       f"Memory loaded — {mem.get('total_trials_ever', 0)} prior trials, "
                       f"best-ever LogLoss={mem.get('best_logloss_ever', 999):.4f}  "
                       f"AUC={mem.get('best_auc_ever', 0):.4f}")

            # If the previous run produced a policy, use it (with extra buffer)
            # to seed this run's Phase 1.  This is cross-run adaptation.
            prev_policy = mem.get("policy")
            if prev_policy:
                self._say("PLAN", "Applying previous policy to warm-start Phase 1 (buffer=2)")
                if prev_policy.get("window_years"):
                    search_space["window_years"] = self._narrow_range(
                        config.SEARCH_WINDOW_YEARS, prev_policy["window_years"], buffer=2,
                    )
                if prev_policy.get("elo_k"):
                    search_space["elo_k"] = self._narrow_range(
                        config.SEARCH_ELO_K, prev_policy["elo_k"], buffer=2,
                    )

        self._say("PLAN", f"Phase 1 space: {self._space_summary(search_space)}")
        self._say("PLAN",
                   f"Budget: Phase1={config.PHASE1_TRIALS}  Phase2={config.PHASE2_TRIALS}  "
                   f"Patience={config.PHASE_PATIENCE}")
        return search_space

    # ══════════════════════════════════════════════════════════════════════════
    #  RUN PHASE  (generic: used for both Explore and Exploit)
    # ══════════════════════════════════════════════════════════════════════════

    def _run_phase(self, phase_name, search_space, df_clean, cols, n_trials, patience):
        target, date_col = cols["target"], cols["date"]

        study = optuna.create_study(
            direction="minimize",          # minimise LogLoss
            sampler=optuna.samplers.TPESampler(
                seed=config.RANDOM_STATE + abs(hash(phase_name)) % 10000,
            ),
        )

        # Warm-start: enqueue best known config if it fits the search space
        bc = self.best_config
        if bc:
            warm = {}
            for key, space_key in [("window_years", "window_years"),
                                   ("elo_k", "elo_k"),
                                   ("model_type", "model_types"),
                                   ("strategy", "strategies"),
                                   ("top_k", "top_k")]:
                val = bc.get(key)
                if val is not None and val in search_space.get(space_key, []):
                    warm[key] = val
            # Include feature type parameters if present
            for feat_key in ["use_roll_features", "use_diff_features", "use_ratio_features",
                           "use_elo_features", "use_home_away_features"]:
                if feat_key in bc:
                    warm[feat_key] = bc[feat_key]
            if warm:
                try:
                    study.enqueue_trial(warm)
                    self._say(phase_name, "Warm-start: enqueued best known config")
                except Exception:
                    pass

        phase_trials = 0
        phase_patience = 0

        PENALTY_LOGLOSS = 9.0  # returned to Optuna on error / constraint violation

        def objective(trial):
            self._improved_this_trial = False
            try:
                return self._run_trial(trial, df_clean, target, date_col, search_space, phase_name)
            except Exception as e:
                self._say("ERROR", f"Trial {trial.number} crashed: {type(e).__name__}: {e}")
                return PENALTY_LOGLOSS

        print()
        print("-" * 62)
        self._say(phase_name, "Entering phase ...")
        print("-" * 62)

        while True:
            study.optimize(objective, n_trials=1, show_progress_bar=False)
            self.total_trials += 1
            phase_trials += 1

            if self._improved_this_trial:
                phase_patience = 0
            else:
                phase_patience += 1

            best_ll = self.best_metrics.get("LogLoss", 999)
            best_auc = self.best_metrics.get("AUC", 0)

            # Stopping
            if best_ll <= config.LOGLOSS_THRESHOLD:
                self._say(phase_name,
                          f"GOAL REACHED!  LL={best_ll:.4f}  AUC={best_auc:.4f}  "
                          f"Acc={self.best_metrics.get('Accuracy', 0):.1%}")
                break
            if phase_patience >= patience:
                self._say(phase_name, f"Patience exhausted ({patience} trials w/o improvement)")
                break
            if phase_trials >= n_trials:
                self._say(phase_name, f"Budget exhausted ({n_trials} trials)")
                break
            if self.total_trials >= config.MAX_TRIALS:
                self._say(phase_name, f"Global safety cap ({config.MAX_TRIALS})")
                break

            # Periodic reflection (display only)
            if phase_trials % config.REFLECTION_INTERVAL == 0:
                self._reflect(study)

        print("-" * 62)
        self._say(phase_name,
                   f"Phase done — {phase_trials} trials, "
                   f"best LL={self.best_metrics.get('LogLoss', 999):.4f}  "
                   f"AUC={self.best_metrics.get('AUC', 0):.4f}  "
                   f"Acc={self.best_metrics.get('Accuracy', 0):.1%}")
        print("-" * 62)
        return study

    # ── Single trial ──────────────────────────────────────────────────────────

    def _run_trial(self, trial, df_clean, target, date_col, search_space, phase_name=None):
        mt = "TARGET"

        # ── DECIDE ───────────────────────────────────────────────
        params = self._sample_params(trial, search_space, phase_name)
        suffix = f" Top{params.get('top_k', '?')}" if params.get("top_k") else ""
        tag = f"{params['model_type']} W={params['window_years']}y K={params['elo_k']}{suffix}"

        # ── ACT ──────────────────────────────────────────────────
        df_f = features.engineer_features(
            df_clean, target, date_col, include_elo=True, elo_k=params["elo_k"],
        )
        df_f = data.create_matchup_df(df_f, target, date_col)
        df_f = features.add_difference_features(df_f)

        feat = self._get_feature_columns(df_f, mt, target)
        train_df, test_df = evaluate.time_aware_split(
            df_f, date_col, window_years=params["window_years"],
        )
        if len(train_df[mt].unique()) < 2:
            return 9.0  # penalty LogLoss

        strat = params["strategy"]
        
        # Step 1: Filter features by type (if feature type selection is enabled)
        feat_before_filter = len(feat)
        feat_after_filter = feat_before_filter  # Initialize to avoid undefined variable
        if strat == "Auto_Select" and any(k.startswith("use_") for k in params.keys()):
            feat = self._filter_features_by_type(feat, params)
            feat_after_filter = len(feat)
        
        # Step 2: Apply feature selection strategy
        if strat == "Auto_Select":
            feat = self._auto_select_features(
                train_df[feat].fillna(0), train_df[mt], params["top_k"],
            )
        elif strat == "Diff_Only":
            feat = [c for c in feat
                    if c.startswith("DIFF_") or c.startswith("RATIO_")
                    or c in ("ELO_DIFF", "HOME_IS_HOME_CALC")]
        elif strat == "Four_Factors_Only":
            ff = config.FOUR_FACTORS_LIST
            feat = [c for c in feat if any(f in c for f in ff) or c == "HOME_IS_HOME_CALC"]
        elif strat == "No_Elo":
            feat = [c for c in feat if "ELO" not in c.upper()]

        if not feat:
            return 9.0  # penalty LogLoss

        # Remove constant features (zero variance) before scaling to avoid warnings
        train_feat_df = train_df[feat].fillna(0)
        variances = train_feat_df.var()
        non_constant_feat = [f for f in feat if variances.get(f, 0) > 1e-8]
        
        if not non_constant_feat:
            return 9.0  # penalty LogLoss - all features are constant
        
        # Update feat list to only include non-constant features
        feat = non_constant_feat

        pre = features.get_preprocessing_pipeline(feat, [])
        train_input = train_df[feat].replace([np.inf, -np.inf], np.nan)
        test_input = test_df[feat].replace([np.inf, -np.inf], np.nan)
        X_tr = pre.fit_transform(train_input)
        X_te = pre.transform(test_input)
        X_tr = self._sanitize_matrix(X_tr)
        X_te = self._sanitize_matrix(X_te)

        mdl = self._build_model(params)
        mdl = models.calibrate_model(mdl, X_tr, train_df[mt])
        try:
            probs = models.predict_proba_stable(mdl, X_te)[:, 1]
        except FloatingPointError:
            self._say("TRIAL", f"#{trial.number:<4} {tag:<38} numerically unstable - PENALISED")
            return 9.0
        m = evaluate.calculate_metrics(test_df[mt], probs)
        auc = m["AUC"]
        logloss = m["LogLoss"]

        # ── AUC floor constraint ─────────────────────────────────
        # If AUC is below the floor, the trial is "invalid" — return a
        # penalty LogLoss so Optuna learns to avoid these regions.
        if auc < config.AUC_FLOOR:
            self._say("TRIAL",
                       f"#{trial.number:<4} {tag:<38} "
                       f"LL={logloss:.4f}  AUC={auc:.4f} < floor {config.AUC_FLOOR} — PENALISED")
            return 9.0

        # ── LEARN ────────────────────────────────────────────────
        m.update({
            "trial": trial.number, "window_years": params["window_years"],
            "elo_k": params["elo_k"], "model_type": params["model_type"],
            "strategy": params["strategy"], "top_k": params.get("top_k"),
            "features_type": "optimized",
            "model": f"{params['model_type']} (W={params['window_years']}y, K={params['elo_k']}, S={params['strategy']})",
        })
        self.experiment_results.append(m)

        # Best = lowest LogLoss (among trials that pass the AUC floor)
        is_best = logloss < self.best_metrics.get("LogLoss", 999)
        if is_best:
            self.best_metrics = m
            self.best_model = mdl
            self.best_preprocessor = pre
            self.best_feature_names = feat
            self.best_config = {
                "window_years": params["window_years"],
                "elo_k": params["elo_k"],
                "model_type": params["model_type"],
                "strategy": params["strategy"],
                "top_k": params.get("top_k"),
            }
            # Add feature type selection to config if enabled
            if strat == "Auto_Select" and any(k.startswith("use_") for k in params.keys()):
                self.best_config.update({
                    "use_roll_features": params.get("use_roll_features", True),
                    "use_diff_features": params.get("use_diff_features", True),
                    "use_ratio_features": params.get("use_ratio_features", True),
                    "use_elo_features": params.get("use_elo_features", True),
                    "use_home_away_features": params.get("use_home_away_features", True),
                })
            self.improvement_log.append((self.total_trials + 1, round(logloss, 4)))
            self._improved_this_trial = True
            # Show full feature list for best model
            if feat and len(feat) > 0:
                self._say("BEST", f"Selected {len(feat)} features:")
                for i, f in enumerate(feat, 1):
                    self._say("BEST", f"  {i:2d}. {f}")
            else:
                self._say("BEST", "Warning: No features selected!")

        self.top_trials.append({
            "auc": auc, "logloss": logloss,
            "model": mdl, "preprocessor": pre,
            "features": feat, "config": m,
            "model_type": params["model_type"],
        })
        # Diverse ensemble: keep the best trial per model type, then
        # fill remaining slots by overall LogLoss.  This ensures the
        # ensemble blends different model families.
        self.top_trials = self._diverse_top_trials(self.top_trials, n=3)

        # ── COMMUNICATE ──────────────────────────────────────────
        acc = m.get("Accuracy", 0)
        brier = m.get("BrierScore", 0)
        marker = "  ** NEW BEST **" if is_best else ""
        self._say("TRIAL",
                   f"#{trial.number:<4} {tag:<38} "
                   f"LL={logloss:.4f}  AUC={auc:.4f}  "
                   f"Acc={acc:.1%}  Best_LL={self.best_metrics['LogLoss']:.4f}{marker}")

        # Show feature type selection (if enabled)
        if strat == "Auto_Select" and any(k.startswith("use_") for k in params.keys()):
            feat_types = []
            if params.get("use_roll_features", True):
                feat_types.append("ROLL")
            if params.get("use_diff_features", True):
                feat_types.append("DIFF")
            if params.get("use_ratio_features", True):
                feat_types.append("RATIO")
            if params.get("use_elo_features", True):
                feat_types.append("ELO")
            if params.get("use_home_away_features", True):
                feat_types.append("HOME/AWAY")
            feat_type_str = "+".join(feat_types) if feat_types else "none"
            self._say("       ", f"     feat_types: [{feat_type_str}] → {feat_before_filter}→{feat_after_filter}→{len(feat)}")
        
        # Show model-internal hyperparameters
        mp = params.get("model_params", {})
        if mp:
            if params["model_type"] == "XGBoost":
                hp = (f"n_est={mp['n_estimators']} depth={mp['max_depth']} "
                      f"lr={mp['learning_rate']:.4f} sub={mp['subsample']:.2f} "
                      f"col={mp['colsample_bytree']:.2f} "
                      f"a={mp.get('reg_alpha', 0):.2f} l={mp.get('reg_lambda', 1):.2f}")
            elif params["model_type"] == "RandomForest":
                hp = (f"n_est={mp['n_estimators']} depth={mp['max_depth']} "
                      f"min_split={mp['min_samples_split']} "
                      f"min_leaf={mp.get('min_samples_leaf', 1)}")
            else:
                hp = str(mp)
            self._say("       ", f"     hypers: {hp}")
        
        # Show selected features (top 5 for brevity, full list saved in metrics)
        if strat == "Auto_Select" and feat and len(feat) > 0:
            top_5_feat = feat[:5]
            feat_preview = ", ".join(str(f) for f in top_5_feat)
            if len(feat) > 5:
                feat_preview += f" ... (+{len(feat)-5} more)"
            self._say("       ", f"     features ({len(feat)}): {feat_preview}")

        if logloss <= config.LOGLOSS_THRESHOLD:
            trial.study.stop()
        return logloss  # Optuna minimises this

    # ── Param sampling ────────────────────────────────────────────────────────

    @staticmethod
    def _sample_params(trial, search_space, phase_name=None):
        models = search_space["model_types"]
        if phase_name == "EXPLORE":
            # Round-robin: each model gets equal trials for fair comparison
            p = {
                "window_years": trial.suggest_categorical("window_years", search_space["window_years"]),
                "elo_k": trial.suggest_categorical("elo_k", search_space["elo_k"]),
                "model_type": models[trial.number % len(models)],
                "strategy": trial.suggest_categorical("strategy", search_space["strategies"]),
            }
        else:
            p = {
                "window_years": trial.suggest_categorical("window_years", search_space["window_years"]),
                "elo_k": trial.suggest_categorical("elo_k", search_space["elo_k"]),
                "model_type": trial.suggest_categorical("model_type", models),
                "strategy": trial.suggest_categorical("strategy", search_space["strategies"]),
            }
        if p["strategy"] == "Auto_Select":
            p["top_k"] = trial.suggest_categorical("top_k", search_space["top_k"])
            
            # Feature type selection: let Optuna choose which feature types to use
            p["use_roll_features"] = trial.suggest_categorical("use_roll_features", [True, False])
            p["use_diff_features"] = trial.suggest_categorical("use_diff_features", [True, False])
            p["use_ratio_features"] = trial.suggest_categorical("use_ratio_features", [True, False])
            p["use_elo_features"] = trial.suggest_categorical("use_elo_features", [True, False])
            p["use_home_away_features"] = trial.suggest_categorical("use_home_away_features", [True, False])

        mp = {}
        if p["model_type"] == "XGBoost":
            # Tighter ranges to prevent overfitting on low-signal sports data
            mp = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 600, step=100),
                "max_depth": trial.suggest_int("xgb_max_depth", 2, 6),
                "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.7, 1.0),
                "reg_alpha": trial.suggest_float("xgb_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("xgb_lambda", 1.0, 5.0),
            }
        elif p["model_type"] == "RandomForest":
            mp = {
                "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500, step=100),
                "max_depth": trial.suggest_int("rf_max_depth", 4, 15),
                "min_samples_split": trial.suggest_int("rf_min_split", 5, 20),
                "min_samples_leaf": trial.suggest_int("rf_min_leaf", 2, 10),
            }
        p["model_params"] = mp
        return p

    # ══════════════════════════════════════════════════════════════════════════
    #  REFLECT  (display-only, used *within* a phase)
    # ══════════════════════════════════════════════════════════════════════════

    def _reflect(self, study):
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if len(completed) < 5:
            return
        # Sort ascending — lower LogLoss is better (Optuna minimises)
        top_n = sorted(completed, key=lambda t: t.value)[:10]
        mc = {}
        ws, ks, tks = [], [], []
        feat_type_counts = {"roll": 0, "diff": 0, "ratio": 0, "elo": 0, "home_away": 0}
        for t in top_n:
            mt = t.params.get("model_type", "?")
            mc[mt] = mc.get(mt, 0) + 1
            ws.append(t.params.get("window_years", 0))
            ks.append(t.params.get("elo_k", 0))
            if "top_k" in t.params:
                tks.append(t.params["top_k"])
            # Count feature type usage
            for feat_key in ["use_roll_features", "use_diff_features", "use_ratio_features",
                           "use_elo_features", "use_home_away_features"]:
                if feat_key in t.params and t.params[feat_key] is True:
                    short_key = feat_key.replace("use_", "").replace("_features", "")
                    feat_type_counts[short_key] += 1
        bm = max(mc, key=mc.get)
        feat_summary = " ".join([f"{k}:{v}" for k, v in feat_type_counts.items() if v > 0])
        print()
        self._say("REFLECT", f"--- {len(completed)} trials | best LL={study.best_value:.4f} | "
                              f"top model: {bm} ({mc[bm]}/10) | "
                              f"W:{min(ws)}-{max(ws)} K:{min(ks)}-{max(ks)} "
                              + (f"Top:{min(tks)}-{max(tks)}" if tks else "")
                              + (f" | Feat: {feat_summary}" if feat_summary else ""))
        if study.best_value > 0.75:
            self._say("REFLECT", "WARNING: LogLoss > 0.75 — predictions are poor. Check data quality.")
        print()

    # ══════════════════════════════════════════════════════════════════════════
    #  REFLECT + ADAPT  (between phases — generates policy.json)
    # ══════════════════════════════════════════════════════════════════════════

    def _reflect_and_adapt(self, study, prev_space):
        """
        The core agentic behaviour: analyse Phase 1 results and produce
        a *narrowed* search space (policy) for Phase 2.
        Writes outputs/policy.json as auditable evidence.
        """
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if len(completed) < 5:
            self._say("ADAPT", "Too few trials — keeping current space.")
            self.policy = prev_space
            return prev_space

        # Sort ascending — lower LogLoss is better
        top_n = sorted(completed, key=lambda t: t.value)[:10]

        # ── Gather top-10 patterns ────────────────────────────────
        mc = {}
        ws, ks, tks = [], [], []
        feat_type_patterns = {
            "roll": [], "diff": [], "ratio": [], "elo": [], "home_away": []
        }
        for t in top_n:
            mt = t.params.get("model_type", "?")
            mc[mt] = mc.get(mt, 0) + 1
            ws.append(t.params.get("window_years", 0))
            ks.append(t.params.get("elo_k", 0))
            if "top_k" in t.params:
                tks.append(t.params["top_k"])
            # Collect feature type patterns
            for feat_key in ["use_roll_features", "use_diff_features", "use_ratio_features",
                           "use_elo_features", "use_home_away_features"]:
                if feat_key in t.params:
                    short_key = feat_key.replace("use_", "").replace("_features", "")
                    feat_type_patterns[short_key].append(t.params[feat_key])

        # ── Build narrowed space ──────────────────────────────────
        new = {}
        new["window_years"] = self._narrow_range(prev_space["window_years"], ws, buffer=1)
        new["elo_k"]        = self._narrow_range(prev_space["elo_k"], ks, buffer=1)
        new["top_k"]        = self._narrow_range(prev_space["top_k"], tks, buffer=1) if tks else list(prev_space["top_k"])
        new["strategies"]   = list(prev_space["strategies"])

        # Keep ALL model types — never drop a model family.
        # Even if a model didn't appear in top-10, it still competes in
        # Phase 2 so the ensemble stays diverse.
        new["model_types"] = list(prev_space["model_types"])

        # ── Log adaptations ───────────────────────────────────────
        best_model = max(mc, key=mc.get)

        print()
        self._say("ADAPT", "=" * 50)
        self._say("ADAPT", "REFLECT -> ADAPT : Generating policy for Phase 2")
        self._say("ADAPT", "-" * 50)
        self._say("ADAPT", f"Analysed {len(completed)} trials (top-10 patterns):")
        self._say("ADAPT", f"  Top model : {best_model} ({mc[best_model]}/10)")
        self._say("ADAPT", f"  Window    : {prev_space['window_years']} -> {new['window_years']}")
        self._say("ADAPT", f"  Elo-K     : {prev_space['elo_k']} -> {new['elo_k']}")
        self._say("ADAPT", f"  Models    : {prev_space['model_types']} -> {new['model_types']}")
        self._say("ADAPT", f"  Top-K     : {prev_space['top_k']} -> {new['top_k']}")
        # Log feature type patterns
        feat_type_summary = []
        for key, vals in feat_type_patterns.items():
            if vals:
                true_count = sum(1 for v in vals if v is True)
                feat_type_summary.append(f"{key}:{true_count}/{len(vals)}")
        if feat_type_summary:
            self._say("ADAPT", f"  Feat types: {', '.join(feat_type_summary)}")

        # ── Write policy.json (auditable artefact) ────────────────
        policy_doc = {
            "generated_at": datetime.now().isoformat(),
            "phase": "EXPLORE -> EXPLOIT transition",
            "based_on_trials": len(completed),
            "best_logloss_at_transition": round(study.best_value, 4),
            "adaptations": {
                "window_years": {"before": prev_space["window_years"],
                                 "after": new["window_years"]},
                "elo_k":        {"before": prev_space["elo_k"],
                                 "after": new["elo_k"]},
                "model_types":  {"before": prev_space["model_types"],
                                 "after": new["model_types"]},
                "top_k":        {"before": prev_space["top_k"],
                                 "after": new["top_k"]},
            },
            "reasoning": {
                "top_model": f"{best_model} ({mc[best_model]}/10 in top-10)",
                "window_sweet_spot": f"{min(ws)}-{max(ws)} years",
                "elo_k_sweet_spot": f"{min(ks)}-{max(ks)}",
                "top_k_sweet_spot": f"{min(tks)}-{max(tks)}" if tks else "N/A",
                "feature_type_patterns": {
                    k: f"{sum(1 for v in vals if v is True)}/{len(vals)} True" 
                    if vals else "N/A"
                    for k, vals in feat_type_patterns.items()
                },
            },
        }
        with open(config.POLICY_FILE, "w") as f:
            json.dump(policy_doc, f, indent=2)

        self._say("ADAPT", f"Policy saved -> {config.POLICY_FILE}")
        self._say("ADAPT", "=" * 50)
        print()

        self.policy = new  # store for memory persistence
        return new

    # ══════════════════════════════════════════════════════════════════════════
    #  DIAGNOSE  (triggered when agent is stuck)
    # ══════════════════════════════════════════════════════════════════════════

    def _diagnose(self, df_clean, cols):
        """
        Agentic action: when stuck, run data-sanity checks and ablation
        experiments to understand *why*, then write diagnostics.md.
        """
        target, date_col = cols["target"], cols["date"]
        elo_k = self.best_config.get("elo_k", config.ELO_K)
        window = self.best_config.get("window_years", 5)

        print()
        self._say("DIAGNOSE", "=" * 50)
        self._say("DIAGNOSE", "Agent is stuck — triggering diagnostic actions")
        self._say("DIAGNOSE", "=" * 50)

        lines = [
            "# Agent Diagnostic Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Best LogLoss at diagnosis: {self.best_metrics.get('LogLoss', 999):.4f}",
            f"Best AUC at diagnosis: {self.best_metrics.get('AUC', 0):.4f}",
            f"Goal: LogLoss <= {config.LOGLOSS_THRESHOLD}  (AUC floor >= {config.AUC_FLOOR})",
            "",
        ]

        # ── Action 1: Data sanity ─────────────────────────────────
        self._say("DIAGNOSE", "Action 1 — Data sanity check ...")
        df_f = features.engineer_features(
            df_clean, target, date_col, include_elo=True, elo_k=elo_k,
        )
        df_f = data.create_matchup_df(df_f, target, date_col)
        df_f = features.add_difference_features(df_f)
        train_df, test_df = evaluate.time_aware_split(df_f, date_col, window_years=window)

        test_bal = test_df["TARGET"].mean()
        n_feat = len(self._get_feature_columns(df_f, "TARGET", target))
        nan_r = train_df.select_dtypes(include=[np.number]).isna().mean().mean()

        lines += [
            "## Data Sanity",
            f"- Train size : {len(train_df):,} matchups",
            f"- Test size  : {len(test_df):,} matchups",
            f"- Test home-win rate : {test_bal:.2%}",
            f"- Available features : {n_feat}",
            f"- Avg NaN ratio      : {nan_r:.4f}",
            "",
        ]
        self._say("DIAGNOSE",
                   f"  Train={len(train_df):,}  Test={len(test_df):,}  "
                   f"Balance={test_bal:.2%}  Feats={n_feat}")

        # ── Action 2: Ablation studies ────────────────────────────
        self._say("DIAGNOSE", "Action 2 — Ablation studies ...")
        ablations = [
            ("Diff_Only + LogReg (baseline)",  "Diff_Only",          "LogisticRegression", None),
            ("Auto_Select + LogReg",           "Auto_Select",        "LogisticRegression", None),
            ("No_Elo + RandomForest",          "No_Elo",             "RandomForest", None),
            ("Four_Factors + RandomForest",    "Four_Factors_Only",  "RandomForest", None),
            # Feature type combination tests
            ("Auto_Select (Roll+Diff only) + XGBoost", "Auto_Select", "XGBoost",
             {"use_roll_features": True, "use_diff_features": True, "use_ratio_features": False,
              "use_elo_features": False, "use_home_away_features": False}),
            ("Auto_Select (No Roll) + XGBoost", "Auto_Select", "XGBoost",
             {"use_roll_features": False, "use_diff_features": True, "use_ratio_features": True,
              "use_elo_features": True, "use_home_away_features": True}),
            ("Auto_Select (All types) + RandomForest", "Auto_Select", "RandomForest",
             {"use_roll_features": True, "use_diff_features": True, "use_ratio_features": True,
              "use_elo_features": True, "use_home_away_features": True}),
        ]

        lines += [
            "## Ablation Studies",
            "| Configuration | AUC | LogLoss | Brier |",
            "|---|---|---|---|",
        ]

        for name, strat, mtype, feat_params in ablations:
            try:
                r = self._quick_eval(
                    df_clean, target, date_col, strat, mtype,
                    window_years=window, elo_k=elo_k, top_k=25,
                    feat_type_params=feat_params,
                )
                lines.append(
                    f"| {name} | {r['AUC']:.4f} | {r['LogLoss']:.4f} | {r['BrierScore']:.4f} |"
                )
                self._say("DIAGNOSE",
                           f"  {name}: LL={r['LogLoss']:.4f}  AUC={r['AUC']:.4f}  "
                           f"Acc={r.get('Accuracy', 0):.1%}")
            except Exception as e:
                lines.append(f"| {name} | FAILED | — | — |")
                self._say("DIAGNOSE", f"  {name}: FAILED ({e})")

        ba = self.best_metrics.get("AUC", 0)
        bl = self.best_metrics.get("LogLoss", 0)
        bb = self.best_metrics.get("BrierScore", 0)
        lines.append(f"| **Best Model (Agent)** | **{ba:.4f}** | **{bl:.4f}** | **{bb:.4f}** |")
        lines.append("")

        # ── Recommendations ───────────────────────────────────────
        lines.append("## Agent Recommendations")
        if test_bal > 0.6 or test_bal < 0.4:
            lines.append(f"- Class imbalance ({test_bal:.2%}). Consider balanced sampling.")
        if nan_r > 0.1:
            lines.append(f"- High NaN ratio ({nan_r:.4f}). Check feature pipeline.")
        if ba < 0.55:
            lines.append("- AUC very low — features may lack predictive signal.")
            lines.append("- Consider: player-level data, injuries, or betting lines.")
        else:
            lines.append(f"- Current best LogLoss={bl:.4f}  AUC={ba:.4f}")
            lines.append(f"- Goal: LogLoss <= {config.LOGLOSS_THRESHOLD}  AUC >= {config.AUC_FLOOR}")
            lines.append("- Consider: wider window, more Phase 2 budget, or threshold adjustment.")
            # Check best config for feature type patterns
            best_feat_types = []
            for feat_key in ["use_roll_features", "use_diff_features", "use_ratio_features",
                           "use_elo_features", "use_home_away_features"]:
                if self.best_config.get(feat_key) is True:
                    best_feat_types.append(feat_key.replace("use_", "").replace("_features", ""))
            if best_feat_types:
                lines.append(f"- Best config uses feature types: {', '.join(best_feat_types)}")
                lines.append("- Consider testing different feature type combinations in ablation studies.")
        lines.append("")

        with open(config.DIAGNOSTICS_FILE, "w") as f:
            f.write("\n".join(lines))

        self._say("DIAGNOSE", f"Report saved -> {config.DIAGNOSTICS_FILE}")
        self._say("DIAGNOSE", "=" * 50)
        print()

    # ══════════════════════════════════════════════════════════════════════════
    #  FINALIZE
    # ══════════════════════════════════════════════════════════════════════════

    def _finalize(self, df_clean, cols, leakage_cols):
        target, date_col = cols["target"], cols["date"]

        print()
        print("=" * 62)
        self._say("FINALIZE", "Saving artefacts ...")

        if self.best_model is None:
            self._say("FINALIZE", "No model found. Nothing to save.")
            return

        self._say("FINALIZE", f"Best config: {self.best_config}")

        # Re-generate features for team-stats export
        df_feat_raw = features.engineer_features(
            df_clean, target, date_col,
            include_elo=True, elo_k=self.best_config["elo_k"],
        )

        # Models
        joblib.dump(self.best_model, os.path.join(config.MODELS_DIR, "best_model.pkl"))
        if self.best_preprocessor:
            joblib.dump(self.best_preprocessor, os.path.join(config.MODELS_DIR, "preprocessor.pkl"))
        joblib.dump(self.top_trials, os.path.join(config.MODELS_DIR, "top_models_ensemble.pkl"))
        self._say("FINALIZE", "Saved: best_model.pkl, preprocessor.pkl, top_models_ensemble.pkl")

        # Config
        with open(os.path.join(config.OUTPUTS_DIR, "best_config.json"), "w") as f:
            json.dump(self.best_config, f, indent=4)

        # Team stats for app.py
        latest = df_feat_raw.sort_values(date_col).groupby("TEAM_ID").tail(1)
        num_cols = df_feat_raw.select_dtypes(include=[np.number]).columns.tolist()
        if "TEAM_ID" in num_cols:
            num_cols.remove("TEAM_ID")
        
        # Remove old CSV columns that conflict with engineered features
        # CSV has ROLL_PTS (NaN), but we generate ROLL_PTS_AVG
        old_roll_cols = ['ROLL_PTS', 'ROLL_AST', 'ROLL_REB', 'ROLL_STL', 'ROLL_BLK', 
                         'ROLL_TOV', 'ROLL_PF', 'ROLL_FG_PCT', 'ROLL_FG3_PCT', 
                         'ROLL_FT_PCT', 'ROLL_WIN_PCT']
        for col in old_roll_cols:
            if col in num_cols:
                num_cols.remove(col)
        
        stats_dict = latest.set_index("TEAM_ID")[num_cols].to_dict("index")
        with open(os.path.join(config.OUTPUTS_DIR, "latest_team_stats.json"), "w") as f:
            json.dump(stats_dict, f, indent=4)

        # Metrics
        self.best_metrics["feature_names"] = self.best_feature_names

        # Memory
        self._save_memory()
        self._say("FINALIZE", "Agent memory + policy persisted for next run")

        # ── Evaluation artefacts ──────────────────────────────────
        self._say("FINALIZE", "Generating evaluation reports ...")
        df_best = data.create_matchup_df(df_feat_raw, target, date_col)
        df_best = features.add_difference_features(df_best)
        _, test_df = evaluate.time_aware_split(
            df_best, date_col, window_years=self.best_config["window_years"],
        )
        X_te = self.best_preprocessor.transform(test_df[self.best_feature_names])
        y_te = test_df["TARGET"]

        # Calibration
        probs = models.predict_proba_stable(self.best_model, X_te)[:, 1]
        evaluate.plot_calibration_curve(y_te, probs, self.best_metrics["model"])

        # Ensemble
        ens_p, ens_y = evaluate.get_ensemble_probs(self.top_trials, df_clean, target, date_col)
        if ens_p is not None and ens_y is not None:
            em = evaluate.calculate_metrics(ens_y, ens_p)
            self._say("FINALIZE",
                       f"Ensemble: LL={em['LogLoss']:.4f}  AUC={em['AUC']:.4f}  "
                       f"Acc={em.get('Accuracy', 0):.1%}")
            self.best_metrics["ensemble_auc"] = em["AUC"]
            self.best_metrics["ensemble_logloss"] = em["LogLoss"]

        # Persist metrics after all post-processing (including ensemble metrics).
        with open(os.path.join(config.OUTPUTS_DIR, "metrics.json"), "w") as f:
            json.dump(self.best_metrics, f, indent=4)

        # Feature importance
        base = self.best_model.estimator if hasattr(self.best_model, "estimator") else self.best_model
        fi = evaluate.get_feature_importance(base, self.best_feature_names)

        # Report
        di = {"shape": df_best.shape, "target": target, "date": date_col}
        rp = report.generate_markdown_report(
            di, leakage_cols, self.experiment_results, self.best_metrics, fi,
        )

        # Summary
        bm = self.best_metrics
        print()
        print("=" * 62)
        print("   OPTIMISATION COMPLETE")
        print("=" * 62)
        print(f"   Best Config  : {self.best_config}")
        print(f"   Best LogLoss : {bm.get('LogLoss', 999):.4f}")
        print(f"   Best AUC     : {bm.get('AUC', 0):.4f}")
        print(f"   Best Accuracy: {bm.get('Accuracy', 0):.1%}")
        print(f"   Best Brier   : {bm.get('BrierScore', 0):.4f}")
        el = bm.get("ensemble_logloss")
        ea = bm.get("ensemble_auc")
        if el and ea:
            print(f"   Ensemble LL  : {el:.4f}")
            print(f"   Ensemble AUC : {ea:.4f}")
        print(f"   Total Trials : {self.total_trials}")
        print(f"   Policy       : {config.POLICY_FILE}")
        print(f"   Report       : {rp}")
        print("=" * 62)
        print()

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def run(self):
        """
        Full agentic pipeline:
            OBSERVE → PLAN → EXPLORE → REFLECT+ADAPT → EXPLOIT → (DIAGNOSE) → FINALIZE
        """
        # ── OBSERVE ──
        result = self._observe()
        if result is None:
            return
        df_clean, cols, leakage_cols = result
        self.start_time = time.time()

        # ── PLAN ──
        search_space = self._plan()

        # ── PHASE 1: EXPLORE (wide search) ──
        print()
        self._say("AGENT", "=" * 50)
        self._say("AGENT", "PHASE 1 — EXPLORATION")
        self._say("AGENT", f"Space: {self._space_summary(search_space)}")
        self._say("AGENT", "=" * 50)

        study1 = self._run_phase(
            "EXPLORE", search_space, df_clean, cols,
            config.PHASE1_TRIALS, config.PHASE_PATIENCE,
        )

        # Early exit if goal already reached
        if self.best_metrics.get("LogLoss", 999) <= config.LOGLOSS_THRESHOLD:
            self._say("AGENT", "Goal reached in Phase 1 — skipping Phase 2.")
            self._finalize(df_clean, cols, leakage_cols)
            return

        # ── REFLECT + ADAPT → policy.json ──
        narrowed = self._reflect_and_adapt(study1, search_space)

        # ── PHASE 2: EXPLOIT (policy-driven narrow search) ──
        self._say("AGENT", "=" * 50)
        self._say("AGENT", "PHASE 2 — EXPLOITATION (policy-driven)")
        self._say("AGENT", f"Narrowed space: {self._space_summary(narrowed)}")
        self._say("AGENT", "=" * 50)

        self._run_phase(
            "EXPLOIT", narrowed, df_clean, cols,
            config.PHASE2_TRIALS, config.PHASE_PATIENCE,
        )

        # ── DIAGNOSE (if still above LogLoss goal) ──
        if self.best_metrics.get("LogLoss", 999) > config.LOGLOSS_THRESHOLD:
            self._diagnose(df_clean, cols)

        # ── FINALIZE ──
        self._finalize(df_clean, cols, leakage_cols)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    agent = AgentOrchestrator()
    agent.run()
