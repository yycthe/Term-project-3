"""
Streamlit UI — NBA Game Predictor
==================================
Three tabs:
  1. Predict Next Game  — select a team, see their next opponent, predict
  2. Prediction History — track all predictions vs actual results
  3. Experiment Report  — model training details
"""

import os
import sys
import subprocess
import streamlit as st
import pandas as pd
import joblib
import json
import uuid
import numpy as np
from datetime import datetime

import config
import features as feat_module

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBA Predictor", layout="wide", page_icon="🏀")

# ═════════════════════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_xgboost_device(model, target_device='cpu'):
    """
    Ensure XGBoost model uses CPU device (always CPU to avoid device mismatch).
    """
    from xgboost import XGBClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    def _set_xgb_device(xgb_model, device='cpu'):
        """Set XGBoost model device to CPU, including booster."""
        if isinstance(xgb_model, XGBClassifier):
            xgb_model.set_params(device=device)
            # Also set booster device if it exists (for already-fitted models)
            if hasattr(xgb_model, 'get_booster'):
                try:
                    booster = xgb_model.get_booster()
                    if hasattr(booster, 'set_param'):
                        booster.set_param('device', device)
                except Exception:
                    pass
    
    if isinstance(model, XGBClassifier):
        _set_xgb_device(model, target_device)
    # Handle CalibratedClassifierCV wrapper
    elif isinstance(model, CalibratedClassifierCV):
        # Check estimators_ first (cv > 1 case)
        if hasattr(model, 'estimators_') and model.estimators_:
            for est in model.estimators_:
                _set_xgb_device(est, target_device)
        # Check base_estimator (fitted model stores the original estimator)
        if hasattr(model, 'base_estimator') and isinstance(model.base_estimator, XGBClassifier):
            _set_xgb_device(model.base_estimator, target_device)
        # Check estimator (single estimator case, or after fit)
        if hasattr(model, 'estimator') and isinstance(model.estimator, XGBClassifier):
            _set_xgb_device(model.estimator, target_device)
    # Generic wrapper with estimator attribute
    elif hasattr(model, 'estimator') and isinstance(model.estimator, XGBClassifier):
        _set_xgb_device(model.estimator, target_device)
    
    return model


@st.cache_resource
def load_artifacts():
    """Load trained model, metrics, team stats, preprocessor, ensemble."""
    try:
        model = joblib.load("models/best_model.pkl")
        # Ensure XGBoost model uses the correct device (matches current system)
        model = _ensure_xgboost_device(model)
        
        ensemble = None
        if os.path.exists("models/top_models_ensemble.pkl"):
            ensemble = joblib.load("models/top_models_ensemble.pkl")
            # Ensure device consistency for all ensemble members
            if ensemble:
                for trial in ensemble:
                    if 'model' in trial:
                        trial['model'] = _ensure_xgboost_device(trial['model'])
        
        with open("outputs/metrics.json", "r") as f:
            metrics = json.load(f)
        with open("outputs/latest_team_stats.json", "r") as f:
            team_stats = json.load(f)
        preprocessor = None
        if os.path.exists("models/preprocessor.pkl"):
            preprocessor = joblib.load("models/preprocessor.pkl")
        return model, metrics, team_stats, preprocessor, ensemble
    except Exception:
        return None, None, None, None, None


@st.cache_data(ttl=300)
def load_nba_teams():
    """Get all 30 NBA teams from nba_api (cached 5 min)."""
    try:
        from nba_fetch import get_all_teams
        teams = get_all_teams()
        return {t["abbreviation"]: t for t in teams}
    except Exception:
        return {}


@st.cache_data(ttl=120, show_spinner=False)
def fetch_next_game(team_abbr):
    """Get the next game for a team (cached 2 min)."""
    try:
        from nba_fetch import get_team_next_game
        result = get_team_next_game(team_abbr)
        # Explicitly return None (not empty dict) if no game found
        return result if result else None
    except Exception as e:
        # Log error for debugging (Streamlit will show in terminal)
        import logging
        import traceback
        logging.error(f"fetch_next_game failed for {team_abbr}: {e}")
        logging.error(traceback.format_exc())
        return None


@st.cache_data(ttl=120, show_spinner=False)
def load_upcoming_games(limit_days=7):
    """Get all upcoming (unplayed) games in the next limit_days days (cached 2 min)."""
    try:
        from nba_fetch import get_all_upcoming_games
        return get_all_upcoming_games(limit_days=limit_days)
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICTION I/O
# ═════════════════════════════════════════════════════════════════════════════

def _load_predictions():
    """Load prediction history from JSON."""
    path = config.PREDICTIONS_FILE
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_predictions(preds):
    """Save prediction history to JSON."""
    with open(config.PREDICTIONS_FILE, "w") as f:
        json.dump(preds, f, indent=2, default=str)


def _dedupe_predictions_by_game_id(predictions):
    """Keep at most one prediction per game_id (most recent wins). Removes home/away duplicate entries."""
    by_id = {}
    for p in reversed(predictions):
        gid = p.get("game_id")
        if gid is not None and gid not in by_id:
            by_id[gid] = p
    return list(by_id.values())


def _settle_on_load():
    """Auto-settle pending predictions on app startup."""
    preds = _load_predictions()
    pending_count = sum(1 for p in preds if p.get("status") == "pending")
    if pending_count == 0:
        return preds, 0
    try:
        from nba_fetch import settle_predictions
        updated = settle_predictions(preds)
        settled_count = pending_count - sum(1 for p in updated if p.get("status") == "pending")
        if settled_count > 0:
            _save_predictions(updated)
        return updated, settled_count
    except Exception:
        return preds, 0


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICTION LOGIC (reused from old app.py)
# ═════════════════════════════════════════════════════════════════════════════

def run_prediction(home_id, away_id, model, metrics, team_stats, preprocessor, ensemble):
    """
    Run the model / ensemble and return (win_prob_home, model_name).
    win_prob_home = probability that the HOME team wins.
    """
    # Ensure XGBoost models use the correct device (matches training device)
    model = _ensure_xgboost_device(model)
    if ensemble:
        for trial in ensemble:
            if 'model' in trial:
                trial['model'] = _ensure_xgboost_device(trial['model'])
    
    h_stats = team_stats.get(str(home_id), team_stats.get(home_id, {}))
    a_stats = team_stats.get(str(away_id), team_stats.get(away_id, {}))

    if not h_stats or not a_stats:
        return None, "N/A"

    # Build full feature row
    full_input_data = {}
    for base_feat, val in h_stats.items():
        full_input_data[f"HOME_{base_feat}"] = val
    for base_feat, val in a_stats.items():
        full_input_data[f"AWAY_{base_feat}"] = val

    full_input_data["HOME_IS_HOME_CALC"] = 1
    full_input_data["AWAY_IS_HOME_CALC"] = 0

    full_input_df = pd.DataFrame([full_input_data])
    full_input_df = feat_module.add_difference_features(full_input_df)

    # Run prediction with device-consistent models (no warnings should appear)
    if ensemble is not None:
        all_probs = []
        for trial in ensemble:
            try:
                trial_features = trial["features"]
                trial_df = full_input_df.reindex(columns=trial_features, fill_value=0)
                X_trial = trial["preprocessor"].transform(trial_df)
                prob = trial["model"].predict_proba(X_trial)[0][1]
                all_probs.append(prob)
            except Exception:
                # Skip failed ensemble members silently
                continue
        if all_probs:
            win_prob = float(np.mean(all_probs))
            model_name = f"Ensemble ({len(all_probs)} models)"
        else:
            # Fallback to single model if all ensemble members failed
            feature_names = metrics.get("feature_names", [])
            final_input_df = full_input_df.reindex(columns=feature_names, fill_value=0)
            if preprocessor is not None:
                input_scaled = preprocessor.transform(final_input_df)
            else:
                input_scaled = final_input_df.values
            win_prob = float(model.predict_proba(input_scaled)[0][1])
            model_name = metrics.get("model", "Best Model").split("(")[0]
    else:
        feature_names = metrics.get("feature_names", [])
        final_input_df = full_input_df.reindex(columns=feature_names, fill_value=0)
        if preprocessor is not None:
            input_scaled = preprocessor.transform(final_input_df)
        else:
            input_scaled = final_input_df.values
        win_prob = float(model.predict_proba(input_scaled)[0][1])
        model_name = metrics.get("model", "Best Model").split("(")[0]

    return win_prob, model_name


def run_training_with_live_logs():
    """
    Run agent.py and stream stdout/stderr logs in the app.
    Returns (success: bool, full_log: str).
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-u", "agent.py"]

    log_placeholder = st.empty()
    logs = []
    update_every = 5

    try:
        with st.spinner("Training in progress... this may take several minutes."):
            proc = subprocess.Popen(
                cmd,
                cwd=root_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            line_count = 0
            for line in proc.stdout:
                logs.append(line.rstrip("\n"))
                line_count += 1
                if line_count % update_every == 0:
                    log_placeholder.text("\n".join(logs[-200:]))

            return_code = proc.wait()
            log_placeholder.text("\n".join(logs[-200:]))
            return return_code == 0, "\n".join(logs)
    except Exception as e:
        logs.append(f"[ERROR] Failed to run training: {e}")
        log_placeholder.text("\n".join(logs[-200:]))
        return False, "\n".join(logs)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🏀 NBA Game Predictor")

    st.markdown("### 🚀 One-click Training")
    st.caption("Run full training from the UI and print live training output.")
    if st.button("Start Training (`agent.py`)", type="primary"):
        success, full_log = run_training_with_live_logs()
        st.session_state["last_training_log"] = full_log
        if success:
            st.success("Training completed successfully. Reloading artifacts...")
            load_artifacts.clear()
            load_nba_teams.clear()
            fetch_next_game.clear()
            load_upcoming_games.clear()
            st.rerun()
        else:
            st.error("Training failed. See log output below.")

    if st.session_state.get("last_training_log"):
        with st.expander("Last training output", expanded=False):
            st.text(st.session_state["last_training_log"][-20000:])

    st.divider()

    # ── Load everything ───────────────────────────────────────────────────
    model, metrics, team_stats, preprocessor, ensemble = load_artifacts()
    nba_teams = load_nba_teams()

    if not model or not team_stats:
        st.error("Model artifacts not found. Run `python agent.py` first to train the model.")
        return

    # ── Auto-settle predictions on load ───────────────────────────────────
    predictions, n_settled = _settle_on_load()
    if n_settled > 0:
        st.toast(f"Settled {n_settled} prediction(s) with actual results!", icon="✅")

    # ── Sidebar: model info ───────────────────────────────────────────────
    st.sidebar.header("Model Info")
    st.sidebar.metric("Model", metrics.get("model", "?").split("(")[0])
    st.sidebar.metric("AUC", f"{metrics.get('AUC', 0):.4f}")
    st.sidebar.metric("LogLoss", f"{metrics.get('LogLoss', 0):.4f}")
    st.sidebar.metric("Accuracy", f"{metrics.get('Accuracy', 0):.1%}")

    pred_count = len(predictions)
    settled = [p for p in predictions if p.get("status") == "settled"]
    correct = [p for p in settled if p.get("correct")]
    if settled:
        acc = len(correct) / len(settled)
        st.sidebar.markdown("---")
        st.sidebar.metric("Predictions Made", pred_count)
        st.sidebar.metric("Settled / Correct", f"{len(settled)} / {len(correct)}")
        st.sidebar.metric("Prediction Accuracy", f"{acc:.1%}")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔮 Predict Next Game",
        "📊 Prediction History",
        "📋 Experiment Report",
    ])

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1 — PREDICT NEXT GAME
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Select a team to predict their next game")

        # Team dropdown
        if nba_teams:
            team_list = sorted(nba_teams.keys())
        else:
            # Fallback: derive from team_stats + CSV
            team_list = sorted(set(
                _id_to_abbr(tid) for tid in team_stats.keys()
            ))

        selected_abbr = st.selectbox(
            "Choose your team",
            options=team_list,
            format_func=lambda a: f"{a} — {nba_teams[a]['full_name']}" if a in nba_teams else a,
        )

        if selected_abbr:
            col_fetch, col_clear = st.columns([3, 1])
            with col_clear:
                if st.button("🔄 Clear Cache", help="Clear schedule cache if games aren't showing"):
                    fetch_next_game.clear()
                    st.rerun()
            
            # Try fetching with cache first
            next_game = fetch_next_game(selected_abbr)
            
            # If cache returned None, try once more without cache (in case cache is stale)
            if next_game is None:
                try:
                    from nba_fetch import get_team_next_game
                    fresh_result = get_team_next_game(selected_abbr)
                    if fresh_result:
                        # Fresh fetch succeeded, clear cache and use fresh result
                        fetch_next_game.clear()
                        next_game = fresh_result
                except Exception:
                    pass  # Will show error in debug info

            if next_game is None:
                st.warning(
                    f"No upcoming game found for **{selected_abbr}**. "
                    "The season may be over, or the schedule API is temporarily unavailable."
                )
                st.caption("💡 Try: Click 'Clear Cache' above, refresh the page, or check if the NBA season is active.")
                
                # Debug: show what teams are available
                with st.expander("🔍 Debug Info"):
                    st.write(f"Selected team: `{selected_abbr}`")
                    st.write(f"Team list length: {len(team_list)}")
                    st.write(f"First 5 teams: {team_list[:5]}")
                    try:
                        from nba_fetch import get_team_upcoming_games
                        upcoming = get_team_upcoming_games(selected_abbr, limit=3)
                        if upcoming:
                            st.success(f"✅ Found {len(upcoming)} upcoming games via alternative method")
                            for g in upcoming:
                                st.write(f"  - {g['game_date']}: {g['home_team']} vs {g['away_team']}")
                        else:
                            st.write("❌ Alternative method also found nothing")
                    except Exception as e:
                        st.error(f"❌ Schedule fetch error: {e}")
            else:
                game_date = next_game["game_date"]
                home_abbr = next_game["home_team"]
                away_abbr = next_game["away_team"]
                is_home = selected_abbr.upper() == home_abbr.upper()
                opponent = away_abbr if is_home else home_abbr

                # Display matchup info
                col_info, col_action = st.columns([3, 1])
                with col_info:
                    st.markdown(f"### {home_abbr} vs {away_abbr}")
                    st.markdown(
                        f"**Date:** {game_date}  \n"
                        f"**Your team:** {selected_abbr} ({'Home' if is_home else 'Away'})  \n"
                        f"**Opponent:** {opponent}  \n"
                        f"**Status:** {next_game.get('game_status_text', 'Scheduled')}"
                    )

                # Check if already predicted
                existing_pred = next(
                    (p for p in predictions
                     if p.get("game_id") == next_game["game_id"]
                     and p.get("predicted_for_team", "").upper() == selected_abbr.upper()),
                    None,
                )

                if existing_pred:
                    st.success(
                        f"You already predicted this game!  \n"
                        f"**Predicted winner:** {existing_pred['predicted_winner']}  \n"
                        f"**Win probability:** {existing_pred['win_probability']:.1%}  \n"
                        f"**Status:** {existing_pred['status']}"
                    )
                    if existing_pred.get("status") == "settled":
                        if existing_pred.get("correct"):
                            st.balloons()
                            st.success(f"Result: {existing_pred['actual_winner']} won — your prediction was CORRECT!")
                        else:
                            st.error(f"Result: {existing_pred['actual_winner']} won — your prediction was wrong.")
                else:
                    with col_action:
                        st.write("")  # spacing
                        predict_btn = st.button(
                            "🔮 Predict!", type="primary"
                        )

                    if predict_btn:
                        home_id = next_game["home_team_id"]
                        away_id = next_game["away_team_id"]

                        with st.spinner("Running model prediction..."):
                            win_prob_home, model_name = run_prediction(
                                home_id, away_id,
                                model, metrics, team_stats, preprocessor, ensemble,
                            )

                        if win_prob_home is None:
                            st.error(
                                "Could not generate prediction — team stats may be "
                                "missing. Run `python agent.py` to refresh data."
                            )
                        else:
                            # Determine winner & team-specific probability
                            if is_home:
                                team_win_prob = win_prob_home
                            else:
                                team_win_prob = 1.0 - win_prob_home

                            predicted_winner = home_abbr if win_prob_home > 0.5 else away_abbr

                            # Save prediction (one per game_id: remove existing for this game first)
                            predictions = [p for p in predictions if p.get("game_id") != next_game["game_id"]]
                            pred_record = {
                                "id": str(uuid.uuid4()),
                                "created_at": datetime.now().isoformat(),
                                "game_id": next_game["game_id"],
                                "game_date": game_date,
                                "home_team": home_abbr,
                                "away_team": away_abbr,
                                "predicted_for_team": selected_abbr,
                                "win_probability": round(team_win_prob, 4),
                                "home_win_probability": round(win_prob_home, 4),
                                "predicted_winner": predicted_winner,
                                "model_used": model_name,
                                "actual_winner": None,
                                "correct": None,
                                "status": "pending",
                            }
                            predictions.append(pred_record)
                            _save_predictions(predictions)

                            # Display result
                            st.divider()
                            r1, r2 = st.columns([1, 2])

                            with r1:
                                st.metric(
                                    f"{selected_abbr} Win Probability",
                                    f"{team_win_prob:.1%}",
                                )
                                if team_win_prob > 0.5:
                                    st.success(f"**{selected_abbr}** is favored!")
                                elif team_win_prob < 0.5:
                                    st.warning(f"**{opponent}** is favored.")
                                else:
                                    st.info("This is a coin flip!")

                                st.caption(f"Model: {model_name}")

                            with r2:
                                _show_comparison(
                                    home_abbr, away_abbr,
                                    next_game["home_team_id"],
                                    next_game["away_team_id"],
                                    team_stats,
                                )

                            st.success("Prediction saved! Check the History tab after the game.")

        # ── Batch predict: select multiple upcoming games ─────────────────────
        st.divider()
        st.subheader("📋 Batch predict multiple games")
        st.caption("Select upcoming games below, then click to predict all selected (from home-team perspective).")
        col_batch, col_refresh = st.columns([4, 1])
        with col_refresh:
            if st.button("🔄 Refresh list", help="Reload upcoming games from schedule"):
                load_upcoming_games.clear()
                st.rerun()
        upcoming = load_upcoming_games(7)
        predicted_game_ids = set(p.get("game_id") for p in predictions)
        games_to_show = [g for g in upcoming if g.get("game_id") and g["game_id"] not in predicted_game_ids]
        MAX_BATCH = 50
        games_to_show = games_to_show[:MAX_BATCH]
        if len(upcoming) > MAX_BATCH:
            st.caption(f"Showing next {MAX_BATCH} games (clear cache to refresh).")

        if not games_to_show:
            if not upcoming:
                st.info("No upcoming games in the next 7 days, or schedule could not be loaded.")
            else:
                st.info("All upcoming games in the next 7 days already have a prediction.")
        else:
            selected_games = []
            for g in games_to_show:
                row_key = f"batch_{g['game_id']}"
                matchup_str = f"{g['home_team']} vs {g['away_team']}"
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    st.text(g["game_date"])
                with c2:
                    st.text(matchup_str)
                with c3:
                    if st.checkbox("Select", key=row_key, label_visibility="collapsed"):
                        selected_games.append(g)
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                batch_selected_btn = st.button(
                    "🔮 Predict selected games",
                    type="primary",
                    disabled=(len(selected_games) == 0),
                )
            with btn_col2:
                batch_all_btn = st.button(
                    "⚡ Predict all listed games",
                    disabled=(len(games_to_show) == 0),
                )

            games_to_predict = None
            if batch_selected_btn and selected_games:
                games_to_predict = selected_games
            elif batch_all_btn:
                games_to_predict = games_to_show

            if games_to_predict:
                predictions = _load_predictions()
                game_ids_to_add = {g["game_id"] for g in games_to_predict}
                predictions = [p for p in predictions if p.get("game_id") not in game_ids_to_add]
                model_name_used = None
                saved = 0
                failed = 0
                with st.spinner(f"Running predictions for {len(games_to_predict)} game(s)..."):
                    for g in games_to_predict:
                        win_prob_home, model_name = run_prediction(
                            g["home_team_id"], g["away_team_id"],
                            model, metrics, team_stats, preprocessor, ensemble,
                        )
                        model_name_used = model_name
                        if win_prob_home is None:
                            failed += 1
                            continue
                        predicted_winner = g["home_team"] if win_prob_home > 0.5 else g["away_team"]
                        # Probability that the predicted winner actually wins
                        team_win_prob = win_prob_home if predicted_winner == g["home_team"] else 1.0 - win_prob_home
                        pred_record = {
                            "id": str(uuid.uuid4()),
                            "created_at": datetime.now().isoformat(),
                            "game_id": g["game_id"],
                            "game_date": g["game_date"],
                            "home_team": g["home_team"],
                            "away_team": g["away_team"],
                            "predicted_for_team": predicted_winner,
                            # win_probability always refers to the predicted winner
                            "win_probability": round(team_win_prob, 4),
                            "home_win_probability": round(win_prob_home, 4),
                            "predicted_winner": predicted_winner,
                            "model_used": model_name,
                            "actual_winner": None,
                            "correct": None,
                            "status": "pending",
                        }
                        predictions.append(pred_record)
                        saved += 1
                    if saved > 0:
                        _save_predictions(predictions)
                if saved > 0:
                    st.success(f"Saved {saved} prediction(s). Check the History tab.")
                    if failed:
                        st.warning(f"{failed} game(s) could not be predicted (missing team stats).")
                    st.rerun()
                else:
                    st.error("No predictions could be saved. Team stats may be missing — try running `python agent.py`.")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2 — PREDICTION HISTORY
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Prediction History")

        predictions = _load_predictions()  # reload to include any just-saved
        predictions = _dedupe_predictions_by_game_id(predictions)

        if not predictions:
            st.info("No predictions yet. Go to the Predict tab to make your first!")
        else:
            # Sort by game date: most recent first
            predictions_sorted = sorted(
                predictions,
                key=lambda p: (p.get("game_date") or "", p.get("created_at") or ""),
                reverse=True,
            )

            # Summary metrics (over filtered set later)
            total = len(predictions_sorted)
            settled_preds = [p for p in predictions_sorted if p.get("status") == "settled"]
            pending_preds = [p for p in predictions_sorted if p.get("status") == "pending"]
            correct_preds = [p for p in settled_preds if p.get("correct")]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Predictions", total)
            m2.metric("Settled", len(settled_preds))
            m3.metric("Pending", len(pending_preds))
            if settled_preds:
                m4.metric("Accuracy", f"{len(correct_preds)/len(settled_preds):.1%}")
            else:
                m4.metric("Accuracy", "N/A")

            st.divider()

            # Date filter: unique dates from predictions (most recent first)
            unique_dates = sorted(
                list({p.get("game_date") for p in predictions_sorted if p.get("game_date")}),
                reverse=True,
            )
            col_filter, col_clean = st.columns([3, 1])
            with col_filter:
                filter_dates = st.multiselect(
                    "Filter by date (leave empty to show all)",
                    options=unique_dates,
                    default=[],
                    format_func=lambda x: x,
                )
            with col_clean:
                raw_count = len(_load_predictions())
                if raw_count > len(predictions_sorted):
                    if st.button("🧹 Clean duplicates", help="Remove duplicate entries (same game, home/away) from saved history"):
                        deduped = _dedupe_predictions_by_game_id(_load_predictions())
                        _save_predictions(deduped)
                        st.success("Duplicates removed.")
                        st.rerun()
            if filter_dates:
                predictions_filtered = [p for p in predictions_sorted if p.get("game_date") in filter_dates]
            else:
                predictions_filtered = predictions_sorted

            if not predictions_filtered:
                st.info("No predictions match the selected date(s).")
            else:
                # Build table (already sorted: most recent first)
                rows = []
                for p in predictions_filtered:
                    status = p.get("status", "pending")
                    if status == "settled":
                        result_icon = "✅" if p.get("correct") else "❌"
                        actual = p.get("actual_winner", "?")
                        score_str = ""
                        if p.get("home_score") is not None and p.get("away_score") is not None:
                            score_str = f"  ({p['home_score']}-{p['away_score']})"
                    else:
                        result_icon = "⏳"
                        actual = "Pending"
                        score_str = ""

                    # Derive probability for the predicted winner.
                    home_prob = p.get("home_win_probability")
                    if home_prob is None:
                        # Fallback for older records that may not have home_win_probability
                        home_prob = p.get("win_probability", 0.0)
                    if p.get("predicted_winner") == p.get("home_team"):
                        team_prob = home_prob
                    elif p.get("predicted_winner") == p.get("away_team"):
                        team_prob = 1.0 - home_prob
                    else:
                        # Unknown mapping; fall back to stored win_probability
                        team_prob = p.get("win_probability", home_prob)

                    rows.append({
                        "Date": p.get("game_date", "?"),
                        "Matchup": f"{p.get('home_team', '?')} vs {p.get('away_team', '?')}",
                        "Predicted": p.get("predicted_winner", "?"),
                        "Win %": f"{team_prob:.1%}",
                        "Actual": f"{actual}{score_str}",
                        "Result": result_icon,
                    })

                df_history = pd.DataFrame(rows)
                st.dataframe(df_history, hide_index=True)
                st.caption(f"Showing {len(predictions_filtered)} prediction(s) — most recent first.")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 3 — EXPERIMENT REPORT
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Model Training Report")
        if os.path.exists("outputs/report.md"):
            with open("outputs/report.md", "r") as f:
                st.markdown(f.read())
        else:
            st.info("No report available. Run `python agent.py` to generate one.")

        # Show calibration curves if they exist
        cal_images = [
            f for f in os.listdir(config.OUTPUTS_DIR)
            if f.startswith("calibration_") and f.endswith(".png")
        ]
        if cal_images:
            st.subheader("Calibration Curves")
            cols = st.columns(min(len(cal_images), 3))
            for i, img_name in enumerate(cal_images):
                with cols[i % 3]:
                    st.image(
                        os.path.join(config.OUTPUTS_DIR, img_name),
                        caption=img_name.replace("calibration_", "").replace(".png", ""),
                    )


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _id_to_abbr(team_id):
    """Best-effort team_id -> abbreviation."""
    try:
        from nba_fetch import _team_id_to_abbr
        mapping = _team_id_to_abbr()
        return mapping.get(str(team_id), str(team_id))
    except Exception:
        return str(team_id)


def _show_comparison(home_abbr, away_abbr, home_id, away_id, team_stats):
    """Show a stat comparison table for two teams."""
    h_stats = team_stats.get(str(home_id), team_stats.get(home_id, {}))
    a_stats = team_stats.get(str(away_id), team_stats.get(away_id, {}))

    if not h_stats or not a_stats:
        st.caption("Detailed stats unavailable for this matchup.")
        return

    comp_data = {
        "Metric": [
            "Win Rate (Roll)", "Elo Rating", "Rest Days",
            "Points (Roll)", "eFG% (Roll)", "TOV% (Roll)", "FT Rate (Roll)",
        ],
        home_abbr: [
            f"{h_stats.get('ROLL_WIN_RATE', h_stats.get('ROLL_WIN_PCT', 0)):.2f}",
            f"{h_stats.get('ELO_PRE', 1500):.0f}",
            str(h_stats.get("REST_DAYS", 0)),
            f"{h_stats.get('ROLL_PTS_AVG', h_stats.get('ROLL_PTS', 0)):.1f}",
            f"{h_stats.get('ROLL_EFG_PCT', 0):.3f}",
            f"{h_stats.get('ROLL_TOV_PCT', 0):.3f}",
            f"{h_stats.get('ROLL_FT_RATE', 0):.3f}",
        ],
        away_abbr: [
            f"{a_stats.get('ROLL_WIN_RATE', a_stats.get('ROLL_WIN_PCT', 0)):.2f}",
            f"{a_stats.get('ELO_PRE', 1500):.0f}",
            str(a_stats.get("REST_DAYS", 0)),
            f"{a_stats.get('ROLL_PTS_AVG', a_stats.get('ROLL_PTS', 0)):.1f}",
            f"{a_stats.get('ROLL_EFG_PCT', 0):.3f}",
            f"{a_stats.get('ROLL_TOV_PCT', 0):.3f}",
            f"{a_stats.get('ROLL_FT_RATE', 0):.3f}",
        ],
    }
    comp_df = pd.DataFrame(comp_data).astype(str)
    st.table(comp_df)


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
