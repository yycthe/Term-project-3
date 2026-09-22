"""Explain the implemented data pipeline without fetching data or retraining."""

from html import escape

import pandas as pd
import streamlit as st

import config


def preparation_steps(summary, metrics):
    """Build the overview from the current CSV profile and saved model settings."""
    rows = f"{summary['rows']:,} team-game rows" if summary else "Historical team-game rows"
    games = f"{summary['games']:,} games" if summary else "One record per matched game"
    window = metrics.get("window_years")
    history = f"Previous {window} seasons" if window is not None else "Earlier seasons"
    selected = len(metrics.get("feature_names", []))
    return [
        ("01", "Collect", rows,
         "Read the historical CSV. At training start, request newer regular-season games from the NBA API."),
        ("02", "Clean", "Dates, labels & result fields",
         "Sort by date, remove rows with missing outcomes, and screen out fields that reveal the game's result."),
        ("03", "Build pre-game features", f"Previous {config.ROLL_WINDOW_DEFAULT} games + Elo",
         "Compute shifted rolling form, pre-game Elo and rest days. Keep home/away context."),
        ("04", "Create matchups", games,
         "Join home and away rows by game, date and season. Add team differences; the label is a home-team win."),
        ("05", "Split by season", history,
         "Use the selected history window for training and the latest season for evaluation. Preserve time order."),
        ("06", "Select & scale", f"{selected} selected inputs" if selected else "Selected model inputs",
         "Rank features on training data, remove constant inputs, then fit mean imputation and standardization."),
    ]


def render_data_preparation(summary, metrics):
    with st.container(border=True):
        st.subheader("Data preparation")
        st.caption("How game logs become model inputs · the current implementation, from collection to prediction.")
        if summary:
            coverage = f"{summary['rows']:,} rows · {summary['columns']} CSV columns"
            if summary.get("teams") is not None:
                coverage += f" · {summary['teams']} teams"
            st.markdown(f"**{coverage}**")
        st.html("""<style>
        .nba-prep-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:6px 0 12px;}
        .nba-prep-step {padding:18px; border:1px solid rgba(128,150,175,.24); border-radius:12px;
          background:linear-gradient(130deg,rgba(70,215,182,.055),rgba(123,169,255,.035));}
        .nba-prep-number {color:#46bfa6; font-size:12px; letter-spacing:2px; font-weight:700;}
        .nba-prep-step h4 {font-size:17px; padding:8px 0; margin:0; line-height:1.35;}
        .nba-prep-value {font-size:13px; font-weight:600; margin:0 0 8px;}
        .nba-prep-step p {font-size:13px; line-height:1.6; opacity:.76; margin:0;}
        @media(max-width:900px) {.nba-prep-grid {grid-template-columns:repeat(2,minmax(0,1fr));}}
        @media(max-width:560px) {.nba-prep-grid {grid-template-columns:1fr;}}
        </style>""")
        cards = []
        for number, title, value, body in preparation_steps(summary, metrics):
            cards.append(f"""<article class="nba-prep-step">
              <div class="nba-prep-number">{number}</div><h4>{escape(title)}</h4>
              <div class="nba-prep-value">{escape(value)}</div><p>{escape(body)}</p></article>""")
        st.html('<div class="nba-prep-grid">' + "".join(cards) + '</div>')

        with st.expander("Source, cleaning & game structure"):
            st.markdown(f"""
**Source and refresh.** The input is `{config.DATA_PATH}`, an existing historical CSV.
At training start, `nba_api.LeagueGameLog` requests the configured current regular season.
Only rows with a date later than the CSV's latest date and a game ID not already present are appended.
New rows are aligned to the CSV's columns; unavailable fields start as missing values.
This refresh does not rebuild or backfill the entire historical dataset.

**Cleaning.** Dates are parsed and sorted. Rows with missing outcomes are removed;
text win/loss labels are mapped to 1/0 when needed. The code first derives effective
field-goal percentage, turnover rate and free-throw rate for later rolling averages.
It then screens columns using result-related names (such as points and box-score fields)
and absolute correlation above 0.90 with the target.

**One row per game.** A source row describes one team's appearance in a game.
`vs.` identifies the home team; `@` identifies the away team. The two sides are joined
on game ID, date and season, with `HOME_` and `AWAY_` feature prefixes.
The prediction target is **1 when the home team wins, 0 when it loses**.
IDs, the season field and direct outcome columns are excluded from candidate model inputs.
""")
            st.caption("The repository contains the historical CSV and its update code; the original multi-season collection script is not included.")

        with st.expander("Pre-game feature calculations"):
            roll = config.ROLL_WINDOW_DEFAULT
            elo_k = metrics.get("elo_k", config.ELO_K)
            definitions = pd.DataFrame([
                ("Recent form", f"Per team: shift results by one game, then average up to the previous {roll} games. At least one prior game is needed."),
                ("Shooting & possession", "eFG% = (FGM + 0.5 × 3PM) / FGA; turnover rate = TOV / (FGA + 0.44 × FTA + TOV); free-throw rate = FTM / FGA. Then shift and roll."),
                ("Elo strength", f"Start each team at {config.ELO_BASE}. Store the rating before each game, then update from its result (selected K = {elo_k})."),
                ("Rest days", "Days since the team's previous game; use 3 days for the first appearance and clip to 0–10 days."),
                ("Matchup contrasts", "Home minus away for whitelisted stats; selected rest/win-rate ratios use a 0.01 offset. Elo advantage = home Elo minus away Elo."),
            ], columns=["Feature group", "Calculation"])
            st.dataframe(definitions, hide_index=True, width="stretch", height=440, row_height=80,
                         column_config={"Feature group": st.column_config.TextColumn(width="small"),
                                        "Calculation": st.column_config.TextColumn(width="large")})
            st.code(f"team_history.shift(1).rolling({roll}, min_periods=1).mean()", language="python")
            st.caption("For a game on day D, recomputed rolling features use earlier games only. Elo is saved before applying day D's result. Earlier completed evaluation-season games can inform later games without refitting the model.")
            st.info("Existing ROLL_* columns in the CSV are retained. The training code recomputes only its implemented feature set, where source fields are available; it does not regenerate every historical rolling column.")

        with st.expander("Season split, feature selection & missing values"):
            if summary:
                st.markdown(f"**Current split:** {summary['train']:,} training games → {summary['evaluation']:,} evaluation games in **{summary['season']}**. Earlier seasons outside the selected window are excluded from model fitting.")
            st.markdown("""
**Time order.** The latest season is used for evaluation; previous seasons form the
training window. If season labels are unavailable, the split falls back to the first
80% / last 20% in date order.

**Feature selection.** For Auto Select, feature-type switches first restrict candidates.
Training-only scores combine normalized Random Forest importance (40%), XGBoost
importance (40%) and absolute target correlation (20%). The selector temporarily fills
missing values with zero; selected inputs with near-zero training variance are removed.

**Model preprocessing.** Infinite input values become missing values. A mean imputer
and StandardScaler are fitted on the training partition: fill with each feature's
training mean, then subtract that mean and divide by its training standard deviation.
These fitted values are reused for evaluation and prediction. Any remaining non-finite transformed values are sanitized
and extreme values clipped before model fitting. The current training call uses numeric
inputs; it does not one-hot encode team names.
""")
            st.caption("The earlier leakage-screening pass runs on the full CSV before splitting, and protects existing rolling columns. It is a heuristic check. The latest season is also reused for tuning and model selection, so these scores are not an untouched final test.")

        with st.expander("What is saved for live predictions"):
            st.markdown("""
Training saves the selected model, fitted preprocessor, selected feature names and
one latest historical feature row per team. The prediction page loads that snapshot,
builds the home/away matchup, and applies the saved transformation.

The team snapshot is refreshed when training finishes. Viewing the page or settling
a prediction does not rebuild the training dataset or retrain the model. Some legacy
rolling columns are omitted from the exported snapshot; the current prediction code
fills absent model columns with zero. The **Data snapshot** panel below reports this
coverage for the selected model.
""")
