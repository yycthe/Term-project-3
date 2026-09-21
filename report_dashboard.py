"""Read-only dashboard for saved training results and model artifacts."""

from html import escape
from pathlib import Path
import json
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config
import evaluate
import features


FAMILIES = ["Random forest", "Logistic regression", "XGBoost"]
COLORS = ["#46d7b6", "#8b9cff", "#f2b66d"]
METRICS = {
    "Log loss": ("LogLoss", False, ".4f"),
    "AUC": ("AUC", True, ".4f"),
    "Accuracy": ("Accuracy", True, ".1%"),
}


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def parse_experiments(markdown):
    """Read the existing report's comparison table, including older reports."""
    match = re.search(r"^## Model Comparison\s*\n(.*?)(?=^## |\Z)", markdown, re.M | re.S)
    columns = ["Trial", "Model", "Configuration", "AUC", "LogLoss", "BrierScore", "Accuracy"]
    rows = []
    if match:
        for line in match.group(1).splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 7:
                continue
            try:
                trial = int(cells[1]) + 1  # Display trials starting at 1.
                scores = [float(value) for value in cells[3:]]
            except ValueError:
                continue
            if not np.isfinite(scores).all():
                continue
            raw_family = cells[0].split("(")[0].strip()
            family = {"RandomForest": "Random forest", "LogisticRegression": "Logistic regression"}.get(raw_family, raw_family)
            rows.append([trial, family, cells[0], *scores])
    return pd.DataFrame(rows, columns=columns).sort_values("Trial")


@st.cache_data(show_spinner=False)
def _dataset_summary(path, version, window_years):
    """Summarize the current CSV without fetching data or running training."""
    try:
        frame = pd.read_csv(path, usecols=lambda c: c in {"GAME_ID", "GAME_DATE", "SEASON_START_YEAR"})
        if not {"GAME_ID", "GAME_DATE", "SEASON_START_YEAR"}.issubset(frame):
            return {}
        frame = frame.drop_duplicates("GAME_ID")
        dates = pd.to_datetime(frame["GAME_DATE"], errors="coerce")
        season = int(frame["SEASON_START_YEAR"].max())
        train = frame["SEASON_START_YEAR"] < season
        if window_years is not None:
            train &= frame["SEASON_START_YEAR"] >= season - int(window_years)
        return {
            "games": len(frame), "start": dates.min().strftime("%d %b %Y"),
            "end": dates.max().strftime("%d %b %Y"),
            "train": int(train.sum()), "evaluation": int((frame["SEASON_START_YEAR"] == season).sum()),
            "season": f"{season}–{str(season + 1)[-2:]}",
        }
    except (OSError, ValueError, KeyError):
        return {}


@st.cache_data(show_spinner=False)
def _model_importance(_model, feature_names, version):
    try:
        importance = evaluate.get_feature_importance(_model, list(feature_names))
    except (ValueError, AttributeError):
        importance = None
    if importance is None:
        return pd.DataFrame(columns=["Feature", "Importance"])
    return importance.rename("Importance").rename_axis("Feature").reset_index()


def missing_snapshot_features(team_stats, feature_names):
    """Check columns supplied for every team using the app's feature builder."""
    if not team_stats:
        return list(feature_names)
    common = set.intersection(*(set(row) for row in team_stats.values()))
    row = {f"{side}_{name}": 0.0 for side in ("HOME", "AWAY") for name in common}
    row.update(HOME_IS_HOME_CALC=1, AWAY_IS_HOME_CALC=0)
    available = features.add_difference_features(pd.DataFrame([row])).columns
    return [name for name in feature_names if name not in available]


def _feature_label(name):
    if name == "ELO_DIFF":
        return "Elo advantage"
    prefix = ""
    for raw, label in [("HOME_", "Home · "), ("AWAY_", "Away · "), ("DIFF_", "Difference · "), ("RATIO_", "Ratio · ")]:
        if name.startswith(raw):
            prefix, name = label, name[len(raw):]
            break
    labels = {
        "ELO_PRE": "Elo rating", "ROLL_FG_PCT": "Field goal %", "ROLL_EFG_PCT": "Effective FG %",
        "ROLL_FG3_PCT": "3-point %", "ROLL_FT_PCT": "Free throw %", "ROLL_FT_RATE": "Free throw rate",
        "ROLL_TOV_PCT": "Turnover rate", "ROLL_WIN_PCT": "Win %", "ROLL_WIN_RATE": "Win rate",
        "ROLL_PTS": "Points", "ROLL_PTS_AVG": "Points", "ROLL_REB": "Rebounds",
        "ROLL_AST": "Assists", "ROLL_STL": "Steals", "ROLL_BLK": "Blocks",
        "ROLL_TOV": "Turnovers", "ROLL_PF": "Fouls", "REST_DAYS": "Rest days",
    }
    return prefix + labels.get(name, name.replace("_", " ").title())


def _chart_style(chart):
    return chart.configure_view(strokeWidth=0).configure_axis(
        gridOpacity=0.12, labelFontSize=11, titleFontSize=11, titlePadding=12,
    ).configure_legend(title=None, orient="top", labelFontSize=11)


def progress_chart(experiments, metric_label):
    metric, higher, fmt = METRICS[metric_label]
    frame = experiments.sort_values("Trial").copy()
    frame["Best so far"] = frame[metric].cummax() if higher else frame[metric].cummin()
    x = alt.X("Trial:Q", title="Trial", axis=alt.Axis(tickMinStep=1, format="d"))
    y = alt.Y(f"{metric}:Q", title=metric_label, scale=alt.Scale(zero=False), axis=alt.Axis(format=fmt))
    base = alt.Chart(frame)
    line = base.mark_line(color="#46d7b6", strokeWidth=2.5, interpolate="step-after", opacity=0.6).encode(
        x=x, y=alt.Y("Best so far:Q", title=metric_label, scale=alt.Scale(zero=False), axis=alt.Axis(format=fmt)),
    )
    points = base.mark_circle(size=100, opacity=0.95).encode(
        x=x, y=y, color=alt.Color("Model:N", scale=alt.Scale(domain=FAMILIES, range=COLORS)),
        tooltip=[alt.Tooltip("Trial:Q", format="d"), "Model:N", alt.Tooltip(f"{metric}:Q", format=fmt), "Configuration:N"],
    )
    return _chart_style((line + points).properties(height=340))


def importance_chart(importance, count):
    frame = importance.head(count).copy()
    frame["Label"] = frame["Feature"].map(_feature_label)
    chart = alt.Chart(frame).mark_bar(color="#7ba9ff", cornerRadiusEnd=4, size=17).encode(
        x=alt.X("Importance:Q", title="Feature importance", axis=alt.Axis(format=".3f")),
        y=alt.Y("Label:N", title=None, sort="-x", axis=alt.Axis(labelLimit=230)),
        tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".4f")],
    )
    return _chart_style(chart.properties(height=max(340, count * 29)))


def _metric_value(metrics, key, fmt):
    value = metrics.get(key)
    return format(value, fmt) if isinstance(value, (int, float)) and np.isfinite(value) else "—"


@st.fragment
def render_report_dashboard(model, metrics, team_stats, artifact_version):
    output_dir = Path(config.OUTPUTS_DIR)
    report_path = output_dir / "report.md"
    try:
        markdown = report_path.read_text()
    except OSError:
        markdown = ""
    experiments = parse_experiments(markdown)
    memory = _read_json(output_dir / "agent_memory.json")
    feature_names = tuple(metrics.get("feature_names", []))
    data_path = Path(config.DATA_PATH)
    summary = _dataset_summary(str(data_path), data_path.stat().st_mtime_ns if data_path.exists() else None, metrics.get("window_years"))
    importance = _model_importance(model, feature_names, artifact_version) if feature_names else pd.DataFrame()
    selected_trial = metrics.get("trial")
    selected_trial = int(selected_trial) + 1 if selected_trial is not None else None
    family = metrics.get("model_type", metrics.get("model", "Selected model").split("(")[0].strip())
    last_run = pd.to_datetime(memory.get("last_run"), errors="coerce")
    updated = last_run.strftime("%d %b %Y") if pd.notna(last_run) else "Not recorded"

    st.html("""<style>
    .nba-report-hero {padding:26px 28px; margin:4px 0 20px; border:1px solid rgba(70,215,182,.28);
      border-radius:16px; background:linear-gradient(115deg,rgba(70,215,182,.10),rgba(123,169,255,.04));}
    .nba-report-eyebrow {font-size:11px; letter-spacing:2px; font-weight:700; color:#46bfa6; margin-bottom:10px;}
    .nba-report-hero h2 {font-size:32px; font-weight:650; letter-spacing:-1px; margin:0 0 8px; padding:0;}
    .nba-report-hero p {opacity:.7; margin:0 0 20px; font-size:14px;}
    .nba-report-tags {display:flex; flex-wrap:wrap; gap:8px; font-size:12px;}
    .nba-report-tags span {border:1px solid rgba(128,150,175,.25); border-radius:6px; padding:5px 10px;}
    .st-key-report_kpis [data-testid="stMetricValue"] {font-size:32px;}
    </style>""")
    trial_tag = f"<span>Selected · Trial {selected_trial}</span>" if selected_trial else ""
    st.html(f"""<section class="nba-report-hero">
      <div class="nba-report-eyebrow">MODEL LAB / EXPERIMENT REPORT</div>
      <h2>Training dashboard</h2>
      <p>Explore the experiments, the selected model, and the data behind its results.</p>
      <div class="nba-report-tags"><span>{escape(str(family))}</span>{trial_tag}
        <span>{len(feature_names)} features</span><span>Last run · {escape(updated)}</span></div>
      </section>""")

    with st.container(key="report_kpis"):
        columns = st.columns(4)
        for col, label, key, fmt, note in zip(
            columns, ["Evaluation accuracy", "AUC", "Log loss", "Brier score"],
            ["Accuracy", "AUC", "LogLoss", "BrierScore"], [".1%", ".4f", ".4f", ".4f"],
            ["Share of games correctly classified", "Ranking quality · higher is better", "Probability error · lower is better", "Squared probability error · lower is better"],
        ):
            with col.container(border=True):
                st.metric(label, _metric_value(metrics, key, fmt), help=note)
                st.caption(note)
    st.caption("Selected-model evaluation during model search. These scores are separate from Prediction History and are not an independent final test.")

    left, right = st.columns([1.15, 1], gap="large")
    with left.container(border=True):
        st.subheader("Training progress")
        metric_label = st.segmented_control("Metric", list(METRICS), default="Log loss", key="report_metric", label_visibility="collapsed") or "Log loss"
        if experiments.empty:
            st.info("Experiment details are not available in the saved report.")
        else:
            st.altair_chart(progress_chart(experiments, metric_label), use_container_width=True, key=f"report_progress_{metric_label}")
            st.caption("Dots show reported trials; the step line tracks the best value seen for the selected metric.")
            attempted = memory.get("total_trials_this_run")
            st.caption(f"{len(experiments)} scored trials shown" + (f" · {attempted} total attempts in the saved run." if attempted else "."))
    with right.container(border=True):
        st.subheader("What the model uses")
        count = st.selectbox("Features to show", [5, 10, 15, 20], index=1, format_func=lambda n: f"Top {n} features", key="report_feature_count", label_visibility="collapsed")
        if importance.empty:
            st.info("Feature importance is not available for this model.")
        else:
            st.altair_chart(importance_chart(importance, count), use_container_width=True, key=f"report_importance_{count}")
            st.caption("Importance describes the selected model's inputs, not win probabilities. Calibrated models average importance across fitted base models.")

    with st.container(border=True):
        heading, filter_col = st.columns([2, 1])
        heading.subheader("Experiment leaderboard")
        families = sorted(experiments["Model"].unique().tolist())
        selected_family = filter_col.selectbox("Model family", ["All models", *families], key="report_family", label_visibility="collapsed")
        if experiments.empty:
            st.info("No scored trials have been saved yet.")
        else:
            ranked = experiments.sort_values("LogLoss").copy()
            ranked.insert(0, "Rank", range(1, len(ranked) + 1))
            ranked.insert(1, "Selected", ranked["Trial"] == selected_trial)
            if selected_family != "All models":
                ranked = ranked[ranked["Model"] == selected_family]
            st.dataframe(
                ranked, hide_index=True, width="stretch", height=310,
                column_order=["Rank", "Selected", "Trial", "Model", "Accuracy", "AUC", "LogLoss", "BrierScore"],
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                    "Selected": st.column_config.CheckboxColumn("Selected", width="small"),
                    "Trial": st.column_config.NumberColumn("Trial", format="%d", width="small"),
                    "Accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
                    "AUC": st.column_config.NumberColumn("AUC", format="%.4f"),
                    "LogLoss": st.column_config.NumberColumn("Log loss ↓", format="%.4f"),
                    "BrierScore": st.column_config.NumberColumn("Brier", format="%.4f"),
                },
            )
            st.caption("Ranked by log loss, the training objective. Only trials retained in the saved report are included.")
            st.download_button("Download experiments · CSV", experiments.to_csv(index=False), "nba_experiments.csv", "text/csv", key="report_download_trials")

    data_col, method_col = st.columns(2, gap="large")
    with data_col.container(border=True):
        st.subheader("Data snapshot")
        if summary:
            st.metric("Games in the current dataset", f"{summary['games']:,}")
            st.caption(f"{summary['start']} — {summary['end']}")
            a, b = st.columns(2)
            a.metric("Training games", f"{summary['train']:,}")
            b.metric("Evaluation games", f"{summary['evaluation']:,}")
            st.caption(f"Latest season: {summary['season']}. Counts use the current CSV and the selected model's history window.")
        else:
            st.info("Dataset coverage is unavailable.")
        missing = missing_snapshot_features(team_stats, feature_names)
        if feature_names:
            st.progress((len(feature_names) - len(missing)) / len(feature_names), text=f"Prediction snapshot · {len(feature_names) - len(missing)} / {len(feature_names)} selected-model feature columns available")
            if missing:
                st.warning(f"{len(missing)} model inputs are missing from the prediction snapshot. The current prediction flow fills missing columns with zero.")
                with st.expander("View missing inputs"):
                    st.write(", ".join(missing))
    with method_col.container(border=True):
        st.subheader("Model setup")
        setup = pd.DataFrame({
            "Setting": ["Model", "History window", "Elo update factor", "Feature strategy", "Selected features"],
            "Value": [str(family), f"{metrics.get('window_years', '—')} seasons", str(metrics.get("elo_k", "—")), str(metrics.get("strategy", "—")).replace("_", " "), str(len(feature_names))],
        })
        st.dataframe(setup, hide_index=True, width="stretch")
        if metrics.get("ensemble_auc") is not None or metrics.get("ensemble_logloss") is not None:
            st.markdown("**Ensemble evaluation**")
            a, b = st.columns(2)
            a.metric("AUC", _metric_value(metrics, "ensemble_auc", ".4f"))
            b.metric("Log loss", _metric_value(metrics, "ensemble_logloss", ".4f"))
        st.caption("The latest season is reused to select models and tune settings. A separate, untouched evaluation period is still needed.")

    calibration_files = sorted(output_dir.glob("calibration_*.png"))
    if calibration_files:
        with st.expander("Saved calibration plots"):
            st.caption("Plots saved by training runs; older runs may also be present.")
            selected_plot = st.selectbox("Calibration plot", calibration_files, format_func=lambda p: p.stem.removeprefix("calibration_").replace("_", " "), key="report_calibration")
            st.image(str(selected_plot), width="stretch")
    if markdown:
        st.download_button("Download original report · Markdown", markdown, "nba_training_report.md", "text/markdown", key="report_download_markdown")
