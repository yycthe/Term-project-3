"""Read-only recovery of the previously verified public history table.

Captured on 2026-09-21 from the deployed app, before Firestore reads hit quota.
This is display evidence, not a replacement for original Firestore documents:
record IDs, creation timestamps, scores and full-precision probabilities were
not preserved. Never upload these rows into the live prediction collection.
"""

import csv
from pathlib import Path


def load_verified_history_snapshot():
    path = Path(__file__).parent / "outputs" / "verified_prediction_history_2026-09-21.csv"
    if not path.exists():
        return []
    with path.open() as source:
        rows = list(csv.DictReader(source))
    return [{
        "game_date": row["date"], "home_team": row["home"], "away_team": row["away"],
        "predicted_winner": row["predicted"], "actual_winner": row["actual"],
        "displayed_win_probability": float(row["probability"]),
        "correct": row["correct"] == "1", "status": "settled",
        "_history_snapshot": True,
    } for row in rows]
