# Firebase Setup (Persistent Memory)

Use this setup to prevent memory loss after cloud sleep/restart.

## 1) Install dependency

Already added to `requirements.txt`:

- `firebase-admin`

## 2) What to store in Firebase

- Prediction history (`predictions`)
- Agent memory (`meta/agent_memory`)

The app now reads/writes these through `storage.py`.

## 3) Where to put API credentials

Put these as environment variables (local shell, Render, Streamlit secrets -> env):

- `USE_FIREBASE=1`
- `FIREBASE_PROJECT_ID=<your-firebase-project-id>`
- `FIREBASE_SERVICE_ACCOUNT_JSON=<full service account JSON as one line>`

Optional alternative:

- `FIREBASE_SERVICE_ACCOUNT_B64=<base64 of service account JSON>`

Notes:
- You only need one of `FIREBASE_SERVICE_ACCOUNT_JSON` or `FIREBASE_SERVICE_ACCOUNT_B64`.
- Do NOT commit raw service account JSON to git.

## 4) Marked code locations

- Config/env keys: `config.py`
- Firebase + fallback logic: `storage.py`
- Prediction persistence calls: `app.py` (`_load_predictions`, `_save_predictions`)
- Agent memory persistence calls: `agent.py` (`_load_memory`, `_save_memory`)

## 5) Local test

### Prediction history during an outage

Cloud history reads are shared across sessions in the same app process for 15 minutes.
Failed reads also back off for 15 minutes; a later page run retries automatically.
A successful read is mirrored atomically to an ignored, project-scoped local cache.
That file is an outage aid, not durable storage across Streamlit redeployments.

If no full cached history survives, the UI can display the verified 120-row public
table captured on 2026-09-21. This read-only CSV preserves the displayed predictions
and results, not original document IDs, timestamps, scores or full-precision values.
It is never uploaded into Firestore. A successful cloud read always takes precedence.

While cloud history cannot be read, editing and saving are blocked. Normal saves
upsert changed documents only; missing rows in an incoming list never delete other
history. User-requested deletions use a separate explicit-ID operation.

Run the outage and write-protection checks with:

```bash
python -m unittest discover -s tests -v
```

Run:

```bash
conda activate nba-predictor
python agent.py
streamlit run app.py
```

If credentials are missing or invalid, code falls back to local JSON files in `outputs/`.
