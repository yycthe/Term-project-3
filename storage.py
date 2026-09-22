import base64
import json
import os
import copy
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

_FIREBASE_READY = False
_FIREBASE_ERROR: Optional[str] = None
_DB = None
_PREDICTIONS_CACHE = None
_PREDICTIONS_DOC_IDS = {}
_PREDICTIONS_RETRY_AT = 0.0
_PREDICTIONS_ERROR = None
_PREDICTIONS_SOURCE = "unavailable"
_PREDICTIONS_LOCK = threading.RLock()
PREDICTIONS_CACHE_SECONDS = 900


class PredictionStorageError(RuntimeError):
    """A prediction mutation was not safely persisted."""


def get_storage_status() -> Dict[str, Any]:
    """Return current backend status for UI/debug."""
    enabled = _firebase_enabled()
    connected = _ensure_firestore() if enabled else False
    error = _PREDICTIONS_ERROR or _FIREBASE_ERROR
    return {
        "firebase_enabled": enabled,
        "firebase_connected": connected and not error,
        "firebase_error": error,
        "predictions_source": _PREDICTIONS_SOURCE,
        "predictions_writable": not enabled or (_PREDICTIONS_SOURCE == "firestore" and not error),
    }


def _get_secret_value(key: str) -> str:
    """Read from env first, then Streamlit secrets if available."""
    val = os.getenv(key, "").strip()
    if val:
        return val
    try:
        import streamlit as st
        s_val = st.secrets.get(key, "")
        return str(s_val).strip() if s_val else ""
    except Exception:
        return ""


def _firebase_enabled() -> bool:
    flag = _get_secret_value("USE_FIREBASE").lower()
    return flag in {"1", "true", "yes", "on"}


def _read_service_account_dict() -> Optional[Dict[str, Any]]:
    """
    Load Firebase service account from env vars.

    Supported formats:
    - FIREBASE_SERVICE_ACCOUNT_JSON: raw JSON string
    - FIREBASE_SERVICE_ACCOUNT_B64: base64-encoded JSON string
    """
    raw_json = _get_secret_value(config.FIREBASE_SERVICE_ACCOUNT_JSON_ENV)
    if raw_json:
        try:
            return json.loads(raw_json)
        except Exception:
            return None

    b64_json = _get_secret_value(config.FIREBASE_SERVICE_ACCOUNT_B64_ENV)
    if b64_json:
        try:
            decoded = base64.b64decode(b64_json).decode("utf-8")
            return json.loads(decoded)
        except Exception:
            return None

    return None


def _ensure_firestore() -> bool:
    global _FIREBASE_READY, _FIREBASE_ERROR, _DB
    if _FIREBASE_READY:
        return True
    if not _firebase_enabled():
        _FIREBASE_ERROR = "Firebase disabled by config."
        return False

    project_id = _get_secret_value(config.FIREBASE_PROJECT_ID_ENV)
    creds_dict = _read_service_account_dict()
    if not project_id or not creds_dict:
        _FIREBASE_ERROR = "Missing Firebase project id or service account json."
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(creds_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"projectId": project_id})
        _DB = firestore.client()
        _FIREBASE_READY = True
        return True
    except Exception as e:
        _FIREBASE_ERROR = str(e)
        return False


def _prediction_key(row):
    return str(row.get("game_id") or row.get("id") or "")


def _read_json_file(path, default):
    try:
        with open(path) as source:
            value = json.load(source)
        return value if isinstance(value, type(default)) else default
    except (OSError, ValueError):
        return default


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as target:
            temporary = target.name
            json.dump(value, target, indent=2, default=str)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _cloud_mirror_path():
    return Path(config.OUTPUTS_DIR) / "predictions_firestore_cache.json"


def _mirror_cloud_history(rows):
    # Cloud data has its own mirror; never confuse it with local development history.
    try:
        _atomic_json(_cloud_mirror_path(), {
            "project": _get_secret_value(config.FIREBASE_PROJECT_ID_ENV),
            "collection": config.FIREBASE_PREDICTIONS_COLLECTION,
            "predictions": rows,
        })
    except OSError:
        pass  # A read-only filesystem must not turn a successful cloud read into failure.


def load_predictions() -> List[Dict[str, Any]]:
    """Share one cloud read per 15 minutes; retain the last good data on failure."""
    global _PREDICTIONS_CACHE, _PREDICTIONS_DOC_IDS, _PREDICTIONS_RETRY_AT
    global _PREDICTIONS_ERROR, _PREDICTIONS_SOURCE, _FIREBASE_ERROR
    if not _firebase_enabled():
        _PREDICTIONS_SOURCE = "local"
        return _read_json_file(config.PREDICTIONS_FILE, [])
    with _PREDICTIONS_LOCK:
        if time.monotonic() < _PREDICTIONS_RETRY_AT:
            return copy.deepcopy(_PREDICTIONS_CACHE or [])
        try:
            if not _ensure_firestore():
                raise PredictionStorageError(_FIREBASE_ERROR or "Firebase initialization failed")
            rows, document_ids = [], {}
            docs = _DB.collection(config.FIREBASE_PREDICTIONS_COLLECTION).stream(timeout=10, retry=None)
            for doc in docs:
                row = doc.to_dict()
                if row:
                    rows.append(row)
                    document_ids.setdefault(_prediction_key(row), []).append(doc.id)
            _PREDICTIONS_CACHE, _PREDICTIONS_DOC_IDS = rows, document_ids
            _PREDICTIONS_ERROR = _FIREBASE_ERROR = None
            _PREDICTIONS_SOURCE = "firestore"
            _mirror_cloud_history(rows)
        except Exception as exc:
            _PREDICTIONS_ERROR = f"load_predictions failed: {exc}"
            if _PREDICTIONS_CACHE is None:
                mirror = _read_json_file(_cloud_mirror_path(), {})
                if (mirror.get("project") == _get_secret_value(config.FIREBASE_PROJECT_ID_ENV)
                        and mirror.get("collection") == config.FIREBASE_PREDICTIONS_COLLECTION
                        and isinstance(mirror.get("predictions"), list)):
                    _PREDICTIONS_CACHE = mirror["predictions"]
            _PREDICTIONS_SOURCE = "cache" if _PREDICTIONS_CACHE is not None else "unavailable"
        _PREDICTIONS_RETRY_AT = time.monotonic() + PREDICTIONS_CACHE_SECONDS
        return copy.deepcopy(_PREDICTIONS_CACHE or [])


def _require_live_history():
    rows = load_predictions()
    if _PREDICTIONS_ERROR or _PREDICTIONS_SOURCE != "firestore":
        raise PredictionStorageError("Cloud history is unavailable. Changes are paused to protect saved records.")
    return rows


def _commit_prediction_changes(upserts, deleted_ids=()):
    global _PREDICTIONS_ERROR, _PREDICTIONS_RETRY_AT
    coll = _DB.collection(config.FIREBASE_PREDICTIONS_COLLECTION)
    operations = [("set", key, payload) for key, payload in upserts.items()]
    operations += [("delete", key, None) for key in deleted_ids]
    try:
        for start in range(0, len(operations), 450):
            batch = _DB.batch()
            for action, key, payload in operations[start:start + 450]:
                if action == "set":
                    batch.set(coll.document(key), payload)
                else:
                    batch.delete(coll.document(key))
            batch.commit()
    except Exception as exc:
        _PREDICTIONS_ERROR = f"Prediction write failed: {exc}"
        _PREDICTIONS_RETRY_AT = time.monotonic() + PREDICTIONS_CACHE_SECONDS
        raise PredictionStorageError("Could not save the cloud change. The last verified history is preserved.") from exc


def save_predictions(preds: List[Dict[str, Any]]) -> None:
    """Upsert changed records only. Omission from a submitted list never deletes."""
    global _PREDICTIONS_CACHE
    if any(row.get("_history_snapshot") for row in preds):
        raise PredictionStorageError("A recovered display snapshot cannot be saved as live predictions.")
    if not _firebase_enabled():
        _atomic_json(config.PREDICTIONS_FILE, preds)
        return
    with _PREDICTIONS_LOCK:
        current = {_prediction_key(row): row for row in _require_live_history()}
        incoming = {_prediction_key(row): row for row in preds if _prediction_key(row)}
        changed = {key: row for key, row in incoming.items() if current.get(key) != row}
        _commit_prediction_changes(changed)
        current.update(copy.deepcopy(changed))
        _PREDICTIONS_CACHE = list(current.values())
        for key in changed:
            ids = _PREDICTIONS_DOC_IDS.setdefault(key, [])
            if key not in ids:
                ids.append(key)
        _mirror_cloud_history(_PREDICTIONS_CACHE)


def delete_predictions(game_ids) -> None:
    """Delete only records explicitly selected by the user, after a healthy read."""
    global _PREDICTIONS_CACHE
    selected = {str(value) for value in game_ids}
    with _PREDICTIONS_LOCK:
        if not _firebase_enabled():
            rows = [row for row in load_predictions() if _prediction_key(row) not in selected]
            _atomic_json(config.PREDICTIONS_FILE, rows)
            return
        rows = _require_live_history()
        known = {_prediction_key(row) for row in rows}
        if not selected.issubset(known):
            raise PredictionStorageError("Some selected records are no longer in the loaded history. Reload before deleting.")
        document_ids = {doc_id for key in selected for doc_id in _PREDICTIONS_DOC_IDS.get(key, [])}
        _commit_prediction_changes({}, document_ids)
        _PREDICTIONS_CACHE = [row for row in rows if _prediction_key(row) not in selected]
        for key in selected:
            _PREDICTIONS_DOC_IDS.pop(key, None)
        _mirror_cloud_history(_PREDICTIONS_CACHE)


def load_agent_memory() -> Dict[str, Any]:
    """Load cross-run memory from Firestore, fallback to local JSON."""
    if _ensure_firestore():
        try:
            doc = (
                _DB.collection(config.FIREBASE_META_COLLECTION)
                .document(config.FIREBASE_AGENT_MEMORY_DOC_ID)
                .get()
            )
            if doc.exists:
                data = doc.to_dict() or {}
                if isinstance(data, dict):
                    return data
        except Exception as e:
            global _FIREBASE_ERROR
            _FIREBASE_ERROR = f"load_agent_memory failed: {e}"

    if os.path.exists(config.MEMORY_FILE):
        try:
            with open(config.MEMORY_FILE, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_agent_memory(mem: Dict[str, Any]) -> None:
    """Save cross-run memory to Firestore, and also mirror to local JSON."""
    if _ensure_firestore():
        try:
            (
                _DB.collection(config.FIREBASE_META_COLLECTION)
                .document(config.FIREBASE_AGENT_MEMORY_DOC_ID)
                .set(mem)
            )
        except Exception as e:
            global _FIREBASE_ERROR
            _FIREBASE_ERROR = f"save_agent_memory failed: {e}"

    with open(config.MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)
