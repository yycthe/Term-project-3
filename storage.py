import base64
import json
import os
from typing import Any, Dict, List, Optional

import config

_FIREBASE_READY = False
_FIREBASE_ERROR: Optional[str] = None
_DB = None


def get_storage_status() -> Dict[str, Any]:
    """Return current backend status for UI/debug."""
    enabled = _firebase_enabled()
    connected = _ensure_firestore() if enabled else False
    return {
        "firebase_enabled": enabled,
        "firebase_connected": connected,
        "firebase_error": _FIREBASE_ERROR,
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
    if _FIREBASE_ERROR:
        return False

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


def load_predictions() -> List[Dict[str, Any]]:
    """Load prediction history from Firestore, fallback to local JSON."""
    if _ensure_firestore():
        try:
            docs = _DB.collection(config.FIREBASE_PREDICTIONS_COLLECTION).stream()
            rows = [d.to_dict() for d in docs if d.to_dict()]
            return rows
        except Exception as e:
            global _FIREBASE_ERROR
            _FIREBASE_ERROR = f"load_predictions failed: {e}"

    if os.path.exists(config.PREDICTIONS_FILE):
        try:
            with open(config.PREDICTIONS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_predictions(preds: List[Dict[str, Any]]) -> None:
    """Save prediction history to Firestore, and also mirror to local JSON."""
    if _ensure_firestore():
        try:
            coll = _DB.collection(config.FIREBASE_PREDICTIONS_COLLECTION)
            incoming: Dict[str, Dict[str, Any]] = {}
            for p in preds:
                doc_id = str(p.get("game_id") or p.get("id") or "")
                if not doc_id:
                    continue
                incoming[doc_id] = p

            existing_ids = set(d.id for d in coll.stream())
            incoming_ids = set(incoming.keys())
            to_delete = existing_ids - incoming_ids

            batch = _DB.batch()
            for doc_id, payload in incoming.items():
                batch.set(coll.document(doc_id), payload)
            for doc_id in to_delete:
                batch.delete(coll.document(doc_id))
            batch.commit()
        except Exception as e:
            global _FIREBASE_ERROR
            _FIREBASE_ERROR = f"save_predictions failed: {e}"

    with open(config.PREDICTIONS_FILE, "w") as f:
        json.dump(preds, f, indent=2, default=str)


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
