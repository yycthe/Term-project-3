"""Regression checks for quota outages and non-destructive history writes."""

import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import config
import storage
from history_snapshot import load_verified_history_snapshot


def prediction(game_id, **values):
    return {"game_id": game_id, "status": "settled", **values}


class PredictionStorageTests(unittest.TestCase):
    def setUp(self):
        importlib.reload(storage)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1000.0
        self.documents = {"a": prediction("a"), "b": prediction("b")}
        self.read_error = None
        self.write_error = None
        self.committed = []
        self.collection = Mock()
        self.collection.stream.side_effect = self.stream
        self.collection.document.side_effect = lambda key: key
        self.db = Mock()
        self.db.collection.return_value = self.collection
        self.db.batch.side_effect = self.batch
        storage._DB = self.db
        for patcher in [
            patch.object(config, "OUTPUTS_DIR", self.tmp.name),
            patch.object(config, "PREDICTIONS_FILE", str(Path(self.tmp.name) / "local.json")),
            patch.object(storage, "_firebase_enabled", return_value=True),
            patch.object(storage, "_ensure_firestore", return_value=True),
            patch.object(storage, "_get_secret_value", return_value="test-project"),
            patch.object(storage.time, "monotonic", side_effect=lambda: self.now),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def stream(self, **kwargs):
        if self.read_error:
            raise RuntimeError(self.read_error)
        return [SimpleNamespace(id=key, to_dict=lambda row=row: copy.deepcopy(row))
                for key, row in self.documents.items()]

    def batch(self):
        operations = []
        batch = Mock()
        batch.set.side_effect = lambda key, row: operations.append(("set", key, copy.deepcopy(row)))
        batch.delete.side_effect = lambda key: operations.append(("delete", key, None))

        def commit():
            if self.write_error:
                raise RuntimeError(self.write_error)
            self.committed.extend(operations)
            for action, key, value in operations:
                if action == "set":
                    self.documents[key] = value
                else:
                    self.documents.pop(key, None)
        batch.commit.side_effect = commit
        return batch

    def expire(self):
        self.now += storage.PREDICTIONS_CACHE_SECONDS + 1

    def test_reruns_share_read_and_cannot_mutate_cache(self):
        rows = storage.load_predictions()
        rows[0]["status"] = "changed outside cache"
        rows.append(prediction("c"))
        for _ in range(20):
            self.assertEqual(storage.load_predictions(), [prediction("a"), prediction("b")])
        self.collection.stream.assert_called_once_with(timeout=10, retry=None)

    def test_quota_failure_retains_last_good_read_and_backs_off(self):
        saved = storage.load_predictions()
        self.expire()
        self.read_error = "429 Quota exceeded."
        self.assertEqual(storage.load_predictions(), saved)
        for _ in range(10):
            self.assertEqual(storage.load_predictions(), saved)
        self.assertEqual(self.collection.stream.call_count, 2)
        self.assertFalse(storage.get_storage_status()["firebase_connected"])
        self.assertFalse(storage.get_storage_status()["predictions_writable"])
        with self.assertRaises(storage.PredictionStorageError):
            storage.save_predictions([prediction("c")])
        with self.assertRaises(storage.PredictionStorageError):
            storage.delete_predictions({"a"})
        self.assertEqual(self.committed, [])

    def test_retry_recovers_without_process_restart(self):
        self.read_error = "429 Quota exceeded."
        self.assertEqual(storage.load_predictions(), [])
        self.read_error = None
        self.expire()
        self.assertEqual(len(storage.load_predictions()), 2)
        self.assertTrue(storage.get_storage_status()["predictions_writable"])
        self.assertIsNone(storage.get_storage_status()["firebase_error"])

    def test_initial_cloud_failure_does_not_use_local_development_history(self):
        Path(config.PREDICTIONS_FILE).write_text(json.dumps([prediction("local-only")]))
        self.read_error = "429 Quota exceeded."
        self.assertEqual(storage.load_predictions(), [])
        with self.assertRaises(storage.PredictionStorageError):
            storage.save_predictions([prediction("new")])
        self.assertEqual(json.loads(Path(config.PREDICTIONS_FILE).read_text()), [prediction("local-only")])

    def test_cloud_mirror_survives_cache_loss_but_stays_read_only(self):
        saved = storage.load_predictions()
        storage._PREDICTIONS_CACHE = None
        self.expire()
        self.read_error = "429 Quota exceeded."
        self.assertEqual(storage.load_predictions(), saved)
        self.assertEqual(storage.get_storage_status()["predictions_source"], "cache")
        self.assertFalse(storage.get_storage_status()["predictions_writable"])

    def test_mirror_from_another_project_is_not_loaded(self):
        storage._cloud_mirror_path().write_text(json.dumps({
            "project": "another-project", "collection": config.FIREBASE_PREDICTIONS_COLLECTION,
            "predictions": [prediction("other")],
        }))
        self.read_error = "429 Quota exceeded."
        self.assertEqual(storage.load_predictions(), [])

    def test_successfully_empty_cloud_does_not_resurrect_cached_records(self):
        storage.load_predictions()
        self.documents.clear()
        self.expire()
        self.assertEqual(storage.load_predictions(), [])
        self.assertTrue(storage.get_storage_status()["predictions_writable"])

    def test_partial_save_upserts_only_changes_and_never_deletes(self):
        storage.load_predictions()
        self.documents["concurrent"] = prediction("concurrent")
        storage.save_predictions([prediction("a", correct=True)])
        self.assertEqual(set(self.documents), {"a", "b", "concurrent"})
        self.assertEqual(self.committed, [("set", "a", prediction("a", correct=True))])
        storage.save_predictions([prediction("a", correct=True)])
        self.assertEqual(len(self.committed), 1)
        self.assertEqual(self.collection.stream.call_count, 1)

    def test_explicit_delete_only_removes_selected_documents(self):
        storage.load_predictions()
        self.documents["concurrent"] = prediction("concurrent")
        storage.delete_predictions({"a"})
        self.assertEqual(set(self.documents), {"b", "concurrent"})
        self.assertEqual(self.committed, [("delete", "a", None)])
        self.assertEqual(storage.load_predictions(), [prediction("b")])

    def test_failed_write_keeps_cache_and_reports_failure(self):
        saved = storage.load_predictions()
        self.write_error = "Unavailable"
        with self.assertRaises(storage.PredictionStorageError):
            storage.save_predictions([prediction("a", correct=True)])
        self.assertEqual(storage.load_predictions(), saved)
        self.assertFalse(storage.get_storage_status()["predictions_writable"])

    def test_display_snapshot_is_never_written_as_live_history(self):
        with self.assertRaises(storage.PredictionStorageError):
            storage.save_predictions(load_verified_history_snapshot())
        self.assertEqual(self.committed, [])


class RecoveredSnapshotTests(unittest.TestCase):
    def test_matches_the_previously_verified_table(self):
        rows = load_verified_history_snapshot()
        self.assertEqual(len(rows), 120)
        self.assertEqual(sum(row["correct"] for row in rows), 87)
        self.assertEqual(len({(r["game_date"], r["home_team"], r["away_team"]) for r in rows}), 120)
        for row in rows:
            self.assertEqual(row["correct"], row["predicted_winner"] == row["actual_winner"])
            self.assertNotIn("game_id", row)
            self.assertNotIn("created_at", row)
        # Fingerprint computed directly from the preserved browser table.
        source = Path(__file__).resolve().parents[1] / "outputs" / "verified_prediction_history_2026-09-21.csv"
        fingerprint = 2166136261
        for char in source.read_text().rstrip("\n"):
            fingerprint = ((fingerprint ^ ord(char)) * 16777619) & 0xffffffff
        self.assertEqual(fingerprint, 642614973)


if __name__ == "__main__":
    unittest.main()
