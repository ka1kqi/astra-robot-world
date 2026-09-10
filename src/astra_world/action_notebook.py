"""Append-only experiment evidence and labeled model notes, durable across sessions."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager


class ActionNotebook:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("""CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, kind TEXT NOT NULL,
            source TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)""")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, experiment_id, kind, payload, *, source="simulator"):
        encoded = json.dumps(payload, allow_nan=False)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO entries (experiment_id,kind,source,created_at,payload) VALUES (?,?,?,?,?)",
                (
                    experiment_id,
                    kind,
                    source,
                    datetime.now(timezone.utc).isoformat(),
                    encoded,
                ),
            )
            row_id = cursor.lastrowid
        return row_id

    @staticmethod
    def _decode(row):
        return {**dict(row), "payload": json.loads(row["payload"])}

    def read(self, experiment_id, limit=50, after_id=0):
        if not self.path.exists():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM entries WHERE experiment_id=? AND id>? ORDER BY id LIMIT ?",
                (experiment_id, after_id, limit),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def search(self, query="", limit=20):
        if not self.path.exists():
            return []
        terms = query.lower().split()[:12]
        clauses = [
            "EXISTS (SELECT 1 FROM json_tree(entries.payload) WHERE type=\"text\" AND lower(atom) LIKE ? ESCAPE '\\'"
            for _ in terms
        ]
        clauses = [clause + ")" for clause in clauses]
        parameters = [
            "%"
            + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            + "%"
            for term in terms
        ]
        sql = (
            "SELECT * FROM entries"
            + (" WHERE " + " AND ".join(clauses) if clauses else "")
            + " ORDER BY id DESC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(sql, (*parameters, limit)).fetchall()
        return [self._decode(row) for row in rows]

    def related(self, goal_kind, limit=3):
        if not self.path.exists():
            return []
        with self._connect() as connection:
            drafts = connection.execute(
                "SELECT * FROM entries WHERE kind='draft' AND json_extract(payload,'$.goal.kind')=? ORDER BY id DESC LIMIT ?",
                (goal_kind, limit),
            ).fetchall()
            results = []
            for draft in drafts:
                data = json.loads(draft["payload"])
                rows = connection.execute(
                    "SELECT * FROM entries WHERE experiment_id=? AND kind IN ('trial','note','state','saved') ORDER BY id DESC LIMIT 8",
                    (draft["experiment_id"],),
                ).fetchall()
                entries = [self._decode(row) for row in reversed(rows)]
                for entry in entries:
                    if entry["kind"] == "trial":
                        payload = entry["payload"]
                        report = payload["report"]
                        entry["payload"] = {
                            "trial_number": payload["trial_number"],
                            "goal_success": report.get("goal_success"),
                            "error_code": report.get("error_code"),
                            "detail": report.get("detail"),
                            "measurements": {
                                k: v
                                for k, v in report.get("measurements", {}).items()
                                if k != "trace"
                            },
                        }
                results.append(
                    {
                        "experiment_id": draft["experiment_id"],
                        "name": data["name"],
                        "goal": data["goal"],
                        "entries": entries,
                    }
                )
        return results

    def add_note(self, experiment_id, text, trial_number=None):
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM entries WHERE experiment_id=? AND kind='draft'",
                (experiment_id,),
            ).fetchone()
            if not exists:
                raise ValueError("No recorded experiment with this ID.")
            if trial_number is not None:
                trials = connection.execute(
                    "SELECT payload FROM entries WHERE experiment_id=? AND kind='trial'",
                    (experiment_id,),
                ).fetchall()
                if not any(
                    json.loads(row["payload"])["trial_number"] == trial_number
                    for row in trials
                ):
                    raise ValueError(
                        "The referenced trial does not exist in this experiment."
                    )
        return self.record(
            experiment_id,
            "note",
            {"text": text, "trial_number": trial_number},
            source="astra",
        )
