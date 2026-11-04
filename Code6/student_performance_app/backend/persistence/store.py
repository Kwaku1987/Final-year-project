"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/persistence/store.py

Purpose:
    Durable storage for assessment records and audit events.

Why this exists
---------------
v3.x kept the assessment history in the Streamlit session only. That
is fine for a demo and wrong for deployment:

  * a lecturer closes the tab and loses the cohort they just assessed;
  * a server restart wipes everything;
  * two lecturers on two devices cannot see each other's work;
  * and there is no audit trail of who trained, saved, deleted or
    exported what -- which is the first thing an institution asks for
    when the system touches student records (Act 843).

The store is deliberately boring: one SQLite file, two tables, no ORM,
no migrations framework. It is enough for a department, easy to back
up (copy the file), and easy to replace with PostgreSQL later because
every call goes through this module.

Failure policy
--------------
Storage must never take the application down. If the database cannot
be opened or written (read-only filesystem, missing permissions, a
locked file), the store degrades to a no-op and logs a warning; the
app continues with in-memory behaviour. Deployment problems should be
visible, not fatal.

Configuration
-------------
    SPAS_DB_PATH=/var/lib/spas/app.db     # default: data/spas.db
    SPAS_DB_ENABLED=0                     # disable persistence

Version:
    1.0
====================================================================
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/spas.db"

ENABLED = os.environ.get("SPAS_DB_ENABLED", "1") == "1"

DB_PATH = Path(
    os.environ.get("SPAS_DB_PATH", DEFAULT_DB_PATH)
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    student_name TEXT NOT NULL,
    student_id   TEXT,
    predicted    TEXT NOT NULL,
    confidence   REAL,
    model_name   TEXT,
    risk_priority TEXT,
    risk_score   INTEGER,
    student_data TEXT,
    explanation  TEXT
);

CREATE INDEX IF NOT EXISTS idx_assessments_created
    ON assessments (created_at DESC);

CREATE TABLE IF NOT EXISTS audit_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    actor      TEXT,
    action     TEXT NOT NULL,
    detail     TEXT,
    model_name TEXT,
    model_version TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_created
    ON audit_events (created_at DESC);
"""


@dataclass
class StoredAssessment:
    """One assessment record as read back from storage."""

    created_at: str
    student_name: str
    student_id: Optional[str]
    predicted: str
    confidence: Optional[float]
    model_name: Optional[str]
    risk_priority: Optional[str]
    risk_score: Optional[int]
    student_data: Dict[str, Any] = field(default_factory=dict)
    explanation: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "student_name": self.student_name,
            "student_id": self.student_id,
            "predicted": self.predicted,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "risk_priority": self.risk_priority,
            "risk_score": self.risk_score,
            "student_data": self.student_data,
            "explanation": self.explanation,
        }


class AssessmentStore:
    """
    SQLite-backed store for assessments and audit events.

    Every method is safe to call even when storage is unavailable:
    failures are logged and swallowed so the UI keeps working.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        path: Optional[Path | str] = None,
        enabled: Optional[bool] = None,
    ) -> None:

        self.path = Path(path or DB_PATH)

        self.enabled = ENABLED if enabled is None else bool(enabled)

        self._healthy: Optional[bool] = None

        if self.enabled:
            self._initialise()

    # -------------------------------------------------------

    @contextmanager
    def _connect(self):

        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
        )

        try:
            yield connection

            connection.commit()

        finally:
            connection.close()

    # -------------------------------------------------------

    def _initialise(self) -> None:
        """Create the file and schema; fall back to no-op on failure."""

        try:

            self.path.parent.mkdir(parents=True, exist_ok=True)

            with self._lock, self._connect() as connection:

                connection.executescript(SCHEMA)

            self._healthy = True

            logger.info("Assessment store ready at %s", self.path)

        except Exception as error:

            self._healthy = False

            self.enabled = False

            logger.warning(
                "Assessment store unavailable (%s); continuing "
                "without persistence. Set SPAS_DB_PATH to a "
                "writable location to enable it.",
                error,
            )

    # -------------------------------------------------------

    @property
    def healthy(self) -> bool:
        return bool(self.enabled and self._healthy)

    # -------------------------------------------------------

    def add_assessment(
        self,
        student_name: str,
        predicted: str,
        student_id: Optional[str] = None,
        confidence: Optional[float] = None,
        model_name: Optional[str] = None,
        risk_priority: Optional[str] = None,
        risk_score: Optional[int] = None,
        student_data: Optional[Dict[str, Any]] = None,
        explanation: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Persist one assessment. Returns True on success.
        """

        if not self.healthy:
            return False

        try:

            with self._lock, self._connect() as connection:

                connection.execute(
                    """
                    INSERT INTO assessments (
                        created_at, student_name, student_id,
                        predicted, confidence, model_name,
                        risk_priority, risk_score,
                        student_data, explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(timespec="seconds"),
                        student_name,
                        student_id,
                        predicted,
                        float(confidence) if confidence is not None else None,
                        model_name,
                        risk_priority,
                        int(risk_score) if risk_score is not None else None,
                        json.dumps(student_data or {}, default=str),
                        json.dumps(explanation or [], default=str),
                    ),
                )

            return True

        except Exception as error:  # pragma: no cover - defensive

            logger.warning("Could not persist assessment: %s", error)

            return False

    # -------------------------------------------------------

    def list_assessments(
        self,
        limit: int = 100,
        student_id: Optional[str] = None,
    ) -> List[StoredAssessment]:
        """
        Most recent assessments first.
        """

        if not self.healthy:
            return []

        try:

            query = (
                "SELECT created_at, student_name, student_id, "
                "predicted, confidence, model_name, risk_priority, "
                "risk_score, student_data, explanation "
                "FROM assessments "
            )

            params: tuple = ()

            if student_id:

                query += "WHERE student_id = ? "

                params = (student_id,)

            query += "ORDER BY id DESC LIMIT ?"

            params = params + (int(limit),)

            with self._lock, self._connect() as connection:

                rows = connection.execute(query, params).fetchall()

            return [
                StoredAssessment(
                    created_at=row[0],
                    student_name=row[1],
                    student_id=row[2],
                    predicted=row[3],
                    confidence=row[4],
                    model_name=row[5],
                    risk_priority=row[6],
                    risk_score=row[7],
                    student_data=json.loads(row[8] or "{}"),
                    explanation=json.loads(row[9] or "[]"),
                )
                for row in rows
            ]

        except Exception as error:  # pragma: no cover - defensive

            logger.warning("Could not read assessments: %s", error)

            return []

    # -------------------------------------------------------

    def count_assessments(self) -> int:
        """Total persisted assessments (0 when storage is off)."""

        if not self.healthy:
            return 0

        try:

            with self._lock, self._connect() as connection:

                row = connection.execute(
                    "SELECT COUNT(*) FROM assessments"
                ).fetchone()

            return int(row[0])

        except Exception:  # pragma: no cover - defensive

            return 0

    # -------------------------------------------------------
    # AUDIT TRAIL
    # -------------------------------------------------------

    def log_event(
        self,
        action: str,
        actor: Optional[str] = None,
        detail: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> bool:
        """
        Append one audit event (who did what, when).

        Actions used by the application: dataset_loaded,
        training_started, model_saved, model_deleted,
        model_loaded, assessment_created, report_exported,
        admin_unlocked, admin_unlock_failed.
        """

        if not self.healthy:
            return False

        try:

            with self._lock, self._connect() as connection:

                connection.execute(
                    """
                    INSERT INTO audit_events (
                        created_at, actor, action, detail,
                        model_name, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(timespec="seconds"),
                        actor,
                        action,
                        detail,
                        model_name,
                        model_version,
                    ),
                )

            logger.info("audit: %s %s", action, detail or "")

            return True

        except Exception as error:  # pragma: no cover - defensive

            logger.warning("Could not write audit event: %s", error)

            return False

    # -------------------------------------------------------

    def list_events(
        self,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:

        if not self.healthy:
            return []

        try:

            with self._lock, self._connect() as connection:

                rows = connection.execute(
                    "SELECT created_at, actor, action, detail, "
                    "model_name, model_version FROM audit_events "
                    "ORDER BY id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()

            return [
                {
                    "created_at": row[0],
                    "actor": row[1],
                    "action": row[2],
                    "detail": row[3],
                    "model_name": row[4],
                    "model_version": row[5],
                }
                for row in rows
            ]

        except Exception:  # pragma: no cover - defensive

            return []


# ============================================================
# MODULE-LEVEL SINGLETON
# ============================================================

_store: Optional[AssessmentStore] = None

_store_lock = threading.Lock()


def get_store() -> AssessmentStore:
    """Return the process-wide store (created on first use)."""

    global _store

    with _store_lock:

        if _store is None:

            _store = AssessmentStore()

        return _store


def reset_store(path: Optional[Path | str] = None) -> AssessmentStore:
    """Rebuild the singleton -- used by tests and after config changes."""

    global _store

    with _store_lock:

        _store = AssessmentStore(path=path)

        return _store


def audit(
    action: str,
    actor: Optional[str] = None,
    detail: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Fire-and-forget audit helper for the controller.

    Never raises: an audit failure must not break a user action.
    """

    try:

        get_store().log_event(
            action,
            actor=actor,
            detail=detail,
            **kwargs,
        )

    except Exception:  # pragma: no cover - defensive

        logger.debug("audit() swallowed an error", exc_info=True)
