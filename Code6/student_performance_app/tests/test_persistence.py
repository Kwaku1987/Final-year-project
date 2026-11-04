"""
Tests for the SQLite persistence layer.

The store must (a) keep assessments across "restarts", (b) record an
audit trail, and (c) never take the application down when storage is
unavailable — a read-only filesystem is a deployment reality.
"""

from __future__ import annotations

import sqlite3

from pathlib import Path

import pytest

from backend.persistence.store import AssessmentStore


@pytest.fixture()
def store(tmp_path) -> AssessmentStore:

    return AssessmentStore(
        path=tmp_path / "test.db",
        enabled=True,
    )


# ============================================================
# SCHEMA / HEALTH
# ============================================================

def test_store_creates_database_and_tables(store, tmp_path):

    assert store.healthy

    assert (tmp_path / "test.db").exists()

    with sqlite3.connect(tmp_path / "test.db") as connection:

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    assert {"assessments", "audit_events"} <= tables


def test_disabled_store_is_a_safe_no_op(tmp_path):

    store = AssessmentStore(
        path=tmp_path / "off.db",
        enabled=False,
    )

    assert store.healthy is False

    assert store.add_assessment("Ama", "Low") is False

    assert store.list_assessments() == []

    assert store.count_assessments() == 0

    assert store.log_event("test") is False


def test_unwritable_location_degrades_instead_of_raising(tmp_path):

    # A path whose parent cannot be created (a file, not a directory).
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")

    store = AssessmentStore(
        path=blocker / "nested" / "app.db",
        enabled=True,
    )

    assert store.healthy is False

    assert store.add_assessment("Ama", "Low") is False


# ============================================================
# ASSESSMENTS
# ============================================================

def test_assessment_is_persisted_and_read_back(store):

    added = store.add_assessment(
        student_name="Ama Mensah",
        student_id="UG-001",
        predicted="Low",
        confidence=0.82,
        model_name="Logistic Regression",
        risk_priority="HIGH",
        risk_score=70,
        student_data={"attendance_rate": 45.0},
        explanation=[{"feature": "attendance_rate"}],
    )

    assert added

    stored = store.list_assessments()

    assert len(stored) == 1

    record = stored[0]

    assert record.student_name == "Ama Mensah"

    assert record.student_id == "UG-001"

    assert record.predicted == "Low"

    assert record.confidence == pytest.approx(0.82)

    assert record.risk_score == 70

    assert record.student_data["attendance_rate"] == 45.0

    assert record.explanation[0]["feature"] == "attendance_rate"


def test_assessments_survive_a_new_store_instance(tmp_path):
    """Simulates a server restart."""

    path = tmp_path / "persist.db"

    first = AssessmentStore(path=path, enabled=True)

    first.add_assessment("Kofi Boateng", "High")

    second = AssessmentStore(path=path, enabled=True)

    stored = second.list_assessments()

    assert len(stored) == 1

    assert stored[0].student_name == "Kofi Boateng"


def test_history_is_newest_first_and_limited(store):

    for index in range(5):

        store.add_assessment(f"Student {index}", "Average")

    recent = store.list_assessments(limit=2)

    assert len(recent) == 2

    assert recent[0].student_name == "Student 4"

    assert recent[1].student_name == "Student 3"

    assert store.count_assessments() == 5


def test_filter_by_student_id(store):

    store.add_assessment("A", "Low", student_id="ID-1")

    store.add_assessment("B", "High", student_id="ID-2")

    store.add_assessment("C", "Average", student_id="ID-1")

    records = store.list_assessments(student_id="ID-1")

    assert {record.student_name for record in records} == {"A", "C"}


# ============================================================
# AUDIT TRAIL
# ============================================================

def test_audit_events_are_recorded(store):

    assert store.log_event(
        "model_saved",
        detail="best=Random Forest",
        model_name="Random Forest",
        model_version="7.1",
    )

    events = store.list_events()

    assert len(events) == 1

    assert events[0]["action"] == "model_saved"

    assert events[0]["model_version"] == "7.1"

    assert events[0]["created_at"]


def test_audit_uses_a_known_action_vocabulary(store):

    actions = [
        "admin_unlocked",
        "dataset_loaded",
        "model_saved",
        "model_loaded",
        "model_deleted",
        "assessment_created",
        "report_exported",
    ]

    for action in actions:

        store.log_event(action)

    assert {event["action"] for event in store.list_events()} == set(
        actions
    )
