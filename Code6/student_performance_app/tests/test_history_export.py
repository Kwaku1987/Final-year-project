"""
Tests for the two "quality of life" additions that ship with v3.3.4.

1. The sidebar's deployment-status panel needs accurate provenance: a
   freshly booted app has *no* model in memory and must say so instead of
   quoting a default, and a loaded bundle must describe its own split
   rather than the session's default.
2. The assessment history export must keep the raw numbers and the input
   features, because the table rounds confidence to one decimal and
   truncates the timestamp.

Both are exercised without a Streamlit runtime.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import streamlit as st

from dashboard.application_state import ApplicationState
from dashboard.dashboard_controller import DashboardController
from dashboard.pages.assessment_page import (
    build_history_export,
    history_csv,
)


# ------------------------------------------------------------
# 1. Model provenance
# ------------------------------------------------------------

def _controller(framework) -> DashboardController:

    controller = object.__new__(DashboardController)

    controller.state = ApplicationState

    controller.framework = framework

    return controller


@pytest.fixture()
def session(monkeypatch) -> dict:
    fake_state: dict = {}

    monkeypatch.setattr(st, "session_state", fake_state, raising=True)

    return fake_state


def test_role_is_not_loaded_before_a_model_exists(session) -> None:
    """A default must never be quoted as a fact."""

    framework = SimpleNamespace(
        trained_models={},
        best_model=None,
        bundle_role="unknown",
        split_metadata={"strategy": "random"},
    )

    controller = _controller(framework)

    assert controller.model_is_loaded is False

    assert controller.model_bundle_role == "not_loaded"


def test_role_after_training_is_reported(session) -> None:
    framework = SimpleNamespace(
        trained_models={"Logistic Regression": object()},
        best_model=object(),
        bundle_role="production",
        split_metadata={"strategy": "temporal"},
    )

    controller = _controller(framework)

    assert controller.model_is_loaded is True

    assert controller.model_bundle_role == "production"

    assert controller.model_validation_strategy == "temporal"


def test_loaded_bundle_split_wins_over_session_default(session) -> None:
    """
    After ``load_best_model`` this session has split nothing, so
    ``split_metadata`` is still its default -- reading it would label a
    temporal bundle "random".
    """

    framework = SimpleNamespace(
        trained_models={"Soft Voting Ensemble": object()},
        best_model=object(),
        bundle_role="production",
        bundle_split={"strategy": "temporal", "test_periods": ["2024-S2"]},
        split_metadata={"strategy": "random"},
    )

    controller = _controller(framework)

    assert controller.model_validation_strategy == "temporal"


def test_unknown_strategy_is_reported_as_unknown(session) -> None:
    framework = SimpleNamespace(
        trained_models={"X": object()},
        best_model=object(),
        bundle_role="production",
        split_metadata={},
    )

    controller = _controller(framework)

    assert controller.model_validation_strategy == "unknown"


def test_shipped_bundle_describes_itself_truthfully(session) -> None:
    """
    The committed demo bundle must be a production model validated
    temporally. If someone re-trains it on a random split -- or on the
    legacy CPI label -- the sidebar would start advertising a number that
    does not mean what it appears to mean, so fail here instead.
    """

    from framework.framework_factory import FrameworkFactory

    bundle = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"

    if not bundle.exists():

        pytest.skip("no committed model bundle to inspect")

    framework = FrameworkFactory.create()

    framework.load_best_model(str(bundle))

    controller = _controller(framework)

    assert controller.model_is_loaded is True

    assert controller.model_bundle_role == "production"

    assert controller.model_validation_strategy == "temporal"


def test_no_framework_is_survivable(session) -> None:
    """The sidebar renders before anything is loaded."""

    controller = object.__new__(DashboardController)

    controller.state = ApplicationState

    controller.framework = None

    assert controller.model_is_loaded is False

    assert controller.model_bundle_role == "not_loaded"

    assert controller.model_validation_strategy == "unknown"


# ------------------------------------------------------------
# 2. History export
# ------------------------------------------------------------

class FakeRecord:
    """Stand-in for ``AssessmentRecord`` (session-only history)."""

    def __init__(self, name: str, confidence: float) -> None:

        self.student_name = name

        self.student_id = ""

        self.prediction = "Low"

        self.confidence = confidence

        self.risk_priority = "High"

        self.risk_score = 7

        self.model_name = "Soft Voting Ensemble"

        self.assessment_date = datetime(2026, 11, 5, 9, 30)

        self.student_data = {"attendance_rate": 61.5, "study_hours": 3}


def test_export_keeps_raw_confidence_and_inputs() -> None:
    rows = build_history_export([FakeRecord("Ama", 0.7345)])

    assert len(rows) == 1

    entry = rows[0]

    # Raw, not the "73.5%" the table shows.
    assert entry["confidence"] == pytest.approx(0.7345)

    assert entry["assessed_at"] == "2026-11-05 09:30"

    assert entry["input_attendance_rate"] == pytest.approx(61.5)

    assert entry["input_study_hours"] == 3

    assert entry["student_id"] == ""


def test_export_handles_stored_dict_records() -> None:
    history = [
        {
            "student_name": "Kwame",
            "student_id": "S-001",
            "predicted": "High",
            "confidence": 0.91,
            "risk_priority": None,
            "risk_score": None,
            "model_name": "Soft Voting Ensemble",
            "created_at": "2026-10-01T08:15:00",
            "student_data": {"sleep_hours": 7.5},
        }
    ]

    entry = build_history_export(history)[0]

    assert entry["prediction"] == "High"

    assert entry["risk_priority"] == ""

    assert entry["assessed_at"] == "2026-10-01T08:15:00"

    assert entry["input_sleep_hours"] == pytest.approx(7.5)


def test_export_is_newest_first() -> None:
    """The table reverses the stored order; the CSV must match it."""

    history = [
        {"student_name": "Oldest", "created_at": "2026-01-01T00:00:00"},
        {"student_name": "Newest", "created_at": "2026-02-01T00:00:00"},
    ]

    rows = build_history_export(history)

    assert [row["student_name"] for row in rows] == ["Newest", "Oldest"]


def test_export_of_an_empty_history_is_an_empty_frame() -> None:
    assert build_history_export([]) == []

    # An empty CSV still parses (header-only / blank), it just has no rows.
    text = history_csv([]).decode("utf-8")

    assert isinstance(text, str)


def test_csv_round_trips_through_pandas() -> None:
    history = [
        FakeRecord("Ama", 0.7345),
        {
            "student_name": "Kwame",
            "predicted": "High",
            "confidence": 0.91,
            "created_at": "2026-10-01T08:15:00",
            "student_data": {"exam_score": 88.0},
        },
    ]

    frame = pd.read_csv(pd.io.common.BytesIO(history_csv(history)))

    assert len(frame) == 2

    assert set(
        [
            "student_name",
            "prediction",
            "confidence",
            "assessed_at",
            "input_attendance_rate",
            "input_exam_score",
        ]
    ).issubset(frame.columns)

    # Both record shapes coexisted in one file, newest first
    # (the store returns oldest-first, like the table's `reversed`).
    assert frame["student_name"].tolist() == ["Kwame", "Ama"]
