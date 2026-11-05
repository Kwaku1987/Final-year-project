"""
Paths must not depend on where the process was started.

The hosted app reported **"No saved model found yet"** while
``Code6/student_performance_app/models/best_model.pkl`` was committed and
worked locally. The cause was a relative default path: Community Cloud
launches ``streamlit run Code6/student_performance_app/app.py`` with the
working directory set to the **repository root**, so
``models/best_model.pkl`` resolved one directory above the bundle.

These tests pin the fix: every default file path is absolute, resolved
against the application directory, and Assessment Mode finds the bundle
even when the process starts at the repository root.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from backend.config import APP_DIR, DEFAULT_DB_PATH, DEFAULT_MODEL_PATH
from dashboard.dashboard_controller import DashboardController


APP_ROOT = Path(__file__).resolve().parents[1]

REPO_ROOT = APP_ROOT.parents[1]

APP = str(APP_ROOT / "app.py")


def test_default_paths_are_absolute() -> None:

    assert Path(DEFAULT_MODEL_PATH).is_absolute()

    assert Path(DEFAULT_DB_PATH).is_absolute()

    assert Path(DashboardController.DEFAULT_MODEL_PATH).is_absolute()


def test_default_paths_live_beside_the_app() -> None:

    assert APP_DIR == APP_ROOT

    assert Path(DEFAULT_MODEL_PATH).parent.parent == APP_DIR

    assert Path(DEFAULT_DB_PATH).parent.parent == APP_DIR


def test_model_path_survives_a_directory_change(tmp_path, monkeypatch) -> None:
    """The bug in one line: a relative path is resolved by the caller."""

    monkeypatch.chdir(tmp_path)

    assert Path(DashboardController.DEFAULT_MODEL_PATH).is_file()


def test_database_path_is_independent_of_the_working_directory(
    tmp_path, monkeypatch
) -> None:
    """Otherwise the hosted app would write its SQLite file to the repo root."""

    from backend.persistence.store import AssessmentStore

    monkeypatch.chdir(tmp_path)

    store = AssessmentStore(enabled=False)

    assert store.path.is_absolute()

    assert store.path.parent.parent == APP_DIR


def test_assessment_mode_finds_the_bundle_from_the_repository_root(
    monkeypatch,
) -> None:
    """
    Reproduce the hosted layout: start the process in the repository root
    and launch the app by path, exactly as Community Cloud does.
    """

    monkeypatch.chdir(REPO_ROOT)

    monkeypatch.setenv("SPAS_DB_ENABLED", "0")

    monkeypatch.setenv(
        "SPAS_ADMIN_PASSWORD_HASH",
        "b1a2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
    )

    app = AppTest.from_file(APP, default_timeout=180)

    app.run()

    assert not app.exception

    text = " ".join(
        [w.value for w in app.warning] + [c.value for c in app.caption]
    )

    assert "No saved model found yet" not in text

    # It should also say where the bundle was found, so a future mismatch
    # is diagnosable from the UI.
    enter = [
        button
        for button in app.button
        if "Enter Assessment Mode" in button.label
    ]

    assert enter, [b.label for b in app.button]

    enter[0].click()

    app.run()

    assert not app.exception


def test_missing_bundle_message_names_the_resolved_path(
    monkeypatch, tmp_path
) -> None:
    """
    When the bundle really is absent, the page must say which path it
    looked at -- the old message could not distinguish "not there" from
    "looked in the wrong place".
    """

    from dashboard.dashboard_controller import DashboardController as Controller

    monkeypatch.setenv("SPAS_DB_ENABLED", "0")

    app = AppTest.from_file(APP, default_timeout=180)

    monkeypatch.setattr(
        Controller,
        "DEFAULT_MODEL_PATH",
        str(tmp_path / "models" / "best_model.pkl"),
    )

    app.run()

    assert not app.exception

    captions = " ".join(c.value for c in app.caption)

    assert "best_model.pkl" in captions

    assert str(tmp_path) in captions
