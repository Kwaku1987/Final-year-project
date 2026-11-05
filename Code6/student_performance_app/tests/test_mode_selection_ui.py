"""
The mode screen, driven end to end with ``AppTest``.

Training Mode used to be behind an administrator password. It is not any
more: the gate was removed at the owner's request, and this file pins
what replaced it --

* no password field, no attempt counter, no lockout: one click enters
  Training Mode;
* the screen still discloses, in one caption, that entering it is
  unrestricted (a deployment should not be surprised by that);
* entering it can never raise, and the sidebar's deployment panel still
  describes what is loaded.

The last test guards the removal itself: if ``utils/auth`` or a password
prompt comes back, this suite fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP_ROOT = Path(__file__).resolve().parents[1]

APP = str(APP_ROOT / "app.py")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No database writes; nothing else is needed to open Training Mode."""

    monkeypatch.setenv("SPAS_DB_ENABLED", "0")


def _boot() -> AppTest:

    app = AppTest.from_file(APP, default_timeout=180)

    app.run()

    return app


def _click(app: AppTest, label: str) -> None:

    matches = [button for button in app.button if label in button.label]

    assert matches, [button.label for button in app.button]

    matches[0].click()

    app.run()


# ------------------------------------------------------------
# Straight through
# ------------------------------------------------------------

def test_the_mode_screen_has_no_password_field() -> None:
    """No prompt, no attempt counter, no "enter password" button."""

    app = _boot()

    assert not app.exception

    assert app.text_input == []

    labels = [button.label for button in app.button]

    assert "Enter Training Mode" in labels

    assert not any("password" in label.lower() for label in labels)


def test_training_mode_is_one_click_away() -> None:
    """The point of the change: no credential stands between the user and the app."""

    app = _boot()

    _click(app, "Enter Training Mode")

    assert not app.exception

    assert app.session_state["application_state"].application_mode == (
        "administrator"
    )


def test_assessment_mode_is_one_click_away() -> None:
    """The lecturer path never needed a password and still does not."""

    app = _boot()

    _click(app, "Enter Assessment Mode")

    assert not app.exception

    assert app.session_state["application_state"].application_mode == (
        "lecturer"
    )


def test_the_screen_says_the_training_door_is_open() -> None:
    """
    Removing the gate is a decision, not an oversight: the mode screen has
    to say so, because a public URL then exposes retraining and model
    deletion to anyone who opens it.
    """

    app = _boot()

    text = " ".join(c.value for c in app.caption)

    assert "No password" in text

    assert "retrain and delete the model" in text


def test_sidebars_deployment_panel_still_describes_the_app() -> None:
    """The panel that answers "what am I looking at?" survives the removal."""

    app = _boot()

    _click(app, "Enter Training Mode")

    assert not app.exception

    captions = " ".join(c.value for c in app.sidebar.caption)

    assert "Model role" in captions

    assert "not_loaded" in captions

    assert "Training Mode" in captions

    assert "History" in captions


# ------------------------------------------------------------
# The removal itself
# ------------------------------------------------------------

def test_no_module_imports_the_removed_auth_module() -> None:
    """
    A dangling ``from utils.auth import ...`` is an ImportError at
    startup, i.e. a blank page on the deployed app.
    """

    assert not (APP_ROOT / "utils" / "auth.py").exists()

    offenders = []

    # tests/ is excluded: this file has to name what is gone in order to
    # assert that it is gone.
    for path in APP_ROOT.rglob("*.py"):

        if {
            "__pycache__",
            ".venv",
            "build",
            "dist",
            "tests",
        }.intersection(path.parts):

            continue

        text = path.read_text(encoding="utf-8")

        if "utils.auth" in text or "utils import auth" in text:

            offenders.append(str(path.relative_to(APP_ROOT)))

    assert not offenders, offenders


def test_no_password_settings_survive_in_the_codebase() -> None:
    """The variable, the default password and the attempt limit are gone."""

    names = {
        "ADMIN_PASSWORD_ENV",
        "ADMIN_PASSWORD_DEFAULT",
        "SPAS_ADMIN_PASSWORD_HASH",
        "AUTH_MAX_ATTEMPTS",
        "verify_admin_password",
        "change-me-admin",
    }

    offenders: dict[str, list[str]] = {}

    # tests/ is excluded: this file has to name what is gone in order to
    # assert that it is gone.
    for path in APP_ROOT.rglob("*.py"):

        if {
            "__pycache__",
            ".venv",
            "build",
            "dist",
            "tests",
        }.intersection(path.parts):

            continue

        text = path.read_text(encoding="utf-8")

        hits = sorted(name for name in names if name in text)

        if hits:

            offenders[str(path.relative_to(APP_ROOT))] = hits

    assert not offenders, offenders
