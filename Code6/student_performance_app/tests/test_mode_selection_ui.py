"""
The mode-selection page, driven end to end with ``AppTest``.

This is the screen the user was stuck on: it answered every failed
unlock with "Incorrect password" whether the cause was a typo, a paste
error, or no credential at all, and five typos left it unusable. These
tests pin the replacement behaviour:

* say what credential is configured (and never print it);
* name the variable to fix when a password really is wrong;
* offer a way out of the attempt limit.

They run the real ``app.py`` in-process, so a change that breaks the page
at import or render time fails here rather than in production.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from backend.config import ADMIN_PASSWORD_ENV, AUTH_MAX_ATTEMPTS


APP = str(Path(__file__).resolve().parents[1] / "app.py")

PASSWORD = "CorrectHorseBatteryStaple"

DIGEST = hashlib.sha256(PASSWORD.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No credential and no database writes unless a test asks for them."""

    monkeypatch.delenv(ADMIN_PASSWORD_ENV, raising=False)

    monkeypatch.setenv("SPAS_DB_ENABLED", "0")


@pytest.fixture()
def hash_credential(monkeypatch) -> str:
    """The deployed configuration: a SHA-256 digest in the secret."""

    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    return DIGEST


def _boot() -> AppTest:

    app = AppTest.from_file(APP, default_timeout=180)

    app.run()

    return app


def test_no_configured_credential_is_reported_as_a_warning() -> None:
    app = _boot()

    assert not app.exception

    warnings = " ".join(w.value for w in app.warning)

    assert "No administrator password is configured" in warnings

    # And it must not silently pretend Training Mode is protected.
    assert "change-me-admin" in warnings


def test_configured_hash_is_described_without_being_printed(
    hash_credential,
) -> None:

    app = _boot()

    assert not app.exception

    captions = " ".join(c.value for c in app.caption)

    assert "SHA-256 hash" in captions

    assert hash_credential not in captions


def test_wrong_password_names_the_variable_to_fix(
    hash_credential,
) -> None:

    app = _boot()

    app.text_input[0].input("definitely-not-it")

    [b for b in app.button if "Enter Training Mode" in b.label][0].click()

    app.run()

    errors = " ".join(e.value for e in app.error)

    assert ADMIN_PASSWORD_ENV in errors

    assert "Incorrect password" in errors

    # The attempt budget is stated, not implied.
    assert f"{AUTH_MAX_ATTEMPTS - 1} attempt(s) left" in errors


def test_lockout_can_be_cleared_without_a_reload(
    hash_credential,
) -> None:
    """
    The old message said "reload the page" -- which does not work,
    because the counter lives in session state.
    """

    app = AppTest.from_file(APP, default_timeout=180)

    app.session_state["admin_auth_attempts"] = AUTH_MAX_ATTEMPTS

    app.run()

    assert not app.exception

    assert any(
        "Too many failed attempts" in e.value for e in app.error
    )

    reset = [
        button for button in app.button if "Reset attempts" in button.label
    ]

    assert reset, [b.label for b in app.button]

    reset[0].click()

    app.run()

    assert app.session_state["admin_auth_attempts"] == 0

    assert any(
        "Enter Training Mode" in button.label for button in app.button
    )


def test_sidebar_reports_the_deployment_state(hash_credential) -> None:
    """
    The panel that answers "what am I actually looking at?" -- the four
    facts that decide whether a number on screen can be trusted.
    """

    app = _boot()

    app.text_input[0].input(PASSWORD)

    [b for b in app.button if "Enter Training Mode" in b.label][0].click()

    app.run()

    assert not app.exception

    captions = " ".join(c.value for c in app.sidebar.caption)

    assert "Model role" in captions

    # No bundle is loaded yet, so it must not claim one.
    assert "not_loaded" in captions

    assert "Training Mode password" in captions

    assert "hash" in captions

    assert "History" in captions


def test_correct_password_still_unlocks_training_mode(
    hash_credential,
) -> None:
    """The fixes must not have moved the goalposts on the happy path."""

    app = _boot()

    app.text_input[0].input(PASSWORD)

    [b for b in app.button if "Enter Training Mode" in b.label][0].click()

    app.run()

    assert not app.exception

    state = app.session_state["application_state"]

    assert state.application_mode == "administrator"
