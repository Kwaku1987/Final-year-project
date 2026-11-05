"""
Tests for the runtime configuration bridge (utils/runtime_env.py).

The point of these tests is a failure mode that is invisible without
them: on Streamlit Community Cloud the app is configured through
``st.secrets``, and if those values never reach ``os.environ`` the app
starts looking perfectly healthy while silently running with the
documented development password. Nothing raises, nothing logs, and the
Training Mode gate is open.

Run: python -m pytest tests/test_runtime_env.py -q
"""

from __future__ import annotations

import os

import pytest

from utils.runtime_env import (
    SECRET_KEYS,
    apply_secrets,
    describe,
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Every test starts from a clean SPAS_* environment."""

    saved = {
        key: os.environ.get(key)
        for key in SECRET_KEYS
    }

    for key in SECRET_KEYS:

        os.environ.pop(key, None)

    yield

    for key, value in saved.items():

        if value is None:

            os.environ.pop(key, None)

        else:

            os.environ[key] = value


# ============================================================
# THE BRIDGE
# ============================================================

def test_secrets_reach_the_environment():

    applied = apply_secrets(
        source={
            "SPAS_DB_ENABLED": "0",
            "SPAS_TARGET_DEFINITION": "exam_band",
        }
    )

    assert "SPAS_DB_ENABLED" in applied

    assert os.environ["SPAS_DB_ENABLED"] == "0"

    assert os.environ["SPAS_TARGET_DEFINITION"] == "exam_band"


def test_a_real_environment_variable_wins():
    """
    Docker, Render and a shell export must stay authoritative: the
    secrets file is a fallback, not an override.
    """

    os.environ["SPAS_TARGET_DEFINITION"] = "exam_band"

    applied = apply_secrets(
        source={"SPAS_TARGET_DEFINITION": "cpi"}
    )

    assert "SPAS_TARGET_DEFINITION" not in applied

    assert os.environ["SPAS_TARGET_DEFINITION"] == "exam_band"


def test_toml_booleans_become_1_and_0():
    """
    TOML has real booleans; the rest of the codebase compares strings
    against "1". Without coercion, SPAS_TUNING = true becomes the string
    "True", the comparison fails, and hyperparameter search is silently
    switched off.
    """

    apply_secrets(
        source={
            "SPAS_TUNING": True,
            "SPAS_CALIBRATION": False,
        }
    )

    assert os.environ["SPAS_TUNING"] == "1"

    assert os.environ["SPAS_CALIBRATION"] == "0"


def test_unknown_keys_are_ignored():
    """A typo in the Secrets box must not inject arbitrary variables."""

    applied = apply_secrets(
        source={
            "SPAS_TYPO_HASH": "oops",
            "AWS_SECRET_ACCESS_KEY": "definitely-not-ours",
        }
    )

    assert applied == []

    assert "AWS_SECRET_ACCESS_KEY" not in os.environ


def test_empty_values_are_not_applied():
    """An empty secret must leave the variable unset, not set to ""."""

    applied = apply_secrets(source={"SPAS_DB_PATH": ""})

    assert applied == []

    assert "SPAS_DB_PATH" not in os.environ


def test_missing_secrets_do_not_break_startup():
    """
    Locally there is no secrets.toml and st.secrets raises on access.
    That must be a no-op, never an exception: the app has to start.
    """

    applied = apply_secrets(source={})

    assert applied == []


def test_the_bridge_is_idempotent():

    source = {"SPAS_DB_ENABLED": "0"}

    first = apply_secrets(source=source)

    second = apply_secrets(source=source)

    assert first == ["SPAS_DB_ENABLED"]

    assert second == []


# ============================================================
# LOGGING
# ============================================================

def test_the_log_line_names_keys_without_values():

    line = describe(["SPAS_DB_ENABLED", "SPAS_TARGET_DEFINITION"])

    assert "SPAS_DB_ENABLED" in line

    assert "SPAS_TARGET_DEFINITION" in line


def test_the_log_line_says_so_when_nothing_was_applied():

    assert "environment only" in describe([])


# ============================================================
# COVERAGE OF THE CONFIG SURFACE
# ============================================================

def test_every_documented_setting_is_bridged():
    """
    A key missing from SECRET_KEYS is a key that cannot be configured
    on Community Cloud. This test makes the list a contract.
    """

    required = {
        "SPAS_DB_PATH",
        "SPAS_DB_ENABLED",
        "SPAS_TARGET_DEFINITION",
        "SPAS_LABEL_METHOD",
        "SPAS_TUNING",
        "SPAS_CALIBRATION",
    }

    assert required.issubset(set(SECRET_KEYS))


def test_the_template_lists_every_key():
    """The template people paste into Cloud must not be missing one."""

    from pathlib import Path

    template = (
        Path(__file__).resolve().parents[1]
        / ".streamlit"
        / "secrets.toml.example"
    )

    text = template.read_text()

    for key in SECRET_KEYS:

        assert key in text, f"{key} is not in secrets.toml.example"
