"""
Regression tests for the administrator credential path.

Every test here corresponds to something the deployed app got wrong:

* a correct password was reported as wrong because the secret had been
  pasted as a whole TOML line, with quotes, with a trailing newline,
  with a ``sha256:`` prefix, or in uppercase hex;
* a plaintext password in ``SPAS_ADMIN_PASSWORD_HASH`` was treated as an
  error instead of a credential;
* five typos locked Training Mode for the rest of the session, because
  the attempt counter lives in ``st.session_state`` -- which survives the
  page reload the UI told the user to perform.

These are unit tests: they swap ``st.session_state`` for a plain dict so
the logic can be exercised without a Streamlit runtime.
"""

from __future__ import annotations

import pytest
import streamlit as st

from backend.config import (
    ADMIN_PASSWORD_DEFAULT,
    ADMIN_PASSWORD_ENV,
    AUTH_MAX_ATTEMPTS,
)
from utils import auth


PASSWORD = "CorrectHorseBatteryStaple"

DIGEST = auth.hash_password(PASSWORD)


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------

@pytest.fixture()
def session(monkeypatch):
    """A fresh, empty session-state dict and no configured secret."""

    fake_state: dict = {}

    monkeypatch.setattr(st, "session_state", fake_state, raising=True)

    monkeypatch.delenv(ADMIN_PASSWORD_ENV, raising=False)

    return fake_state


# ------------------------------------------------------------
# Normalisation of a pasted secret
# ------------------------------------------------------------

@pytest.mark.parametrize(
    "pasted",
    [
        DIGEST,
        f"{DIGEST}\n",
        f"  {DIGEST}  ",
        f'"{DIGEST}"',
        f"'{DIGEST}'",
        f"{ADMIN_PASSWORD_ENV} = \"{DIGEST}\"",
        f"{ADMIN_PASSWORD_ENV}={DIGEST}",
        f"sha256:{DIGEST}",
        DIGEST.upper(),
    ],
)
def test_normalize_credential_accepts_common_paste_shapes(pasted: str) -> None:
    """All of these mean the same SHA-256 credential."""

    assert auth.normalize_credential(pasted) == DIGEST


@pytest.mark.parametrize("empty", [None, "", "   ", "\n"])
def test_normalize_credential_reduces_blanks_to_empty(empty) -> None:
    assert auth.normalize_credential(empty) == ""


def test_normalize_credential_keeps_plaintext_passwords_intact() -> None:
    """A plaintext password is not a hash and must not be mangled."""

    assert auth.normalize_credential("  my pass word  ") == "my pass word"

    assert auth.normalize_credential('"my pass word"') == "my pass word"


def test_normalize_credential_does_not_strip_non_spas_equals_lines() -> None:
    """Only a pasted ``SPAS_*`` line is unpacked."""

    raw = "a=b=c"

    assert auth.normalize_credential(raw) == raw


# ------------------------------------------------------------
# What kind of credential is configured
# ------------------------------------------------------------

def test_kind_is_default_when_nothing_is_configured(session) -> None:
    assert auth.credential_kind() == "default"

    assert auth.is_using_default_password() is True

    assert auth.credential_status()["configured"] is False


def test_kind_is_hash_for_a_digest(session, monkeypatch) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    status = auth.credential_status()

    assert status["kind"] == "hash"

    assert status["length"] == 64

    assert status["configured"] is True


def test_kind_is_plaintext_for_a_password(session, monkeypatch) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, PASSWORD)

    assert auth.credential_kind() == "plaintext"

    assert auth.credential_status()["length"] == len(PASSWORD)


def test_credential_status_never_contains_the_secret(
    session, monkeypatch
) -> None:
    """The status dict is rendered in the UI, so it must stay non-secret."""

    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    rendered = repr(auth.credential_status())

    assert DIGEST not in rendered

    assert PASSWORD not in rendered


# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

def test_verify_accepts_the_matching_password(session, monkeypatch) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    assert auth.verify_admin_password(PASSWORD) is True

    assert auth.is_admin_unlocked() is True


def test_verify_accepts_a_secret_pasted_as_a_toml_line(
    session, monkeypatch
) -> None:
    """The exact failure the deployed app had: the whole line was pasted."""

    monkeypatch.setenv(
        ADMIN_PASSWORD_ENV, f'{ADMIN_PASSWORD_ENV} = "{DIGEST}"'
    )

    assert auth.verify_admin_password(PASSWORD) is True


def test_verify_accepts_a_plaintext_secret(session, monkeypatch) -> None:
    """Plaintext in the secrets box works; the UI says it should be hashed."""

    monkeypatch.setenv(ADMIN_PASSWORD_ENV, PASSWORD)

    assert auth.credential_kind() == "plaintext"

    assert auth.verify_admin_password(PASSWORD) is True

    assert auth.verify_admin_password("not the password") is False


def test_verify_falls_back_to_the_documented_development_password(
    session,
) -> None:
    assert auth.is_using_default_password() is True

    assert auth.verify_admin_password(ADMIN_PASSWORD_DEFAULT) is True


def test_verify_rejects_the_development_password_once_a_secret_is_set(
    session, monkeypatch
) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    assert auth.verify_admin_password(ADMIN_PASSWORD_DEFAULT) is False

    assert auth.is_admin_unlocked() is False


def test_verify_rejects_empty_and_wrong_passwords(session, monkeypatch) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    assert auth.verify_admin_password("") is False

    assert auth.verify_admin_password("wrong") is False

    assert auth.attempts_remaining() == AUTH_MAX_ATTEMPTS - 2


# ------------------------------------------------------------
# Lockout: the guard, and the way out of it
# ------------------------------------------------------------

def test_repeated_failures_lock_the_form(session, monkeypatch) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    for _ in range(AUTH_MAX_ATTEMPTS):

        assert auth.verify_admin_password("wrong") is False

    assert auth.attempts_remaining() == 0

    # Blocked even with the right password -- this is the guard working.
    assert auth.verify_admin_password(PASSWORD) is False


def test_reset_attempts_restores_access(session, monkeypatch) -> None:
    """
    The bug: the counter lives in session state, which survives a reload,
    so "reload the page" left the administrator locked out for good.
    """

    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    for _ in range(AUTH_MAX_ATTEMPTS):

        auth.verify_admin_password("wrong")

    assert auth.attempts_remaining() == 0

    auth.reset_attempts()

    assert auth.attempts_remaining() == AUTH_MAX_ATTEMPTS

    assert auth.verify_admin_password(PASSWORD) is True

    assert auth.is_admin_unlocked() is True


def test_successful_unlock_clears_the_counter(session, monkeypatch) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    auth.verify_admin_password("wrong")

    auth.verify_admin_password("wrong")

    assert auth.attempts_remaining() == AUTH_MAX_ATTEMPTS - 2

    assert auth.verify_admin_password(PASSWORD) is True

    assert auth.attempts_remaining() == AUTH_MAX_ATTEMPTS


def test_lock_admin_relocks_and_clears_attempts(
    session, monkeypatch
) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    auth.verify_admin_password(PASSWORD)

    auth.verify_admin_password("wrong")

    auth.lock_admin()

    assert auth.is_admin_unlocked() is False

    assert auth.attempts_remaining() == AUTH_MAX_ATTEMPTS


# ------------------------------------------------------------
# CLI diagnostics
# ------------------------------------------------------------

def test_cli_check_reports_hash_without_leaking_it(
    session, monkeypatch, capsys
) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    assert auth._main(["utils.auth", "check"]) == 0

    output = capsys.readouterr().out

    assert "hash" in output

    assert "64 chars" in output

    assert DIGEST not in output


def test_cli_check_exits_nonzero_when_unconfigured(
    session, capsys
) -> None:
    assert auth._main(["utils.auth", "check"]) == 1

    output = capsys.readouterr().out

    assert "NOT CONFIGURED" in output


def test_cli_check_flags_a_plaintext_secret(
    session, monkeypatch, capsys
) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, PASSWORD)

    assert auth._main(["utils.auth", "check"]) == 0

    output = capsys.readouterr().out

    assert "plaintext" in output

    assert PASSWORD not in output


def test_cli_verify_matches_and_mismatches(
    session, monkeypatch, capsys
) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DIGEST)

    assert auth._main(["utils.auth", "verify", PASSWORD]) == 0

    assert capsys.readouterr().out.strip() == "match"

    assert auth._main(["utils.auth", "verify", "nope"]) == 1

    assert capsys.readouterr().out.strip() == "no match"


def test_cli_without_a_known_command_prints_usage(session, capsys) -> None:
    assert auth._main(["utils.auth"]) == 1

    assert "Usage" in capsys.readouterr().out
