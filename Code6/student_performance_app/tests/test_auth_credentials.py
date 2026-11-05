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
# The deployed failure: "plaintext, 66 characters"
# ------------------------------------------------------------

# sha256("Admin123456") -- the credential that was actually configured on
# the live app while the app reported it as a 66-character plaintext.
DEPLOYED_DIGEST = (
    "1cd73874884db6a0f9f21d853d7e9eacdc773c39ee389060f5e96ae0bcb4773a"
)
DEPLOYED_PASSWORD = "Admin123456"


@pytest.mark.parametrize(
    "wrapped",
    [
        f"`{DEPLOYED_DIGEST}`",          # backticks from a chat message
        f"\u201c{DEPLOYED_DIGEST}\u201d",  # curly quotes from a phone
        f"'{DEPLOYED_DIGEST}' ",
        f"[{DEPLOYED_DIGEST}]",
        f"{DEPLOYED_DIGEST},",
        f"{DEPLOYED_DIGEST}\u00a0",     # non-breaking space
        f"\ufeff{DEPLOYED_DIGEST}\u200b",
    ],
)
def test_digest_is_recovered_from_packaging(session, monkeypatch, wrapped):
    """
    The live app reported the secret as "plaintext, 66 characters": a
    64-character digest plus two characters of packaging. The password was
    correct and could never match, because the comparison saw the
    packaging as part of the credential.
    """

    monkeypatch.setenv(ADMIN_PASSWORD_ENV, wrapped)

    status = auth.credential_status()

    assert status["kind"] == "hash", status

    assert status["length"] == 64

    assert auth.verify_admin_password(DEPLOYED_PASSWORD) is True


def test_packaging_is_reported_so_the_mistake_is_visible(
    session, monkeypatch
) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, f"`{DEPLOYED_DIGEST}`")

    status = auth.credential_status()

    assert status["packaging"] == ["backtick x2"]

    assert status["fingerprint"] == (
        DEPLOYED_DIGEST[:6] + "…" + DEPLOYED_DIGEST[-4:]
    )

    diagnostics = auth.credential_diagnostics()

    assert diagnostics["raw_length"] == 66

    assert diagnostics["clean_length"] == 64

    assert any("packaging" in warning for warning in diagnostics["warnings"])

    # Still never the secret itself.
    assert DEPLOYED_DIGEST not in repr(diagnostics)


def test_hex_of_the_wrong_length_is_not_mistaken_for_a_digest(
    session, monkeypatch
) -> None:
    """66 hex characters is not 64: do not silently truncate to a digest."""

    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DEPLOYED_DIGEST + "ab")

    assert auth.credential_kind() == "plaintext"

    diagnostics = auth.credential_diagnostics()

    assert diagnostics["clean_length"] == 66

    assert any(
        "hexadecimal but 66 characters" in warning
        for warning in diagnostics["warnings"]
    )


# ------------------------------------------------------------
# The other two ways a "wrong password" happens
# ------------------------------------------------------------

def test_typing_the_digest_is_diagnosed(session, monkeypatch) -> None:
    """The digest is what is stored; the password is what is typed."""

    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DEPLOYED_DIGEST)

    assert auth.verify_admin_password(DEPLOYED_DIGEST) is False

    assert auth.verification_hint() == "typed_the_digest"


def test_a_double_hashed_secret_is_diagnosed(session, monkeypatch) -> None:
    """
    ``python -m utils.auth hash "<the digest>"`` produces a secret that no
    password can satisfy. Saying so is the difference between a five-minute
    fix and an afternoon.
    """

    monkeypatch.setenv(
        ADMIN_PASSWORD_ENV, auth.hash_password(DEPLOYED_DIGEST)
    )

    assert auth.verify_admin_password(DEPLOYED_PASSWORD) is False

    assert auth.verification_hint() == "double_hashed"


def test_hint_is_cleared_by_a_successful_unlock(session, monkeypatch) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DEPLOYED_DIGEST)

    auth.verify_admin_password(DEPLOYED_DIGEST)

    assert auth.verification_hint() == "typed_the_digest"

    assert auth.verify_admin_password(DEPLOYED_PASSWORD) is True

    assert auth.verification_hint() is None


def test_no_hint_for_a_plain_typo(session, monkeypatch) -> None:
    monkeypatch.setenv(ADMIN_PASSWORD_ENV, DEPLOYED_DIGEST)

    assert auth.verify_admin_password("Adm1n123456") is False

    assert auth.verification_hint() is None


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
