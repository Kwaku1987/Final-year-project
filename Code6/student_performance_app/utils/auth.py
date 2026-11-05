"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    utils/auth.py

Purpose:
    Minimal access control for Training Mode.

Why this exists
---------------
In v2.x the "Administrator" / "Lecturer" choice was a button, not
a control: anyone could enter Training Mode, upload a dataset,
retrain, and overwrite or delete the model that lecturers depend
on. That is a data-protection problem as well as an operational
one -- student marks are personal data under Ghana's Data
Protection Act, 2012 (Act 843).

This module gates Training Mode behind a password whose SHA-256
digest is supplied through the environment:

    export SPAS_ADMIN_PASSWORD_HASH=$(python -m utils.auth hash "s3cret")
    streamlit run app.py

If no hash is configured, the app falls back to a documented
development password and says so loudly in the UI. That keeps the
demo runnable while making the insecure default impossible to miss.

What this is NOT
----------------
This is a single shared secret for a demo/teaching application.
It is not a user account system, gives no per-user audit trail, and
does not encrypt anything at rest. Production use needs proper
identity (institutional SSO), HTTPS, and a real audit log.

Version:
    1.0
====================================================================
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re

from typing import Optional

import streamlit as st

from backend.config import (
    ADMIN_PASSWORD_DEFAULT,
    ADMIN_PASSWORD_ENV,
    AUTH_MAX_ATTEMPTS,
)

logger = logging.getLogger(__name__)

SESSION_KEY = "admin_authenticated"

ATTEMPTS_KEY = "admin_auth_attempts"


# ============================================================
# HASHING
# ============================================================

def hash_password(password: str) -> str:
    """SHA-256 hex digest of a password."""

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def normalize_credential(value: Optional[str]) -> str:
    """
    Turn whatever was pasted into the Secrets box into a usable value.

    People paste things that are *nearly* right, and every one of these
    variants used to be rejected as "wrong password" with no clue why:

        SPAS_ADMIN_PASSWORD_HASH = "abc..."   (the whole TOML line)
        "abc..."                              (the surrounding quotes)
        abc...\n                              (a trailing newline)
        sha256:abc...                         (an OpenSSL-style prefix)
        ABC...                                (uppercase hex)

    All of those mean the same credential. Normalising here is not a
    security relaxation: it changes nothing about *what* is compared,
    only strips packaging that is never part of a secret.
    """

    if value is None:

        return ""

    text = str(value).strip().strip('"').strip("'").strip()

    # A pasted TOML/env line, e.g. SPAS_ADMIN_PASSWORD_HASH = "abc"
    if "=" in text:

        head, _, tail = text.partition("=")

        if head.strip().upper().startswith("SPAS_"):

            text = tail.strip().strip('"').strip("'").strip()

    if text.lower().startswith("sha256:"):

        text = text.split(":", 1)[1].strip()

    # Hex digests are case-insensitive; normalise so a capitalised
    # copy-and-paste still matches.
    if _HEX_DIGEST.match(text.lower()):

        return text.lower()

    return text


def configured_hash() -> Optional[str]:
    """
    The configured credential, or None when nothing is set.

    Despite the name this returns whatever was supplied: a SHA-256
    digest when the value is 64 hex characters, otherwise the plain
    password. ``credential_kind()`` says which.
    """

    value = normalize_credential(
        os.environ.get(ADMIN_PASSWORD_ENV, "")
    )

    return value or None


def credential_kind() -> str:
    """
    "hash" | "plaintext" | "default".

    "plaintext" is not an error — it works — but it means the secret is
    stored readable, which is worth saying out loud in the UI.
    """

    value = configured_hash()

    if value is None:

        return "default"

    return "hash" if _HEX_DIGEST.match(value) else "plaintext"


def credential_status() -> dict:
    """Non-secret description of the configured credential, for the UI."""

    value = configured_hash()

    return {
        "configured": value is not None,
        "kind": credential_kind(),
        "length": len(value) if value else 0,
        "using_default": value is None,
        "variable": ADMIN_PASSWORD_ENV,
    }


def is_using_default_password() -> bool:
    """True when no hash is configured (development fallback)."""

    return configured_hash() is None


# ============================================================
# SESSION STATE
# ============================================================

def is_admin_unlocked() -> bool:
    """Whether Training Mode has been unlocked this session."""

    return bool(st.session_state.get(SESSION_KEY, False))


def lock_admin() -> None:
    """Re-lock Training Mode (used on mode switch / reset)."""

    st.session_state[SESSION_KEY] = False

    st.session_state[ATTEMPTS_KEY] = 0


def _attempts() -> int:
    return int(st.session_state.get(ATTEMPTS_KEY, 0))


def _record_failure() -> None:
    st.session_state[ATTEMPTS_KEY] = _attempts() + 1


def attempts_remaining() -> int:
    return max(AUTH_MAX_ATTEMPTS - _attempts(), 0)


def reset_attempts() -> None:
    """
    Clear the failed-attempt counter for this session.

    The counter lives in ``st.session_state``, which survives a browser
    reload -- so without this, five typos locked Training Mode for the
    rest of the session with no way back. The counter is a courtesy
    guard against fat-fingering, not a security boundary: an attacker
    who can reach this page can also start a fresh session. The real
    boundary is the password itself.
    """

    st.session_state[ATTEMPTS_KEY] = 0

    logger.info("Admin unlock attempt counter reset.")


# ============================================================
# VERIFICATION
# ============================================================

def verify_admin_password(password: str) -> bool:
    """
    Check a password and update session state on success.

    Uses hmac.compare_digest for a constant-time comparison.
    """

    if _attempts() >= AUTH_MAX_ATTEMPTS:

        logger.warning(
            "Admin unlock blocked: too many failed attempts."
        )

        return False

    expected = configured_hash()

    if expected is None:

        expected = ADMIN_PASSWORD_DEFAULT

        logger.warning(
            "No %s configured -- falling back to the documented "
            "development password.",
            ADMIN_PASSWORD_ENV,
        )

    provided = password or ""

    # Compare digests in both branches so the comparison is constant
    # time whether the stored credential is a hash or a plaintext
    # password (a plaintext secret compared directly would leak its
    # length and prefix through timing).
    if _HEX_DIGEST.match(expected):

        matched = hmac.compare_digest(
            hash_password(provided),
            expected,
        )

    else:

        matched = hmac.compare_digest(
            hash_password(provided),
            hash_password(expected),
        )

    if matched:

        st.session_state[SESSION_KEY] = True

        st.session_state[ATTEMPTS_KEY] = 0

        logger.info("Training Mode unlocked.")

        return True

    _record_failure()

    logger.warning(
        "Failed Training Mode unlock (%d/%d).",
        _attempts(),
        AUTH_MAX_ATTEMPTS,
    )

    return False


# ============================================================
# CLI HELPER
# ============================================================
# python -m utils.auth hash "my-password"
# ============================================================

def _main(argv: list[str]) -> int:
    """
    CLI entry point.

    ``argv`` is the raw ``sys.argv``, so ``python -m utils.auth check``
    arrives as a *two*-element list -- the argument-count guards are per
    subcommand, not global (a single global ``>= 3`` silently turned
    ``check`` into a usage message, which is how this was found).
    """

    if len(argv) >= 3 and argv[1] == "hash":

        print(hash_password(argv[2]))

        return 0

    if len(argv) >= 2 and argv[1] == "check":

        # Diagnose a deployment without printing the secret.
        status = credential_status()

        print(f"{ADMIN_PASSWORD_ENV}: {status['kind']} ({status['length']} chars)")

        if status["kind"] == "hash":

            verdict = "OK — SHA-256 hash configured"

        elif status["kind"] == "plaintext":

            verdict = (
                "OK — plaintext password configured. It works, but store "
                "the hash instead: python -m utils.auth hash \"<password>\""
            )

        else:

            verdict = (
                "NOT CONFIGURED — the development password is active. "
                "Set the variable before deploying."
            )

        print(verdict)

        print(
            f"\nTo verify a password against it: "
            f"python -m utils.auth verify <password>"
        )

        return 0 if status["configured"] else 1

    if len(argv) >= 3 and argv[1] == "verify":

        expected = configured_hash() or ADMIN_PASSWORD_DEFAULT

        provided = argv[2]

        if _HEX_DIGEST.match(expected):

            ok = hash_password(provided) == expected

        else:

            ok = provided == expected

        print("match" if ok else "no match")

        return 0 if ok else 1

    print(
        "Usage:\n"
        "  python -m utils.auth hash \"<password>\"    # print the SHA-256 digest\n"
        "  python -m utils.auth check                 # diagnose the configured secret\n"
        "  python -m utils.auth verify <password>     # test a password non-interactively\n\n"
        "Then set the credential:\n"
        f"  export {ADMIN_PASSWORD_ENV}=<hash>"
    )

    return 1


if __name__ == "__main__":  # pragma: no cover

    import sys

    raise SystemExit(_main(sys.argv))
