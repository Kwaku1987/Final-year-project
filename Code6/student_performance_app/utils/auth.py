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


def configured_hash() -> Optional[str]:
    """The admin password hash from the environment, if any."""

    value = os.environ.get(ADMIN_PASSWORD_ENV, "").strip()

    return value or None


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

        expected = hash_password(ADMIN_PASSWORD_DEFAULT)

        logger.warning(
            "No %s configured -- falling back to the documented "
            "development password.",
            ADMIN_PASSWORD_ENV,
        )

    provided = hash_password(password or "")

    if hmac.compare_digest(provided, expected):

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

    if len(argv) >= 3 and argv[1] == "hash":

        print(hash_password(argv[2]))

        return 0

    print(
        "Usage:\n"
        "  python -m utils.auth hash \"<password>\"\n\n"
        "Then export the result:\n"
        f"  export {ADMIN_PASSWORD_ENV}=<hash>"
    )

    return 1


if __name__ == "__main__":  # pragma: no cover

    import sys

    raise SystemExit(_main(sys.argv))
