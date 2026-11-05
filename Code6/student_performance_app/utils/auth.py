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

# Why the last attempt failed, beyond "it did" -- see verification_hint().
HINT_KEY = "admin_auth_hint"


# ============================================================
# HASHING
# ============================================================

def hash_password(password: str) -> str:
    """SHA-256 hex digest of a password."""

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# A 64-character hex run that is not part of a longer hex string. Used to
# recover a digest that arrived wrapped in something -- backticks from a
# chat message, curly quotes from a phone keyboard, a trailing comma.
_DIGEST_ANYWHERE = re.compile(
    r"(?<![0-9a-fA-F])([0-9a-fA-F]{64})(?![0-9a-fA-F])"
)

# Characters that are packaging, never part of a credential. Straight
# quotes and whitespace were already handled; the rest are what a secret
# picks up when it is copied out of a chat window or a phone.
_PACKAGING_CHARS = (
    " \t\r\n"
    "\"'`"
    "\u2018\u2019"  # curly single quotes
    "\u201c\u201d"  # curly double quotes
    "\u00a0"        # non-breaking space
    "\u200b\u200c\u200d"  # zero-width space / non-joiner / joiner
    "\ufeff"        # byte-order mark
)

_PACKAGING_NAMES = {
    "`": "backtick",
    "\u2018": "curly quote",
    "\u2019": "curly quote",
    "\u201c": "curly quote",
    "\u201d": "curly quote",
    "\u00a0": "non-breaking space",
    "\u200b": "zero-width space",
    "\u200c": "zero-width non-joiner",
    "\u200d": "zero-width joiner",
    "\ufeff": "byte-order mark",
    "\n": "newline",
    "\r": "carriage return",
    "\t": "tab",
}


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

    text = str(value).strip(_PACKAGING_CHARS)

    # A pasted TOML/env line, e.g. SPAS_ADMIN_PASSWORD_HASH = "abc"
    if "=" in text:

        head, _, tail = text.partition("=")

        if head.strip().upper().startswith("SPAS_"):

            text = tail.strip(_PACKAGING_CHARS)

    if text.lower().startswith("sha256:"):

        text = text.split(":", 1)[1].strip(_PACKAGING_CHARS)

    # A digest wrapped in anything else -- `abc…`, "abc…" from a phone
    # keyboard, "abc…," in a sentence -- is still that digest. This is
    # what a secret that reported itself as "plaintext, 66 characters"
    # actually was: a 64-character SHA-256 with two characters of
    # packaging around it.
    match = _DIGEST_ANYWHERE.search(text)

    if match:

        return match.group(1).lower()

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
    """
    Non-secret description of the configured credential, for the UI.

    ``packaging`` lists what had to be stripped before the value could be
    used (backticks, curly quotes, a trailing newline …). It is empty for
    a clean secret, which makes a paste mistake visible instead of
    mysterious. ``fingerprint`` is only ever produced for a digest: six
    leading and four trailing hex characters identify which secret is
    loaded without disclosing it.
    """

    raw = os.environ.get(ADMIN_PASSWORD_ENV, "")

    value = configured_hash()

    packaging = credential_packaging(raw)

    return {
        "configured": value is not None,
        "kind": credential_kind(),
        "length": len(value) if value else 0,
        "using_default": value is None,
        "variable": ADMIN_PASSWORD_ENV,
        "packaging": packaging,
        "fingerprint": (
            f"{value[:6]}…{value[-4:]}"
            if value and _HEX_DIGEST.match(value)
            else ""
        ),
    }


def credential_packaging(raw: str) -> list[str]:
    """Describe the non-credential characters that were stripped."""

    removed: list[str] = []

    for char, name in _PACKAGING_NAMES.items():

        count = raw.count(char)

        if count:

            removed.append(f"{name} x{count}")

    return removed


def credential_diagnostics() -> dict:
    """
    Everything a deployment needs to answer "why is my password wrong?".

    Never includes the secret itself -- only its shape.
    """

    raw = os.environ.get(ADMIN_PASSWORD_ENV, "")

    clean = normalize_credential(raw)

    status = credential_status()

    warnings: list[str] = []

    if not clean:

        warnings.append(
            f"{ADMIN_PASSWORD_ENV} is not set: the documented development "
            "password is active."
        )

    else:

        is_hex = bool(re.fullmatch(r"[0-9a-fA-F]+", clean))

        if is_hex and len(clean) != 64:

            warnings.append(
                f"the value is hexadecimal but {len(clean)} characters "
                "long; a SHA-256 digest is exactly 64, so this is being "
                "treated as a plaintext password."
            )

        if status["kind"] == "plaintext":

            warnings.append(
                "the value is being treated as a plaintext password "
                "(it is not a 64-character SHA-256 digest)."
            )

        if status["packaging"]:

            warnings.append(
                "packaging was stripped from the secret ("
                + ", ".join(status["packaging"])
                + "); this is repaired automatically, and is the most "
                "common reason a correct-looking value never matched."
            )

        if len(raw) != len(clean) and not status["packaging"]:

            warnings.append(
                f"the raw secret is {len(raw)} characters but its usable "
                f"form is {len(clean)}."
            )

    return {
        "raw_length": len(raw),
        "clean_length": len(clean),
        **status,
        "warnings": warnings,
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


def verification_hint() -> Optional[str]:
    """
    The most recent diagnosis of a failed unlock, if there is one.

    Values:
        "double_hashed"   the stored secret is SHA-256 of SHA-256
        "typed_the_digest" the entered text *is* the stored digest
        None              nothing specific to say

    Both are real mistakes people make with a secrets box, and both look
    identical to a wrong password from the outside.
    """

    return st.session_state.get(HINT_KEY)


def clear_verification_hint() -> None:

    st.session_state[HINT_KEY] = None


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

        st.session_state[HINT_KEY] = None

        logger.info("Training Mode unlocked.")

        return True

    _record_failure()

    st.session_state[HINT_KEY] = _diagnose_mismatch(expected, provided)

    logger.warning(
        "Failed Training Mode unlock (%d/%d)%s.",
        _attempts(),
        AUTH_MAX_ATTEMPTS,
        (
            f" -- {st.session_state[HINT_KEY]}"
            if st.session_state[HINT_KEY]
            else ""
        ),
    )

    return False


def _diagnose_mismatch(expected: str, provided: str) -> Optional[str]:
    """
    Name the two failures that a "wrong password" message hides.

    Neither is accepted -- they are only diagnosed, because a double-hashed
    secret means the deployment stored the wrong thing, and typing the
    digest is not authentication.
    """

    if not provided:

        return None

    if _HEX_DIGEST.match(expected):

        # The administrator pasted the *stored hash* into the password box.
        if hmac.compare_digest(provided.lower(), expected):

            return "typed_the_digest"

        # The secret is hash(hash(password)): the digest was hashed again,
        # so no password can ever match it.
        if hmac.compare_digest(hash_password(hash_password(provided)), expected):

            return "double_hashed"

    elif _HEX_DIGEST.match(provided.strip().lower()):

        return "typed_the_digest"

    return None


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
        status = credential_diagnostics()

        print(
            f"{ADMIN_PASSWORD_ENV}: {status['kind']} "
            f"({status['clean_length']} chars usable, "
            f"{status['raw_length']} raw)"
        )

        if status["fingerprint"]:

            print(f"  digest fingerprint: {status['fingerprint']}")

        if status["packaging"]:

            print(
                "  packaging stripped: "
                + ", ".join(status["packaging"])
            )

        for warning in status["warnings"]:

            print(f"  note: {warning}")

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
