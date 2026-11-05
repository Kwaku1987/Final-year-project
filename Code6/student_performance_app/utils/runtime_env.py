"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    utils/runtime_env.py

Purpose:
    Make configuration work on hosts that only expose ``st.secrets``.

Why this exists
---------------
Every setting in this project is read from the environment
(``SPAS_ADMIN_PASSWORD_HASH``, ``SPAS_DB_PATH``, ``SPAS_TARGET_DEFINITION``,
...). That is correct for Docker, Compose, Render and a bare server, where
environment variables are how you configure a process.

Streamlit Community Cloud is different: it gives you a *Secrets* box that
populates ``st.secrets``, and environment variables set in its UI do not
reliably reach ``os.environ``. Without a bridge the app would start on
Community Cloud with no password hash at all and silently fall back to the
documented development password -- the worst possible failure mode, because
nothing looks broken.

So: copy known secret keys into ``os.environ`` before anything reads config.

Rules
-----
* **Never overwrite** an existing environment variable. Real env vars (Docker,
  Render, a shell export) win over the secrets file. That keeps the
  container deployments authoritative and makes this module inert there.
* **Only known keys** are copied. A stray secret cannot inject an arbitrary
  variable into the process, so a typo in the Secrets box fails loudly as
  "not set" instead of doing something unexpected.
* **TOML booleans become 1/0**, because the rest of the codebase compares
  strings (``os.environ.get("SPAS_TUNING") == "1"``). ``True`` was becoming
  the string "True" and silently disabling the feature.
* **Failure is silent by design.** Locally there is no ``secrets.toml``, and
  ``st.secrets`` raises when accessed. Missing secrets must not stop the app
  from starting; ``apply_secrets()`` returns what it applied so the caller can
  log it.

Usage
-----
At the very top of ``app.py``, before importing anything that reads config::

    from utils.runtime_env import apply_secrets
    apply_secrets()
====================================================================
"""

from __future__ import annotations

import os

from typing import Any, Iterable, Mapping, Optional

# Every environment variable the application understands. Keep this in sync
# with .env.example -- a key that is missing here will not be picked up from
# Streamlit Cloud's Secrets box.
SECRET_KEYS = (
    # required in production
    "SPAS_ADMIN_PASSWORD_HASH",
    # storage
    "SPAS_DB_PATH",
    "SPAS_DB_ENABLED",
    # model behaviour
    "SPAS_TARGET_DEFINITION",
    "SPAS_LABEL_METHOD",
    "SPAS_TUNING",
    "SPAS_CALIBRATION",
)

_TRUE = {"true", "yes", "on", "1"}

_FALSE = {"false", "no", "off", "0"}

_TRUE_WORDS = {"true", "false", "yes", "no", "on", "off"}

_MISSING = object()


def _coerce(value: Any) -> str:
    """
    Turn a TOML value into the string the rest of the code expects.

    ``True`` -> ``"1"``, ``False`` -> ``"0"`` so flags keep working;
    numbers and strings pass through.
    """

    if isinstance(value, bool):

        return "1" if value else "0"

    return str(value)


def _lookup(source: Any, key: str) -> Any:
    """Read one key from a mapping-like or Streamlit secrets object."""

    try:

        return source[key]

    except Exception:

        return _MISSING


def apply_secrets(
    source: Optional[Any] = None,
    keys: Iterable[str] = SECRET_KEYS,
    environ: Optional[Mapping[str, str]] = None,
) -> list[str]:
    """
    Copy known keys from ``st.secrets`` into ``os.environ``.

    Parameters
    ----------
    source
        Anything mapping-like. Defaults to ``st.secrets``. Tests pass a dict.
    keys
        Keys to consider; defaults to ``SECRET_KEYS``.
    environ
        Mapping to read "already set?" from. Defaults to ``os.environ``.

    Returns
    -------
    list[str]
        The keys actually applied, in order. Empty on a normal local run,
        which is not an error.
    """

    current = os.environ if environ is None else environ

    if source is None:

        try:

            import streamlit as st

            source = st.secrets

            # Touch the object: st.secrets is lazy and raises only on access.
            _ = list(source.keys())

        except Exception:

            # No secrets configured (local run, Docker, Render). Nothing to do.
            return []

    applied: list[str] = []

    for key in keys:

        if key in current and str(current[key]).strip():

            # A real environment variable wins.
            continue

        value = _lookup(source, key)

        if value is _MISSING or value is None:

            continue

        text = _coerce(value).strip()

        if not text:

            continue

        os.environ[key] = text

        applied.append(key)

    return applied


def describe(applied: Iterable[str]) -> str:
    """One line for the log: which keys came from secrets, redacted."""

    keys = list(applied)

    if not keys:

        return "runtime configuration: environment only"

    safe = [
        # Never log a credential, not even a hash.
        key if "PASSWORD" not in key else f"{key} (redacted)"
        for key in keys
    ]

    return "runtime configuration from st.secrets: " + ", ".join(safe)
