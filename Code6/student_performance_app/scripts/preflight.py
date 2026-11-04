"""
====================================================================
Student Performance Assessment System
====================================================================

Script:
    scripts/preflight.py

Purpose:
    Answer one question before you deploy: "is this host and this
    checkout actually ready?"

Checks
------
  environment   : Python version, required packages importable
  dependencies  : installed versions match requirements.txt pins
  model bundle  : the pickle exists, loads, and reports its contract
  data storage  : the SQLite path and model directory are writable
  security      : Training Mode password configured (strict mode),
                  no development password in use, CORS/XSRF enabled
  configuration : target definition, tuning/calibration flags
  tests         : test suite present (and optionally run)

Usage
-----
    python scripts/preflight.py                # development check
    python scripts/preflight.py --production    # strict: fails on
                                                # insecure defaults
    python scripts/preflight.py --run-tests     # also run pytest

Exit codes
----------
    0  ready
    1  at least one FAIL (fix before deploying)
    (WARN items never fail the run; read them.)
====================================================================
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sqlite3
import sys

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(APP_DIR))

RESULTS: list[tuple[str, str, str]] = []

OK = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def record(status: str, area: str, message: str) -> None:
    RESULTS.append((status, area, message))


def check_python() -> None:

    version = sys.version_info

    # 3.12 is a hard floor, not a preference: xgboost 3.4.1 and shap
    # 0.54.1 are pinned in requirements.txt and both declare
    # requires-python >= 3.12. On 3.11, pip fails before the app starts.
    if version >= (3, 12):
        record(OK, "environment", f"Python {version.major}.{version.minor}")
    else:
        record(
            FAIL,
            "environment",
            f"Python {version.major}.{version.minor} is too old: the "
            "pinned xgboost and shap releases need 3.12+, so "
            "requirements.txt will not install",
        )


def check_imports() -> None:

    required = {
        "streamlit": "streamlit",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
        "matplotlib": "matplotlib",
        "openpyxl": "openpyxl",
        "reportlab": "reportlab",
    }

    for module, package in required.items():

        try:
            importlib.import_module(module)
            record(OK, "packages", f"{package} importable")
        except Exception as error:
            record(FAIL, "packages", f"{package} missing: {error}")

    # Optional extras
    for module in ("shap", "xgboost"):

        try:
            importlib.import_module(module)
            record(OK, "packages", f"{module} available")
        except Exception:
            record(
                WARN,
                "packages",
                f"{module} not installed -- the related feature "
                "degrades gracefully",
            )


def check_pins() -> None:

    requirements = APP_DIR / "requirements.txt"

    if not requirements.exists():
        record(FAIL, "dependencies", "requirements.txt is missing")
        return

    pins = {}

    for line in requirements.read_text().splitlines():

        match = re.match(
            r"^([A-Za-z0-9_.\-]+)==([0-9A-Za-z_.\-]+)$",
            line.strip(),
        )

        if match:
            pins[match.group(1).lower()] = match.group(2)

    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:  # pragma: no cover
        record(WARN, "dependencies", "importlib.metadata unavailable")
        return

    for package, expected in pins.items():

        try:
            installed = version(package)
        except PackageNotFoundError:
            record(FAIL, "dependencies", f"{package} not installed")
            continue

        if installed == expected:
            record(OK, "dependencies", f"{package}=={installed}")
        else:
            record(
                WARN,
                "dependencies",
                f"{package} {installed} installed, {expected} pinned "
                "-- a model bundle saved under a different "
                "scikit-learn may not load",
            )


def check_model_bundle(production: bool = False) -> None:

    model_path = APP_DIR / "models" / "best_model.pkl"

    if not model_path.exists():
        record(
            FAIL,
            "model",
            f"no model bundle at {model_path}. Train one with "
            "scripts/train_model.py (Assessment Mode needs it)",
        )
        return

    size_mb = model_path.stat().st_size / (1024 * 1024)

    record(OK, "model", f"bundle present ({size_mb:.1f} MB)")

    try:

        import joblib

        bundle = joblib.load(model_path)

        version = bundle.get("model_version", "unknown")

        best = bundle.get("best_model_name", "unknown")

        target = bundle.get("target_definition", "cpi (legacy)")

        record(
            OK,
            "model",
            f"loads: version={version}, best={best}, target={target}",
        )

        if "target_definition" not in bundle:
            record(
                WARN,
                "model",
                "bundle predates v3.0: it was trained on the legacy "
                "circular target. Retrain before quoting its metrics.",
            )

        # v7.2: is the bundle allowed to be the production model?
        definition = bundle.get("target_definition", "cpi")

        role = bundle.get(
            "bundle_role",
            "rule_reconstruction_baseline"
            if definition == "cpi"
            else "unknown",
        )

        if role == "rule_reconstruction_baseline":

            record(
                FAIL if production else WARN,
                "model",
                f"bundle_role={role} (target={definition!r}): the label "
                "is computed from the predictors, so its metrics measure "
                "rule reconstruction. Serve an exam_band bundle instead.",
            )

        elif role == "unknown":

            record(
                WARN,
                "model",
                "bundle records no bundle_role (written before v7.2); "
                "retrain to get a certified production bundle",
            )

        else:
            record(OK, "model", f"bundle_role={role}")

        split = bundle.get("split") or {}

        if split.get("strategy") == "temporal":
            record(
                OK,
                "model",
                "metrics come from a temporal split: "
                f"train {', '.join(split.get('train_periods') or [])} -> "
                f"test {', '.join(split.get('test_periods') or [])}",
            )
        elif split.get("strategy") == "random":
            record(
                WARN,
                "model",
                "metrics come from a random split, so they measure fit "
                "within one cohort -- run --split temporal before "
                "claiming forward generalisation",
            )
        else:
            record(WARN, "model", "bundle records no split provenance")

        metrics = bundle.get("metrics") or {}

        if metrics.get("baseline_majority") is None:
            record(
                WARN,
                "model",
                "bundle stores no baseline; treat its accuracy as "
                "uninterpretable until retrained",
            )

    except Exception as error:

        record(
            FAIL,
            "model",
            f"bundle could not be loaded: {error}. Check the "
            "scikit-learn version pin.",
        )


def check_storage() -> None:

    from backend.persistence.store import AssessmentStore

    db_path = os.environ.get(
        "SPAS_DB_PATH",
        str(APP_DIR / "data" / "spas.db"),
    )

    store = AssessmentStore(path=db_path, enabled=True)

    if store.healthy:
        record(OK, "storage", f"assessment store writable at {db_path}")
    else:
        record(
            WARN,
            "storage",
            f"persistence unavailable at {db_path} -- assessments "
            "will be lost on restart. Point SPAS_DB_PATH at a "
            "writable volume.",
        )

    models_dir = APP_DIR / "models"

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        probe = models_dir / ".preflight_probe"
        probe.write_text("ok")
        probe.unlink()
        record(OK, "storage", f"models directory writable ({models_dir})")
    except Exception as error:
        record(
            WARN,
            "storage",
            f"models directory not writable ({error}) -- Training "
            "Mode cannot save a model here",
        )


def check_security(production: bool) -> None:

    from utils.auth import is_using_default_password

    if is_using_default_password():

        message = (
            "SPAS_ADMIN_PASSWORD_HASH is not set: Training Mode "
            "uses the documented development password"
        )

        record(FAIL if production else WARN, "security", message)

    else:
        record(OK, "security", "administrator password hash configured")

    config = APP_DIR / ".streamlit" / "config.toml"

    if config.exists():

        text = config.read_text()

        cors = "enableCORS = true" in text
        xsrf = "enableXsrfProtection = true" in text

        if cors and xsrf:
            record(OK, "security", "CORS and XSRF protection enabled")
        else:
            record(
                WARN,
                "security",
                "CORS/XSRF protection disabled in config.toml -- "
                "acceptable for a local preview only",
            )

        if "showErrorDetails = false" in text:
            record(OK, "security", "tracebacks hidden from users")
        else:
            record(
                WARN,
                "security",
                "showErrorDetails is on: users may see tracebacks",
            )

    else:
        record(FAIL, "security", ".streamlit/config.toml is missing")

    record(
        OK if not production else WARN,
        "security",
        "Assessment Mode has no password (by design: it never "
        "touches the training data) -- confirm with your department",
    )


def check_configuration() -> None:

    from backend.config import (
        CALIBRATION,
        LABEL_POLICY,
        PREDICTOR_COLUMNS,
        TARGET_DEFINITION,
        TUNING,
    )

    record(OK, "configuration", f"target definition = {TARGET_DEFINITION}")

    if TARGET_DEFINITION == "cpi":
        record(
            WARN,
            "configuration",
            "legacy cpi target active: metrics measure rule "
            "reconstruction, not prediction",
        )

    record(
        OK,
        "configuration",
        f"label policy = {LABEL_POLICY['method']}, "
        f"{len(PREDICTOR_COLUMNS)} predictors",
    )

    record(
        OK,
        "configuration",
        f"tuning={'on' if TUNING['enabled'] else 'off'}, "
        f"calibration={'on' if CALIBRATION['enabled'] else 'off'}",
    )


def check_tests(run_tests: bool) -> None:

    tests_dir = APP_DIR / "tests"

    if not tests_dir.exists():
        record(FAIL, "tests", "tests/ directory missing")
        return

    count = len(list(tests_dir.glob("test_*.py")))

    record(OK, "tests", f"{count} test modules present")

    if not run_tests:
        record(
            WARN,
            "tests",
            "not executed (pass --run-tests to run them here)",
        )
        return

    import pytest

    exit_code = pytest.main(["-q", str(tests_dir)])

    record(
        OK if exit_code == 0 else FAIL,
        "tests",
        f"pytest exit code {exit_code}",
    )


def check_database_integrity() -> None:

    db_path = os.environ.get(
        "SPAS_DB_PATH",
        str(APP_DIR / "data" / "spas.db"),
    )

    if not Path(db_path).exists():
        return

    try:

        with sqlite3.connect(db_path) as connection:

            result = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()

        if result and result[0] == "ok":
            record(OK, "storage", "SQLite integrity_check = ok")
        else:
            record(FAIL, "storage", f"SQLite integrity_check: {result}")

    except Exception as error:

        record(WARN, "storage", f"could not verify database: {error}")


def main(argv=None) -> int:

    parser = argparse.ArgumentParser(
        description="Check whether this host is ready to run the app."
    )

    parser.add_argument(
        "--production",
        action="store_true",
        help="strict mode: insecure defaults become failures",
    )

    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="also run the pytest suite",
    )

    args = parser.parse_args(argv)

    print("=" * 66)
    print("PREFLIGHT — Student Performance Assessment System")
    print("=" * 66)

    check_python()
    check_imports()
    check_pins()
    check_model_bundle(production=args.production)
    check_storage()
    check_security(production=args.production)
    check_configuration()
    check_database_integrity()
    check_tests(run_tests=args.run_tests)

    print()

    for status, area, message in RESULTS:

        icon = {OK: "✅", WARN: "⚠️ ", FAIL: "❌"}[status]

        print(f"{icon} [{area:<13}] {message}")

    failures = sum(1 for status, _, _ in RESULTS if status == FAIL)

    warnings = sum(1 for status, _, _ in RESULTS if status == WARN)

    print()
    print("-" * 66)

    if failures:
        print(
            f"NOT READY — {failures} failure(s), {warnings} warning(s). "
            "Fix the failures above before deploying."
        )
        return 1

    if warnings:
        print(
            f"READY WITH WARNINGS — {warnings} warning(s). Read them; "
            "they are the things that bite in production."
        )
        return 0

    print("READY — no failures, no warnings.")
    return 0


if __name__ == "__main__":

    raise SystemExit(main())
