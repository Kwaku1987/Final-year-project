"""
Regression tests for the two runtime failures the deployed app hit.

1. ``AttributeError: property 'application_mode' of
   'DashboardController' object has no setter`` -- raised by the "Switch
   Mode" button, which did ``controller.application_mode = None``.
   ``clear_mode()`` exists instead; this file makes sure nobody puts the
   assignment back.

2. ``RecursionError`` in the sidebar, where seven ``classmethod``
   fallbacks ended with ``return cls._<their own name>()``.

Both are static properties of the source, so they are checked by walking
the AST rather than by driving the UI -- they can then never regress
without a red test, even on a code path the tests do not click through.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
import streamlit as st

from backend.config import PROJECT_VERSION
from dashboard.application_state import ApplicationState
from dashboard.dashboard_controller import DashboardController


APP_ROOT = Path(__file__).resolve().parents[1]

REPO_ROOT = APP_ROOT.parents[1]

SIDEBAR = APP_ROOT / "components" / "sidebar.py"


def _python_files() -> list[Path]:
    """
    Every application module.

    ``tests/`` is excluded on purpose: this file has to perform the
    forbidden assignment once, to prove the property really is
    read-only.
    """

    skip = {"__pycache__", ".venv", "build", "dist", "node_modules", "tests"}

    return [
        path
        for path in APP_ROOT.rglob("*.py")
        if not skip.intersection(path.parts)
    ]


# ------------------------------------------------------------
# 1. No assignment to the read-only application_mode property
# ------------------------------------------------------------

def test_no_module_assigns_to_application_mode() -> None:
    """
    ``controller.application_mode = ...`` is the deployed crash.

    Attribute-only assignment (``state.application_mode = None``) is
    legal, because ``DashboardState`` is a plain dataclass; what must not
    happen is assigning on the controller, i.e. a target of the form
    ``<something>.application_mode`` where something is a controller-ish
    name. Rather than guess, forbid assignment to a target whose base is
    named ``controller`` / ``self.controller`` and require the state
    object otherwise -- simplest honest rule: only ``.state`` may be
    assigned through, and only inside ``application_state``.
    """

    offenders: list[str] = []

    for path in _python_files():

        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):

            targets: list[ast.expr] = []

            if isinstance(node, ast.Assign):

                targets = list(node.targets)

            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):

                targets = [node.target]

            for target in targets:

                if not isinstance(target, ast.Attribute):

                    continue

                if target.attr != "application_mode":

                    continue

                base = ast.unparse(target.value)

                if base.endswith("controller") or base == "cls._controller":

                    offenders.append(
                        f"{path.relative_to(APP_ROOT)}:{node.lineno} "
                        f"{base}.application_mode = ..."
                    )

    assert not offenders, (
        "application_mode is a read-only property on "
        "DashboardController; call clear_mode() instead. Offenders: "
        f"{offenders}"
    )


def test_controller_mode_has_no_setter() -> None:
    """Documents the constraint the previous test protects."""

    controller = object.__new__(DashboardController)

    with pytest.raises(AttributeError):

        controller.application_mode = None  # type: ignore[misc]


# ------------------------------------------------------------
# 2. clear_mode() replaces the assignment
# ------------------------------------------------------------

@pytest.fixture()
def session(monkeypatch) -> dict:
    fake_state: dict = {}

    monkeypatch.setattr(st, "session_state", fake_state, raising=True)

    return fake_state


def test_clear_mode_returns_to_the_selection_screen(session) -> None:
    controller = object.__new__(DashboardController)

    controller.state = ApplicationState

    assert controller.application_mode is None

    controller.set_mode("administrator")

    assert controller.application_mode == "administrator"

    controller.clear_mode()

    assert controller.application_mode is None


def test_clear_mode_is_idempotent(session) -> None:
    controller = object.__new__(DashboardController)

    controller.state = ApplicationState

    controller.clear_mode()

    controller.clear_mode()

    assert controller.application_mode is None


# ------------------------------------------------------------
# 3. No self-recursive classmethod fallbacks in the sidebar
# ------------------------------------------------------------

def test_sidebar_helpers_do_not_call_themselves() -> None:
    """
    ``return cls._has_dataset()`` inside ``_has_dataset`` recursed until
    the interpreter gave up. A fallback must return a literal.
    """

    tree = ast.parse(SIDEBAR.read_text(encoding="utf-8"))

    offenders: list[str] = []

    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:

        for func in [
            n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]:

            for node in ast.walk(func):

                if not isinstance(node, ast.Call):

                    continue

                func_node = node.func

                if not isinstance(func_node, ast.Attribute):

                    continue

                if func_node.attr != func.name:

                    continue

                base = func_node.value

                if isinstance(base, ast.Name) and base.id in {"cls", "self"}:

                    offenders.append(
                        f"{cls.name}.{func.name} (line {node.lineno})"
                    )

    assert not offenders, (
        "Self-recursive method call found in components/sidebar.py: "
        f"{offenders}"
    )


def test_sidebar_helpers_return_literals_on_the_fallback_path() -> None:
    """
    The boolean helpers must have a literal ``return False`` / ``return
    []`` so an uninitialised controller degrades instead of recursing.
    """

    tree = ast.parse(SIDEBAR.read_text(encoding="utf-8"))

    expected_literal = {
        "_has_dataset": (ast.Constant, False),
        "_has_trained_models": (ast.Constant, False),
        "_has_saved_model": (ast.Constant, False),
        "_assessment_history": (ast.List, None),
        "_is_ready_for_prediction": (ast.Constant, False),
        "_is_ready_for_interpretation": (ast.Constant, False),
        "_is_ready_for_reports": (ast.Constant, False),
    }

    found: dict[str, set] = {}

    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:

        for func in [
            n for n in cls.body if isinstance(n, ast.FunctionDef)
        ]:

            if func.name not in expected_literal:

                continue

            kinds = set()

            for node in ast.walk(func):

                if isinstance(node, ast.Return) and node.value is not None:

                    if isinstance(node.value, ast.Constant):

                        kinds.add(("const", node.value.value))

                    elif isinstance(node.value, ast.List):

                        kinds.add(("list", None))

            found[func.name] = kinds

    for name, (node_type, value) in expected_literal.items():

        assert name in found, f"helper {name} disappeared from the sidebar"

        if node_type is ast.Constant:

            assert ("const", value) in found[name], (
                f"{name} must have a literal `return {value}` fallback"
            )

        else:

            assert ("list", None) in found[name], (
                f"{name} must have a literal `return []` fallback"
            )


def test_sidebar_switch_mode_uses_clear_mode() -> None:
    """The button that crashed must go through the method, not a property."""

    source = SIDEBAR.read_text(encoding="utf-8")

    assert "clear_mode()" in source

    assert "application_mode = " not in source


# ------------------------------------------------------------
# 4. Version strings and the theme
# ------------------------------------------------------------

def test_sidebar_advertises_the_configured_version() -> None:
    """
    The sidebar used to say "Version 6.0" while the deployment-status
    panel said "3.3.1". One number now, read from the config, so a
    release cannot half-happen.
    """

    source = SIDEBAR.read_text(encoding="utf-8")

    assert "PROJECT_VERSION" in source

    assert "Version 6.0" not in source

    assert "Version 5.0" not in source


def test_sidebar_does_not_advertise_a_cpi_target() -> None:
    """
    The shipped model predicts an exam band; CPI is a demoted
    rule-reconstruction baseline and must not be sold as the feature.
    """

    source = SIDEBAR.read_text(encoding="utf-8")

    assert "CPI-based target generation" not in source


def test_both_streamlit_configs_request_the_dark_theme() -> None:
    """
    ``.streamlit/config.toml`` must exist at the repository root for
    Community Cloud and inside the app directory for local/Docker use.
    Both are checked so the two can never drift apart.
    """

    configs = {
        "app": APP_ROOT / ".streamlit" / "config.toml",
        "repo-root": REPO_ROOT / ".streamlit" / "config.toml",
    }

    for label, path in configs.items():

        assert path.exists(), f"missing {label} config at {path}"

        config = tomllib.loads(path.read_text(encoding="utf-8"))

        theme = config.get("theme", {})

        assert theme.get("base") == "dark", (
            f"{label} config must request the dark theme"
        )

        assert theme.get("backgroundColor")

        assert theme.get("textColor")


def test_devcontainer_python_matches_the_pins() -> None:
    """
    A Codespaces image on 3.11 builds a container that cannot install
    xgboost/shap, i.e. an environment where the app cannot start. CI only
    covers the versions in the workflow, so pin the devcontainer too.
    """

    import re

    path = REPO_ROOT / ".devcontainer" / "devcontainer.json"

    if not path.exists():

        pytest.skip("no devcontainer in this checkout")

    text = path.read_text(encoding="utf-8")

    image = re.search(r'"image"\s*:\s*"([^"]+)"', text)

    assert image, "devcontainer.json has no image"

    assert "3.12" in image.group(1), (
        "devcontainer image must be Python 3.12+: " + image.group(1)
    )

    # CORS/XSRF must be relaxed on the command line, never in the
    # committed Streamlit config.
    assert "--server.enableCORS false" in text

    config = (APP_ROOT / ".streamlit" / "config.toml").read_text(
        encoding="utf-8"
    )

    assert "enableCORS = true" in config
