"""
===============================================================
Sidebar Component
===============================================================

Project:
    Student Performance Assessment System

Module:
    components/sidebar.py

Description:
    Reusable Streamlit sidebar component responsible for:

    - Navigation
    - Workflow Progress
    - System Status
    - Quick Information

Version:
    6.0

Notes
-----
v5.x read every status flag from ApplicationState while app.py
held the DashboardController in st.session_state. Two sources of
truth meant the sidebar could say "Models: trained" while the page
said "no models". v6.0 takes the controller as its single source
of truth (passed in by app.py) and only falls back to
ApplicationState for notifications.
===============================================================
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import streamlit as st

from backend.config import PROJECT_VERSION

from dashboard.application_state import ApplicationState


# ============================================================
# Navigation Pages
# ============================================================

class DashboardPage(str, Enum):
    """Supported dashboard pages."""

    HOME = "🏠 Home"

    DATASET = "📂 Dataset"

    TRAINING = "⚙ Model Training"

    PREDICTION = "🔮 Prediction"

    ASSESSMENT = "📋 Assess Student"

    BATCH = "👥 Batch Prediction"

    INTERPRETATION = "🧠 Explainable AI"

    REPORTS = "📄 Reports"


# Pages shown per application mode. HOME and INTERPRETATION are
# shared -- Administrator sees the model just trained this
# session, Lecturer sees the loaded saved model. Batch prediction
# is useful in both modes -- an Administrator testing a freshly
# trained model against many students, or a Lecturer running an
# entire class roster through the saved model.
ADMINISTRATOR_PAGES = [
    DashboardPage.HOME,
    DashboardPage.DATASET,
    DashboardPage.TRAINING,
    DashboardPage.PREDICTION,
    DashboardPage.BATCH,
    DashboardPage.INTERPRETATION,
    DashboardPage.REPORTS,
]

LECTURER_PAGES = [
    DashboardPage.HOME,
    DashboardPage.ASSESSMENT,
    DashboardPage.BATCH,
    DashboardPage.INTERPRETATION,
]


# ============================================================
# Sidebar Component
# ============================================================

class Sidebar:
    """
    Reusable sidebar renderer.

    Every dashboard page should simply call:

        Sidebar.render(current_page)
    """

    APP_TITLE = "🎓 Student Performance"

    # One version, taken from the backend config, so the sidebar can
    # never advertise a different release from the one CHANGELOG.md and
    # the deployment-status panel describe.
    APP_VERSION = f"Version {PROJECT_VERSION}"

    # Controller captured by render(), so the helper methods
    # below always read the same state the pages read.
    _controller = None

    # --------------------------------------------------------

    @classmethod
    def render(
        cls,
        current_page: DashboardPage,
        mode: str = "administrator",
        controller=None,
    ) -> DashboardPage:
        """
        Draw the complete sidebar.

        Parameters
        ----------
        controller:
            The DashboardController for this session. When
            supplied, it is the single source of truth for every
            status flag shown here.

        Parameters
        ----------
        current_page
            Current active page.

        mode
            "administrator" (Training Mode) or "lecturer"
            (Assessment Mode) -- determines which pages and
            workflow/status items are shown.

        Returns
        -------
        DashboardPage
            Selected page.
        """

        cls._controller = controller

        with st.sidebar:

            cls._header()

            cls._mode_badge(mode)

            page = cls._navigation(current_page, mode)

            st.divider()

            cls._workflow(mode)

            st.divider()

            cls._system_status(mode)

            st.divider()

            cls._deployment_status()

            st.divider()

            cls._project_info()

        return page


    # ========================================================
    # STATE HELPERS (controller first, ApplicationState last)
    # ========================================================

    @classmethod
    def _has_dataset(cls) -> bool:

        if cls._controller is not None:
            return bool(cls._controller.has_dataset)

        # No controller: nothing has been loaded. (This used to call
        # itself, so the first click on a fresh session ended in
        # RecursionError instead of "not yet".)
        return False

    @classmethod
    def _has_trained_models(cls) -> bool:

        if cls._controller is not None:
            return bool(cls._controller.has_trained_models)

        return False

    @classmethod
    def _has_saved_model(cls) -> bool:

        if cls._controller is not None:
            return bool(cls._controller.has_saved_model)

        return False

    @classmethod
    def _assessment_history(cls) -> list:

        if cls._controller is not None:
            return list(
                cls._controller.get_assessment_history()
            )

        return []

    @classmethod
    def _is_ready_for_prediction(cls) -> bool:

        if cls._controller is not None:
            return bool(
                cls._controller.is_ready_for_prediction
            )

        return False

    @classmethod
    def _is_ready_for_interpretation(cls) -> bool:

        if cls._controller is not None:
            return (
                cls._controller.state
                .get_interpretation_report()
                is not None
            )

        return False

    @classmethod
    def _is_ready_for_reports(cls) -> bool:

        if cls._controller is not None:
            return (
                cls._controller.get_report() is not None
            )

        return False

    @classmethod
    def _switch_mode(cls) -> None:
        """Leave the current mode, re-locking Training Mode."""

        if cls._controller is not None:

            # application_mode is a read-only property; the old
            # assignment raised AttributeError in the deployed app,
            # which is what "Switch Mode" did every time. The
            # controller clears the state itself.
            cls._controller.clear_mode()

        try:

            from utils.auth import lock_admin

            lock_admin()

        except Exception:  # pragma: no cover
            pass

    @classmethod
    def _reset(cls) -> None:

        if cls._controller is not None:
            cls._controller.reset()

    # =======================================================
    # Mode Badge / Switch
    # =======================================================

    @classmethod
    def _mode_badge(cls, mode: str) -> None:

        label = (
            "🛠️ Training Mode (Administrator)"
            if mode == "administrator"
            else "📋 Assessment Mode (Lecturer/Advisor)"
        )

        st.caption(label)

        if st.button(
            "🔀 Switch Mode",
            use_container_width=True,
        ):

            cls._switch_mode()

            st.rerun()

    # =======================================================
    # Header
    # =======================================================

    @classmethod
    def _header(cls) -> None:

        st.title(cls.APP_TITLE)

        st.caption(cls.APP_VERSION)

    # =======================================================
    # Navigation
    # =======================================================

    @classmethod
    def _navigation(
        cls,
        current_page: DashboardPage,
        mode: str = "administrator",
    ) -> DashboardPage:

        pages = (
            LECTURER_PAGES
            if mode == "lecturer"
            else ADMINISTRATOR_PAGES
        )

        labels = [page.value for page in pages]

        try:

            index = pages.index(current_page)

        except ValueError:

            index = 0

        selected = st.radio(

            "Navigation",

            labels,

            index=index

        )

        return pages[labels.index(selected)]
    # =======================================================
    # Workflow Progress
    # =======================================================

    @classmethod
    def _workflow(cls, mode: str = "administrator") -> None:
        """
        Display the user's progress through the application.

        The workflow is based on the current state of the
        dashboard and updates automatically as the user uploads
        data, trains models, generates predictions, and creates
        reports (Administrator), or loads a saved model and
        assesses students (Lecturer).
        """

        st.subheader("Workflow Progress")

        if mode == "lecturer":

            cls._workflow_item(
                label="Load Saved Model",
                completed=cls._has_saved_model()
            )

            cls._workflow_item(
                label="Assess Students",
                completed=bool(
                    cls._assessment_history()
                )
            )

            return

        dataset_ready = cls._has_dataset()

        training_ready = cls._has_trained_models()

        prediction_ready = (
            cls._is_ready_for_prediction()
        )

        interpretation_ready = (
            cls._is_ready_for_interpretation()
        )

        report_ready = (
            cls._is_ready_for_reports()
        )

        cls._workflow_item(
            label="Upload Dataset",
            completed=dataset_ready
        )

        cls._workflow_item(
            label="Train Models",
            completed=training_ready
        )

        cls._workflow_item(
            label="Predict Performance",
            completed=prediction_ready
        )

        cls._workflow_item(
            label="Interpret Results",
            completed=interpretation_ready
        )

        cls._workflow_item(
            label="Generate Report",
            completed=report_ready
        )

    # -------------------------------------------------------

    @staticmethod
    def _workflow_item(
        label: str,
        completed: bool
    ) -> None:
        """
        Display one workflow item.

        Parameters
        ----------
        label:
            Name of the workflow step.

        completed:
            True when the step has been completed.
        """

        icon = "✅" if completed else "⬜"

        st.write(f"{icon} {label}")

    # =======================================================
    # System Status
    # =======================================================

    @classmethod
    def _system_status(cls, mode: str = "administrator") -> None:
        """
        Display the current state of important application
        components.
        """

        st.subheader("System Status")

        if mode == "lecturer":

            cls._status_item(
                label="Saved Model",
                active=cls._has_saved_model(),
                active_text="Loaded",
                inactive_text="Not Loaded"
            )

            history_count = len(
                cls._assessment_history()
            )

            cls._status_item(
                label="Assessments Logged",
                active=history_count > 0,
                active_text=f"{history_count}",
                inactive_text="None yet"
            )

            return

        cls._status_item(
            label="Dataset",
            active=cls._has_dataset(),
            active_text="Loaded",
            inactive_text="Not Loaded"
        )

        cls._status_item(
            label="Models",
            active=cls._has_trained_models(),
            active_text="Trained",
            inactive_text="Not Trained"
        )

        cls._status_item(
            label="Prediction",
            active=cls._is_ready_for_prediction(),
            active_text="Ready",
            inactive_text="Unavailable"
        )

        cls._status_item(
            label="Interpretation",
            active=(
                cls._is_ready_for_interpretation()
            ),
            active_text="Generated",
            inactive_text="Not Generated"
        )

        cls._status_item(
            label="Report",
            active=cls._is_ready_for_reports(),
            active_text="Ready",
            inactive_text="Not Ready"
        )

    # -------------------------------------------------------

    @staticmethod
    def _status_item(
        label: str,
        active: bool,
        active_text: str,
        inactive_text: str
    ) -> None:
        """
        Display one system status item.

        Parameters
        ----------
        label:
            Name of the system component.

        active:
            Current status of the component.

        active_text:
            Text displayed when active.

        inactive_text:
            Text displayed when inactive.
        """

        if active:

            st.markdown(
                f"**{label}:** 🟢 {active_text}"
            )

        else:

            st.markdown(
                f"**{label}:** ⚪ {inactive_text}"
            )

    # =======================================================
    # Quick Actions
    # =======================================================

    @classmethod
    def render_quick_actions(cls) -> None:
        """
        Display optional quick-action buttons.

        This method is separate from render() because quick
        actions may be placed only on selected pages.
        """

        st.subheader("Quick Actions")

        if st.button(
            "🔄 Reset Application",
            use_container_width=True,
            help=(
                "Clear the current dataset, models, predictions, "
                "reports, and dashboard progress."
            )
        ):

            cls._reset()

            st.rerun()

    # =======================================================
    # Project Information
    # =======================================================

    @classmethod
    def _deployment_status(cls) -> None:
        """
        What this deployment is actually running.

        Four facts decide whether a number on screen may be trusted:
        which model is loaded, how it was validated, whether history is
        being stored, and whether Training Mode is protected by a real
        password. They are one expander away instead of three pages.
        """

        try:

            from utils.auth import credential_status

            status = credential_status()

        except Exception:  # pragma: no cover - defensive

            status = {"kind": "unknown", "configured": False}

        with st.expander("🔍 Deployment status", expanded=False):

            version = PROJECT_VERSION

            st.caption(f"**Version** {version}")

            if cls._controller is not None:

                role = cls._controller.model_bundle_role

                strategy = cls._controller.model_validation_strategy

                persistent = cls._controller.history_is_persistent

            else:  # pragma: no cover - defensive

                role, strategy, persistent = (
                    "not_loaded",
                    "unknown",
                    False,
                )

            icons = {
                "production": "✅",
                "rule_reconstruction_baseline": "⚠️",
                "not_loaded": "⬜",
                "unknown": "❔",
            }

            st.caption(
                f"{icons.get(role, '❔')} **Model role** `{role}`"
            )

            if role == "not_loaded":

                st.caption(
                    "No model is in memory yet. Enter Assessment Mode "
                    "(or train one) and this panel describes what is "
                    "actually loaded."
                )

            if role == "rule_reconstruction_baseline":

                st.caption(
                    "Scores from this bundle measure rule reconstruction, "
                    "not prediction."
                )

            strategy_icons = {
                "temporal": "✅",
                "random": "⚠️",
                "unknown": "❔",
            }

            st.caption(
                f"{strategy_icons.get(strategy, '❔')} **Validation** "
                f"`{strategy}`"
            )

            if strategy == "random":

                st.caption(
                    "Random split of one cohort — it measures fit, not "
                    "next-semester transfer."
                )

            st.caption(
                f"{'✅' if persistent else '⚠️'} **History** "
                + (
                    "stored (SQLite)"
                    if persistent
                    else "session-only — nothing survives a restart"
                )
            )

            kind = status.get("kind", "unknown")

            credential_icons = {
                "hash": "✅",
                "plaintext": "⚠️",
                "default": "❌",
            }

            st.caption(
                f"{credential_icons.get(kind, '❔')} **Training Mode "
                f"password** `{kind}`"
            )

            if kind == "default":

                st.caption(
                    "The development password is active. Do not share "
                    "this URL."
                )

    @classmethod
    def _project_info(cls) -> None:
        """
        Display a compact project information section.
        """

        with st.expander(
            "ℹ️ About This System",
            expanded=False
        ):

            st.write(
                "The Student Performance Assessment System "
                "uses machine learning to classify student "
                "academic performance as Low, Average, or High."
            )

            st.caption(
                "Features include dataset validation, target "
                "definition from exam performance, model "
                "comparison, student prediction, explainable AI, "
                "recommendations, and report generation."
            )

            st.caption(
                "Student Performance Assessment System "
                f"• v{PROJECT_VERSION}"
            )

    # =======================================================
    # Notifications
    # =======================================================

    @classmethod
    def render_notifications(cls) -> None:
        """
        Display queued application notifications.

        Notifications are cleared after they are displayed so
        that old messages do not repeatedly appear after every
        Streamlit rerun.
        """

        notifications = (
            ApplicationState.get_notifications()
        )

        if not notifications:

            return

        for notification in notifications:

            level = notification.level.lower()

            message = notification.message

            if level == "success":

                st.success(message)

            elif level == "warning":

                st.warning(message)

            elif level == "error":

                st.error(message)

            else:

                st.info(message)

        ApplicationState.clear_notifications()