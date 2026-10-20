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
    5.0
===============================================================
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import streamlit as st

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

    APP_VERSION = "Version 5.0"

    # --------------------------------------------------------

    @classmethod
    def render(
        cls,
        current_page: DashboardPage,
        mode: str = "administrator",
    ) -> DashboardPage:
        """
        Draw the complete sidebar.

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

        with st.sidebar:

            cls._header()

            cls._mode_badge(mode)

            page = cls._navigation(current_page, mode)

            st.divider()

            cls._workflow(mode)

            st.divider()

            cls._system_status(mode)

            st.divider()

            cls._project_info()

        return page

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

            ApplicationState.clear_application_mode()

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
                completed=ApplicationState.has_saved_model()
            )

            cls._workflow_item(
                label="Assess Students",
                completed=bool(
                    ApplicationState.get_assessment_history()
                )
            )

            return

        dataset_ready = ApplicationState.has_dataset()

        training_ready = ApplicationState.has_trained_models()

        prediction_ready = (
            ApplicationState.is_ready_for_prediction()
        )

        interpretation_ready = (
            ApplicationState.is_ready_for_interpretation()
        )

        report_ready = (
            ApplicationState.is_ready_for_reports()
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
                active=ApplicationState.has_saved_model(),
                active_text="Loaded",
                inactive_text="Not Loaded"
            )

            history_count = len(
                ApplicationState.get_assessment_history()
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
            active=ApplicationState.has_dataset(),
            active_text="Loaded",
            inactive_text="Not Loaded"
        )

        cls._status_item(
            label="Models",
            active=ApplicationState.has_trained_models(),
            active_text="Trained",
            inactive_text="Not Trained"
        )

        cls._status_item(
            label="Prediction",
            active=ApplicationState.is_ready_for_prediction(),
            active_text="Ready",
            inactive_text="Unavailable"
        )

        cls._status_item(
            label="Interpretation",
            active=(
                ApplicationState.is_ready_for_interpretation()
            ),
            active_text="Generated",
            inactive_text="Not Generated"
        )

        cls._status_item(
            label="Report",
            active=ApplicationState.is_ready_for_reports(),
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

            ApplicationState.reset_application()

            st.rerun()

    # =======================================================
    # Project Information
    # =======================================================

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
                "Features include dataset validation, CPI-based "
                "target generation, model comparison, student "
                "prediction, explainable AI, recommendations, "
                "and report generation."
            )

            st.caption(
                "Student Performance Assessment System "
                "• Version 5.0"
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