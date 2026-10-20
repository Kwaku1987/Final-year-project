"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    app.py

Purpose:
    Streamlit application entry point. Configures logging, sets
    up the page, and routes between dashboard pages based on the
    sidebar selection.

Notes
-----
Streamlit reruns this entire script on every user interaction
(every button click, slider change, etc.). Without caching, that
would mean calling DashboardController() -- which builds a brand
new StudentPerformanceFramework via create_framework() -- on every
single interaction, silently wiping out any trained models,
uploaded dataset, or predictions each time. The controller is
cached in st.session_state so exactly one instance persists for
the duration of the browser session.

Run with:
    streamlit run app.py
====================================================================
"""

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from dashboard.dashboard_controller import DashboardController

from components.sidebar import (
    Sidebar,
    DashboardPage,
)

from dashboard.pages.home_page import render_home_page
from dashboard.pages.dataset_page import render_dataset_page
from dashboard.pages.training_page import render_training_page
from dashboard.pages.prediction_page import render_prediction_page
from dashboard.pages.assessment_page import render_assessment_page
from dashboard.pages.batch_prediction_page import (
    render_batch_prediction_page,
)
from dashboard.pages.interpretation_page import (
    render_interpretation_page,
)
from dashboard.pages.reports_page import render_reports_page
from dashboard.pages.mode_selection_page import (
    render_mode_selection_page,
)

from utils.styles import load_theme


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Assessment System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(load_theme(), unsafe_allow_html=True)


# ============================================================
# CONTROLLER (cached across reruns)
# ============================================================

def get_controller() -> DashboardController:
    """
    Return the single DashboardController for this browser
    session, creating it once on first run.
    """

    if "controller" not in st.session_state:

        logger.info(
            "No controller in session_state -- creating one."
        )

        st.session_state.controller = DashboardController()

    return st.session_state.controller


# ============================================================
# PAGE ROUTER
# ============================================================

_PAGE_RENDERERS = {

    DashboardPage.HOME: render_home_page,

    DashboardPage.DATASET: render_dataset_page,

    DashboardPage.TRAINING: render_training_page,

    DashboardPage.PREDICTION: render_prediction_page,

    DashboardPage.ASSESSMENT: render_assessment_page,

    DashboardPage.BATCH: render_batch_prediction_page,

    DashboardPage.INTERPRETATION: render_interpretation_page,

    DashboardPage.REPORTS: render_reports_page,

}


def main() -> None:

    controller = get_controller()

    # Mode-selection gate: nothing else renders until the user
    # has chosen Administrator (Training Mode) or Lecturer
    # (Assessment Mode).
    if controller.application_mode is None:

        render_mode_selection_page(controller)

        return

    if "current_page" not in st.session_state:

        st.session_state.current_page = DashboardPage.HOME

    selected_page = Sidebar.render(
        st.session_state.current_page,
        mode=controller.application_mode,
    )

    st.session_state.current_page = selected_page

    Sidebar.render_notifications()

    renderer = _PAGE_RENDERERS.get(
        selected_page,
        render_home_page,
    )

    renderer(controller)


if __name__ == "__main__":

    main()
