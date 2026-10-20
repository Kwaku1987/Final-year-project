"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/mode_selection_page.py

Purpose:
    First screen shown each session: choose Administrator
    (Training Mode) or Lecturer/Advisor (Assessment Mode).
====================================================================
"""

from __future__ import annotations

import streamlit as st

from dashboard.dashboard_controller import DashboardController


def render_mode_selection_page(
    controller: DashboardController,
) -> None:

    st.title("🎓 Student Performance Assessment System")

    st.caption(
        "Predict student academic performance (Low / Average / "
        "High) using attendance, study habits, and assessment "
        "scores."
    )

    st.divider()

    st.subheader("Choose how you'd like to use this system")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🛠️ Training Mode")

        st.caption("For Administrators")

        st.write(
            "Upload a dataset, generate the Composite "
            "Performance Index, train and compare models, and "
            "save the best one for lecturers to use."
        )

        if st.button(
            "Enter Training Mode",
            type="primary",
            use_container_width=True,
        ):

            controller.set_mode("administrator")

            st.rerun()

    with col2:

        st.markdown("### 📋 Assessment Mode")

        st.caption("For Lecturers / Academic Advisors")

        st.write(
            "Load the saved model and assess individual "
            "students by entering their details -- no training "
            "required."
        )

        model_exists = controller.framework.has_saved_model(
            controller.DEFAULT_MODEL_PATH
        )

        if not model_exists:

            st.warning(
                "No saved model found yet. An Administrator "
                "needs to train and save one first."
            )

        if st.button(
            "Enter Assessment Mode",
            type="primary",
            use_container_width=True,
        ):

            controller.set_mode("lecturer")

            st.rerun()
