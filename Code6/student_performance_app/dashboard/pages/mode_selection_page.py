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

from utils.auth import (
    attempts_remaining,
    is_admin_unlocked,
    is_using_default_password,
    verify_admin_password,
)


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
            "Upload a dataset, build the target labels, train "
            "and compare models, and save the best one for "
            "lecturers to use."
        )

        if is_using_default_password():

            st.caption(
                "🔐 Development password in use "
                "(`change-me-admin`). Set "
                "`SPAS_ADMIN_PASSWORD_HASH` before deploying."
            )

        password = st.text_input(
            "Administrator password",
            type="password",
            key="admin_password_input",
            help=(
                "Training Mode can retrain and delete the model "
                "lecturers depend on, so it is password-gated."
            ),
        )

        if attempts_remaining() <= 0:

            st.error(
                "Too many failed attempts in this session. "
                "Reload the page to try again."
            )

        elif st.button(
            "Enter Training Mode",
            type="primary",
            use_container_width=True,
        ):

            succeeded = verify_admin_password(password)

            controller.log_admin_unlock(succeeded)

            if succeeded:

                controller.set_mode("administrator")

                st.rerun()

            else:

                st.error(
                    "Incorrect password. "
                    f"{attempts_remaining()} attempt(s) left."
                )

    with col2:

        st.markdown("### 📋 Assessment Mode")

        st.caption("For Lecturers / Academic Advisors")

        st.write(
            "Load the saved model and assess individual "
            "students by entering their details -- no training "
            "required, and no access to the training data."
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
