"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/home_page.py

Purpose:
    Landing page: project overview and current workflow status,
    tailored to the active application mode.

Notes
-----
st.metric() truncates its value text (e.g. "⬜ Pending" became
"Pen...") when a column is too narrow to fit it -- five columns
across the page didn't leave enough room. Switched to icon-only
metric values, which always fit regardless of column width.
====================================================================
"""

from __future__ import annotations

import streamlit as st

from dashboard.dashboard_controller import DashboardController


def render_home_page(controller: DashboardController) -> None:

    if controller.is_lecturer:

        _render_lecturer_home(controller)

    else:

        _render_administrator_home(controller)


# ================================================================
# ADMINISTRATOR / TRAINING MODE
# ================================================================

def _render_administrator_home(
    controller: DashboardController,
) -> None:

    st.title("🎓 Student Performance Assessment System")

    st.caption(
        "Training Mode (Administrator) — predict student "
        "academic performance (Low / Average / High) using "
        "attendance, study habits, and assessment scores."
    )

    st.divider()

    status = controller.status()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Dataset",
        "✅" if status["dataset_validated"] else "⬜",
    )

    col2.metric(
        "Data Prepared",
        "✅" if status["data_prepared"] else "⬜",
    )

    col3.metric(
        "Models Trained",
        "✅" if status["models_trained"] else "⬜",
    )

    col4.metric(
        "Best Model",
        "✅" if status["best_model_selected"] else "⬜",
    )

    col5.metric(
        "Interpretation",
        "✅" if status["interpretation_ready"] else "⬜",
    )

    st.divider()

    if not status["dataset_validated"]:

        st.info(
            "👋 Get started by heading to the **📂 Dataset** page "
            "to upload a student performance dataset."
        )

    elif not status["models_trained"]:

        st.info(
            "Dataset is loaded. Head to **⚙ Model Training** to "
            "train and compare models."
        )

    elif not status["best_model_selected"]:

        st.info(
            "Models are trained. Evaluate them on the "
            "**⚙ Model Training** page to select the best one."
        )

    else:

        st.success(
            f"Best model selected: **{controller.best_model_name}**. "
            "Head to **🔮 Prediction** to predict a student's "
            "performance, or **🧠 Explainable AI** to see what "
            "drives the model's decisions."
        )

        if controller.has_saved_model:

            st.caption(
                "✅ This model is saved and available to "
                "Lecturers in Assessment Mode."
            )

        else:

            st.caption(
                "💡 Use the **Run Training Pipeline** button on "
                "the Training page to save this model for "
                "Lecturers to use in Assessment Mode."
            )

    st.divider()

    st.subheader("How it works")

    st.markdown(
        """
1. **📂 Dataset** — Upload a student dataset and validate it.
2. **⚙ Model Training** — Generate the Composite Performance
   Index (CPI), prepare features, train/compare models, and
   save the best one.
3. **🔮 Prediction** — Enter a student's details and predict
   their performance tier.
4. **🧠 Explainable AI** — See which factors most influence the
   model's predictions.
5. **📄 Reports** — Generate a consolidated summary report.
"""
    )


# ================================================================
# LECTURER / ASSESSMENT MODE
# ================================================================

def _render_lecturer_home(
    controller: DashboardController,
) -> None:

    st.title("🎓 Student Performance Assessment System")

    st.caption(
        "Assessment Mode (Lecturer/Advisor) — assess individual "
        "students using the model an Administrator has trained "
        "and saved."
    )

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric(
        "Saved Model",
        "✅" if controller.has_saved_model else "⬜",
    )

    history_count = len(controller.get_assessment_history())

    col2.metric(
        "Assessments Logged",
        str(history_count),
    )

    st.divider()

    if not controller.has_saved_model:

        st.warning(
            "No saved model is available yet. An Administrator "
            "needs to train and save one first (Training Mode)."
        )

    elif not controller.is_ready_for_prediction:

        st.info(
            "A saved model is available. Head to "
            "**📋 Assess Student** to load it and begin."
        )

    else:

        st.success(
            f"Model loaded: **{controller.best_model_name}**. "
            "Head to **📋 Assess Student** to assess a student, "
            "or **🧠 Explainable AI** to see what drives its "
            "predictions."
        )

    st.divider()

    st.subheader("How it works")

    st.markdown(
        """
1. **📋 Assess Student** — Load the saved model, enter a
   student's name, ID (optional), and details, and get a
   predicted performance tier.
2. **🧠 Explainable AI** — See which factors most influence the
   loaded model's predictions.

Every assessment is logged so you can review past results.
"""
    )
