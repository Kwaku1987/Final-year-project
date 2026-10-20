"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/prediction_page.py

Purpose:
    Collect a single student's details and predict their
    performance tier. For predicting many students at once, see
    the dedicated 👥 Batch Prediction page.
====================================================================
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.dashboard_controller import DashboardController
from backend.config import CPI_CONFIG


def render_prediction_page(controller: DashboardController) -> None:

    st.title("🔮 Prediction")

    if not controller.is_ready_for_prediction:

        st.warning(
            "Please train and evaluate models on the "
            "**⚙ Model Training** page first."
        )

        return

    st.caption(
        f"Best model: **{controller.best_model_name}**"
    )

    _render_single_prediction(controller)


# ================================================================
# SINGLE STUDENT
# ================================================================

def _render_single_prediction(
    controller: DashboardController,
) -> None:

    available_models = controller.get_available_models()

    model_options = ["🏆 Best Model (auto)"] + available_models

    with st.form("prediction_form"):

        selected_option = st.selectbox(
            "Predict using",
            model_options,
            help=(
                "Use the auto-selected best model, or choose a "
                "specific model to compare predictions."
            ),
        )

        st.subheader("Student Details")

        col1, col2 = st.columns(2)

        with col1:

            attendance_rate = st.slider(
                "Attendance Rate (%)", 0, 100, 85,
            )

            study_hours = st.slider(
                "Daily Study Hours", 0.0,
                float(CPI_CONFIG["study_hours_max"]) * 2, 4.0,
            )

            sleep_hours = st.slider(
                "Daily Sleep Hours", 0.0,
                float(CPI_CONFIG["sleep_hours_max"]) * 2, 7.0,
            )

            social_media_hours = st.slider(
                "Daily Social Media Hours", 0.0,
                float(CPI_CONFIG["social_media_hours_max"]) * 2, 2.0,
            )

        with col2:

            assignment_score = st.slider(
                "Assignment Score (%)", 0, 100, 75,
            )

            test_score = st.slider(
                "Test Score (%)", 0, 100, 75,
            )

            exam_score = st.slider(
                "Exam Score (%)", 0, 100, 75,
            )

        submitted = st.form_submit_button(
            "🔮 Predict Performance", type="primary",
        )

    if not submitted:

        return

    model_name = (
        None
        if selected_option == "🏆 Best Model (auto)"
        else selected_option
    )

    student_data = {
        "attendance_rate": attendance_rate,
        "study_hours": study_hours,
        "sleep_hours": sleep_hours,
        "social_media_hours": social_media_hours,
        "assignment_score": assignment_score,
        "test_score": test_score,
        "exam_score": exam_score,
    }

    with st.spinner("Predicting..."):

        try:

            prediction = controller.predict_student(
                student_data,
                model_name=model_name,
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")

            return

    st.divider()

    label_colors = {
        "High": "🟢",
        "Average": "🟡",
        "Low": "🔴",
    }

    icon = label_colors.get(prediction.predicted_label, "⚪")

    st.subheader(
        f"{icon} Predicted Performance: "
        f"**{prediction.predicted_label}**"
    )

    st.caption(f"Model used: {prediction.model_name}")

    st.metric(
        "Confidence",
        f"{prediction.confidence * 100:.1f}%",
    )

    if getattr(prediction, "summary", None):

        st.write(prediction.summary)

    if prediction.probabilities:

        st.write("**Class Probabilities**")

        probabilities_df = pd.DataFrame(
            {
                "Performance Level": list(
                    prediction.probabilities.keys()
                ),
                "Probability": list(
                    prediction.probabilities.values()
                ),
            }
        ).set_index("Performance Level")

        st.bar_chart(probabilities_df)

    if getattr(prediction, "top_features", None):

        st.write("**Why this prediction?**")

        st.caption(
            "The factors that most influenced this "
            "prediction, and this student's values for them."
        )

        for feature in prediction.top_features:

            st.write(
                f"- **{feature['display_name']}**: "
                f"{feature['value']:.1f} "
                f"(influence: {feature['importance']:.1%})"
            )

    st.divider()

    st.subheader("🚨 Intervention Priority")

    risk = getattr(prediction, "intervention", None)

    if risk is None:

        st.caption("Risk assessment unavailable for this prediction.")

    else:

        risk_col1, risk_col2 = st.columns([1, 2])

        with risk_col1:

            st.metric("Risk Score", risk.score)

        with risk_col2:

            priority_display = {
                "CRITICAL": st.error,
                "HIGH": st.warning,
                "MEDIUM": st.warning,
                "LOW": st.success,
            }

            display_fn = priority_display.get(
                risk.priority, st.info
            )

            display_fn(f"**{risk.priority}** — {risk.level}")

        st.write(risk.message)

        if risk.reasons:

            st.write("**Contributing Factors**")

            for reason in risk.reasons:

                st.write(f"- {reason}")

    if getattr(prediction, "recommendations", None):

        st.write("**Recommendations**")

        for recommendation in prediction.recommendations:

            st.write(f"- {recommendation}")

    st.divider()

    try:

        pdf_bytes = controller.export_student_pdf(
            student_data, prediction,
        )

        st.download_button(
            "⬇ Download PDF Report",
            data=pdf_bytes,
            file_name="student_prediction_report.pdf",
            mime="application/pdf",
        )

    except Exception as error:

        st.caption(f"PDF report unavailable: {error}")
