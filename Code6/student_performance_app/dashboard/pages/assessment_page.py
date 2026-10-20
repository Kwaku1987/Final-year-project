"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/assessment_page.py

Purpose:
    Lecturer / Assessment Mode: load the saved model, assess a
    student by name (and optional ID), and review assessment
    history.
====================================================================
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.dashboard_controller import DashboardController
from backend.config import CPI_CONFIG


def render_assessment_page(controller: DashboardController) -> None:

    st.title("📋 Assess Student")

    if not controller.has_saved_model:

        st.warning(
            "No saved model is available yet. Ask an "
            "Administrator to train and save one in Training "
            "Mode first."
        )

        return

    if not controller.is_ready_for_prediction:

        st.info(
            "Load the saved model to begin assessing students."
        )

        if st.button("📥 Load Saved Model", type="primary"):

            with st.spinner("Loading saved model..."):

                try:

                    controller.enter_assessment_mode()

                except Exception as error:

                    st.error(f"Could not load saved model: {error}")

                    return

            st.rerun()

        return

    st.caption(
        f"Best model: **{controller.best_model_name}**"
    )

    model_info = controller.get_model_information()

    if model_info.get("training_date"):

        with st.expander("ℹ️ Model Information"):

            info_col1, info_col2 = st.columns(2)

            with info_col1:

                st.write(
                    f"**Model:** {model_info['model_name']}"
                )

                st.write(
                    f"**Trained on:** {model_info['training_date']}"
                )

                if model_info.get("training_samples"):

                    st.write(
                        "**Training samples:** "
                        f"{model_info['training_samples']}"
                    )

            with info_col2:

                if model_info.get("accuracy") is not None:

                    st.write(
                        f"**Accuracy:** {model_info['accuracy']:.1%}"
                    )

                if model_info.get("f1_score") is not None:

                    st.write(
                        f"**F1 Score:** {model_info['f1_score']:.1%}"
                    )

    available_models = controller.get_available_models()

    model_options = ["🏆 Best Model (auto)"] + available_models

    with st.form("assessment_form"):

        st.subheader("Student Identity")

        id_col1, id_col2 = st.columns(2)

        with id_col1:

            student_name = st.text_input("Student Name")

        with id_col2:

            student_id = st.text_input("Student ID (Optional)")

        selected_option = st.selectbox(
            "Assess using",
            model_options,
            help=(
                "Use the auto-selected best model, or choose a "
                "specific model to compare assessments."
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
            "📋 Assess Student", type="primary",
        )

    if submitted:

        if not student_name.strip():

            st.error("Student Name is required.")

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

        with st.spinner("Assessing..."):

            try:

                prediction, record = controller.assess_student(
                    student_name=student_name,
                    student_id=student_id,
                    student_data=student_data,
                    model_name=model_name,
                )

            except Exception as error:

                st.error(f"Assessment failed: {error}")

                return

        st.divider()

        label_colors = {
            "High": "🟢",
            "Average": "🟡",
            "Low": "🔴",
        }

        icon = label_colors.get(prediction.predicted_label, "⚪")

        id_suffix = f" ({record.student_id})" if record.student_id else ""

        st.subheader(
            f"{icon} {record.student_name}{id_suffix}: "
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
                student_data,
                prediction,
                student_name=record.student_name,
                student_id=record.student_id,
            )

            st.download_button(
                "⬇ Download PDF Report",
                data=pdf_bytes,
                file_name=(
                    f"{record.student_name.replace(' ', '_')}"
                    "_assessment_report.pdf"
                ),
                mime="application/pdf",
            )

        except Exception as error:

            st.caption(f"PDF report unavailable: {error}")

    st.divider()

    st.subheader("📜 Assessment History")

    history = controller.get_assessment_history()

    if not history:

        st.caption("No assessments logged yet this session.")

    else:

        history_df = pd.DataFrame([
            {
                "Student Name": record.student_name,
                "Student ID": record.student_id or "—",
                "Prediction": record.prediction,
                "Confidence": f"{record.confidence * 100:.1f}%",
                "Priority": record.risk_priority or "—",
                "Date": record.assessment_date.strftime(
                    "%Y-%m-%d %H:%M"
                ),
            }
            for record in reversed(history)
        ])

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )
