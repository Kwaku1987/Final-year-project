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
from backend.config import CPI_CONFIG, FEATURE_COLUMNS


# ================================================================
# INPUT SPECS
# ================================================================
# label, minimum, maximum, default -- per feature column.
# ================================================================

INPUT_SPECS = {
    "attendance_rate": ("Attendance Rate (%)", 0, 100, 85),
    "study_hours": (
        "Daily Study Hours",
        0.0,
        float(CPI_CONFIG["study_hours_max"]) * 2,
        4.0,
    ),
    "sleep_hours": (
        "Daily Sleep Hours",
        0.0,
        float(CPI_CONFIG["sleep_hours_max"]) * 2,
        7.0,
    ),
    "social_media_hours": (
        "Daily Social Media Hours",
        0.0,
        float(CPI_CONFIG["social_media_hours_max"]) * 2,
        2.0,
    ),
    "assignment_score": ("Assignment Score (%)", 0, 100, 75),
    "test_score": ("Test Score (%)", 0, 100, 75),
    "exam_score": ("Exam Score (%)", 0, 100, 75),
}


# ================================================================
# EXPLANATION RENDERER
# ================================================================

def _render_explanation(prediction) -> None:
    """
    Show why this student got this prediction.

    Prefers SHAP contributions (instance-level, signed) and
    falls back to global feature importance paired with the
    student's values when SHAP is unavailable.
    """

    instance_explanation = getattr(
        prediction, "instance_explanation", None
    )

    if instance_explanation:

        st.write("**Why this prediction?**")

        st.caption(
            "SHAP contributions for this student: how much each "
            "factor pushed the prediction towards (positive) or "
            "away from (negative) the predicted label."
        )

        explanation_df = pd.DataFrame(instance_explanation)[
            ["display_name", "value", "contribution", "direction"]
        ].rename(
            columns={
                "display_name": "Factor",
                "value": "Student's value",
                "contribution": "Contribution",
                "direction": "Effect",
            }
        )

        st.dataframe(
            explanation_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            explanation_df[["Factor", "Contribution"]]
            .set_index("Factor")
        )

        return

    if getattr(prediction, "top_features", None):

        st.write("**Why this prediction?**")

        st.caption(
            "SHAP is not installed, so this shows the model's "
            "globally most important factors alongside this "
            "student's values (not an instance-level "
            "explanation). Install `shap` for per-student "
            "attribution."
        )

        for feature in prediction.top_features:

            st.write(
                f"- **{feature['display_name']}**: "
                f"{feature['value']:.1f} "
                f"(global influence: {feature['importance']:.1%})"
            )



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

        # v3.0: only ask for the columns the model was actually
        # trained on. Under the default "exam_band" definition the
        # examination score IS the label, so it is deliberately
        # not an input -- asking for it would make the prediction
        # circular all over again.
        features = controller.required_features

        excluded = [
            column
            for column in FEATURE_COLUMNS
            if column not in features
        ]

        if excluded:

            st.caption(
                "Not requested because the model predicts it: "
                + ", ".join(excluded)
                + "."
            )

        columns = st.columns(2)

        student_data = {}

        for index, feature in enumerate(features):

            label, minimum, maximum, default = (
                INPUT_SPECS.get(
                    feature,
                    (feature.replace("_", " ").title(), 0, 100, 0),
                )
            )

            with columns[index % 2]:

                student_data[feature] = st.slider(
                    label,
                    minimum,
                    maximum,
                    default,
                    help=f"Column: `{feature}`",
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

    _render_explanation(prediction)

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
