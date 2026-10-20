"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/interpretation_page.py

Purpose:
    Show feature importance and explainability insights for the
    best model.
====================================================================
"""

from __future__ import annotations

import streamlit as st

from dashboard.dashboard_controller import DashboardController


def render_interpretation_page(controller: DashboardController) -> None:

    st.title("🧠 Explainable AI")

    if not controller.is_ready_for_prediction:

        st.warning(
            "Please train and evaluate models on the "
            "**⚙ Model Training** page first."
        )

        return

    if st.button("🧠 Generate Interpretation", type="primary"):

        with st.spinner("Interpreting the best model..."):

            try:

                controller.interpret_model()

            except Exception as error:

                st.error(f"Interpretation failed: {error}")

                return

        st.rerun()

    report = controller.state.get_interpretation_report()

    if report is None:

        st.info(
            "Click **Generate Interpretation** to explain the "
            f"best model ({controller.best_model_name})."
        )

        return

    st.divider()

    st.subheader(f"Model: {report.model_name} ({report.model_type})")

    st.write(report.executive_summary)

    st.divider()

    st.subheader("📊 Feature Importance")

    if report.feature_importance is not None:

        importance = (
            report.feature_importance[["feature", "importance"]]
            .set_index("feature")
        )

        st.bar_chart(importance)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("💡 Key Insights")

        for insight in report.key_insights:

            st.write(f"- {insight}")

    with col2:

        st.subheader("✅ Recommendations")

        for recommendation in report.recommendations:

            st.write(f"- {recommendation}")

    if report.top_features:

        st.divider()

        st.subheader("🔝 Top Contributing Features")

        for insight in report.top_features:

            with st.expander(
                f"#{insight.rank}  {insight.feature}  "
                f"({insight.direction})"
            ):

                st.write(f"**Importance:** {insight.importance:.4f}")

                st.write(insight.interpretation)
