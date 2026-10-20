"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/reports_page.py

Purpose:
    Generate and display a consolidated project report.
====================================================================
"""

from __future__ import annotations

import json

import streamlit as st

from dashboard.dashboard_controller import DashboardController


def render_reports_page(controller: DashboardController) -> None:

    st.title("📄 Reports")

    if not controller.is_ready_for_prediction:

        st.warning(
            "Please train and evaluate models on the "
            "**⚙ Model Training** page first."
        )

        return

    if st.button("📄 Generate Report", type="primary"):

        with st.spinner("Generating report..."):

            try:

                controller.generate_report()

            except Exception as error:

                st.error(f"Report generation failed: {error}")

                return

        st.rerun()

    report = controller.get_report()

    if report is None:

        st.info("Click **Generate Report** to build a summary.")

        return

    st.divider()

    st.subheader("Project Summary")

    st.write(f"**Project:** {report.get('project', 'N/A')}")

    st.subheader(f"🏆 Best Model: {report.get('best_model', 'N/A')}")

    comparison = report.get("comparison_results")

    if comparison:

        rows = []

        for model_name, evaluation in comparison.items():

            metrics = evaluation.get("metrics", {})

            rows.append({
                "Model": model_name,
                "Accuracy": round(metrics.get("accuracy", 0), 4),
                "F1 Score": round(
                    metrics.get("f1_score", metrics.get("f1", 0)), 4
                ),
            })

        if rows:

            import pandas as pd

            st.dataframe(
                pd.DataFrame(rows), use_container_width=True,
                hide_index=True,
            )

    interpretation = report.get("interpretation")

    if interpretation:

        st.subheader("Executive Summary")

        st.write(
            interpretation.get("executive_summary", "")
            if isinstance(interpretation, dict)
            else getattr(interpretation, "executive_summary", "")
        )

    st.divider()

    st.subheader("⬇ Download")

    try:

        report_json = json.dumps(report, indent=2, default=str)

        st.download_button(
            "Download Report (JSON)",
            data=report_json,
            file_name="student_performance_report.json",
            mime="application/json",
        )

    except Exception:

        st.caption(
            "Report contains non-serializable objects and "
            "can't be downloaded as JSON directly."
        )
