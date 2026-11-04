"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/dataset_page.py

Purpose:
    Upload, validate, and preview the student dataset.

Notes
-----
v3.0 wires EDAAnalyzer in. Its matplotlib methods call plt.show(),
which does not render inside Streamlit, but its *analysis*
methods (generate_report(), correlation_analysis(),
multicollinearity_check(), outlier_analysis(),
target_distribution()) return DataFrames -- and those are the
findings a reader of a final-year report actually needs. They are
shown in the "Exploratory Data Analysis" section below, so the
657-line module is no longer dead code.

Validation warnings raised by DatasetValidator are surfaced here
too, instead of being lost in the server console.
====================================================================
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.dashboard_controller import DashboardController
from backend.config import FEATURE_COLUMNS


def render_dataset_page(controller: DashboardController) -> None:

    st.title("📂 Dataset")

    st.caption(
        "Upload a student performance dataset (CSV or Excel). "
        "Required columns: "
        + ", ".join(FEATURE_COLUMNS)
    )

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):

                dataframe = pd.read_csv(uploaded_file)

            else:

                dataframe = pd.read_excel(uploaded_file)

        except Exception as error:

            st.error(f"Could not read the file: {error}")

            return

        st.write("**Preview (first 5 rows)**")

        st.dataframe(dataframe.head(), use_container_width=True)

        if st.button("✅ Validate & Load Dataset", type="primary"):

            with st.spinner("Validating dataset..."):

                try:

                    controller.load_dataset(dataframe)

                except Exception as error:

                    st.error(f"Validation failed: {error}")

                    return

            st.success("Dataset validated and loaded successfully!")

            st.rerun()

    st.divider()

    if not controller.has_dataset:

        st.info("No dataset loaded yet.")

        return

    summary = controller.dataset_summary()

    warnings = controller.validation_warnings

    if warnings:

        with st.expander(
            f"⚠️ {len(warnings)} data note(s) from validation",
            expanded=True,
        ):

            for warning in warnings:

                st.write(f"- {warning}")

    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    col1.metric("Rows", summary["rows"])

    col2.metric("Columns", summary["columns"])

    dataframe = controller.state.get_dataset()

    with st.expander("📊 Feature Distributions", expanded=True):

        numeric_columns = [
            column
            for column in FEATURE_COLUMNS
            if column in dataframe.columns
        ]

        selected_column = st.selectbox(
            "Choose a feature to visualize",
            numeric_columns,
        )

        if selected_column:

            st.bar_chart(
                dataframe[selected_column]
                .value_counts(bins=10)
                .sort_index()
            )

    with st.expander("🔍 Missing Values"):

        st.dataframe(
            summary["missing"].to_frame("Missing Values"),
            use_container_width=True,
        )

    with st.expander("📋 Full Dataset"):

        st.dataframe(dataframe, use_container_width=True)

    _render_eda(controller)

    st.divider()

    st.subheader("Next Step")

    st.write(
        "Once your dataset looks correct, head to "
        "**⚙ Model Training** to build the target labels and "
        "train models."
    )


# ================================================================
# EXPLORATORY DATA ANALYSIS
# ================================================================

def _render_eda(controller: DashboardController) -> None:
    """
    Surface backend/eda/eda_analyzer.py findings in the dashboard.
    """

    with st.expander("📈 Exploratory Data Analysis", expanded=False):

        if not st.button("Run EDA", key="run_eda_button"):

            st.caption(
                "Correlation, multicollinearity, outliers and "
                "target distribution, computed by "
                "`backend/eda/eda_analyzer.py`."
            )

            return

        with st.spinner("Running exploratory analysis..."):

            report = controller.run_eda()

        if not report or report.get("error"):

            st.warning(
                "EDA could not run: "
                f"{(report or {}).get('error', 'unknown error')}"
            )

            return

        summary = report.get("summary_statistics")

        if summary is not None:

            st.write("**Summary statistics**")

            st.dataframe(
                summary,
                use_container_width=True,
            )

        target = report.get("target_distribution")

        if target is not None:

            st.write("**Target distribution**")

            st.dataframe(
                target,
                use_container_width=True,
            )

        correlation = report.get("correlation_matrix")

        if correlation is not None:

            st.write("**Correlation matrix**")

            st.dataframe(
                correlation.round(3),
                use_container_width=True,
            )

        multicollineaity = report.get("multicollinearity")

        if multicollineaity is not None:

            st.write("**Multicollinearity check**")

            st.dataframe(
                multicollineaity,
                use_container_width=True,
            )

        outliers = report.get("outliers")

        if outliers is not None:

            st.write("**Outlier analysis**")

            st.dataframe(
                outliers,
                use_container_width=True,
            )
