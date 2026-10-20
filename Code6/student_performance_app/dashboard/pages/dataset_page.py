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
EDAAnalyzer's plotting methods call plt.show(), which doesn't
render inside Streamlit (Streamlit needs st.pyplot()/native
chart calls instead). Rather than adapt EDAAnalyzer for this page,
visuals here are built directly with Streamlit-native charting
(st.bar_chart, st.dataframe) which renders correctly without
changes. EDAAnalyzer remains well-suited to notebook/console use.
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

    st.divider()

    st.subheader("Next Step")

    st.write(
        "Once your dataset looks correct, head to "
        "**⚙ Model Training** to generate the Composite "
        "Performance Index and train models."
    )
