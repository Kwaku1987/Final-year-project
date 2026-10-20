"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/batch_prediction_page.py

Purpose:
    Upload a class roster (CSV/Excel) and predict performance for
    every student at once. Available in both Administrator and
    Lecturer mode.
====================================================================
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.dashboard_controller import DashboardController
from backend.config import FEATURE_COLUMNS


def render_batch_prediction_page(
    controller: DashboardController,
) -> None:

    st.title("👥 Batch Prediction")

    if not controller.is_ready_for_prediction:

        st.warning(
            "No trained or loaded model is available yet. "
            "Train a model (Training Mode) or load the saved "
            "model (Assessment Mode) first."
        )

        return

    st.caption(
        f"Model: **{controller.best_model_name}**. Upload a "
        "roster with a column for each of: "
        + ", ".join(FEATURE_COLUMNS)
    )

    available_models = controller.get_available_models()

    model_options = ["🏆 Best Model (auto)"] + available_models

    selected_option = st.selectbox(
        "Predict using", model_options,
    )

    model_name = (
        None
        if selected_option == "🏆 Best Model (auto)"
        else selected_option
    )

    uploaded_file = st.file_uploader(
        "Upload class roster",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file is None:

        return

    try:

        if uploaded_file.name.endswith(".csv"):

            roster = pd.read_csv(uploaded_file)

        else:

            roster = pd.read_excel(uploaded_file)

    except Exception as error:

        st.error(f"Could not read the file: {error}")

        return

    st.write("**Preview**")

    st.dataframe(roster.head(), use_container_width=True)

    if st.button("👥 Predict for All Students", type="primary"):

        with st.spinner(
            f"Predicting for {len(roster)} student(s)..."
        ):

            try:

                results = controller.predict_batch(
                    roster, model_name=model_name,
                )

            except ValueError as error:

                st.error(str(error))

                return

            except Exception as error:

                st.error(f"Batch prediction failed: {error}")

                return

        st.session_state["batch_results"] = results

    results = st.session_state.get("batch_results")

    if results is None:

        return

    st.divider()

    st.subheader("Results")

    error_count = int((results["Predicted Performance"] == "ERROR").sum())

    if error_count:

        st.warning(
            f"{error_count} row(s) failed to predict -- check "
            "for missing or invalid values in those rows."
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High",
        int((results["Predicted Performance"] == "High").sum()),
    )

    col2.metric(
        "Average",
        int((results["Predicted Performance"] == "Average").sum()),
    )

    col3.metric(
        "Low",
        int((results["Predicted Performance"] == "Low").sum()),
    )

    st.dataframe(results, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("⬇ Download")

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        csv_bytes = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="batch_predictions.csv",
            mime="text/csv",
        )

    with download_col2:

        try:

            excel_bytes = controller.export_batch_excel(results)

            st.download_button(
                "Download Excel (with summary)",
                data=excel_bytes,
                file_name="batch_predictions.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".spreadsheetml.sheet"
                ),
            )

        except Exception as error:

            st.caption(f"Excel export unavailable: {error}")
