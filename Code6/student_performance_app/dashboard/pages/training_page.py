"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/pages/training_page.py

Purpose:
    Generate CPI target, prepare data, train models, and compare
    their performance.
====================================================================
"""

from __future__ import annotations

import streamlit as st

from dashboard.dashboard_controller import DashboardController


def render_training_page(controller: DashboardController) -> None:

    st.title("⚙ Model Training")

    if not controller.has_dataset:

        st.warning(
            "Please upload and validate a dataset on the "
            "**📂 Dataset** page first."
        )

        return

    st.caption(
        "This runs the full pipeline: generates the Composite "
        "Performance Index (CPI), prepares features, trains "
        "Logistic Regression and Random Forest, selects the "
        "best model, and saves it for Lecturers to use in "
        "Assessment Mode."
    )

    if st.button("🚀 Run Training Pipeline", type="primary"):

        dataframe = controller.state.get_dataset()

        with st.spinner(
            "Generating CPI, preparing data, training models, "
            "and saving the best one..."
        ):

            try:

                controller.train_and_save_system(dataframe)

            except Exception as error:

                st.error(f"Training pipeline failed: {error}")

                return

        st.success(
            f"Training complete! Best model: "
            f"**{controller.best_model_name}** (saved for "
            "Assessment Mode)."
        )

        st.rerun()

    st.divider()

    if not controller.has_trained_models:

        st.info("No models trained yet.")

        return

    st.subheader("Model Comparison")

    comparison = controller.state.get_comparison_results()

    if comparison is not None:

        # comparison here is the dict returned by
        # StudentPerformanceFramework.evaluate_models() -- shape
        # depends on ModelEvaluator's per-model evaluate() output
        # aggregated by the framework. Fall back to a simple
        # metrics table per model.
        rows = []

        for model_name, evaluation in comparison.items():

            metrics = evaluation.get("metrics", {})

            rows.append({
                "Model": model_name,
                "Accuracy": round(metrics.get("accuracy", 0), 4),
                "Precision": round(metrics.get("precision", 0), 4),
                "Recall": round(metrics.get("recall", 0), 4),
                "F1 Score": round(
                    metrics.get("f1_score", metrics.get("f1", 0)), 4
                ),
            })

        if rows:

            import pandas as pd

            comparison_df = pd.DataFrame(rows).sort_values(
                "Accuracy", ascending=False
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

    st.success(f"🏆 Best Model: **{controller.best_model_name}**")

    st.divider()

    st.subheader("Manage Saved Model")

    if controller.has_saved_model:

        model_info = controller.get_model_information()

        info_col1, info_col2 = st.columns(2)

        with info_col1:

            st.write(f"**Model:** {model_info.get('model_name')}")

            if model_info.get("training_date"):

                st.write(
                    f"**Trained on:** {model_info['training_date']}"
                )

        with info_col2:

            if model_info.get("accuracy") is not None:

                st.write(
                    f"**Accuracy:** {model_info['accuracy']:.1%}"
                )

            if model_info.get("training_samples"):

                st.write(
                    "**Training samples:** "
                    f"{model_info['training_samples']}"
                )

        confirm_delete = st.checkbox(
            "I understand this will remove the model Lecturers "
            "are currently using in Assessment Mode."
        )

        if st.button(
            "🗑️ Delete Saved Model",
            disabled=not confirm_delete,
        ):

            controller.delete_saved_model()

            st.rerun()

    else:

        st.caption(
            "No model is currently saved for Assessment Mode."
        )

    st.divider()

    st.subheader("Next Step")

    st.write(
        "Head to **🔮 Prediction** to predict a student's "
        "performance, or **🧠 Explainable AI** to understand "
        "what drives the model's decisions."
    )
