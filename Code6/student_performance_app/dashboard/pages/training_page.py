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
from backend.config import TARGET_DESCRIPTIONS


def render_training_page(controller: DashboardController) -> None:

    st.title("⚙ Model Training")

    if not controller.has_dataset:

        st.warning(
            "Please upload and validate a dataset on the "
            "**📂 Dataset** page first."
        )

        return

    st.caption(
        "This runs the full pipeline: builds the target labels, "
        "prepares features, trains Logistic Regression, Random "
        "Forest and a soft-voting ensemble, cross-validates them, "
        "selects the best model, and saves it for Lecturers to "
        "use in Assessment Mode."
    )

    # ------------------------------------------------------
    # Target definition
    # ------------------------------------------------------

    with st.expander("🎯 Target definition", expanded=False):

        st.write(
            f"**Active definition:** `{controller.target_definition}`"
        )

        st.caption(
            TARGET_DESCRIPTIONS.get(
                controller.target_definition, ""
            )
        )

        st.write(
            "**Predictors used:** "
            + ", ".join(controller.required_features)
        )

        st.caption(
            "The column the label is derived from is always "
            "excluded from the predictors. Switch definition "
            "with the SPAS_TARGET_DEFINITION environment "
            "variable (`exam_band` = default, `cpi` = legacy)."
        )

    # ------------------------------------------------------
    # Validation warnings
    # ------------------------------------------------------

    warnings = controller.validation_warnings

    if warnings:

        with st.expander(
            f"⚠️ {len(warnings)} data note(s) from validation",
            expanded=False,
        ):

            for warning in warnings:

                st.write(f"- {warning}")

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
        import pandas as pd

        cross_validation = controller.cross_validation

        rows = []

        for model_name, evaluation in comparison.items():

            metrics = evaluation.get("metrics", {})

            cv = cross_validation.get(model_name, {})

            rows.append({
                "Model": model_name,
                "Accuracy": round(metrics.get("accuracy", 0), 4),
                "Macro F1": round(metrics.get("macro_f1", 0), 4),
                "Balanced Acc.": round(
                    metrics.get("balanced_accuracy", 0), 4
                ),
                "ROC-AUC": (
                    round(metrics["roc_auc"], 4)
                    if metrics.get("roc_auc") is not None
                    else None
                ),
                "Baseline": round(
                    metrics.get("baseline_majority", 0), 4
                ),
                "CV Mean": (
                    round(cv.get("mean_accuracy", 0), 4)
                    if cv else None
                ),
                "CV Std": (
                    round(cv.get("std_accuracy", 0), 4)
                    if cv else None
                ),
            })

        if rows:

            comparison_df = pd.DataFrame(rows).sort_values(
                "Macro F1", ascending=False
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

            # ----------------------------------------------
            # Read the table honestly
            # ----------------------------------------------
            st.caption(
                "**Baseline** is the accuracy of always "
                "predicting the most common class. Read every "
                "accuracy next to it: on an imbalanced target a "
                "model can beat 90% accuracy while being useless. "
                "**Macro F1** and **balanced accuracy** treat all "
                "three classes equally, so they fall when the "
                "model ignores the minority class. **CV** is "
                "5-fold cross-validation on the training fold "
                "only -- the columns above are held-out test "
                "metrics."
            )

            if controller.target_definition == "cpi":

                st.warning(
                    "This run uses the legacy `cpi` target, which "
                    "is computed from the same columns used as "
                    "predictors. These scores measure how well "
                    "the model reproduces the labelling rule, "
                    "not predictive accuracy. Switch to "
                    "`exam_band` for a genuine prediction task."
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
