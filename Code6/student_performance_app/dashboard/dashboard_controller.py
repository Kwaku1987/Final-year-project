"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    dashboard/dashboard_controller.py

Author:
    Brenya Kwaku Twum

Purpose
-------
Central application orchestrator. Every dashboard page
communicates only with this class.

Responsibilities
----------------
- Coordinate dashboard workflow
- Manage ApplicationState
- Coordinate the ML pipeline
- Handle notifications
- Handle errors

Notes
-----
The uploaded draft had several structural problems: duplicate
method definitions (two best_model properties, two status()
methods), bare statements sitting directly in the class body
outside any method (self.framework = FrameworkFactory.create(),
self.framework.load_dataset(...), a stray script line calling
framework.execute_training_pipeline(...)), and wrong import paths
(services.* instead of backend.*). Those are all removed/fixed
here.

More importantly, the draft self-constructed its own
DatasetValidator(), CPIEngine(), ModelTrainer(), ModelEvaluator(),
ModelInterpreter() and called the raw-dataframe 5-model
trainer.train(...) plus whole-TrainingResult evaluator.evaluate(...)
-- the architecture you decided against for ModelInterpreter and
StudentPerformanceFramework earlier in this project, and moved
away from for ModelTrainer/ModelEvaluator via their
framework-compatible adapters. This controller instead wraps the
already-built, already-tested StudentPerformanceFramework (via
framework_factory.create_framework()) and delegates to it, staying
consistent with everything else in this codebase.

Version
-------
5.3 (adapted)
====================================================================
"""

from __future__ import annotations

import logging

from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd

from framework.framework_factory import (
    create_framework,
)

from dashboard.application_state import (
    ApplicationState,
    AssessmentRecord,
)

from backend.intervention.intervention_engine import (
    InterventionEngine,
)

from backend.reporting.report_engine import ReportEngine
from backend.reporting.pdf_exporter import PDFExporter
from backend.reporting.excel_exporter import ExcelExporter

logger = logging.getLogger(__name__)


# ============================================================
# DASHBOARD CONTROLLER
# ============================================================

class DashboardController:
    """
    Central dashboard controller.

    Every dashboard page communicates only with this class.
    """

    # =======================================================
    # INITIALIZATION
    # =======================================================

    def __init__(self):

        logger.info(
            "Creating DashboardController..."
        )

        self.framework = create_framework()

        self.state = ApplicationState

        self.state.initialize()

        self.state.set_framework(
            self.framework
        )

        self._last_report: Optional[Dict[str, Any]] = None

        self.intervention_engine = InterventionEngine()

        self.report_engine = ReportEngine()

        self.pdf_exporter = PDFExporter()

        self.excel_exporter = ExcelExporter()

        logger.info(
            "DashboardController ready."
        )

    # =======================================================
    # PROPERTIES
    # =======================================================

    @property
    def has_dataset(self) -> bool:

        return self.state.has_dataset()

    @property
    def has_trained_models(self) -> bool:

        return self.state.has_trained_models()

    @property
    def is_ready_for_prediction(self) -> bool:

        return self.state.is_ready_for_prediction()

    @property
    def best_model(self) -> Optional[Any]:

        return self.framework.best_model

    @property
    def best_model_name(self) -> Optional[str]:

        return self.framework.best_model_name

    # =======================================================
    # APPLICATION MODE (Administrator / Lecturer)
    # =======================================================

    @property
    def application_mode(self) -> Optional[str]:

        return self.state.get_application_mode()

    @property
    def is_administrator(self) -> bool:

        return self.state.is_administrator()

    @property
    def is_lecturer(self) -> bool:

        return self.state.is_lecturer()

    @property
    def has_saved_model(self) -> bool:
        """
        True when a saved model bundle exists on disk.

        Checks the filesystem directly rather than relying on
        an in-memory flag: a fresh session (e.g. a Lecturer
        opening the app in a new browser tab) has no way to
        know a model exists just because *this* controller
        instance hasn't trained or loaded one -- an
        Administrator may have saved one in a completely
        separate session.
        """

        exists = self.framework.has_saved_model(
            self.DEFAULT_MODEL_PATH
        )

        # Keep ApplicationState's flag in sync too, since other
        # code (e.g. the sidebar) reads it directly.
        self.state.set_saved_model_available(exists)

        return exists

    def set_mode(self, mode: str) -> None:
        """
        Choose Administrator (Training Mode) or Lecturer
        (Assessment Mode).
        """

        self.state.set_application_mode(mode)

        logger.info("Application mode set to: %s", mode)

    # =======================================================
    # LOAD DATASET
    # =======================================================

    def load_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate an uploaded dataset and store it.
        """

        logger.info("Loading dataset...")

        if dataframe is None:
            raise ValueError(
                "Dataset cannot be None."
            )

        try:

            validated = self.framework.validate_dataset(
                dataframe
            )

            self.state.set_dataset(
                validated
            )

            self.state.success(
                "Dataset uploaded and validated successfully."
            )

            logger.info(
                "Dataset validation successful."
            )

            return validated

        except Exception as error:

            self.state.error(
                f"Dataset validation failed: {error}"
            )

            raise

    # =======================================================
    # PREPARE DATASET
    # =======================================================

    def prepare_dataset(self) -> Any:
        """
        Generate CPI target and prepare data for training.
        """

        logger.info(
            "Preparing dataset..."
        )

        try:

            prepared = self.framework.prepare_data()

            self.state.set_prepared_data(
                prepared
            )

            self.state.success(
                "Dataset prepared for training."
            )

            return prepared

        except Exception as error:

            self.state.error(
                f"Data preparation failed: {error}"
            )

            raise

    # =======================================================
    # TRAIN MODELS
    # =======================================================

    def train_models(self) -> Dict[str, Any]:
        """
        Train the supported models.
        """

        logger.info(
            "Training models..."
        )

        try:

            trained = self.framework.train_models()

            self.state.set_trained_models(
                trained
            )

            self.state.success(
                "Models trained successfully."
            )

            return trained

        except Exception as error:

            self.state.error(
                f"Model training failed: {error}"
            )

            raise

    # =======================================================
    # EVALUATE MODELS
    # =======================================================

    def evaluate_models(self) -> Any:
        """
        Evaluate trained models and select the best one.
        """

        logger.info(
            "Evaluating models..."
        )

        try:

            evaluated = self.framework.evaluate_models()

            self.state.set_comparison_results(
                evaluated
            )

            self.state.set_best_model(
                self.framework.best_model
            )

            self.state.success(
                "Best model selected: "
                f"{self.framework.best_model_name}"
            )

            return evaluated

        except Exception as error:

            self.state.error(
                f"Model evaluation failed: {error}"
            )

            raise

    # =======================================================
    # RUN FULL PIPELINE
    # =======================================================

    def run_full_pipeline(
        self,
        dataframe: pd.DataFrame,
    ) -> Any:
        """
        Run the complete end-to-end workflow: validate, prepare,
        train, evaluate, and interpret, in one call.
        """

        logger.info(
            "Running full pipeline..."
        )

        try:

            result = self.framework.run_pipeline(
                dataframe
            )

            self.state.set_dataset(
                self.framework.validated_data
            )

            self.state.set_prepared_data(
                self.framework.prepared_data
            )

            self.state.set_trained_models(
                self.framework.trained_models
            )

            self.state.set_comparison_results(
                self.framework.comparison_results
            )

            self.state.set_best_model(
                self.framework.best_model
            )

            self.state.success(
                "Pipeline completed. Best model: "
                f"{result.best_model_name}"
            )

            return result

        except Exception as error:

            self.state.error(
                f"Pipeline failed: {error}"
            )

            raise

    # =======================================================
    # ADMINISTRATOR / TRAINING MODE : TRAIN & SAVE
    # =======================================================

    DEFAULT_MODEL_PATH = "models/best_model.pkl"

    def train_and_save_system(
        self,
        dataframe: pd.DataFrame,
        model_path: Optional[str] = None,
    ) -> Any:
        """
        Administrator / Training Mode workflow: run the full
        pipeline, then save the best model so it can be loaded
        later in Assessment Mode without retraining.
        """

        path = model_path or self.DEFAULT_MODEL_PATH

        result = self.run_full_pipeline(dataframe)

        try:

            saved_path = self.framework.save_best_model(
                path
            )

            self.state.set_saved_model_available(True)

            self.state.success(
                f"Model saved to {saved_path}. "
                "It's now available in Assessment Mode."
            )

        except Exception as error:

            self.state.error(
                f"Training succeeded, but saving the model "
                f"failed: {error}"
            )

            raise

        return result

    # =======================================================
    # LECTURER / ASSESSMENT MODE : LOAD SAVED MODEL
    # =======================================================

    def enter_assessment_mode(
        self,
        model_path: Optional[str] = None,
    ) -> str:
        """
        Lecturer / Assessment Mode workflow: load a previously
        saved model, without needing a training pipeline run in
        this session.

        Returns
        -------
        str
            The name of the loaded model.
        """

        path = model_path or self.DEFAULT_MODEL_PATH

        try:

            model_name = self.framework.load_best_model(
                path
            )

            self.state.set_best_model(
                self.framework.best_model
            )

            self.state.set_saved_model_available(True)

            self.state.success(
                f"Loaded saved model: {model_name}"
            )

            return model_name

        except FileNotFoundError as error:

            self.state.error(str(error))

            raise

        except Exception as error:

            self.state.error(
                f"Failed to load saved model: {error}"
            )

            raise

    # =======================================================
    # ASSESS STUDENT (Lecturer / Assessment Mode)
    # =======================================================

    def assess_student(
        self,
        student_name: str,
        student_id: Optional[str],
        student_data: Dict[str, float],
        model_name: Optional[str] = None,
    ) -> Any:
        """
        Predict a student's performance and log it to the
        assessment history. Distinct from predict_student()
        (used in Administrator Mode) in that it also records
        the student's identity for the Lecturer's records.

        Parameters
        ----------
        model_name:
            Optional. Assess using a specific loaded model
            (e.g. "Logistic Regression" or "Random Forest")
            instead of the best model. See
            get_available_models().
        """

        if not student_name or not student_name.strip():

            raise ValueError(
                "Student name is required."
            )

        prediction = self.predict_student(
            student_data,
            model_name=model_name,
        )

        risk = getattr(prediction, "intervention", None)

        record = AssessmentRecord(
            student_name=student_name.strip(),
            student_id=(
                student_id.strip() if student_id else None
            ),
            prediction=prediction.predicted_label,
            confidence=prediction.confidence,
            recommendations=list(
                getattr(prediction, "recommendations", [])
                or []
            ),
            risk_priority=risk.priority if risk else None,
            risk_score=risk.score if risk else None,
        )

        self.state.add_assessment_record(record)

        return prediction, record

    # =======================================================
    # ASSESSMENT HISTORY (Lecturer / Assessment Mode)
    # =======================================================

    def get_assessment_history(self) -> list:

        return self.state.get_assessment_history()

    # =======================================================
    # REPORT EXPORTS
    # =======================================================

    def export_student_pdf(
        self,
        student_data: Dict[str, float],
        prediction: Any,
        student_name: Optional[str] = None,
        student_id: Optional[str] = None,
    ) -> bytes:
        """
        Build a PDF report for one student's prediction.

        Returns
        -------
        bytes
            PDF bytes, ready for st.download_button(data=...).
        """

        report = self.report_engine.generate_student_report(
            student=student_data,
            prediction=prediction,
            student_name=student_name,
            student_id=student_id,
        )

        return self.pdf_exporter.export_student_report(report)

    # -------------------------------------------------------

    def export_batch_excel(
        self,
        results: pd.DataFrame,
    ) -> bytes:
        """
        Build an Excel workbook (Predictions + Summary sheets)
        for a batch prediction result.

        Returns
        -------
        bytes
            Excel bytes, ready for st.download_button(data=...).
        """

        summary = self.report_engine.generate_batch_report(
            results
        )

        return self.excel_exporter.export(results, summary)

    # =======================================================
    # AVAILABLE MODELS (for manual model toggling)
    # =======================================================

    def get_available_models(self) -> list:
        """
        Return the names of all trained/loaded models available
        for prediction, e.g. for populating a "predict with"
        dropdown in the UI.
        """

        return self.framework.get_available_models()

    # =======================================================
    # MODEL INFORMATION (training date, samples, metrics)
    # =======================================================

    def get_model_information(self) -> dict:
        """
        Return a display-ready summary of the active model:
        name, version, training date, training sample count,
        and accuracy/precision/recall/f1. Values are None for
        anything not available (no model yet, or an older saved
        bundle from before this metadata existed).
        """

        return self.framework.get_model_information()

    # =======================================================
    # DELETE SAVED MODEL (Administrator / Training Mode)
    # =======================================================

    def delete_saved_model(
        self,
        model_path: Optional[str] = None,
    ) -> bool:
        """
        Delete the saved model bundle from disk, e.g. before
        retraining with a different dataset.

        Returns
        -------
        bool
            True if a saved model was deleted, False if there
            was nothing to delete.
        """

        path = model_path or self.DEFAULT_MODEL_PATH

        deleted = self.framework.delete_saved_model(path)

        self.state.set_saved_model_available(False)

        if deleted:

            self.state.success(
                "Saved model deleted."
            )

        else:

            self.state.warning(
                "No saved model was found to delete."
            )

        return deleted

    # =======================================================
    # PREDICT STUDENT
    # =======================================================

    def predict_student(
        self,
        student_data: Dict[str, float],
        model_name: Optional[str] = None,
    ) -> Any:
        """
        Predict a single student's performance level.

        Parameters
        ----------
        model_name:
            Optional. Predict using a specific trained/loaded
            model instead of the auto-selected best model. See
            get_available_models().
        """

        logger.info(
            "Generating prediction..."
        )

        try:

            prediction = self.framework.predict(
                student_data,
                model_name=model_name,
            )

            self.state.set_prediction_result(
                prediction
            )

            self.state.success(
                f"Prediction generated: {prediction.predicted_label}"
            )

            return prediction

        except Exception as error:

            self.state.error(
                f"Prediction failed: {error}"
            )

            raise

    # =======================================================
    # BATCH PREDICTION
    # =======================================================

    def predict_batch(
        self,
        dataframe: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Predict performance for every student in a dataframe at
        once (e.g. an uploaded class roster).

        Parameters
        ----------
        dataframe:
            Must contain a column for every entry in
            FEATURE_COLUMNS. Extra columns are preserved and
            carried through to the output unchanged.

        model_name:
            Optional. Predict every row using a specific
            trained/loaded model instead of the auto-selected
            best model.

        Returns
        -------
        pd.DataFrame
            The original dataframe with four columns appended:
            "Predicted Performance", "Confidence (%)",
            "Risk Priority", "Risk Score".
        """

        from backend.config import FEATURE_COLUMNS

        missing = [
            column
            for column in FEATURE_COLUMNS
            if column not in dataframe.columns
        ]

        if missing:

            raise ValueError(
                f"Missing required columns: {missing}"
            )

        logger.info(
            "Running batch prediction on %d student(s)...",
            len(dataframe),
        )

        results = dataframe.copy().reset_index(drop=True)

        predicted_labels = []

        confidences = []

        risk_priorities = []

        risk_scores = []

        for _, row in results.iterrows():

            student_data = {
                column: float(row[column])
                for column in FEATURE_COLUMNS
            }

            try:

                prediction = self.framework.predict(
                    student_data,
                    model_name=model_name,
                )

                predicted_labels.append(
                    prediction.predicted_label
                )

                confidences.append(
                    round(prediction.confidence * 100, 1)
                )

                intervention = getattr(
                    prediction, "intervention", None
                )

                risk_priorities.append(
                    intervention.priority
                    if intervention else None
                )

                risk_scores.append(
                    intervention.score
                    if intervention else None
                )

            except Exception as error:

                logger.warning(
                    "Batch prediction failed for row: %s",
                    error,
                )

                predicted_labels.append("ERROR")

                confidences.append(None)

                risk_priorities.append(None)

                risk_scores.append(None)

        results["Predicted Performance"] = predicted_labels

        results["Confidence (%)"] = confidences

        results["Risk Priority"] = risk_priorities

        results["Risk Score"] = risk_scores

        self.state.success(
            f"Batch prediction completed for {len(results)} "
            "student(s)."
        )

        return results

    # =======================================================
    # RISK ASSESSMENT (Intervention Priority)
    # =======================================================

    def assess_risk(
        self,
        student_data: Dict[str, float],
        prediction,
    ) -> Any:
        """
        Return the intervention risk assessment for a
        prediction.

        predict() already computes this as part of every
        prediction (see PredictionResult.intervention) -- this
        method is a thin, backward-compatible accessor rather
        than a second computation, so pages that already call
        it don't need to change.
        """

        if getattr(prediction, "intervention", None) is not None:

            return prediction.intervention

        return self.intervention_engine.assess(
            student=student_data,
            prediction=prediction.predicted_label,
            confidence=prediction.confidence,
        )

    # =======================================================
    # INTERPRET MODEL
    # =======================================================

    def interpret_model(self) -> Any:
        """
        Generate the interpretation report for the best model.
        """

        logger.info(
            "Interpreting model..."
        )

        try:

            report = self.framework.interpret()

            self.state.set_interpretation_report(
                report
            )

            self.state.success(
                "Model interpretation generated."
            )

            return report

        except Exception as error:

            self.state.error(
                f"Interpretation failed: {error}"
            )

            raise

    # =======================================================
    # GENERATE REPORT
    # =======================================================

    def generate_report(self) -> Dict[str, Any]:
        """
        Build the consolidated dashboard-ready report.
        """

        logger.info(
            "Generating report..."
        )

        try:

            report = self.framework.generate_report()

            self._last_report = report

            self.state.mark_report_ready()

            self.state.success(
                "Report generated successfully."
            )

            return report

        except Exception as error:

            self.state.error(
                f"Report generation failed: {error}"
            )

            raise

    # =======================================================
    # GET REPORT
    # =======================================================

    def get_report(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recently generated report, if any.
        """

        return self._last_report

    # =======================================================
    # DATASET SUMMARY
    # =======================================================

    def dataset_summary(self) -> Optional[Dict[str, Any]]:
        """
        Return a summary of the currently loaded dataset.
        """

        dataframe = self.state.get_dataset()

        if dataframe is None:

            return None

        return {

            "rows":
                len(dataframe),

            "columns":
                len(dataframe.columns),

            "features":
                list(dataframe.columns),

            "missing":
                dataframe.isnull().sum(),

        }

    # =======================================================
    # STATUS
    # =======================================================

    def status(self) -> Dict[str, Any]:
        """
        Return the current framework status.
        """

        return self.framework.get_status()

    # =======================================================
    # RESET
    # =======================================================

    def reset(self) -> None:
        """
        Reset the controller and application state, building a
        fresh framework instance.
        """

        logger.info(
            "Resetting DashboardController..."
        )

        self.framework = create_framework()

        self.state.reset_application()

        self.state.set_framework(
            self.framework
        )

        self._last_report = None

        logger.info(
            "DashboardController reset completed."
        )
