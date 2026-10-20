"""
===============================================================
Application State Manager
===============================================================

Project:
    Student Performance Assessment System

Module:
    dashboard/application_state.py

Description:
    Centralized state manager for the Streamlit dashboard.

    This class acts as the single source of truth for every
    dashboard page. Rather than allowing individual pages to
    manipulate Streamlit session_state directly, every page
    communicates through this class.

Notes
-----
Internal storage is composed of DatasetState / TrainingState /
PredictionState sub-objects (dataset upload+validation+prep,
training+evaluation+best-model, prediction+interpretation,
respectively) rather than one flat set of fields. Every public
classmethod on ApplicationState keeps the exact same name,
signature, and behavior as before this restructuring -- only
where each value lives internally changed. No other file needed
to change: the controller, sidebar, and every dashboard page only
ever call these classmethods, never touch DashboardState's fields
directly.

Author:
    Brenya Kwaku Twum

Version:
    6.0 (nested state)
===============================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Workflow Stages
# ------------------------------------------------------------

WORKFLOW_STAGES = [

    "HOME",

    "DATASET",

    "VALIDATION",

    "PREPARATION",

    "TRAINING",

    "EVALUATION",

    "PREDICTION",

    "INTERPRETATION",

    "REPORT"

]


# ------------------------------------------------------------
# Notification Model
# ------------------------------------------------------------

@dataclass
class Notification:
    """
    Represents a dashboard notification.

    Attributes
    ----------
    level : str

        success

        warning

        error

        info

    message : str
    """

    level: str

    message: str


@dataclass
class AssessmentRecord:
    """
    One student assessment logged in Assessment Mode
    (Lecturer/Advisor).

    programme, level, and semester are optional -- only
    student_name is required and student_id is explicitly
    optional, matching how the Assessment page collects this
    information. They're available for institutions that want
    to record them, without requiring every assessment to
    provide them.
    """

    student_name: str

    student_id: Optional[str]

    prediction: str

    confidence: float

    recommendations: List[str] = field(default_factory=list)

    programme: Optional[str] = None

    level: Optional[str] = None

    semester: Optional[str] = None

    risk_priority: Optional[str] = None

    risk_score: Optional[int] = None

    assessment_date: datetime = field(
        default_factory=datetime.now
    )


# ------------------------------------------------------------
# Dataset State
# ------------------------------------------------------------

@dataclass
class DatasetState:
    """
    Everything related to the uploaded dataset and its
    preparation for training.
    """

    dataframe: Optional[pd.DataFrame] = None

    uploaded: bool = False

    prepared_data: Optional[Any] = None


# ------------------------------------------------------------
# Training State
# ------------------------------------------------------------

@dataclass
class TrainingState:
    """
    Everything related to model training, evaluation, and the
    selected best model.
    """

    trained_models: Dict[str, Any] = field(default_factory=dict)

    models_trained: bool = False

    best_model: Optional[Any] = None

    comparison_results: Optional[Any] = None

    # True once a best model has been selected -- i.e. the
    # system is ready to make predictions.
    prediction_ready: bool = False


# ------------------------------------------------------------
# Prediction State
# ------------------------------------------------------------

@dataclass
class PredictionState:
    """
    Everything related to the latest prediction and its
    interpretation.
    """

    prediction_result: Optional[Any] = None

    interpretation_report: Optional[Any] = None

    interpretation_ready: bool = False


# ------------------------------------------------------------
# Dashboard State
# ------------------------------------------------------------

@dataclass
class DashboardState:
    """
    Stores every runtime object used throughout the dashboard,
    composed of the three sub-states above plus cross-cutting
    values that don't belong to just one of them (framework
    instance, notifications, application mode, saved-model
    availability, assessment history, report readiness).
    """

    framework: Optional[Any] = None

    dataset: DatasetState = field(default_factory=DatasetState)

    training: TrainingState = field(
        default_factory=TrainingState
    )

    prediction: PredictionState = field(
        default_factory=PredictionState
    )

    current_stage: str = "HOME"

    notifications: List[Notification] = field(default_factory=list)

    report_ready: bool = False

    # --------------------------------------------------------
    # Application Mode (Administrator / Lecturer)
    # --------------------------------------------------------

    application_mode: Optional[str] = None

    saved_model_available: bool = False

    assessment_history: List[AssessmentRecord] = field(
        default_factory=list
    )


# ------------------------------------------------------------
# Application State Manager
# ------------------------------------------------------------

class ApplicationState:
    """
    Central manager for all dashboard state.

    Every Streamlit page should use this class.

    Never write directly to st.session_state.
    """

    SESSION_KEY = "application_state"

    # --------------------------------------------------------

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize application state.

        This method should be called once
        inside app.py.
        """

        if cls.SESSION_KEY not in st.session_state:

            logger.info(
                "Creating dashboard state..."
            )

            st.session_state[
                cls.SESSION_KEY
            ] = DashboardState()

    # --------------------------------------------------------

    @classmethod
    def state(cls) -> DashboardState:
        """
        Return dashboard state.
        """

        cls.initialize()

        return st.session_state[
            cls.SESSION_KEY
        ]

    # ========================================================
    # Framework
    # ========================================================

    @classmethod
    def set_framework(cls, framework: Any) -> None:
        """
        Store the StudentPerformanceFramework instance.
        """

        logger.info("Framework registered.")

        cls.state().framework = framework

    # --------------------------------------------------------

    @classmethod
    def get_framework(cls) -> Optional[Any]:
        """
        Return the active framework instance.
        """

        return cls.state().framework

    # ========================================================
    # Dataset
    # ========================================================

    @classmethod
    def set_dataset(cls, dataframe: pd.DataFrame) -> None:
        """
        Store the uploaded dataset.
        """

        logger.info(
            "Dataset loaded (%s rows, %s columns).",
            dataframe.shape[0],
            dataframe.shape[1]
        )

        dataset = cls.state().dataset

        dataset.dataframe = dataframe

        dataset.uploaded = True

        cls.state().current_stage = "DATASET"

    # --------------------------------------------------------

    @classmethod
    def get_dataset(cls) -> Optional[pd.DataFrame]:
        """
        Return the current dataset.
        """

        return cls.state().dataset.dataframe

    # --------------------------------------------------------

    @classmethod
    def has_dataset(cls) -> bool:
        """
        Check whether a dataset exists.
        """

        return cls.state().dataset.uploaded

    # --------------------------------------------------------

    @classmethod
    def clear_dataset(cls) -> None:
        """
        Remove the current dataset.
        """

        logger.info("Dataset cleared.")

        state = cls.state()

        state.dataset = DatasetState()

        state.training = TrainingState()

        state.prediction = PredictionState()

        state.report_ready = False

        state.current_stage = "HOME"

    # ========================================================
    # Prepared Data
    # ========================================================

    @classmethod
    def set_prepared_data(cls, prepared_data: Any) -> None:
        """
        Store processed ML-ready data.
        """

        logger.info("Prepared dataset stored.")

        state = cls.state()

        state.dataset.prepared_data = prepared_data

        state.current_stage = "PREPARATION"

    # --------------------------------------------------------

    @classmethod
    def get_prepared_data(cls) -> Optional[Any]:
        """
        Return prepared data.
        """

        return cls.state().dataset.prepared_data

    # ========================================================
    # Trained Models
    # ========================================================

    @classmethod
    def set_trained_models(
        cls,
        models: dict
    ) -> None:
        """
        Store all trained models.
        """

        logger.info(
            "%d trained model(s) stored.",
            len(models)
        )

        state = cls.state()

        state.training.trained_models = models

        state.training.models_trained = True

        state.current_stage = "TRAINING"

    # --------------------------------------------------------

    @classmethod
    def get_trained_models(cls) -> dict:
        """
        Return trained models.
        """

        return cls.state().training.trained_models

    # --------------------------------------------------------

    @classmethod
    def has_trained_models(cls) -> bool:
        """
        Check if models exist.
        """

        return cls.state().training.models_trained

    # --------------------------------------------------------

    @classmethod
    def reset_models(cls) -> None:
        """
        Remove every trained model.

        Used whenever a new dataset is uploaded.
        """

        logger.info("Resetting trained models.")

        state = cls.state()

        state.training.trained_models.clear()

        state.training.best_model = None

        state.training.comparison_results = None

        state.training.models_trained = False

        state.training.prediction_ready = False

        state.prediction.prediction_result = None

        state.prediction.interpretation_report = None

        state.prediction.interpretation_ready = False

        state.report_ready = False

        state.current_stage = "PREPARATION"

    # ========================================================
    # Best Model
    # ========================================================

    @classmethod
    def set_best_model(
        cls,
        model: Any
    ) -> None:
        """
        Store the best performing model.
        """

        logger.info("Best model selected.")

        state = cls.state()

        state.training.best_model = model

        state.training.prediction_ready = True

        state.current_stage = "EVALUATION"

    # --------------------------------------------------------

    @classmethod
    def get_best_model(cls) -> Optional[Any]:
        """
        Return the best trained model.
        """

        return cls.state().training.best_model

    # ========================================================
    # Comparison Results
    # ========================================================

    @classmethod
    def set_comparison_results(
        cls,
        results: Any
    ) -> None:
        """
        Store model comparison results.
        """

        logger.info("Comparison results stored.")

        cls.state().training.comparison_results = results

    # --------------------------------------------------------

    @classmethod
    def get_comparison_results(cls) -> Optional[Any]:
        """
        Return comparison results.
        """

        return cls.state().training.comparison_results

    # ========================================================
    # Prediction
    # ========================================================

    @classmethod
    def set_prediction_result(cls, prediction: Any) -> None:
        """
        Store the latest prediction result.
        """

        logger.info("Prediction result stored.")

        state = cls.state()

        state.prediction.prediction_result = prediction

        state.current_stage = "PREDICTION"

    # --------------------------------------------------------

    @classmethod
    def get_prediction_result(cls) -> Optional[Any]:
        """
        Return the latest prediction.
        """

        return cls.state().prediction.prediction_result

    # ========================================================
    # Interpretation
    # ========================================================

    @classmethod
    def set_interpretation_report(
        cls,
        report: Any
    ) -> None:
        """
        Store interpretation report.
        """

        logger.info("Interpretation report generated.")

        state = cls.state()

        state.prediction.interpretation_report = report

        state.prediction.interpretation_ready = True

        state.current_stage = "INTERPRETATION"

    # --------------------------------------------------------

    @classmethod
    def get_interpretation_report(cls) -> Optional[Any]:
        """
        Return interpretation report.
        """

        return cls.state().prediction.interpretation_report

    # ========================================================
    # Report Status
    # ========================================================

    @classmethod
    def mark_report_ready(cls) -> None:
        """
        Mark report generation complete.
        """

        logger.info("Executive report ready.")

        state = cls.state()

        state.report_ready = True

        state.current_stage = "REPORT"

    # ========================================================
    # Notifications
    # ========================================================

    @classmethod
    def add_notification(
        cls,
        level: str,
        message: str
    ) -> None:
        """
        Add dashboard notification.
        """

        cls.state().notifications.append(

            Notification(
                level=level,
                message=message
            )

        )

    # --------------------------------------------------------

    @classmethod
    def success(
        cls,
        message: str
    ) -> None:
        cls.add_notification(
            "success",
            message
        )

    # --------------------------------------------------------

    @classmethod
    def warning(
        cls,
        message: str
    ) -> None:
        cls.add_notification(
            "warning",
            message
        )

    # --------------------------------------------------------

    @classmethod
    def error(
        cls,
        message: str
    ) -> None:
        cls.add_notification(
            "error",
            message
        )

    # --------------------------------------------------------

    @classmethod
    def info(
        cls,
        message: str
    ) -> None:
        cls.add_notification(
            "info",
            message
        )

    # --------------------------------------------------------

    @classmethod
    def get_notifications(cls) -> List[Notification]:
        """
        Return all notifications.
        """

        return cls.state().notifications

    # --------------------------------------------------------

    @classmethod
    def clear_notifications(cls) -> None:
        """
        Remove every notification.
        """

        cls.state().notifications.clear()

    # ========================================================
    # Workflow
    # ========================================================

    @classmethod
    def set_stage(
        cls,
        stage: str
    ) -> None:
        """
        Set current workflow stage.
        """

        if stage not in WORKFLOW_STAGES:

            raise ValueError(
                f"Unknown workflow stage: {stage}"
            )

        logger.info(
            "Workflow stage -> %s",
            stage
        )

        cls.state().current_stage = stage

    # --------------------------------------------------------

    @classmethod
    def get_stage(cls) -> str:
        """
        Return current workflow stage.
        """

        return cls.state().current_stage

    # ========================================================
    # Status Helpers
    # ========================================================

    @classmethod
    def is_ready_for_training(cls) -> bool:
        """
        Training requires prepared data.
        """

        return cls.state().dataset.prepared_data is not None

    # --------------------------------------------------------

    @classmethod
    def is_ready_for_prediction(cls) -> bool:
        """
        Prediction requires best model.
        """

        return cls.state().training.prediction_ready

    # --------------------------------------------------------

    @classmethod
    def is_ready_for_interpretation(cls) -> bool:
        """
        Interpretation requires report.
        """

        return cls.state().prediction.interpretation_ready

    # --------------------------------------------------------

    @classmethod
    def is_ready_for_reports(cls) -> bool:
        """
        Report export availability.
        """

        return cls.state().report_ready

    # ========================================================
    # Diagnostics
    # ========================================================

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """
        Return dashboard state for debugging.
        """

        state = cls.state()

        return {

            "current_stage":
                state.current_stage,

            "dataset_loaded":
                state.dataset.uploaded,

            "models_trained":
                state.training.models_trained,

            "prediction_ready":
                state.training.prediction_ready,

            "interpretation_ready":
                state.prediction.interpretation_ready,

            "report_ready":
                state.report_ready,

            "notifications":
                len(state.notifications)

        }

    # ========================================================
    # Application Mode (Administrator / Lecturer)
    # ========================================================

    VALID_MODES = ("administrator", "lecturer")

    @classmethod
    def set_application_mode(cls, mode: str) -> None:
        """
        Set the active application mode.

        Parameters
        ----------
        mode : str
            "administrator" (Training Mode) or
            "lecturer" (Assessment Mode).
        """

        if mode not in cls.VALID_MODES:

            raise ValueError(
                f"Invalid application mode: {mode!r}. "
                f"Must be one of {cls.VALID_MODES}."
            )

        logger.info(
            "Application mode set to: %s",
            mode,
        )

        cls.state().application_mode = mode

    # --------------------------------------------------------

    @classmethod
    def get_application_mode(cls) -> Optional[str]:
        """
        Return the current application mode, or None if not
        yet chosen.
        """

        return cls.state().application_mode

    # --------------------------------------------------------

    @classmethod
    def is_administrator(cls) -> bool:
        """
        True when in Administrator / Training Mode.
        """

        return (
            cls.state().application_mode == "administrator"
        )

    # --------------------------------------------------------

    @classmethod
    def is_lecturer(cls) -> bool:
        """
        True when in Lecturer / Assessment Mode.
        """

        return cls.state().application_mode == "lecturer"

    # --------------------------------------------------------

    @classmethod
    def clear_application_mode(cls) -> None:
        """
        Return to the mode-selection screen.
        """

        cls.state().application_mode = None

    # ========================================================
    # Saved Model (for Assessment Mode)
    # ========================================================

    @classmethod
    def set_saved_model_available(
        cls,
        available: bool = True,
    ) -> None:
        """
        Record whether a saved model bundle exists on disk.
        """

        cls.state().saved_model_available = available

    # --------------------------------------------------------

    @classmethod
    def has_saved_model(cls) -> bool:
        """
        True when a saved model bundle is available.
        """

        return cls.state().saved_model_available

    # ========================================================
    # Assessment History (Lecturer / Assessment Mode)
    # ========================================================

    @classmethod
    def add_assessment_record(
        cls,
        record: AssessmentRecord,
    ) -> None:
        """
        Log one completed student assessment.
        """

        cls.state().assessment_history.append(record)

        logger.info(
            "Logged assessment for %s",
            record.student_name,
        )

    # --------------------------------------------------------

    @classmethod
    def get_assessment_history(
        cls,
    ) -> List[AssessmentRecord]:
        """
        Return all logged assessments, most recent last.
        """

        return cls.state().assessment_history

    # --------------------------------------------------------

    @classmethod
    def clear_assessment_history(cls) -> None:
        """
        Clear all logged assessments.
        """

        cls.state().assessment_history = []

    # ========================================================
    # Reset
    # ========================================================

    @classmethod
    def reset_application(cls) -> None:
        """
        Reset dashboard.

        Preserves the framework object, the current application
        mode, and the assessment history -- a routine reset
        (e.g. clearing a dataset to try a different one)
        shouldn't kick an Administrator/Lecturer back to the
        mode-selection screen or erase a Lecturer's logged
        assessments. Use clear_application_mode() /
        clear_assessment_history() explicitly for those.
        """

        logger.info(
            "Resetting dashboard state..."
        )

        framework = cls.get_framework()

        mode = cls.get_application_mode()

        history = cls.get_assessment_history()

        saved_model = cls.has_saved_model()

        st.session_state[
            cls.SESSION_KEY
        ] = DashboardState()

        cls.set_framework(framework)

        if mode is not None:

            cls.set_application_mode(mode)

        cls.state().assessment_history = history

        cls.set_saved_model_available(saved_model)

        logger.info(
            "Dashboard reset complete."
        )
