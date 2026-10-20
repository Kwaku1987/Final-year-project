"""
Student Performance Assessment System
=====================================

Module:
    framework/student_performance_framework.py

Purpose:
    Provide one high-level interface to the complete student
    performance prediction workflow.

Design pattern:
    Facade Pattern

The dashboard should communicate with this class instead of
directly coordinating the validator, CPI engine, preparation
pipeline, trainer, evaluator, and interpreter.

Version:
    4.1 - Frozen Backend
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression

from backend.interpretation.model_interpreter import (
    InterpretationReport,
    ModelInterpreter,
)

from backend.intervention.intervention_engine import (
    InterventionEngine,
)

from backend.config import (
    FEATURE_COLUMNS as _CONFIG_FEATURE_COLUMNS,
)


logger = logging.getLogger(__name__)


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class PredictionResult:
    """
    Represents one student-performance prediction.
    """

    predicted_label: str

    predicted_code: int

    confidence: float

    probabilities: Dict[str, float]

    model_name: str

    student_data: Dict[str, float]

    summary: str

    recommendations: list[str] = field(
        default_factory=list
    )

    # Per-prediction explanation: the globally most-important
    # features, paired with this specific student's values for
    # them (not a true instance-level explanation like SHAP --
    # just "here's what mattered most, and what this student's
    # numbers were").
    top_features: list = field(
        default_factory=list
    )

    # Academic risk score/priority for this student, from
    # InterventionEngine. None if intervention assessment
    # wasn't available.
    intervention: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the prediction into a serializable dictionary.
        """

        return {
            "predicted_label": (
                self.predicted_label
            ),
            "predicted_code": (
                self.predicted_code
            ),
            "confidence": (
                self.confidence
            ),
            "probabilities": (
                self.probabilities
            ),
            "model_name": (
                self.model_name
            ),
            "student_data": (
                self.student_data
            ),
            "summary": self.summary,
            "recommendations": (
                self.recommendations
            ),
            "top_features": (
                self.top_features
            ),
            "intervention": (
                self.intervention
            ),
        }


@dataclass
class FrameworkResult:
    """
    Result returned by run_pipeline().
    """

    validated_data: pd.DataFrame

    prepared_data: Any

    trained_models: Dict[str, Any]

    comparison_results: Any

    best_model_name: str

    best_model: Any

    interpretation_report: (
        InterpretationReport
    )


# ============================================================
# STUDENT PERFORMANCE FRAMEWORK
# ============================================================

class StudentPerformanceFramework:
    """
    High-level coordinator for the complete application.

    The class follows the Facade Pattern and exposes a small,
    stable public API:

        validate_dataset()
        prepare_data()
        train_models()
        predict()
        interpret()
        generate_report()
        run_pipeline()

    Dependencies are injected through the constructor so the
    framework remains independent of the project's internal
    folder structure.
    """

    DEFAULT_LABEL_MAPPING = {
        0: "Low",
        1: "Average",
        2: "High",
    }

    # Sourced from backend/config.py, the single shared source
    # of truth for feature columns across all backend components.
    REQUIRED_FEATURES = list(_CONFIG_FEATURE_COLUMNS)

    # Bumped when the saved model bundle's structure changes in
    # a way that matters for consumers reading model_version.
    MODEL_BUNDLE_VERSION = "6.0"

    def __init__(
        self,
        validator: Any,
        cpi_engine: Any,
        preparation_pipeline: Any,
        trainer: Any,
        evaluator: Any,
        interpreter: Optional[
            ModelInterpreter
        ] = None,
        label_mapping: Optional[
            Dict[int, str]
        ] = None,
        required_features: Optional[
            Sequence[str]
        ] = None,
    ) -> None:
        """
        Initialize the facade.

        Args:
            validator:
                Existing DatasetValidator instance.

            cpi_engine:
                Existing CPIEngine instance.

            preparation_pipeline:
                Existing DataPreparationPipeline instance.

            trainer:
                Existing ModelTrainer instance.

            evaluator:
                Existing ModelEvaluator instance.

            interpreter:
                Optional ModelInterpreter. A default interpreter
                is created when one is not supplied.

            label_mapping:
                Numeric-to-performance-level mapping.

            required_features:
                Ordered model input features.
        """

        self.validator = validator

        self.cpi_engine = cpi_engine

        self.preparation_pipeline = (
            preparation_pipeline
        )

        self.trainer = trainer

        self.evaluator = evaluator

        self.interpreter = (
            interpreter
            or ModelInterpreter()
        )

        self.intervention_engine = InterventionEngine()

        self.label_mapping = (
            label_mapping
            or self.DEFAULT_LABEL_MAPPING.copy()
        )

        self.required_features = list(
            required_features
            or self.REQUIRED_FEATURES
        )

        # Runtime state
        self.raw_data: Optional[
            pd.DataFrame
        ] = None

        self.validated_data: Optional[
            pd.DataFrame
        ] = None

        self.prepared_data: Optional[
            Any
        ] = None

        self.trained_models: Dict[
            str,
            Any,
        ] = {}

        self.comparison_results: Optional[
            Any
        ] = None

        self.best_model_name: Optional[
            str
        ] = None

        self.best_model: Optional[
            Any
        ] = None

        # Whether self.best_model needs scaled input (True for
        # Logistic Regression, trained on X_train_scaled; False
        # for tree-based models like Random Forest, trained on
        # X_train_original). Set in _select_best_model() and
        # used by _prepare_prediction_input() so scaling is only
        # applied when the actual best model requires it --
        # previously, the scaler was applied unconditionally
        # whenever one existed, which would silently corrupt
        # predictions any time a non-scaled model (e.g. Random
        # Forest) won best-model selection.
        self.best_model_requires_scaling: bool = False

        # Populated by save_best_model()/load_best_model().
        self.model_version: Optional[str] = None

        self.model_training_date: Optional[str] = None

        self.model_training_samples: Optional[int] = None

        self.model_metrics: Optional[Dict[str, float]] = None

        self.interpretation_report: Optional[
            InterpretationReport
        ] = None

        logger.info(
            "StudentPerformanceFramework initialized."
        )

    # ========================================================
    # DATASET VALIDATION
    # ========================================================

    def validate_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate an uploaded dataset.

        The validator may return either:

        - A validated DataFrame
        - A validation result object containing a DataFrame

        The method normalizes both patterns.
        """

        self._require_dataframe(
            dataframe
        )

        logger.info(
            "Validating dataset with %d rows.",
            len(dataframe),
        )

        self.raw_data = (
            dataframe.copy()
        )

        result = self.validator.validate(
            dataframe.copy()
        )

        validated = (
            self._extract_dataframe(
                result=result,
                fallback=dataframe,
            )
        )

        self.validated_data = (
            validated.copy()
        )

        logger.info(
            "Dataset validation completed."
        )

        return self.validated_data

    # ========================================================
    # DATA PREPARATION
    # ========================================================

    def prepare_data(
        self,
        dataframe: Optional[
            pd.DataFrame
        ] = None,
    ) -> Any:
        """
        Generate CPI-based performance labels and prepare data.

        Expected sequence:

            Validation
                ->
            CPI target generation
                ->
            Feature preparation
                ->
            ML-ready data
        """

        source = (
            dataframe
            if dataframe is not None
            else self.validated_data
        )

        if source is None:

            raise RuntimeError(
                "No validated dataset is available. "
                "Call validate_dataset() first."
            )

        working_data = (
            source.copy()
        )

        logger.info(
            "Generating CPI target variable."
        )

        working_data = (
            self._build_target_variable(
                working_data
            )
        )

        self.validated_data = (
            working_data.copy()
        )

        logger.info(
            "Preparing machine-learning data."
        )

        self.prepared_data = (
            self._call_prepare_method(
                working_data
            )
        )

        return self.prepared_data

    # ========================================================
    # MODEL TRAINING
    # ========================================================

    def train_models(
        self,
    ) -> Dict[str, Any]:
        """
        Train Logistic Regression and Random Forest.

        The trainer can support either:

            train_logistic_regression()
            train_random_forest()

        or methods accepting prepared_data as an argument.
        """

        if self.prepared_data is None:

            raise RuntimeError(
                "Prepared data is unavailable. "
                "Call prepare_data() first."
            )

        logger.info(
            "Training Logistic Regression."
        )

        logistic_model = (
            self._call_component_method(
                component=self.trainer,
                method_name=(
                    "train_logistic_regression"
                ),
                data=self.prepared_data,
            )
        )

        logger.info(
            "Training Random Forest."
        )

        random_forest_model = (
            self._call_component_method(
                component=self.trainer,
                method_name=(
                    "train_random_forest"
                ),
                data=self.prepared_data,
            )
        )

        self.trained_models = {
            "Logistic Regression": (
                logistic_model
            ),
            "Random Forest": (
                random_forest_model
            ),
        }

        logger.info(
            "Model training completed."
        )

        return self.trained_models

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    def evaluate_models(
        self,
    ) -> Any:
        """
        Evaluate trained models and select the best model.
        """

        if not self.trained_models:

            raise RuntimeError(
                "No trained models are available. "
                "Call train_models() first."
            )

        logger.info(
            "Evaluating trained models."
        )

        evaluated_models: Dict[
            str,
            Any,
        ] = {}

        for name, model in (
            self.trained_models.items()
        ):

            evaluated_models[name] = (
                self._evaluate_model(
                    model
                )
            )

        self.trained_models = (
            evaluated_models
        )

        self.comparison_results = (
            self._compare_models(
                evaluated_models
            )
        )

        (
            self.best_model_name,
            self.best_model,
        ) = self._select_best_model(
            evaluated_models
        )

        self.best_model_requires_scaling = isinstance(
            self._extract_estimator(
                self.best_model
            ),
            LogisticRegression,
        )

        logger.info(
            "Best model selected: %s",
            self.best_model_name,
        )

        return self.comparison_results

    def get_available_models(self) -> list[str]:
        """
        Return the names of all currently trained (or loaded)
        models available for prediction -- e.g. for populating
        a "predict with" dropdown in the UI.
        """

        if not self.trained_models:

            return []

        return list(self.trained_models.keys())

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        student_data: Dict[
            str,
            float,
        ],
        model_name: Optional[str] = None,
    ) -> PredictionResult:
        """
        Predict a student's performance level.

        The method:

        1. Validates input features.
        2. Builds a one-row DataFrame.
        3. Applies the trained preprocessing pipeline when
           available.
        4. Generates a class prediction.
        5. Generates class probabilities.
        6. Produces a readable summary and recommendations.

        Parameters
        ----------
        student_data:
            The student's feature values.

        model_name:
            Optional. Predict using a specific trained model
            (e.g. "Logistic Regression" or "Random Forest")
            instead of the auto-selected best model. Must be a
            key in self.trained_models (or, after
            load_models(), one of the models restored from a
            saved bundle). Defaults to the best model when
            omitted.
        """

        if model_name is not None:

            if (
                not self.trained_models
                or model_name not in self.trained_models
            ):

                raise ValueError(
                    f"Model {model_name!r} is not available. "
                    f"Available models: "
                    f"{self.get_available_models()}"
                )

            selected_result = self.trained_models[
                model_name
            ]

            selected_display_name = model_name

        elif self.best_model is not None:

            selected_result = self.best_model

            selected_display_name = (
                self.best_model_name
            )

        else:

            raise RuntimeError(
                "No best model is available. "
                "Train and evaluate the models first."
            )

        model = (
            self._extract_estimator(
                selected_result
            )
        )

        # Logistic Regression needs scaled input; tree-based
        # models (Random Forest) were trained on the original,
        # unscaled features. This works for any selected model,
        # not just the auto-selected best one.
        requires_scaling = isinstance(
            model,
            LogisticRegression,
        )

        self._validate_student_data(
            student_data
        )

        student_frame = (
            pd.DataFrame(
                [
                    {
                        feature: float(
                            student_data[
                                feature
                            ]
                        )
                        for feature
                        in self.required_features
                    }
                ]
            )
        )

        prediction_input = (
            self._prepare_prediction_input(
                student_frame,
                requires_scaling=requires_scaling,
            )
        )

        raw_prediction = (
            model.predict(
                prediction_input
            )
        )

        raw_value = (
            np.asarray(
                raw_prediction
            ).ravel()[0]
        )

        # CPIEngine writes string labels ("Low" / "Average" /
        # "High") into the target column, so models trained on
        # it predict strings, not 0/1/2 codes. Handle both so
        # this keeps working regardless of which target
        # encoding a given CPIEngine/trainer combination uses.
        if isinstance(raw_value, (str, np.str_, bytes)):

            predicted_label = str(raw_value)

            reverse_mapping = {
                label: code
                for code, label
                in self.label_mapping.items()
            }

            predicted_code = reverse_mapping.get(
                predicted_label,
                -1,
            )

        else:

            predicted_code = int(raw_value)

            predicted_label = (
                self.label_mapping.get(
                    predicted_code,
                    str(predicted_code),
                )
            )

        probabilities = (
            self._predict_probabilities(
                model=model,
                prediction_input=(
                    prediction_input
                ),
            )
        )

        confidence = (
            max(
                probabilities.values()
            )
            if probabilities
            else 1.0
        )

        summary = (
            self._build_prediction_summary(
                predicted_label=(
                    predicted_label
                ),
                confidence=confidence,
                student_data=(
                    student_data
                ),
            )
        )

        recommendations = (
            self._prediction_recommendations(
                predicted_label=(
                    predicted_label
                ),
                student_data=(
                    student_data
                ),
            )
        )

        # Per-prediction explanation: the globally most
        # important features for this model, paired with this
        # student's actual values. Best-effort -- a model type
        # extract_feature_importance() doesn't support shouldn't
        # break prediction itself.
        top_features: list = []

        try:

            importance_table = (
                self.interpreter.extract_feature_importance(
                    model,
                    self.required_features,
                )
            )

            for _, row in importance_table.head(3).iterrows():

                feature_name = row["feature"]

                top_features.append({
                    "feature": feature_name,
                    "display_name": row["display_name"],
                    "importance": float(row["importance"]),
                    "value": student_data.get(feature_name),
                })

        except Exception as error:

            logger.warning(
                "Could not compute per-prediction feature "
                "explanation: %s",
                error,
            )

        # Academic risk / intervention assessment. Best-effort
        # for the same reason -- a malformed student_data field
        # shouldn't break the prediction itself.
        intervention = None

        try:

            intervention = (
                self.intervention_engine.assess(
                    student=student_data,
                    prediction=predicted_label,
                    confidence=confidence,
                )
            )

        except Exception as error:

            logger.warning(
                "Could not compute intervention assessment: %s",
                error,
            )

        return PredictionResult(
            predicted_label=(
                predicted_label
            ),
            predicted_code=(
                predicted_code
            ),
            confidence=confidence,
            probabilities=(
                probabilities
            ),
            model_name=(
                selected_display_name
                or model.__class__.__name__
            ),
            student_data={
                key: float(value)
                for key, value
                in student_data.items()
            },
            summary=summary,
            recommendations=(
                recommendations
            ),
            top_features=top_features,
            intervention=intervention,
        )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    def interpret(
        self,
    ) -> InterpretationReport:
        """
        Interpret the selected best model.
        """

        if self.best_model is None:

            raise RuntimeError(
                "No best model is available. "
                "Evaluate the models first."
            )

        estimator = (
            self._extract_estimator(
                self.best_model
            )
        )

        evaluation_metrics = (
            self._extract_metrics(
                self.best_model
            )
        )

        self.interpretation_report = (
            self.interpreter.interpret(
                model=estimator,
                feature_names=(
                    self.required_features
                ),
                model_name=(
                    self.best_model_name
                ),
                evaluation_results=(
                    evaluation_metrics
                ),
            )
        )

        return self.interpretation_report

    # ========================================================
    # REPORT
    # ========================================================

    def generate_report(
        self,
    ) -> Dict[str, Any]:
        """
        Generate a dashboard-ready executive report.
        """

        if (
            self.interpretation_report
            is None
        ):

            self.interpret()

        return {
            "project": (
                "Student Performance "
                "Assessment System"
            ),
            "best_model": (
                self.best_model_name
            ),
            "comparison_results": (
                self.comparison_results
            ),
            "interpretation": (
                self.interpretation_report
                .to_dict()
            ),
            "workflow_status": (
                self.get_status()
            ),
        }

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def run_pipeline(
        self,
        dataframe: pd.DataFrame,
    ) -> FrameworkResult:
        """
        Execute the complete end-to-end workflow.

        Sequence:

            Validate
                ->
            Generate CPI target
                ->
            Prepare data
                ->
            Train models
                ->
            Evaluate models
                ->
            Select best model
                ->
            Interpret best model
        """

        validated = (
            self.validate_dataset(
                dataframe
            )
        )

        prepared = (
            self.prepare_data(
                validated
            )
        )

        trained = (
            self.train_models()
        )

        comparison = (
            self.evaluate_models()
        )

        report = (
            self.interpret()
        )

        return FrameworkResult(
            validated_data=(
                self.validated_data
            ),
            prepared_data=prepared,
            trained_models=trained,
            comparison_results=(
                comparison
            ),
            best_model_name=(
                self.best_model_name
            ),
            best_model=(
                self.best_model
            ),
            interpretation_report=(
                report
            ),
        )

    # ========================================================
    # MODEL PERSISTENCE
    # ========================================================
    # Supports Assessment Mode (Lecturer/Advisor): the best
    # model, its scaler, feature names, whether it needs scaled
    # input, and the label mapping are saved together as one
    # self-contained bundle, so a fresh session can load it and
    # call predict() without re-running the training pipeline.
    # ========================================================

    def save_best_model(
        self,
        model_path: str = "models/best_model.pkl",
    ) -> str:
        """
        Save ALL currently trained models as one self-contained
        bundle (not just the best one) -- so a later session can
        load the bundle and still toggle between models via
        predict(student_data, model_name=...), not just use the
        auto-selected best model.

        Returns
        -------
        str
            The path the bundle was saved to.
        """

        if self.best_model is None or not self.trained_models:

            raise ValueError(
                "No trained model available to save. "
                "Train and evaluate models first."
            )

        path = Path(model_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        scaler = None

        if isinstance(self.prepared_data, dict):

            scaler = self.prepared_data.get("scaler")

        models = {
            name: self._extract_estimator(result)
            for name, result in self.trained_models.items()
        }

        requires_scaling = {
            name: isinstance(estimator, LogisticRegression)
            for name, estimator in models.items()
        }

        training_samples = (
            len(self.validated_data)
            if self.validated_data is not None
            else None
        )

        metrics = None

        if isinstance(self.comparison_results, dict):

            best_result = self.comparison_results.get(
                self.best_model_name
            )

            if isinstance(best_result, dict):

                metrics = best_result.get("metrics")

        training_date = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        bundle = {

            "models": models,

            "requires_scaling": requires_scaling,

            "best_model_name": self.best_model_name,

            "scaler": scaler,

            "feature_names": list(self.required_features),

            "label_mapping": dict(self.label_mapping),

            # Metadata -- for a "model info" display (e.g. in
            # Assessment Mode, so a Lecturer can see when/how
            # the loaded model was trained).
            "model_version": self.MODEL_BUNDLE_VERSION,

            "training_date": training_date,

            "training_samples": training_samples,

            "metrics": metrics,

        }

        joblib.dump(bundle, path)

        self.model_version = self.MODEL_BUNDLE_VERSION

        self.model_training_date = training_date

        self.model_training_samples = training_samples

        self.model_metrics = metrics

        logger.info(
            "Saved %d model(s) to %s: %s",
            len(models),
            path,
            list(models.keys()),
        )

        return str(path)

    # --------------------------------------------------------

    def load_best_model(
        self,
        model_path: str = "models/best_model.pkl",
    ) -> str:
        """
        Load a previously saved model bundle, restoring enough
        state for predict() to work -- including toggling
        between all originally-trained models via
        predict(student_data, model_name=...) -- without a
        training pipeline.

        Returns
        -------
        str
            The name of the best model in the loaded bundle
            (used by default when predict() is called without
            an explicit model_name).
        """

        path = Path(model_path)

        if not path.exists():

            raise FileNotFoundError(
                f"No saved model found at {path}. "
                "Train and save a model first (Administrator / "
                "Training Mode)."
            )

        bundle = joblib.load(path)

        # Backward compatible with bundles saved by an earlier
        # version of this method, which stored a single "model"
        # instead of a "models" dict.
        if "models" in bundle:

            models = bundle["models"]

            requires_scaling_map = bundle["requires_scaling"]

            best_name = bundle["best_model_name"]

        else:

            best_name = bundle["model_name"]

            models = {best_name: bundle["model"]}

            requires_scaling_map = {
                best_name: bundle["requires_scaling"],
            }

        self.trained_models = models

        self.best_model_name = best_name

        self.best_model = models[best_name]

        self.best_model_requires_scaling = (
            requires_scaling_map[best_name]
        )

        self.label_mapping = bundle["label_mapping"]

        # Metadata -- absent in bundles saved before this was
        # added, hence the .get() defaults.
        self.model_version = bundle.get("model_version")

        self.model_training_date = bundle.get("training_date")

        self.model_training_samples = bundle.get(
            "training_samples"
        )

        self.model_metrics = bundle.get("metrics")

        # Minimal prepared_data -- just enough for
        # _prepare_prediction_input() to find the scaler.
        self.prepared_data = {
            "scaler": bundle["scaler"],
            "feature_names": bundle["feature_names"],
        }

        logger.info(
            "Loaded %d model(s) from %s. Best: %s",
            len(models),
            path,
            self.best_model_name,
        )

        return self.best_model_name

    # --------------------------------------------------------

    @staticmethod
    def has_saved_model(
        model_path: str = "models/best_model.pkl",
    ) -> bool:
        """
        Check whether a saved model bundle exists on disk.
        """

        return Path(model_path).exists()

    # --------------------------------------------------------

    @staticmethod
    def delete_saved_model(
        model_path: str = "models/best_model.pkl",
    ) -> bool:
        """
        Delete a saved model bundle from disk, so an
        Administrator can clear it (e.g. before retraining with
        a different dataset).

        Returns
        -------
        bool
            True if a file was deleted, False if there was
            nothing to delete.
        """

        path = Path(model_path)

        if not path.exists():

            return False

        path.unlink()

        logger.info(
            "Deleted saved model bundle at %s",
            path,
        )

        return True

    # --------------------------------------------------------

    def get_model_information(self) -> Dict[str, Any]:
        """
        Return a display-ready summary of the active model --
        for a "model info" panel (e.g. on the Assessment page,
        so a Lecturer can see when/how the loaded model was
        trained).

        Values are None for anything not available (e.g. no
        model trained/loaded yet, or a bundle saved before this
        metadata existed).
        """

        metrics = self.model_metrics or {}

        return {

            "model_name": self.best_model_name,

            "version": self.model_version,

            "training_date": self.model_training_date,

            "training_samples": self.model_training_samples,

            "accuracy": metrics.get("accuracy"),

            "precision": metrics.get("precision"),

            "recall": metrics.get("recall"),

            "f1_score": metrics.get(
                "f1_score",
                metrics.get("f1"),
            ),

        }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(
        self,
    ) -> Dict[str, bool]:
        """
        Return the current framework status.
        """

        return {
            "dataset_validated": (
                self.validated_data
                is not None
            ),
            "data_prepared": (
                self.prepared_data
                is not None
            ),
            "models_trained": bool(
                self.trained_models
            ),
            "best_model_selected": (
                self.best_model
                is not None
            ),
            "interpretation_ready": (
                self.interpretation_report
                is not None
            ),
        }

    # ========================================================
    # INTERNAL CPI AND PREPARATION METHODS
    # ========================================================

    def _build_target_variable(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Call the CPI engine using the supported frozen method.

        Supported method names:

            build_target_variable()
            generate_target_variable()
        """

        if hasattr(
            self.cpi_engine,
            "build_target_variable",
        ):

            return (
                self.cpi_engine
                .build_target_variable(
                    dataframe
                )
            )

        if hasattr(
            self.cpi_engine,
            "generate_target_variable",
        ):

            return (
                self.cpi_engine
                .generate_target_variable(
                    dataframe
                )
            )

        raise AttributeError(
            "CPIEngine must provide "
            "'build_target_variable()' or "
            "'generate_target_variable()'."
        )

    def _call_prepare_method(
        self,
        dataframe: pd.DataFrame,
    ) -> Any:
        """
        Call the data preparation pipeline.
        """

        for method_name in (
            "prepare",
            "prepare_data",
            "fit_transform",
        ):

            if hasattr(
                self.preparation_pipeline,
                method_name,
            ):

                method = getattr(
                    self.preparation_pipeline,
                    method_name,
                )

                return method(
                    dataframe
                )

        raise AttributeError(
            "DataPreparationPipeline does not "
            "provide a supported preparation method."
        )

    # ========================================================
    # EVALUATION METHODS
    # ========================================================

    def _evaluate_model(
        self,
        model: Any,
    ) -> Any:
        """
        Evaluate one trained model.
        """

        if not hasattr(
            self.evaluator,
            "evaluate",
        ):

            return model

        evaluate = (
            self.evaluator.evaluate
        )

        try:

            return evaluate(
                model
            )

        except TypeError:

            return evaluate(
                model,
                self.prepared_data,
            )

    def _compare_models(
        self,
        models: Dict[
            str,
            Any,
        ],
    ) -> Any:
        """
        Compare all evaluated models.
        """

        if not hasattr(
            self.evaluator,
            "compare_models",
        ):

            return models

        compare = (
            self.evaluator
            .compare_models
        )

        try:

            return compare(
                models
            )

        except TypeError:

            return compare(
                *models.values()
            )

    def _select_best_model(
        self,
        models: Dict[
            str,
            Any,
        ],
    ) -> tuple[str, Any]:
        """
        Select the best model using the evaluator.

        Falls back to the highest available accuracy or F1
        score when the evaluator does not provide a dedicated
        selection method.
        """

        if hasattr(
            self.evaluator,
            "select_best_model",
        ):

            selected = (
                self.evaluator
                .select_best_model(
                    self.comparison_results
                )
            )

            if isinstance(
                selected,
                tuple,
            ):

                name, model = selected

                return (
                    str(name),
                    model,
                )

            if isinstance(
                selected,
                str,
            ):

                return (
                    selected,
                    models[selected],
                )

            for name, model in (
                models.items()
            ):

                if selected is model:

                    return (
                        name,
                        model,
                    )

        scored_models = []

        for name, model_result in (
            models.items()
        ):

            metrics = (
                self._extract_metrics(
                    model_result
                )
            )

            score = (
                metrics.get(
                    "f1_score",
                    metrics.get(
                        "f1",
                        metrics.get(
                            "accuracy",
                            0.0,
                        ),
                    ),
                )
            )

            scored_models.append(
                (
                    float(score),
                    name,
                    model_result,
                )
            )

        if not scored_models:

            raise RuntimeError(
                "Unable to select a best model."
            )

        _, name, model = max(
            scored_models,
            key=lambda item: item[0],
        )

        return name, model

    # ========================================================
    # MODEL AND METRIC EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_estimator(
        model_result: Any,
    ) -> Any:
        """
        Extract the fitted estimator from a model result.
        """

        if hasattr(
            model_result,
            "predict",
        ):

            return model_result

        for attribute in (
            "model",
            "estimator",
            "trained_model",
        ):

            if hasattr(
                model_result,
                attribute,
            ):

                candidate = getattr(
                    model_result,
                    attribute,
                )

                if hasattr(
                    candidate,
                    "predict",
                ):

                    return candidate

        if isinstance(
            model_result,
            dict,
        ):

            for key in (
                "model",
                "estimator",
                "trained_model",
            ):

                candidate = (
                    model_result.get(
                        key
                    )
                )

                if (
                    candidate is not None
                    and hasattr(
                        candidate,
                        "predict",
                    )
                ):

                    return candidate

        raise ValueError(
            "Unable to locate a fitted estimator "
            "inside the model result."
        )

    @staticmethod
    def _extract_metrics(
        model_result: Any,
    ) -> Dict[str, Any]:
        """
        Extract evaluation metrics from common result formats.
        """

        if isinstance(
            model_result,
            dict,
        ):

            for key in (
                "metrics",
                "evaluation_results",
                "results",
            ):

                value = (
                    model_result.get(
                        key
                    )
                )

                if isinstance(
                    value,
                    dict,
                ):

                    return value

            return {
                key: value
                for key, value
                in model_result.items()
                if isinstance(
                    value,
                    (
                        int,
                        float,
                        np.number,
                    ),
                )
            }

        for attribute in (
            "metrics",
            "evaluation_results",
            "results",
        ):

            if hasattr(
                model_result,
                attribute,
            ):

                value = getattr(
                    model_result,
                    attribute,
                )

                if isinstance(
                    value,
                    dict,
                ):

                    return value

        return {}

    # ========================================================
    # PREDICTION PREPARATION
    # ========================================================

    def _prepare_prediction_input(
        self,
        student_frame: pd.DataFrame,
        requires_scaling: Optional[bool] = None,
    ) -> Any:
        """
        Apply the stored preprocessing pipeline when available.

        Parameters
        ----------
        requires_scaling:
            Whether the model this input is being prepared for
            needs scaled features. Defaults to
            self.best_model_requires_scaling when not given,
            for backward compatibility with any other internal
            callers.
        """

        prepared = (
            self.prepared_data
        )

        if prepared is None:

            return student_frame

        if requires_scaling is None:

            requires_scaling = (
                self.best_model_requires_scaling
            )

        if not requires_scaling:

            return student_frame

        for attribute in (
            "preprocessor",
            "scaler",
            "feature_scaler",
        ):

            if hasattr(
                prepared,
                attribute,
            ):

                transformer = getattr(
                    prepared,
                    attribute,
                )

                if hasattr(
                    transformer,
                    "transform",
                ):

                    return (
                        transformer.transform(
                            student_frame
                        )
                    )

        if isinstance(
            prepared,
            dict,
        ):

            for key in (
                "preprocessor",
                "scaler",
                "feature_scaler",
            ):

                transformer = (
                    prepared.get(
                        key
                    )
                )

                if (
                    transformer is not None
                    and hasattr(
                        transformer,
                        "transform",
                    )
                ):

                    return (
                        transformer.transform(
                            student_frame
                        )
                    )

        return student_frame

    def _predict_probabilities(
        self,
        model: Any,
        prediction_input: Any,
    ) -> Dict[str, float]:
        """
        Generate class probabilities when supported.
        """

        if not hasattr(
            model,
            "predict_proba",
        ):

            return {}

        values = (
            np.asarray(
                model.predict_proba(
                    prediction_input
                )
            )
            .ravel()
        )

        classes = getattr(
            model,
            "classes_",
            range(
                len(values)
            ),
        )

        def _label_for(class_code: Any) -> str:

            if isinstance(
                class_code,
                (str, np.str_, bytes),
            ):

                return str(class_code)

            return self.label_mapping.get(
                int(class_code),
                str(class_code),
            )

        return {
            _label_for(class_code): float(probability)
            for class_code, probability
            in zip(
                classes,
                values,
            )
        }

    # ========================================================
    # PREDICTION EXPLANATIONS
    # ========================================================

    def _build_prediction_summary(
        self,
        predicted_label: str,
        confidence: float,
        student_data: Dict[
            str,
            float,
        ],
    ) -> str:
        """
        Generate a readable prediction summary.
        """

        confidence_text = (
            "high"
            if confidence >= 0.90
            else (
                "moderate"
                if confidence >= 0.70
                else "limited"
            )
        )

        return (
            f"The model predicts a {predicted_label} "
            f"academic performance level with "
            f"{confidence:.1%} confidence. "
            f"This represents {confidence_text} "
            f"prediction confidence and should be "
            f"considered alongside educator judgment."
        )

    @staticmethod
    def _prediction_recommendations(
        predicted_label: str,
        student_data: Dict[
            str,
            float,
        ],
    ) -> list[str]:
        """
        Generate recommendations based on the predicted class.
        """

        normalized = (
            predicted_label
            .strip()
            .lower()
        )

        if normalized == "low":

            return [
                "Provide early academic support and "
                "monitor progress regularly.",
                "Review attendance and identify barriers "
                "to consistent participation.",
                "Create a structured study and revision "
                "plan.",
                "Use assignment and test feedback to "
                "address specific learning gaps.",
            ]

        if normalized == "average":

            return [
                "Provide targeted support in weaker "
                "academic areas.",
                "Encourage more consistent study habits.",
                "Monitor continuous assessment results "
                "before major examinations.",
                "Set achievable improvement goals and "
                "review progress regularly.",
            ]

        return [
            "Maintain consistent attendance and study "
            "habits.",
            "Continue strong performance in assignments "
            "and assessments.",
            "Provide enrichment opportunities where "
            "appropriate.",
            "Monitor performance to ensure that strong "
            "results are sustained.",
        ]

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_student_data(
        self,
        student_data: Dict[
            str,
            float,
        ],
    ) -> None:
        """
        Validate student prediction input.
        """

        missing = [
            feature
            for feature
            in self.required_features
            if feature
            not in student_data
        ]

        if missing:

            raise ValueError(
                "Student data is missing required "
                f"features: {missing}"
            )

        for feature in (
            self.required_features
        ):

            try:

                value = float(
                    student_data[
                        feature
                    ]
                )

            except (
                TypeError,
                ValueError,
            ) as error:

                raise ValueError(
                    f"Feature '{feature}' must be "
                    "numeric."
                ) from error

            if not np.isfinite(
                value
            ):

                raise ValueError(
                    f"Feature '{feature}' must be "
                    "a finite number."
                )

    @staticmethod
    def _require_dataframe(
        dataframe: Any,
    ) -> None:
        """
        Validate a DataFrame input.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Expected a pandas DataFrame."
            )

        if dataframe.empty:

            raise ValueError(
                "The dataset is empty."
            )

    @staticmethod
    def _extract_dataframe(
        result: Any,
        fallback: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract a DataFrame from common validation-result
        structures.
        """

        if isinstance(
            result,
            pd.DataFrame,
        ):

            return result

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "dataframe",
                "data",
                "validated_data",
            ):

                candidate = (
                    result.get(
                        key
                    )
                )

                if isinstance(
                    candidate,
                    pd.DataFrame,
                ):

                    return candidate

        for attribute in (
            "dataframe",
            "data",
            "validated_data",
        ):

            if hasattr(
                result,
                attribute,
            ):

                candidate = getattr(
                    result,
                    attribute,
                )

                if isinstance(
                    candidate,
                    pd.DataFrame,
                ):

                    return candidate

        return fallback.copy()

    @staticmethod
    def _call_component_method(
        component: Any,
        method_name: str,
        data: Any,
    ) -> Any:
        """
        Call a component method with or without prepared data.
        """

        if not hasattr(
            component,
            method_name,
        ):

            raise AttributeError(
                f"{component.__class__.__name__} "
                f"does not provide '{method_name}()'."
            )

        method = getattr(
            component,
            method_name,
        )

        try:

            return method()

        except TypeError:

            return method(
                data
            )