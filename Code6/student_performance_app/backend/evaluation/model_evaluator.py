"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/evaluation/model_evaluator.py

Author:
    Brenya Kwaku Twum

Purpose
-------
Evaluates trained machine learning models.

Responsibilities
----------------
- Evaluate trained models
- Compute classification metrics
- Generate confusion matrices
- Generate classification reports
- Compute ROC-AUC and feature importance
- Rank models
- Select best model

Notes
-----
Reconstructed from a notebook-style paste with several issues:
a recursive self-call bug in _extract_feature_importance() (it
called itself at its own end, alongside _compute_roc_auc() --
clearly a paste duplication, since that would recurse forever),
a dead scratch Pipeline([...]) expression sitting outside any
method, and a duplicated/broken second copy of
_validate_training_result() merged into leftover
_extract_feature_importance() fragments. Those were removed, and
the recursive call was replaced with a single call from
_evaluate_model() so ROC-AUC and feature importance run once per
model as part of the normal evaluation loop.

evaluate(model, prepared_data) was added because
StudentPerformanceFramework._evaluate_model() calls
self.evaluator.evaluate(model) first, and on TypeError retries
self.evaluator.evaluate(model, self.prepared_data) -- one model,
not a whole TrainingResult. The original evaluate(training_result)
(full multi-model comparison across everything trained by
ModelTrainer.train()) is kept, renamed to evaluate_all(), as a
standalone entry point -- e.g. for a "compare all models"
dashboard page -- since it isn't what framework.py calls during
the normal predict pipeline (which only trains Logistic Regression
+ Random Forest via the trainer's framework-compatible adapters).

Also fixed the import path: services.model_trainer ->
backend.training.model_trainer, matching this project's actual
package layout.

Version
-------
5.2 (adapted)
====================================================================
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from dataclasses import field

from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from sklearn.preprocessing import (
    label_binarize,
)

from sklearn.inspection import (
    permutation_importance,
)

from backend.training.model_trainer import (
    TrainingResult,
)

logger = logging.getLogger(__name__)

# ============================================================
# EVALUATION RESULT
# ============================================================

@dataclass
class EvaluationResult:
    """
    Stores evaluation artifacts for every model.
    """

    metrics: Dict[str, Dict] = field(
        default_factory=dict
    )

    confusion_matrices: Dict[str, np.ndarray] = field(
        default_factory=dict
    )

    classification_reports: Dict[str, Dict] = field(
        default_factory=dict
    )

    probability_predictions: Dict[str, np.ndarray] = field(
        default_factory=dict
    )

    roc_auc_scores: Dict[str, float] = field(
        default_factory=dict
    )

    feature_importance: Dict[str, pd.DataFrame] = field(
        default_factory=dict
    )

    comparison_table: Optional[pd.DataFrame] = None

    best_model_name: Optional[str] = None

    best_model: Optional[object] = None

# ============================================================
# MODEL EVALUATOR
# ============================================================

class ModelEvaluator:
    """
    Evaluates trained models.
    """

    def __init__(self):

        logger.info(
            "ModelEvaluator initialized."
        )

    # =======================================================
    # FRAMEWORK-COMPATIBLE ADAPTER
    # =======================================================
    # This is what StudentPerformanceFramework._evaluate_model()
    # actually calls: one already-fitted model + the prepared_data
    # dict from DataPreparationPipeline.prepare(). Returns a dict
    # shaped for framework.py's _extract_metrics() (looks for a
    # "metrics" key).
    # =======================================================

    def evaluate(
        self,
        model,
        prepared_data: dict,
    ) -> dict:
        """
        Evaluate a single already-fitted model against the
        held-out test set in prepared_data.
        """

        logger.info(
            "Evaluating %s",
            type(model).__name__,
        )

        # Logistic Regression was trained on scaled features;
        # everything else (Random Forest, etc.) on the original,
        # unscaled features -- matches how ModelTrainer's
        # framework-compatible adapters trained them.
        if isinstance(model, LogisticRegression):

            X_test = prepared_data["X_test_scaled"]

        else:

            X_test = prepared_data["X_test_original"]

        y_test = prepared_data["y_test"]

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions,
            average="weighted", zero_division=0,
        )

        recall = recall_score(
            y_test, predictions,
            average="weighted", zero_division=0,
        )

        f1 = f1_score(
            y_test, predictions,
            average="weighted", zero_division=0,
        )

        cm = confusion_matrix(y_test, predictions)

        report = classification_report(
            y_test, predictions,
            output_dict=True, zero_division=0,
        )

        return {

            "model": model,

            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f1_score": f1,
            },

            "confusion_matrix": cm,

            "classification_report": report,
        }

    # =======================================================
    # FULL PIPELINE : EVALUATE ALL MODELS
    # =======================================================
    # Standalone entry point -- evaluates every model in a
    # TrainingResult (as produced by ModelTrainer.train(), the
    # full 5-model pipeline), with confusion matrices,
    # classification reports, ROC-AUC, and feature importance.
    # Not currently called by framework.py.
    # =======================================================

    def evaluate_all(
        self,
        training_result: TrainingResult,
    ) -> EvaluationResult:
        """
        Evaluate all trained models.
        """

        logger.info(
            "Beginning model evaluation."
        )

        self._validate_training_result(
            training_result
        )

        result = EvaluationResult()

        for (

            model_name,

            model,

        ) in training_result.trained_models.items():

            self._evaluate_model(

                model_name,

                model,

                training_result,

                result,

            )

        result.comparison_table = (
            self._build_comparison_table(
                training_result,
                result,
            )
        )

        (

            result.best_model_name,

            result.best_model,

        ) = self._select_best_model(

            training_result,

            result,

        )

        logger.info(
            "Evaluation complete."
        )

        return result

    # =======================================================
    # EVALUATE SINGLE MODEL (used by evaluate_all's loop)
    # =======================================================

    def _evaluate_model(

        self,

        model_name: str,

        model,

        training_result: TrainingResult,

        evaluation_result: EvaluationResult,

    ):

        logger.info(
            "Evaluating %s",
            model_name,
        )

        X_test = training_result.X_test

        y_test = training_result.y_test

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        cm = confusion_matrix(
            y_test,
            predictions,
        )

        report = classification_report(

            y_test,

            predictions,

            output_dict=True,

            zero_division=0,

        )

        evaluation_result.metrics[
            model_name
        ] = {

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

        }

        evaluation_result.confusion_matrices[
            model_name
        ] = cm

        evaluation_result.classification_reports[
            model_name
        ] = report

        self._extract_feature_importance(

            model_name,

            model,

            training_result,

            evaluation_result,

        )

    # =======================================================
    # BUILD COMPARISON TABLE
    # =======================================================

    def _build_comparison_table(
        self,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
    ) -> pd.DataFrame:
        """
        Build a comparison table for all models.
        """

        rows = []

        for model_name in training_result.trained_models:

            metrics = evaluation_result.metrics[
                model_name
            ]

            cv = training_result.cross_validation[
                model_name
            ]

            rows.append({

                "Model":
                    model_name,

                "Accuracy":
                    metrics["accuracy"],

                "Precision":
                    metrics["precision"],

                "Recall":
                    metrics["recall"],

                "F1 Score":
                    metrics["f1"],

                "CV Mean":
                    cv["mean_accuracy"],

                "CV Std":
                    cv["std_accuracy"],

                "Training Time":
                    training_result.training_times[
                        model_name
                    ]

            })

        comparison = pd.DataFrame(rows)

        comparison = comparison.sort_values(

            by="Accuracy",

            ascending=False,

        )

        comparison.reset_index(

            drop=True,

            inplace=True,

        )

        logger.info(
            "Comparison table generated."
        )

        return comparison

    # =======================================================
    # BEST MODEL
    # =======================================================

    def _select_best_model(
        self,
        training_result: TrainingResult,
        evaluation_result: EvaluationResult,
    ):
        """
        Select the best-performing model.
        """

        table = evaluation_result.comparison_table

        winner = table.iloc[0]["Model"]

        logger.info(

            "Best model: %s",

            winner,

        )

        return (

            winner,

            training_result.trained_models[
                winner
            ],

        )

    # =======================================================
    # MODEL METRICS
    # =======================================================

    def get_metrics(
        self,
        evaluation_result: EvaluationResult,
        model_name: str,
    ) -> Dict:
        """
        Return metrics for one model.
        """

        return evaluation_result.metrics[
            model_name
        ]

    # =======================================================
    # CLASSIFICATION REPORT
    # =======================================================

    def get_classification_report(
        self,
        evaluation_result: EvaluationResult,
        model_name: str,
    ) -> Dict:

        return evaluation_result.classification_reports[
            model_name
        ]

    # =======================================================
    # CONFUSION MATRIX
    # =======================================================

    def get_confusion_matrix(
        self,
        evaluation_result: EvaluationResult,
        model_name: str,
    ) -> np.ndarray:

        return evaluation_result.confusion_matrices[
            model_name
        ]

    # =======================================================
    # PROBABILITY PREDICTIONS
    # =======================================================

    def _predict_probabilities(
        self,
        model,
        X_test,
    ):
        """
        Return probability predictions when supported.
        """

        if hasattr(
            model,
            "predict_proba",
        ):

            return model.predict_proba(
                X_test
            )

        return None

    # =======================================================
    # ROC AUC
    # =======================================================

    def _compute_roc_auc(
        self,
        model_name,
        model,
        training_result,
        evaluation_result,
    ):
        """
        Compute multiclass ROC-AUC.
        """

        probabilities = self._predict_probabilities(

            model,

            training_result.X_test,

        )

        if probabilities is None:

            return

        y_true = label_binarize(

            training_result.y_test,

            classes=np.unique(
                training_result.y_train
            ),

        )

        # Binary classification (2 classes) needs special
        # handling: label_binarize collapses to a single
        # column, but roc_auc_score's multi_class="ovr" path
        # expects one column per class.
        if y_true.shape[1] == 1:

            y_true = np.hstack(
                [1 - y_true, y_true]
            )

        score = roc_auc_score(

            y_true,

            probabilities,

            multi_class="ovr",

            average="weighted",

        )

        evaluation_result.probability_predictions[
            model_name
        ] = probabilities

        evaluation_result.roc_auc_scores[
            model_name
        ] = score

    # =======================================================
    # FEATURE IMPORTANCE
    # =======================================================

    def _extract_feature_importance(
        self,
        model_name,
        model,
        training_result,
        evaluation_result,
    ):
        """
        Extract feature importance.
        """

        feature_names = (
            training_result.feature_names
        )

        estimator = model

        if hasattr(
            model,
            "named_steps",
        ):

            estimator = model.named_steps[
                "classifier"
            ]

        importance = None

        if hasattr(
            estimator,
            "feature_importances_",
        ):

            importance = (
                estimator.feature_importances_
            )

        elif hasattr(
            estimator,
            "coef_",
        ):

            importance = np.mean(

                np.abs(
                    estimator.coef_
                ),

                axis=0,

            )

        else:

            permutation = permutation_importance(

                estimator,

                training_result.X_test,

                training_result.y_test,

                random_state=42,

                n_repeats=10,

            )

            importance = (
                permutation.importances_mean
            )

        dataframe = pd.DataFrame({

            "Feature":
                feature_names,

            "Importance":
                importance,

        })

        dataframe = dataframe.sort_values(

            by="Importance",

            ascending=False,

        )

        dataframe.reset_index(

            drop=True,

            inplace=True,

        )

        evaluation_result.feature_importance[
            model_name
        ] = dataframe

        self._compute_roc_auc(

            model_name,

            model,

            training_result,

            evaluation_result,

        )

    # =======================================================
    # VALIDATION
    # =======================================================

    @staticmethod
    def _validate_training_result(
        training_result: TrainingResult,
    ) -> None:
        """
        Validate training result.
        """

        if training_result is None:

            raise ValueError(
                "TrainingResult cannot be None."
            )

        if not training_result.trained_models:

            raise ValueError(
                "No trained models available."
            )

        if training_result.X_test is None:

            raise ValueError(
                "Testing dataset missing."
            )

        if training_result.y_test is None:

            raise ValueError(
                "Testing labels missing."
            )

    # =======================================================
    # SUMMARY
    # =======================================================

    def evaluation_summary(
        self,
        evaluation_result: EvaluationResult,
    ) -> pd.DataFrame:
        """
        Return comparison table.
        """

        return evaluation_result.comparison_table.copy()

    # =======================================================
    # EXPORT
    # =======================================================

    def export_results(
        self,
        evaluation_result: EvaluationResult,
        filepath: str,
    ) -> None:
        """
        Export evaluation summary.
        """

        evaluation_result.comparison_table.to_csv(

            filepath,

            index=False,

        )

        logger.info(

            "Evaluation exported to %s",

            filepath,

        )

    # =======================================================
    # BEST MODEL METRICS
    # =======================================================

    def best_model_metrics(
        self,
        evaluation_result: EvaluationResult,
    ) -> Dict:

        return evaluation_result.metrics[
            evaluation_result.best_model_name
        ]

    # =======================================================
    # FEATURE IMPORTANCE
    # =======================================================

    def get_feature_importance(
        self,
        evaluation_result: EvaluationResult,
        model_name: str,
    ) -> pd.DataFrame:

        return evaluation_result.feature_importance[
            model_name
        ]
