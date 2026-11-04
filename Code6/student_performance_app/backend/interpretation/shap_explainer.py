"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/interpretation/shap_explainer.py

Purpose:
    Instance-level explanations for a single student's prediction.

Why this exists
---------------
The "Explainable AI" page in v2.x showed the globally most
important features paired with the student's values. That answers
"what matters to the model overall", not "why did THIS student get
THIS label" -- the second question is the one a lecturer actually
asks before acting.

This module answers the second question with SHAP: each feature gets
a signed contribution towards (or against) the predicted class for
this one student.

Design notes
------------
* SHAP is an OPTIONAL dependency. If it is not installed, or the
  model type is unsupported, explain() returns None and the caller
  falls back to the global-importance view. Nothing crashes.
* Pipelines are handled by transforming the input through the
  pipeline's preprocessing steps before explaining the final
  estimator.
* Contributions are reported in the model's own units (log-odds
  for linear models, probability-ish for trees). We deliberately do
  not pretend they are comparable across model types; the UI labels
  the method used.

Version:
    1.0
====================================================================
"""

from __future__ import annotations

import logging

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    SHAP_AVAILABLE = False

logger.info(
    "SHAP %s.",
    "available" if SHAP_AVAILABLE else "NOT installed (instance-level "
    "explanations disabled; install 'shap' to enable)",
)


DISPLAY_NAMES = {
    "attendance_rate": "Attendance Rate",
    "study_hours": "Study Hours",
    "sleep_hours": "Sleep Hours",
    "social_media_hours": "Social Media Hours",
    "assignment_score": "Assignment Score",
    "test_score": "Test Score",
    "exam_score": "Examination Score",
}


class ShapExplainer:
    """
    Computes per-student SHAP contributions for a trained model.
    """

    def __init__(self) -> None:

        self._explainers: Dict[int, Any] = {}

    # -------------------------------------------------------

    @property
    def available(self) -> bool:
        return SHAP_AVAILABLE

    # -------------------------------------------------------

    @staticmethod
    def _final_estimator(model: Any) -> Any:
        """
        Unwrap a Pipeline to its final estimator.
        """

        if isinstance(model, Pipeline):
            return model.steps[-1][1]

        if hasattr(model, "steps") and model.steps:
            return model.steps[-1][1]

        # VotingClassifier: explain the first branch, which is
        # the most interpretable (linear) member.
        if hasattr(model, "estimators_") and getattr(
            model, "estimators_", None
        ):
            return ShapExplainer._final_estimator(
                model.estimators_[0]
            )

        return model

    # -------------------------------------------------------

    @staticmethod
    def _transform_input(
        model: Any,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the row through any preprocessing steps that sit in
        front of the final estimator, so SHAP sees the same
        numbers the estimator sees.
        """

        if isinstance(model, Pipeline):

            transformed = features

            for name, step in model.steps[:-1]:

                if hasattr(step, "transform"):
                    transformed = step.transform(transformed)

            if not isinstance(transformed, pd.DataFrame):

                transformed = pd.DataFrame(
                    transformed,
                    columns=list(features.columns),
                )

            return transformed

        return features

    # -------------------------------------------------------

    def _explainer_for(
        self,
        estimator: Any,
        background: pd.DataFrame,
    ) -> Optional[Any]:
        """
        Build (and cache) the right explainer for the estimator.

        `background` should be a sample of training rows: SHAP
        compares this student against the population, so a
        one-row background would make every contribution zero.
        """

        key = (id(estimator), len(background))

        if key in self._explainers:
            return self._explainers[key]

        explainer = None

        try:

            if hasattr(estimator, "coef_"):

                masker = shap.maskers.Independent(background)

                explainer = shap.LinearExplainer(
                    estimator,
                    masker,
                )

            elif hasattr(estimator, "feature_importances_"):

                explainer = shap.TreeExplainer(estimator)

            else:

                explainer = shap.Explainer(
                    estimator.predict_proba,
                    background,
                )

        except Exception as error:  # pragma: no cover - defensive
            logger.warning(
                "Could not build SHAP explainer for %s: %s",
                type(estimator).__name__,
                error,
            )
            return None

        self._explainers[key] = explainer

        return explainer

    # -------------------------------------------------------

    def explain(
        self,
        model: Any,
        student_frame: pd.DataFrame,
        predicted_index: Optional[int] = None,
        top_n: int = 5,
        background: Optional[pd.DataFrame] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Explain one prediction.

        Parameters
        ----------
        model:
            Fitted estimator (or Pipeline, or VotingClassifier).

        student_frame:
            One-row DataFrame of raw feature values, in the
            model's expected column order.

        predicted_index:
            Index of the predicted class, used to pick the right
            slice of a multiclass SHAP output.

        top_n:
            How many contributions to return.

        Returns
        -------
        list of dict or None
            [{"feature", "display_name", "value", "contribution",
              "direction", "method"}, ...] sorted by |contribution|,
            or None when SHAP cannot explain this model.
        """

        if not SHAP_AVAILABLE:
            return None

        try:

            estimator = self._final_estimator(model)

            transformed = self._transform_input(model, student_frame)

            if background is None or len(background) < 2:

                # Degenerate: explain relative to the origin.
                background = transformed

            else:

                background = self._transform_input(
                    model, background
                )

            explainer = self._explainer_for(estimator, background)

            if explainer is None:
                return None

            values = explainer.shap_values(transformed)

            values = self._select_class(
                values,
                predicted_index,
            )

            if values is None:
                return None

            values = np.asarray(values).ravel()

            columns = list(student_frame.columns)

            if values.shape[0] != len(columns):
                logger.debug(
                    "SHAP shape %s does not match %d features.",
                    values.shape,
                    len(columns),
                )
                return None

            contributions = []

            for column, value, contribution in zip(
                columns,
                student_frame.iloc[0].tolist(),
                values.tolist(),
            ):

                contributions.append(
                    {
                        "feature": column,
                        "display_name": DISPLAY_NAMES.get(
                            column, column.replace("_", " ").title()
                        ),
                        "value": value,
                        "contribution": float(contribution),
                        "direction": (
                            "Supports the prediction"
                            if contribution >= 0
                            else "Argues against the prediction"
                        ),
                        "method": "SHAP",
                    }
                )

            contributions.sort(
                key=lambda item: abs(item["contribution"]),
                reverse=True,
            )

            return contributions[:top_n]

        except Exception as error:  # pragma: no cover - defensive
            logger.warning(
                "SHAP explanation failed (%s); falling back to "
                "global importance.",
                error,
            )
            return None

    # -------------------------------------------------------

    @staticmethod
    def _select_class(values: Any, predicted_index: Optional[int]):
        """
        Normalise the many shapes shap_values() can return:

            list of arrays (one per class)
            3-D array (samples, features, classes)
            2-D array (samples, features)
        """

        if isinstance(values, list):

            if predicted_index is not None and 0 <= predicted_index < len(values):
                return values[predicted_index]

            return values[0]

        array = np.asarray(values)

        if array.ndim == 3:

            index = (
                predicted_index
                if predicted_index is not None
                and predicted_index < array.shape[2]
                else 0
            )

            return array[0, :, index]

        if array.ndim == 2:

            return array[0]

        return None
