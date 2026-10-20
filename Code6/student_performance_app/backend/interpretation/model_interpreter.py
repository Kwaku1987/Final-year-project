"""
Student Performance Assessment System
=====================================

Module:
    backend/interpretation/model_interpreter.py

Purpose:
    Interpret trained machine-learning models and convert their
    technical outputs into understandable educational insights.

Supported model types:
    - Logistic Regression
    - Random Forest
    - Other estimators exposing coef_ or feature_importances_

Version:
    4.1 - Frozen Backend
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class FeatureInsight:
    """
    Represents the interpretation of one input feature.

    Attributes:
        feature:
            Name of the feature.

        importance:
            Relative importance score.

        rank:
            Position of the feature after sorting by importance.

        direction:
            Positive, Negative, Mixed, or Not Available.

        interpretation:
            Human-readable explanation.
    """

    feature: str
    importance: float
    rank: int
    direction: str
    interpretation: str


@dataclass
class InterpretationReport:
    """
    Complete interpretation output generated for a model.
    """

    model_name: str

    model_type: str

    feature_importance: pd.DataFrame

    top_features: List[FeatureInsight]

    key_insights: List[str]

    recommendations: List[str]

    executive_summary: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the report into a serializable dictionary.
        """

        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "feature_importance": (
                self.feature_importance.to_dict(
                    orient="records"
                )
            ),
            "top_features": [
                asdict(item)
                for item in self.top_features
            ],
            "key_insights": self.key_insights,
            "recommendations": self.recommendations,
            "executive_summary": (
                self.executive_summary
            ),
            "metadata": self.metadata,
        }


# ============================================================
# MODEL INTERPRETER
# ============================================================

class ModelInterpreter:
    """
    Interprets trained machine-learning models.

    The class uses:

    - Absolute coefficients for Logistic Regression.
    - Feature importance values for Random Forest.
    - A generic fallback for compatible estimators.

    The output is an InterpretationReport containing:

    - Ranked feature importance
    - Feature-level explanations
    - Key insights
    - Educational recommendations
    - Executive summary
    """

    POSITIVE_FEATURES = {
        "attendance_rate",
        "study_hours",
        "sleep_hours",
        "assignment_score",
        "test_score",
        "exam_score",
    }

    NEGATIVE_FEATURES = {
        "time_on_social_media",
    }

    FEATURE_LABELS = {
        "attendance_rate": "Attendance Rate",
        "study_hours": "Study Hours",
        "sleep_hours": "Sleep Hours",
        "time_on_social_media": (
            "Time on Social Media"
        ),
        "assignment_score": "Assignment Score",
        "test_score": "Test Score",
        "exam_score": "Examination Score",
    }

    def __init__(
        self,
        top_n: int = 5,
        importance_threshold: float = 0.05,
    ) -> None:
        """
        Initialize the interpreter.

        Args:
            top_n:
                Maximum number of important features to include
                in the main interpretation.

            importance_threshold:
                Minimum normalized importance required for a
                feature to be described as influential.
        """

        if top_n < 1:
            raise ValueError(
                "top_n must be at least 1."
            )

        if not 0 <= importance_threshold <= 1:
            raise ValueError(
                "importance_threshold must be "
                "between 0 and 1."
            )

        self.top_n = top_n

        self.importance_threshold = (
            importance_threshold
        )

        logger.info(
            "ModelInterpreter initialized."
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def interpret(
        self,
        model: Any,
        feature_names: Sequence[str],
        model_name: Optional[str] = None,
        evaluation_results: Optional[
            Dict[str, Any]
        ] = None,
    ) -> InterpretationReport:
        """
        Generate a complete interpretation report.

        Args:
            model:
                A fitted machine-learning estimator.

            feature_names:
                Ordered names of the model input features.

            model_name:
                Optional display name.

            evaluation_results:
                Optional model evaluation metrics.

        Returns:
            A complete InterpretationReport.

        Raises:
            ValueError:
                If the model or feature names are invalid.
        """

        self._validate_inputs(
            model=model,
            feature_names=feature_names,
        )

        resolved_name = (
            model_name
            or model.__class__.__name__
        )

        model_type = (
            model.__class__.__name__
        )

        importance_table = (
            self.extract_feature_importance(
                model=model,
                feature_names=feature_names,
            )
        )

        top_features = (
            self._build_feature_insights(
                importance_table
            )
        )

        key_insights = (
            self._generate_key_insights(
                top_features=top_features,
                model_name=resolved_name,
            )
        )

        recommendations = (
            self._generate_recommendations(
                top_features=top_features
            )
        )

        executive_summary = (
            self._generate_executive_summary(
                model_name=resolved_name,
                top_features=top_features,
                evaluation_results=(
                    evaluation_results
                ),
            )
        )

        metadata = {
            "total_features": len(
                feature_names
            ),
            "top_features_returned": len(
                top_features
            ),
            "importance_method": (
                importance_table[
                    "importance_method"
                ].iloc[0]
            ),
        }

        logger.info(
            "Interpretation completed for %s.",
            resolved_name,
        )

        return InterpretationReport(
            model_name=resolved_name,
            model_type=model_type,
            feature_importance=(
                importance_table
            ),
            top_features=top_features,
            key_insights=key_insights,
            recommendations=recommendations,
            executive_summary=(
                executive_summary
            ),
            metadata=metadata,
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def extract_feature_importance(
        self,
        model: Any,
        feature_names: Sequence[str],
    ) -> pd.DataFrame:
        """
        Extract and normalize model feature importance.

        Logistic Regression:
            Uses mean absolute coefficient magnitude across
            all classes.

        Random Forest:
            Uses feature_importances_.

        Generic estimators:
            Use either coef_ or feature_importances_ when
            available.
        """

        names = list(feature_names)

        if hasattr(
            model,
            "feature_importances_"
        ):

            raw_importance = np.asarray(
                model.feature_importances_,
                dtype=float,
            )

            method = (
                "feature_importances"
            )

            directions = [
                "Not Available"
                for _ in names
            ]

        elif hasattr(model, "coef_"):

            coefficients = np.asarray(
                model.coef_,
                dtype=float,
            )

            if coefficients.ndim == 1:

                coefficient_values = (
                    coefficients
                )

                raw_importance = np.abs(
                    coefficient_values
                )

                directions = [
                    self._coefficient_direction(
                        value
                    )
                    for value in coefficient_values
                ]

            else:

                # Multiclass: sklearn's multinomial
                # (softmax) coefficients are constructed so
                # that, for every feature, the coefficients
                # across classes sum to ~0 -- an
                # identifiability artifact of the
                # parameterization, not a meaningful signal.
                # np.mean(coefficients, axis=0) therefore
                # collapses to floating-point noise near zero
                # for every feature, which silently corrupts
                # both the importance ranking and the
                # direction label. Taking the mean of the
                # absolute values instead (matching this
                # method's own docstring) preserves each
                # class's magnitude rather than letting
                # opposite-signed classes cancel out.
                raw_importance = np.mean(
                    np.abs(coefficients),
                    axis=0,
                )

                # Direction: for each feature, report the
                # sign of the coefficient in whichever class
                # that feature affects most strongly (largest
                # magnitude) -- the mean signed coefficient
                # isn't meaningful here for the same reason
                # above.
                dominant_class = np.argmax(
                    np.abs(coefficients),
                    axis=0,
                )

                directions = [
                    self._coefficient_direction(
                        coefficients[
                            dominant_class[index],
                            index,
                        ]
                    )
                    for index in range(
                        coefficients.shape[1]
                    )
                ]

            method = (
                "absolute_coefficients"
            )

        else:

            raise ValueError(
                "The model does not expose "
                "'coef_' or 'feature_importances_'."
            )

        if len(raw_importance) != len(names):

            raise ValueError(
                "The number of extracted importance "
                "values does not match the number of "
                "feature names."
            )

        total = float(
            np.sum(raw_importance)
        )

        if total > 0:

            normalized = (
                raw_importance / total
            )

        else:

            normalized = np.zeros(
                len(raw_importance)
            )

        table = pd.DataFrame(
            {
                "feature": names,
                "display_name": [
                    self._display_name(
                        name
                    )
                    for name in names
                ],
                "raw_importance": (
                    raw_importance
                ),
                "importance": normalized,
                "direction": directions,
                "importance_method": (
                    method
                ),
            }
        )

        table = (
            table
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        table.insert(
            0,
            "rank",
            range(
                1,
                len(table) + 1,
            ),
        )

        return table

    # ========================================================
    # INSIGHTS
    # ========================================================

    def _build_feature_insights(
        self,
        importance_table: pd.DataFrame,
    ) -> List[FeatureInsight]:
        """
        Convert the highest-ranked rows into feature insights.
        """

        selected = (
            importance_table
            .head(self.top_n)
            .copy()
        )

        insights: List[
            FeatureInsight
        ] = []

        for _, row in selected.iterrows():

            feature = str(
                row["feature"]
            )

            importance = float(
                row["importance"]
            )

            direction = str(
                row["direction"]
            )

            interpretation = (
                self._describe_feature(
                    feature=feature,
                    importance=importance,
                    direction=direction,
                )
            )

            insights.append(
                FeatureInsight(
                    feature=feature,
                    importance=importance,
                    rank=int(
                        row["rank"]
                    ),
                    direction=direction,
                    interpretation=(
                        interpretation
                    ),
                )
            )

        return insights

    def _generate_key_insights(
        self,
        top_features: List[
            FeatureInsight
        ],
        model_name: str,
    ) -> List[str]:
        """
        Generate concise findings for educators.
        """

        insights: List[str] = []

        if not top_features:

            return [
                "No feature importance "
                "information was available."
            ]

        most_important = (
            top_features[0]
        )

        insights.append(
            f"{self._display_name(most_important.feature)} "
            f"was the most influential predictor in "
            f"the {model_name} model."
        )

        positive_features = [
            item
            for item in top_features
            if item.feature
            in self.POSITIVE_FEATURES
        ]

        negative_features = [
            item
            for item in top_features
            if item.feature
            in self.NEGATIVE_FEATURES
        ]

        if positive_features:

            names = ", ".join(
                self._display_name(
                    item.feature
                )
                for item in positive_features[
                    :3
                ]
            )

            insights.append(
                f"Academic and learning-related factors "
                f"such as {names} contributed strongly "
                f"to the model's predictions."
            )

        if negative_features:

            names = ", ".join(
                self._display_name(
                    item.feature
                )
                for item in negative_features
            )

            insights.append(
                f"{names} was identified as a relevant "
                f"lifestyle factor and should be monitored "
                f"alongside academic indicators."
            )

        insights.append(
            "Feature importance indicates the relative "
            "influence of variables within the model; it "
            "does not establish causation."
        )

        return insights

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    def _generate_recommendations(
        self,
        top_features: List[
            FeatureInsight
        ],
    ) -> List[str]:
        """
        Generate educational recommendations from important
        features.
        """

        recommendations: List[str] = []

        features = {
            item.feature
            for item in top_features
        }

        if "attendance_rate" in features:

            recommendations.append(
                "Monitor attendance regularly and provide "
                "early support for students with declining "
                "attendance."
            )

        if "study_hours" in features:

            recommendations.append(
                "Encourage consistent study schedules and "
                "provide structured study support where "
                "necessary."
            )

        if "assignment_score" in features:

            recommendations.append(
                "Use assignment performance to identify "
                "learning gaps before major examinations."
            )

        if "test_score" in features:

            recommendations.append(
                "Provide targeted revision and feedback "
                "for students with weak test performance."
            )

        if "exam_score" in features:

            recommendations.append(
                "Use examination results together with "
                "continuous assessment to guide academic "
                "interventions."
            )

        if "sleep_hours" in features:

            recommendations.append(
                "Promote healthy sleep routines as part "
                "of balanced study and wellbeing practices."
            )

        if (
            "time_on_social_media"
            in features
        ):

            recommendations.append(
                "Encourage balanced and purposeful social "
                "media use so that excessive use does not "
                "displace study time."
            )

        if not recommendations:

            recommendations.append(
                "Review the highest-ranked predictors and "
                "use them with educator judgment when "
                "planning student support."
            )

        return recommendations

    # ========================================================
    # EXECUTIVE SUMMARY
    # ========================================================

    def _generate_executive_summary(
        self,
        model_name: str,
        top_features: List[
            FeatureInsight
        ],
        evaluation_results: Optional[
            Dict[str, Any]
        ],
    ) -> str:
        """
        Generate a concise management-level summary.
        """

        if top_features:

            feature_names = ", ".join(
                self._display_name(
                    item.feature
                )
                for item in top_features[
                    :3
                ]
            )

            feature_sentence = (
                f"The most influential predictors were "
                f"{feature_names}."
            )

        else:

            feature_sentence = (
                "No ranked feature information was "
                "available."
            )

        metric_sentence = ""

        if evaluation_results:

            accuracy = (
                evaluation_results.get(
                    "accuracy"
                )
            )

            f1_score = (
                evaluation_results.get(
                    "f1_score"
                )
                or evaluation_results.get(
                    "f1"
                )
            )

            metrics: List[str] = []

            if accuracy is not None:

                metrics.append(
                    f"accuracy of "
                    f"{float(accuracy):.2%}"
                )

            if f1_score is not None:

                metrics.append(
                    f"F1-score of "
                    f"{float(f1_score):.2%}"
                )

            if metrics:

                metric_sentence = (
                    " The model achieved "
                    + " and ".join(metrics)
                    + "."
                )

        return (
            f"The {model_name} model was interpreted to "
            f"identify the factors most strongly associated "
            f"with predicted student performance."
            f"{metric_sentence} "
            f"{feature_sentence} "
            f"These findings should support early "
            f"intervention and educator decision-making, "
            f"not replace professional judgment."
        )

    # ========================================================
    # FEATURE DESCRIPTIONS
    # ========================================================

    def _describe_feature(
        self,
        feature: str,
        importance: float,
        direction: str,
    ) -> str:
        """
        Produce a feature-specific explanation.
        """

        name = self._display_name(
            feature
        )

        percentage = (
            importance * 100
        )

        influence = (
            "highly influential"
            if importance
            >= self.importance_threshold
            else "a supporting predictor"
        )

        if direction == "Positive":

            direction_text = (
                "Higher values were generally associated "
                "with stronger predicted performance."
            )

        elif direction == "Negative":

            direction_text = (
                "Higher values were generally associated "
                "with weaker predicted performance."
            )

        elif feature in self.POSITIVE_FEATURES:

            direction_text = (
                "This factor is academically beneficial "
                "when maintained at an appropriate level."
            )

        elif feature in self.NEGATIVE_FEATURES:

            direction_text = (
                "This lifestyle factor should be balanced "
                "with academic responsibilities."
            )

        else:

            direction_text = (
                "The model identified this variable as "
                "relevant, but a single direction was not "
                "available."
            )

        return (
            f"{name} accounted for approximately "
            f"{percentage:.1f}% of the model's normalized "
            f"feature importance and was {influence}. "
            f"{direction_text}"
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _coefficient_direction(
        coefficient: float,
    ) -> str:
        """
        Determine the sign of a coefficient.
        """

        if coefficient > 0:

            return "Positive"

        if coefficient < 0:

            return "Negative"

        return "Neutral"

    def _display_name(
        self,
        feature: str,
    ) -> str:
        """
        Convert a technical feature name into a display label.
        """

        if feature in self.FEATURE_LABELS:

            return self.FEATURE_LABELS[
                feature
            ]

        return (
            feature
            .replace("_", " ")
            .title()
        )

    @staticmethod
    def _validate_inputs(
        model: Any,
        feature_names: Sequence[str],
    ) -> None:
        """
        Validate interpretation inputs.
        """

        if model is None:

            raise ValueError(
                "A trained model is required."
            )

        if not feature_names:

            raise ValueError(
                "At least one feature name is required."
            )