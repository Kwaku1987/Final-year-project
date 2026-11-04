"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/cpi/cpi_engine.py

Purpose:
    Build the supervised-learning target for the dataset, either as

        "exam_band"  -- the end-of-term examination band (default), or
        "cpi"        -- the Composite Performance Index band (legacy).

    Both definitions are re-expressed as Low / Average / High using a
    configurable labelling policy (quantile thirds by default, or the
    original fixed 60/80 cut-offs).

Notes
-----
Version 2.x computed the CPI and then trained models to predict it
from the same columns the CPI is made of, which made the task
circular (see docs/METHODOLOGY.md). The exam_band definition fixes
that: the exam score becomes the label and is excluded from the
predictor set by backend/config.py.

This module no longer prints to stdout. Everything is logged, and
the thresholds actually used are stored on the instance
(self.thresholds) so they can be written into the model bundle and
surfaced in the UI.

Version:
    3.0
====================================================================
"""

from __future__ import annotations

import logging

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from backend.config import (
    CPI_CONFIG,
    LABEL_POLICY,
    PERFORMANCE_LABELS,
    TARGET_COLUMN,
    TARGET_DEFINITION,
    TARGET_SOURCE_COLUMN,
)

logger = logging.getLogger(__name__)


# ============================================================
# CPI ENGINE
# ============================================================

class CPIEngine:
    """
    Computes the Composite Performance Index and the
    Low / Average / High target labels.
    """

    def __init__(
        self,
        config=None,
        label_policy: Optional[dict] = None,
        target_definition: Optional[str] = None,
    ):
        self.config = config or CPI_CONFIG

        self.weights = self.config["weights"]

        self.label_policy = dict(
            label_policy or LABEL_POLICY
        )

        self.target_definition = (
            target_definition or TARGET_DEFINITION
        )

        # Filled in by build_target_variable().
        self.thresholds: Optional[Tuple[float, float]] = None

        self.threshold_method: Optional[str] = None

        self.label_distribution: Optional[Dict[str, int]] = None

    # -------------------------------------------------------
    # NORMALISATION HELPERS
    # -------------------------------------------------------

    def normalize_positive(self, value, maximum):
        score = (value / maximum) * 100
        return max(0, min(score, 100))

    # -------------------------------------------------------

    def normalize_negative(self, value, maximum):
        score = (1 - (value / maximum)) * 100
        return max(0, min(score, 100))

    # -------------------------------------------------------
    # COMPOSITE PERFORMANCE INDEX
    # -------------------------------------------------------

    def calculate_cpi(self, row):
        study_score = self.normalize_positive(
            row["study_hours"],
            self.config["study_hours_max"]
        )
        sleep_score = self.normalize_positive(
            row["sleep_hours"],
            self.config["sleep_hours_max"]
        )
        social_score = self.normalize_negative(
            row["social_media_hours"],
            self.config["social_media_hours_max"]
        )
        cpi = (
            self.weights["exam_score"] * row["exam_score"]
            + self.weights["test_score"] * row["test_score"]
            + self.weights["assignment_score"] * row["assignment_score"]
            + self.weights["attendance_rate"] * row["attendance_rate"]
            + self.weights["study_score"] * study_score
            + self.weights["sleep_score"] * sleep_score
            + self.weights["social_score"] * social_score
        )
        return round(cpi, 2)

    # -------------------------------------------------------
    # LABELLING
    # -------------------------------------------------------

    def _resolve_thresholds(
        self,
        values: pd.Series,
    ) -> Tuple[float, float]:
        """
        Resolve the Low/High cut-offs for this dataset.

        quantile -> thirds of the observed distribution, so the
                    three classes are balanced and every class is
                    learnable and measurable.
        fixed    -> the original 60 / 80 cut-offs.
        """

        method = self.label_policy.get("method", "quantile")

        if method == "fixed":

            low, high = self.label_policy.get(
                "fixed_thresholds",
                (60.0, 80.0),
            )

            self.threshold_method = "fixed"

            return float(low), float(high)

        low_q = float(
            self.label_policy.get("low_quantile", 1 / 3)
        )

        high_q = float(
            self.label_policy.get("high_quantile", 2 / 3)
        )

        low = float(values.quantile(low_q))

        high = float(values.quantile(high_q))

        if high <= low:
            # Degenerate distribution -- fall back to the fixed
            # cut-offs rather than emitting a useless label set.
            logger.warning(
                "Quantile thresholds collapsed (low=%.2f, "
                "high=%.2f); falling back to fixed thresholds.",
                low,
                high,
            )

            self.threshold_method = "fixed"

            return (
                float(
                    self.label_policy.get(
                        "fixed_thresholds",
                        (60.0, 80.0),
                    )[0]
                ),
                float(
                    self.label_policy.get(
                        "fixed_thresholds",
                        (60.0, 80.0),
                    )[1]
                ),
            )

        self.threshold_method = "quantile"

        return round(low, 2), round(high, 2)

    # -------------------------------------------------------

    def classify_performance(
        self,
        value,
        thresholds: Optional[Tuple[float, float]] = None,
    ) -> str:
        """
        Map a numeric index (CPI or exam score) to a label using
        the thresholds resolved for this dataset.
        """

        if thresholds is None:

            thresholds = self.thresholds or tuple(
                self.label_policy.get(
                    "fixed_thresholds",
                    (60.0, 80.0),
                )
            )

        low, high = thresholds

        if value >= high:
            return "High"

        if value >= low:
            return "Average"

        return "Low"

    # -------------------------------------------------------
    # TARGET VARIABLE
    # -------------------------------------------------------

    def build_target_variable(
        self,
        df: pd.DataFrame,
        definition: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add the performance_level column to the dataframe.

        Parameters
        ----------
        df:
            Validated dataset containing the required columns.

        definition:
            "exam_band" or "cpi". Defaults to the engine's
            configured definition.
        """

        definition = definition or self.target_definition

        df = df.copy()

        source_column = TARGET_SOURCE_COLUMN.get(definition)

        if definition == "cpi":

            df["CPI"] = df.apply(
                self.calculate_cpi,
                axis=1,
            )

            index_values = df["CPI"]

        elif definition == "exam_band":

            if source_column not in df.columns:

                raise ValueError(
                    f"Target definition {definition!r} needs the "
                    f"column {source_column!r}, which is missing."
                )

            index_values = df[source_column].astype(float)

        else:

            raise ValueError(
                f"Unknown target definition {definition!r}. "
                "Use 'exam_band' or 'cpi'."
            )

        self.target_definition = definition

        self.thresholds = self._resolve_thresholds(index_values)

        df[TARGET_COLUMN] = index_values.apply(
            lambda value: self.classify_performance(
                value,
                self.thresholds,
            )
        )

        self.label_distribution = {
            str(label): int(count)
            for label, count in df[TARGET_COLUMN]
            .value_counts()
            .items()
        }

        # Guard against a degenerate label set: with heavy ties
        # (e.g. scores clipped at 100) quantile thresholds can
        # still collapse onto one class. Say so instead of
        # silently training on a lopsided target.
        total = sum(self.label_distribution.values()) or 1

        self.balanced = all(
            count / total <= 0.60
            for count in self.label_distribution.values()
        )

        if not self.balanced:

            logger.warning(
                "Class distribution is not balanced (%s). "
                "Headline accuracy will be misleading -- read "
                "macro F1 and balanced accuracy instead, or "
                "revisit the score scale / thresholds.",
                self.label_distribution,
            )

        logger.info(
            "Target built (definition=%s, method=%s, "
            "thresholds=%.2f/%.2f): %s",
            definition,
            self.threshold_method,
            self.thresholds[0],
            self.thresholds[1],
            self.label_distribution,
        )

        return df

    # Backwards-compatible alias (older framework builds call it).
    generate_target_variable = build_target_variable

    # -------------------------------------------------------
    # REPORTING
    # -------------------------------------------------------

    def describe_target(self) -> Dict[str, object]:
        """
        Machine-readable description of the target that was
        built -- stored in the model bundle and shown in the UI.
        """

        low, high = self.thresholds or (None, None)

        return {
            "definition": self.target_definition,
            "method": self.threshold_method,
            "low_threshold": low,
            "high_threshold": high,
            "distribution": self.label_distribution,
            "balanced": getattr(self, "balanced", None),
            "labels": list(PERFORMANCE_LABELS.values()),
        }

    # -------------------------------------------------------

    def ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce the required columns to float, raising a clear
        error naming the offending column.
        """

        required = list(self.weights.keys() - {"study_score", "sleep_score", "social_score"})

        for column in required:

            if column not in df.columns:

                raise ValueError(
                    f"Missing column required for CPI: {column}"
                )

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        return df


# ============================================================

def build_quantile_thresholds(
    values: pd.Series,
    low_quantile: float = 1 / 3,
    high_quantile: float = 2 / 3,
) -> Tuple[float, float]:

    return (
        float(np.quantile(values, low_quantile)),
        float(np.quantile(values, high_quantile)),
    )
