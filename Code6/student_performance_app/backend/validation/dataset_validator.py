"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/validation/dataset_validator.py

Purpose:
    Validate the raw student dataset before target generation and
    model training, and normalise score scales so the CPI weights
    and the models see consistent units.

Notes
-----
Version 3.0 changes:

* No more print() to stdout. Everything is logged, and every
  check appends a structured message to self.warnings /
  self.report so the dashboard can display validation findings
  to the user instead of hiding them in the server console.

* normalize_score_scales() applies SCORE_STALE_RULES from
  backend/config.py. assignment_score arrives on a 0-20 scale
  while its CPI weight assumes 0-100, which quietly cut its real
  contribution from 15 points to ~1.5. Any column whose observed
  maximum is at or below the rule's detect_max is rescaled to a
  percentage and a warning is recorded.

* validate_target_labels() still only runs when the target column
  is already present; the normal pipeline validates before the
  target exists, so it mainly guards re-validation.

Version:
    3.0
====================================================================
"""

from __future__ import annotations

import logging

from typing import Dict, List

import pandas as pd

from backend.config import (
    COLUMN_MAPPING,
    EXPECTED_LABELS,
    FEATURE_COLUMNS,
    SCORE_SCALE_RULES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


# ============================================================
# DATASET VALIDATOR CLASS
# ============================================================

class DatasetValidator:
    """
    Validates a student dataset before preprocessing and
    machine learning.
    """

    def __init__(self,
                 feature_columns=FEATURE_COLUMNS,
                 target_column=TARGET_COLUMN,
                 expected_labels=EXPECTED_LABELS):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.expected_labels = expected_labels

        # Populated by validate().
        self.warnings: List[str] = []
        self.report: Dict[str, object] = {}
        self.rescaled_columns: Dict[str, Dict[str, float]] = {}

    # -------------------------------------------------------

    def _warn(self, message: str) -> None:
        logger.warning(message)
        if message not in self.warnings:
            self.warnings.append(message)

    # -------------------------------------------------------

    def _info(self, message: str, *args) -> None:
        logger.info(message, *args)

    # -------------------------------------------------------

    def standardize_columns(self, df):
        """
        Rename known alternative column names to the project's
        standard schema.

        Skips a rename if the target column name already exists in
        the DataFrame, to avoid silently producing two columns with
        the same name (which would break any downstream
        df[new_name] lookup by returning a DataFrame instead of a
        Series).
        """
        renamed = {}
        skipped = {}
        for old_name, new_name in COLUMN_MAPPING.items():
            if old_name not in df.columns:
                continue
            if new_name in df.columns:
                skipped[old_name] = new_name
                continue
            df = df.rename(columns={old_name: new_name})
            renamed[old_name] = new_name

        if renamed:
            self._info(f"Renamed columns: {renamed}")

        if skipped:
            self._warn(
                "Skipped renaming because both names are present "
                f"(check your data): {skipped}"
            )

        self.report["renamed_columns"] = renamed
        self.report["skipped_renames"] = skipped

        return df

    # -------------------------------------------------------

    def normalize_score_scales(self, df):
        """
        Rescale columns that are not on a 0-100 scale.

        A column is rescaled when its observed maximum is at or
        below the rule's detect_max (see SCORE_SCALE_RULES).
        """

        if not SCORE_SCALE_RULES:
            return df

        for column, (detect_max, underlying_max) in (
            SCORE_SCALE_RULES.items()
        ):
            if column not in df.columns:
                continue

            series = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            observed_max = series.max()

            if pd.isna(observed_max):
                continue

            if observed_max <= detect_max and observed_max > 0:

                factor = 100.0 / underlying_max

                df[column] = (series * factor).clip(0, 100)

                self.rescaled_columns[column] = {
                    "observed_max": float(observed_max),
                    "assumed_scale": float(underlying_max),
                    "factor": float(factor),
                }

                self._warn(
                    f"{column} appears to be scored out of "
                    f"{underlying_max:g} (max observed "
                    f"{observed_max:g}) but the project expects "
                    f"0-100. Rescaled by {factor:g}x so its weight "
                    "applies correctly."
                )

        self.report["rescaled_columns"] = self.rescaled_columns

        return df

    # -------------------------------------------------------

    def validate_columns(self, df):
        required = self.feature_columns
        missing = [
            col for col in required
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )
        self._info("Required columns validated.")

    # -------------------------------------------------------

    def validate_missing_values(self, df):
        total_missing = int(df.isnull().sum().sum())

        per_column = {
            column: int(count)
            for column, count in df.isnull().sum().items()
            if count
        }

        if total_missing == 0:
            self._info("No missing values.")
        else:
            self._warn(
                f"Missing values detected: {total_missing} "
                f"across {len(per_column)} column(s)."
            )

        self.report["missing_values"] = per_column
        self.report["total_missing"] = total_missing

    # -------------------------------------------------------

    def validate_duplicates(self, df):
        duplicates = int(df.duplicated().sum())

        if duplicates == 0:
            self._info("No duplicate records.")
        else:
            self._warn(
                f"Duplicate rows: {duplicates}. Consider whether "
                "they are genuine repeat submissions."
            )

        self.report["duplicates"] = duplicates

    # -------------------------------------------------------

    def validate_data_types(self, df):
        invalid = [
            col for col in self.feature_columns
            if col in df.columns
            and not pd.api.types.is_numeric_dtype(df[col])
        ]

        if not invalid:
            self._info("All predictor variables are numeric.")
        else:
            self._warn(
                f"Non-numeric predictor columns: {invalid}. "
                "They must be converted before training."
            )

        self.report["non_numeric_columns"] = invalid

    # -------------------------------------------------------

    def validate_ranges(self, df):
        checks = {
            "attendance_rate": (0, 100),
            "assignment_score": (0, 100),
            "test_score": (0, 100),
            "exam_score": (0, 100),
            "study_hours": (0, 24),
            "sleep_hours": (0, 24),
            "social_media_hours": (0, 24),
        }

        for feature, (minimum, maximum) in checks.items():

            if feature not in df.columns:
                continue

            if minimum is not None:
                if (df[feature] < minimum).any():
                    raise ValueError(
                        f"{feature} contains values below {minimum}"
                    )

            if maximum is not None:
                if (df[feature] > maximum).any():
                    raise ValueError(
                        f"{feature} contains values above {maximum}"
                    )

        self.report["ranges_validated"] = True
        self._info("Feature ranges validated.")

    # -------------------------------------------------------

    def validate_target_labels(self, df):
        """
        Check target labels when the target column is present.
        """

        if self.target_column not in df.columns:
            self._info(
                "Target column not yet present (created later by "
                "the target builder) -- skipping label checks."
            )
            return

        labels = set(df[self.target_column].dropna().unique())
        expected = set(self.expected_labels)

        if labels.issubset(expected):
            self._info("Target labels are valid.")
        else:
            self._warn(
                f"Unexpected labels found: {labels - expected}"
            )

        self.report["target_labels"] = sorted(
            str(label) for label in labels
        )

    # -------------------------------------------------------

    def dataset_summary(self, df):
        self.report["rows"] = int(df.shape[0])
        self.report["columns"] = int(df.shape[1])

        self._info(
            "Dataset summary: %d rows x %d columns.",
            df.shape[0],
            df.shape[1],
        )

    # -------------------------------------------------------

    def validate(self, df):
        """
        Run every validation step and return the cleaned frame.

        Parameters
        ----------
        df:
            Raw uploaded dataset.

        Returns
        -------
        pandas.DataFrame
            Standardised, scale-normalised dataframe.
        """

        self.warnings = []
        self.report = {}
        self.rescaled_columns = {}

        self._info("Starting dataset validation.")

        df = df.copy()

        df = self.standardize_columns(df)

        self.validate_columns(df)

        self.validate_missing_values(df)

        self.validate_duplicates(df)

        self.validate_data_types(df)

        df = self.normalize_score_scales(df)

        self.validate_ranges(df)

        self.validate_target_labels(df)

        self.dataset_summary(df)

        self.report["warnings"] = list(self.warnings)

        self._info(
            "Dataset ready with %d warning(s).",
            len(self.warnings),
        )

        return df
