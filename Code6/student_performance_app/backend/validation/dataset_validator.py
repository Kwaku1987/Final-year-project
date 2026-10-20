"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/validation/dataset_validator.py

Purpose:
    Validate the raw student dataset before CPI target
    generation and model training.

Notes
-----
validate_target_labels() checks df[target_column] against
expected_labels, but only when target_column is already present
in df. Validation normally runs before the CPI engine creates the
target column (see StudentPerformanceFramework.prepare_data(),
which validates first and only then calls
cpi_engine.build_target_variable()), so in the normal pipeline
flow this check is skipped -- there's nothing to check yet. It
will run for real if this validator is ever called on data that
already has a target column (e.g. re-validating post-CPI data).

Version:
    2.0
====================================================================
"""

from __future__ import annotations

import pandas as pd

from backend.config import (
    COLUMN_MAPPING,
    EXPECTED_LABELS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


# ============================================================
# DATASET VALIDATOR CLASS
# ============================================================

class DatasetValidator:
    """
    Validates a student dataset before preprocessing
    and machine learning.
    """

    def __init__(self,
                 feature_columns=FEATURE_COLUMNS,
                 target_column=TARGET_COLUMN,
                 expected_labels=EXPECTED_LABELS):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.expected_labels = expected_labels

    # -------------------------------------------------------

    def standardize_columns(self, df):
        """
        Rename known alternative column names
        to the project's standard schema.

        Skips a rename if the target column name already exists
        in the DataFrame, to avoid silently producing two
        columns with the same name (which would break any
        downstream df[new_name] lookup by returning a DataFrame
        instead of a Series).
        """
        renamed = {}
        skipped = {}
        for old_name, new_name in COLUMN_MAPPING.items():
            if old_name not in df.columns:
                continue
            if new_name in df.columns:
                skipped[old_name] = new_name
                continue
            df.rename(
                columns={old_name: new_name},
                inplace=True
            )
            renamed[old_name] = new_name
        if renamed:
            print("\nStandardized Columns")
            for old, new in renamed.items():
                print(f"  {old}  -->  {new}")
        else:
            print("✓ No column renaming required.")
        if skipped:
            print("\n⚠ Skipped renaming (target column already exists):")
            for old, new in skipped.items():
                print(f"  {old}  -x->  {new}  (both present -- check your data)")
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
        print("✓ Required columns validated")

    # -------------------------------------------------------

    def validate_missing_values(self, df):
        missing = df.isnull().sum().sum()
        if missing == 0:
            print("✓ No missing values")
        else:
            print(f"⚠ Missing values detected: {missing}")

    # -------------------------------------------------------

    def validate_duplicates(self, df):
        duplicates = df.duplicated().sum()
        if duplicates == 0:
            print("✓ No duplicate records")
        else:
            print(f"⚠ Duplicate rows: {duplicates}")

    # -------------------------------------------------------

    def validate_data_types(self, df):
        invalid = [
            col for col in self.feature_columns
            if col in df.columns
            and not pd.api.types.is_numeric_dtype(df[col])
        ]
        if not invalid:
            print("✓ All predictor variables are numeric")
        else:
            print(f"⚠ Non-numeric columns detected: {invalid}")

    # -------------------------------------------------------

    def validate_ranges(self, df):
        checks = {
            "attendance_rate": (0, 100),
            "assignment_score": (0, 100),
            "test_score": (0, 100),
            "exam_score": (0, 100),
            "study_hours": (0, None),
            "sleep_hours": (0, 24),
            "social_media_hours": (0, 24)
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
        print("✓ Feature ranges validated")

    # -------------------------------------------------------

    def validate_target_labels(self, df):
        """
        Check target labels when the target column is present.

        Skipped when target_column isn't in df yet -- this
        validator normally runs before the CPI engine creates
        the target column, so there is nothing to check most
        of the time.
        """
        if self.target_column not in df.columns:
            print(
                "… Target column not yet present "
                "(created later by CPIEngine) — "
                "skipping target label checks"
            )
            return

        labels = set(df[self.target_column].dropna().unique())
        expected = set(self.expected_labels)

        if labels.issubset(expected):
            print("✓ Target labels are valid")
        else:
            print(f"⚠ Unexpected labels found: {labels - expected}")

    # -------------------------------------------------------

    def dataset_summary(self, df):
        print("\nDataset Summary")
        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")

    # -------------------------------------------------------

    def validate(self, df):
        print("=" * 60)
        print("DATASET VALIDATION")
        print("=" * 60)
        df = self.standardize_columns(df)
        self.validate_columns(df)
        self.validate_missing_values(df)
        self.validate_duplicates(df)
        self.validate_data_types(df)
        self.validate_ranges(df)
        self.validate_target_labels(df)
        self.dataset_summary(df)
        print("=" * 60)
        print("DATASET READY")
        print("=" * 60)
        return df
