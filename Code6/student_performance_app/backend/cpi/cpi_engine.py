"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/cpi/cpi_engine.py

Purpose:
    Compute the Composite Performance Index (CPI) for each
    student and classify them into Low / Average / High
    performance tiers.

Notes
-----
This engine writes STRING labels ("Low" / "Average" / "High")
into the target column, not numeric codes. StudentPerformance-
Framework.predict() has been updated to handle string-label
predictions directly (see framework/student_performance_framework.py),
since models trained on this target will predict strings, not the
0/1/2 codes in DEFAULT_LABEL_MAPPING.

Version:
    2.0
====================================================================
"""

from __future__ import annotations

from backend.config import TARGET_COLUMN


# ============================================================
# CPI ENGINE
# ============================================================

class CPIEngine:
    def __init__(self, config):
        self.config = config
        self.weights = config["weights"]

    # -------------------------------------------------------

    def normalize_positive(self, value, maximum):
        score = (value / maximum) * 100
        return max(0, min(score, 100))

    # -------------------------------------------------------

    def normalize_negative(self, value, maximum):
        score = (1 - (value / maximum)) * 100
        return max(0, min(score, 100))

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

    def classify_performance(self, cpi):
        if cpi >= 80:
            return "High"
        elif cpi >= 60:
            return "Average"
        return "Low"

    # -------------------------------------------------------

    def build_target_variable(self, df):
        print("=" * 60)
        print("BUILDING TARGET VARIABLE")
        print("=" * 60)
        df["CPI"] = df.apply(
            self.calculate_cpi,
            axis=1
        )
        df[TARGET_COLUMN] = df["CPI"].apply(
            self.classify_performance
        )
        print("\nPerformance Distribution\n")
        print(df[TARGET_COLUMN].value_counts())
        print("\n✓ CPI successfully generated.")
        return df
