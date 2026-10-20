"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/config.py

Purpose:
    Single source of truth for project-wide configuration:
    feature columns, target column, columns to drop, expected
    labels, and CPI weighting.

    Every backend component (DatasetValidator, CPIEngine,
    DataPreparationPipeline, ModelTrainer, ModelEvaluator,
    StudentPerformanceFramework) should import these constants
    rather than hardcoding its own copy. This is what caused the
    social_media_hours / time_on_social_media mismatch earlier --
    keeping one shared config avoids that class of bug going
    forward.

Version:
    2.0
====================================================================
"""

from __future__ import annotations


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_VERSION = "2.0"

TEST_SIZE = 0.20

RANDOM_STATE = 42

STRATIFY = True

FEATURE_COLUMNS = [
    "attendance_rate",
    "study_hours",
    "sleep_hours",
    "social_media_hours",
    "assignment_score",
    "test_score",
    "exam_score",
]

TARGET_COLUMN = "performance_level"

DROP_COLUMNS = [
    "performance_tier",
    "placement_status",
]

EXPECTED_LABELS = [
    "Low",
    "Average",
    "High",
]

CPI_CONFIG = {
    "study_hours_max": 8,
    "sleep_hours_max": 8,
    "social_media_hours_max": 8,
    "weights": {
        "exam_score": 0.40,
        "test_score": 0.20,
        "assignment_score": 0.15,
        "attendance_rate": 0.10,
        "study_score": 0.10,
        "sleep_score": 0.03,
        "social_score": 0.02,
    },
}

# ============================================================
# EDA THRESHOLDS
# ============================================================
# NOTE: common defaults -- adjust to your project's needs.
CORRELATION_THRESHOLD = 0.3        # |r| >= this vs CPI is flagged as notable
MULTICOLLINEARITY_THRESHOLD = 0.8  # |r| >= this between two predictors is flagged

# ============================================================
# PERFORMANCE LEVEL ENCODING
# ============================================================
PERFORMANCE_MAPPING = {
    "Low": 0,
    "Average": 1,
    "High": 2,
}

PERFORMANCE_LABELS = {
    0: "Low",
    1: "Average",
    2: "High",
}

# ============================================================
# COLUMN STANDARDIZATION MAP
# ============================================================
COLUMN_MAPPING = {
    # Attendance
    "attendance": "attendance_rate",
    "attendance_percentage": "attendance_rate",
    "attendance_percent": "attendance_rate",
    # Social Media
    "internet_usage": "social_media_hours",
    "time_on_social_media": "social_media_hours",
    "social_media_time": "social_media_hours",
    "social_media_usage": "social_media_hours",
    # Assignment
    "assignments_completed": "assignment_score",
    "assignment_marks": "assignment_score",
    # Test
    "previous_score": "test_score",
    "quiz_score": "test_score",
    "midterm_score": "test_score",
    # Exam
    "final_exam_score": "exam_score",
}
