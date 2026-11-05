"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/config.py

Purpose:
    Single source of truth for project-wide configuration:
    feature columns, predictor columns, target definition,
    target-labelling policy, score-scale rules, expected labels,
    and CPI weighting.

    Every backend component (DatasetValidator, CPIEngine,
    DataPreparationPipeline, ModelTrainer, ModelEvaluator,
    StudentPerformanceFramework) imports these constants rather
    than hardcoding its own copy. This is what caused the
    social_media_hours / time_on_social_media mismatch earlier --
    keeping one shared config avoids that class of bug going
    forward.

Version:
    3.0 -- methodology fixes
====================================================================

CHANGES IN 3.0 (all four are documented in docs/METHODOLOGY.md)
---------------------------------------------------------------
1. TARGET DEFINITION
   v2.x labelled every student with the Composite Performance
   Index (CPI) and then trained models to predict that CPI label
   *from the same seven columns the CPI was computed from*. That
   is circular: the target is a deterministic function of the
   predictors, so a 97-98% "accuracy" only measured how well the
   model re-learned an arithmetic formula.

   The framework now supports two target definitions:

     "exam_band" (default, recommended)
         Target  : exam_score bucketed into Low/Average/High.
         Predictors: everything EXCEPT exam_score -- attendance,
         study/sleep/social-media hours, assignment and test
         scores. This is a genuine early-warning task: term-time
         behaviour and continuous assessment are used to predict
         the end-of-term examination outcome, which is not yet
         known when the prediction is made.

     "cpi" (legacy)
         Target  : the Composite Performance Index label.
         Predictors: all seven columns.
         Kept for comparison / continuity with earlier chapters.
         The UI and reports label results produced under this
         definition as "rule reconstruction", not prediction.

2. PREDICTOR COLUMNS
   PREDICTOR_COLUMNS is derived from the active target definition,
   so a model can never accidentally be trained on its own target
   source column.

3. TARGET-LABELLING POLICY
   A fixed CPI cut-off of 60/80 on a weighted average that centres
   near 66 produced a ~1% "High" class -- unlearnable and
   unmeasurable. LABEL_POLICY now supports "quantile" (default:
   balanced thirds of the actual distribution) or "fixed"
   (the original 60/80 cut-offs), and the chosen thresholds are
   recorded in the metrics, the interpretation report and the
   saved model bundle.

4. SCORE-SCALE RULES
   assignment_score arrives on a 0-20 scale in the shipped data
   (stored mean 10.0, sd 6.0) while every other score is 0-100.
   With a 0.15 CPI weight that quietly cut its real contribution
   from 15 points to about 1.5. SCORE_SCALE_RULES rescales such
   columns to percentages before the CPI or the model sees them,
   and the validator raises a loud warning when it does.
"""

from __future__ import annotations

import os

from pathlib import Path

from typing import Dict, List, Optional, Sequence, Tuple

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_VERSION = "3.3.5"

# ------------------------------------------------------------
# APPLICATION DIRECTORY
# ------------------------------------------------------------
# Every default file path is resolved against this directory rather
# than the process working directory.
#
# Why this exists: Streamlit Community Cloud launches the app with the
# working directory set to the REPOSITORY ROOT (it runs
# ``streamlit run Code6/student_performance_app/app.py``), while a local
# run happens inside Code6/student_performance_app. With the old
# relative defaults, the hosted app looked for
# ``<repo>/models/best_model.pkl`` -- one directory above the bundle
# that is committed -- so the deployed Assessment Mode reported "No
# saved model found yet" while the same code worked locally. Paths must
# not depend on where the process happens to be started.
APP_DIR = Path(__file__).resolve().parents[1]

# The production model bundle.
DEFAULT_MODEL_PATH = APP_DIR / "models" / "best_model.pkl"

# The SQLite file used when SPAS_DB_PATH is not set.
DEFAULT_DB_PATH = APP_DIR / "data" / "spas.db"

TEST_SIZE = 0.20

RANDOM_STATE = 42

STRATIFY = True

# Cross-validation folds used by the training path.
CV_FOLDS = 5

# Feature governance file written next to the trained bundle.
MODEL_CARD_PATH = "docs/MODEL_CARD.md"

# ------------------------------------------------------------
# Master schema: every column the project understands.
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# TARGET DEFINITION
# ------------------------------------------------------------
# "exam_band" -> predict the examination band from early signals.
# "cpi"       -> legacy: predict the composite index label.
#
# Override at runtime without editing code:
#     SPAS_TARGET_DEFINITION=cpi streamlit run app.py
# ------------------------------------------------------------

TARGET_DEFINITIONS = ("exam_band", "cpi")

DEFAULT_TARGET_DEFINITION = "exam_band"

TARGET_DEFINITION = os.environ.get(
    "SPAS_TARGET_DEFINITION",
    DEFAULT_TARGET_DEFINITION,
).strip().lower()

if TARGET_DEFINITION not in TARGET_DEFINITIONS:
    raise ValueError(
        f"Unknown SPAS_TARGET_DEFINITION {TARGET_DEFINITION!r}. "
        f"Choose one of {TARGET_DEFINITIONS}."
    )

# The raw column each definition derives its label from
# (None = derived from several columns, i.e. the CPI).
TARGET_SOURCE_COLUMN: Dict[str, Optional[str]] = {
    "exam_band": "exam_score",
    "cpi": None,
}

# Columns that must never be used as predictors for a definition,
# because they are the label source itself.
EXCLUDED_PREDICTORS: Dict[str, Tuple[str, ...]] = {
    "exam_band": ("exam_score",),
    "cpi": (),
}

# Human-readable description shown in the UI and reports.
TARGET_DESCRIPTIONS = {
    "exam_band": (
        "End-of-term examination band (Low/Average/High) "
        "predicted from attendance, study habits, sleep, "
        "social-media use, assignments and class tests."
    ),
    "cpi": (
        "Composite Performance Index band (legacy definition: "
        "the label is computed from the same columns used as "
        "predictors, so scores measure rule reconstruction "
        "rather than prediction)."
    ),
}


def predictor_columns(
    definition: Optional[str] = None,
) -> List[str]:
    """
    Return the predictor columns for a target definition.

    The target source column(s) are excluded so the model can
    never be trained on its own label.
    """

    definition = definition or TARGET_DEFINITION

    if definition not in TARGET_DEFINITIONS:

        raise ValueError(
            f"Unknown target definition {definition!r}. "
            f"Choose one of {TARGET_DEFINITIONS}."
        )

    excluded = set(
        EXCLUDED_PREDICTORS.get(definition, ())
    )

    return [
        column
        for column in FEATURE_COLUMNS
        if column not in excluded
    ]


def is_leaky_definition(
    definition: Optional[str] = None,
) -> bool:
    """
    True when the definition's predictors include the column the
    label is computed from (i.e. the legacy CPI definition).
    """

    definition = definition or TARGET_DEFINITION

    source = TARGET_SOURCE_COLUMN.get(definition)

    if source is None:
        # CPI is a function of every feature column.
        return set(predictor_columns(definition)) >= set(
            FEATURE_COLUMNS
        )

    return source in predictor_columns(definition)


# Predictors actually used by the default configuration.
PREDICTOR_COLUMNS = predictor_columns()

# ------------------------------------------------------------
# TARGET-LABELLING POLICY
# ------------------------------------------------------------
# method="quantile" -> balanced thirds of the observed
#                      distribution (Low = bottom 33%,
#                      High = top 33%). Booleans keep the class
#                      balance learnable and the metrics honest.
# method="fixed"    -> the original fixed cut-offs.
# ------------------------------------------------------------

LABEL_POLICY = {
    "method": os.environ.get("SPAS_LABEL_METHOD", "quantile"),
    "low_quantile": 1 / 3,
    "high_quantile": 2 / 3,
    "fixed_thresholds": (60.0, 80.0),
}

if LABEL_POLICY["method"] not in ("quantile", "fixed"):
    raise ValueError(
        "LABEL_POLICY['method'] must be 'quantile' or 'fixed'."
    )

# ------------------------------------------------------------
# SCORE-SCALE RULES
# ------------------------------------------------------------
# column -> (detect_max, underlying_max)
# If the observed maximum of a column is <= detect_max, the
# column is assumed to be scored out of `underlying_max` and is
# rescaled to a 0-100 percentage.
# ------------------------------------------------------------

SCORE_SCALE_RULES: Dict[str, Tuple[float, float]] = {
    "assignment_score": (25.0, 20.0),
}

# ============================================================
# CPI CONFIGURATION (legacy + still used by the "cpi" target)
# ============================================================

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

# ============================================================
# HYPERPARAMETER SEARCH AND PROBABILITY CALIBRATION
# ============================================================
# Two problems this addresses (see docs/METHODOLOGY.md §5):
#
#  1. The models compared in v3.0 were two fixed configurations,
#     so "best model" meant "best of two untuned defaults".
#  2. Logistic Regression and Random Forest disagreed sharply on
#     borderline students (e.g. High at 100% vs Average at 63%)
#     because their probabilities were not on the same,
#     trustworthy scale. A confidence figure a lecturer cannot
#     interpret is worse than no figure.
#
# Both stages run on the TRAINING fold only; the hold-out test
# fold stays untouched so the reported metrics remain honest.
# Set SPAS_TUNING=0 / SPAS_CALIBRATION=0 to skip either stage.
# ------------------------------------------------------------

TUNING = {
    "enabled": os.environ.get("SPAS_TUNING", "1") == "1",
    "cv": 3,
    "n_iter": 8,
    "scoring": "f1_macro",
    "random_state": RANDOM_STATE,
}

TUNING_GRIDS = {
    "Random Forest": {
        "n_estimators": [200],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    },
    "Logistic Regression": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    },
}

CALIBRATION = {
    "enabled": os.environ.get("SPAS_CALIBRATION", "1") == "1",
    "method": "sigmoid",
    "cv": 3,
}

# ============================================================
# SECURITY / ACCESS CONTROL
# ============================================================
# Training Mode (dataset upload, retraining, model deletion) is
# gated behind a password. Set SPAS_ADMIN_PASSWORD_HASH to a
# SHA-256 hex digest produced by:
#
#     python -m utils.auth hash "your-password"
#
# If the variable is unset, utils.auth falls back to the
# documented development password and warns loudly in the UI.
# ------------------------------------------------------------

ADMIN_PASSWORD_ENV = "SPAS_ADMIN_PASSWORD_HASH"

ADMIN_PASSWORD_DEFAULT = "change-me-admin"

AUTH_MAX_ATTEMPTS = 5
