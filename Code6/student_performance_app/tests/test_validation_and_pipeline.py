"""
Tests for dataset validation and the preparation pipeline.

These cover the scale bug (assignment_score arriving on a 0-20
scale) and the predictor governance that stops a model being
trained on its own label.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.config import PREDICTOR_COLUMNS

from backend.validation.dataset_validator import DatasetValidator

from backend.cpi.cpi_engine import CPIEngine

from backend.preprocessing.data_preparation_pipeline import (
    DataPreparationPipeline,
)

from tests.conftest import make_cohort


# ============================================================
# VALIDATION
# ============================================================

def test_valid_dataset_passes_and_reports_no_errors(cohort):

    validator = DatasetValidator()

    validated = validator.validate(cohort)

    assert len(validated) == len(cohort)

    assert validator.report["rows"] == len(cohort)


def test_missing_required_column_raises():

    frame = make_cohort(50).drop(columns=["test_score"])

    with pytest.raises(ValueError, match="Missing required columns"):

        DatasetValidator().validate(frame)


def test_out_of_range_values_raise():

    frame = make_cohort(50)

    frame.loc[0, "attendance_rate"] = 140

    with pytest.raises(ValueError, match="above 100"):

        DatasetValidator().validate(frame)


def test_column_aliases_are_standardised():

    frame = make_cohort(50).rename(
        columns={
            "attendance_rate": "attendance",
            "test_score": "midterm_score",
        }
    )

    validated = DatasetValidator().validate(frame)

    assert "attendance_rate" in validated.columns

    assert "test_score" in validated.columns


def test_assignment_on_a_20_point_scale_is_rescaled_to_percentages():
    """The bug that silently cut a 15% weight to ~1.5 points."""

    frame = make_cohort(200, assignment_on_20_scale=True)

    assert frame["assignment_score"].max() <= 20

    validator = DatasetValidator()

    validated = validator.validate(frame)

    assert validated["assignment_score"].max() > 20

    assert validated["assignment_score"].max() <= 100

    assert "assignment_score" in validator.rescaled_columns

    assert any(
        "assignment_score" in warning
        for warning in validator.warnings
    )


def test_percentage_assignment_is_left_alone(cohort):
    """A genuine 0-100 column must not be multiplied by 5."""

    validator = DatasetValidator()

    validated = validator.validate(cohort)

    assert "assignment_score" not in validator.rescaled_columns

    assert validated["assignment_score"].max() == pytest.approx(
        cohort["assignment_score"].max()
    )


def test_warnings_are_collected_not_printed():

    frame = make_cohort(60)

    frame.loc[0:2, "exam_score"] = frame.loc[0, "exam_score"]

    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    validator = DatasetValidator()

    validator.validate(frame)

    assert isinstance(validator.warnings, list)

    assert any(
        "Duplicate rows" in warning
        for warning in validator.warnings
    )


# ============================================================
# PREPARATION
# ============================================================

def test_prepare_uses_predictors_only(cohort):

    labelled = CPIEngine().build_target_variable(
        cohort,
        definition="exam_band",
    )

    prepared = DataPreparationPipeline().prepare(labelled)

    assert prepared["feature_names"] == PREDICTOR_COLUMNS

    assert "exam_score" not in prepared["feature_names"]

    assert prepared["predictor_columns"] == PREDICTOR_COLUMNS


def test_prepare_splits_scales_and_encodes(cohort):

    labelled = CPIEngine().build_target_variable(
        cohort,
        definition="exam_band",
    )

    prepared = DataPreparationPipeline().prepare(labelled)

    total = len(prepared["X_train_original"]) + len(
        prepared["X_test_original"]
    )

    assert total == len(labelled)

    # 80/20 split
    test_share = len(prepared["X_test_original"]) / total

    assert 0.15 <= test_share <= 0.25

    # Scaled copy has zero mean, unit variance
    assert prepared["X_train_scaled"].mean().abs().max() < 1e-6

    # Targets are ordinal codes, not labels
    assert set(prepared["y_train"].unique()) <= {0, 1, 2}

    scaler = prepared["scaler"]

    assert hasattr(scaler, "transform")
