"""
Tests for configuration governance and the target builder.

These encode the two central methodology fixes:
  1. the target source column is never a predictor, and
  2. labels are balanced by default, with thresholds recorded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.config import (
    EXPECTED_LABELS,
    FEATURE_COLUMNS,
    LABEL_POLICY,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    is_leaky_definition,
    predictor_columns,
)

from backend.cpi.cpi_engine import CPIEngine


# ============================================================
# CONFIG GOVERNANCE
# ============================================================

def test_default_predictors_exclude_the_exam_score():
    """The label source must never be a predictor."""

    assert "exam_score" not in PREDICTOR_COLUMNS

    assert PREDICTOR_COLUMNS == predictor_columns("exam_band")


def test_cpi_definition_uses_every_column_and_is_flagged_leaky():
    """The legacy definition is kept, but must self-report."""

    assert predictor_columns("cpi") == FEATURE_COLUMNS

    assert is_leaky_definition("cpi") is True

    assert is_leaky_definition("exam_band") is False


def test_unknown_definition_raises():

    with pytest.raises(ValueError):

        predictor_columns("nonsense")


# ============================================================
# CPI MATHS
# ============================================================

def test_cpi_matches_hand_calculation():

    engine = CPIEngine()

    row = {
        "exam_score": 80,
        "test_score": 70,
        "assignment_score": 90,
        "attendance_rate": 100,
        "study_hours": 8,
        "sleep_hours": 8,
        "social_media_hours": 0,
    }

    expected = (
        0.40 * 80
        + 0.20 * 70
        + 0.15 * 90
        + 0.10 * 100
        + 0.10 * 100
        + 0.03 * 100
        + 0.02 * 100
    )

    assert engine.calculate_cpi(row) == pytest.approx(expected, abs=0.01)


def test_cpi_hours_are_clipped_to_the_configured_maximum():

    engine = CPIEngine()

    row = {
        "exam_score": 0,
        "test_score": 0,
        "assignment_score": 0,
        "attendance_rate": 0,
        "study_hours": 24,      # capped at 8
        "sleep_hours": 24,      # capped at 8
        "social_media_hours": 0,
    }

    # 0.10 * 100 (study) + 0.03 * 100 (sleep) + 0.02 * 100 (social)
    assert engine.calculate_cpi(row) == pytest.approx(15.0, abs=0.01)


# ============================================================
# LABELLING
# ============================================================

def test_quantile_policy_produces_balanced_classes(cohort):

    engine = CPIEngine()

    labelled = engine.build_target_variable(
        cohort,
        definition="exam_band",
    )

    counts = labelled[TARGET_COLUMN].value_counts()

    assert set(counts.index) == set(EXPECTED_LABELS)

    # Every class should hold roughly a third (±10 points).
    shares = counts / counts.sum()

    for share in shares:

        assert 0.23 <= share <= 0.44


def test_thresholds_are_recorded_for_audit(cohort):

    engine = CPIEngine()

    engine.build_target_variable(cohort, definition="exam_band")

    description = engine.describe_target()

    assert description["definition"] == "exam_band"

    assert description["method"] in ("quantile", "fixed")

    assert description["low_threshold"] < description["high_threshold"]


def test_fixed_policy_reproduces_the_original_cut_offs(cohort):

    engine = CPIEngine(
        label_policy={
            "method": "fixed",
            "fixed_thresholds": (60.0, 80.0),
        }
    )

    labelled = engine.build_target_variable(
        cohort,
        definition="exam_band",
    )

    low, high = 60.0, 80.0

    for value, label in zip(
        cohort["exam_score"],
        labelled[TARGET_COLUMN],
    ):

        if value >= high:
            assert label == "High"
        elif value >= low:
            assert label == "Average"
        else:
            assert label == "Low"


def test_cpi_definition_adds_a_cpi_column(cohort):

    engine = CPIEngine()

    labelled = engine.build_target_variable(
        cohort,
        definition="cpi",
    )

    assert "CPI" in labelled.columns

    assert labelled["CPI"].between(0, 100).all()


def test_unknown_definition_is_rejected(cohort):

    with pytest.raises(ValueError):

        CPIEngine().build_target_variable(cohort, definition="nope")


def test_classify_performance_uses_given_thresholds():

    engine = CPIEngine()

    assert engine.classify_performance(90, (60, 80)) == "High"

    assert engine.classify_performance(70, (60, 80)) == "Average"

    assert engine.classify_performance(10, (60, 80)) == "Low"
