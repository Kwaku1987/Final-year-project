"""
Tests for temporal validation and fairness disaggregation.

These encode the two properties that make the analyses trustworthy:

* a temporal split never lets a period appear on both sides -- if it
  did, the "next semester" number would be a lie;
* a fairness table never quietly averages away a subgroup too small to
  support a conclusion, and never hides a class the model refuses to
  predict for a group.

Run: python -m pytest tests/test_temporal_and_fairness.py -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.config import PREDICTOR_COLUMNS
from backend.evaluation.fairness import (
    DEFAULT_MIN_SUPPORT,
    disaggregate,
    disparity_summary,
    fairness_report,
    to_markdown,
)
from backend.preprocessing.data_preparation_pipeline import (
    DataPreparationPipeline,
)
from backend.preprocessing.split_strategy import (
    TemporalSplitError,
    available_periods,
    compare_splits,
    period_counts,
    resolve_period_column,
    temporal_split,
)


# ============================================================
# FIXTURES
# ============================================================

PERIODS = ["2024/25-S1", "2024/25-S2", "2025/26-S1"]


def make_timeframe(rows_per_period: int = 60, seed: int = 7) -> pd.DataFrame:
    """A small frame with a period column and a target."""

    import numpy as np

    rng = np.random.default_rng(seed)

    frames = []

    for index, period in enumerate(PERIODS):

        n = rows_per_period

        frames.append(
            pd.DataFrame(
                {
                    "semester": [period] * n,
                    "study_hours": rng.normal(6, 2, n).clip(0, 12),
                    "gender": rng.choice(["Female", "Male"], n),
                    "performance_level": rng.choice(
                        ["Low", "Average", "High"],
                        n,
                        p=[0.34, 0.33, 0.33],
                    ),
                    "_order": [index] * n,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# ============================================================
# PERIOD HANDLING
# ============================================================

def test_periods_are_ordered_chronologically_not_alphabetically():

    frame = make_timeframe(10)

    assert available_periods(frame) == PERIODS


def test_period_column_is_detected_from_an_alias():

    frame = make_timeframe(10).rename(
        columns={"semester": "academic_year"}
    )

    assert resolve_period_column(frame) == "academic_year"


def test_missing_period_column_raises_with_an_actionable_message():

    frame = make_timeframe(10).drop(columns=["semester"])

    with pytest.raises(TemporalSplitError) as error:

        resolve_period_column(frame)

    # The message must name the problem and the fix: "add a column",
    # not just "KeyError".
    assert "semester" in str(error.value)
    assert "random split" in str(error.value)


def test_period_counts_are_reported_per_period():

    frame = make_timeframe(25)

    assert period_counts(frame) == {
        period: 25 for period in PERIODS
    }


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def test_temporal_split_holds_out_the_latest_period_by_default():

    frame = make_timeframe(40)

    X = frame[["study_hours"]]

    y = frame["performance_level"]

    split = temporal_split(X, y, frame)

    metadata = split["metadata"]

    assert metadata.test_periods == ["2025/26-S1"]

    assert metadata.train_periods == ["2024/25-S1", "2024/25-S2"]

    assert metadata.n_test == 40

    assert metadata.n_train == 80


def test_no_period_appears_on_both_sides_of_a_temporal_split():
    """
    The single property that makes the number meaningful.

    Checked on the row indices, not just the labels, so a mask bug
    cannot pass this test.
    """

    frame = make_timeframe(30)

    X = frame[["study_hours"]]

    y = frame["performance_level"]

    split = temporal_split(X, y, frame)

    train_periods = set(
        frame.loc[split["train_index"], "semester"]
    )

    test_periods = set(
        frame.loc[split["test_index"], "semester"]
    )

    assert not (train_periods & test_periods)

    # And every row landed somewhere: no silent data loss.
    assert (
        len(split["train_index"]) + len(split["test_index"])
        == len(frame)
    )


def test_two_periods_can_be_held_out_together():

    frame = make_timeframe(60)

    split = temporal_split(
        frame[["study_hours"]],
        frame["performance_level"],
        frame,
        test_periods=["2024/25-S2", "2025/26-S1"],
    )

    assert split["metadata"].train_periods == ["2024/25-S1"]

    assert split["metadata"].n_test == 120


def test_unknown_test_period_lists_the_available_ones():

    frame = make_timeframe(30)

    with pytest.raises(TemporalSplitError) as error:

        temporal_split(
            frame[["study_hours"]],
            frame["performance_level"],
            frame,
            test_periods=["2026/27-S1"],
        )

    assert "2026/27-S1" in str(error.value)

    assert "2024/25-S1" in str(error.value)


def test_a_single_period_cannot_be_split_temporally():

    frame = make_timeframe(30)

    frame["semester"] = "2024/25-S1"

    with pytest.raises(TemporalSplitError) as error:

        temporal_split(
            frame[["study_hours"]],
            frame["performance_level"],
            frame,
        )

    assert "at least two" in str(error.value)


def test_a_test_period_too_small_to_conclude_anything_is_refused():

    frame = make_timeframe(40)

    tiny = frame[frame["semester"] != "2025/26-S1"]

    extra = frame[frame["semester"] == "2025/26-S1"].head(5)

    frame = pd.concat([tiny, extra], ignore_index=True)

    with pytest.raises(TemporalSplitError) as error:

        temporal_split(
            frame[["study_hours"]],
            frame["performance_level"],
            frame,
        )

    assert "minimum" in str(error.value)


def test_single_class_on_the_test_side_is_warned_about_not_hidden():

    frame = make_timeframe(40)

    frame.loc[
        frame["semester"] == "2025/26-S1", "performance_level"
    ] = "High"

    split = temporal_split(
        frame[["study_hours"]],
        frame["performance_level"],
        frame,
    )

    assert any(
        "single class" in warning
        for warning in split["metadata"].warnings
    )


# ============================================================
# PIPELINE INTEGRATION
# ============================================================

@pytest.fixture
def labelled_cohort(cohort):
    """The shared cohort, labelled and stamped with periods."""

    from backend.cpi.cpi_engine import CPIEngine

    frame = CPIEngine().build_target_variable(
        cohort.copy(),
        definition="exam_band",
    )

    # Three chronological blocks, oldest first -- the shape a real
    # multi-semester export has.
    blocks = [
        ("2024/25-S1", 0.00, 0.40),
        ("2024/25-S2", 0.40, 0.75),
        ("2025/26-S1", 0.75, 1.00),
    ]

    semester = pd.Series(index=frame.index, dtype=object)

    for period, start, end in blocks:

        semester.iloc[
            int(start * len(frame)):int(end * len(frame))
        ] = period

    frame["semester"] = semester

    return frame


def test_pipeline_uses_the_temporal_split_when_asked(labelled_cohort):

    cohort = labelled_cohort

    pipeline = DataPreparationPipeline(
        predictor_columns=PREDICTOR_COLUMNS
    )

    prepared = pipeline.prepare(
        cohort.copy(),
        split_strategy="temporal",
        test_periods=["2025/26-S1"],
    )

    assert prepared["split_strategy"] == "temporal"

    assert prepared["split_metadata"]["test_periods"] == ["2025/26-S1"]

    # The prepared test rows must all come from the held-out period.
    test_periods = set(
        cohort.loc[prepared["test_index"], "semester"]
    )

    assert test_periods == {"2025/26-S1"}


def test_pipeline_defaults_to_a_random_split(labelled_cohort):

    pipeline = DataPreparationPipeline(
        predictor_columns=PREDICTOR_COLUMNS
    )

    prepared = pipeline.prepare(labelled_cohort.copy())

    assert prepared["split_strategy"] == "random"

    # A random split cannot avoid mixing periods -- that is exactly
    # why it overstates forward performance.
    assert prepared["split_metadata"]["train_periods"] == []


def test_unknown_split_strategy_is_rejected(labelled_cohort):

    pipeline = DataPreparationPipeline(
        predictor_columns=PREDICTOR_COLUMNS
    )

    with pytest.raises(ValueError):

        pipeline.prepare(
            labelled_cohort.copy(),
            split_strategy="by-eye",
        )


# ============================================================
# FAIRNESS
# ============================================================

def test_perfect_predictions_produce_no_disparity():

    y = [0, 1, 2] * 40

    groups = pd.Series(["A", "B"] * 60)

    results = disaggregate(y, y, groups, column="programme")

    summary = disparity_summary(results)

    assert summary["gap"] == 0.0

    assert summary["flags"] == []


def test_disparity_is_measured_not_averaged_away():
    """
    67% accuracy overall must not hide 100% for one group and 33% for
    the other. The gap is the finding.
    """

    y_true = [0, 0, 0] * 40

    y_pred = [0, 0, 0] * 40

    # Group A: always right. Group B: always wrong.
    y_pred[60:] = [1] * 60

    groups = pd.Series(["A"] * 60 + ["B"] * 60)

    results = disaggregate(
        y_true, y_pred, groups, column="programme"
    )

    summary = disparity_summary(results)

    assert summary["best_group"] == "A"

    assert summary["worst_group"] == "B"

    assert summary["best_value"] == pytest.approx(1.0)

    assert summary["worst_value"] == pytest.approx(0.0)

    assert summary["gap"] == pytest.approx(1.0)

    assert any("differs by" in flag for flag in summary["flags"])


def test_small_subgroups_are_flagged_rather_than_trusted():

    y = [0, 1, 2] * 40

    groups = pd.Series(["Big"] * 100 + ["Tiny"] * 20)

    y_pred = list(y)

    results = disaggregate(y, y_pred, groups, column="programme")

    by_name = {item.group: item for item in results}

    assert by_name["Big"].reliable is True

    assert by_name["Tiny"].reliable is False

    assert any("insufficient support" in warning
               for warning in by_name["Tiny"].warnings)

    # And an unreliable subgroup cannot set the gap: with only one
    # reliable group, no disparity conclusion is possible at all.
    summary = disparity_summary(results)

    assert "Tiny" in summary["groups_excluded"]

    assert summary["gap"] is None

    assert any(
        "only one subgroup" in flag for flag in summary["flags"]
    )


def test_a_class_the_model_never_predicts_for_a_group_is_reported():

    y_true = [0, 1, 2] * 20

    y_pred = [0, 1, 2] * 20

    # Group B never gets "High" predicted.
    y_pred = [
        0 if value == 2 else value
        for value in y_pred
    ]

    groups = pd.Series(["A"] * 30 + ["B"] * 30)

    results = disaggregate(
        y_true, y_pred, groups, min_support=10
    )

    by_name = {item.group: item for item in results}

    assert any(
        "never predicted 'High'" in warning
        for warning in by_name["B"].warnings
    )


def test_misaligned_inputs_are_refused():
    """
    A fairness table built on shifted rows is worse than none: it
    reports somebody else's performance as this group's.
    """

    with pytest.raises(ValueError):

        disaggregate(
            y_true=[0, 1, 2],
            y_pred=[0, 1],
            groups=pd.Series(["A", "B", "C"]),
        )


def test_fairness_report_skips_columns_that_cannot_be_used():

    frame = pd.DataFrame(
        {
            "gender": ["F", "M"] * 40,
            "programme": ["One"] * 80,   # a single value: unusable
        }
    )

    y = [0, 1] * 40

    report = fairness_report(y, y, frame)

    assert "gender" in report["columns"]

    assert "programme" in report["skipped"]

    markdown = to_markdown(report)

    assert "By gender" in markdown


def test_fairness_report_says_so_when_there_is_nothing_to_disaggregate():

    frame = pd.DataFrame({"study_hours": [1, 2, 3] * 20})

    y = [0, 1, 2] * 20

    report = fairness_report(y, y, frame)

    assert report["columns"] == {}

    assert "cannot be assessed" in report["note"]


def test_default_support_threshold_is_stated():

    # The docs quote this number; if it changes, the docs must too.
    assert DEFAULT_MIN_SUPPORT == 30


# ============================================================
# RANDOM vs TEMPORAL
# ============================================================

def test_split_comparison_reports_the_gap_both_ways():

    random_metrics = {
        "Logistic Regression": {"macro_f1": 0.65},
        "Random Forest": {"macro_f1": 0.63},
    }

    temporal_metrics = {
        "Logistic Regression": {"macro_f1": 0.52},
        "Random Forest": {"macro_f1": 0.51},
    }

    rows = compare_splits(random_metrics, temporal_metrics)

    by_name = {row["model"]: row for row in rows}

    assert by_name["Logistic Regression"]["gap"] == pytest.approx(-0.13)

    assert by_name["Random Forest"]["gap"] == pytest.approx(-0.12)

    # Worst degradation first: the risk belongs at the top of the
    # table, not buried.
    assert rows[0]["model"] == "Logistic Regression"


def test_split_comparison_refuses_to_invent_a_missing_number():

    rows = compare_splits(
        {"Logistic Regression": {"macro_f1": 0.65}},
        {},
    )

    assert rows[0]["temporal"] is None

    assert rows[0]["gap"] is None
