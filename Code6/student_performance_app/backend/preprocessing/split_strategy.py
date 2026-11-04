"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/preprocessing/split_strategy.py

Purpose:
    Split a cohort **by time**, not at random.

Why this exists
---------------
A random 80/20 split of one cohort measures one thing well: whether the
model fits the data it was trained on. It does not measure whether the
model still works next semester. Rows from the same semester share
cohort effects (a different lecturer, a different timetable, a marking
re-scaling), so a random split lets the model see the future it is
supposed to predict.

Temporal validation asks the harder question:

    train on 2024/25, test on 2025/26

That is the only split that proves the model generalises forward in
time, and it is what an early-warning system is actually for.

Contract
--------
    split = temporal_split(X, y, dataframe, "semester")
    split["X_train"], split["X_test"], split["y_train"], split["y_test"]
    split["metadata"] -> what was trained on what, for the report

The default test period is the most recent one present in the data.
Every earlier period is training data. Periods are never split across
both sides -- that is the whole point, and it is asserted.

Failure modes are explicit: a missing period column, an unknown period
name, a test side with fewer than `min_test_rows` rows, or a period
with a single class all raise `TemporalSplitError` with a message that
says what to do. A silent fallback to a random split would be worse
than a crash: it would report a temporal result that never happened.
====================================================================
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# The column name used when the caller does not name one. Matches the
# column the synthetic generator writes and the alias the validator
# recognises.
DEFAULT_PERIOD_COLUMN = "semester"

# Aliases accepted so a real dataset does not have to be renamed
# before it can be validated temporally.
PERIOD_COLUMN_ALIASES = (
    "semester",
    "session",
    "term",
    "academic_period",
    "year",
    "academic_year",
    "cohort",
    "period",
)

MIN_TEST_ROWS = 20

MIN_TRAIN_ROWS = 50


class TemporalSplitError(ValueError):
    """Raised when a period-based split cannot be produced honestly."""


# ============================================================
# PERIOD HANDLING
# ============================================================

def _period_sort_key(value: str) -> tuple:
    """
    Sort period labels the way a human reads them.

    "2024/25-S1" < "2024/25-S2" < "2025/26-S1", and "2024" < "2025",
    because the numeric components are compared first and the text
    after them breaks the tie.
    """

    text = str(value)

    numbers = tuple(
        int(part) for part in re.findall(r"\d+", text)
    )

    letters = tuple(re.findall(r"[A-Za-z]+", text))

    return (numbers, letters, text)


def resolve_period_column(
    dataframe: pd.DataFrame,
    period_column: Optional[str] = None,
) -> str:
    """
    Return the name of the column holding the academic period.

    Raises
    ------
    TemporalSplitError
        If no usable column exists, listing the aliases that were
        checked -- so the fix is obvious from the message.
    """

    if period_column is not None:

        if period_column not in dataframe.columns:

            raise TemporalSplitError(
                f"Period column {period_column!r} is not in the dataset. "
                f"Available columns: {list(dataframe.columns)}"
            )

        return period_column

    lowered = {
        str(column).lower(): column
        for column in dataframe.columns
    }

    for alias in PERIOD_COLUMN_ALIASES:

        if alias in lowered:

            return lowered[alias]

    raise TemporalSplitError(
        "No academic-period column found. Temporal validation needs "
        f"one of {list(PERIOD_COLUMN_ALIASES)}. Add a semester column "
        "to your data, or pass --period-column explicitly. Without it "
        "the honest option is a random split, and a random split "
        "cannot tell you whether the model works next semester."
    )


def available_periods(
    dataframe: pd.DataFrame,
    period_column: Optional[str] = None,
) -> List[str]:
    """Every distinct period label, oldest first."""

    column = resolve_period_column(dataframe, period_column)

    values = (
        dataframe[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(values, key=_period_sort_key)


def period_counts(
    dataframe: pd.DataFrame,
    period_column: Optional[str] = None,
) -> Dict[str, int]:
    """Row count per period, oldest first."""

    column = resolve_period_column(dataframe, period_column)

    counts = (
        dataframe[column]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    return {
        period: int(counts.get(period, 0))
        for period in sorted(counts, key=_period_sort_key)
    }


# ============================================================
# THE SPLIT
# ============================================================

@dataclass
class SplitMetadata:
    """What was trained on what -- travels into the report."""

    strategy: str = "random"

    period_column: Optional[str] = None

    train_periods: List[str] = field(default_factory=list)

    test_periods: List[str] = field(default_factory=list)

    n_train: int = 0

    n_test: int = 0

    train_label_distribution: Dict[str, int] = field(
        default_factory=dict
    )

    test_label_distribution: Dict[str, int] = field(
        default_factory=dict
    )

    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:

        return {
            "strategy": self.strategy,
            "period_column": self.period_column,
            "train_periods": self.train_periods,
            "test_periods": self.test_periods,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_label_distribution": self.train_label_distribution,
            "test_label_distribution": self.test_label_distribution,
            "warnings": self.warnings,
        }

    def describe(self) -> str:
        """One line a human can paste into a report."""

        if self.strategy != "temporal":

            return (
                f"Random split: {self.n_train} train / "
                f"{self.n_test} test rows, drawn from the same cohort."
            )

        return (
            f"Temporal split: trained on {', '.join(self.train_periods)} "
            f"({self.n_train} rows), tested on "
            f"{', '.join(self.test_periods)} ({self.n_test} rows). "
            "No period appears on both sides."
        )


def _label_distribution(y: pd.Series) -> Dict[str, int]:

    counts = y.astype(str).value_counts().to_dict()

    return {str(key): int(value) for key, value in counts.items()}


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    dataframe: pd.DataFrame,
    period_column: Optional[str] = None,
    test_periods: Optional[Iterable[str]] = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    min_test_rows: int = MIN_TEST_ROWS,
) -> Dict[str, Any]:
    """
    Hold out whole periods at the end of the timeline.

    Parameters
    ----------
    X, y
        Predictors and target, indexed the same way as `dataframe`.
    dataframe
        The full frame carrying the period column. Read-only.
    period_column
        Column holding the academic period. Auto-detected from
        `PERIOD_COLUMN_ALIASES` when omitted.
    test_periods
        Periods to hold out. Defaults to the most recent period only.
    min_train_rows, min_test_rows
        Refuse to produce a split too small to mean anything.

    Returns
    -------
    dict
        X_train, X_test, y_train, y_test, train_index, test_index,
        metadata (a SplitMetadata).

    Raises
    ------
    TemporalSplitError
        On a missing column, an unknown period, an overlap between
        the two sides, or a side too small to evaluate.
    """

    column = resolve_period_column(dataframe, period_column)

    periods = available_periods(dataframe, column)

    if len(periods) < 2:

        raise TemporalSplitError(
            f"Temporal validation needs at least two distinct values in "
            f"{column!r}; found {periods}. Either add historical data or "
            "use the random split -- and say so in the report."
        )

    if test_periods is None:

        held_out = [periods[-1]]

    else:

        held_out = [str(period) for period in test_periods]

        unknown = [p for p in held_out if p not in periods]

        if unknown:

            raise TemporalSplitError(
                f"Unknown test period(s) {unknown}. Available: {periods}"
            )

    held_out = sorted(set(held_out), key=_period_sort_key)

    training = [
        period for period in periods if period not in held_out
    ]

    if not training:

        raise TemporalSplitError(
            f"Every period ({held_out}) was held out, leaving nothing to "
            "train on. Hold out the latest period, not all of them."
        )

    overlap = sorted(set(training) & set(held_out))

    if overlap:  # pragma: no cover - impossible by construction

        raise TemporalSplitError(
            f"Period(s) {overlap} appear on both sides of the split."
        )

    # Align the period labels to the rows actually in X.
    period_series = (
        dataframe.loc[X.index, column].astype(str)
    )

    train_mask = period_series.isin(training)

    test_mask = period_series.isin(held_out)

    X_train = X.loc[train_mask]

    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]

    y_test = y.loc[test_mask]

    metadata = SplitMetadata(
        strategy="temporal",
        period_column=column,
        train_periods=training,
        test_periods=held_out,
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        train_label_distribution=_label_distribution(y_train),
        test_label_distribution=_label_distribution(y_test),
    )

    if len(X_train) < min_train_rows:

        raise TemporalSplitError(
            f"Only {len(X_train)} training rows before "
            f"{', '.join(held_out)} (minimum {min_train_rows}). "
            "Hold out fewer periods, or collect another semester."
        )

    if len(X_test) < min_test_rows:

        raise TemporalSplitError(
            f"Only {len(X_test)} rows in the test period "
            f"{', '.join(held_out)} (minimum {min_test_rows}). "
            "A test set that small cannot support a conclusion -- "
            "hold out two periods instead."
        )

    for side, labels in (
        ("training", metadata.train_label_distribution),
        ("test", metadata.test_label_distribution),
    ):

        if len(labels) < 2:

            metadata.warnings.append(
                f"The {side} side contains a single class ({labels}). "
                "Metrics computed on it are not a measure of "
                "classification."
            )

    missing = int(period_series.isna().sum())

    if missing:

        metadata.warnings.append(
            f"{missing} row(s) have no {column!r} value and were "
            "excluded from both sides of the split."
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_index": X_train.index,
        "test_index": X_test.index,
        "metadata": metadata,
    }


def random_split_metadata(
    n_train: int,
    n_test: int,
    y_train: Optional[pd.Series] = None,
    y_test: Optional[pd.Series] = None,
) -> SplitMetadata:
    """Metadata for a conventional random split, so both are reportable."""

    return SplitMetadata(
        strategy="random",
        n_train=int(n_train),
        n_test=int(n_test),
        train_label_distribution=(
            _label_distribution(y_train) if y_train is not None else {}
        ),
        test_label_distribution=(
            _label_distribution(y_test) if y_test is not None else {}
        ),
    )


def compare_splits(
    random_metrics: Dict[str, Dict],
    temporal_metrics: Dict[str, Dict],
    metric: str = "macro_f1",
) -> List[Dict[str, Any]]:
    """
    Line up the same model under both splits.

    The gap is the interesting number: a model that scores 0.65
    randomly and 0.52 next semester has told you something a single
    number never could. Falls back to the next metric in the list when
    the preferred one is missing, and reports None rather than
    inventing a value.
    """

    rows: List[Dict[str, Any]] = []

    for name, random_result in random_metrics.items():

        temporal_result = temporal_metrics.get(name, {})

        row: Dict[str, Any] = {
            "model": name,
            "metric": metric,
            "random": None,
            "temporal": None,
            "gap": None,
        }

        for key in (metric, "accuracy"):

            value = (random_result or {}).get(key)

            if value is not None:

                row["random"] = float(value)
                row["metric"] = key
                break

        for key in (metric, "accuracy"):

            value = (temporal_result or {}).get(key)

            if value is not None:

                row["temporal"] = float(value)
                break

        if row["random"] is not None and row["temporal"] is not None:

            row["gap"] = row["temporal"] - row["random"]

        rows.append(row)

    # Worst degradation first: the largest DROP belongs at the top of
    # the table, where somebody will read it.
    rows.sort(
        key=lambda item: (
            item["gap"] if item["gap"] is not None else float("inf")
        ),
    )

    return rows
