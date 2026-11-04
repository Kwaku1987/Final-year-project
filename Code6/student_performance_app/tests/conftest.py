"""
Shared fixtures for the test suite.

Everything here is synthetic and documented as such: the point of
the tests is to verify the pipeline's behaviour, not to claim any
performance figure. See docs/METHODOLOGY.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


FEATURE_COLUMNS = [
    "attendance_rate",
    "study_hours",
    "sleep_hours",
    "social_media_hours",
    "assignment_score",
    "test_score",
    "exam_score",
]


def make_cohort(
    n: int = 400,
    seed: int = 0,
    assignment_on_20_scale: bool = False,
) -> pd.DataFrame:
    """
    Build a synthetic cohort with a plausible (but invented)
    signal structure:

        study hours -> attendance -> test score -> exam score

    The exam score is never a direct function of the predictors,
    so the task is genuinely predictive rather than circular.
    """

    rng = np.random.default_rng(seed)

    z = rng.normal(0, 1, n)

    study = np.clip(6 + 2 * z, 0, 12)

    attendance = np.clip(
        70 + 12 * rng.normal(0, 1, n) + 3 * (study - 6),
        0,
        100,
    )

    sleep = np.clip(6.5 + 1.2 * rng.normal(0, 1, n), 0, 12)

    social = np.clip(
        6 - 0.8 * (study - 6) / 2 + 1.5 * rng.normal(0, 1, n),
        0,
        12,
    )

    assignment = np.clip(
        10 + 2.2 * (study - 6) / 2 + 1.2 * rng.normal(0, 1, n),
        0,
        20,
    )

    test = np.clip(
        60
        + 3.5 * (study - 6)
        + 0.35 * attendance
        + 8 * rng.normal(0, 1, n),
        0,
        100,
    )

    exam = np.clip(
        30
        + 4.5 * (study - 6)
        + 0.35 * attendance
        + 0.22 * test
        + 10 * rng.normal(0, 1, n),
        0,
        100,
    )

    if not assignment_on_20_scale:

        assignment = assignment * 5

    return pd.DataFrame(
        {
            "attendance_rate": attendance,
            "study_hours": study,
            "sleep_hours": sleep,
            "social_media_hours": social,
            "assignment_score": assignment,
            "test_score": test,
            "exam_score": exam,
        }
    )


@pytest.fixture(scope="session")
def cohort() -> pd.DataFrame:

    return make_cohort(500, seed=1)


@pytest.fixture(scope="session")
def small_cohort() -> pd.DataFrame:

    return make_cohort(200, seed=2)
