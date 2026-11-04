"""
Tests for model interpretation.

Regression: a VotingClassifier exposes neither coef_ nor
feature_importances_, so when the ensemble won model selection the
whole interpretation report used to fail with ValueError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.interpretation.model_interpreter import ModelInterpreter


FEATURES = [
    "attendance_rate",
    "study_hours",
    "sleep_hours",
    "social_media_hours",
    "assignment_score",
    "test_score",
]


def _toy_problem(n: int = 150, seed: int = 0):

    rng = np.random.default_rng(seed)

    X = pd.DataFrame(
        rng.normal(0, 1, (n, len(FEATURES))),
        columns=FEATURES,
    )

    # Label driven mostly by study_hours and test_score.
    score = 2 * X["study_hours"] + 1.5 * X["test_score"]

    y = np.where(score > 1, 2, np.where(score > -1, 1, 0))

    return X, y


def test_linear_model_is_interpreted():

    X, y = _toy_problem()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    ).fit(X, y)

    table = ModelInterpreter().extract_feature_importance(model, FEATURES)

    # The table is ranked by importance, so compare as sets.
    assert set(table["feature"]) == set(FEATURES)

    assert table["importance"].sum() == pytest.approx(1.0, abs=1e-6)

    assert (
        table.sort_values("importance").iloc[-1]["feature"]
        in {"study_hours", "test_score"}
    )


def test_tree_model_is_interpreted():

    X, y = _toy_problem()

    model = RandomForestClassifier(
        n_estimators=20, random_state=0
    ).fit(X, y)

    table = ModelInterpreter().extract_feature_importance(model, FEATURES)

    assert table["importance"].sum() == pytest.approx(1.0, abs=1e-6)


def test_ensemble_is_interpreted_via_a_member():
    """A VotingClassifier must not break the report."""

    X, y = _toy_problem()

    ensemble = VotingClassifier(
        estimators=[
            (
                "logistic_regression",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=500)),
                    ]
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=20, random_state=0
                ),
            ),
        ],
        voting="soft",
    ).fit(X, y)

    table = ModelInterpreter().extract_feature_importance(
        ensemble, FEATURES
    )

    assert table["importance"].sum() == pytest.approx(1.0, abs=1e-6)

    assert "ensemble member" in table["importance_method"].iloc[0]


def test_unsupported_model_still_raises_a_clear_error():

    class Nothing:
        pass

    with pytest.raises(ValueError, match="coef_"):

        ModelInterpreter().extract_feature_importance(
            Nothing(), FEATURES
        )
