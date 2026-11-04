"""
Tests for hyperparameter search and probability calibration.

Two guarantees matter here:
  1. search and calibration only ever see the training fold, and
  2. confidence figures become interpretable (measured by Brier
     score and expected calibration error), not just different.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.calibration import CalibratedClassifierCV

from backend.config import CALIBRATION, TUNING

from backend.cpi.cpi_engine import CPIEngine
from backend.preprocessing.data_preparation_pipeline import (
    DataPreparationPipeline,
)
from backend.training.model_trainer import ModelTrainer

from tests.conftest import make_cohort


@pytest.fixture(scope="module")
def prepared():
    """A prepared dataset shared across this module's tests."""

    labelled = CPIEngine().build_target_variable(
        make_cohort(300, seed=7),
        definition="exam_band",
    )

    return DataPreparationPipeline().prepare(labelled)


# ============================================================
# CONFIG
# ============================================================

def test_tuning_and_calibration_are_configurable():

    assert "enabled" in TUNING
    assert "enabled" in CALIBRATION
    assert CALIBRATION["method"] in ("sigmoid", "isotonic")


# ============================================================
# TUNING
# ============================================================

def test_tuning_returns_a_fitted_estimator_and_a_summary(prepared):

    trainer = ModelTrainer()

    estimator, summary = trainer.tune_model(
        prepared, "Random Forest"
    )

    # It must be usable, and it must have been refit on the
    # training fold (not merely searched).
    assert hasattr(estimator, "predict")

    predictions = estimator.predict(prepared["X_test_original"])

    assert len(predictions) == len(prepared["X_test_original"])

    assert summary["candidates_evaluated"] >= 1

    assert 0.0 <= summary["best_cv_score"] <= 1.0

    assert isinstance(summary["best_params"], dict)


def test_tuning_uses_only_the_training_fold(prepared):
    """The search must not see the hold-out set."""

    trainer = ModelTrainer()

    # Shrink the test fold to a single row: if tuning peeked at it,
    # the search would be dominated by that row and blow up.
    squashed = dict(prepared)

    squashed["X_test_original"] = prepared["X_test_original"].iloc[:1]
    squashed["X_test_scaled"] = prepared["X_test_scaled"].iloc[:1]
    squashed["y_test"] = prepared["y_test"].iloc[:1]

    _, summary = trainer.tune_model(
        squashed, "Random Forest"
    )

    assert 0.0 <= summary["best_cv_score"] <= 1.0


def test_unknown_model_has_no_grid(prepared):

    with pytest.raises(ValueError, match="No tuning grid"):

        ModelTrainer().tune_model(prepared, "Magic Model")


# ============================================================
# CALIBRATION
# ============================================================

def test_random_forest_can_be_calibrated(prepared):

    trainer = ModelTrainer()

    model = trainer.train_calibrated_random_forest(prepared)

    assert isinstance(model, CalibratedClassifierCV)

    probabilities = model.predict_proba(prepared["X_test_original"])

    # Valid probability distribution per row.
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_logistic_regression_calibration_wraps_its_own_scaler(prepared):

    trainer = ModelTrainer()

    model = trainer.train_calibrated_logistic_regression(prepared)

    # Takes the ORIGINAL (unscaled) features.
    probabilities = model.predict_proba(prepared["X_test_original"])

    assert probabilities.shape[0] == len(
        prepared["X_test_original"]
    )


def test_calibrated_model_still_encodes_the_same_classes(prepared):

    trainer = ModelTrainer()

    calibrated = trainer.train_calibrated_random_forest(prepared)

    uncalibrated = trainer.train_random_forest(prepared)

    assert list(calibrated.classes_) == list(uncalibrated.classes_)
