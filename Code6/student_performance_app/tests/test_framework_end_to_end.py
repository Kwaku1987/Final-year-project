"""
End-to-end tests for the framework facade: train, evaluate,
honest metrics, predict, explain, save, load.
"""

from __future__ import annotations

import os

import pytest

from framework.framework_factory import FrameworkFactory


@pytest.fixture(scope="module")
def trained():
    """One trained framework shared by the tests in this module."""

    factory = FrameworkFactory.create()

    from tests.conftest import make_cohort

    result = factory.run_pipeline(make_cohort(400, seed=3))

    return factory, result


# ============================================================
# TARGET + PREDICTORS
# ============================================================

def test_predictors_never_include_the_label_source(trained):

    framework, _ = trained

    assert "exam_score" not in framework.required_features

    assert framework.target_description is not None

    assert framework.target_definition in ("exam_band", "cpi")


# ============================================================
# TRAINING + CV
# ============================================================

def test_three_models_are_trained(trained):

    framework, _ = trained

    assert set(framework.trained_models) == {
        "Logistic Regression",
        "Random Forest",
        "Soft-Voting Ensemble",
    }


def test_cross_validation_is_recorded(trained):

    framework, _ = trained

    assert framework.cross_validation

    for scores in framework.cross_validation.values():

        assert 0.0 <= scores["mean_accuracy"] <= 1.0

        assert scores["folds"] >= 2


def test_scaling_requirements_are_recorded_per_model(trained):

    framework, _ = trained

    assert framework.model_requires_scaling[
        "Logistic Regression"
    ] is True

    # The ensemble carries its own scaler inside a Pipeline.
    assert framework.model_requires_scaling[
        "Soft-Voting Ensemble"
    ] is False

    assert framework.model_requires_scaling[
        "Random Forest"
    ] is False


# ============================================================
# HONEST METRICS
# ============================================================

def test_metrics_include_honest_fields(trained):

    framework, _ = trained

    for result in framework.comparison_results.values():

        metrics = result["metrics"]

        for key in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "baseline_majority",
            "n_test",
            "per_class",
        ):

            assert key in metrics

        # Every label should appear in the per-class breakdown.
        assert set(metrics["per_class"]) == {
            "Low",
            "Average",
            "High",
        }


def test_accuracy_is_reported_against_a_baseline(trained):

    framework, _ = trained

    for result in framework.comparison_results.values():

        metrics = result["metrics"]

        assert (
            metrics["lift_over_baseline"]
            == pytest.approx(
                metrics["accuracy"]
                - metrics["baseline_majority"]
            )
        )


def test_interpretation_carries_methodology_caveats(trained):

    framework, _ = trained

    caveats = framework.interpretation_report.caveats

    assert caveats

    assert any(
        "Target definition" in caveat for caveat in caveats
    )

    assert any(
        "baseline" in caveat.lower() for caveat in caveats
    )


# ============================================================
# PREDICTION
# ============================================================

def _student(framework):

    return {
        "attendance_rate": 88.0,
        "study_hours": 8.0,
        "sleep_hours": 7.0,
        "social_media_hours": 2.0,
        "assignment_score": 80.0,
        "test_score": 78.0,
    }


def test_every_model_can_predict(trained):

    framework, _ = trained

    for name in framework.get_available_models():

        prediction = framework.predict(
            _student(framework),
            model_name=name,
        )

        assert prediction.predicted_label in (
            "Low",
            "Average",
            "High",
        )

        assert 0.0 <= prediction.confidence <= 1.0

        assert prediction.probabilities


def test_intervention_always_gives_reasons(trained):

    framework, _ = trained

    prediction = framework.predict(_student(framework))

    assert prediction.intervention is not None

    assert prediction.intervention.reasons


def test_missing_feature_raises(trained):

    framework, _ = trained

    with pytest.raises(ValueError):

        framework.predict({"attendance_rate": 50.0})


# ============================================================
# PERSISTENCE
# ============================================================

def test_save_and_load_roundtrip_preserves_the_contract(
    trained,
    tmp_path,
):

    framework, _ = trained

    path = tmp_path / "bundle.pkl"

    framework.save_best_model(str(path))

    reloaded = FrameworkFactory.create()

    reloaded.load_best_model(str(path))

    assert reloaded.best_model_name == framework.best_model_name

    assert reloaded.required_features == framework.required_features

    assert reloaded.target_definition == framework.target_definition

    assert reloaded.model_requires_scaling == (
        framework.model_requires_scaling
    )

    first = framework.predict(_student(framework))

    second = reloaded.predict(_student(framework))

    assert first.predicted_label == second.predicted_label


def test_bundle_records_environment_and_caveats(trained, tmp_path):

    import joblib

    framework, _ = trained

    path = tmp_path / "bundle.pkl"

    framework.save_best_model(str(path))

    bundle = joblib.load(path)

    for key in (
        "sklearn_version",
        "target_definition",
        "predictor_columns",
        "metrics",
        "caveats",
        "cross_validation",
    ):

        assert key in bundle

    assert bundle["predictor_columns"] == framework.required_features


def test_loading_a_missing_bundle_raises_a_clear_error():

    factory = FrameworkFactory.create()

    with pytest.raises(FileNotFoundError):

        factory.load_best_model("models/does-not-exist.pkl")
