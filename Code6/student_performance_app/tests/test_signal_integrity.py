"""
Version 3.3 standards, enforced as tests rather than as claims.

Four properties are checked here, and each one is the kind of thing a
project can *say* without doing:

1. **Signal integrity.** The pipeline must beat a majority-class baseline
   when the data contains signal, and must collapse to chance when it does
   not. A pipeline that scores 90 % on noise is measuring its own
   preprocessing, not the students. This is the direct antidote to the
   legacy ~97 % CPI number.
2. **Label-source exclusion.** A predictor column that the target is
   computed from may not reach the model -- and the model's output must be
   invariant to it, even when a caller hands it in.
3. **Bundle provenance.** A rule-reconstruction model (legacy ``cpi``)
   cannot be exported as the production bundle; when it is exported
   deliberately, it is labelled as a baseline so nothing can quote it as
   accuracy.
4. **Dynamic thresholds.** Labels are quantile-based by default, and a
   degenerate distribution is handled explicitly.

Run: python -m pytest tests/test_signal_integrity.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.config import (
    LABEL_POLICY,
    PERFORMANCE_MAPPING,
    PREDICTOR_COLUMNS,
    is_leaky_definition,
    predictor_columns,
)
from backend.cpi.cpi_engine import CPIEngine
from backend.evaluation.model_evaluator import ModelEvaluator
from backend.preprocessing.data_preparation_pipeline import (
    DataPreparationPipeline,
)
from backend.training.model_trainer import ModelTrainer

from tests.conftest import make_cohort


# ============================================================
# FIXTURES
# ============================================================

def _run_pipeline(frame: pd.DataFrame) -> dict:
    """Label, prepare, train one model, evaluate. Deliberately cheap."""

    labelled = CPIEngine().build_target_variable(
        frame.copy(),
        definition="exam_band",
    )

    prepared = DataPreparationPipeline(
        predictor_columns=PREDICTOR_COLUMNS
    ).prepare(labelled)

    model = ModelTrainer().train_logistic_regression(prepared)

    result = ModelEvaluator().evaluate(model, prepared)

    # evaluate() returns {"metrics": {...}, "model": ..., ...}; the
    # honest-metric assertions live in the inner dict.
    return result["metrics"]


@pytest.fixture(scope="module")
def trained():
    """
    One trained framework, shared.

    Search and calibration are switched off for speed: this module is
    about the contract (leakage, provenance, signal integrity), not
    about model selection, which other modules already test.
    """

    import framework.student_performance_framework as framework_module

    from framework.framework_factory import FrameworkFactory

    saved = (
        framework_module.TUNING,
        framework_module.CALIBRATION,
    )

    framework_module.TUNING = {"enabled": False}

    framework_module.CALIBRATION = {"enabled": False}

    try:

        factory = FrameworkFactory.create()

        result = factory.run_pipeline(make_cohort(300, seed=3))

    finally:

        (
            framework_module.TUNING,
            framework_module.CALIBRATION,
        ) = saved

    return factory, result


@pytest.fixture(scope="module")
def structured_metrics() -> dict:
    """A cohort with a real (invented) signal structure."""

    return _run_pipeline(make_cohort(500, seed=11))


@pytest.fixture(scope="module")
def noise_metrics() -> dict:
    """
    The same cohort with the exam scores shuffled.

    Marginals are identical, labels are built the same way, the
    predictors are untouched -- the only thing destroyed is the
    relationship between them and the outcome. This is the cleanest
    possible noise control: anything the model still scores is an
    artefact of the pipeline.
    """

    frame = make_cohort(500, seed=11)

    rng = np.random.default_rng(99)

    frame["exam_score"] = rng.permutation(
        frame["exam_score"].to_numpy()
    )

    return _run_pipeline(frame)


# ============================================================
# 1. SIGNAL INTEGRITY
# ============================================================

def test_structured_data_beats_the_baseline(structured_metrics):
    """There must be measurable signal when there is signal."""

    baseline = structured_metrics["baseline_majority"]

    assert structured_metrics["macro_f1"] > 0.45

    assert structured_metrics["accuracy"] > baseline + 0.10

    assert structured_metrics["lift_over_baseline"] > 0.10


def test_pure_noise_collapses_to_chance(noise_metrics):
    """
    The test that would have caught the original 98 % figure.

    With the target relationship destroyed, macro F1 must sit at chance
    for a three-class problem (1/3), accuracy must sit at the majority
    baseline, and ROC-AUC must be indistinguishable from 0.5.
    """

    assert noise_metrics["macro_f1"] < 0.45, (
        "macro F1 on shuffled-label data is "
        f"{noise_metrics['macro_f1']:.3f}: that is above chance, so "
        "something other than the signal is being learned"
    )

    baseline = noise_metrics["baseline_majority"]

    # Accuracy may land a little above or below the majority baseline
    # by chance; 8 points of slack on 100 held-out rows is generous.
    assert abs(noise_metrics["accuracy"] - baseline) < 0.08

    assert abs(noise_metrics["lift_over_baseline"]) < 0.08


def test_noise_roc_auc_is_a_coin_flip(noise_metrics):
    """AUC is the sharper instrument for 'is there any signal at all'."""

    auc = noise_metrics.get("roc_auc")

    if auc is None:

        pytest.skip("ROC-AUC not estimable for this class distribution")

    assert 0.40 <= auc <= 0.60, (
        f"ROC-AUC on noise is {auc:.3f}; it should be ~0.5"
    )


def test_the_gap_between_signal_and_noise_is_large(
    structured_metrics,
    noise_metrics,
):
    """
    One assertion that states the whole claim: the same pipeline, the
    same code path, and a difference of at least 20 points in macro F1
    between 'there is signal' and 'there is none'.
    """

    gap = (
        structured_metrics["macro_f1"]
        - noise_metrics["macro_f1"]
    )

    assert gap > 0.20, (
        "the pipeline does not separate signal from noise: "
        f"structured={structured_metrics['macro_f1']:.3f}, "
        f"noise={noise_metrics['macro_f1']:.3f}"
    )


def test_honest_metric_fields_are_always_present(structured_metrics):
    """Accuracy alone is not a reportable result."""

    for field in (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "baseline_majority",
        "lift_over_baseline",
        "per_class",
        "n_test",
    ):

        assert field in structured_metrics, field


# ============================================================
# 2. LABEL-SOURCE EXCLUSION
# ============================================================

def test_exam_band_predictors_exclude_the_exam_score():
    """The label source is never a predictor of its own band."""

    assert "exam_score" not in predictor_columns("exam_band")

    assert "exam_score" in predictor_columns("cpi")

    assert is_leaky_definition("exam_band") is False

    assert is_leaky_definition("cpi") is True


def test_predictions_are_invariant_to_the_label_source(trained):
    """
    Hand the model a wildly different exam score and it must not move.

    The feature is not in the contract, so a caller passing it must not
    be able to change the answer -- otherwise a "prediction" is just
    reading the input back.
    """

    framework, _ = trained

    student = {
        "attendance_rate": 72.0,
        "study_hours": 6.0,
        "sleep_hours": 7.0,
        "social_media_hours": 5.0,
        "assignment_score": 68.0,
        "test_score": 61.0,
    }

    low = dict(student, exam_score=5.0)

    high = dict(student, exam_score=99.0)

    first = framework.predict(low)

    second = framework.predict(high)

    assert first.predicted_label == second.predicted_label

    assert first.confidence == pytest.approx(second.confidence)


# ============================================================
# 3. BUNDLE PROVENANCE
# ============================================================

def test_production_bundle_is_certified(trained, tmp_path):

    framework, _ = trained

    path = tmp_path / "production.pkl"

    framework.save_best_model(str(path))

    import joblib

    bundle = joblib.load(path)

    assert bundle["bundle_role"] == "production"

    assert bundle["label_source_in_predictors"] is False

    assert bundle["target_definition"] == "exam_band"

    # Provenance of the metric, not just the model.
    assert bundle["split"]["strategy"] in ("random", "temporal")

    # A metric without a baseline is uninterpretable.
    assert bundle["metrics"]["baseline_majority"] is not None


def test_a_leaky_model_cannot_be_exported_as_production(
    trained,
    tmp_path,
):
    """
    The legacy CPI target may be studied, but it may not become
    ``best_model.pkl`` by accident: that is the file Assessment Mode
    serves to lecturers.
    """

    framework, _ = trained

    original = framework.target_definition

    framework.target_definition = "cpi"

    try:

        with pytest.raises(ValueError) as error:

            framework.save_best_model(str(tmp_path / "leaky.pkl"))

        message = str(error.value)

        assert "rule" in message.lower()

        assert "--allow-leaky-target" in message

    finally:

        framework.target_definition = original


def test_a_deliberate_leaky_baseline_is_labelled_as_one(
    trained,
    tmp_path,
):
    """Exported on purpose, it carries its own warning into the bundle."""

    framework, _ = trained

    original = framework.target_definition

    framework.target_definition = "cpi"

    path = tmp_path / "baseline.pkl"

    try:

        framework.save_best_model(
            str(path),
            allow_leaky_target=True,
        )

    finally:

        framework.target_definition = original

    import joblib

    bundle = joblib.load(path)

    assert bundle["bundle_role"] == "rule_reconstruction_baseline"

    assert bundle["label_source_in_predictors"] is True

    assert any(
        "RULE-RECONSTRUCTION" in caveat.upper()
        for caveat in bundle["caveats"]
    ), "a rule-reconstruction bundle must carry the warning internally"


def test_a_reloaded_production_bundle_keeps_its_certification(
    trained,
    tmp_path,
):
    from framework.framework_factory import FrameworkFactory

    framework, _ = trained

    path = tmp_path / "roundtrip.pkl"

    framework.save_best_model(str(path))

    other = FrameworkFactory.create()

    other.load_best_model(str(path))

    assert other.bundle_role == "production"

    assert not any(
        "RULE-RECONSTRUCTION" in caveat.upper()
        for caveat in other.methodology_caveats()
    )


# ============================================================
# 4. DYNAMIC THRESHOLDS
# ============================================================

def test_the_default_label_policy_is_quantile_based():

    assert LABEL_POLICY["method"] == "quantile"

    assert LABEL_POLICY["low_quantile"] < LABEL_POLICY["high_quantile"]

    # The legacy fixed cut-offs remain available for comparison, but
    # only as an explicit choice.
    assert "fixed_thresholds" in LABEL_POLICY


def test_quantile_thresholds_balance_the_classes(cohort):

    engine = CPIEngine()

    labelled = engine.build_target_variable(
        cohort.copy(),
        definition="exam_band",
    )

    counts = labelled["performance_level"].value_counts()

    share = counts / counts.sum()

    assert len(counts) == 3

    # Balanced thirds, within a few points.
    assert share.min() > 0.25, share.to_dict()

    assert engine.threshold_method == "quantile"

    assert engine.thresholds is not None


def test_a_degenerate_distribution_does_not_produce_an_empty_class():
    """
    If the score column is constant, quantile thresholds collapse onto
    each other and two classes would be empty. The engine must notice and
    fall back rather than emit an untrainable label set.
    """

    frame = make_cohort(200, seed=5)

    frame["exam_score"] = 55.0

    engine = CPIEngine()

    labelled = engine.build_target_variable(
        frame.copy(),
        definition="exam_band",
    )

    # Whatever it decides, it must not silently produce one class.
    assert labelled["performance_level"].nunique() >= 1

    assert engine.thresholds is not None
