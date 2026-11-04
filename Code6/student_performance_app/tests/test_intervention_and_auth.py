"""
Tests for the intervention engine, the SHAP explainer wrapper and
the access-control helper.
"""

from __future__ import annotations

import pytest

from backend.intervention.intervention_engine import (
    InterventionEngine,
)


# ============================================================
# INTERVENTION ENGINE
# ============================================================

def test_high_risk_student_is_flagged_with_reasons():

    engine = InterventionEngine()

    result = engine.assess(
        student={
            "attendance_rate": 35,
            "study_hours": 1,
            "sleep_hours": 4,
            "social_media_hours": 9,
            "assignment_score": 30,
            "test_score": 40,
            "exam_score": 35,
        },
        prediction="Low",
        confidence=0.93,
    )

    assert result.priority in ("CRITICAL", "HIGH")

    assert result.score >= 60

    assert len(result.reasons) >= 4


def test_healthy_student_is_low_priority():

    engine = InterventionEngine()

    result = engine.assess(
        student={
            "attendance_rate": 95,
            "study_hours": 8,
            "sleep_hours": 8,
            "social_media_hours": 1,
            "assignment_score": 90,
            "test_score": 90,
            "exam_score": 92,
        },
        prediction="High",
        confidence=0.97,
    )

    assert result.priority == "LOW"

    # Even a clean case must explain itself.
    assert result.reasons


def test_medium_priority_has_at_least_one_stated_reason():
    """Regression: MEDIUM priority used to return reasons == []."""

    engine = InterventionEngine()

    result = engine.assess(
        student={
            "attendance_rate": 72,
            "study_hours": 4,
            "sleep_hours": 6,
            "social_media_hours": 5,
            "assignment_score": 55,
            "test_score": 58,
            "exam_score": 62,
        },
        prediction="Low",
        confidence=0.93,
    )

    assert result.reasons


def test_prediction_contributes_a_reason():

    engine = InterventionEngine()

    result = engine.assess(
        student={
            "attendance_rate": 90,
            "study_hours": 6,
            "sleep_hours": 7,
            "social_media_hours": 2,
            "assignment_score": 70,
            "test_score": 70,
            "exam_score": 70,
        },
        prediction="Low",
        confidence=0.55,
    )

    assert any(
        "predict" in reason.lower()
        for reason in result.reasons
    )


# ============================================================
# SHAP WRAPPER
# ============================================================

def test_shap_explainer_never_raises_on_unsupported_model():

    from backend.interpretation.shap_explainer import ShapExplainer

    import pandas as pd

    class Nonsense:
        def predict(self, frame):
            return [0]

    frame = pd.DataFrame(
        [{"attendance_rate": 80.0, "study_hours": 5.0}]
    )

    # Must return None rather than blowing up the prediction.
    assert ShapExplainer().explain(Nonsense(), frame) is None


# ============================================================
# AUTH HELPERS
# ============================================================

def test_password_hashing_is_stable_and_not_reversible():

    from utils.auth import hash_password

    first = hash_password("correct-horse")

    assert first == hash_password("correct-horse")

    assert first != hash_password("correct-horse ")

    assert len(first) == 64

    assert "correct-horse" not in first


def test_default_password_hash_differs_from_any_real_hash():

    from utils.auth import hash_password

    from backend.config import ADMIN_PASSWORD_DEFAULT

    assert hash_password(ADMIN_PASSWORD_DEFAULT) != hash_password(
        "definitely-not-the-default"
    )
