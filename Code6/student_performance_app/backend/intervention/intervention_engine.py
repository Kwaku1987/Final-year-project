"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/intervention/intervention_engine.py

Purpose:
    Compute an academic risk score and intervention priority for
    a student, based on their academic metrics, the model's
    prediction, and its confidence.

Notes
-----
The source material for this module had two versions: an earlier
one that only branched on the prediction label (Low/Average/High
-> HIGH/MEDIUM/LOW priority), and a later, more complete one that
computes an actual weighted risk score from individual factors
(attendance, study hours, sleep, social media, assignment/test/exam
scores) plus the prediction and confidence, with specific reasons
attached. This module implements the latter, since it gives
actionable, explainable output rather than just relabeling the
prediction.

Version:
    1.0
====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class InterventionResult:
    """
    The outcome of an academic risk assessment.
    """

    priority: str

    level: str

    score: int

    message: str

    reasons: List[str] = field(default_factory=list)


class InterventionEngine:
    """
    Computes an academic risk score and intervention priority.
    """

    def assess(
        self,
        student: Dict[str, float],
        prediction: str,
        confidence: float,
    ) -> InterventionResult:
        """
        Assess a student's academic risk.

        Parameters
        ----------
        student:
            The student's feature values (attendance_rate,
            study_hours, sleep_hours, social_media_hours,
            assignment_score, test_score, exam_score).

        prediction:
            The predicted performance label ("Low", "Average",
            "High").

        confidence:
            The prediction's confidence, as a fraction (0.0-1.0)
            or a percentage (0-100) -- both are handled.

        Returns
        -------
        InterventionResult
        """

        score = 0

        reasons: List[str] = []

        # ------------------------------------------------
        # Attendance
        # ------------------------------------------------

        attendance = student.get("attendance_rate", 100)

        if attendance < 50:

            score += 30

            reasons.append("Very poor attendance")

        elif attendance < 70:

            score += 15

            reasons.append("Low attendance")

        # ------------------------------------------------
        # Study Hours
        # ------------------------------------------------

        study = student.get("study_hours", 8)

        if study < 2:

            score += 20

            reasons.append("Insufficient study hours")

        elif study < 4:

            score += 10

            reasons.append("Study time can be improved")

        # ------------------------------------------------
        # Sleep
        # ------------------------------------------------

        sleep = student.get("sleep_hours", 8)

        if sleep < 5:

            score += 15

            reasons.append("Poor sleeping habits")

        # ------------------------------------------------
        # Social Media
        # ------------------------------------------------

        social = student.get("social_media_hours", 0)

        if social > 8:

            score += 15

            reasons.append("Excessive social media usage")

        elif social > 5:

            score += 8

            reasons.append("High social media usage")

        # ------------------------------------------------
        # Assignment
        # ------------------------------------------------

        assignment = student.get("assignment_score", 100)

        if assignment < 50:

            score += 20

            reasons.append("Weak assignment performance")

        # ------------------------------------------------
        # Test
        # ------------------------------------------------

        test = student.get("test_score", 100)

        if test < 50:

            score += 20

            reasons.append("Weak test performance")

        # ------------------------------------------------
        # Exam
        # ------------------------------------------------

        exam = student.get("exam_score", 100)

        if exam < 50:

            score += 30

            reasons.append("Poor examination performance")

        # ------------------------------------------------
        # Prediction
        # ------------------------------------------------

        if prediction == "Low":

            score += 30

        elif prediction == "Average":

            score += 10

        # ------------------------------------------------
        # Confidence -- normalize to a 0-100 scale first,
        # since predict() reports confidence as a 0.0-1.0
        # fraction while this scoring was originally written
        # assuming a 0-100 percentage.
        # ------------------------------------------------

        confidence_percent = (
            confidence * 100
            if confidence <= 1.0
            else confidence
        )

        if confidence_percent > 90:

            score += 10

        # ------------------------------------------------
        # Priority tier
        # ------------------------------------------------

        if score >= 90:

            return InterventionResult(

                priority="CRITICAL",

                level="Immediate Academic Intervention",

                score=score,

                message="Student requires urgent academic support.",

                reasons=reasons,

            )

        elif score >= 60:

            return InterventionResult(

                priority="HIGH",

                level="High Priority Monitoring",

                score=score,

                message="Student should be monitored weekly.",

                reasons=reasons,

            )

        elif score >= 35:

            return InterventionResult(

                priority="MEDIUM",

                level="Academic Monitoring",

                score=score,

                message="Provide additional academic guidance.",

                reasons=reasons,

            )

        else:

            return InterventionResult(

                priority="LOW",

                level="Maintain Performance",

                score=score,

                message="Continue current learning strategy.",

                reasons=reasons,

            )
