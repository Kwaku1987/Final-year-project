"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/reporting/report_engine.py

Purpose:
    Build report dictionaries from a PredictionResult (single
    student) or a batch prediction dataframe, ready for the CSV/
    Excel/PDF exporters.

Notes
-----
Adapted from a scratch draft that worked with a generic dict
shaped like {"prediction":..., "confidence":..., ...}. This
version takes the actual PredictionResult dataclass our
framework.predict() returns, and converts confidence/
probabilities from this project's 0-1 fraction convention to the
0-100 percentages the PDF/Excel exporters display.

Version:
    1.0
====================================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import pandas as pd


class ReportEngine:

    def __init__(self):
        pass

    # --------------------------------------------------------
    # SINGLE STUDENT REPORT
    # --------------------------------------------------------

    def generate_student_report(
        self,
        student: Dict[str, float],
        prediction: Any,
        student_name: str = None,
        student_id: str = None,
    ) -> dict:
        """
        Build a report dict for one student's prediction, ready
        for PDFExporter.export_student_report().

        Parameters
        ----------
        student:
            The student_data dict used for prediction.

        prediction:
            The PredictionResult returned by
            framework.predict() / controller.predict_student().

        student_name, student_id:
            Optional identity fields (Assessment Mode only --
            Administrator predictions don't collect these).
        """

        risk = getattr(prediction, "intervention", None)

        return {

            "generated_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "student_name": student_name,

            "student_id": student_id,

            "student_information": student,

            "prediction": prediction.predicted_label,

            "confidence": prediction.confidence * 100,

            "probabilities": {
                label: value * 100
                for label, value in
                (prediction.probabilities or {}).items()
            },

            "top_features": list(
                getattr(prediction, "top_features", []) or []
            ),

            "recommendations": list(
                getattr(prediction, "recommendations", []) or []
            ),

            "risk_priority": risk.priority if risk else None,

            "risk_score": risk.score if risk else None,

            "risk_reasons": risk.reasons if risk else [],

        }

    # --------------------------------------------------------
    # BATCH REPORT
    # --------------------------------------------------------

    def generate_batch_report(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Build a summary dict for a batch prediction result
        (the dataframe controller.predict_batch() returns).
        """

        return {

            "generated_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "students_processed": len(dataframe),

            "high_predictions": int(
                (dataframe["Predicted Performance"] == "High").sum()
            ),

            "average_predictions": int(
                (dataframe["Predicted Performance"] == "Average").sum()
            ),

            "low_predictions": int(
                (dataframe["Predicted Performance"] == "Low").sum()
            ),

            "average_confidence": round(
                dataframe["Confidence (%)"].mean(), 2
            ),

        }
