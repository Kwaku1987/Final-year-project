"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/reporting/pdf_exporter.py

Purpose:
    Generate a formatted PDF report for one student's prediction.

Notes
-----
Writes to an in-memory BytesIO buffer rather than a file on disk
-- st.download_button() just needs bytes, and writing to disk in
a deployed Streamlit app risks permission issues and leftover
files for no benefit.

Version:
    1.0
====================================================================
"""

from __future__ import annotations

import io

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)


class PDFExporter:
    """
    Generates a PDF report for one student's prediction.
    """

    def export_student_report(self, report: dict) -> bytes:
        """
        Build the PDF and return its bytes, ready for
        st.download_button(data=...).
        """

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(buffer)

        styles = getSampleStyleSheet()

        story = []

        story.append(
            Paragraph(
                "<b>Student Performance Assessment System</b>",
                styles["Title"],
            )
        )

        story.append(
            Paragraph(
                "Academic Performance Assessment Report",
                styles["Heading2"],
            )
        )

        story.append(Spacer(1, 20))

        # ----------------------------------------------------
        # Student Identity (Assessment Mode only)
        # ----------------------------------------------------

        if report.get("student_name"):

            story.append(
                Paragraph(
                    "<b>Student</b>",
                    styles["Heading2"],
                )
            )

            identity = report["student_name"]

            if report.get("student_id"):

                identity += f" ({report['student_id']})"

            story.append(
                Paragraph(identity, styles["BodyText"])
            )

            story.append(Spacer(1, 15))

        # ----------------------------------------------------
        # Student Information
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "<b>Student Information</b>",
                styles["Heading2"],
            )
        )

        student = report["student_information"]

        for key, value in student.items():

            story.append(
                Paragraph(
                    f"{key}: {value}",
                    styles["BodyText"],
                )
            )

        story.append(Spacer(1, 15))

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "<b>Prediction</b>",
                styles["Heading2"],
            )
        )

        story.append(
            Paragraph(
                report["prediction"],
                styles["BodyText"],
            )
        )

        story.append(
            Paragraph(
                f"Confidence: {report['confidence']:.2f}%",
                styles["BodyText"],
            )
        )

        story.append(Spacer(1, 15))

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        if report.get("probabilities"):

            story.append(
                Paragraph(
                    "<b>Prediction Probabilities</b>",
                    styles["Heading2"],
                )
            )

            for label, probability in (
                report["probabilities"].items()
            ):

                story.append(
                    Paragraph(
                        f"{label}: {probability:.2f}%",
                        styles["BodyText"],
                    )
                )

            story.append(Spacer(1, 15))

        # ----------------------------------------------------
        # Top Contributing Factors
        # ----------------------------------------------------

        if report.get("top_features"):

            story.append(
                Paragraph(
                    "<b>Top Contributing Factors</b>",
                    styles["Heading2"],
                )
            )

            for feature in report["top_features"]:

                display = feature.get(
                    "display_name", feature.get("feature", "")
                )

                value = feature.get("value")

                story.append(
                    Paragraph(
                        f"• {display}: {value}",
                        styles["BodyText"],
                    )
                )

            story.append(Spacer(1, 15))

        # ----------------------------------------------------
        # Intervention / Risk Assessment
        # ----------------------------------------------------

        if report.get("risk_priority"):

            story.append(
                Paragraph(
                    "<b>Intervention Priority</b>",
                    styles["Heading2"],
                )
            )

            story.append(
                Paragraph(
                    f"{report['risk_priority']} "
                    f"(Risk Score: {report['risk_score']})",
                    styles["BodyText"],
                )
            )

            for reason in report.get("risk_reasons", []):

                story.append(
                    Paragraph(
                        f"• {reason}",
                        styles["BodyText"],
                    )
                )

            story.append(Spacer(1, 15))

        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        if report.get("recommendations"):

            story.append(
                Paragraph(
                    "<b>Recommendations</b>",
                    styles["Heading2"],
                )
            )

            for recommendation in report["recommendations"]:

                story.append(
                    Paragraph(
                        f"• {recommendation}",
                        styles["BodyText"],
                    )
                )

            story.append(Spacer(1, 20))

        story.append(
            Paragraph(
                f"Generated: {report['generated_at']}",
                styles["Italic"],
            )
        )

        doc.build(story)

        return buffer.getvalue()
