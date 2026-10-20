"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/reporting/excel_exporter.py

Purpose:
    Export a batch prediction result to Excel, with a Predictions
    sheet and a Summary sheet.

Notes
-----
Writes to an in-memory BytesIO buffer, same reasoning as
pdf_exporter.py -- st.download_button() just needs bytes.

Version:
    1.0
====================================================================
"""

from __future__ import annotations

import io

import pandas as pd


class ExcelExporter:

    @staticmethod
    def export(
        dataframe: pd.DataFrame,
        summary: dict,
    ) -> bytes:
        """
        Build the Excel workbook and return its bytes, ready
        for st.download_button(data=...).
        """

        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:

            dataframe.to_excel(
                writer,
                sheet_name="Predictions",
                index=False,
            )

            summary_df = pd.DataFrame(
                summary.items(),
                columns=["Metric", "Value"],
            )

            summary_df.to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )

        return buffer.getvalue()
