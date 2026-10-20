"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/preprocessing/target_encoder.py

Purpose:
    Validate and encode the target variable (Low/Average/High)
    into the ordinal integer codes the models train on.

Notes
-----
Reconstructed from a notebook fragment that gave the constructor
and a validate_target_labels()-style check, but not a complete
encode() body. encode() here validates first, then applies the
same ordinal PERFORMANCE_MAPPING (Low=0, Average=1, High=2) used
elsewhere in the project -- this matches
StudentPerformanceFramework.DEFAULT_LABEL_MAPPING.

Version:
    1.0
====================================================================
"""

from __future__ import annotations

import logging

import pandas as pd

from backend.config import (
    PERFORMANCE_LABELS,
    PERFORMANCE_MAPPING,
)

logger = logging.getLogger(__name__)

VALID_PERFORMANCE_LEVELS = set(PERFORMANCE_MAPPING.keys())


# ============================================================
# TARGET ENCODER
# ============================================================

class TargetEncoder:

    def __init__(self):

        self.mapping = PERFORMANCE_MAPPING

        self.labels = PERFORMANCE_LABELS

        self.valid_labels = VALID_PERFORMANCE_LEVELS

    # -------------------------------------------------------

    def validate(
        self,
        y: pd.Series
    ) -> None:
        """
        Validate that all target labels are recognised.
        """

        labels = set(y.unique())

        unknown = labels - self.valid_labels

        if unknown:

            raise ValueError(
                f"Unknown performance levels found: {unknown}"
            )

        logger.info("Target labels validated successfully.")

    # -------------------------------------------------------

    def encode(
        self,
        y: pd.Series
    ) -> pd.Series:
        """
        Validate and encode target labels into ordinal codes
        (Low=0, Average=1, High=2).
        """

        self.validate(y)

        return y.map(self.mapping).astype(int)
