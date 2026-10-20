"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/preprocessing/data_preparation_pipeline.py

Purpose:
    Coordinates all feature engineering tasks before model
    training: feature/target selection, train/test split,
    scaling, and target encoding.

Notes
-----
The notebook version this was reconstructed from has
DataPreparationPipeline take df in __init__() and prepare() take
no arguments. That doesn't match how StudentPerformanceFramework
actually calls it: framework.py's _call_prepare_method() calls
self.preparation_pipeline.prepare(dataframe) -- passing the
dataframe as an argument to prepare(), against a pipeline instance
that framework_factory.py builds once, with no arguments, before
any data exists. So here, prepare() takes the dataframe instead of
the constructor -- this matches framework_factory.py's existing
DataPreparationPipeline() call and framework.py's prepare(dataframe)
call, with no changes needed to either.

prepare() returns a dict (not a PreparedData object), per your
confirmation.

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

from backend.preprocessing.feature_engineer import FeatureEngineer
from backend.preprocessing.target_encoder import TargetEncoder

logger = logging.getLogger(__name__)


# ============================================================
# DATA PREPARATION PIPELINE
# ============================================================

class DataPreparationPipeline:
    """
    Coordinates all feature engineering tasks
    before model training.
    """

    def __init__(self):

        self.target_encoder = TargetEncoder()

        logger.info(
            "Data Preparation Pipeline initialized."
        )

    # -------------------------------------------------------

    def prepare(self, dataframe: pd.DataFrame) -> dict:
        """
        Run the complete data preparation sequence on a
        validated, CPI-labeled dataframe.

        Returns
        -------
        dict
            Model-ready data: scaled/original feature sets,
            encoded targets, scaler, feature names, and the
            performance-level mapping/labels.
        """

        logger.info(
            "Starting complete data preparation..."
        )

        feature_engineer = FeatureEngineer(dataframe)

        # ---------------------------------
        # Module 1 : feature / target selection
        # ---------------------------------

        X, y = feature_engineer.feature_selection()

        # ---------------------------------
        # Module 2 : train/test split
        # ---------------------------------

        split_data = feature_engineer.train_test_split_data(
            X, y
        )

        feature_engineer.validate_split(split_data)

        # ---------------------------------
        # Module 3 : feature scaling
        # ---------------------------------

        scaled_data = feature_engineer.scale_features(
            split_data
        )

        feature_engineer.validate_scaling(scaled_data)

        # ---------------------------------
        # Module 4 : target encoding
        # ---------------------------------

        y_train = self.target_encoder.encode(
            split_data["y_train"]
        )

        y_test = self.target_encoder.encode(
            split_data["y_test"]
        )

        logger.info(
            "Data preparation completed."
        )

        return {

            # Logistic Regression
            "X_train_scaled":
                scaled_data["X_train_scaled"],

            "X_test_scaled":
                scaled_data["X_test_scaled"],

            # Random Forest
            "X_train_original":
                scaled_data["X_train_original"],

            "X_test_original":
                scaled_data["X_test_original"],

            # Encoded Targets
            "y_train": y_train,

            "y_test": y_test,

            # Utilities
            "scaler":
                scaled_data["scaler"],

            "feature_names":
                scaled_data["feature_names"],

            "mapping": PERFORMANCE_MAPPING,

            "labels": PERFORMANCE_LABELS,
        }
