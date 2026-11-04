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

from typing import List, Optional

import pandas as pd

from backend.config import (
    PERFORMANCE_LABELS,
    PERFORMANCE_MAPPING,
    PREDICTOR_COLUMNS,
)

from backend.preprocessing.feature_engineer import FeatureEngineer
from backend.preprocessing.split_strategy import (
    SplitMetadata,
    random_split_metadata,
    temporal_split,
)
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

    def __init__(self, predictor_columns=None):

        self.target_encoder = TargetEncoder()

        self.predictor_columns = list(
            predictor_columns or PREDICTOR_COLUMNS
        )

        # How the data was split, for the report. "random" unless the
        # caller asks for temporal validation.
        self.split_metadata: SplitMetadata = SplitMetadata(
            strategy="random"
        )

        logger.info(
            "Data Preparation Pipeline initialized."
        )

    # -------------------------------------------------------

    def prepare(
        self,
        dataframe: pd.DataFrame,
        split_strategy: str = "random",
        period_column: Optional[str] = None,
        test_periods: Optional[List[str]] = None,
    ) -> dict:
        """
        Run the complete data preparation sequence on a
        validated, CPI-labeled dataframe.

        Parameters
        ----------
        split_strategy
            "random" (default) or "temporal". Temporal holds out whole
            academic periods at the end of the timeline, which is the
            only split that measures forward generalisation.
        period_column, test_periods
            Only meaningful for the temporal strategy; see
            backend/preprocessing/split_strategy.py.

        Returns
        -------
        dict
            Model-ready data: scaled/original feature sets,
            encoded targets, scaler, feature names, the
            performance-level mapping/labels, and the split metadata
            (strategy, periods, indices) so results can say what they
            were measured on.
        """

        logger.info(
            "Starting complete data preparation..."
        )

        feature_engineer = FeatureEngineer(
            dataframe,
            predictor_columns=self.predictor_columns,
        )

        # ---------------------------------
        # Module 1 : feature / target selection
        # ---------------------------------

        X, y = feature_engineer.feature_selection()

        # ---------------------------------
        # Module 2 : train/test split
        # ---------------------------------

        if str(split_strategy).lower() == "temporal":

            temporal = temporal_split(
                X,
                y,
                dataframe,
                period_column=period_column,
                test_periods=test_periods,
            )

            split_data = {
                "X_train": temporal["X_train"],
                "X_test": temporal["X_test"],
                "y_train": temporal["y_train"],
                "y_test": temporal["y_test"],
            }

            self.split_metadata = temporal["metadata"]

            logger.info(
                "Temporal split: train on %s, test on %s.",
                self.split_metadata.train_periods,
                self.split_metadata.test_periods,
            )

        elif str(split_strategy).lower() == "random":

            split_data = feature_engineer.train_test_split_data(
                X, y
            )

            self.split_metadata = random_split_metadata(
                len(split_data["X_train"]),
                len(split_data["X_test"]),
                split_data["y_train"],
                split_data["y_test"],
            )

        else:

            raise ValueError(
                f"Unknown split strategy {split_strategy!r}; expected "
                "'random' or 'temporal'."
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

            "predictor_columns":
                list(self.predictor_columns),

            "mapping": PERFORMANCE_MAPPING,

            "labels": PERFORMANCE_LABELS,

            # Split provenance. test_index lets the caller pull
            # demographic columns for the held-out rows, which is how
            # fairness disaggregation stays aligned with the metrics.
            "split_strategy": self.split_metadata.strategy,

            "split_metadata": self.split_metadata.to_dict(),

            "train_index": list(split_data["X_train"].index),

            "test_index": list(split_data["X_test"].index),
        }
