"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/preprocessing/feature_engineer.py

Purpose:
    Turn a validated, CPI-labeled dataset into model-ready
    training data: feature/target selection, train/test split,
    feature scaling, and target encoding.

Status: NOT YET WIRED INTO framework_factory.py / framework.py
--------------------------------------------------------------------
This class is complete for the modules given so far (feature_
selection, train_test_split_data, scale_features, encode_target,
plus their validate_* companions), but framework_factory.py still
builds a separate DataPreparationPipeline
(backend/preprocessing/data_preparation_pipeline.py) as the
preparation_pipeline component, and StudentPerformanceFramework
calls one of prepare()/prepare_data()/fit_transform() on it --
none of which exist here. Per your instruction, FeatureEngineer is
being kept as its own class for now; you'll send a
prepare_for_training() (or similar) method that chains these
modules together, at which point I'll wire it into the framework.

encode_target() uses the manual PERFORMANCE_MAPPING (ordinal:
Low=0, Average=1, High=2), not sklearn's LabelEncoder (which would
assign codes alphabetically: Average=0, High=1, Low=2). This
matches StudentPerformanceFramework.DEFAULT_LABEL_MAPPING, which
assumes {0: "Low", 1: "Average", 2: "High"}.

Version:
    1.0 (partial -- awaiting prepare_for_training())
====================================================================
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.config import (
    FEATURE_COLUMNS,
    PERFORMANCE_LABELS,
    PERFORMANCE_MAPPING,
    RANDOM_STATE,
    STRATIFY,
    TARGET_COLUMN,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ============================================================
# FEATURE ENGINEER
# ============================================================

class FeatureEngineer:

    def __init__(self, df):
        self.df = df.copy()

    # -------------------------------------------------------
    # MODULE 1 : FEATURE / TARGET SELECTION
    # -------------------------------------------------------

    def feature_selection(self):
        """
        Select predictor variables and target variable
        for machine learning.
        """

        logger.info("Starting feature selection...")

        X = self.df[FEATURE_COLUMNS].copy()

        y = self.df[TARGET_COLUMN].copy()

        logger.info("Feature selection completed.")

        return X, y

    # -------------------------------------------------------
    # MODULE 2 : TRAIN / TEST SPLIT
    # -------------------------------------------------------

    def train_test_split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict:
        """
        Split predictor and target variables into
        training and testing datasets.

        Returns
        -------
        dict
            Dictionary containing training and testing sets.
        """

        logger.info("Starting train/test split...")

        stratify = y if STRATIFY else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify
        )

        logger.info("Train/test split completed.")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

    # -------------------------------------------------------

    def validate_split(
        self,
        split_data: dict
    ) -> None:
        """
        Validate the train/test split.
        """

        logger.info("Validating train/test split...")

        X_train = split_data["X_train"]
        X_test = split_data["X_test"]
        y_train = split_data["y_train"]
        y_test = split_data["y_test"]

        print("\n" + "=" * 60)
        print("TRAIN / TEST SPLIT SUMMARY")
        print("=" * 60)

        print(f"Training Samples : {len(X_train)}")
        print(f"Testing Samples  : {len(X_test)}")

        print("\nTraining Shape")
        print(X_train.shape)

        print("\nTesting Shape")
        print(X_test.shape)

        print("\nTraining Class Distribution")
        print(y_train.value_counts(normalize=True).round(3))

        print("\nTesting Class Distribution")
        print(y_test.value_counts(normalize=True).round(3))

        logger.info("Train/test split validated.")

    # -------------------------------------------------------
    # MODULE 3 : FEATURE SCALING
    # -------------------------------------------------------

    def scale_features(
        self,
        split_data: dict
    ) -> dict:
        """
        Scale predictor variables using StandardScaler.
        """

        logger.info("Scaling features...")

        X_train = split_data["X_train"]
        X_test = split_data["X_test"]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        logger.info("Feature scaling completed.")

        return {

            "X_train_scaled": pd.DataFrame(
                X_train_scaled,
                columns=X_train.columns,
                index=X_train.index
            ),

            "X_test_scaled": pd.DataFrame(
                X_test_scaled,
                columns=X_test.columns,
                index=X_test.index
            ),

            "scaler": scaler,

            "feature_names": list(X_train.columns),

            # Keep originals for Random Forest
            "X_train_original": X_train.copy(),
            "X_test_original": X_test.copy()
        }

    # -------------------------------------------------------

    def validate_scaling(
        self,
        scaled_data: dict
    ):
        """
        Validate feature scaling.
        """

        logger.info("Validating scaling...")

        X_train = scaled_data["X_train_scaled"]

        print("\n" + "=" * 60)
        print("FEATURE SCALING SUMMARY")
        print("=" * 60)

        print("Means")

        print(X_train.mean().round(3))

        print("\nStandard Deviations")

        print(X_train.std().round(3))

        logger.info("Scaling validation completed.")

    # -------------------------------------------------------
    # MODULE 4 : TARGET ENCODING
    # -------------------------------------------------------

    def encode_target(
        self,
        split_data: dict
    ) -> dict:
        """
        Encode the target variable using the project's
        predefined performance mapping (Low=0, Average=1,
        High=2 -- matches StudentPerformanceFramework.
        DEFAULT_LABEL_MAPPING).
        """

        logger.info("Encoding target labels...")

        y_train = (
            split_data["y_train"]
            .map(PERFORMANCE_MAPPING)
            .astype(int)
        )

        y_test = (
            split_data["y_test"]
            .map(PERFORMANCE_MAPPING)
            .astype(int)
        )

        logger.info("Target encoding completed.")

        return {

            "y_train": y_train,

            "y_test": y_test,

            "mapping": PERFORMANCE_MAPPING,

            "labels": PERFORMANCE_LABELS
        }

    # -------------------------------------------------------

    def validate_encoding(
        self,
        encoded_data: dict
    ):
        """
        Validate manual target encoding.
        """

        logger.info("Validating target encoding...")

        print("\n" + "=" * 60)
        print("TARGET LABEL ENCODING")
        print("=" * 60)

        for label, code in PERFORMANCE_MAPPING.items():
            print(f"{label:<10} -> {code}")

        print("\nEncoded Training Labels")

        print(encoded_data["y_train"].head())

        logger.info("Encoding validation completed.")

    # -------------------------------------------------------
    # TODO: prepare_for_training() -- chains modules 1-4 and
    # returns the dict ModelTrainer expects. Send whenever
    # ready and I'll wire this into framework_factory.py /
    # StudentPerformanceFramework.
    # -------------------------------------------------------
