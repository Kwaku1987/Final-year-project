"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/training/model_trainer.py

Author:
    Brenya Kwaku Twum

Purpose
-------
Handles the complete machine learning training pipeline.

Responsibilities
----------------
- Prepare training datasets
- Perform train/test split
- Scale features when required
- Train supported ML models
- Perform cross validation
- Record training statistics
- Return structured training results

Notes
-----
This module was reconstructed from a notebook-style paste that
contained several duplicate/dead code fragments (a second copy of
_train_all_models()/_perform_cross_validation()/return sitting
after get_best_model()'s return statement, a stray dict literal,
top-level script calls, an unused ModelTrainingRecord dataclass
superseded by TrainingResult). Those were removed; train()'s body
was reassembled from scattered pieces (a validate_inputs() call
and a _perform_cross_validation() call had been physically
misplaced elsewhere in the file but clearly belonged in train()).

train_logistic_regression(prepared_data) and
train_random_forest(prepared_data) were added so this class
matches what StudentPerformanceFramework.train_models() actually
calls: two methods, each taking the prepared_data dict produced by
DataPreparationPipeline.prepare() (already scaled + ordinally
encoded), each returning a single fitted model. The original
train(dataframe, feature_columns, target_column) full-pipeline
method (5 models, its own internal split/scaling, cross-
validation) is kept intact as a separate, more thorough entry
point -- e.g. for a "compare all models" dashboard page -- but
isn't what framework.py calls during the normal predict pipeline.

Version
-------
5.2 (adapted)
====================================================================
"""

from __future__ import annotations

import logging
import time

from dataclasses import dataclass
from dataclasses import field

from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.tree import (
    DecisionTreeClassifier,
)

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.ensemble import (
    GradientBoostingClassifier,
)

from backend.preprocessing.target_encoder import (
    TargetEncoder,
)

# ------------------------------------------------------------
# Optional XGBoost
# ------------------------------------------------------------

try:

    from xgboost import (
        XGBClassifier,
    )

    XGBOOST_AVAILABLE = True

except ImportError:

    XGBOOST_AVAILABLE = False

# ------------------------------------------------------------

logger = logging.getLogger(__name__)

# ============================================================
# TRAINING RESULT
# ============================================================

@dataclass
class TrainingResult:
    """
    Stores every artifact produced during training.
    """

    trained_models: Dict[str, object] = field(
        default_factory=dict
    )

    scalers: Dict[str, Optional[StandardScaler]] = field(
        default_factory=dict
    )

    cross_validation: Dict[str, Dict] = field(
        default_factory=dict
    )

    training_times: Dict[str, float] = field(
        default_factory=dict
    )

    feature_names: List[str] = field(
        default_factory=list
    )

    X_train: Optional[pd.DataFrame] = None

    X_test: Optional[pd.DataFrame] = None

    y_train: Optional[pd.Series] = None

    y_test: Optional[pd.Series] = None

# ============================================================
# MODEL TRAINER
# ============================================================

class ModelTrainer:
    """
    Handles the training of all machine learning models.
    """

    # --------------------------------------------------------

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        cv_folds: int = 5,
    ):

        self.test_size = test_size

        self.random_state = random_state

        self.cv_folds = cv_folds

        logger.info(
            "ModelTrainer initialized."
        )

    # =======================================================
    # FRAMEWORK-COMPATIBLE ADAPTERS
    # =======================================================
    # These are what StudentPerformanceFramework.train_models()
    # actually calls. Each takes the prepared_data dict from
    # DataPreparationPipeline.prepare() and returns one fitted
    # model.
    # =======================================================

    def train_logistic_regression(self, prepared_data: dict):
        """
        Train Logistic Regression on the already-scaled
        training features and ordinally-encoded target.
        """

        logger.info("Training Logistic Regression...")

        start_time = time.perf_counter()

        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            solver="lbfgs",
        )

        model.fit(
            prepared_data["X_train_scaled"],
            prepared_data["y_train"],
        )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Logistic Regression trained in %.3f seconds.",
            elapsed,
        )

        return model

    # -------------------------------------------------------

    def train_random_forest(self, prepared_data: dict):
        """
        Train Random Forest on the original (unscaled)
        training features and ordinally-encoded target.
        Tree-based models don't need feature scaling.
        """

        logger.info("Training Random Forest...")

        start_time = time.perf_counter()

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
        )

        model.fit(
            prepared_data["X_train_original"],
            prepared_data["y_train"],
        )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Random Forest trained in %.3f seconds.",
            elapsed,
        )

        return model

    # =======================================================
    # FULL PIPELINE : TRAIN ALL SUPPORTED MODELS
    # =======================================================
    # Standalone entry point -- trains 5 models with its own
    # internal split/scaling/cross-validation, from a raw
    # dataframe. Not currently called by framework.py.
    # =======================================================

    def train(
        self,
        dataframe: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
    ) -> TrainingResult:
        """
        Train every supported model.

        Parameters
        ----------
        dataframe
        feature_columns
        target_column

        Returns
        -------
        TrainingResult
        """

        logger.info(
            "Beginning model training."
        )

        self._validate_inputs(
            dataframe,
            feature_columns,
            target_column,
        )

        result = TrainingResult()

        (
            X_train,
            X_test,
            y_train,
            y_test,

        ) = self._prepare_dataset(

            dataframe,

            feature_columns,

            target_column,

        )

        result.feature_names = feature_columns

        result.X_train = X_train

        result.X_test = X_test

        result.y_train = y_train

        result.y_test = y_test

        # --------------------------------------------------

        self._train_all_models(
            result
        )

        self._perform_cross_validation(
            result
        )

        logger.info(
            "Training completed."
        )

        return result

    # =======================================================
    # DATA PREPARATION
    # =======================================================

    def _prepare_dataset(
        self,
        dataframe,
        feature_columns,
        target_column,
    ):

        logger.info(
            "Preparing training dataset."
        )

        X = dataframe[
            feature_columns
        ]

        y = dataframe[
            target_column
        ]

        (
            X_train,
            X_test,
            y_train,
            y_test,

        ) = train_test_split(

            X,

            y,

            test_size=self.test_size,

            random_state=self.random_state,

            stratify=y,

        )

        # Encode target labels to ordinal codes (Low=0,
        # Average=1, High=2). Most sklearn classifiers tolerate
        # string class labels, but XGBoost requires integer,
        # 0-indexed classes -- encoding here keeps all 5 models
        # trained on the same target representation and matches
        # the ordinal mapping used elsewhere in the project.
        target_encoder = TargetEncoder()

        y_train = target_encoder.encode(y_train)

        y_test = target_encoder.encode(y_test)

        logger.info(
            "Training samples: %d",
            len(X_train),
        )

        logger.info(
            "Testing samples: %d",
            len(X_test),
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )

    # =======================================================
    # TRAIN ALL MODELS
    # =======================================================

    def _train_all_models(
        self,
        result: TrainingResult,
    ):

        logger.info(
            "Training all supported models."
        )

        self._train_logistic_regression(
            result
        )

        self._train_decision_tree(
            result
        )

        self._train_random_forest_variant(
            result
        )

        self._train_gradient_boosting(
            result
        )

        if XGBOOST_AVAILABLE:

            self._train_xgboost(
                result
            )

    # =======================================================
    # GENERIC TRAINING HELPER
    # =======================================================

    def _fit_model(
        self,
        result: TrainingResult,
        model_name: str,
        estimator,
        use_scaler: bool = False,
    ) -> None:
        """
        Train a single model and register it.
        """

        logger.info(
            "Training %s...",
            model_name,
        )

        start_time = time.perf_counter()

        X_train = result.X_train
        y_train = result.y_train

        scaler = None

        if use_scaler:

            scaler = StandardScaler()

            model = Pipeline([
                ("scaler", scaler),
                ("classifier", estimator),
            ])

        else:

            model = estimator

        model.fit(
            X_train,
            y_train,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        result.trained_models[
            model_name
        ] = model

        result.scalers[
            model_name
        ] = scaler

        result.training_times[
            model_name
        ] = elapsed

        logger.info(
            "%s trained in %.3f seconds.",
            model_name,
            elapsed,
        )

    # =======================================================
    # LOGISTIC REGRESSION
    # =======================================================

    def _train_logistic_regression(
        self,
        result: TrainingResult,
    ) -> None:

        estimator = LogisticRegression(

            random_state=self.random_state,

            max_iter=1000,

            solver="lbfgs",

        )

        self._fit_model(

            result=result,

            model_name="Logistic Regression",

            estimator=estimator,

            use_scaler=True,

        )

    # =======================================================
    # DECISION TREE
    # =======================================================

    def _train_decision_tree(
        self,
        result: TrainingResult,
    ) -> None:

        estimator = DecisionTreeClassifier(

            random_state=self.random_state,

        )

        self._fit_model(

            result=result,

            model_name="Decision Tree",

            estimator=estimator,

        )

    # =======================================================
    # RANDOM FOREST
    # =======================================================

    def _train_random_forest_variant(
        self,
        result: TrainingResult,
    ) -> None:
        """
        Trains a Random Forest as part of the full 5-model
        train() pipeline. Named distinctly from
        train_random_forest() (the framework.py adapter above)
        to avoid confusion between the two entry points.
        """

        estimator = RandomForestClassifier(

            n_estimators=200,

            random_state=self.random_state,

            n_jobs=-1,

        )

        self._fit_model(

            result=result,

            model_name="Random Forest",

            estimator=estimator,

        )

    # =======================================================
    # GRADIENT BOOSTING
    # =======================================================

    def _train_gradient_boosting(
        self,
        result: TrainingResult,
    ) -> None:

        estimator = GradientBoostingClassifier(

            random_state=self.random_state,

        )

        self._fit_model(

            result=result,

            model_name="Gradient Boosting",

            estimator=estimator,

        )

    # =======================================================
    # XGBOOST
    # =======================================================

    def _train_xgboost(
        self,
        result: TrainingResult,
    ) -> None:

        estimator = XGBClassifier(

            random_state=self.random_state,

            eval_metric="mlogloss",

        )

        self._fit_model(

            result=result,

            model_name="XGBoost",

            estimator=estimator,

        )

    # =======================================================
    # CROSS VALIDATION
    # =======================================================

    def _perform_cross_validation(
        self,
        result: TrainingResult,
    ) -> None:
        """
        Perform cross validation for every trained model.
        """

        logger.info(
            "Beginning cross validation."
        )

        for model_name, model in result.trained_models.items():

            scores = cross_val_score(

                estimator=model,

                X=result.X_train,

                y=result.y_train,

                cv=self.cv_folds,

                scoring="accuracy",

                n_jobs=-1,

            )

            result.cross_validation[
                model_name
            ] = {

                "scores": scores,

                "mean_accuracy": float(
                    np.mean(scores)
                ),

                "std_accuracy": float(
                    np.std(scores)
                ),

            }

            logger.info(
                "%s CV Mean = %.4f",
                model_name,
                np.mean(scores),
            )

    # =======================================================
    # TRAINING SUMMARY
    # =======================================================

    def training_summary(
        self,
        result: TrainingResult,
    ) -> pd.DataFrame:
        """
        Create a summary of all trained models.
        """

        rows = []

        for model_name in result.trained_models:

            cv = result.cross_validation[
                model_name
            ]

            rows.append({

                "Model": model_name,

                "Training Time (s)":
                    result.training_times[
                        model_name
                    ],

                "CV Mean Accuracy":
                    cv[
                        "mean_accuracy"
                    ],

                "CV Std":
                    cv[
                        "std_accuracy"
                    ],

            })

        dataframe = pd.DataFrame(
            rows
        )

        dataframe = dataframe.sort_values(

            by="CV Mean Accuracy",

            ascending=False,

        )

        dataframe.reset_index(

            drop=True,

            inplace=True,

        )

        return dataframe

    # =======================================================
    # BEST MODEL
    # =======================================================

    def get_best_model(
        self,
        result: TrainingResult,
    ):
        """
        Return the model with the highest
        cross validation accuracy.
        """

        best_name = max(

            result.cross_validation,

            key=lambda model:

            result.cross_validation[
                model
            ][
                "mean_accuracy"
            ],

        )

        logger.info(
            "Best model selected: %s",
            best_name,
        )

        return (

            best_name,

            result.trained_models[
                best_name
            ],

        )

    # =======================================================
    # DATA VALIDATION
    # =======================================================

    @staticmethod
    def _validate_inputs(
        dataframe: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
    ) -> None:
        """
        Validate training inputs.
        """

        if dataframe is None:
            raise ValueError(
                "Input dataframe cannot be None."
            )

        if dataframe.empty:
            raise ValueError(
                "Input dataframe is empty."
            )

        if not feature_columns:
            raise ValueError(
                "No feature columns provided."
            )

        if target_column not in dataframe.columns:
            raise ValueError(
                f"Target column '{target_column}' not found."
            )

        missing = [
            column
            for column in feature_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}"
            )

    # =======================================================
    # AVAILABLE MODELS
    # =======================================================

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Return the supported models.
        """

        models = [

            "Logistic Regression",

            "Decision Tree",

            "Random Forest",

            "Gradient Boosting",

        ]

        if XGBOOST_AVAILABLE:
            models.append("XGBoost")

        return models

    # =======================================================
    # TRAINING STATISTICS
    # =======================================================

    @staticmethod
    def training_statistics(
        result: TrainingResult,
    ) -> Dict:
        """
        Compute overall training statistics.
        """

        total_time = sum(
            result.training_times.values()
        )

        return {

            "number_of_models":
                len(result.trained_models),

            "total_training_time":
                total_time,

            "average_training_time":
                total_time
                / max(
                    len(result.trained_models),
                    1,
                ),

        }

    # =======================================================
    # EXPORT SUMMARY
    # =======================================================

    def export_training_summary(
        self,
        result: TrainingResult,
        filepath: str,
    ) -> None:
        """
        Export training summary to CSV.
        """

        summary = self.training_summary(
            result
        )

        summary.to_csv(
            filepath,
            index=False,
        )

        logger.info(
            "Training summary exported to %s",
            filepath,
        )
