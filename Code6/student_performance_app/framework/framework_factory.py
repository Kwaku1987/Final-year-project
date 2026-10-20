"""
===============================================================
Student Performance Assessment System
===============================================================

Module:
    framework/framework_factory.py

Purpose:
    Create and configure the complete machine-learning
    framework.

Design Pattern:
    Factory Pattern

The dashboard must not create individual backend objects.
Instead, it requests one fully configured
StudentPerformanceFramework from this factory.

Version:
    5.0
===============================================================
"""

from __future__ import annotations

import logging
from typing import Optional

from backend.validation.dataset_validator import (
    DatasetValidator,
)

from backend.cpi.cpi_engine import (
    CPIEngine,
)

from backend.config import (
    CPI_CONFIG,
)

from backend.preprocessing.data_preparation_pipeline import (
    DataPreparationPipeline,
)

from backend.training.model_trainer import (
    ModelTrainer,
)

from backend.evaluation.model_evaluator import (
    ModelEvaluator,
)

from backend.interpretation.model_interpreter import (
    ModelInterpreter,
)

from framework.student_performance_framework import (
    StudentPerformanceFramework,
)


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# FRAMEWORK FACTORY
# ============================================================

class FrameworkFactory:
    """
    Factory responsible for constructing the complete Student
    Performance Prediction Framework.

    The factory centralizes dependency creation so that:

    - Dashboard pages remain simple.
    - Backend configuration is not duplicated.
    - All components use the same configuration.
    - The application has one consistent framework setup.
    """

    # ========================================================
    # CREATE FRAMEWORK
    # ========================================================

    @classmethod
    def create(
        cls,
    ) -> StudentPerformanceFramework:
        """
        Create a fully configured framework.

        Returns
        -------
        StudentPerformanceFramework
            A ready-to-use framework containing all backend
            components.
        """

        logger.info(
            "Creating Student Performance Framework..."
        )

        # ----------------------------------------------------
        # 1. Dataset Validation
        # ----------------------------------------------------

        validator = (
            cls._create_validator()
        )

        # ----------------------------------------------------
        # 2. CPI Engine
        # ----------------------------------------------------

        cpi_engine = (
            cls._create_cpi_engine()
        )

        # ----------------------------------------------------
        # 3. Data Preparation
        # ----------------------------------------------------

        preparation_pipeline = (
            cls._create_preparation_pipeline()
        )

        # ----------------------------------------------------
        # 4. Model Training
        # ----------------------------------------------------

        trainer = (
            cls._create_model_trainer()
        )

        # ----------------------------------------------------
        # 5. Model Evaluation
        # ----------------------------------------------------

        evaluator = (
            cls._create_model_evaluator()
        )

        # ----------------------------------------------------
        # 6. Model Interpretation
        # ----------------------------------------------------

        interpreter = (
            cls._create_model_interpreter()
        )

        # ----------------------------------------------------
        # 7. Framework Assembly
        # ----------------------------------------------------

        framework = (
            StudentPerformanceFramework(

                validator=validator,

                cpi_engine=cpi_engine,

                preparation_pipeline=(
                    preparation_pipeline
                ),

                trainer=trainer,

                evaluator=evaluator,

                interpreter=interpreter,

            )
        )

        logger.info(
            "Student Performance Framework created "
            "successfully."
        )

        return framework

    # ========================================================
    # COMPONENT FACTORIES
    # ========================================================

    @staticmethod
    def _create_validator() -> DatasetValidator:
        """
        Create the dataset validator.
        """

        logger.info(
            "Initializing DatasetValidator."
        )

        return DatasetValidator()

    # --------------------------------------------------------

    @staticmethod
    def _create_cpi_engine() -> CPIEngine:
        """
        Create the CPI engine.
        """

        logger.info(
            "Initializing CPIEngine."
        )

        return CPIEngine(CPI_CONFIG)

    # --------------------------------------------------------

    @staticmethod
    def _create_preparation_pipeline(
    ) -> DataPreparationPipeline:
        """
        Create the data preparation pipeline.
        """

        logger.info(
            "Initializing "
            "DataPreparationPipeline."
        )

        return DataPreparationPipeline()

    # --------------------------------------------------------

    @staticmethod
    def _create_model_trainer() -> ModelTrainer:
        """
        Create the model trainer.
        """

        logger.info(
            "Initializing ModelTrainer."
        )

        return ModelTrainer()

    # --------------------------------------------------------

    @staticmethod
    def _create_model_evaluator(
    ) -> ModelEvaluator:
        """
        Create the model evaluator.
        """

        logger.info(
            "Initializing ModelEvaluator."
        )

        return ModelEvaluator()

    # --------------------------------------------------------

    @staticmethod
    def _create_model_interpreter(
    ) -> ModelInterpreter:
        """
        Create the model interpreter.
        """

        logger.info(
            "Initializing ModelInterpreter."
        )

        return ModelInterpreter(
            top_n=5,
            importance_threshold=0.05,
        )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def create_framework(
) -> StudentPerformanceFramework:
    """
    Create and return a fully configured framework.

    This function provides a short and clean interface for
    application startup.

    Example
    -------

    framework = create_framework()
    """

    return FrameworkFactory.create()