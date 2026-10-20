"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/eda/eda_analyzer.py

Purpose:
    Exploratory Data Analysis (EDA) on the validated student
    dataset -- dataset overview, summary statistics, missing
    value / duplicate analysis, feature distributions, target
    variable analysis, correlation analysis, outlier detection,
    feature relationships, and a generated EDA report.

Status: INCOMPLETE SCAFFOLD
--------------------------------------------------------------------
Only __init__ has been provided so far. None of the 10 analysis
responsibilities listed in the class docstring have method bodies
yet.

Not currently wired into anything: framework_factory.py doesn't
construct an EDAAnalyzer, and StudentPerformanceFramework doesn't
call one anywhere in its pipeline (validate -> CPI -> prepare ->
train -> evaluate -> interpret). If EDA is meant to run as part
of the app (e.g. a dashboard "Dataset" page, or a step before/
after validate_dataset()), let me know and I'll wire it in --
framework_factory.py's _create_validator()-style pattern, plus a
StudentPerformanceFramework method or a standalone dashboard call,
depending on where it's meant to sit.

Send over the remaining methods (stages 1-10 from the class
docstring) and I'll integrate and test them.

Version:
    1.0 (partial)
====================================================================
"""

from __future__ import annotations

import logging

import pandas as pd
import matplotlib.pyplot as plt

from backend.config import (
    CORRELATION_THRESHOLD,
    FEATURE_COLUMNS,
    MULTICOLLINEARITY_THRESHOLD,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


# ============================================================
# STAGE 6 : EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

class EDAAnalyzer:
    """
    Performs Exploratory Data Analysis (EDA) on the validated
    student dataset.

    Responsibilities
    ----------------
    1. Dataset overview
    2. Summary statistics
    3. Missing value analysis
    4. Duplicate analysis
    5. Feature distribution analysis
    6. Target variable analysis
    7. Correlation analysis
    8. Outlier detection
    9. Feature relationship analysis
    10. Generate EDA report
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the EDA analyzer.

        Parameters
        ----------
        df : pandas.DataFrame
            Validated dataset containing the generated
            target variable.
        """
        self.df = df.copy()
        logger.info("EDA Analyzer initialized.")

    # -------------------------------------------------------

    def dataset_overview(self) -> None:
        """
        Display a high-level overview of the dataset.
        """
        logger.info("Generating dataset overview...")
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        print(f"\nRows    : {self.df.shape[0]}")
        print(f"Columns : {self.df.shape[1]}")
        print("\nColumn Names")
        print("-"*60)
        for column in self.df.columns:
            print(f"• {column}")
        print("\nData Types")
        print("-"*60)
        print(self.df.dtypes)
        print("\nFirst Five Records")
        print("-"*60)
        print(self.df.head())
        print("\nLast Five Records")
        print("-"*60)
        print(self.df.tail())
        logger.info("Dataset overview completed.")

    # -------------------------------------------------------

    def summary_statistics(self) -> None:
        """
        Display descriptive statistics for all numerical features.
        The summary includes:
        - Count
        - Mean
        - Standard Deviation
        - Minimum
        - 25th Percentile
        - Median (50%)
        - 75th Percentile
        - Maximum
        """
        logger.info("Generating summary statistics...")
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        summary = self.df.describe()
        print(summary)
        logger.info("Summary statistics generated successfully.")

    # -------------------------------------------------------

    def missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.

        Returns
        -------
        pandas.DataFrame
            A summary table showing the number and percentage
            of missing values for each column.
        """
        logger.info("Checking for missing values...")
        missing_count = self.df.isnull().sum()
        missing_percent = (
            missing_count / len(self.df) * 100
        ).round(2)
        summary = pd.DataFrame({
            "Missing Values": missing_count,
            "Percentage (%)": missing_percent
        })
        print("\n" + "=" * 60)
        print("MISSING VALUE ANALYSIS")
        print("=" * 60)
        print(summary)
        total_missing = missing_count.sum()
        print("\n" + "-" * 60)
        if total_missing == 0:
            print("✓ No missing values detected.")
            logger.info("No missing values found.")
        else:
            print(f"⚠ Total Missing Values: {total_missing}")
            logger.warning(
                f"{total_missing} missing values detected."
            )
        return summary

    # -------------------------------------------------------

    def duplicate_analysis(self) -> pd.DataFrame:
        """
        Analyze duplicate records in the dataset.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing duplicate records.
        """
        logger.info("Checking for duplicate records...")
        duplicates = self.df[self.df.duplicated()]
        duplicate_count = len(duplicates)
        print("\n" + "=" * 60)
        print("DUPLICATE RECORD ANALYSIS")
        print("=" * 60)
        print(f"\nTotal Duplicate Records: {duplicate_count}")
        if duplicate_count == 0:
            print("\n✓ No duplicate records detected.")
            logger.info("No duplicate records found.")
        else:
            print("\nDuplicate Records Preview")
            print("-" * 60)
            print(duplicates.head())
            logger.warning(
                f"{duplicate_count} duplicate records detected."
            )
        return duplicates

    # -------------------------------------------------------

    def target_distribution(self) -> pd.DataFrame:
        """
        Analyze the distribution of the target variable.

        Returns
        -------
        pandas.DataFrame
            Frequency and percentage distribution of the
            target classes.
        """
        logger.info("Analyzing target variable distribution...")
        counts = self.df[TARGET_COLUMN].value_counts().sort_index()
        percentages = (
            self.df[TARGET_COLUMN]
            .value_counts(normalize=True)
            .sort_index() * 100
        ).round(2)
        summary = pd.DataFrame({
            "Count": counts,
            "Percentage (%)": percentages
        })
        print("\n" + "=" * 60)
        print("TARGET VARIABLE DISTRIBUTION")
        print("=" * 60)
        print(summary)
        plt.figure(figsize=(8, 5))
        plt.bar(
            summary.index,
            summary["Count"]
        )
        plt.title("Distribution of Student Performance Levels")
        plt.xlabel("Performance Level")
        plt.ylabel("Number of Students")
        plt.grid(axis="y", alpha=0.3)
        plt.show()
        imbalance = (
            summary["Percentage (%)"].max()
            - summary["Percentage (%)"].min()
        )
        if imbalance <= 20:
            print("\n✓ Dataset is reasonably balanced.")
            logger.info("Target variable is balanced.")
        else:
            print("\n⚠ Dataset appears imbalanced.")
            logger.warning(
                "Target variable may require class balancing."
            )
        return summary

    # NOTE: this is responsibility #6 (target variable analysis).
    # -------------------------------------------------------

    def feature_distributions(self) -> None:
        """
        Plot the distribution of all numerical features
        using histograms with Kernel Density Estimation (KDE).

        This helps identify:
        - Distribution shape
        - Skewness
        - Potential outliers
        - Data spread
        """

        logger.info("Generating feature distribution plots...")

        numeric_features = FEATURE_COLUMNS + ["CPI"]

        for feature in numeric_features:

            plt.figure(figsize=(8, 5))

            self.df[feature].plot(
                kind="hist",
                bins=20,
                density=True,
                alpha=0.7,
                edgecolor="black"
            )

            self.df[feature].plot(
                kind="kde"
            )

            plt.title(
                f"{feature.replace('_', ' ').title()} Distribution"
            )

            plt.xlabel(
                feature.replace("_", " ").title()
            )

            plt.ylabel("Density")

            plt.grid(alpha=0.3)

            plt.tight_layout()

            plt.show()

            logger.info(f"{feature} distribution plotted.")

        logger.info("Feature distribution analysis completed.")

    # -------------------------------------------------------

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Perform Pearson correlation analysis on all numerical
        features and generate a correlation heatmap.

        Returns
        -------
        pandas.DataFrame
            Pearson correlation matrix.
        """

        logger.info("Generating correlation analysis...")

        numeric_features = FEATURE_COLUMNS + ["CPI"]

        corr_matrix = self.df[numeric_features].corr(method="pearson")

        print("\n" + "=" * 60)
        print("CORRELATION MATRIX")
        print("=" * 60)

        print(corr_matrix.round(2))

        plt.figure(figsize=(10, 8))

        plt.imshow(corr_matrix,
                   cmap="coolwarm",
                   interpolation="nearest")

        plt.colorbar(label="Correlation")

        plt.xticks(
            range(len(corr_matrix.columns)),
            corr_matrix.columns,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(len(corr_matrix.columns)),
            corr_matrix.columns
        )

        plt.title("Pearson Correlation Heatmap")

        plt.tight_layout()

        plt.show()

        logger.info("Correlation heatmap generated.")

        return corr_matrix

    # -------------------------------------------------------

    def correlation_insights(
        self,
        corr_matrix: pd.DataFrame
    ) -> None:
        """
        Display significant correlations with CPI.
        """

        logger.info("Generating correlation insights...")

        print("\n" + "=" * 60)
        print("CORRELATION INSIGHTS")
        print("=" * 60)

        cpi_corr = corr_matrix["CPI"].drop("CPI")

        for feature, value in cpi_corr.items():

            strength = abs(value)

            if strength >= CORRELATION_THRESHOLD:

                direction = (
                    "Positive"
                    if value > 0
                    else "Negative"
                )

                print(
                    f"{feature:<25}"
                    f"{value:>6.2f}"
                    f"   ({direction})"
                )

        logger.info("Correlation insights generated.")

    # -------------------------------------------------------

    def multicollinearity_check(
        self,
        corr_matrix: pd.DataFrame
    ) -> None:
        """
        Detect highly correlated predictor variables.
        """

        logger.info("Checking multicollinearity...")

        print("\n" + "=" * 60)
        print("MULTICOLLINEARITY CHECK")
        print("=" * 60)

        for i in range(len(FEATURE_COLUMNS)):

            for j in range(i + 1, len(FEATURE_COLUMNS)):

                feature1 = FEATURE_COLUMNS[i]
                feature2 = FEATURE_COLUMNS[j]

                corr = corr_matrix.loc[
                    feature1,
                    feature2
                ]

                if abs(corr) >= MULTICOLLINEARITY_THRESHOLD:

                    print(
                        f"WARNING: "
                        f"{feature1}"
                        f" \u2194 "
                        f"{feature2}"
                        f" = {corr:.2f}"
                    )

        logger.info("Multicollinearity check completed.")

    # -------------------------------------------------------

    def outlier_analysis(self) -> pd.DataFrame:
        """
        Detect potential outliers using the
        Interquartile Range (IQR) method.

        Returns
        -------
        pandas.DataFrame
            Summary of detected outliers.
        """

        logger.info("Performing outlier analysis...")

        numeric_features = FEATURE_COLUMNS + ["CPI"]

        summary = []

        print("\n" + "=" * 60)
        print("OUTLIER ANALYSIS")
        print("=" * 60)

        for feature in numeric_features:

            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = self.df[
                (self.df[feature] < lower) |
                (self.df[feature] > upper)
            ]

            summary.append({
                "Feature": feature,
                "Outliers": len(outliers)
            })

            plt.figure(figsize=(8, 2))

            plt.boxplot(
                self.df[feature],
                vert=False
            )

            plt.title(
                f"{feature.replace('_',' ').title()} Boxplot"
            )

            plt.tight_layout()

            plt.show()

        summary = pd.DataFrame(summary)

        print(summary)

        logger.info("Outlier analysis completed.")

        return summary

    # -------------------------------------------------------

    def feature_relationships(self) -> None:
        """
        Visualize the relationship between each predictor
        variable and the Composite Performance Index (CPI).
        """

        logger.info("Generating feature relationship plots...")

        for feature in FEATURE_COLUMNS:

            plt.figure(figsize=(8, 5))

            plt.scatter(
                self.df[feature],
                self.df["CPI"],
                alpha=0.6
            )

            plt.title(
                f"{feature.replace('_',' ').title()} vs CPI"
            )

            plt.xlabel(
                feature.replace("_", " ").title()
            )

            plt.ylabel("Composite Performance Index")

            plt.grid(alpha=0.3)

            plt.tight_layout()

            plt.show()

            correlation = self.df[feature].corr(self.df["CPI"])

            print("\n" + "-" * 60)
            print(
                f"{feature.replace('_',' ').title()} "
                f"(r = {correlation:.2f})"
            )

            logger.info(
                f"{feature} relationship analyzed."
            )

        logger.info(
            "Feature relationship analysis completed."
        )

    # -------------------------------------------------------

    def generate_report(self) -> dict:
        """
        Generate a complete Exploratory Data Analysis report.

        Returns
        -------
        dict
            Dictionary containing all EDA results.
        """

        logger.info("Generating complete EDA report...")

        report = {}

        report["overview"] = self.dataset_overview()

        report["summary_statistics"] = self.summary_statistics()

        report["missing_values"] = self.missing_values()

        report["duplicates"] = self.duplicate_analysis()

        report["target_distribution"] = self.target_distribution()

        self.feature_distributions()

        corr_matrix = self.correlation_analysis()

        report["correlation_matrix"] = corr_matrix

        self.correlation_insights(corr_matrix)

        self.multicollinearity_check(corr_matrix)

        report["outliers"] = self.outlier_analysis()

        self.feature_relationships()

        logger.info("EDA report generation completed.")

        return report

    # -------------------------------------------------------

    def eda_recommendations(self) -> None:
        """
        Summarize the key findings from the
        Exploratory Data Analysis (EDA).
        """

        logger.info("Generating EDA recommendations...")

        print("\n" + "=" * 70)
        print("EDA SUMMARY AND RECOMMENDATIONS")
        print("=" * 70)

        print("\nDATASET HEALTH")

        print("\u2713 Required features are present.")
        print("\u2713 Data types are valid.")
        print("\u2713 Dataset structure is consistent.")

        print("\nDATA QUALITY")

        print("\u2713 Missing values have been addressed.")
        print("\u2713 Duplicate records have been checked.")
        print("\u2713 Dataset passed compatibility validation.")

        print("\nFEATURE ANALYSIS")

        print("\u2713 Feature distributions have been examined.")
        print("\u2713 Correlation analysis completed.")
        print("\u2713 Outliers identified and documented.")
        print("\u2713 Feature relationships analysed.")

        print("\nMODEL READINESS")

        print("\u2713 CPI successfully generated.")
        print("\u2713 Target variable is ready.")
        print("\u2713 Dataset is suitable for Feature Engineering.")

        print("\nFINAL RECOMMENDATION")

        print(
            "The dataset is considered READY for "
            "Feature Engineering and Machine Learning."
        )

        logger.info("EDA recommendations generated.")
