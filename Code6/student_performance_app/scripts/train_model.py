"""
====================================================================
Student Performance Assessment System
====================================================================

Script:
    scripts/train_model.py

Purpose:
    Train the whole pipeline from the command line and write a model
    bundle -- the same work the ⚙ Model Training page does, but
    reproducible, scriptable and CI-friendly.

Why this exists
---------------
In v2.x the only way to produce models/best_model.pkl was through the
Streamlit UI, so the reported metrics could not be reproduced by
anyone reading the repository. This script makes the training run
auditable:

    python scripts/train_model.py --data data/cohort.csv \
        --out models/best_model.pkl --metrics reports/metrics.json

Usage
-----
    # retrain on your dataset with the default (honest) definition
    python scripts/train_model.py --data data/cohort.csv

    # legacy definition, for comparison only
    python scripts/train_model.py --data data/cohort.csv --definition cpi

    # fixed thresholds instead of quantile labelling
    python scripts/train_model.py --data data/cohort.csv --label-method fixed

Exit codes
----------
    0  success
    1  the training run failed (bad data, missing columns, ...)
====================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("train_model")


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel dataset."""

    if not path.exists():

        raise FileNotFoundError(f"No such file: {path}")

    if path.suffix.lower() in (".xlsx", ".xls"):

        return pd.read_excel(path)

    return pd.read_csv(path)


def summarize(framework) -> dict:
    """Collect the metrics that must accompany any claim."""

    comparison = {}

    for name, result in framework.comparison_results.items():

        metrics = result.get("metrics", {})

        comparison[name] = {
            key: metrics.get(key)
            for key in (
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
                "roc_auc",
                "baseline_majority",
                "lift_over_baseline",
                "n_test",
            )
        }

        comparison[name]["per_class"] = metrics.get("per_class")

    return {
        "target": framework.target_description,
        "predictors": list(framework.required_features),
        "split": getattr(framework, "split_metadata", None),
        "fairness": getattr(framework, "fairness_report", None),
        "cross_validation": framework.cross_validation,
        "models": comparison,
        "best_model": framework.best_model_name,
        "caveats": framework.methodology_caveats(),
        "validation_warnings": framework.validator_warnings,
    }


def main(argv=None) -> int:

    parser = argparse.ArgumentParser(
        description="Train and export the student performance models."
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="CSV/XLSX dataset with the project's feature columns",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/best_model.pkl"),
        help="where to write the model bundle",
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="optional path for a JSON metrics report",
    )

    parser.add_argument(
        "--definition",
        choices=("exam_band", "cpi"),
        default=None,
        help=(
            "target definition; exam_band (default) predicts the "
            "examination band from early signals, cpi is the legacy "
            "circular definition"
        ),
    )

    parser.add_argument(
        "--label-method",
        choices=("quantile", "fixed"),
        default=None,
        help="how Low/Average/High thresholds are chosen",
    )

    parser.add_argument(
        "--allow-leaky-target",
        action="store_true",
        help=(
            "export a rule-reconstruction baseline. Required for "
            "--definition cpi: that label is a formula over the "
            "predictors, so its metrics describe rule reproduction, not "
            "prediction, and it may never be reported as accuracy"
        ),
    )

    parser.add_argument(
        "--min-support",
        type=int,
        default=30,
        help=(
            "minimum subgroup size before a fairness row counts as "
            "reliable (default: 30)"
        ),
    )

    parser.add_argument(
        "--split",
        choices=("random", "temporal"),
        default="random",
        help=(
            "random (default) holds out a random 20%% of the cohort; "
            "temporal holds out the most recent semester(s), which is "
            "the only split that measures forward generalisation"
        ),
    )

    parser.add_argument(
        "--period-column",
        default=None,
        help=(
            "column holding the academic period (auto-detected from "
            "semester/session/term/academic_year when omitted)"
        ),
    )

    parser.add_argument(
        "--test-period",
        action="append",
        default=None,
        dest="test_periods",
        help=(
            "period to hold out; repeatable. Defaults to the most "
            "recent period in the data (temporal split only)"
        ),
    )

    parser.add_argument(
        "--fairness",
        action="store_true",
        help=(
            "also disaggregate the selected model's hold-out "
            "performance by gender / programme / entry qualification, "
            "where those columns exist"
        ),
    )

    args = parser.parse_args(argv)

    if args.definition == "cpi" and not args.allow_leaky_target:

        print(
            "REFUSED: --definition cpi builds the label from the same "
            "columns used as predictors (CPI is a weighted average of "
            "attendance, study hours, assignments, tests and the exam). "
            "Its ~97% 'accuracy' is rule reconstruction, not prediction, "
            "and it is banned from production metrics.\n"
            "\n"
            "  production model:      --definition exam_band (default)\n"
            "  baseline experiment:   --definition cpi --allow-leaky-target\n"
            "\n"
            "The baseline export is labelled bundle_role="
            "\"rule_reconstruction_baseline\" so nothing can quote it "
            "as accuracy by accident.",
            file=sys.stderr,
        )

        return 2

    # Apply overrides before importing config-dependent modules.
    if args.definition:

        import os

        os.environ["SPAS_TARGET_DEFINITION"] = args.definition

    if args.label_method:

        import os

        os.environ["SPAS_LABEL_METHOD"] = args.label_method

    # Imported late so the environment overrides take effect.
    from framework.framework_factory import FrameworkFactory

    try:

        dataframe = load_dataset(args.data)

    except Exception as error:

        logger.error("Could not read %s: %s", args.data, error)

        return 1

    logger.info(
        "Loaded %d rows x %d columns from %s",
        dataframe.shape[0],
        dataframe.shape[1],
        args.data,
    )

    framework = FrameworkFactory.create()

    try:

        framework.run_pipeline(
            dataframe,
            split_strategy=args.split,
            period_column=args.period_column,
            test_periods=args.test_periods,
        )

        if args.fairness:

            framework.evaluate_subgroups(
                min_support=args.min_support
            )

        saved_to = framework.save_best_model(
            str(args.out),
            allow_leaky_target=args.allow_leaky_target,
        )

    except Exception as error:

        logger.error("Training failed: %s", error)

        return 1

    report = summarize(framework)

    if framework.target_definition == "cpi":

        print(
            "=" * 66,
            "\nRULE-RECONSTRUCTION BASELINE — NOT A PREDICTIVE MODEL\n"
            "The target is computed from the predictors, so the metrics "
            "below\nmeasure formula reproduction. Do not report them as "
            "accuracy.\n" + "=" * 66,
            file=sys.stderr,
        )

    logger.info("Split: %s", getattr(framework, "split_summary", args.split))

    logger.info("Best model: %s", report["best_model"])
    logger.info(
        "Target: %s (method=%s, thresholds=%s/%s, distribution=%s)",
        report["target"]["definition"],
        report["target"]["method"],
        report["target"]["low_threshold"],
        report["target"]["high_threshold"],
        report["target"]["distribution"],
    )

    for name, metrics in report["models"].items():

        logger.info(
            "%-24s accuracy=%.3f macroF1=%.3f balanced=%.3f "
            "baseline=%.3f",
            name,
            metrics["accuracy"] or 0.0,
            metrics["macro_f1"] or 0.0,
            metrics["balanced_accuracy"] or 0.0,
            metrics["baseline_majority"] or 0.0,
        )

    if args.fairness and framework.fairness_report:

        report["fairness_markdown"] = framework.fairness_markdown()

        for column, payload in (
            framework.fairness_report.get("columns") or {}
        ).items():

            summary = payload["summary"]

            logger.info(
                "Fairness by %s: %d subgroup(s), gap=%s (%s vs %s)",
                column,
                len(payload["subgroups"]),
                summary.get("gap"),
                summary.get("best_group"),
                summary.get("worst_group"),
            )

            for flag in summary.get("flags", []):

                logger.warning("FAIRNESS (%s): %s", column, flag)

    for caveat in report["caveats"]:

        logger.info("CAVEAT: %s", caveat)

    if args.metrics:

        args.metrics.parent.mkdir(parents=True, exist_ok=True)

        args.metrics.write_text(
            json.dumps(report, indent=2, default=str)
        )

        logger.info("Metrics written to %s", args.metrics)

    logger.info("Model bundle saved to %s", saved_to)

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
