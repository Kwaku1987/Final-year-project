"""
====================================================================
Student Performance Assessment System
====================================================================

Script:
    scripts/report_results.py

Purpose:
    Produce the tables you actually paste into a report: a model
    comparison table, per-class metrics for the selected model, the
    confusion matrix, cross-validation summaries, calibration
    quality and the label distribution.

Why this exists
---------------
Copying numbers out of a Streamlit page by hand is how results become
transcription errors. This script runs the same pipeline
headlessly, writes a Markdown file and a JSON file, and prints the
caveats that must accompany the numbers.

Usage
-----
    python scripts/report_results.py --data data/cohort.csv \
        --out reports/results.md --json reports/results.json

    python scripts/report_results.py --data data/cohort.csv \
        --definition cpi          # legacy comparison

Exit codes: 0 success, 1 failure.
====================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s | %(message)s",
)


def load_dataset(path: Path) -> pd.DataFrame:

    if path.suffix.lower() in (".xlsx", ".xls"):

        return pd.read_excel(path)

    return pd.read_csv(path)


def build_tables(framework) -> dict:
    """Collect every number a results chapter needs."""

    comparison = []

    for name, result in framework.comparison_results.items():

        metrics = result["metrics"]

        cv = framework.cross_validation.get(name, {})

        comparison.append(
            {
                "model": name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "roc_auc": metrics.get("roc_auc"),
                "brier_score": metrics.get("brier_score"),
                "calibration_error": metrics.get("calibration_error"),
                "baseline": metrics.get("baseline_majority"),
                "lift": metrics.get("lift_over_baseline"),
                "cv_mean": cv.get("mean_accuracy"),
                "cv_std": cv.get("std_accuracy"),
                "n_test": metrics.get("n_test"),
                "per_class": metrics.get("per_class"),
            }
        )

    best = framework.best_model_name

    best_result = framework.comparison_results.get(best, {})

    confusion = best_result.get("confusion_matrix")

    framework.interpret()

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": framework.target_description,
        "predictors": list(framework.required_features),
        "best_model": best,
        "best_model_is_calibrated": framework.best_model_is_calibrated,
        "tuning": framework.tuning_results,
        "comparison": comparison,
        "confusion_matrix": (
            confusion.tolist() if confusion is not None else None
        ),
        "label_distribution": (
            framework.validated_data["performance_level"]
            .value_counts()
            .to_dict()
            if framework.validated_data is not None
            else None
        ),
        "caveats": framework.methodology_caveats(),
        "validation_warnings": framework.validator_warnings,
        "feature_importance": (
            framework.interpretation_report.feature_importance[
                ["feature", "importance", "direction"]
            ].to_dict(orient="records")
            if framework.interpretation_report is not None
            else []
        ),
    }


def to_markdown(payload: dict) -> str:
    """Render the collected numbers as a Markdown fragment."""

    lines = []

    lines.append("# Results (generated)")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("")

    target = payload["target"]

    lines.append("## Target")
    lines.append("")
    lines.append(f"- Definition: `{target['definition']}`")
    lines.append(f"- Threshold method: `{target['method']}`")
    lines.append(
        f"- Thresholds: {target['low_threshold']} / "
        f"{target['high_threshold']}"
    )
    lines.append(f"- Distribution: {target['distribution']}")
    lines.append(
        f"- Predictors ({len(payload['predictors'])}): "
        + ", ".join(payload["predictors"])
    )
    lines.append("")

    if payload["label_distribution"]:

        lines.append(
            f"- Class counts in the full set: "
            f"{payload['label_distribution']}"
        )
        lines.append("")

    lines.append("## Model comparison")
    lines.append("")
    lines.append(
        "| Model | Accuracy | Macro F1 | Balanced acc. | ROC-AUC | "
        "Brier | ECE | Baseline | Lift | CV mean ± SD |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|"
    )

    for row in payload["comparison"]:

        def fmt(value, digits=3):

            if value is None:
                return "—"

            return f"{value:.{digits}f}"

        lines.append(
            "| {model} | {acc} | {f1} | {bal} | {auc} | {brier} | "
            "{ece} | {base} | {lift} | {cv} ± {sd} |".format(
                model=row["model"],
                acc=fmt(row["accuracy"]),
                f1=fmt(row["macro_f1"]),
                bal=fmt(row["balanced_accuracy"]),
                auc=fmt(row["roc_auc"]),
                brier=fmt(row["brier_score"]),
                ece=fmt(row["calibration_error"]),
                base=fmt(row["baseline"]),
                lift=(
                    f"{row['lift']:+.3f}"
                    if row["lift"] is not None
                    else "—"
                ),
                cv=fmt(row["cv_mean"]),
                sd=fmt(row["cv_std"]),
            )
        )

    lines.append("")

    best_row = next(
        (
            row
            for row in payload["comparison"]
            if row["model"] == payload["best_model"]
        ),
        None,
    )

    if best_row and best_row["per_class"]:

        lines.append(
            f"## Per-class metrics — {payload['best_model']} "
            f"(calibrated: {payload['best_model_is_calibrated']})"
        )
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")

        for label, values in best_row["per_class"].items():

            lines.append(
                f"| {label} | {values['precision']:.3f} | "
                f"{values['recall']:.3f} | {values['f1']:.3f} | "
                f"{values['support']} |"
            )

        lines.append("")

    if payload["confusion_matrix"]:

        lines.append("## Confusion matrix (rows = actual, cols = predicted)")
        lines.append("")
        lines.append("| actual \\ predicted | Low | Average | High |")
        lines.append("|---|---|---|---|")

        for label, row in zip(
            ("Low", "Average", "High"),
            payload["confusion_matrix"],
        ):

            lines.append(
                f"| **{label}** | " + " | ".join(str(v) for v in row) + " |"
            )

        lines.append("")

    if payload["tuning"]:

        lines.append("## Hyperparameter search")
        lines.append("")

        for name, summary in payload["tuning"].items():

            lines.append(f"- **{name}** — {summary['best_params']}")
            lines.append(
                f"  - {summary['candidates_evaluated']} candidates, "
                f"scoring `{summary['scoring']}`, best CV "
                f"{summary['best_cv_score']:.4f} (training fold only)"
            )

        lines.append("")

    if payload["feature_importance"]:

        lines.append("## Global feature importance (selected model)")
        lines.append("")
        lines.append("| Rank | Feature | Importance | Direction |")
        lines.append("|---|---|---|---|")

        for rank, item in enumerate(
            payload["feature_importance"], start=1
        ):

            lines.append(
                f"| {rank} | {item['feature']} | "
                f"{item['importance']:.3f} | {item['direction']} |"
            )

        lines.append("")

    lines.append("## Caveats (must be quoted with the numbers)")
    lines.append("")

    for caveat in payload["caveats"]:

        lines.append(f"- {caveat}")

    lines.append("")

    if payload["validation_warnings"]:

        lines.append("## Data notes")
        lines.append("")

        for warning in payload["validation_warnings"]:

            lines.append(f"- {warning}")

        lines.append("")

    return "\n".join(lines)


def main(argv=None) -> int:

    parser = argparse.ArgumentParser(
        description="Generate results tables for the report."
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="CSV/XLSX dataset",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/results.md"),
        help="Markdown output path",
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional JSON output path",
    )

    parser.add_argument(
        "--definition",
        choices=("exam_band", "cpi"),
        default=None,
        help="target definition override",
    )

    args = parser.parse_args(argv)

    if args.definition:

        import os

        os.environ["SPAS_TARGET_DEFINITION"] = args.definition

    from framework.framework_factory import FrameworkFactory

    try:

        dataframe = load_dataset(args.data)

    except Exception as error:

        print(f"Could not read {args.data}: {error}", file=sys.stderr)

        return 1

    framework = FrameworkFactory.create()

    try:

        framework.run_pipeline(dataframe)

    except Exception as error:

        print(f"Pipeline failed: {error}", file=sys.stderr)

        return 1

    payload = build_tables(framework)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    args.out.write_text(to_markdown(payload))

    print(f"Wrote {args.out}")

    if args.json:

        args.json.parent.mkdir(parents=True, exist_ok=True)

        args.json.write_text(
            json.dumps(payload, indent=2, default=str)
        )

        print(f"Wrote {args.json}")

    print(
        f"Best model: {payload['best_model']} "
        f"(calibrated: {payload['best_model_is_calibrated']})"
    )

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
