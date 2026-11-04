"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    backend/evaluation/fairness.py

Purpose:
    Disaggregate model performance by subgroup.

Why this exists
---------------
A single accuracy figure is an average, and averages hide the students
a system is most likely to harm. If the model is 0.65 accurate overall
but 0.42 accurate for one programme -- or never predicts "High" for one
group -- the average is not describing what happens to that group. On a
model that influences how a student is treated, that is the number that
matters.

What this reports, per subgroup
-------------------------------
    n                support (how much the number is worth)
    accuracy
    macro F1         the headline, because classes are balanced
    per-class recall the "who gets missed" question
    predicted distribution   the "who never gets flagged" question

and a disparity summary: best group, worst group, the gap between
them, and a warning when a subgroup is too small to convict anyone
(`min_support`).

Honesty rules built into the code
---------------------------------
* A subgroup below `min_support` is **labelled unreliable**, not
  quietly averaged in; its gap enters the summary as "insufficient
  evidence" rather than as a finding.
* Nothing here corrects, re-weights or "fixes" a disparity. Detecting
  one is the job; deciding what to do about it is the institution's.
* Encoded targets (0/1/2) and string labels are both accepted, and the
  label names in the output come from `PERFORMANCE_LABELS`, so a table
  cannot disagree with the rest of the system about what "Low" means.
====================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from sklearn.metrics import accuracy_score

from backend.config import PERFORMANCE_LABELS


# Subgroups smaller than this are reported but marked unreliable. 30 is
# a defensible default for a three-class problem: below it, a single
# extra correct prediction moves accuracy by more than 3 points.
DEFAULT_MIN_SUPPORT = 30

# Demographic columns worth checking when they exist. Nothing here is
# a predictor -- see backend/config.py predictor_columns().
CANDIDATE_GROUP_COLUMNS = (
    "gender",
    "sex",
    "programme",
    "program",
    "course",
    "department",
    "entry_qualification",
    "entry_mode",
    "level",
)


@dataclass
class SubgroupMetrics:
    """Performance for one subgroup."""

    group: str

    column: str

    n: int

    accuracy: float

    macro_f1: float

    per_class_recall: Dict[str, Optional[float]] = field(
        default_factory=dict
    )

    predicted_distribution: Dict[str, int] = field(
        default_factory=dict
    )

    actual_distribution: Dict[str, int] = field(
        default_factory=dict
    )

    warnings: List[str] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        """False when the subgroup is too small to draw a conclusion."""

        return not any(
            "insufficient support" in warning
            for warning in self.warnings
        )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "column": self.column,
            "group": self.group,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class_recall": {
                key: (
                    round(value, 4) if value is not None else None
                )
                for key, value in self.per_class_recall.items()
            },
            "predicted_distribution": self.predicted_distribution,
            "actual_distribution": self.actual_distribution,
            "reliable": self.reliable,
            "warnings": self.warnings,
        }


def _label_name(label: Any) -> str:
    """
    Turn an encoded class (0/1/2) into its name.

    Already-nameable labels pass through unchanged, so this works
    whether the caller hands over encoded or string targets.
    """

    try:

        index = int(label)

    except (TypeError, ValueError):

        return str(label)

    if index in PERFORMANCE_LABELS:

        return str(PERFORMANCE_LABELS[index])

    return str(label)


def _class_labels(y_true: pd.Series, y_pred: pd.Series) -> List[Any]:

    seen = pd.unique(
        pd.concat([y_true, y_pred], ignore_index=True)
    )

    try:

        return sorted(seen)

    except TypeError:  # pragma: no cover - mixed types

        return list(seen)


def disaggregate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    groups: pd.Series,
    column: str = "group",
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> List[SubgroupMetrics]:
    """
    Compute per-subgroup performance.

    Parameters
    ----------
    y_true, y_pred
        Aligned truth and prediction, encoded or named.
    groups
        Subgroup label per row, aligned with y_true.
    column
        Name of the demographic column (for the report).
    min_support
        Below this, the subgroup is marked unreliable.

    Returns
    -------
    list[SubgroupMetrics]
        Ordered by support, largest first, so an unreliable subgroup is
        never the first line an examiner reads.
    """

    true_series = pd.Series(list(y_true)).reset_index(drop=True)

    pred_series = pd.Series(list(y_pred)).reset_index(drop=True)

    group_series = pd.Series(list(groups)).reset_index(drop=True)

    if not (
        len(true_series) == len(pred_series) == len(group_series)
    ):

        raise ValueError(
            "y_true, y_pred and groups must be the same length "
            f"(got {len(true_series)}, {len(pred_series)}, "
            f"{len(group_series)}). A misaligned fairness table is "
            "worse than none."
        )

    frame = pd.DataFrame(
        {
            "y_true": true_series,
            "y_pred": pred_series,
            "group": group_series.astype(str),
        }
    )

    frame = frame[frame["group"].notna()]

    labels = _class_labels(frame["y_true"], frame["y_pred"])

    results: List[SubgroupMetrics] = []

    for group, subset in frame.groupby("group", sort=True):

        y_t = subset["y_true"]

        y_p = subset["y_pred"]

        n = int(len(subset))

        # Per-class figures are computed by hand so that
        #   * a class the model never predicts cannot vanish from the
        #     macro average, and
        #   * a class with no students in this subgroup is reported as
        #     "not estimable" (None) instead of a misleading 0.000.
        recalls: List[Optional[float]] = []

        f1_scores = []

        present = set(pd.unique(y_t)) | set(pd.unique(y_p))

        for label in labels:

            tp = int(
                ((y_t == label) & (y_p == label)).sum()
            )

            false_positive = int(
                ((y_t != label) & (y_p == label)).sum()
            )

            false_negative = int(
                ((y_t == label) & (y_p != label)).sum()
            )

            actual_support = tp + false_negative

            precision = (
                tp / (tp + false_positive)
                if (tp + false_positive)
                else 0.0
            )

            recall = (
                tp / actual_support if actual_support else 0.0
            )

            recalls.append(
                float(recall) if actual_support else None
            )

            # The macro average runs over the classes this subgroup
            # actually contains: a group with one class has a
            # single-class problem, not a model failure on the others.
            if label in present:

                f1_scores.append(
                    2 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0
                )

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        item = SubgroupMetrics(
            group=str(group),
            column=column,
            n=n,
            accuracy=float(accuracy_score(y_t, y_p)),
            macro_f1=float(macro_f1),
            per_class_recall={
                _label_name(label): (
                    float(value) if value is not None else None
                )
                for label, value in zip(labels, recalls)
            },
            predicted_distribution={
                _label_name(key): int(value)
                for key, value in y_p.value_counts().to_dict().items()
            },
            actual_distribution={
                _label_name(key): int(value)
                for key, value in y_t.value_counts().to_dict().items()
            },
        )

        if n < min_support:

            item.warnings.append(
                f"insufficient support: n={n} < {min_support}; treat this "
                "row as illustrative only"
            )

        # A group the model never places in a class: that is the
        # mechanism by which an early-warning system silently ignores
        # students, so it is stated rather than left in the table.
        for label in labels:

            name = _label_name(label)

            if item.predicted_distribution.get(name, 0) == 0:

                item.warnings.append(
                    f"never predicted '{name}' for this group"
                )

        missing_actual = [
            _label_name(label)
            for label in labels
            if item.actual_distribution.get(_label_name(label), 0) == 0
        ]

        if missing_actual:

            item.warnings.append(
                "no students in class(es) "
                + ", ".join(missing_actual)
                + " -- recall for those classes is not estimable here"
            )

        results.append(item)

    results.sort(key=lambda item: item.n, reverse=True)

    return results


def disparity_summary(
    results: Iterable[SubgroupMetrics],
    metric: str = "macro_f1",
) -> Dict[str, Any]:
    """
    Best/worst subgroup and the gap between them.

    Only reliable subgroups (n >= min_support) are allowed to define
    the gap; an unreliable one is reported as excluded rather than
    being allowed to look like a finding.
    """

    items = list(results)

    reliable = [item for item in items if item.reliable]

    excluded = [item.group for item in items if not item.reliable]

    summary: Dict[str, Any] = {
        "metric": metric,
        "groups_evaluated": len(items),
        "groups_reliable": len(reliable),
        "groups_excluded": excluded,
        "best_group": None,
        "best_value": None,
        "worst_group": None,
        "worst_value": None,
        "gap": None,
        "flags": [],
    }

    if not reliable:

        summary["flags"].append(
            "no subgroup had sufficient support; no disparity conclusion "
            "is possible"
        )

        return summary

    def value_of(item: SubgroupMetrics) -> float:

        return float(getattr(item, metric, 0.0) or 0.0)

    if len(reliable) < 2:

        summary["best_group"] = reliable[0].group
        summary["best_value"] = round(value_of(reliable[0]), 4)
        summary["worst_group"] = reliable[0].group
        summary["worst_value"] = round(value_of(reliable[0]), 4)
        summary["flags"].append(
            "only one subgroup had sufficient support "
            f"({reliable[0].group}, n={reliable[0].n}); no disparity "
            "conclusion is possible -- collecting more rows for the "
            "other subgroups is the prerequisite"
        )
        return summary

    best = max(reliable, key=value_of)

    worst = min(reliable, key=value_of)

    summary.update(
        {
            "best_group": best.group,
            "best_value": round(value_of(best), 4),
            "worst_group": worst.group,
            "worst_value": round(value_of(worst), 4),
            "gap": round(value_of(best) - value_of(worst), 4),
        }
    )

    if summary["gap"] and summary["gap"] >= 0.10:

        summary["flags"].append(
            f"{metric} differs by {summary['gap']:.3f} between "
            f"{best.group!r} and {worst.group!r} -- investigate before "
            "this model informs any decision about a student"
        )

    for item in reliable:

        if item.warnings:

            summary["flags"].extend(
                f"{item.group}: {warning}"
                for warning in item.warnings
                if "insufficient support" not in warning
            )

    return summary


def fairness_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    dataframe: pd.DataFrame,
    group_columns: Optional[Iterable[str]] = None,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> Dict[str, Any]:
    """
    Run `disaggregate` over every usable demographic column.

    Columns not present in the frame are skipped with a note, so a
    dataset without a gender column produces a shorter report, not an
    exception.
    """

    if group_columns is None:

        requested = [
            column
            for column in CANDIDATE_GROUP_COLUMNS
            if column in dataframe.columns
        ]

    else:

        requested = list(group_columns)

    report: Dict[str, Any] = {
        "columns": {},
        "skipped": [],
        "min_support": min_support,
    }

    for column in requested:

        if column not in dataframe.columns:

            report["skipped"].append(column)

            continue

        values = dataframe[column].astype(str)

        if values.nunique() < 2:

            report["skipped"].append(column)

            continue

        results = disaggregate(
            y_true=y_true,
            y_pred=y_pred,
            groups=values,
            column=column,
            min_support=min_support,
        )

        report["columns"][column] = {
            "subgroups": [item.to_dict() for item in results],
            "summary": disparity_summary(results),
        }

    if not report["columns"]:

        report["note"] = (
            "No demographic column suitable for disaggregation was found "
            f"(looked for {list(CANDIDATE_GROUP_COLUMNS)}). Fairness "
            "cannot be assessed from this dataset -- which is itself a "
            "finding worth reporting."
        )

    return report


def to_markdown(report: Dict[str, Any]) -> str:
    """Render a fairness report as Markdown tables."""

    lines: List[str] = []

    lines.append(
        "All figures are hold-out test-set figures, disaggregated by "
        "subgroup. Subgroups with n < "
        f"{report.get('min_support')} are marked unreliable."
    )
    lines.append("")

    if report.get("note"):

        lines.append(f"_{report['note']}_")
        lines.append("")

    for column, payload in report.get("columns", {}).items():

        summary = payload["summary"]

        lines.append(f"### By {column}")

        lines.append("")

        lines.append(
            "| Subgroup | n | Accuracy | Macro F1 | "
            + " | ".join(
                f"Recall {name}"
                for name in PERFORMANCE_LABELS.values()
            )
            + " | Flags |"
        )

        lines.append(
            "|---|---|---|---|" + "---|" * len(PERFORMANCE_LABELS) + "---|"
        )

        for row in payload["subgroups"]:

            recalls = " | ".join(
                (
                    f"{row['per_class_recall'][name]:.3f}"
                    if row["per_class_recall"].get(name) is not None
                    else "n/a"
                )
                for name in PERFORMANCE_LABELS.values()
            )

            flags = []

            if not row["reliable"]:

                flags.append("unreliable (small n)")

            if any(
                "never predicted" in warning
                for warning in row["warnings"]
            ):

                flags.append("never predicted a class")

            lines.append(
                f"| {row['group']} | {row['n']} | "
                f"{row['accuracy']:.3f} | {row['macro_f1']:.3f} | "
                f"{recalls} | {', '.join(flags) or '—'} |"
            )

        lines.append("")

        if summary.get("gap") is not None:

            lines.append(
                f"Gap in {summary['metric']}: "
                f"{summary['best_group']} {summary['best_value']:.3f} vs "
                f"{summary['worst_group']} {summary['worst_value']:.3f} "
                f"= **{summary['gap']:.3f}**."
            )

            lines.append("")

        for flag in summary.get("flags", []):

            lines.append(f"- ⚠️ {flag}")

        if summary.get("flags"):

            lines.append("")

    return "\n".join(lines)
