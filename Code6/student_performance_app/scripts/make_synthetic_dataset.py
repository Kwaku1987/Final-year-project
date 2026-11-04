"""
====================================================================
Student Performance Assessment System
====================================================================

Script:
    scripts/make_synthetic_dataset.py

Purpose:
    Generate a SYNTHETIC student cohort for demonstrations and tests.

Why this exists
---------------
The repository ships no student data (correctly -- it would be
personal data). Without a dataset nobody can reproduce the pipeline,
run the tests, or see the app working end to end. This script fills
that gap with data that is obviously invented, documented, and
seeded.

Every row is produced by the explicit formulas below. The exam score
is influenced by behaviour and earlier assessment, but never
determined by them -- there is genuine noise, so an honest model
scores well below 100%.

    study_hours      ~ N(6, 2)                  clipped 0..12
    attendance_rate  ~ 70 + 12 N(0,1) + 3(study - 6)   clipped 0..100
    sleep_hours      ~ N(6.5, 1.2)              clipped 0..12
    social_media     ~ N(6, 1.5) - 0.4(study - 6)      clipped 0..12
    assignment_score ~ N(10, 1.2) + 1.1(study - 6)     clipped 0..20
                       (0-20 scale on purpose: the pipeline detects
                        and rescales it -- see docs/METHODOLOGY.md)
    test_score       ~ 60 + 3.5(study - 6) + 0.35 attendance + 8 N(0,1)
    exam_score       ~ 30 + 4.5(study - 6) + 0.35 attendance
                       + 0.22 test_score + 10 N(0,1)

Usage
-----
    python scripts/make_synthetic_dataset.py --rows 2000 \
        --out data/synthetic_cohort.csv

    python scripts/make_synthetic_dataset.py --format xlsx \
        --out data/synthetic_cohort.xlsx

Output columns match backend/config.py FEATURE_COLUMNS, so the file
can be uploaded directly to the 📂 Dataset page.

Context columns (v3.3)
----------------------
Four columns are written that are NOT predictors. They exist so the
two questions a single accuracy figure cannot answer become testable:

    semester            2024/25-S1, 2024/25-S2, 2025/26-S1
    gender              Female / Male
    programme           BSc IT / BSc Computer Science / BSc IS
    entry_qualification WASSCE / Diploma / Mature / Transfer

* `semester` makes temporal validation possible (train on the first
  two periods, test on the last) -- see
  backend/preprocessing/split_strategy.py and
  `train_model.py --split temporal`.
* the other three make fairness disaggregation possible -- see
  backend/evaluation/fairness.py.

Two small structural effects are INJECTED on purpose, so those
analyses have something to find rather than reporting a tidy zero:

* each semester carries a cohort effect on the exam score
  (-2.0, +0.5, +1.5 marks) -- a stand-in for differences in marking,
  timetable or cohort intake;
* each programme carries an attendance offset (-3, +4, 0 points) --
  a stand-in for differences in contact hours.

Neither is a claim about real students. They are properties of this
generator, and the docs say so.
====================================================================
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

import numpy as np
import pandas as pd

# Make `backend` importable when run as a script.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from backend.config import FEATURE_COLUMNS  # noqa: E402


# Context columns: carried alongside the predictors so temporal
# validation and fairness disaggregation are possible, never fed to
# the model (see backend/config.py predictor_columns()).
CONTEXT_COLUMNS = [
    "semester",
    "gender",
    "programme",
    "entry_qualification",
]

SEMESTERS = ["2024/25-S1", "2024/25-S2", "2025/26-S1"]

# Injected, documented structure -- not a claim about real students.
SEMESTER_EFFECT = {
    "2024/25-S1": -2.0,
    "2024/25-S2": 0.5,
    "2025/26-S1": 1.5,
}

SEMESTER_SHARE = [0.40, 0.35, 0.25]

PROGRAMMES = [
    "BSc Information Technology",
    "BSc Computer Science",
    "BSc Information Systems",
]

PROGRAMME_ATTENDANCE_EFFECT = {
    "BSc Information Technology": 4.0,
    "BSc Computer Science": -3.0,
    "BSc Information Systems": 0.0,
}

ENTRY_QUALIFICATIONS = ["WASSCE", "Diploma", "Mature", "Transfer"]

ENTRY_SHARE = [0.68, 0.18, 0.08, 0.06]


def generate(
    rows: int,
    seed: int,
    include_context: bool = True,
) -> pd.DataFrame:
    """Generate the synthetic cohort described in the module docstring."""

    rng = np.random.default_rng(seed)

    study = np.clip(rng.normal(6, 2, rows), 0, 12)

    if include_context:
        semester = rng.choice(SEMESTERS, size=rows, p=SEMESTER_SHARE)
        programme = rng.choice(PROGRAMMES, size=rows)
        gender = rng.choice(["Female", "Male"], size=rows, p=[0.48, 0.52])
        entry = rng.choice(
            ENTRY_QUALIFICATIONS, size=rows, p=ENTRY_SHARE
        )
    else:
        semester = np.array(["2024/25-S1"] * rows)
        programme = np.array(["BSc Information Technology"] * rows)
        gender = np.array(["Female"] * rows)
        entry = np.array(["WASSCE"] * rows)

    attendance_effect = np.array(
        [PROGRAMME_ATTENDANCE_EFFECT[p] for p in programme]
    )

    attendance = np.clip(
        70
        + 12 * rng.normal(0, 1, rows)
        + 3 * (study - 6)
        + attendance_effect,
        0,
        100,
    )

    sleep = np.clip(rng.normal(6.5, 1.2, rows), 0, 12)

    social = np.clip(
        rng.normal(6, 1.5, rows) - 0.4 * (study - 6),
        0,
        12,
    )

    # Deliberately a 0-20 scale: the validator detects this, warns,
    # and rescales to a percentage before the CPI or the model sees
    # it (v3.0 fix for the mis-weighted assignment score).
    assignment = np.clip(
        rng.normal(10, 1.2, rows) + 1.1 * (study - 6),
        0,
        20,
    )

    test = np.clip(
        60
        + 3.5 * (study - 6)
        + 0.35 * attendance
        + 8 * rng.normal(0, 1, rows),
        0,
        100,
    )

    semester_effect = np.array(
        [SEMESTER_EFFECT[s] for s in semester]
    )

    exam = np.clip(
        30
        + 4.5 * (study - 6)
        + 0.35 * attendance
        + 0.22 * test
        + semester_effect
        + 10 * rng.normal(0, 1, rows),
        0,
        100,
    )

    frame = pd.DataFrame(
        {
            "attendance_rate": attendance,
            "study_hours": study,
            "sleep_hours": sleep,
            "social_media_hours": social,
            "assignment_score": assignment,
            "test_score": test,
            "exam_score": exam,
        }
    )

    # Round to realistic reporting precision.
    frame = frame.round(2)

    if include_context:

        frame["semester"] = semester
        frame["gender"] = gender
        frame["programme"] = programme
        frame["entry_qualification"] = entry

        return frame[FEATURE_COLUMNS + CONTEXT_COLUMNS]

    return frame[FEATURE_COLUMNS]


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic student cohort (no real data) for "
            "demonstration and testing."
        )
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=2000,
        help="number of students to generate (default: 2000)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed, for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/synthetic_cohort.csv"),
        help="output path (default: data/synthetic_cohort.csv)",
    )

    parser.add_argument(
        "--no-context",
        action="store_true",
        help=(
            "omit the semester/gender/programme/entry_qualification "
            "columns (then temporal validation and fairness "
            "disaggregation have nothing to work with)"
        ),
    )

    parser.add_argument(
        "--format",
        choices=("csv", "xlsx"),
        default=None,
        help="output format; inferred from --out when omitted",
    )

    args = parser.parse_args()

    if args.rows < 50:
        parser.error("--rows must be at least 50")

    frame = generate(
        args.rows, args.seed, include_context=not args.no_context
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    file_format = args.format or args.out.suffix.lstrip(".").lower()

    if file_format == "xlsx":

        frame.to_excel(args.out, index=False)

    else:

        frame.to_csv(args.out, index=False)

    print(f"Wrote {len(frame)} synthetic student rows to {args.out}")
    print("This data is INVENTED. Do not present it as real student data.")
    print(frame.describe().round(2).to_string())

    if not args.no_context:

        print()
        print("Context columns (NOT predictors):")
        for column in CONTEXT_COLUMNS:

            counts = frame[column].value_counts().to_dict()

            print(f"  {column}: {counts}")

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
