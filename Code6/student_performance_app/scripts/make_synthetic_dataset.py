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


def generate(rows: int, seed: int) -> pd.DataFrame:
    """Generate the synthetic cohort described in the module docstring."""

    rng = np.random.default_rng(seed)

    study = np.clip(rng.normal(6, 2, rows), 0, 12)

    attendance = np.clip(
        70 + 12 * rng.normal(0, 1, rows) + 3 * (study - 6),
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

    exam = np.clip(
        30
        + 4.5 * (study - 6)
        + 0.35 * attendance
        + 0.22 * test
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
        "--format",
        choices=("csv", "xlsx"),
        default=None,
        help="output format; inferred from --out when omitted",
    )

    args = parser.parse_args()

    if args.rows < 50:
        parser.error("--rows must be at least 50")

    frame = generate(args.rows, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    file_format = args.format or args.out.suffix.lstrip(".").lower()

    if file_format == "xlsx":

        frame.to_excel(args.out, index=False)

    else:

        frame.to_csv(args.out, index=False)

    print(f"Wrote {len(frame)} synthetic student rows to {args.out}")
    print("This data is INVENTED. Do not present it as real student data.")
    print(frame.describe().round(2).to_string())

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
