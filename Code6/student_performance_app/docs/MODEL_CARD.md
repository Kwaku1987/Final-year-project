# Model Card — `models/best_model.pkl`

A model card for the bundle that ships with this repository, plus the rules for
any bundle produced from now on. Written to the standard used in the
accountability literature (Mitchell et al., 2019): what it is, what it was
trained on, what it can and cannot be used for.

---

## 1. Status of the committed bundle

**Certified production bundle.** Since v3.3.1 the shipped file is an
`exam_band` model, not the legacy CPI artefact.

| Field | Value (read directly from the pickle) |
|---|---|
| Bundle version | `7.2` |
| `bundle_role` | `production` |
| `label_source_in_predictors` | `false` |
| Target | `exam_band` — Low/Average/High bands of the **examination score**, quantile thresholds **64.67 / 80.34** |
| Predictors | six term-time columns; **`exam_score` is not among them** |
| Training samples | 2,000 (1,510 train / 490 test) |
| Validation | **temporal** — trained on 2024/25-S1 + 2024/25-S2, tested on 2025/26-S1 |
| Algorithms in the bundle | Logistic Regression, Random Forest, Soft-Voting Ensemble, Random Forest (Tuned), Random Forest (Calibrated), Logistic Regression (Calibrated) — **Soft-Voting Ensemble selected** |
| Hold-out metrics | accuracy 0.653 · macro F1 **0.655** · balanced accuracy 0.660 · ROC-AUC 0.837 · Brier 0.441 · ECE 0.027 |
| Baseline | majority class 0.363 → **lift +0.29** |
| Fairness | embedded: gender gap 0.065, programme 0.053, entry qualification 0.046 (macro F1) |
| scikit-learn | `1.9.1` (pinned in `requirements.txt`) |
| Data | **synthetic** — `scripts/make_synthetic_dataset.py --rows 2000`, seed 42 |

> ⚠️ **The metrics describe the synthetic cohort.** They are not evidence about
> real students, and the temporal result is benign because the generator's
> semester effect is an additive shift (see `docs/TEMPORAL_AND_FAIRNESS.md` §2).
> Retrain on a real, de-identified cohort before quoting any figure.

**Why the 97.8 % figure is gone.** It came from a target computed from the same
columns the model consumed (the CPI formula). That is rule reconstruction, not
prediction, and since v3.2 it cannot reach a production bundle at all:
`save_best_model()` refuses a leaky target unless `allow_leaky_target=True` is
passed explicitly, and when it is, the bundle is stamped
`bundle_role="rule_reconstruction_baseline"` and carries the warning in its own
`caveats`. `scripts/preflight.py --production` fails on such a bundle.

**Why the committed artefact is still a compromise.** It is trained on synthetic
data, and its labels come from the examination score of an invented cohort. It
is the honest demonstration model, not a validated predictor.

### Why the pickle is pinned to scikit-learn 1.9.x

The bundle was written by scikit-learn 1.9.0. Loading it with 1.6.1 raises:

```
AttributeError: 'LogisticRegression' object has no attribute 'multi_class'
```

`requirements.txt` therefore pins `scikit-learn==1.9.1`, and
`StudentPerformanceFramework._safe_load_bundle()` converts a version clash into
an actionable message instead of a stack trace. If you retrain, re-pin.

### Size

9.9 MB committed to Git (six fitted candidates, two of them calibrated, written
with `joblib.dump(..., compress=3)`; uncompressed it is ~50 MB). That is smaller
than the legacy artefact it replaces, despite carrying six models, the tuning
summary, the fairness report and the explanation background.

Bundles are build outputs: `.gitignore` covers `models/*.pkl` and whitelists only
`best_model.pkl`. For anything larger, use a release asset or Git LFS rather than
the repository history.

---

## 2. Intended use

**Intended**

* Demonstration and coursework: a lecturer-facing decision-*support* view of a
  student's likely examination band, with the factors behind it.
* Teaching material for a documented ML pipeline (validation → target building
  → training → honest evaluation → explainability → reporting).

**Not intended**

* Any automated academic decision (progression, probation, scholarship,
  disciplinary action).
* Ranking students against each other or against students from other cohorts.
* Use with a cohort whose feature distributions differ materially from the
  training cohort — thresholds are relative to that cohort.
* Any claim of causal effect ("more sleep improves grades").

---

## 3. Training data

* **v3.0 and later:** supplied by the operator of the system. The repository
  ships **no real student data**. `scripts/make_synthetic_dataset.py` generates
  a clearly-labelled synthetic cohort for demonstration and testing only.
* **The replaced legacy bundle (still in git history):** 10,000 records,
  provenance not recorded in
  the repository. Feature means recoverable from the stored scaler:

  | Feature | Mean | SD |
  |---|---|---|
  | attendance_rate | 69.85 | 17.64 |
  | study_hours | 5.99 | 3.17 |
  | sleep_hours | 6.50 | 1.71 |
  | social_media_hours | 6.06 | 3.13 |
  | assignment_score | 10.00 | 6.04 |
  | test_score | 64.80 | 17.51 |
  | exam_score | 86.73 | 14.96 |

  The `assignment_score` mean of 10 with SD 6 indicates a 0–20 scale, which
  v3.0 now detects and rescales (see `docs/METHODOLOGY.md` §2.3).

---

## 4. Evaluation

Current pipeline (`scripts/train_model.py`, `pytest tests -q`): 80/20 stratified
hold-out plus 5-fold cross-validation on the training fold; accuracy, macro F1,
balanced accuracy, per-class precision/recall/F1, one-vs-rest ROC-AUC, and a
majority-class baseline for every model.

**Every reported accuracy must be read next to the baseline.** A model that
scores 0.90 accuracy on a target with a 0.90 majority class has learned nothing.

---

## 5. Ethical and legal notes

* Student marks and identifiers are personal data under Ghana's **Data
  Protection Act, 2012 (Act 843)**. Lawful basis, retention and access control
  are the deploying institution's responsibility; see
  `docs/DATA_PROTECTION.md`.
* Training Mode has no password gate (see `docs/DATA_PROTECTION.md`); Assessment
  Mode requires no training data.
* No fairness disaggregation is performed. Before any real deployment, metrics
  should be reported by gender, programme and entry qualification, and subgroup
  performance compared.
* Predictions are advisory. Human judgement must remain decisive.

---

## 6. Retraining checklist (release gate)

1. `pip install -r requirements.txt` (pins enforced).
2. Data validated — read the warnings the app surfaces; confirm ranges and the
   `assignment_score` scale.
3. Target definition chosen and justified (`exam_band` unless you can defend the
   legacy CPI framing), thresholds recorded.
4. Metrics reported **with** baseline, macro F1, balanced accuracy, per-class
   breakdown, Brier score and ECE; ideally across semesters, not just a random
   split. `scripts/report_results.py` produces them.
5. Caveats verified in the interpretation report (`report.caveats`).
6. `python -m pytest tests -q` green.
7. Tuning and calibration decisions recorded — which search ran, which
   configurations were compared, and whether calibration improved Brier/ECE on
   this cohort (keep it only if it does).
8. This file updated: date, sample count, version pins, chosen model, metric
   table, known limitations.
