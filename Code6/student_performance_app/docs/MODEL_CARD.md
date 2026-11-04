# Model Card — `models/best_model.pkl`

A model card for the bundle that ships with this repository, plus the rules for
any bundle produced from now on. Written to the standard used in the
accountability literature (Mitchell et al., 2019): what it is, what it was
trained on, what it can and cannot be used for.

---

## 1. Status of the committed bundle

| Field | Value (read directly from the pickle) |
|---|---|
| Bundle version | `6.0` (pre-v3.0) |
| Created | `2026-08-25 21:37:40` |
| Training samples | 10,000 |
| scikit-learn version | `1.9.0` |
| Algorithms | Logistic Regression (**selected as best**) and Random Forest (200 trees) |
| Recorded metrics | accuracy 0.9780 · precision 0.9778 · recall 0.9780 · F1 0.9779 |
| Predictors | all seven columns, **including `exam_score`** |
| Target | Composite Performance Index band (legacy definition) |

> ⚠️ **This bundle is a legacy artefact and its headline numbers must not be
> presented as predictive accuracy.** It was trained on a target computed from
> the same seven columns it predicts from, so 97.8 % measures how well the model
> reproduces the CPI formula. See `docs/METHODOLOGY.md` §1.
>
> It is retained only so that Assessment Mode can be demonstrated without
> retraining. The app labels it as legacy at load time
> (`framework.legacy_bundle == True`) and attaches that caveat to every report.

### Why the pickle is pinned to scikit-learn 1.9.x

The bundle was written by scikit-learn 1.9.0. Loading it with 1.6.1 raises:

```
AttributeError: 'LogisticRegression' object has no attribute 'multi_class'
```

`requirements.txt` therefore pins `scikit-learn==1.9.1`, and
`StudentPerformanceFramework._safe_load_bundle()` converts a version clash into
an actionable message instead of a stack trace. If you retrain, re-pin.

### Size

16.9 MB, committed to Git. That is tolerable once, not as a habit: future
bundles are covered by `.gitignore` (`models/*.pkl`), and a bundle this size
belongs in a release asset or Git LFS rather than in the history.

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
* **The committed legacy bundle:** 10,000 records, provenance not recorded in
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
* Training Mode is password-gated; Assessment Mode requires no training data.
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
