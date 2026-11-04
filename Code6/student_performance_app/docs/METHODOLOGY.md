# Methodology, Validation and Limitations

This document exists because the *interesting* part of this project is not the
Streamlit dashboard — it is the question **"what exactly is being predicted, and
does the evaluation support that claim?"** Every figure the application reports
should be read alongside this page.

---

## 1. The original problem: a circular target

Version 2.x of this system did the following:

1. Computed a **Composite Performance Index (CPI)** for every student from seven
   columns — `attendance_rate`, `study_hours`, `sleep_hours`,
   `social_media_hours`, `assignment_score`, `test_score`, `exam_score`.
2. Bucketed the CPI into `Low` / `Average` / `High`.
3. Trained classifiers to predict that bucket **from those same seven columns**.

Because the label is a deterministic function of the inputs
(`label = f(features)`), the classifiers were not predicting anything unknown.
They were re-learning the arithmetic of the CPI. That is why the shipped model
reports **97.8 % accuracy**, and why an equivalent pipeline trained on *pure
random noise* also reports ~98 % — a model cannot score better on real data than
on noise unless the task contains no real uncertainty.

Two consequences an examiner would raise immediately:

* *"Why use machine learning to reproduce a formula you defined yourself?"*
* *"What is being predicted that was not already known?"*

Version 3.0 fixes this by separating **what the label is** from **what the model
may see**.

---

## 2. The fix: prediction, not rule reconstruction

### 2.1 Two target definitions (`backend/config.py`)

| Definition | Label | Predictors | Status |
|---|---|---|---|
| `exam_band` (**default**) | End-of-term **examination band** (Low/Average/High) | attendance, study, sleep, social media, assignment, test score | Genuine prediction |
| `cpi` (legacy) | Composite Performance Index band | all seven columns | Rule reconstruction |

Under `exam_band` the examination score **is** the label and is therefore
**excluded from the predictors** — the prediction is made from term-time
behaviour and continuous assessment, which are available *before* the exam
result is known. `config.predictor_columns()` derives the predictor list from
the active definition, so a model can never accidentally be trained on the
source of its own label.

Switch definition without touching code:

```bash
SPAS_TARGET_DEFINITION=exam_band  streamlit run app.py   # default
SPAS_TARGET_DEFINITION=cpi        streamlit run app.py   # legacy, for comparison
```

`config.is_leaky_definition()` flags the legacy mode, and the UI, the
interpretation report and the saved bundle all carry that warning with the
results.

### 2.2 Balanced labels (`LABEL_POLICY`)

The old rule used fixed CPI cut-offs of 60 / 80 on a weighted average that
centres near 66, which produced roughly **77 % Average, 22 % Low and ~1 % High**.
A class with 1 % of the data cannot be learned, and a 1 %-class problem makes
headline accuracy meaningless.

Labels are now assigned by **quantile** by default (balanced thirds of the
observed distribution), with the fixed cut-offs still available:

```python
LABEL_POLICY = {
    "method": "quantile",          # or "fixed"
    "low_quantile": 1 / 3,
    "high_quantile": 2 / 3,
    "fixed_thresholds": (60.0, 80.0),
}
```

The thresholds actually used are recorded in
`CPIEngine.describe_target()`, written into the model bundle
(`target_info`), and displayed in the UI. Labels are therefore **relative to the
cohort**, not absolute — that is a stated design choice, not an accident.

If ties collapse the quantiles (e.g. many scores clipped at 100), the engine
falls back to the fixed thresholds and reports `balanced: false` rather than
silently training on a lopsided target.

### 2.3 Correct score scales (`SCORE_SCALE_RULES`)

The shipped data carries `assignment_score` on a **0–20 scale** (mean 10.0,
sd 6.0) while the CPI weight of 0.15 assumes 0–100. A perfect 20/20 assignment
therefore contributed ~3 points instead of 15 — the weight was effectively
**ten times smaller than documented**, and the CPI was dominated by the exam
score alone.

`DatasetValidator.normalize_score_scales()` now detects the scale (observed max
≤ 25 → assumed 0–20 → rescaled ×5), rescales to a percentage, and records a
warning that the UI displays. A genuine 0–100 column is left untouched.

---

## 3. Evaluation protocol

| What | How |
|---|---|
| Split | 80/20 hold-out, stratified, `random_state=42` |
| Cross-validation | Stratified 5-fold, **training fold only** |
| Scaling | `StandardScaler` fitted on the training fold; used by the linear models only |
| Models | Logistic Regression, Random Forest (200 trees), soft-voting ensemble, tuned Random Forest, calibrated Random Forest and calibrated Logistic Regression |
| Hyperparameter search | `RandomizedSearchCV`, 8 candidates, scoring `f1_macro`, 3 folds — **training fold only** |
| Calibration | `CalibratedClassifierCV` (sigmoid/Platt), 3 internal folds — **training fold only** |
| Calibration quality | Brier score and expected calibration error (ECE) per model |
| Selection | Highest macro F1 on the hold-out test set |

### Reported metrics, and why

`Metric` / `Why it is reported`:

* **Accuracy** — headline figure, but meaningless alone.
* **Macro F1** — unweighted mean over the three classes; collapses when the model
  ignores the minority class.
* **Balanced accuracy** — mean per-class recall.
* **Per-class precision / recall / F1 / support** — shows *which* band the model
  fails on.
* **ROC-AUC (one-vs-rest, weighted)** — threshold-free separability.
* **Baseline (majority class)** — the accuracy of always predicting the most
  common label. **Always compare accuracy with this, not with zero.**
* **Lift over baseline** — accuracy − baseline.
* **Brier score** — squared error of the probability vector; the number that
  decides whether the confidence figure can be trusted.
* **Expected calibration error (ECE)** — mean |confidence − accuracy| across
  confidence bins.

On the documented 2,000-row synthetic cohort (`scripts/make_synthetic_dataset.py`),
the selected model reports **0.655 accuracy / 0.653 macro F1 against a 0.335
baseline**. On pure-noise data the same pipeline reports chance level (~0.33),
which is the correct behaviour and the check that proves no leakage remains.

Tuning and calibration are reported honestly even when they do not help: on the
reference cohort both were neutral-to-slightly-negative (tuned forest 0.616 vs
0.638 untuned, calibrated Brier 0.455 vs 0.451), and the write-up says so rather
than quietly dropping the comparison.

---

## 4. Explainability

Two different questions, two different tools:

| Question | Method | Where |
|---|---|---|
| Which factors matter to the model overall? | Normalised coefficients (linear) / Gini importance (trees) | 🧠 Explainable AI page |
| Why did **this** student get **this** label? | **SHAP** contributions for that instance | 🔮 Prediction / 📋 Assess Student pages |

`backend/interpretation/shap_explainer.py` wraps SHAP with three safeguards:
SHAP is optional (the app degrades to global importance if it is missing),
Pipelines are unwrapped and their preprocessing applied before explaining, and
contributions are measured against a **sample of training rows** stored in the
model bundle — without that population baseline, every contribution would be
zero.

SHAP values are in the model's own units (log-odds for the linear models,
probability-scale for trees). They are signed contributions *towards or away
from the predicted class*, not causal effects, and not comparable across model
families. The UI says so.

---

## 5. Limitations (state these explicitly in the report)

1. **Labels are relative.** Under `exam_band` + quantile labelling, "High" means
   "top third of this cohort", not "above 80 %". Absolute thresholds are
   configurable (`LABEL_POLICY.method = "fixed"`) but were empirically
   unbalanced.
2. **Temporal validation is implemented, but the synthetic result is benign.**
   `backend/preprocessing/split_strategy.py` holds out whole academic periods
   (train 2024/25, test 2025/26) and `train_model.py --split temporal` runs it.
   On the synthetic cohort the change is roughly zero, which says more about the
   generator than about the model: its semester effect is a plain additive shift,
   and an additive shift preserves the signal structure. On real data this split
   is the one that matters — see `docs/TEMPORAL_AND_FAIRNESS.md` §2.
3. **Self-reported inputs.** `study_hours`, `sleep_hours` and
   `social_media_hours` are typically self-report and are the weakest features
   in the model.
4. **No causal claims.** Feature importance and SHAP describe associations
   learned by the model; they do not license statements such as "sleeping one
   more hour raises a grade".
5. **SAMPLED FROM A SINGLE INSTITUTION (if applicable).** Cohort-specific
   thresholds and small class counts limit external validity.
6. **Fairness disaggregation is implemented, and it already shows a pattern.**
   `backend/evaluation/fairness.py` reports accuracy, macro F1 and per-class
   recall by gender, programme and entry qualification, marks subgroups below 30
   students as unreliable, and refuses to let an unreliable subgroup set the gap.
   On the synthetic cohort the recall for the **Average** band is the weakest of
   the three classes in every subgroup (0.48–0.63). Subgroup parity on synthetic
   data is not evidence of parity on real data — see
   `docs/TEMPORAL_AND_FAIRNESS.md` §3.
7. **Calibration is not a performance gain.** Platt scaling can only rescale
   confidence; it cannot add information, and on the reference cohort it slightly
   increased Brier score and ECE. Whether to keep it is an empirical decision to
   be made on the real cohort from Table 5.4 of the results chapter.
8. **The 16.9 MB bundle in `models/` is the LEGACY model** trained on the
   circular CPI target (see `docs/MODEL_CARD.md`). It is kept so Assessment Mode
   can be demonstrated; retrain with `scripts/train_model.py` before quoting any
   number from it.

---

## 6. Reproducing the numbers

```bash
pip install -r requirements.txt

# the generator also writes semester / gender / programme /
# entry_qualification as CONTEXT columns (never predictors), so the
# temporal and fairness analyses below have something to work with

# generate a clearly-labelled synthetic cohort (no real student data)
python scripts/make_synthetic_dataset.py --rows 2000 --out data/synthetic_cohort.csv

# retrain, evaluate, export the bundle + a metrics JSON
python scripts/train_model.py --data data/synthetic_cohort.csv \
       --out models/best_model.pkl --metrics reports/metrics.json

# regenerate the report tables (comparison, per-class, confusion matrix, CV,
# calibration quality, tuning summary, feature importance)
python scripts/report_results.py --data data/synthetic_cohort.csv \
       --out reports/results.md --json reports/results.json

# verify the behaviour the pipeline claims
python -m pytest tests -q      # 53 tests
```

Every synthetic row is generated from an explicit, documented formula in
`scripts/make_synthetic_dataset.py`. Figures produced from it demonstrate that
the *code* works; they are not findings about real students.
