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
| Models | Logistic Regression, Random Forest (200 trees), soft-voting ensemble |
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

On a well-separated synthetic cohort with no leaked label, this pipeline reports
roughly 0.8 accuracy against a 0.55 baseline with macro F1 ≈ 0.6 — a believable
result for a noisy social-science problem. On pure-noise data it reports chance
level (~0.33), which is the correct behaviour.

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
2. **No temporal validation.** All results come from a random split of one
   cohort. A genuine early-warning system should be validated
   **across semesters** (train on 2024/25, test on 2025/26) — that is the only
   split that proves the model generalises forward in time.
3. **Self-reported inputs.** `study_hours`, `sleep_hours` and
   `social_media_hours` are typically self-report and are the weakest features
   in the model.
4. **No causal claims.** Feature importance and SHAP describe associations
   learned by the model; they do not license statements such as "sleeping one
   more hour raises a grade".
5. **SAMPLED FROM A SINGLE INSTITUTION (if applicable).** Cohort-specific
   thresholds and small class counts limit external validity.
6. **No fairness analysis.** Results are not disaggregated by gender, programme
   or entry qualification. Any deployment decision affecting students should be,
   and the model should not be used as the sole basis for one.
7. **The 16.9 MB bundle in `models/` is the LEGACY model** trained on the
   circular CPI target (see `docs/MODEL_CARD.md`). It is kept so Assessment Mode
   can be demonstrated; retrain with `scripts/train_model.py` before quoting any
   number from it.

---

## 6. Reproducing the numbers

```bash
pip install -r requirements.txt

# generate a clearly-labelled synthetic cohort (no real student data)
python scripts/make_synthetic_dataset.py --rows 2000 --out data/synthetic_cohort.csv

# retrain, evaluate, export the bundle + a metrics JSON
python scripts/train_model.py --data data/synthetic_cohort.csv --out models/best_model.pkl

# verify the behaviour the pipeline claims
python -m pytest tests -q
```

Every synthetic row is generated from an explicit, documented formula in
`scripts/make_synthetic_dataset.py`. Figures produced from it demonstrate that
the *code* works; they are not findings about real students.
