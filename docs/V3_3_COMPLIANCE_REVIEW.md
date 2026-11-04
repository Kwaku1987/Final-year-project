# Version 3.3 Compliance Review

**Scope:** training pipelines, target definitions, validation strategy, test suite,
retraining CLI and the shipped model bundle.
**Method:** each requirement was checked against the code, then re-verified by
running it — `pytest`, the CLI, `preflight.py` and the smoke test. Nothing below
is asserted on the strength of a docstring.

Version 3.3.1 · Python 3.12+ · `scikit-learn==1.9.1` (pinned) · **100 tests passing**

---

## 1. Verdict

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1a | `exam_band` labels derived purely from examination performance | **Pass** | `backend/config.py: predictor_columns()` → 6 predictors, `exam_score` absent; `is_leaky_definition("exam_band") is False` |
| 1b | Label source strictly excluded from predictor features | **Pass, enforced** | `tests/test_signal_integrity.py::test_predictions_are_invariant_to_the_label_source` — passing `exam_score=5` vs `99` yields an identical prediction |
| 1c | Legacy CPI treated explicitly as rule reconstruction / baseline | **Now enforced** | `save_best_model()` refuses a leaky target; `train_model.py` exits 2 without `--allow-leaky-target`; bundle stamped `bundle_role="rule_reconstruction_baseline"` |
| 2 | Dynamic quantile thresholding replaces fixed cut-offs | **Pass** | `LABEL_POLICY["method"] == "quantile"` (thirds of the observed distribution); `fixed` reachable only via `--label-method fixed`; degenerate distribution handled with an explicit fallback |
| 3a | Majority baseline, macro F1, balanced accuracy always reported | **Pass** | `ModelEvaluator.evaluate()`; asserted in `test_honest_metric_fields_are_always_present` |
| 3b | Pipeline verified against random noise | **Added and passing** | §3 below — macro F1 **0.289** vs **0.671** structured; ROC-AUC **0.507** ≈ chance |
| 4a | Temporal splits partition by semester, no period on both sides | **Pass** | `temporal_split()`; `test_no_period_appears_on_both_sides_of_a_temporal_split` checks **row indices**, not labels |
| 4b | Disaggregation by gender / programme / entry qualification | **Pass** | `backend/evaluation/fairness.py`; report embedded in the bundle and printed by the CLI |
| 5a | CLI produces bundles + metrics JSON to these standards | **Pass** | `train_model.py` writes `bundle_role`, `split`, `fairness` into both artefacts; `reports/metrics.json` verified |
| 5b | Existing tests pass, and enforce the rules | **Pass** | `pytest tests -q` → **100 passed in 42 s** (86 → 100) |
| — | Legacy 97.8 % CPI accuracy kept out of production metrics | **Pass** | It appears only in explanatory text ("why this is wrong") and once in the chapter's v2-vs-v3 comparison row; no production bundle, report table or metric contains it |

---

## 2. What changed in this pass (3.3.1)

The standards were mostly implemented in 3.1–3.3. What was missing was
**enforcement** — several rules were documented but nothing failed when they were
broken. Four gaps were closed.

### 2.1 A leaky target can no longer become the production bundle

`StudentPerformanceFramework.save_best_model()` now refuses a `cpi` target:

```
ValueError: Refusing to save a 'cpi' model as the production bundle: this target
is computed from the same columns used as predictors, so its metrics measure how
well the model reproduces the labelling rule ... pass allow_leaky_target=True
(CLI: --allow-leaky-target) to export this explicitly as a rule-reconstruction
baseline.
```

Refusing is the point: `models/best_model.pkl` is the file Assessment Mode serves
to lecturers, and the original 97.8 % figure lived in exactly that file. A
deliberate baseline export still works, but it is stamped
`bundle_role="rule_reconstruction_baseline"`, `label_source_in_predictors=true`,
and carries the warning inside its own `caveats` — so it stays quotable-as-a-baseline
and unquotable-as-accuracy no matter who opens it.

The CLI enforces the same rule before doing any work:

```console
$ python scripts/train_model.py --data data/cohort.csv --definition cpi
REFUSED: --definition cpi builds the label from the same columns used as
predictors ... Its ~97% 'accuracy' is rule reconstruction, not prediction, and it
is banned from production metrics.
  production model:      --definition exam_band (default)
  baseline experiment:   --definition cpi --allow-leaky-target
$ echo $?
2
```

### 2.2 The bundle records *how* it was measured, not just what it is

`MODEL_BUNDLE_VERSION` 7.1 → **7.2**. New keys:

| Key | Meaning |
|---|---|
| `bundle_role` | `production` \| `rule_reconstruction_baseline` |
| `label_source_in_predictors` | boolean, read from `is_leaky_definition()` |
| `split` | `strategy`, `train_periods`, `test_periods`, `n_train`, `n_test` |
| `fairness` | the subgroup report, so it travels with the model |

`load_best_model()` restores all four, infers the role for bundles written before
7.2, and appends a `RULE-RECONSTRUCTION BUNDLE` caveat to every report produced
from such a model.

### 2.3 `preflight.py` fails a deployment that serves a rule reconstructor

```
✅ [model] bundle_role=production
✅ [model] metrics come from a temporal split: train 2024/25-S1, 2024/25-S2 -> test 2025/26-S1
```

On the old artefact the same check reads `❌ bundle_role=rule_reconstruction_baseline
... Serve an exam_band bundle instead` and `--production` exits 1.

### 2.4 The shipped bundle was replaced

The committed pickles were, in order: the v6.0 legacy CPI artefact (97.8 %), then
a v7.2 bundle that is actually certified. New contents:

| Field | Value |
|---|---|
| `model_version` | 7.2 |
| `bundle_role` | `production` |
| Target | `exam_band`, quantile thresholds 64.67 / 80.34 |
| Predictors | 6, no `exam_score` |
| Validation | temporal: 2024/25-S1 + 2024/25-S2 (1,510) → 2025/26-S1 (490) |
| Best model | Soft-Voting Ensemble |
| Hold-out | accuracy 0.653 · macro F1 **0.655** · balanced 0.660 · ROC-AUC 0.837 · ECE 0.027 |
| Baseline / lift | 0.363 / **+0.29** |
| Fairness | embedded (gender 0.065, programme 0.053, entry qualification 0.046) |
| Size | **9.9 MB** (compressed; 49.7 MB uncompressed) |

`joblib.dump(..., compress=3)` was added because six fitted candidates — two of
them calibrated wrappers that duplicate their base estimator — pushed the
uncompressed bundle to ~50 MB. Compressed it is smaller than the legacy file it
replaced, and load time is 1.6 s.

---

## 3. The noise control (the test that would have caught 97.8 %)

`tests/test_signal_integrity.py` runs **the same code path** twice on the same
cohort: once intact, once with the exam scores shuffled. Marginals, predictors,
label thresholds and the pipeline are identical; only the relationship between
predictors and outcome is destroyed. Logistic Regression, 500 rows, 100 held out.

| | Accuracy | Macro F1 | Balanced acc. | ROC-AUC | Baseline | Lift |
|---|---|---|---|---|---|---|
| **Structured** | 0.670 | **0.671** | 0.669 | 0.847 | 0.340 | **+0.330** |
| **Noise** | 0.300 | **0.289** | 0.298 | **0.507** | 0.340 | **−0.040** |

Three assertions follow from that table, and they are what the suite now enforces:

* on noise, macro F1 < 0.45 (chance for three classes) and ROC-AUC within
  0.40–0.60 — i.e. the pipeline reports a coin flip when it is looking at one;
* accuracy on noise lands within 8 points of the majority baseline, because a
  three-class problem with no signal can only ever predict the majority class;
* the signal-vs-noise gap exceeds 0.20 macro F1, so the two regimes cannot be
  confused.

A pipeline that reported 97.8 % on this control would be measuring its own
preprocessing. That is exactly what the CPI definition did: the label *was* the
weighted average of the inputs, so the "model" was asked to rediscover a formula
it had been handed.

---

## 4. Requirement-by-requirement detail

### 4.1 Target definitions and leakage

`backend/config.py`:

* `TARGET_DEFINITIONS = ("exam_band", "cpi")`, `TARGET_DEFINITION = "exam_band"`.
* `predictor_columns("exam_band")` → `attendance_rate`, `study_hours`,
  `sleep_hours`, `social_media_hours`, `assignment_score`, `test_score`.
  `exam_score` — the column the label is built from — is absent by construction,
  not by convention.
* `predictor_columns("cpi")` includes `exam_score`; `is_leaky_definition("cpi")`
  is `True` and drives the caveat, the bundle role and the save guard.

Independently verified end-to-end: two identical students differing only in
`exam_score` receive the same label and the same confidence to full precision.

### 4.2 Dynamic quantile thresholds

`LABEL_POLICY = {method: "quantile", low_quantile: 1/3, high_quantile: 2/3,
fixed_thresholds: (60, 80)}`. `CPIEngine._resolve_thresholds()` takes quantiles of
the observed score distribution, records them on the instance, writes them into
`target_info` and the bundle, and falls back explicitly when a degenerate
distribution collapses the two thresholds (`high <= low`). The fixed 60/80
cut-offs remain reachable only through `--label-method fixed` — retained because
the review's §"why the original was wrong" argument needs a reproducible
comparison, not because they are a live default.

Balanced distribution on the reference cohort: Low 665 / Average 668 / High 667.

### 4.3 Honest metrics and baselines

Every `evaluate()` result carries `accuracy`, `macro_f1`, `balanced_accuracy`,
`per_class`, `roc_auc`, `brier_score`, `calibration_error`, `baseline_majority`,
`lift_over_baseline` and `n_test`. The majority baseline is 0.363 on the reference
cohort, so the headline 0.653 is reported as **+0.29 over guessing the largest
class**, never on its own.

### 4.4 Temporal and subgroup validation

* `temporal_split()` refuses a missing period column, an unknown period, a
  training side under 50 rows, a test side under 20 rows, or a single-class side.
  No silent fallback to a random split exists — it would produce a temporal claim
  that never happened.
* The overlap assertion runs on **row indices**: `set(train_index) ∩ set(test_index) == ∅`
  and every row lands on exactly one side.
* `fairness.py` reports accuracy, macro F1 and per-class recall per subgroup,
  flags `n < 30` as unreliable and **excludes it from the gap**, refuses to draw a
  conclusion from a single reliable subgroup, and names any class the model never
  predicts for a group. It never re-weights — detecting a disparity is the job;
  acting on it is the institution's.

### 4.5 CLI outputs

```
$ python scripts/train_model.py --data data/cohort.csv \
      --split temporal --fairness --out models/best_model.pkl \
      --metrics reports/metrics.json
```

`reports/metrics.json` keys: `target`, `predictors`, `split`, `fairness`,
`cross_validation`, `models`, `best_model`, `caveats`, `validation_warnings`,
`fairness_markdown`.

### 4.6 Tests

`pytest tests -q` → **100 passed in 42 s**. The 14 added in this pass live in
`tests/test_signal_integrity.py`; the suite as a whole now fails if:

* the pipeline scores above chance on noise, or fails to beat the baseline on
  signal;
* the label source reaches the predictor list, or moves a prediction;
* a leaky model is exported as production, or a baseline export loses its stamp;
* a reloaded production bundle acquires a rule-reconstruction caveat;
* the default threshold policy stops being quantile;
* a degenerate score column silently produces one class.

---

## 5. Compatibility

* **Python 3.12+** enforced: `xgboost==3.4.1` and `shap==0.54.1` both declare
  `requires-python >= 3.12`, so `preflight.py` fails 3.11 with that reason rather
  than letting pip fail later. CI runs 3.12 and 3.13.
* **`scikit-learn==1.9.1`** pinned in `requirements.txt`; the bundle records
  `sklearn_version` and `_safe_load_bundle()` converts a mismatch into an
  actionable message.
* No dependency was added or changed in this pass.

---

## 6. Known gaps (stated, not hidden)

1. **The reference cohort is synthetic.** Every figure above describes
   `make_synthetic_dataset.py`. Compliance with the *method* is what the suite
   verifies; it says nothing about real students.
2. **The temporal result is benign by construction.** The generator's semester
   effect is an additive shift, which preserves signal structure, so transfer
   looks free. Real cohort effects are not additive — see
   `TEMPORAL_AND_FAIRNESS.md` §2.
3. **Fairness needs the columns.** No `gender`/`programme` in the export means no
   fairness table; the report says so rather than omitting the section quietly.
4. **Group fairness is not individual fairness.** Disaggregation detects
   group-level disparities; it cannot certify that any individual decision is
   just.
5. **The legacy 97.8 % figure still exists** in `CHANGELOG.md`, in the "why this
   was wrong" sections, and in one comparison row of the Results chapter. It is
   labelled as the rejected number everywhere. Removing it entirely would delete
   the evidence that the problem was understood.

---

## 7. Reproducing this review

```bash
cd Code6/student_performance_app

python -m pytest tests -q                       # 100 passed
python -m pytest tests/test_signal_integrity.py -q   # the noise control

python scripts/train_model.py --data data/cohort.csv \
    --split temporal --fairness \
    --out models/best_model.pkl --metrics reports/metrics.json

python scripts/report_results.py --data data/cohort.csv --temporal \
    --out reports/results.md --json reports/results.json

python scripts/preflight.py --production        # fails on a leaky bundle
python scripts/train_model.py --data data/cohort.csv --definition cpi   # exit 2
```
