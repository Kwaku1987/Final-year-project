# Changelog

All notable changes to this project. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow the
`PROJECT_VERSION` / `MODEL_BUNDLE_VERSION` constants in the code.

---

## [3.0] — 2026-11-04 — "Make the evaluation honest"

This release is mostly about **one methodological problem** and the six
engineering problems it exposed. Every item below is covered by a test in
`Code6/student_performance_app/tests/`.

### Fixed — methodology

* **Circular target (critical).** Models were trained to predict a CPI label
  computed from the very columns used as predictors, so 97.8 % "accuracy"
  measured rule reconstruction, not prediction. The pipeline now defaults to the
  `exam_band` definition: the examination score is the label and is **excluded
  from the predictors**. The legacy definition remains available
  (`SPAS_TARGET_DEFINITION=cpi`) and is labelled as rule reconstruction in the
  UI, the interpretation report and the model bundle.
* **Unlearnable class balance (high).** Fixed CPI cut-offs of 60/80 produced
  ~77 % Average and ~1 % High. Labels now use quantile thresholds
  (`LABEL_POLICY.method = "quantile"`), with the fixed cut-offs still selectable;
  the thresholds actually used are recorded and displayed. A degenerate
  distribution is detected and reported (`balanced: false`) rather than trained
  on silently.
* **Mis-scaled `assignment_score` (high).** The column arrives on a 0–20 scale
  while its CPI weight assumed 0–100, so its real contribution was ~1.5 points
  instead of 15. `DatasetValidator.normalize_score_scales()` now detects and
  rescales it, and warns loudly.
* **Misleading metrics (high).** Evaluation reported weighted accuracy alone.
  It now reports per-class precision/recall/F1/support, **macro F1**, **balanced
  accuracy**, one-vs-rest **ROC-AUC**, and the **majority-class baseline** with
  lift over it — everywhere, including the training page and exported reports.

### Added

* **SHAP instance-level explanations** (`backend/interpretation/shap_explainer.py`):
  "why did *this* student get *this* label", with the population baseline stored
  in the model bundle so explanations keep working after loading. Optional
  dependency — the app degrades gracefully to global importance.
* **Password-gated Training Mode** (`utils/auth.py`, `SPAS_ADMIN_PASSWORD_HASH`,
  hashed comparison, attempt limiting). Previously the Administrator/Lecturer
  choice was a button, so anyone could retrain or delete the live model.
* **CLI trainer** (`scripts/train_model.py`) and **synthetic cohort generator**
  (`scripts/make_synthetic_dataset.py`), so the reported
  numbers are reproducible outside the UI.
* **Test suite** — 44 pytest tests covering target building, validation, scale
  repair, preparation, training, honest metrics, prediction, SHAP graceful
  degradation, bundle round-trip, intervention reasons and password hashing.
* **Documentation**: `docs/METHODOLOGY.md`, `docs/MODEL_CARD.md`,
  `docs/DATA_PROTECTION.md`, rewritten `README.md`, this changelog.
* **Cross-validation** (stratified 5-fold on the training fold) now runs on the
  training path and is reported per model.
* **Soft-voting ensemble** as a third candidate model, using an internal
  scaler pipeline.
* **EDA page section**, wiring in `backend/eda/eda_analyzer.py` (previously 657
  lines of dead code).
* Model bundle is now **self-describing** (`model_version` 7.0): sklearn/Python
  version, target definition + thresholds, predictor order, CV results,
  validation warnings, caveats and the explanation background sample.

### Changed

* **Dependencies pinned** (`requirements.txt`). The committed bundle needs
  scikit-learn 1.9.x — with 1.6.x it fails with
  `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`.
* **Per-model scaling requirements** are recorded at train/load time instead of
  being re-derived with `isinstance(model, LogisticRegression)`, which was wrong
  for any `Pipeline`.
* **Sidebar reads the controller** as its single source of truth; it previously
  read `ApplicationState` while `app.py` held the controller, so the two could
  disagree ("Models: trained" beside "no models").
* **Target-builder and validator use `logging`, not `print()`** — 109 `print()`
  calls were invisible in a Streamlit server console; they are now log records,
  and validation warnings are surfaced in the UI.
* Prediction, batch and assessment pages ask **only for the predictors the model
  was trained on** (six, under the default definition — not seven).
* Loading a version-mismatched bundle now raises an actionable `RuntimeError`
  instead of an obscure traceback.

### Removed

* 87 committed `.pyc` files; `.gitignore` added.
* Blocking `print()` output from the CPI path (appeared in the server console on
  every training run).

### Known limitations at release

Legacy bundle retained (retrain before quoting its metrics); no temporal
(across-semester) validation; no fairness disaggregation; no licence yet.

---

## [2.0] — 2026-08-25 — Initial uploaded code

Single squashed commit ("Add files via upload"): Streamlit app, Facade/Factory
backend, CPI-based labelling, two models, PDF/Excel reporting, and a 16.9 MB
trained bundle reporting 97.8 % accuracy on a circular target.
