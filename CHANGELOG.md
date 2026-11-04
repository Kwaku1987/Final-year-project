# Changelog

All notable changes to this project. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow the
`PROJECT_VERSION` / `MODEL_BUNDLE_VERSION` constants in the code.

---

## [3.2] — 2026-11-04 — "Deployment readiness"

Closes the gap between "runs on my laptop" and "runs on a server": state that
survives a restart, a container image, a task runner, CI, and a runbook. No
modelling changes — the metrics and the bundle are untouched.

### Added

* **Persistence** (`backend/persistence/store.py`): SQLite with an
  `assessments` table and an `audit_events` table. `get_store()` returns a
  process-wide singleton; `SPAS_DB_PATH` (default `data/spas.db`) and
  `SPAS_DB_ENABLED` control it. Every method is failure-tolerant: an unwritable
  path degrades to session-only history with a warning instead of an exception,
  so a misconfigured volume cannot take the app down.
* **Audit trail**: `admin_unlocked`, `admin_unlock_failed`, `dataset_loaded`,
  `model_saved`, `model_loaded`, `model_deleted`, `assessment_created`,
  `report_exported` are recorded with actor, model name and model version. This
  is the Appointing Authority / Act 843 evidence trail, not just logging.
* **Assessment history is durable**: `get_assessment_history()` reads the store
  first and falls back to session records; the Assessment page says which one
  you are looking at.
* **Container image** (`Dockerfile`): `python:3.12-slim`, non-root uid 10001,
  `libgomp1` for the sklearn/XGBoost wheels, `/data` volume for the database,
  stdlib healthcheck on `/_stcore/health`, and dependency layer cached ahead of
  source so code changes do not re-resolve scikit-learn.
* **`docker-compose.yml`**: loopback-bound port, `spas-data` volume, model
  bundle mounted read-only (a bad save cannot overwrite the released model),
  2 GB memory limit, healthcheck.
* **`.env.example`** and `Makefile` targets: `install`, `run`, `test`, `lint`,
  `preflight`, `smoke`, `train`, `report`, `docker-build`, `docker-up`,
  `docker-down`, `docker-logs`, `docker-smoke`, `clean`.
* **`scripts/preflight.py`**: host readiness — Python version, importable
  packages, installed versions against the pins, bundle presence and contract,
  writable storage, SQLite integrity, password hash, CORS/XSRF posture, target
  definition. `--production` turns insecure defaults into failures.
* **`scripts/smoke_test.py`**: deployment readiness against a *running* server —
  health endpoint, HTML shell, static bundle, headers. `--json` for CI.
* **CI** (`.github/workflows/ci.yml`): tests on Python 3.11 and 3.12, a
  byte-compile pass, strict preflight, then a Docker build, health wait and
  container smoke test.
* **`docs/DEPLOYMENT.md`**: the runbook — local, Docker, compose, Nginx/Caddy
  TLS with the WebSocket headers Streamlit requires, volume layout, audit-trail
  queries, backup and restore, upgrade, retraining, rollback, Act 843 duties,
  and a troubleshooting table.
* **`tests/test_persistence.py`**: 9 tests — round-trip across store instances
  (restart simulation), ordering and limits, unwritable-path degradation,
  audit vocabulary. 62 tests total.

### Changed

* **`.streamlit/config.toml` is production-safe**: `enableCORS` and
  `enableXsrfProtection` are **on**, `showErrorDetails = false`,
  `maxUploadSize = 25`, `toolbarMode = "minimal"`, static serving off. Embedded
  previews override CORS/XSRF with CLI flags instead of weakening the committed
  file.
* **`.gitignore`**: the SQLite database is runtime state and is never committed;
  a root `.gitignore` keeps `.env` and local backups out of git.
* **README**: deployment section, runbook link, 62-test badge, persistence and
  audit documented, and the "known limitations" list corrected — temporal
  validation and fairness disaggregation do **not** exist and are now stated as
  open items rather than implied gaps.

### Not fixed (deliberately)

* **No temporal validation** (train one semester, test the next) and **no
  fairness disaggregation** by gender, programme or entry qualification. Both
  are cheap to add and both change which numbers you are allowed to report, so
  they are left as explicit open items rather than quietly mocked up.
* **Single shared Training Mode password.** Per-user identity and role
  separation belong at the proxy (institutional SSO), not in a Streamlit app.
* **The committed bundle is still the legacy `cpi`-target model.** Retrain
  before quoting its metrics.

---

## [3.1] — 2026-11-04 — "Search, calibration, and a results chapter"

Closes out the remaining items from the code review: the two model-quality
gaps (untuned models, untrustworthy confidence) and the missing academic
deliverables (results chapter, licence).

### Added

* **Hyperparameter search** (`ModelTrainer.tune_model`): `RandomizedSearchCV`
  over a documented grid, scoring macro F1, 3 folds, **training fold only**.
  Adds "Random Forest (Tuned)" as a candidate; the chosen configuration and
  search summary are recorded in the bundle and shown in the UI.
* **Probability calibration** (`ModelTrainer.train_calibrated`):
  `CalibratedClassifierCV` (sigmoid/Platt) for both Random Forest and Logistic
  Regression, each carrying its own preprocessing. Fixes the credibility problem
  where the two models disagreed wildly on borderline students (High at 100 %
  versus Average at 63 %) and no confidence figure was interpretable.
* **Calibration quality metrics**: multiclass **Brier score** and **expected
  calibration error (ECE)** in `ModelEvaluator`, so "is this confidence
  trustworthy?" is measured rather than assumed.
* **`scripts/report_results.py`**: generates the report tables (comparison,
  per-class, confusion matrix, CV, calibration quality, tuning summary, feature
  importance, caveats) as Markdown + JSON, so figures are never transcribed by
  hand.
* **`docs/REPORT_CHAPTER_RESULTS_AND_DISCUSSION.md`**: a complete draft Chapter 5
  with ten numbered tables, threats-to-validity analysis and the v2.0-vs-v3.0
  comparison. Every number is produced by the script above and labelled with its
  cohort.
* **MIT `LICENSE`** (source code only; student data explicitly excluded).
* Tests for tuning and calibration (53 tests total), including a check that the
  search cannot see the hold-out fold.

### Fixed

* **Interpreting calibrated wrappers.** `CalibratedClassifierCV` exposes neither
  `coef_` nor `feature_importances_`, so the report and the per-prediction
  explanation failed for calibrated models. The interpreter and SHAP explainer
  now unwrap `calibrated_classifiers_` to the base estimator.
* **SHAP with the new Explainer API.** Newer SHAP objects have `.values` rather
  than `.shap_values()`, which silently disabled instance explanations. Both APIs
  are now supported.
* **Scaling flags for wrapped models.** Calibrated wrappers are recorded as
  `requires_scaling=False` (they preprocess internally), consistent with
  Pipeline handling.

### Notes on honesty

On the reference synthetic cohort, tuning and calibration did **not** improve
the result (tuned forest 0.616 vs 0.638 untuned macro F1; calibrated Brier 0.455
vs 0.451). Both are reported with the evidence in the results chapter rather
than dropped, because a pipeline that can only report improvements is not
measuring anything.

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
