# Changelog

All notable changes to this project. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions follow the
`PROJECT_VERSION` / `MODEL_BUNDLE_VERSION` constants in the code.

---

## [3.3.6] — 2026-11-05 — "No password: Training Mode is one click away"

The administrator password is gone, at the owner's request, and so is everything
that existed to serve it. **This is a deliberate reduction in protection, not a
fix**, and the documentation says so rather than quietly dropping the subject.

Removed:

* `utils/auth.py` — the SHA-256 verification, the attempt counter, the lockout,
  the credential diagnostics and the `hash` / `check` / `verify` CLI. Deleted, not
  left disabled, so there is no dead path that looks like a gate.
* the password field, "Incorrect password" branches, "Reset attempts" button and
  the diagnosis panel on the mode-selection page: **"Enter Training Mode" is now
  a single click.**
* `SPAS_ADMIN_PASSWORD_HASH` from `backend/config.py`, `utils/runtime_env.py`,
  `.env.example`, `secrets.toml.example`, `render.yaml`, the `Dockerfile`
  example, CI and the compose docs; `ADMIN_PASSWORD_DEFAULT` and
  `AUTH_MAX_ATTEMPTS` with it.
* `app.py` no longer re-locks a "stale" administrator session, because there is no
  session to re-lock.

Kept, because removing it would have made the deployment dishonest:

* the mode screen and the sidebar say, in one line each, that Training Mode is
  open and that anyone who can reach the URL can retrain or delete the model;
* entering Training Mode is still audited, now as `training_mode_entered`
  (`log_admin_unlock` → `log_training_mode_entry`);
* `preflight.py` reports the missing gate as a warning instead of implying
  protection that no longer exists;
* CORS/XSRF stay on, tracebacks stay hidden, uploads stay capped.

Tests went from 194 to 144 (the credential suite was deleted with the code it
tested) and now include the removal itself: no `utils.auth` import anywhere, no
password widget on the mode screen, one click into Training Mode, and no
`SPAS_ADMIN_PASSWORD_HASH` / `change-me-admin` left in the source.

---

## [3.3.5] — 2026-11-05 — "The two things the live app still got wrong"

Both were reported from the deployed app, and both were invisible from inside
the code: the password was rejected, and the committed model was "not found".

* **The password.** The deployed secret was reported as *"plaintext, 66
  characters"* — a 64-character SHA-256 digest wrapped in two characters of
  packaging (a backtick, a brace, a curly quote, a non-breaking space: whatever
  it was copied through). `normalize_credential()` now strips the packaging and,
  failing that, **recovers a 64-hex digest from anywhere inside the value**; a hex
  string of the wrong length is *not* truncated to a digest, it is reported as
  such. `credential_diagnostics()` (behind `python -m utils.auth check`) prints
  the raw versus usable length, everything it stripped, and a fingerprint
  (first six / last four characters) so you can recognise which secret is
  loaded without printing it.
* **Two failures that look like a wrong password** are now named: typing the
  stored **digest** into the password box, and storing
  `hash(hash(password))` — i.e. running `utils.auth hash` on a digest, which no
  password can satisfy. Neither is accepted; both are diagnosed, because the fix
  is in the secret, not the typing. A successful unlock clears the hint.
* **The model.** The hosted app said "No saved model found yet" while
  `models/best_model.pkl` was committed, because **Community Cloud starts the
  process in the repository root** (`streamlit run Code6/student_performance_app/
  app.py`) whereas a local run starts inside the app directory: the relative
  default `models/best_model.pkl` resolved one level above the bundle.
  `backend/config.py` now derives `APP_DIR` from `__file__`, and the model and
  SQLite paths are absolute (`DEFAULT_MODEL_PATH`, `DEFAULT_DB_PATH`); the
  framework's four `model_path` defaults and `DashboardController` use them. The
  "not found" message prints the resolved path, so the next mismatch is
  diagnosable from the UI rather than by reading the source.
* 21 new tests: the deployed 66-character secret in seven packaging shapes,
  digest-recovery boundaries (66 hex chars must not become a digest), the two
  unlock hints, the same packaged secret unlocking Training Mode through the
  real page, and `tests/test_paths_and_deployment.py` — which boots the app with
  the working directory set to the **repository root** and asserts Assessment
  Mode finds the bundle. **194 total**, ~46 s.

---

## [3.3.4] — 2026-11-05 — "Fix the deployed app: dark theme, real password diagnosis, no crash"

Three faults reported from the live app, plus the feature work that was worth
doing while those files were open. Every fix is pinned by a test.

* **"Switch Mode" crashed with `AttributeError`.** `application_mode` is a
  read-only property on `DashboardController`; the sidebar and `app.py` both did
  `controller.application_mode = None`. There is now a `clear_mode()` method and
  no assignment anywhere — an AST test over every module fails if one comes back.
* **Seven sidebar helpers were infinitely recursive.** Each `classmethod` ended
  with `return cls._<its own name>()`, so the first call blew the stack.
  Fallbacks return `False` / `[]`. A test walks the AST of `components/sidebar.py`
  and refuses any self-call.
* **The admin password was always reported wrong.** The comparison now
  normalises what was actually pasted: the whole `NAME = "value"` line, wrapping
  quotes, trailing newlines, a `sha256:` prefix, and uppercase hex. A **plaintext**
  password in `SPAS_ADMIN_PASSWORD_HASH` is accepted as a credential instead of
  being treated as an error (both branches compare digests, so the check stays
  constant-time). New `python -m utils.auth check` reports `hash` / `plaintext` /
  `default` without printing the secret, and `verify <password>` answers a single
  question with an exit code. The per-subcommand argv guards were wrong (`check`
  takes no argument, so it was unreachable from the shell) — caught by the new
  tests, not by reading.
* **The lockout was permanent.** The failed-attempt counter lives in
  `st.session_state`, which survives the page reload the old message recommended.
  There is a **"Reset attempts and try again"** button, and the page says plainly
  that the counter is a typo guard rather than a security boundary.
* **The mode-selection page diagnoses instead of rejecting.** It states which
  credential is configured (hash / plaintext / none), names the variable, and on
  failure expands a *"Still failing? Diagnose it"* panel with the checker commands
  and the four real causes.
* **New: sidebar "Deployment status" panel.** Model role
  (`production` / `rule_reconstruction_baseline` / `not_loaded`), validation
  strategy read from **the bundle's own split record**, whether history is
  persisted, and the password mode — the four facts that decide whether a number
  on screen can be trusted. Before a bundle is loaded it says `not_loaded`
  instead of quoting a default: `bundle_split` now overrides this session's
  `split_metadata`, which would otherwise label a temporal bundle "random".
* **New: CSV export of assessment history**, with the raw confidence and the
  input features that produced each prediction — the table shows one decimal and
  a truncated timestamp.
* **Dark theme** in both `.streamlit/config.toml` files (root and app directory),
  with pinned colours; a test asserts both request `base = "dark"` and that they
  cannot drift.
* **One version number**: the sidebar no longer advertises "Version 6.0" /
  "Version 5.0"; it prints `PROJECT_VERSION`. Its feature caption no longer
  claims "CPI-based target generation" — the shipped model predicts an exam band,
  and CPI is a demoted rule-reconstruction baseline.
* **The devcontainer is now usable.** A collaborator's Codespaces config
  (`059fb4f`) used a Python **3.11** image and launched Streamlit from the
  repository root; the pins do not install on 3.11, so the container could never
  have run the app. Now 3.12, launched from the app directory, with the CORS/XSRF
  relaxations on the command line rather than in the committed config — and a
  test that keeps it that way.
* 62 new tests — `test_auth_credentials.py` (34), `test_ui_regressions.py` (11),
  `test_history_export.py` (11), `test_mode_selection_ui.py` (6); **173 total**
  (111 before), ~45 s. The UI ones drive the real `app.py` through
  `streamlit.testing`, so the crash paths are exercised, not just described.

---

## [3.3.3] — 2026-11-05 — "Streamlit Community Cloud"

Makes the hosted Streamlit path work. Three things were missing, and each
fails silently rather than loudly, which is why they are worth listing.

* **`utils/runtime_env.py`** — a `st.secrets` → `os.environ` bridge, called
  before anything reads config. Community Cloud configures apps through secrets,
  not environment variables, so without this the deployed app started with no
  password hash and fell back to the documented development password: healthy
  looking, Training Mode wide open. Rules in the bridge: a real environment
  variable always wins (so Docker/Render are unaffected), only known keys are
  copied, TOML booleans become `"1"`/`"0"` (the codebase compares strings, so
  `SPAS_TUNING = true` was silently disabling search), empty values are skipped,
  and missing secrets are a no-op. Verified in a simulated Cloud run: hash from
  secrets → `is_using_default_password()` False, `SPAS_TUNING = true` →
  `TUNING["enabled"] is True`, `SPAS_DB_ENABLED = "0"` → session-only history.
* **Root `requirements.txt`** — `-r Code6/student_performance_app/requirements.txt`.
  Cloud installs from the repository root and does not search subdirectories; the
  app would have deployed unpinned and failed on `import xgboost`.
* **Root `.streamlit/config.toml`** — Cloud launches from the repo root, so the
  app-directory config was never read and the deployed app fell back to
  Streamlit's defaults (tracebacks visible to users, no upload cap).
* **`.streamlit/secrets.toml.example`** — the exact TOML to paste into the
  Secrets box, with the reason for each value. A test asserts the template lists
  every key the bridge accepts, and one phantom key (`SPAS_MAX_UPLOAD_MB`) was
  removed because nothing reads it — the upload cap lives in `config.toml`.
* **`docs/DEPLOYMENT.md` §4.4** — the wizard values (main file path, Python
  3.12, secrets), the memory envelope (690 MB floor / ~2.7 GB ceiling against a
  ~400 MB idle footprint), and an explicit answer to "it is asking for a
  database": there isn't one — SQLite is a local file and Cloud has no persistent
  disk, so `SPAS_DB_ENABLED=0` is the honest setting.
* 11 tests (`tests/test_runtime_env.py`); **111 total**.

---

## [3.3.2] — 2026-11-04 — "Portable port, Render blueprint"

* **The image honours `$PORT`.** Render, Cloud Run, Railway and Fly inject a port
  and route traffic only to it; the `CMD` hardcoded 8501, so the container would
  have built and then been marked unhealthy. The `CMD` is now
  `sh -c "exec streamlit run app.py --server.port=${PORT:-8501} ..."` — `exec`
  keeps SIGTERM reaching Streamlit, and with no `PORT` set the default is
  unchanged, so compose and plain `docker run` behave exactly as before. The
  `HEALTHCHECK` honours `$PORT` for the same reason. Verified by running the
  exact command with `PORT=8599`: it listened there and `/_stcore/health`
  returned `ok`.
* **`render.yaml`**: one-click free-tier blueprint (Docker runtime, health path
  `/_stcore/health`, `SPAS_DB_ENABLED=0` because free Render has no persistent
  disk, password hash marked `sync: false` so it is never read from git).
* **`docs/DEPLOYMENT.md` §4.3** documents the measured memory profile against
  Render's 512 MB limit: 402 MB idle serving the bundle, ~750 MB during a
  training run. Assessment Mode fits; Training Mode does not, and the guide says
  so instead of letting an OOM kill explain it.

---

## [3.3.1] — 2026-11-04 — "Enforcement pass"

The 3.1–3.3 standards were documented and mostly implemented; several of them
were not *enforced*, meaning a rule could be broken without anything failing.
This pass closes that gap and replaces the shipped bundle with a compliant one.
Full audit: [`docs/V3_3_COMPLIANCE_REVIEW.md`](docs/V3_3_COMPLIANCE_REVIEW.md).

### Added

* **Signal-integrity controls** (`tests/test_signal_integrity.py`, 14 tests). The
  same pipeline is run on an intact cohort and on the same cohort with the exam
  scores shuffled: 0.671 macro F1 / 0.847 ROC-AUC versus **0.289 / 0.507**. The
  suite now fails if the pipeline scores above chance on noise, if it fails to
  beat the majority baseline on signal, or if the two regimes come within 0.20
  macro F1 of each other. This is the direct antidote to the legacy 97.8 %:
  a pipeline that reports accuracy on a shuffled target is measuring its own
  preprocessing.
* **A leaky target can no longer be exported as the production bundle.**
  `save_best_model()` raises unless `allow_leaky_target=True`; the CLI exits 2
  with the reason before doing any work. A deliberate baseline export is stamped
  `bundle_role="rule_reconstruction_baseline"` + `label_source_in_predictors=true`
  and carries the warning inside its own caveats.
* **Bundle provenance** (`MODEL_BUNDLE_VERSION` 7.1 → 7.2): `bundle_role`,
  `label_source_in_predictors`, `split` (strategy + periods + row counts) and
  `fairness` now travel with the model. `load_best_model()` restores them,
  infers the role for older bundles, and attaches a RULE-RECONSTRUCTION caveat
  where it applies.
* **`preflight.py` certifies the bundle**: `--production` now fails on a
  rule-reconstruction bundle, warns when the split provenance is missing or
  random-only, and reports the temporal periods the metrics came from.
* **Label-source invariance test**: two students identical except `exam_score`
  must receive the same label and confidence to full precision.
* **CLI**: `train_model.py --allow-leaky-target`, `--min-support`.

### Changed

* **The committed bundle is now a certified production model**: `exam_band`,
  `bundle_role="production"`, temporal split (1,510 train → 490 test), Soft-Voting
  Ensemble, macro F1 0.655, lift +0.29 over a 0.363 baseline, fairness report
  embedded. The v6.0 legacy CPI artefact (the source of the 97.8 % figure) is
  gone from the working tree and remains only in git history.
* **`joblib.dump(..., compress=3)`**: six fitted candidates — two of them
  calibrated wrappers that duplicate their base estimator — made the uncompressed
  bundle ~50 MB. Compressed it is **9.9 MB**, smaller than the legacy file it
  replaced, with a 1.6 s load.
* `MODEL_CARD.md` §1 rewritten around the new artefact, including why the 97.8 %
  number is unreachable now rather than merely discouraged.
* **100 tests** (was 86).

### Unchanged (deliberately)

* The fixed 60/80 threshold policy still exists behind `--label-method fixed`:
  the argument that it was wrong needs a reproducible comparison.
* The 97.8 % figure still appears in the "why this was wrong" sections and in one
  comparison row of the Results chapter, labelled as the rejected number.
  Deleting the evidence is not the same as fixing the method.

---

## [3.3] — 2026-11-04 — "Temporal validation and fairness disaggregation"

The two items that 3.1 and 3.2 listed as limitations are now implemented,
tested and reported. Neither changes the model; both change what you are
allowed to claim about it.

### Added

* **Temporal validation** (`backend/preprocessing/split_strategy.py`):
  `temporal_split()` holds out whole academic periods — train 2024/25, test
  2025/26 — instead of a random 80/20. Period column auto-detected
  (`semester`, `session`, `term`, `academic_year`, `academic_period`, `cohort`,
  `period`) or named explicitly. Refusals are explicit: missing column, unknown
  period, fewer than 50 training rows, fewer than 20 test rows, or a single class
  on either side all raise a message that says what to do. `compare_splits()`
  lines up the same model under both designs and reports the change, worst first.
  A silent fallback to a random split was deliberately not implemented: it would
  produce a temporal claim that never happened.
* **Fairness disaggregation** (`backend/evaluation/fairness.py`): hold-out
  accuracy, macro F1 and per-class recall by `gender` / `programme` /
  `entry_qualification` (and aliases), plus predicted-distribution and
  disparity summaries. Subgroups below `min_support` (30) are flagged
  *unreliable* and excluded from the gap; a single reliable subgroup yields "no
  disparity conclusion is possible" rather than a tidy 0.000; a class the model
  never predicts for a group is named. Nothing is re-weighted or "corrected" —
  detecting a disparity is this module's job, deciding what to do about it is the
  institution's.
* **Pipeline and facade wiring**: `prepare()` / `prepare_data()` /
  `run_pipeline()` accept `split_strategy`, `period_column`, `test_periods`;
  the prepared dict now carries `split_metadata`, `train_index` and `test_index`
  so group columns stay aligned with the hold-out rows.
  `framework.evaluate_subgroups()`, `framework.temporal_comparison()` and
  `framework.split_summary` expose both analyses; the split is recorded in the
  caveats that travel with every metric.
* **Synthetic generator context columns** (`semester`, `gender`, `programme`,
  `entry_qualification`) — **never predictors**, added so both analyses are
  demonstrable on the reference cohort. Two effects are injected on purpose
  (a semester offset on the exam score, a programme attendance offset) and
  documented as properties of the generator, not claims about students.
  `--no-context` reproduces the old column set.
* **CLI**: `train_model.py --split temporal --period-column --test-period
  --fairness`, and `report_results.py --temporal --no-fairness --min-support`.
  Fairness tables are on by default: a results chapter should not be able to
  omit them by forgetting a flag.
* **`docs/TEMPORAL_AND_FAIRNESS.md`**: method, refusals, the current synthetic
  results, and an explicit reading of what those results do **not** prove.

### Fixed

* **The Python floor is 3.12, not 3.11.** `xgboost==3.4.1` and
  `shap==0.54.1` both declare `requires-python >= 3.12`, so the 3.11 CI job
  failed at *Install dependencies* before running a single test. Caught by the
  first CI run, which is what it is for. CI now runs 3.12 and 3.13,
  `preflight.py` refuses 3.11 with a message that names the reason, and the
  README badge and deployment prerequisites say 3.12.

### Results on the synthetic cohort (2,000 rows, seed 42)

* **Temporal validation does not degrade performance here** — change between
  −0.03 and +0.01 macro F1 depending on the model. That is a property of the
  generator: its semester effect is an additive shift, which preserves the signal
  structure. Stated that way in the docs rather than reported as time-robustness.
* **Fairness gaps are small by construction** (gender 0.065, programme 0.053,
  entry qualification 0.046 macro F1), which makes the synthetic cohort a useful
  "no bias to find" baseline for real data.
* **One finding is not small:** recall for the **Average** band is 0.48–0.63 in
  every subgroup — borderline students are where the model is weakest, and they
  are precisely the students an early-warning flag acts on. Same finding as the
  per-class table, now shown to hold across groups.

### Changed

* 24 new tests (`tests/test_temporal_and_fairness.py`); **86 tests total**.
  `test_no_period_appears_on_both_sides_of_a_temporal_split` checks row indices,
  not labels, so a mask bug cannot pass it; `test_disparity_is_measured_not_averaged_away`
  requires a 100%-vs-0% split to surface as a 1.0 gap rather than a 67% average.
* `MODEL_CARD`/`METHODOLOGY` limitations 2 and 6 rewritten from "not done" to
  "implemented, here is what the synthetic result does and does not show".

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
