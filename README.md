# Student Performance Assessment System

A machine-learning decision-*support* system that predicts a student's
**end-of-term examination band** (Low / Average / High) from term-time behaviour
and continuous assessment — attendance, study hours, sleep, social-media use,
assignment and class-test scores — and explains every prediction.

Built as a final-year project: Streamlit front end, scikit-learn back end,
role-based modes for an administrator (train) and a lecturer (assess), model
persistence, per-student SHAP explanations, risk/intervention advice, and
PDF/Excel reporting.

![status](https://img.shields.io/badge/tests-173%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
![licence](https://img.shields.io/badge/licence-MIT-green)

---

## Why this version exists (read this before quoting any number)

An earlier version of this project computed a **Composite Performance Index**
from seven columns and then trained models to predict that index **from the same
seven columns**. The label was a deterministic function of the inputs, so the
reported 97.8 % "accuracy" measured how well the model reproduced its own
formula — not how well it predicts anything. The same pipeline scored ~98 % on
pure random noise.

This version separates **what the label is** from **what the model may see**:

* the examination score becomes the **label**, and is **excluded from the
  predictors**;
* labels are **balanced** by quantile thresholds (the old fixed cut-offs produced
  a ~1 % "High" class);
* every metric is reported with a **majority-class baseline**, **macro F1** and
  **balanced accuracy**;
* the old framing is still available (`SPAS_TARGET_DEFINITION=cpi`) but is
  labelled *rule reconstruction* everywhere it appears.

The result is a believable **0.655 accuracy / 0.653 macro F1 against a 0.335
baseline** on the documented synthetic cohort, and **chance-level accuracy on
noise** — which is what an honest evaluation should show. Full reasoning, limitations and the
reproduction steps: **[`docs/METHODOLOGY.md`](Code6/student_performance_app/docs/METHODOLOGY.md)**.

---

## Quick start

```bash
git clone https://github.com/Kwaku1987/Final-year-project.git
cd Final-year-project/Code6/student_performance_app

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # exact pins -- required, see note below

python scripts/preflight.py             # is this host ready?
streamlit run app.py
```

Then:

* **Lecturer / Assessment Mode** → *Load Saved Model* → assess students
  immediately with the bundled model (no dataset needed).
* **Administrator / Training Mode** → password (`change-me-admin` by default; set
  `SPAS_ADMIN_PASSWORD_HASH`, see [`docs/DATA_PROTECTION.md`](Code6/student_performance_app/docs/DATA_PROTECTION.md))
  → upload a dataset → train → save. The secret may be a SHA-256 digest **or**
  the plaintext password; `python -m utils.auth check` says which one the app
  found, and `python -m utils.auth verify "your-password"` tests it.

> ⚠️ **The version pins matter.** `models/best_model.pkl` was written by
> scikit-learn 1.9.x; with an older scikit-learn it fails to load with
> `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`.
> `requirements.txt` pins working versions.

No dataset handy? Generate a clearly-labelled synthetic one:

```bash
python scripts/make_synthetic_dataset.py --rows 2000 --out data/synthetic_cohort.csv
python scripts/train_model.py --data data/synthetic_cohort.csv \
       --out models/best_model.pkl --metrics reports/metrics.json
```

---

## Features

| Area | What it does |
|---|---|
| **Dataset** | CSV/XLSX upload, schema + range validation, column-alias mapping, automatic score-scale repair, missing/duplicate reporting, EDA (correlation, multicollinearity, outliers, target distribution) |
| **Training** | Target building, stratified 80/20 split, scaling, six candidate models (Logistic Regression, Random Forest, soft-voting ensemble, tuned forest, calibrated forest and calibrated logistic regression), 5-fold CV, `RandomizedSearchCV` hyperparameter search, sigmoid probability calibration, model comparison with honest metrics, saved bundle |
| **Prediction** | Single-student prediction with class probabilities (calibrated where selected), SHAP per-student explanation, intervention priority with stated reasons, PDF report |
| **Batch** | Whole-roster CSV/XLSX prediction with risk columns, CSV export |
| **Explainable AI** | Global feature importance (coefficients / Gini), plain-English insights, recommendations, methodology caveats |
| **Assessment Mode** | Lecturer workflow: load the saved model, assess by name/ID, history, PDF |
| **Reports** | PDF student report, Excel batch report |
| **Governance** | Password-gated Training Mode, single-source config, model card, version guard on load, MIT licence |
| **Persistence** | Assessments and an audit trail of every privileged action (`admin_unlocked`, `dataset_loaded`, `model_saved`, …) survive restarts, in SQLite |
| **Deployment status** | Sidebar panel states the loaded model's role and validation strategy, whether history is stored, and whether Training Mode is really password-protected |
| **Validation design** | Temporal split (train earlier semesters, test the latest) alongside the random split, with the change reported per model |
| **Fairness** | Hold-out performance disaggregated by gender, programme and entry qualification; small subgroups flagged, never averaged away |

### Two questions, two explanations

| Question | Method | Where |
|---|---|---|
| Which factors matter overall? | Normalised coefficients / Gini importance | 🧠 Explainable AI |
| Why did *this* student get *this* label? | **SHAP** contributions, measured against a stored sample of training rows | 🔮 Prediction, 📋 Assess Student |

---

## Project layout

```
Code6/student_performance_app/
├── app.py                        Streamlit entry point + router
├── requirements.txt              exact pins
├── models/best_model.pkl         certified exam_band bundle (v7.2, temporal split)
├── framework/
│   ├── student_performance_framework.py   Facade: the public API
│   └── framework_factory.py               Factory: wires everything
├── dashboard/
│   ├── dashboard_controller.py            UI ↔ backend orchestration
│   ├── application_state.py               session state, notifications
│   └── pages/                            9 pages (home, dataset, training, …)
├── backend/
│   ├── config.py                 single source of truth for schema/weights/policy
│   ├── validation/               dataset validator + scale repair
│   ├── cpi/                      CPI + target builder (exam_band | cpi)
│   ├── preprocessing/            split (random | temporal), scale, encode
│   ├── training/                 LR, RF, ensemble, CV
│   ├── evaluation/               honest metrics, baselines, fairness
│   ├── interpretation/           global importance + SHAP
│   ├── intervention/             risk score → priority → reasons
│   ├── reporting/                PDF / Excel exporters
│   ├── persistence/              SQLite store: assessments + audit trail
│   └── eda/                      exploratory analysis (wired into the UI)
├── components/sidebar.py
├── utils/                        auth (password gate), theme
├── scripts/                      synthetic data generator, CLI trainer,
│                                  results-table generator, preflight,
│                                  smoke test
├── tests/                        173 pytest tests
└── docs/                         METHODOLOGY · MODEL_CARD · DATA_PROTECTION
```

Architecture: **Facade** (`StudentPerformanceFramework`) over eight backend
services, built by a **Factory**, driven by a **controller** and a session state
object, with **one shared config** so no component invents its own schema.

---

## Tests

```bash
python -m pytest tests -q          # 173 passed, ~45 s
```

The suite encodes the fixes: the label source is never a predictor, labels are
balanced, the 0–20 assignment scale is repaired, metrics carry a baseline and
calibration quality (Brier, ECE), search and calibration never touch the test
fold,
interventions always state a reason, bundles round-trip their contract, a
version-mismatched bundle fails with a readable message instead of a traceback,
an assessment survives a store restart, the audit trail
records every privileged action. Since 3.3 the suite also pins the two
claims that are easiest to fake: a temporal split may not let a semester appear
on both sides, and a fairness table may not average away a subgroup too small to
support a conclusion. Since 3.3.4 it also pins the deployed-app fixes: nothing
assigns to the read-only `application_mode` property, no sidebar helper calls
itself, a pasted credential in any of its common shapes unlocks Training Mode,
the attempt limit can be cleared, the committed bundle still describes itself as
a production model validated temporally, and both Streamlit configs ask for the
dark theme.

---

## Documentation

* [`docs/METHODOLOGY.md`](Code6/student_performance_app/docs/METHODOLOGY.md) — target design, evaluation protocol, limitations
* [`docs/TEMPORAL_AND_FAIRNESS.md`](Code6/student_performance_app/docs/TEMPORAL_AND_FAIRNESS.md) — validation across semesters, subgroup performance, and how to read both
* [`docs/MODEL_CARD.md`](Code6/student_performance_app/docs/MODEL_CARD.md) — what the shipped model is, and what it must not be used for
* [`docs/DATA_PROTECTION.md`](Code6/student_performance_app/docs/DATA_PROTECTION.md) — access control and Act 843 obligations
* [`docs/REPORT_CHAPTER_RESULTS_AND_DISCUSSION.md`](Code6/student_performance_app/docs/REPORT_CHAPTER_RESULTS_AND_DISCUSSION.md) — draft Results & Discussion chapter, tables generated by script
* [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — runbook: Docker, TLS proxy, backup/restore, upgrade and rollback
* [`docs/V3_3_COMPLIANCE_REVIEW.md`](docs/V3_3_COMPLIANCE_REVIEW.md) — requirement-by-requirement audit of the v3.3 standards, with the noise control results
* [`CHANGELOG.md`](CHANGELOG.md) — what changed in this revision and why

---

## Deployment

```bash
make preflight                        # check the host: python, pins, bundle, storage, security
make docker-build && make docker-up   # single-host stack (needs .env)
make smoke                            # verify the running deployment serves
```

The stack is a `python:3.12-slim` image running as uid 10001, a healthcheck on
`/_stcore/health`, SQLite persistence on a named volume, and the model bundle
mounted read-only so a bad save cannot overwrite the released model. It binds to
`127.0.0.1` — a TLS reverse proxy belongs in front of it.

Full runbook (local, Docker, compose, Render free tier, Nginx/Caddy TLS, backup
and restore of the database, retraining, rollback, Act 843 duties,
troubleshooting): **[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)**.

The image honours `$PORT`, so it runs unmodified on Render, Cloud Run, Railway or
Fly as well as on compose. For **Streamlit Community Cloud** the repository root
now carries the `requirements.txt` and `.streamlit/config.toml` that Cloud reads
(it does not look inside subdirectories), and `utils/runtime_env.py` maps
`st.secrets` into the environment so the password hash and DB settings actually
take effect. A free-tier Render blueprint is included
([`render.yaml`](render.yaml)) — good for a public Assessment-Mode demo; training
needs ≥ 1 GB of RAM and belongs on the compose stack or a small VPS.

> ⚠️ Deployment-ready is not the same as fit to serve students. Five items
> remain, and they are decisions rather than code: real-cohort retraining, the
> temporal and fairness runs on that cohort, institutional sign-off, per-user
> authentication (there is one shared Training Mode password), and Act 843
> operational duties. See §9 of the deployment guide.

---

## Known limitations

Relative (cohort-specific) thresholds · **temporal validation exists but only
reports one horizon** (and on the synthetic cohort the change is near zero
because the generator's semester effect is a benign additive shift — see
[`docs/TEMPORAL_AND_FAIRNESS.md`](Code6/student_performance_app/docs/TEMPORAL_AND_FAIRNESS.md)) ·
**fairness disaggregation exists but only as good as the columns you supply**
(no `gender`/`programme` column, no fairness table — the report says so rather
than omitting the section) · the **Average** band is the weakest class in every
subgroup (recall 0.48–0.63) · self-reported behavioural inputs · no causal
claims · the committed bundle is a compliant `exam_band` model with no
label-source leakage, but it is trained on **synthetic** data — retrain before
quoting any number from it.

## Licence

**MIT** — see [`LICENSE`](LICENSE). The licence covers the source code only; it
grants no rights over student data and does not remove the obligations of
Ghana's Data Protection Act, 2012 (Act 843).

**Author:** Brenya Kwaku Twum · GitHub [@Kwaku1987](https://github.com/Kwaku1987)
