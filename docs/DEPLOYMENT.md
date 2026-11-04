# Deployment Guide

**Student Performance Assessment System** — how to take this repository from a
working checkout to something a department can actually use.

Version 3.2 · Applies to `main` @ `de1b1a6` and later

---

## 0. Read this first: what this system is and is not

| It is | It is not |
|---|---|
| A decision-*support* dashboard with a transparent ML model | An autonomous grading system |
| Trained on a **synthetic** cohort (`data/cohort.csv`) | Validated on your real students — it is not |
| Honest about its limits (bundle level `cpi`, `exam_band`, baseline, caveats) | A source of accuracy numbers you can quote to an examiner |

Nothing in this guide makes the model predictive of real students. Deploying it
to real users without retraining on a real, de-identified cohort is a
**methodological** decision, not a technical one, and it is yours to make — see
§9 "Before you serve real students".

The app has two modes with different threat models:

- **Assessment Mode** — no password. It collects a form, runs the model, shows
  explanations. It never reads the training data and never writes to it.
- **Training Mode** — password protected. Loads a CSV, trains, compares models,
  saves/deletes bundles. This is the privileged surface.

---

## 1. What is in the repository

```
.
├── Dockerfile                  # production image (non-root, healthcheck)
├── docker-compose.yml          # single-host stack, loopback-bound
├── .env.example                # every supported environment variable
├── Makefile                    # install / run / test / preflight / train / docker-*
├── .github/workflows/ci.yml    # tests, strict preflight, image build + smoke test
└── Code6/student_performance_app/
    ├── app.py                  # Streamlit entry point
    ├── requirements.txt        # exact pins — the contract for reproducibility
    ├── .streamlit/config.toml  # production-safe server defaults
    ├── models/best_model.pkl   # the released model bundle
    ├── data/                   # training CSV (never shipped in the image)
    ├── scripts/
    │   ├── preflight.py        # "is this host ready?" — run before every deploy
    │   ├── smoke_test.py       # "is the running deployment serving?" — run after
    │   ├── train_model.py      # retrain and export a bundle
    │   ├── report_results.py   # regenerate the results tables
    │   └── make_synthetic_dataset.py
    ├── backend/persistence/    # SQLite store: assessments + audit trail
    ├── tests/                  # 62 tests
    └── docs/                   # methodology, model card, data protection
```

---

## 2. Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.11 | 3.12 is what the image ships |
| RAM | 2 GB | scikit-learn + pandas + shap |
| Disk | 1 GB | 16 MB model, DB grows slowly |
| Docker | 24.x + Compose v2 | only for the container path |
| A model bundle | `models/best_model.pkl` | Assessment Mode refuses to run without one |

Dependencies are pinned in `requirements.txt`. Do not loosen the pins in
production: a scikit-learn minor bump can make a saved bundle unpicklable, and
the failure appears at request time, not at install time. `preflight.py`
compares installed versions against the pins and warns before that happens.

---

## 3. Local run (no Docker)

```bash
cd Code6/student_performance_app

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional but recommended: set a real Training Mode password
python -m utils.auth hash "your-password"      # prints a SHA-256 digest
export SPAS_ADMIN_PASSWORD_HASH=<digest pasted here>

python scripts/preflight.py                    # expect "READY WITH WARNINGS"
streamlit run app.py                           # http://localhost:8501
```

From the repository root, `make install && make preflight && make run` does the
same thing.

**If there is no bundle yet**, Assessment Mode will refuse to start. Train one:

```bash
python scripts/make_synthetic_dataset.py --rows 2000 --out data/cohort.csv
python scripts/train_model.py --data data/cohort.csv \
    --out models/best_model.pkl --metrics reports/metrics.json
```

On a 2-core laptop this takes about 2–4 minutes with tuning and calibration on.

---

## 4. Docker

### 4.1 Build

```bash
docker build -t spas:3.2 .
```

The image is `python:3.12-slim`, installs `libgomp1` (needed by the
scikit-learn/XGBoost wheels), copies `requirements.txt` first so the dependency
layer caches, then copies the app, creates uid/gid `10001`, and drops to it.

### 4.2 Run

```bash
docker run -d --name spas \
  -p 127.0.0.1:8501:8501 \
  -e SPAS_ADMIN_PASSWORD_HASH="$SPAS_ADMIN_PASSWORD_HASH" \
  -e SPAS_DB_PATH=/data/spas.db \
  -v spas-data:/data \
  -v "$PWD/Code6/student_performance_app/models:/app/models:ro" \
  --restart unless-stopped \
  spas:3.2
```

Three deliberate choices in that command:

- `-p 127.0.0.1:...` — the container is **not** exposed to the internet. §6 adds
  TLS in front of it.
- `-v spas-data:/data` — assessments and the audit trail survive redeploys.
  Without it, history is in an anonymous layer and dies with the container.
- `models:ro` — the app can read a bundle but not overwrite it at runtime, so a
  bad save cannot corrupt the released model. Save new bundles through a
  controlled retrain (§8) instead.

### 4.3 Compose (the normal path)

```bash
cp .env.example .env
# edit .env: set SPAS_ADMIN_PASSWORD_HASH
docker compose up -d --build
docker compose logs -f
docker compose ps          # STATUS should read "healthy" after ~40s
```

Stop without losing data: `docker compose down`. Never use `down -v` unless you
intend to destroy the assessment history.

---

## 5. Verifying a deployment

Two different questions, two different scripts.

```bash
# Is this host ready? (before starting)
cd Code6/student_performance_app && python scripts/preflight.py --production

# Is the running deployment serving? (after starting)
python scripts/smoke_test.py --url http://127.0.0.1:8501
python scripts/smoke_test.py --url https://spas.your-domain.edu.gh --json
```

`preflight.py --production` fails on insecure defaults (no password hash, CORS
or XSRF disabled, tracebacks shown). `smoke_test.py` checks the health endpoint,
the HTML shell, the static bundle, and response headers — the four things that
are usually broken when "the app doesn't load".

In Docker, `docker compose ps` reading `healthy` means the image's built-in
healthcheck (`/_stcore/health` returning `ok`) is passing. A container that
starts but stays `starting` for more than a minute is almost always a missing
model bundle or an unwritable `/data`; check `docker compose logs app`.

---

## 6. HTTPS: reverse proxy

Streamlit's server is not hardened for direct internet exposure. Terminate TLS
in front of it and let it see only loopback traffic.

### 6.1 Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name spas.your-domain.edu.gh;

    ssl_certificate     /etc/letsencrypt/live/spas.your-domain.edu.gh/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/spas.your-domain.edu.gh/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    client_max_body_size 25m;          # matches maxUploadSize in config.toml

    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;

        # Streamlit is a WebSocket app; these three lines are mandatory
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        proxy_read_timeout 3600s;      # training runs outlast default timeouts
        proxy_buffering    off;
    }
}

server {
    listen 80;
    server_name spas.your-domain.edu.gh;
    return 301 https://$host$request_uri;
}
```

Certificate: `certbot --nginx -d spas.your-domain.edu.gh`, then confirm
auto-renewal with `certbot renew --dry-run`.

### 6.2 Caddy (shorter, automatic TLS)

```caddy
spas.your-domain.edu.gh {
    reverse_proxy 127.0.0.1:8501
    request_body { max_size 25MB }
}
```

### 6.3 Keep CORS/XSRF on

`.streamlit/config.toml` ships with `enableCORS = true` and
`enableXsrfProtection = true`. Because the browser now talks to one origin (the
proxy), these work correctly and should stay on.

**The one exception:** an embedded iframe preview (e.g. a sandboxed developer
preview) will be blocked by these settings. Override per-launch instead of
weakening the committed config:

```bash
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

If you ever find yourself editing `config.toml` to disable them, you have
probably misconfigured the proxy.

---

## 7. Data, storage and backup

### 7.1 What is stored

| Path | Contents | Persistence |
|---|---|---|
| `/data/spas.db` | `assessments` table + `audit_events` table | **volume required** |
| `/app/models/*.pkl` | model bundles | mounted `:ro` in compose |
| Session state | current form, current training run | dies with the tab |

The store degrades safely: if `SPAS_DB_PATH` is unwritable the app logs a
warning and keeps working with session-only history rather than crashing. That
is a deployment smell, not a feature — `preflight.py` warns about it, and the
Assessment page shows a caption saying history is not being saved.

`SPAS_DB_ENABLED=0` makes the app deliberately stateless (a demo). Do not use it
where an audit trail is required.

### 7.2 Audit trail

Every consequential action is recorded in `audit_events`:
`admin_unlocked`, `admin_unlock_failed`, `dataset_loaded`, `model_saved`,
`model_loaded`, `model_deleted`, `assessment_created`, `report_exported`.

This is the Appointing Authority / Data Protection Commission story: who
unlocked Training Mode, when a dataset was loaded, when a bundle changed. Read
it with:

```bash
docker compose exec app python -c "
from backend.persistence.store import get_store
for e in get_store().list_events(limit=20):
    print(e["created_at"], e["action"], e["actor"], e["detail"])
"
```

### 7.3 Backup

SQLite is a single file; copy it *consistently*, not mid-write:

```bash
docker compose exec app python - <<'PY'
import sqlite3, datetime
src = sqlite3.connect("/data/spas.db")
dst = sqlite3.connect(f"/data/backup-{datetime.date.today()}.db")
with dst:
    src.backup(dst)
dst.close(); src.close()
PY

docker compose cp app:/data/backup-$(date +%F).db ./backups/
```

Schedule that (cron or a host timer) and copy the result off the machine. A
backup that lives only on the same disk is not a backup.

### 7.4 Restore

```bash
docker compose stop app
docker run --rm -v spas-data:/data -v "$PWD/backups:/backups" alpine \
  sh -c "cp /backups/spas-2026-11-04.db /data/spas.db"
docker compose start app
```

Verify afterwards with `sqlite3 spas.db "PRAGMA integrity_check;"` → `ok`, and
with the `count_assessments()` figure you recorded before the restore.

### 7.5 Retention

The Data Protection Act, 2012 (Act 843) principles in `docs/DATA_PROTECTION.md`
apply to the deployed system, not just the code: collect the minimum, keep it no
longer than needed, and be able to delete on request. Practically, agree a
retention window (e.g. one academic year after the assessment cycle) and delete
rows by `student_id` in one statement. Do not export the database to a personal
laptop "just in case".

---

## 8. Upgrades, retraining and rollback

### 8.1 Code upgrade

```bash
git pull
make test                       # 62 tests, ~40 s — do not skip this
docker compose build
docker compose up -d
make smoke                      # verifies the running deployment
```

`docker compose up -d` recreates the container; the `spas-data` volume and the
mounted `models` directory are untouched.

### 8.2 Retraining

Retraining replaces evidence. Do it deliberately:

```bash
python scripts/train_model.py --data data/real_cohort.csv \
    --out models/best_model.pkl --metrics reports/metrics.json
python scripts/report_results.py --data data/real_cohort.csv \
    --out reports/results.md --json reports/results.json
```

Then check, in the new bundle: `target_definition` is what you intended,
`baseline_majority` exists (a metric without a baseline is uninterpretable), and
macro-F1 on the hold-out beats that baseline. Update `docs/MODEL_CARD.md` with
the new figures — a model card that describes a previous model is worse than no
model card.

### 8.3 Rollback

Keep the previous bundle and pin it by date:

```bash
cp models/best_model.pkl models/archive/best_model-2026-11-04.pkl
# rollback:
cp models/archive/best_model-2026-10-01.pkl models/best_model.pkl
docker compose restart app
```

Because `models/` is mounted `:ro` from the host, rollback is a file copy plus a
restart — no rebuild, no downtime beyond a few seconds. Record the rollback in
`CHANGELOG.md`; the audit trail will not explain it for you.

### 8.4 Version guard

`MODEL_BUNDLE_VERSION` and `PROJECT_VERSION` live in `backend/config.py`. If you
change the bundle's structure (new keys, changed schema), bump the bundle
version — older code should refuse a newer bundle rather than misread it.

---

## 9. Before you serve real students

Technical readiness is not the same as fitness to deploy. Four items remain, and
they are **not** in this repository because they are decisions, not code:

1. **Retrain on a real, de-identified cohort.** The shipped bundle is synthetic.
   Every metric in `reports/results.md` describes generated data.
   The "Average" band is the weakest class (macro-F1 ≈ 0.50 on the synthetic
   cohort) — that is exactly the band a student would contest.
2. **Institutional sign-off.** The output is advisory. State in writing, to
   whoever owns the process, that the band is a prompt for a human decision and
   not a grade.
3. **Authentication beyond a shared password.** Training Mode uses one shared
   password, and there is no per-user identity or role separation. For anything
   beyond a single trusted operator, put the app behind institutional SSO at the
   proxy layer (e.g. OAuth2-proxy or your campus IdP) so `audit_events.actor`
   reflects a real person.
4. **Act 843 operational duties.** Register the processing, publish a privacy
   notice, keep the retention schedule, and have a named person who can action a
   deletion request. `docs/DATA_PROTECTION.md` is the starting draft.

If those four are not done, deploy on an internal, loopback-bound host for a
pilot with a small number of staff — not to students.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `docker compose ps` never leaves `starting` | no model bundle, or `/data` not writable | `docker compose logs app`; train a bundle (§3) |
| Assessment Mode shows "no model" | `models/best_model.pkl` missing in the container | mount `models/` or bake it in |
| History empty after restart | persistence disabled or DB not on a volume | check `SPAS_DB_PATH`, `SPAS_DB_ENABLED`, and the volume |
| `preflight.py` warns "bundle predates v3.0" | old bundle, legacy `cpi` target | retrain (§8.2) |
| App loads but widgets don't respond | reverse proxy is not upgrading WebSockets | add the `Upgrade`/`Connection` headers (§6.1) |
| `413 Request Entity Too Large` on upload | proxy body limit below 25 MB | raise `client_max_body_size` |
| `502` after a few minutes of training | proxy read timeout | `proxy_read_timeout 3600s` |
| `Permission denied` writing a model | image runs as uid 10001, mounted dir owned by root | `chown -R 10001:10001 models/` |
| Warning about CORS/XSRF in logs | config and proxy disagree | keep both enabled behind TLS; override only for previews |

---

## 11. Quick reference

```bash
make install      # venv + pinned deps
make test         # 62 tests
make preflight    # is this host ready?
make run          # local, :8501
make smoke        # is it serving? (make smoke URL=https://...)
make train        # retrain on data/cohort.csv
make report       # regenerate reports/results.md
make docker-build # build the image
make docker-up    # compose up -d --build
make docker-smoke # smoke-test through the host port
make docker-down  # stop, keep the volume
```

Environment variables that matter: `SPAS_ADMIN_PASSWORD_HASH` (required in
production), `SPAS_DB_PATH`, `SPAS_DB_ENABLED`, `SPAS_TARGET_DEFINITION`,
`SPAS_LABEL_METHOD`, `SPAS_TUNING`, `SPAS_CALIBRATION`. All are documented in
`.env.example`.

---

*Related documents: `docs/METHODOLOGY.md` (how the model is built and validated),
`docs/MODEL_CARD.md` (the released bundle's contract and limits),
`docs/DATA_PROTECTION.md` (Act 843 handling), `CHANGELOG.md` (what changed and
when).*
