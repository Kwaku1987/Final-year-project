# ============================================================
# Student Performance Assessment System -- task runner
# ============================================================
# Run `make` (or `make help`) to list targets.
# ============================================================

APP_DIR := Code6/student_performance_app
PY ?= python3
IMAGE ?= spas:3.3

.DEFAULT_GOAL := help
.PHONY: help install run test lint preflight smoke train report docker-build docker-up docker-down docker-logs docker-smoke clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create a venv and install pinned dependencies
	cd $(APP_DIR) && $(PY) -m venv .venv && \
		.venv/bin/pip install --upgrade pip && \
		.venv/bin/pip install -r requirements.txt

run: ## Run the app locally on :8501
	cd $(APP_DIR) && $(PY) -m streamlit run app.py

test: ## Run the test suite
	cd $(APP_DIR) && $(PY) -m pytest tests -q

lint: ## Byte-compile every module (catches syntax errors fast)
	cd $(APP_DIR) && $(PY) -m compileall -q . && echo "compile OK"

preflight: ## Check this host is ready to run the app
	cd $(APP_DIR) && $(PY) scripts/preflight.py

smoke: ## Smoke-test a running deployment (make smoke URL=...)
	cd $(APP_DIR) && $(PY) scripts/smoke_test.py --url $(or $(URL),http://127.0.0.1:8501)

train: ## Retrain on data/cohort.csv and export the bundle
	cd $(APP_DIR) && $(PY) scripts/make_synthetic_dataset.py --rows 2000 --out data/cohort.csv
	cd $(APP_DIR) && $(PY) scripts/train_model.py --data data/cohort.csv \
		--out models/best_model.pkl --metrics reports/metrics.json

report: ## Regenerate the report tables from data/cohort.csv
	cd $(APP_DIR) && $(PY) scripts/report_results.py --data data/cohort.csv \
		--out reports/results.md --json reports/results.json

docker-build: ## Build the production image
	docker build -t $(IMAGE) .

docker-up: ## Start the stack (needs .env)
	docker compose up -d --build

docker-down: ## Stop the stack (keeps the data volume)
	docker compose down

docker-smoke: ## Smoke-test the container through the host port
	cd $(APP_DIR) && $(PY) scripts/smoke_test.py --url http://127.0.0.1:8501

docker-logs: ## Follow application logs
	docker compose logs -f

clean: ## Remove caches and generated artefacts
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache $(APP_DIR)/.pytest_cache \
		$(APP_DIR)/data $(APP_DIR)/reports
