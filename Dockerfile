# ============================================================
# Student Performance Assessment System -- production image
# ============================================================
# Build:
#     docker build -t spas:3.3 .
# Run:
#     docker run -p 8501:8501 \
#         -e SPAS_ADMIN_PASSWORD_HASH=... \
#         -v spas-data:/data \
#         -v "$PWD/Code6/student_performance_app/models:/app/models" \
#         spas:3.3
#
# Notes
# -----
# * Dependencies are installed from requirements.txt (exact pins)
#   BEFORE the source is copied, so code changes do not invalidate
#   the dependency layer.
# * The container runs as a non-root user.
# * Persistence lives in /data (assessments + audit trail), which
#   should be a mounted volume so it survives redeploys.
# * The model bundle lives in /app/models and can either be baked
#   into the image or mounted, depending on your release process.
# ============================================================

FROM python:3.12-slim AS base

# --- runtime basics -----------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    SPAS_DB_PATH=/data/spas.db

# libgomp1 is needed by scikit-learn / xgboost wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- dependencies (cached layer) -----------------------------
COPY Code6/student_performance_app/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# --- application ---------------------------------------------
COPY Code6/student_performance_app/ ./

# --- non-root user -------------------------------------------
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /data /app/models \
    && chown -R app:app /app /data

USER app

# --- health check --------------------------------------------
# Uses the stdlib so the image needs no curl/wget, and honours $PORT
# so it still works on a platform that assigns one (Render, Cloud Run,
# Railway, Fly). Defaults to 8501 locally.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
port=os.environ.get('PORT','8501'); \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+port+'/_stcore/health', timeout=4).read().strip()==b'ok' else 1)"

EXPOSE 8501

# Streamlit serves its own HTTP/WebSocket stack; no gunicorn needed.
#
# PORT handling: platforms such as Render, Cloud Run, Railway and Fly
# inject $PORT and route traffic only to it. `exec` replaces the shell
# so SIGTERM still reaches Streamlit and shutdown stays graceful.
# Docker Compose / plain `docker run` set no PORT, so it falls back to
# 8501 and the existing port mapping keeps working unchanged.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
