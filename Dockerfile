# Diabetes API — FastAPI over the Kedro pipelines.
#
# Two-phase copy: dependencies first (they rarely change, so the layer stays
# cached), source code second. Editing a node then rebuilds only the last,
# cheap layers instead of reinstalling scikit-learn from scratch.

FROM python:3.14-slim

# uv is a static Rust binary: copy it from the official image rather than
# pip-installing it. Pinned, like the dependencies in uv.lock: `latest` would
# let the same Dockerfile build differently from one day to the next.
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /usr/local/bin/

# Pre-compile .py to .pyc at install time for a faster cold start.
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# --- phase 1: third-party dependencies only -------------------------------
# README.md is required because pyproject.toml declares `readme = "README.md"`;
# without it the build of the `diabetes` package itself fails in phase 2.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# --- phase 2: the project ---------------------------------------------------
COPY src/ src/
RUN uv sync --frozen --no-dev

# Documentation only — publishing the port is the job of `-p` / compose `ports`.
EXPOSE 8000

# Runs as root, matching the course Dockerfile. A non-root user is the stronger
# practice, but ./data is bind-mounted from the host: unless the container UID
# matches the host's, the pipelines lose write access to it. Fixing that
# properly means passing --user $(id -u):$(id -g) at run time.

# The venv built above, used directly. `uv run` would re-sync the environment
# on every start, *including* the dev group (jupyterlab, kedro-viz, pytest...):
# a download at each cold start, and a crash on a host without internet.
ENV PATH="/app/.venv/bin:$PATH"

# Exec form: uvicorn becomes PID 1 and receives signals directly, so
# `docker compose down` stops it cleanly instead of timing out.
# --host 0.0.0.0 is mandatory: binding to localhost would be unreachable from
# outside the container's network namespace.
CMD ["uvicorn", "diabetes.api:app", "--host", "0.0.0.0", "--port", "8000"]
